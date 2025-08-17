from math import prod

import torch
from torch_geometric.data import Data, Batch
from torch_geometric.nn import knn_graph, global_add_pool
from torch_geometric.nn.models import GAT

from diffusion_co_design.vmas.schema import ScenarioConfig


class E3Critic(torch.nn.Module):
    def __init__(
        self,
        scenario: ScenarioConfig,
        node_emb_dim: int = 128,
        num_layers: int = 3,
        k: int = 5,
    ):
        super().__init__()
        self.scenario = scenario
        self.n_obstacles = len(scenario.obstacle_sizes)
        self.k = k

        # Observations
        # agent_pos (2) vel (2) goal_pos (2) lidar (12)

        # State
        # obstacle_pos (n_obstacles, 2)

        self.model = GAT(
            in_channels=-1,
            edge_dim=5,
            hidden_channels=node_emb_dim,
            num_layers=num_layers,
            out_channels=1,
            act="relu",
            add_self_loops=False,
            v2=True,
        )

    def forward(self, obstacle_pos, agent_pos, goal_pos, agent_vel):
        B_all = obstacle_pos.shape[:-2]
        B, N = prod(B_all), self.scenario.get_num_agents()

        # Shape checks
        obstacle_pos = obstacle_pos.reshape(-1, self.n_obstacles, 2)
        agent_pos = agent_pos.reshape(-1, N, 2)
        goal_pos = goal_pos.reshape(-1, N, 2)
        agent_vel = agent_vel.reshape(-1, N, 2)

        assert obstacle_pos.shape == (B, self.n_obstacles, 2), obstacle_pos.shape
        assert agent_pos.shape == (B, N, 2), agent_pos.shape
        assert goal_pos.shape == (B, N, 2), goal_pos.shape
        assert agent_vel.shape == (B, N, 2), agent_vel.shape

        # Construct graph
        data_list = [
            self.construct_graph(
                obstacle_pos=obstacle_pos[i],
                agent_pos=agent_pos[i],
                goal_pos=goal_pos[i],
                agent_vel=agent_vel[i],
            )
            for i in range(B)
        ]

        data = Batch.from_data_list(data_list)

        out = self.model(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=data.batch,
        )

        is_agent = data.x[:, 0] == 1.0
        out = out[is_agent]
        out = global_add_pool(out, data.batch[is_agent])

        out = (
            out.view(*B_all, 1)
            .expand(*B_all, self.scenario.get_num_agents())
            .unsqueeze(-1)
        )
        return out

    def construct_graph(self, obstacle_pos, agent_pos, goal_pos, agent_vel):
        N = self.scenario.get_num_agents() * 2 + self.n_obstacles

        # Graph topology
        pos = torch.cat([agent_pos, goal_pos, obstacle_pos], dim=-2)
        assert pos.shape == (N, 2), pos.shape
        edge_index = knn_graph(pos, k=self.k, loop=True)
        # add direct edges between agents and their goals
        agent_goal_edges = torch.arange(self.n_obstacles, device=pos.device)
        agent_goal_edges = torch.stack(
            [agent_goal_edges, agent_goal_edges + self.scenario.get_num_agents()],
            dim=0,
        )
        edge_index = torch.cat([edge_index, agent_goal_edges], dim=-1)
        # remove duplicate edges
        edge_index = torch.unique(edge_index, dim=-1)

        # Node features:
        # One-hot encoding of node types
        # Velocity norm
        # Radius
        x = torch.zeros((N, 3 + 1 + 1), device=pos.device)
        x[: self.scenario.get_num_agents(), 0] = 1.0
        x[self.scenario.get_num_agents() : self.scenario.get_num_agents() * 2, 1] = 1.0
        x[self.scenario.get_num_agents() * 2 :, 2] = 1.0

        x[: self.scenario.get_num_agents(), 3] = torch.linalg.vector_norm(
            agent_vel, dim=-1
        )
        entity_radius = torch.zeros(N, device=pos.device)
        entity_radius[: self.scenario.get_num_agents()] = 0.05
        entity_radius[self.scenario.get_num_agents() * 2 :] = torch.tensor(
            self.scenario.obstacle_sizes, device=pos.device
        )
        x[:, 4] = entity_radius

        # Edge features:
        # Boolean indicating if the edge is between an agent and its goal
        # Absolute distance
        # Absolute collision distance
        # Vel dot and prod
        edge_attr = torch.zeros((edge_index.shape[1], 1 + 1 + 1 + 2), device=pos.device)

        agent_goal_edges = torch.logical_and(
            edge_index[0, :] < self.scenario.get_num_agents(),
            edge_index[1, :] == edge_index[0, :] + self.scenario.get_num_agents(),
        )
        edge_attr[agent_goal_edges, 0] = 1.0

        from_pos = pos[edge_index[0, :]]
        to_pos = pos[edge_index[1, :]]

        dist = torch.linalg.vector_norm(from_pos - to_pos, dim=-1)
        edge_attr[:, 1] = dist

        radius_sum = entity_radius[edge_index[0, :]] + entity_radius[edge_index[1, :]]
        edge_attr[:, 2] = dist - radius_sum

        vel = torch.zeros(N, 2, device=pos.device)
        vel[: self.scenario.get_num_agents()] = agent_vel
        from_vel = vel[edge_index[0, :]]
        to_vel = vel[edge_index[1, :]]
        rel_vel = from_vel - to_vel
        pos_diff = from_pos - to_pos
        pos_diff = pos_diff / dist.unsqueeze(-1).clamp_min(1e-6)

        edge_attr[:, 3] = torch.sum(rel_vel * pos_diff, dim=-1)
        edge_attr[:, 4] = (
            rel_vel[:, 0] * pos_diff[:, 1] - rel_vel[:, 1] * pos_diff[:, 0]
        )

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
        return data
