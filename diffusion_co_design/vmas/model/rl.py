from math import prod

import torch
from tensordict.nn import TensorDictModule, InteractionType
from torchrl.data.utils import DEVICE_TYPING
from torchrl.modules import ProbabilisticActor, MultiAgentMLP
from torchrl.modules import NormalParamExtractor, TanhNormal
from torch_geometric.data import Data, Batch
from torch_geometric.nn import knn_graph, global_add_pool
from torch_geometric.nn.models import GAT

from diffusion_co_design.vmas.schema import (
    ActorConfig,
    CriticConfig,
    ActorCriticConfig,
    ScenarioConfig,
)
from diffusion_co_design.vmas.scenario.obstacle_navigation import DesignableVmasEnv
from diffusion_co_design.vmas.static import GROUP_NAME


def create_policy(env: DesignableVmasEnv, cfg: ActorConfig, device: DEVICE_TYPING):
    policy_net = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=env.observation_spec[GROUP_NAME, "observation"].shape[
                -1
            ],  # n_obs_per_agent
            n_agent_outputs=2 * env.action_spec.shape[-1],
            n_agents=env.n_agents,
            centralised=False,
            share_params=True,
            device=device,
            depth=cfg.depth,
            num_cells=cfg.hidden_size,
            activation_class=torch.nn.Tanh,
        ),
        NormalParamExtractor(),
    )

    policy_module = TensorDictModule(
        policy_net,
        in_keys=[(GROUP_NAME, "observation")],
        out_keys=[(GROUP_NAME, "loc"), (GROUP_NAME, "scale")],
    )

    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.unbatched_action_spec,
        in_keys=[(GROUP_NAME, "loc"), (GROUP_NAME, "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.unbatched_action_spec[env.action_key].space.low,
            "high": env.unbatched_action_spec[env.action_key].space.high,
        },
        return_log_prob=True,
        log_prob_key=(GROUP_NAME, "sample_log_prob"),
        default_interaction_type=InteractionType.RANDOM,
    )

    return policy


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

    def forward(self, obs, state):
        B_all = obs.shape[:-2]
        B, N = prod(B_all), self.scenario.get_num_agents()

        # Shape checks
        obs = obs.reshape(-1, N, obs.shape[-1])
        assert obs.shape == (B, N, 18), obs.shape
        state = state.reshape(-1, self.n_obstacles, state.shape[-1])
        assert state.shape == (B, self.n_obstacles, 2), state.shape

        # Construct graph
        data_list: list[Data] = []
        for i in range(B):
            obstacle_pos = state[i]
            agent_pos = obs[i, :, :2]
            goal_pos = obs[i, :, 4:6]

            # Note: discard lidar for now, position information should be sufficient
            agent_vel = obs[i, :, 2:4]

            data_list.append(
                self.construct_graph(
                    obstacle_pos=obstacle_pos,
                    agent_pos=agent_pos,
                    goal_pos=goal_pos,
                    agent_vel=agent_vel,
                )
            )

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

        out = out.view(*B_all, 1).expand(-1, self.scenario.get_num_agents())
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
        entity_radius = torch.zeros(N)
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


def create_critic(
    env: DesignableVmasEnv,
    scenario: ScenarioConfig,
    cfg: CriticConfig,
    device: DEVICE_TYPING,
):
    critic_net = E3Critic(
        scenario=scenario,
        node_emb_dim=cfg.hidden_size,
        num_layers=cfg.depth,
    ).to(device=device)

    critic = TensorDictModule(
        critic_net,
        in_keys=[(GROUP_NAME, "observation"), "state"],
        out_keys=[(GROUP_NAME, "state_value")],
    )
    return critic


def vmas_models(
    env: DesignableVmasEnv,
    scenario: ScenarioConfig,
    actor_critic_cfg: ActorCriticConfig,
    device: DEVICE_TYPING = torch.device("cpu"),
):
    policy = create_policy(env, actor_critic_cfg.actor, device)
    critic = create_critic(env, scenario, actor_critic_cfg.critic, device)

    td = env.reset()
    with torch.no_grad():
        policy(td)
        critic(td)

    return policy, critic
