from functools import cache

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, radius_graph
from torch_geometric.data import Data, Batch
from guided_diffusion.unet import SimpleFlowModel
from torch_scatter import scatter

from guided_diffusion.unet import timestep_embedding
from diffusion_co_design.pretrain.rware.generate import (
    WarehouseRandomGeneratorConfig,
)
from diffusion_co_design.pretrain.rware.generate import get_position


# Connectivity
def fully_connected(n):
    row, col = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    edge_index = torch.stack([row.flatten(), col.flatten()], dim=0)
    return edge_index


def shelf_radius_graph_goals_connected(
    shelf_pos, is_shelf_mask, radius, batch, goal_edge_index
):
    shelf_edge_index = radius_graph(shelf_pos, r=radius, batch=batch[is_shelf_mask])
    shelf_indices = torch.where(is_shelf_mask)[0]
    shelf_edge_index = shelf_indices[shelf_edge_index]

    return torch.cat([shelf_edge_index, goal_edge_index], dim=1)


class E3GNNLayer(MessagePassing):
    def __init__(
        self,
        node_embedding_dim: int = 32,
        edge_embedding_dim: int = 16,  # Messages
        graph_embedding_dim: int = 32,
        normalise_pos: bool = True,
        pos_aggr: str = "mean",
        update_node_features: bool = True,
        use_attention: bool = True,
    ):
        super().__init__(aggr="add")

        self.normalise_pos = normalise_pos

        self.message_mlp = nn.Sequential(
            nn.Linear(
                node_embedding_dim * 2 + graph_embedding_dim + 1,
                edge_embedding_dim,
            ),
            nn.SiLU(),
            nn.Linear(edge_embedding_dim, edge_embedding_dim),
            nn.SiLU(),
        )

        self.pos_mlp = nn.Sequential(
            nn.Linear(edge_embedding_dim, edge_embedding_dim),
            nn.SiLU(),
            nn.Linear(edge_embedding_dim, 1),
        )
        self.pos_aggr = pos_aggr

        if update_node_features:
            self.node_mlp = nn.Sequential(
                nn.Linear(edge_embedding_dim + node_embedding_dim, node_embedding_dim),
                nn.SiLU(),
                nn.Linear(node_embedding_dim, node_embedding_dim),
                nn.LeakyReLU(),
            )
        else:
            self.node_mlp = None

        if use_attention:
            self.att_mlp = nn.Sequential(nn.Linear(edge_embedding_dim, 1), nn.Sigmoid())
        else:
            self.att_mlp = None

    def forward(self, x, edge_index, pos, graph_embedding, batch):
        edge_batch = batch[edge_index[0]]

        if graph_embedding is not None:
            graph_embedding = graph_embedding[edge_batch]
        out = self.propagate(edge_index, x=x, pos=pos, graph_embedding=graph_embedding)
        return out

    def message(self, x_i, x_j, pos_i, pos_j, graph_embedding=None):
        # Position encodings
        pos_diff = pos_i - pos_j
        radial = torch.sum(pos_diff**2, dim=-1, keepdim=True)
        if self.normalise_pos:
            norm = torch.sqrt(radial).detach() + 1e-6
            pos_diff = pos_diff / norm

        # Messages
        if graph_embedding is not None:
            msg = torch.cat([x_i, x_j, radial, graph_embedding], dim=-1)
        else:
            msg = torch.cat([x_i, x_j, radial], dim=-1)
        msg = self.message_mlp(msg)
        if self.att_mlp:
            msg = msg * self.att_mlp(msg)

        return (msg, pos_diff)

    def aggregate(self, inputs, index):
        msg, pos_diff = inputs
        pos_vec = pos_diff * self.pos_mlp(msg)
        aggr_h = scatter(msg, index, dim=0, reduce=self.aggr)
        aggr_pos = scatter(pos_vec, index, dim=0, reduce=self.pos_aggr)
        return aggr_h, aggr_pos

    def update(self, inputs, x, pos):
        aggr_h, aggr_pos = inputs
        upd_out = self.node_mlp(torch.cat([x, aggr_h], dim=-1)) if self.node_mlp else 0
        return upd_out + x, aggr_pos + pos


class WarehouseGNNBase(nn.Module):
    def __init__(
        self,
        scenario: WarehouseRandomGeneratorConfig,
        use_radius_graph: bool = True,
        radius: float = 0.5,
        include_color_features: bool = False,
    ):
        super().__init__()
        self.include_color_features = include_color_features
        self.feature_dim = 2 if not include_color_features else 2 + scenario.n_colors
        self.scenario = scenario
        self.num_nodes = scenario.n_goals + scenario.n_shelves

        self.use_radius_graph = use_radius_graph
        self.r = radius

    def generate_scenario_graph(self) -> Data:
        scenario = self.scenario
        pos = torch.zeros((self.num_nodes, 2), dtype=torch.float32)
        h = torch.zeros(
            (self.num_nodes, self.feature_dim), dtype=torch.float32
        )  # One-hot for type

        assert scenario.goal_idxs is not None
        for i, idx in enumerate(scenario.goal_idxs):
            h[i, 0] = 1
            x, y = get_position(idx, scenario.size)
            pos[i, 0] = (x / scenario.size) * 2 - 1
            pos[i, 1] = (y / scenario.size) * 2 - 1
            if self.include_color_features:
                assert scenario.goal_colors is not None
                h[i, 2 + scenario.goal_colors[i]] = 1
        for i in range(scenario.n_shelves):
            h[i + scenario.n_goals, 1] = 1

        edge_index = fully_connected(self.num_nodes)
        is_goal_edge = edge_index[0] < scenario.n_goals

        data = Data(h=h, edge_index=edge_index, pos=pos, is_goal_edge=is_goal_edge)
        return data

    def make_graph_from_data(
        self, pos: torch.Tensor, color: torch.Tensor | None = None
    ):
        assert pos.ndim in [2, 3]
        if pos.ndim == 2:
            pos = pos.unsqueeze(0)

        batch_size = pos.shape[0]
        assert pos.shape[1] == self.scenario.n_shelves
        assert pos.shape[2] == 2
        pos = pos.view(-1, 2)

        graph, is_shelf_mask = self.get_batch_graph(batch_size, device=pos.device)

        if self.use_radius_graph:
            graph.edge_index = shelf_radius_graph_goals_connected(
                shelf_pos=pos,
                is_shelf_mask=is_shelf_mask,
                radius=self.r,
                batch=graph.batch,
                goal_edge_index=graph.edge_index[:, graph.is_goal_edge],
            )

        graph.pos[is_shelf_mask] = pos

        if self.include_color_features:
            assert color is not None
            color = color.view(-1, self.scenario.n_colors)
            graph.h[is_shelf_mask][:, 2:] = color

        return graph, is_shelf_mask

    @cache
    def _get_batch_graph(self, batch_size: int) -> Batch:
        graphs = [self.generate_scenario_graph() for _ in range(batch_size)]
        batch_graph = Batch.from_data_list(graphs)

        is_shelf_mask = batch_graph.h[:, 1] == 1
        return batch_graph, is_shelf_mask

    def get_batch_graph(self, batch_size: int, device="cpu") -> Batch:
        graph, is_shelf_mask = self._get_batch_graph(batch_size)

        return graph.clone().to(device), is_shelf_mask.to(device)


class WarehouseDiffusionModel(WarehouseGNNBase):
    def __init__(
        self,
        scenario: WarehouseRandomGeneratorConfig,
        node_embedding_dim: int = 32,
        edge_embedding_dim: int = 32,
        timestep_embedding_dim: int = 32,
        num_layers: int = 5,
        use_radius_graph: bool = True,
        radius: float = 0.5,
    ):
        super().__init__(
            scenario=scenario,
            use_radius_graph=use_radius_graph,
            radius=radius,
            include_color_features=False,
        )
        self.feature_dim = 2
        self.scenario = scenario
        self.num_nodes = scenario.n_goals + scenario.n_shelves
        self.num_layers = num_layers

        self.embedding_dim = node_embedding_dim
        self.timestep_embedding_dim = timestep_embedding_dim

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            layer = E3GNNLayer(
                node_embedding_dim=node_embedding_dim,
                edge_embedding_dim=edge_embedding_dim,
                graph_embedding_dim=timestep_embedding_dim,
                update_node_features=i < num_layers - 1,
            )
            self.convs.append(layer)

        self.timestep_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            t_layer = nn.Linear(timestep_embedding_dim, timestep_embedding_dim)
            self.timestep_layers.append(t_layer)

        self.h_in = nn.Linear(2, node_embedding_dim)
        self.activation = nn.SiLU()

        self.use_radius_graph = use_radius_graph
        self.r = radius

    def forward(self, pos: torch.Tensor, timesteps=None):
        shape = pos.shape

        batch_size = shape[0]
        graph, is_shelf_mask = self.make_graph_from_data(pos)

        if timesteps is None:
            timesteps = torch.zeros(batch_size, dtype=torch.long, device=pos.device)
        timesteps_emb = timestep_embedding(timesteps, self.timestep_embedding_dim)

        h = self.h_in(graph.h)
        pos = graph.pos

        for i in range(self.num_layers):
            g_layer = self.convs[i]
            h, pos = g_layer(h, graph.edge_index, pos, timesteps_emb, graph.batch)
            if i < self.num_layers - 1:
                t_layer = self.timestep_layers[i]
                timesteps_emb = t_layer(timesteps_emb)
                timesteps_emb = self.activation(timesteps_emb)

        out = pos[is_shelf_mask]
        return out.view(shape)


class WarehouseDiffusionMLP(nn.Module):
    def __init__(self, scenario: WarehouseRandomGeneratorConfig):
        super().__init__()
        self.scenario = scenario
        self.model = SimpleFlowModel(
            data_shape=(2 * scenario.n_shelves, 1),
            hidden_dim=1024,
        )

    def forward(self, pos: torch.Tensor, timesteps):
        B = pos.shape[0]
        pos = pos.reshape(B, -1)
        out = self.model(pos, timesteps)
        return out.reshape(B, self.scenario.n_shelves, 2)
