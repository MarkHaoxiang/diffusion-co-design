import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, radius_graph
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx, to_undirected
import networkx as nx
from torch_scatter import scatter

from rware.rendering import _SHELF_COLORS
from diffusion_co_design.rware.schema import ScenarioConfig
from diffusion_co_design.rware.diffusion.generate import get_position


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


def visualize_warehouse_graph(
    data: Data,
    ax,
    include_color_features: bool = True,
):
    graph = to_networkx(data, to_undirected=True)
    pos = data.pos.numpy(force=True)  # Positions
    h = data.h.numpy(force=True)  # Node features

    goal_shape, shelf_shape = "o", "s"
    node_colors = []
    node_shapes = []
    for i, feat in enumerate(h):
        if include_color_features:
            color = feat[2:].argmax().item()
            node_colors.append([x / 255 for x in _SHELF_COLORS[color]])
        else:
            if feat[0] == 1:
                node_colors.append("green")
            else:
                node_colors.append("saddlebrown")

        if feat[0] == 1:
            node_shapes.append(goal_shape)
        elif feat[1] == 1:
            node_shapes.append(shelf_shape)
        else:
            raise ValueError("Unknown object")

    # Initialize plot
    ax.set_title("Warehouse Graph Representation")
    ax.set_aspect("equal")
    ax.axis("off")

    for shape in [goal_shape, shelf_shape]:
        nodes = [i for i, s in enumerate(node_shapes) if s == shape]
        nx.draw_networkx_nodes(
            G=graph,
            pos=pos,
            nodelist=nodes,
            node_size=100,
            node_color=[node_colors[i] for i in nodes],
            node_shape=shape,
            ax=ax,
        )

    nx.draw_networkx_edges(G=graph, pos=pos, ax=ax, alpha=0.5)


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
                nn.SiLU(),
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


class WarehouseGNNLayer(MessagePassing):
    def __init__(
        self,
        node_embedding_dim: int = 32,
        edge_embedding_dim: int = 16,  # Messages
        graph_embedding_dim: int = 32,
        pos_aggr: str = "mean",
    ):
        super().__init__(aggr="add")

        self.msg_mlp = nn.Sequential(
            nn.Linear(
                node_embedding_dim * 2 + graph_embedding_dim + 2,
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
            nn.SiLU(),
        )
        self.pos_aggr = pos_aggr

        self.node_mlp = nn.Sequential(
            nn.Linear(edge_embedding_dim + node_embedding_dim, node_embedding_dim),
            nn.SiLU(),
            nn.Linear(node_embedding_dim, node_embedding_dim),
            nn.SiLU(),
        )

    def forward(self, x, edge_index, pos, graph_embedding, batch):
        edge_batch = batch[edge_index[0]]

        if graph_embedding is not None:
            graph_embedding = graph_embedding[edge_batch]
        out = self.propagate(edge_index, x=x, pos=pos, graph_embedding=graph_embedding)
        return out

    def message(self, x_i, x_j, pos_i, pos_j, graph_embedding=None):
        # Position encodings
        pos_diff = pos_i - pos_j
        d_x = pos_diff[:, 0]
        d_y = pos_diff[:, 1]

        abs_d_x = torch.abs(d_x).unsqueeze(-1)
        abs_d_y = torch.abs(d_y).unsqueeze(-1)

        # Messages
        msg_in = [x_i, x_j, abs_d_x, abs_d_y]
        if graph_embedding is not None:
            msg_in.append(graph_embedding)

        msg = self.msg_mlp(torch.cat(msg_in, dim=-1))

        return (msg, pos_diff)

    def aggregate(self, inputs, index):
        msg, pos_diff = inputs
        # pos_vec = pos_diff * self.pos_mlp(msg)
        aggr_h = scatter(msg, index, dim=0, reduce=self.aggr)
        # aggr_pos = scatter(pos_vec, index, dim=0, reduce=self.pos_aggr)
        # return aggr_h, aggr_pos
        return aggr_h

    def update(self, inputs, x, pos):
        aggr_h = inputs
        # aggr_h, aggr_pos = inputs
        upd_out = self.node_mlp(torch.cat([x, aggr_h], dim=-1)) if self.node_mlp else 0
        return upd_out + x


class WarehouseGNNBase(nn.Module):
    def __init__(
        self,
        scenario: ScenarioConfig,
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

    def generate_scenario_graph(
        self, shelf_pos: torch.Tensor, shelf_colors: torch.Tensor | None
    ) -> Data:
        scenario = self.scenario
        pos = torch.zeros(
            (self.num_nodes, 2), dtype=torch.float32, device=shelf_pos.device
        )
        h = torch.zeros(
            (self.num_nodes, self.feature_dim),
            dtype=torch.float32,
            device=shelf_pos.device,
        )  # One-hot for type

        assert scenario.goal_idxs is not None
        for i, idx in enumerate(scenario.goal_idxs):
            h[i, 0] = 1
            x, y = get_position(idx, scenario.size)
            pos[i, 0] = (x / (scenario.size - 1)) * 2 - 1
            pos[i, 1] = (y / (scenario.size - 1)) * 2 - 1
            if self.include_color_features:
                assert scenario.goal_colors is not None
                h[i, 2 + scenario.goal_colors[i]] = 1
        for i in range(scenario.n_shelves):
            h[i + scenario.n_goals, 1] = 1

        edge_index = fully_connected(self.num_nodes)
        is_goal_edge = edge_index[0] < scenario.n_goals
        edge_index = edge_index[:, is_goal_edge].to(device=shelf_pos.device)

        shelf_edge_index = radius_graph(shelf_pos, r=self.r, batch=None)
        is_shelf_mask = (h[:, 1] == 1).to(device=shelf_pos.device)
        shelf_indices = torch.where(is_shelf_mask)[0]
        shelf_edge_index = shelf_indices[shelf_edge_index]

        edge_index = torch.cat([edge_index, shelf_edge_index], dim=1)
        edge_index = to_undirected(edge_index)
        # edge_index, _ = coalesce(edge_index, None, self.num_nodes, sort_by_row=False)
        pos[is_shelf_mask] = shelf_pos

        if self.include_color_features:
            assert shelf_colors is not None
            h[is_shelf_mask, 2:] = shelf_colors

        data = Data(h=h, edge_index=edge_index, pos=pos)
        return data

    def make_graph_batch_from_data(
        self, pos: torch.Tensor, color: torch.Tensor | None = None
    ):
        assert pos.ndim in [2, 3]
        if pos.ndim == 2:
            pos = pos.unsqueeze(0)

        batch_size = pos.shape[0]
        assert pos.shape[1] == self.scenario.n_shelves
        assert pos.shape[2] == 2

        graph_list = []
        for i in range(batch_size):
            graph = self.generate_scenario_graph(
                pos[i], color[i] if color is not None else None
            )
            graph_list.append(graph)
        graph = Batch.from_data_list(graph_list)
        is_shelf_mask = graph.h[:, 1] == 1

        return graph, is_shelf_mask


# class WarehouseDiffusionModel(WarehouseGNNBase):
#     def __init__(
#         self,
#         scenario: WarehouseRandomGeneratorConfig,
#         node_embedding_dim: int = 32,
#         edge_embedding_dim: int = 32,
#         timestep_embedding_dim: int = 32,
#         num_layers: int = 5,
#         use_radius_graph: bool = True,
#         radius: float = 0.5,
#     ):
#         super().__init__(
#             scenario=scenario,
#             use_radius_graph=use_radius_graph,
#             radius=radius,
#             include_color_features=False,
#         )
#         self.feature_dim = 2
#         self.scenario = scenario
#         self.num_nodes = scenario.n_goals + scenario.n_shelves
#         self.num_layers = num_layers

#         self.embedding_dim = node_embedding_dim
#         self.timestep_embedding_dim = timestep_embedding_dim

#         self.convs = nn.ModuleList()
#         for i in range(num_layers):
#             layer = E3GNNLayer(
#                 node_embedding_dim=node_embedding_dim,
#                 edge_embedding_dim=edge_embedding_dim,
#                 graph_embedding_dim=timestep_embedding_dim,
#                 update_node_features=i < num_layers - 1,
#             )
#             self.convs.append(layer)

#         self.timestep_layers = nn.ModuleList()
#         for _ in range(num_layers - 1):
#             t_layer = nn.Linear(timestep_embedding_dim, timestep_embedding_dim)
#             self.timestep_layers.append(t_layer)

#         self.h_in = nn.Linear(2, node_embedding_dim)
#         self.activation = nn.SiLU()

#         self.use_radius_graph = use_radius_graph
#         self.r = radius

#     def forward(self, pos: torch.Tensor, timesteps=None):
#         shape = pos.shape

#         batch_size = shape[0]
#         graph, is_shelf_mask = self.make_graph_batch_from_data(pos)

#         if timesteps is None:
#             timesteps = torch.zeros(batch_size, dtype=torch.long, device=pos.device)
#         timesteps_emb = timestep_embedding(timesteps, self.timestep_embedding_dim)

#         h = self.h_in(graph.h)
#         pos = graph.pos

#         for i in range(self.num_layers):
#             g_layer = self.convs[i]
#             h, pos = g_layer(h, graph.edge_index, pos, timesteps_emb, graph.batch)
#             if i < self.num_layers - 1:
#                 t_layer = self.timestep_layers[i]
#                 timesteps_emb = t_layer(timesteps_emb)
#                 timesteps_emb = self.activation(timesteps_emb)

#         out = pos[is_shelf_mask]
#         return out.view(shape)
