import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter


class EnvCritic(nn.Module):
    supports_distillation: bool = False

    def predict_theta_value_with_hint(self, *args, **kwargs):
        if self.supports_distillation:
            return self.forward(*args, **kwargs)
        else:
            return self.forward(*args, **kwargs), None

    def predict_theta_value(self, *args, **kwargs):
        return self.predict_theta_value_with_hint(*args, **kwargs)[0]


def fully_connected(n):
    row, col = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    edge_index = torch.stack([row.flatten(), col.flatten()], dim=0)
    return edge_index


class EGNNLayer(MessagePassing):
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
