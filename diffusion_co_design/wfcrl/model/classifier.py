from abc import abstractmethod
from typing import Literal
from math import prod

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from diffusion_co_design.common.nn.geometric import fully_connected
from diffusion_co_design.common.nn import EnvCritic as _EnvCritic
from diffusion_co_design.wfcrl.schema import ScenarioConfig


class EnvCritic(_EnvCritic):
    def __init__(self, post_hook: torch.nn.Module | None = None):
        super().__init__()
        self.post_hook = post_hook

    def forward(self, x: torch.Tensor):
        out = self._forward(x)
        if self.post_hook is not None:
            return self.post_hook(out)
        return out

    @abstractmethod
    def _forward(self, layout: torch.Tensor):
        raise NotImplementedError()


class MLPCritic(EnvCritic):
    def __init__(
        self,
        cfg: ScenarioConfig,
        embedding_size: int = 256,
        depth: int = 2,
        post_hook: torch.nn.Module | None = None,
    ):
        super().__init__(post_hook=post_hook)
        self.scenario = cfg
        layers: list[nn.Module] = []

        for _ in range(depth):
            layers.append(nn.Linear(embedding_size, embedding_size))
            layers.append(nn.SiLU())

        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_turbines * 2, embedding_size),  # [B, N*2] -> [B, E]
            nn.SiLU(),
            *layers,
            nn.Linear(embedding_size, 1),
        )

    def _forward(self, layout: torch.Tensor):
        # x: [B, N, 2]
        layout = layout.flatten(start_dim=1)  # [B, N*2]
        return self.mlp(layout).squeeze(-1)  # [B]


class GNNCritic(EnvCritic):
    def __init__(
        self,
        cfg: ScenarioConfig,
        node_emb_dim: int = 64,
        edge_emb_dim: int = 16,
        n_layers: int = 4,
        aggr: Literal["add", "mean", "max"] = "add",
        post_hook: torch.nn.Module | None = None,
    ):
        super().__init__(post_hook=post_hook)
        self.scenario = cfg
        self.n_turbines = self.scenario.n_turbines  # N

        edge_index = fully_connected(self.n_turbines)  # [2, E = (N)(N-1)/2]
        # Remove self loops
        self.edge_index = edge_index[:, edge_index[0] != edge_index[1]]

        self.node_emb_dim = node_emb_dim
        self.node_in = nn.Linear(3, node_emb_dim)
        self.edge_in = nn.Linear(3, edge_emb_dim)

        self.message_mlp_list = nn.ModuleList()
        self.upd_mlp_list = nn.ModuleList()
        for _ in range(n_layers):
            self.message_mlp_list.append(
                nn.Sequential(
                    nn.Linear(node_emb_dim * 2 + edge_emb_dim, edge_emb_dim),
                    nn.SiLU(),
                    nn.LayerNorm(edge_emb_dim),
                    nn.Linear(edge_emb_dim, edge_emb_dim),
                    nn.SiLU(),
                )
            )

            self.upd_mlp_list.append(
                nn.Sequential(
                    nn.Linear(node_emb_dim + edge_emb_dim, node_emb_dim),
                    nn.SiLU(),
                    nn.LayerNorm(node_emb_dim),
                    nn.Linear(node_emb_dim, node_emb_dim),
                    nn.SiLU(),
                )
            )

        self.att_mlp = nn.Sequential(nn.Linear(edge_emb_dim, 1), nn.Sigmoid())
        self.aggr = aggr

        self.out_mlp = nn.Linear(node_emb_dim, 1)

    def _forward(
        self,
        layout: torch.Tensor,  # [*B, N, 2]
    ):
        device = layout.device
        has_batch = len(layout.shape) > 2
        if not has_batch:
            layout = layout.unsqueeze(0)

        self.edge_index = self.edge_index.to(device=device)
        B_all = layout.shape[:-2]
        B = prod(B_all)
        N = self.n_turbines
        E = len(self.edge_index[0])

        # Shape checks
        layout = layout.reshape(-1, N, 2)
        assert layout.shape == (B, N, 2), layout.shape

        # Fully connected graph
        src, dst = self.edge_index[0], self.edge_index[1]

        # Feature engineering
        pos_diff = layout[:, dst] - layout[:, src]  # [B, E, 2]
        radial = torch.norm(pos_diff, dim=-1, keepdim=True)  # [B, E, 1]

        assert pos_diff.shape == (B, E, 2), pos_diff.shape
        assert radial.shape == (B, E, 1), radial.shape

        # In layers
        edge_features = torch.cat([radial, pos_diff], dim=-1)  # [B, E, 3]
        assert edge_features.shape == (B, E, 3), edge_features.shape
        e = self.edge_in(edge_features)  # [B, E, edge_emb_dim]
        node_features = torch.cat(
            [scatter(edge_features, src, dim=1, dim_size=N, reduce=self.aggr)], dim=-1
        )  # [B, N, 3]
        assert node_features.shape == (B, N, 3), node_features.shape
        h = self.node_in(node_features)  # [B, N, node_emb_dim]

        # Message passing
        for i, (message_mlp, upd_mlp) in enumerate(
            zip(self.message_mlp_list, self.upd_mlp_list)
        ):
            is_final_layer = i == len(self.message_mlp_list) - 1

            # Message
            msg = message_mlp(
                torch.cat([h[:, src], h[:, dst], e], dim=-1)
            )  # [B, E, edge_emb_dim]

            # Attention
            msg = msg * self.att_mlp(msg)

            h_aggr = scatter(
                msg, src, dim=1, dim_size=N, reduce=self.aggr
            )  # [B, N, edge_emb_dim]
            h_aggr = upd_mlp(torch.cat([h, h_aggr], dim=-1))  # [B, N, node_emb_dim]

            # Residual connections
            h = h + h_aggr
            if not is_final_layer:
                e = e + msg

        assert h.shape == (B, N, self.node_emb_dim)
        h = h.reshape(*B_all, N, self.node_emb_dim)
        h = self.out_mlp(h)  # [*B, N, 1]
        h = h.mean(dim=-2)  # [*B, 1]
        if not has_batch:
            h = h.squeeze(0)

        return h.squeeze(-1)  # [*B]
