from abc import abstractmethod
from typing import Literal
from math import prod

import torch
import torch.nn as nn
from torch_geometric.utils import scatter
from torch_geometric.data import Data, Batch
from torch_geometric.nn.models import PNA

from diffusion_co_design.common.nn.geometric import Connectivity, KNN, graph_topology
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
        n_layers: int = 4,
        post_hook: torch.nn.Module | None = None,
        connectivity: Connectivity = KNN(k=5),
    ):
        super().__init__(post_hook=post_hook)
        self.scenario = cfg
        self.n_turbines = self.scenario.n_turbines  # N

        if isinstance(connectivity, KNN):
            deg = torch.zeros(connectivity.k + 1, dtype=torch.long)
            deg[-1] = cfg.get_num_agents()
        else:
            deg = torch.zeros(cfg.get_num_agents() + 1, dtype=torch.long)
            deg[-1] = cfg.get_num_agents()
        self.connectivity = connectivity

        self.model = PNA(
            aggregators=["sum", "mean", "min", "max", "std"],
            scalers=["identity"],
            deg=deg,
            in_channels=5,
            edge_dim=3,
            hidden_channels=node_emb_dim,
            num_layers=n_layers,
            out_channels=node_emb_dim,
            act="relu",
        )

        self.out_mlp = nn.Sequential(
            nn.Linear(node_emb_dim, node_emb_dim),
            nn.ReLU(),
            nn.Linear(node_emb_dim, 1),
        )

    def _forward(
        self,
        layout: torch.Tensor,  # [*B, N, 2]
    ):
        device = layout.device
        has_batch = len(layout.shape) > 2
        if not has_batch:
            layout = layout.unsqueeze(0)

        B_all = layout.shape[:-2]
        B, N = prod(B_all), self.n_turbines

        # Shape checks
        layout = layout.reshape(-1, N, 2)
        assert layout.shape == (B, N, 2), layout.shape

        data_list: list[Data] = []
        for i in range(B):
            pos = layout[i]
            edge_index = graph_topology(pos, connectivity=KNN(k=5)).to(device)

            src, dst = edge_index
            pos_diff = pos[dst] - pos[src]
            radial = torch.norm(pos_diff, dim=-1, keepdim=True)

            edge_attr = torch.cat([radial, pos_diff], dim=-1)  # [E, 3]
            x = torch.cat(
                [scatter(edge_attr, src, dim=0, dim_size=N, reduce="mean"), pos], dim=-1
            )

            assert edge_attr.shape == (edge_index.shape[1], 3), edge_attr.shape
            assert x.shape == (N, 5), x.shape

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
            data_list.append(data)

        data = Batch.from_data_list(data_list)

        # Feature engineering
        h = self.model(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=data.batch,
        )
        assert h.shape == (B, N, self.node_emb_dim)
        h = h.reshape(*B_all, N, self.node_emb_dim)
        h = self.out_mlp(h)  # [*B, N, 1]
        h = h.mean(dim=-2)  # [*B, 1]
        if not has_batch:
            h = h.squeeze(0)

        return h.squeeze(-1)  # [*B]
