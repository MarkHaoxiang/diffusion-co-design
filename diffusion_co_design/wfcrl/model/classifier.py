import torch
import torch.nn as nn

from diffusion_co_design.wfcrl.schema import ScenarioConfig


class EnvironmentCritic(nn.Module):
    def __init__(self, cfg: ScenarioConfig, embedding_size: int = 256, depth: int = 2):
        super().__init__()
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

    def predict(self, x: torch.Tensor):
        return self.forward(x)

    def forward(self, x: torch.Tensor):
        # x: [B, N, 2]
        x = x.flatten(start_dim=1)  # [B, N*2]
        return self.mlp(x).squeeze(-1)  # [B]
