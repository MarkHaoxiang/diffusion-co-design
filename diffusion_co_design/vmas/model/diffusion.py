from math import prod
import torch
import torch.nn as nn
from guided_diffusion.unet import SimpleFlowModel, timestep_embedding
from guided_diffusion.script_util import diffusion_defaults, create_gaussian_diffusion

from diffusion_co_design.vmas.schema import (
    ScenarioConfigType,
    LocalPlacementScenarioConfig,
    GlobalPlacementScenarioConfig,
)


class DiffusionMLP(nn.Module):
    def __init__(self, scenario: ScenarioConfigType):
        super().__init__()
        self.scenario = scenario
        self.model = SimpleFlowModel(
            data_shape=(prod(scenario.diffusion_shape), 1),
            hidden_dim=512,
        )

    def forward(self, pos: torch.Tensor, timesteps):
        B = pos.shape[0]
        pos = pos.reshape(B, -1)
        out = self.model(pos, timesteps)
        return out.reshape(B, *self.scenario.diffusion_shape)


class IndependentDiffusionMLP(nn.Module):
    def __init__(self, scenario: LocalPlacementScenarioConfig):
        super().__init__()
        self.scenario = scenario

        hidden_dim = 64
        self.hidden_dim = 64
        self.layer_1 = nn.Linear(1, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, 1)

        self.emb_layer_1 = nn.Linear(hidden_dim, hidden_dim)
        self.emb_layer_2 = nn.Linear(hidden_dim, hidden_dim)

        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor, timesteps):
        B = x.shape[0]
        x = x.reshape(B, -1, 1)

        emb = timestep_embedding(timesteps, self.hidden_dim)
        emb = emb.unsqueeze(dim=1)

        x = self.layer_1(x)
        x = self.activation(x)
        emb = self.emb_layer_1(emb)
        emb = self.activation(emb)

        x = self.layer_2(x + emb)
        x = self.activation(x)
        emb = self.emb_layer_2(emb)
        emb = self.activation(emb)

        x = self.layer_3(x + emb)

        return x.reshape(B, *self.scenario.diffusion_shape)


def diffusion_setup(
    scenario: ScenarioConfigType,
    diffusion_steps: int = 1000,
):
    diffusion_args = diffusion_defaults()
    del diffusion_args["diffusion_steps"]
    del diffusion_args["use_ldm"]
    del diffusion_args["ldm_config_path"]
    diffusion_args["steps"] = diffusion_steps
    diffusion_args["timestep_respacing"] = "ddim50"

    diffusion = create_gaussian_diffusion(**diffusion_args)
    if isinstance(scenario, GlobalPlacementScenarioConfig):
        model: nn.Module = DiffusionMLP(scenario=scenario)
    elif isinstance(scenario, LocalPlacementScenarioConfig):
        model = IndependentDiffusionMLP(scenario=scenario)

    return model, diffusion
