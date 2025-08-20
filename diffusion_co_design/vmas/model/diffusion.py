from math import prod
import torch
import torch.nn as nn
from guided_diffusion.unet import SimpleFlowModel
from guided_diffusion.script_util import diffusion_defaults, create_gaussian_diffusion

from diffusion_co_design.vmas.schema import ScenarioConfigType


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
    model = DiffusionMLP(scenario=scenario)

    return model, diffusion
