import torch
import torch.nn as nn
from guided_diffusion.unet import SimpleFlowModel
from guided_diffusion.script_util import diffusion_defaults, create_gaussian_diffusion

from diffusion_co_design.wfcrl.schema import ScenarioConfig


class WfcrlDiffusionMLP(nn.Module):
    def __init__(self, scenario: ScenarioConfig):
        super().__init__()
        self.scenario = scenario
        self.model = SimpleFlowModel(
            data_shape=(2 * scenario.n_turbines, 1),
            hidden_dim=512,
        )

    def forward(self, pos: torch.Tensor, timesteps):
        B = pos.shape[0]
        pos = pos.reshape(B, -1)
        out = self.model(pos, timesteps)
        return out.reshape(B, self.scenario.n_turbines, 2)


def diffusion_setup(
    scenario: ScenarioConfig,
    diffusion_steps: int = 1000,
):
    diffusion_args = diffusion_defaults()
    del diffusion_args["diffusion_steps"]
    del diffusion_args["use_ldm"]
    del diffusion_args["ldm_config_path"]
    diffusion_args["steps"] = diffusion_steps
    diffusion_args["timestep_respacing"] = "ddim50"

    diffusion = create_gaussian_diffusion(**diffusion_args)
    model = WfcrlDiffusionMLP(scenario=scenario)

    return model, diffusion
