import torch
import torch.nn as nn
from guided_diffusion.unet import SimpleFlowModel
from guided_diffusion.script_util import (
    diffusion_defaults,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    create_gaussian_diffusion,
)

from diffusion_co_design.rware.schema import ScenarioConfig, Representation


class WarehouseDiffusionMLP(nn.Module):
    def __init__(self, scenario: ScenarioConfig):
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


def diffusion_setup(
    scenario: ScenarioConfig,
    representation: Representation,
    diffusion_steps: int = 1000,
):
    size = scenario.size
    n_colors = scenario.n_colors
    n_shelves = scenario.n_shelves
    if representation == "image":
        model_diffusion_args = model_and_diffusion_defaults()
        model_diffusion_args["image_size"] = size
        model_diffusion_args["image_channels"] = n_colors
        model_diffusion_args["num_channels"] = 128
        model_diffusion_args["num_res_blocks"] = 3
        model_diffusion_args["diffusion_steps"] = diffusion_steps
        model_diffusion_args["timestep_respacing"] = "ddim50"

        model, diffusion = create_model_and_diffusion(**model_diffusion_args)
    elif representation in ["flat", "graph"]:
        diffusion_args = diffusion_defaults()
        del diffusion_args["diffusion_steps"]
        del diffusion_args["use_ldm"]
        del diffusion_args["ldm_config_path"]
        diffusion_args["steps"] = diffusion_steps
        diffusion_args["timestep_respacing"] = "ddim50"

        diffusion = create_gaussian_diffusion(**diffusion_args)

        if representation == "flat":
            model = SimpleFlowModel(
                # data_shape=((2 + n_colors) * n_shelves, 1),
                data_shape=(2 * n_shelves, 1),
                hidden_dim=2048,
            )
        else:
            # model = WarehouseDiffusionModel(
            #     scenario=scenario,
            #     node_embedding_dim=128,
            #     edge_embedding_dim=64,
            #     timestep_embedding_dim=64,
            #     num_layers=5,
            #     use_radius_graph=True,
            # )
            model = WarehouseDiffusionMLP(scenario=scenario)

    return model, diffusion
