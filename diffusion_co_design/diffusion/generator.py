import torch
from torch import Generator

from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel


class GeneratorConfig:
    generator_model_path: str
    size: int = 32
    batch_size: int = 32


class RwareGenerator:
    def __init__(self, cfg: GeneratorConfig, rng: Generator | None = None):
        super().__init__()

        self.size = cfg.size
        self.batch_size = cfg.batch_size
        self.image_channels = 3
        self.clip_denoised = True

        if rng is None:
            self.rng = Generator(device=dist_util.dev())
        else:
            self.rng = rng

        # Create diffusion model
        model_diffusion_args = model_and_diffusion_defaults()
        model_diffusion_args["image_size"] = self.size
        model_diffusion_args["image_channels"] = self.image_channels
        model_diffusion_args["num_channels"] = 128
        model_diffusion_args["num_res_blocks"] = 3
        model_diffusion_args["diffusion_steps"] = 1000
        model_diffusion_args["noise_schedule"] = "linear"
        model_diffusion_args["timestep_respacing"] = "ddim50"

        self.model, self.diffusion = create_model_and_diffusion(**model_diffusion_args)

        self.model.load_state_dict(
            dist_util.load_state_dict(cfg.generator_model_path, map_location="cpu")
        )
        dist_util.setup_dist()
        self.model.to(dist_util.dev())
        self.model.eval()

    def generate_batch(self):

        initial_noise = torch.randn((self.batch_size, *self.shape), generator=self.rng)

        # TODO: I'm not sure why this is needed
        def model_fn(x, t, y=None):
            return self.model(x, t, None)

        sample = self.diffusion.ddim_sample_loop(
            model=model_fn,
            shape=self.shape,
            noise=initial_noise,
            clip_denoised=self.clip_denoised,
        )
        sample = (
            ((sample + 1) * 127.5)
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .contiguous()
        )

        # TODO: The distributed loop is weird: why is all_gather needed here?
        gathered_samples = [
            torch.zeros_like(sample) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(
            gathered_samples, sample
        )  # gather not supported with NCCL

        return [sample.cpu().numpy() for sample in gathered_samples]

    @property
    def shape(self):
        return (self.batch_size, self.image_channels, self.size, self.size)
