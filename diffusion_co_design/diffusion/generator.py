import numpy as np
import torch
from torch import Generator

from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)

from pydantic import BaseModel


class GeneratorConfig(BaseModel):
    generator_model_path: str
    size: int = 16
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

        initial_noise = torch.randn(
            self.shape, generator=self.rng, device=dist_util.dev()
        )

        sample = self.diffusion.ddim_sample_loop(
            model=self.model,
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
        return np.concatenate(
            [sample.cpu().numpy() for sample in gathered_samples], axis=0
        )

    @property
    def shape(self):
        return (self.batch_size, self.image_channels, self.size, self.size)


if __name__ == "__main__":
    import os
    import shutil
    from diffusion_co_design.utils import BASE_DIR
    from PIL import Image

    cfg = GeneratorConfig(
        generator_model_path=os.path.join(
            BASE_DIR, "diffusion_pretrain", "default", "model100000.pt"
        )
    )

    # Generate a unguided batch
    generator = RwareGenerator(cfg)
    environment_batch = generator.generate_batch()

    # Save to disk
    data_dir = "test_diffusion_output"
    if os.path.exists(data_dir):
        shutil.rmtree(path=data_dir)
    os.makedirs(data_dir)

    for i, env in enumerate(environment_batch):
        im = Image.fromarray(env)
        im.save(data_dir + "/%04d" % i + ".png")
