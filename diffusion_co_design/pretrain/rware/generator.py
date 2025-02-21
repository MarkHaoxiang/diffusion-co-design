from functools import partial
import numpy as np
import torch
from pydantic import BaseModel
from torch import nn
from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
from guided_diffusion.resample import create_named_schedule_sampler

from diffusion_co_design.pretrain.rware.generate import generate
from diffusion_co_design.pretrain.rware.transform import storage_to_rgb


# Using Universal Guided Diffusion


class OptimizerDetails:
    def __init__(self):
        self.num_recurrences = 4
        self.operation_func = None
        self.optimizer = "Adam"
        self.lr = 0.0002
        self.loss_func = None
        self.backward_steps = 0
        self.loss_cutoff = None
        self.lr_scheduler = None
        self.warm_start = None
        self.old_img = None
        self.fact = 0.5
        self.print = False
        self.print_every = None
        self.folder = None
        self.tv_loss = None
        self.use_forward = True
        self.forward_guidance_wt = 5.0
        self.other_guidance_func = None
        self.other_criterion = None
        self.original_guidance = False
        self.sampling_type = "ddim"
        self.loss_save = None


class GeneratorConfig(BaseModel):
    generator_model_path: str
    size: int = 16
    batch_size: int = 32
    num_channels: int = 1


device = dist_util.dev()


def get_model_and_diffusion_defaults(size, image_channels):
    model_diffusion_args = model_and_diffusion_defaults()
    model_diffusion_args["image_size"] = size
    model_diffusion_args["image_channels"] = image_channels
    model_diffusion_args["num_channels"] = 128
    model_diffusion_args["num_res_blocks"] = 3
    model_diffusion_args["diffusion_steps"] = 1000
    model_diffusion_args["noise_schedule"] = "linear"
    model_diffusion_args["timestep_respacing"] = "ddim50"

    return model_diffusion_args


class Generator:
    def __init__(
        self,
        cfg: GeneratorConfig,
        rng: torch.Generator | None = None,
        guidance_wt: float = 10.0,
    ):
        super().__init__()

        self.size = cfg.size
        self.batch_size = cfg.batch_size
        self.image_channels = cfg.num_channels
        self.clip_denoised = True

        self.guidance_weight = guidance_wt

        if rng is None:
            self.rng = torch.Generator(device)
        else:
            self.rng = rng

        # Create diffusion model
        model_diffusion_args = get_model_and_diffusion_defaults(
            self.size, self.image_channels
        )

        self.model, self.diffusion = create_model_and_diffusion(**model_diffusion_args)

        self.model.load_state_dict(
            dist_util.load_state_dict(cfg.generator_model_path, map_location="cpu")
        )
        dist_util.setup_dist()
        self.model.to(device)
        self.model.eval()

        # Schedule sampler (for training time dependent diffusion)
        self.schedule_sampler = create_named_schedule_sampler("uniform", self.diffusion)

    def generate_batch(
        self, value: nn.Module | None = None, use_operation: bool = False
    ):
        initial_noise = torch.randn(self.shape, generator=self.rng, device=device)

        if value is not None:

            def cond_fn(x: torch.Tensor, t):
                with torch.enable_grad():
                    x_in = x.detach().requires_grad_(True)
                    out = value(x_in, t).sum() * self.guidance_weight
                    return torch.autograd.grad(outputs=out, inputs=x_in)[0]
        else:
            cond_fn = None

        if not use_operation:
            sample = self.diffusion.ddim_sample_loop(
                model=self.model,
                shape=self.shape,
                noise=initial_noise,
                clip_denoised=self.clip_denoised,
                cond_fn=cond_fn,
            )
        else:
            operation = OptimizerDetails()
            sample = self.diffusion.ddim_sample_loop_operation(
                model=self.model,
                shape=self.shape,
                noise=initial_noise,
                operated_image=None,
                operation=operation,
                cond_fn=cond_fn,
            )

        # Storage
        if self.image_channels == 3:
            sample = (
                ((sample + 1) * 127.5)
                .clamp(0, 255)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .contiguous()
            )
        elif self.image_channels == 1:
            sample = (
                ((sample + 1) * 0.5)
                .clamp(0, 1)
                .round()
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .contiguous()
            )
        else:
            assert False, "3 channels for RGB, 1 channel for storage only"
        return sample.numpy(force=True)

    @property
    def shape(self):
        return (self.batch_size, self.image_channels, self.size, self.size)


# uv run python -m diffusion_co_design.pretrain.rware.generator unguided
# uv run python -m diffusion_co_design.pretrain.rware.generator guided

if __name__ == "__main__":
    import argparse
    import os
    import shutil
    from PIL import Image

    from guided_diffusion.script_util import create_classifier, classifier_defaults

    from diffusion_co_design.utils import OUTPUT_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "option",
        type=str,
        choices=["unguided", "guided"],
    )
    args = parser.parse_args()

    cfg = GeneratorConfig(
        generator_model_path=os.path.join(
            OUTPUT_DIR, "diffusion_pretrain", "rware_16_50_5_5_random", "model100000.pt"
        )
    )

    if args.option == "unguided":
        # Generate a unguided batch
        generator = Generator(cfg)
        environment_batch = generator.generate_batch()
    elif args.option == "guided":
        model_dict = classifier_defaults()
        model_dict["image_size"] = cfg.size
        # model_dict["image_channels"] = 3
        model_dict["image_channels"] = 1
        model_dict["classifier_width"] = 128
        model_dict["classifier_depth"] = 2
        model_dict["classifier_attention_resolutions"] = "16, 8, 4"
        model_dict["output_dim"] = 1
        model = create_classifier(**model_dict).to(device)

        # class PseudoValue(torch.nn.Module):
        #     def __init__(self, size):
        #         super().__init__()
        #         self.size = size

        #     def forward(self, x, t):
        #         indices = (
        #             torch.arange(self.size, dtype=torch.float32)
        #             .view(-1, 1)
        #             .expand(self.size, self.size)
        #         ).to(x.device)
        #         channel = x[..., 0, :, :]
        #         return (channel * indices).sum(dim=(-2, -1))
        # model = PseudoValue(cfg.size)
        generator = Generator(cfg, guidance_wt=10)
        environment_batch = generator.generate_batch(value=model)
    else:
        raise NotImplementedError()

    # Save to disk
    data_dir = "test_diffusion_output"
    if os.path.exists(data_dir):
        shutil.rmtree(path=data_dir)
    os.makedirs(data_dir)

    for i, env in enumerate(environment_batch):
        env = env.transpose(2, 0, 1)
        print(env.sum())
        env = storage_to_rgb(
            env, [1, 88, 132, 233, 162], [39, 185, 237, 238, 159], channel_first=False
        )
        im = Image.fromarray(env)
        # im = Image.fromarray(env)
        im.save(data_dir + "/%04d" % i + ".png")
