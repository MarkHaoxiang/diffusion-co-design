import torch
from torch import nn
from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.respace import SpacedDiffusion, _WrappedModel


# Using Universal Guided Diffusion


class OptimizerDetails:
    def __init__(self):
        self.num_recurrences = 4  # Self-recurrence to increase inference time compute
        self.operation_func = None  # Set as none, probably some sort of mask
        self.optimizer = "Adam"  # Optim for backprop on expected image
        self.lr = 0.0002
        self.loss_func = None  # Important: this is the guidance model. Also 'criterion', takes in pred_xstart and any extra info in operated_image
        self.backward_steps = 0
        self.loss_cutoff = None  # disabled, ignore
        self.lr_scheduler = None
        self.warm_start = None
        self.old_img = None  # Ignore
        self.fact = 0  # Ignore
        self.print = False  # Ignore
        self.print_every = None  # Ignore
        self.folder = None  # Ignore
        self.tv_loss = None  # Ignore
        self.use_forward = True  # Set true to use forward. Not sure if this should be enabled when using backward.
        self.forward_guidance_wt = 5.0
        self.other_guidance_func = None
        self.other_criterion = None
        self.original_guidance = False
        self.sampling_type = "ddim"
        self.loss_save = None
        self.operated_image = None


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
        generator_model_path: str,
        num_channels: int,
        size: int,
        batch_size: int = 32,
        rng: torch.Generator | None = None,
        guidance_wt: float = 50.0,
    ):
        super().__init__()

        self.size = size
        self.batch_size = batch_size
        self.image_channels = num_channels
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
            dist_util.load_state_dict(generator_model_path, map_location="cpu")
        )
        self.model.to(device)
        self.model.eval()
        if isinstance(self.diffusion, SpacedDiffusion):
            self.model = _WrappedModel(
                self.model,
                self.diffusion.timestep_map,
                self.diffusion.rescale_timesteps,
                self.diffusion.original_num_steps,
            )

        dist_util.setup_dist()

        # Schedule sampler (for training time dependent diffusion)
        self.schedule_sampler = create_named_schedule_sampler("uniform", self.diffusion)

    def generate_batch(
        self,
        value: nn.Module | None = None,
        use_operation: bool = False,
        operation_override: OptimizerDetails | None = None,
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
            assert value is not None
            operation = (
                OptimizerDetails() if operation_override is None else operation_override
            )
            operation.forward_guidance_wt = self.guidance_weight

            def criterion(x: torch.Tensor, additional: torch.Tensor | None = None):
                if additional is not None:
                    additional_batch = additional.unsqueeze(0).expand(
                        x.shape[0], -1, -1, -1
                    )
                    x = torch.cat([x, additional_batch], dim=1)
                return -value(x)

            operation.loss_func = criterion
            sample = self.diffusion.ddim_sample_loop_operation(
                model=self.model,
                shape=self.shape,
                noise=initial_noise,
                operated_image=operation.operated_image,
                operation=operation,
                model_kwargs={},
            )

        # Storage
        sample = (
            ((sample + 1) * 0.5)
            .clamp(0, 1)
            .round()
            .to(torch.uint8)
            # .permute(0, 2, 3, 1)
            .contiguous()
        )
        return sample.numpy(force=True)

    @property
    def shape(self):
        return (self.batch_size, self.image_channels, self.size, self.size)
