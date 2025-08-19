from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
from diffusion_co_design.common.nn import EnvCritic
from guided_diffusion import dist_util
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.respace import SpacedDiffusion, _WrappedModel
from diffusion_co_design.common.pydra import Config


class DiffusionOperation(Config):
    num_recurrences: int = 4
    backward_lr: float = 0.01
    backward_steps: int = 0
    forward_guidance_wt: float = 5.0
    forward_guidance_annealing: bool = False


@dataclass
class OptimizerDetails:
    def __init__(self):
        self.num_recurrences = 4  # Self-recurrence to increase inference time compute
        self.operation_func = None  # Set as none, probably some sort of mask
        self.optimizer = "Adam"  # Optim for backprop on expected image
        self.lr = 0.01
        self.loss_func = None  # Important: this is the guidance model. Also 'criterion', takes in pred_xstart and any extra info in operated_image
        self.backward_steps = 0
        self.backward_mixed_precision: bool = True
        self.loss_cutoff = None  # disabled, ignore
        self.lr_scheduler = None
        self.warm_start = None
        self.old_img = None  # Ignore
        self.fact = 0  # Ignore
        self.print = False  # Ignore
        self.print_every = None  # Ignore
        self.folder = None  # Ignore
        self.use_forward = True  # Set true to use forward.
        self.forward_guidance_wt = None
        self.other_guidance_func = None
        self.other_criterion = None
        self.original_guidance = False
        self.sampling_type = "ddim"
        self.loss_save = None
        self.operated_image = None
        self.projection_constraint = None  # Projection constraint after on z_(t-1)


class BaseGenerator(ABC):
    def __init__(
        self,
        generator_model_path: str,
        model: nn.Module,
        diffusion: SpacedDiffusion,
        batch_size: int = 32,
        rng: torch.Generator | None = None,
        default_guidance_wt: float = 50.0,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        self.batch_size = batch_size
        self.clip_denoised = True

        self.guidance_weight = default_guidance_wt

        if rng is None:
            self.rng = torch.Generator(device)
        else:
            self.rng = rng

        # Create diffusion mode

        self.model, self.diffusion = model, diffusion

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

        self.device = device

        # Schedule sampler (for training time dependent diffusion)
        self.schedule_sampler = create_named_schedule_sampler("uniform", self.diffusion)

    def generate_batch(
        self,
        batch_size: int | None = None,
        value: EnvCritic | None = None,
        use_operation: bool = False,
        operation_override: OptimizerDetails | None = None,
    ):
        shape = self.shape(batch_size)
        initial_noise = torch.randn(shape, generator=self.rng, device=self.device)

        if value is not None:

            def cond_fn(x: torch.Tensor, t):
                with torch.enable_grad():
                    x_in = x.detach().requires_grad_(True)
                    out = (
                        value.predict_theta_value(x_in, t).sum() * self.guidance_weight
                    )
                    return torch.autograd.grad(outputs=out, inputs=x_in)[0]
        else:
            cond_fn = None

        if not use_operation:
            sample = self.diffusion.ddim_sample_loop(
                model=self.model,
                shape=shape,
                noise=initial_noise,
                clip_denoised=self.clip_denoised,
                cond_fn=cond_fn,
            )
        else:
            assert value is not None
            operation = (
                OptimizerDetails() if operation_override is None else operation_override
            )

            if operation.forward_guidance_wt is None:
                operation.forward_guidance_wt = self.guidance_weight

            def criterion(x: torch.Tensor, additional: torch.Tensor | None = None):
                # if additional is not None:
                #     additional_batch = additional.unsqueeze(0).expand(
                #         x.shape[0], -1, -1, -1
                #     )
                #     x = torch.cat([x, additional_batch], dim=1)
                assert additional is None
                return -value.predict_theta_value(x)

            operation.loss_func = criterion
            sample = self.diffusion.ddim_sample_loop_operation(
                model=self.model,
                shape=shape,
                noise=initial_noise,
                operated_image=operation.operated_image,
                operation=operation,
                model_kwargs={},
            )

        # Storage
        return sample

    @abstractmethod
    def shape(self, batch_size: int | None = None) -> tuple[int, ...]:
        raise NotImplementedError()
