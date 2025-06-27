import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
import pickle as pkl
from typing import Literal, Generic, TypeVar

from tensordict import TensorDict
from tensordict.nn import TensorDictModule
import torch
import torch.nn as nn
from torchrl.data import ReplayBuffer
from guided_diffusion import dist_util
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.respace import SpacedDiffusion, _WrappedModel
from diffusion_co_design.common.pydra import Config


SC = TypeVar("SC", bound=Config)


class BaseDesigner(nn.Module, Generic[SC], ABC):
    def __init__(self, scenario: SC, environment_repeats: int = 1):
        super().__init__()
        self.scenario = scenario
        self.update_counter = 0
        self.environment_repeats = environment_repeats
        self.environment_repeat_counter = 0
        self.previous_environment = None

    @abstractmethod
    def _generate_environment_weights(self, objective):
        raise NotImplementedError()

    def generate_environment_weights(self, objective=None):
        self.environment_repeat_counter += 1
        if self.environment_repeat_counter >= self.environment_repeats:
            self.environment_repeat_counter = 0
            self.previous_environment = None
        if self.previous_environment is not None:
            return self.previous_environment
        else:
            self.previous_environment = self._generate_environment_weights(objective)
            return self.previous_environment

    def to_td_module(self):
        return TensorDictModule(
            self,
            in_keys=[("environment_design", "objective")],
            out_keys=[("environment_design", "layout_weights")],
        )

    def update(self, sampling_td: TensorDict):
        self.update_counter += 1

    def reset(self, **kwargs):
        self.environment_repeat_counter = 0
        self.previous_environment = None

    def get_logs(self) -> dict:
        return {}

    def get_model(self) -> None | nn.Module:
        return None

    def get_training_buffer(self) -> None | ReplayBuffer:
        return None


class BaseRandomDesigner(BaseDesigner[SC]):
    @abstractmethod
    def _generate_random_environment(self, objective):
        raise NotImplementedError()

    def forward(self, objective):
        env = self._generate_random_environment(objective)
        return env

    def _generate_environment_weights(self, objective):
        return self.forward(objective)


class ValueLearner:
    pass


class BaseValueDesigner(BaseDesigner[SC]):
    pass


class BaseSamplingDesigner(BaseValueDesigner[SC]):
    pass


class BaseDicodeDesigner(BaseValueDesigner[SC]):
    pass


class BaseDiskDesigner(BaseDesigner):
    """Hack for parallel execution"""

    # So this is hacky
    # For some reason, normal mp queues break across multiple environments
    # So we just read the shared environment generation buffer from disk instead

    def __init__(
        self,
        lock,
        artifact_dir,
        environment_repeats: int = 1,
        master_designer=None,
    ):
        super().__init__(environment_repeats=environment_repeats)
        self.is_master = master_designer is not None
        self.master_designer = master_designer
        self.lock = lock
        self.buffer_path = os.path.join(artifact_dir, "designer_buffer.pkl")

    def force_regenerate(
        self, batch_size: int, mode: Literal["train", "eval"] = "train"
    ):
        assert self.master_designer

        def regenerate():
            with self.lock:
                batch = self.master_designer.reset_env_buffer(batch_size)
                with open(self.buffer_path, "wb") as f:
                    pkl.dump(batch, f)

        match mode:
            case "train":
                with open(self.buffer_path, "rb") as f:
                    batch_len = len(pkl.load(f))
                if batch_len < batch_size:
                    regenerate()
            case "eval":
                regenerate()
            case _:
                raise ValueError(f"Unknown mode {mode}")

    def forward(self, objective):
        return self.generate_environment_weights(objective)

    def update(self, sampling_td):
        super().update(sampling_td)
        if self.is_master:
            self.master_designer.update(sampling_td)

    def reset(self, batch_size: int | None = None, **kwargs):  # type: ignore
        super().reset(**kwargs)
        if self.is_master:
            self.master_designer.reset()  # type: ignore
            if batch_size is not None:
                self.force_regenerate(batch_size=batch_size, mode="eval")
                self.environment_repeat_counter = 1

    def _generate_environment_weights(self, objective):
        with self.lock:
            with open(self.buffer_path, "rb") as f:
                batch = pkl.load(f)
                # print(len(batch), "items in buffer")
            if len(batch) <= 0 and self.is_master:
                batch = self.master_designer.reset_env_buffer()
            elif len(batch) <= 0 and not self.is_master:
                assert False, "Hack failed"
            res = batch.pop()
            with open(self.buffer_path, "wb") as f:
                pkl.dump(batch, f)
        return res

    def get_logs(self):
        if self.is_master:
            return self.master_designer.get_logs()
        return super().get_logs()

    def get_model(self):
        if self.is_master:
            return self.master_designer.get_model()
        return super().get_model()


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


class DiffusionOperation(Config):
    num_recurrences: int = 4
    backward_lr: float = 0.01
    backward_steps: int = 0
    forward_guidance_wt: float = 5.0
    forward_guidance_annealing: bool = False


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
        value: nn.Module | None = None,
        use_operation: bool = False,
        operation_override: OptimizerDetails | None = None,
    ):
        shape = self.shape(batch_size)
        initial_noise = torch.randn(shape, generator=self.rng, device=self.device)

        # print(operation_override.__dict__)
        # assert False

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
                return -value(x)

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
