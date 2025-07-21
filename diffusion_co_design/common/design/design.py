from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from multiprocessing.synchronize import Lock
from typing import Any, Literal

from tensordict import TensorDict
from tensordict.nn import TensorDictModule
import torch
import torch.nn as nn
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torchrl.objectives.value.functional import reward2go

from diffusion_co_design.common.design.base import (
    DesignProducer,
    ENVIRONMENT_DESIGN_KEY,
)
from diffusion_co_design.common.design.diffusion import (
    OptimizerDetails,
    DiffusionOperation,
    BaseGenerator,
)
from diffusion_co_design.common.pydra import Config


class DesignerConfig(Config):
    kind: str  # type: ignore
    environment_repeats: int = 1


@dataclass
class DesignerParams[SC]:
    scenario: SC
    artifact_dir: Path
    lock: Lock
    environment_repeats: int = 1


class Designer[SC](DesignProducer):
    def __init__(self, designer_setting: DesignerParams[SC]):
        super().__init__(
            designer_setting.artifact_dir,
            designer_setting.lock,
            designer_setting.environment_repeats,
        )
        self.scenario = designer_setting.scenario


class RandomDesigner[SC](Designer[SC]):
    def generate_layout_batch(self, batch_size: int):
        return self.generate_random_layouts(batch_size)

    @abstractmethod
    def generate_random_layouts(self, batch_size: int):
        raise NotImplementedError()


class FixedDesigner[SC](Designer[SC]):
    def __init__(self, designer_setting: DesignerParams[SC], layout: Any):
        super().__init__(designer_setting)
        self.layout = torch.nn.Parameter(layout, requires_grad=False)

    def generate_layout_batch(self, batch_size: int):
        return [self.layout.data for _ in range(batch_size)]


@dataclass
class ValueLearnerHyperparameters:
    buffer_size: int = 2048
    train_batch_size: int = 64
    lr: float = 3e-5
    weight_decay: float = 0.0
    train_early_start: int = 0
    clip_grad_norm: float | None = 1.0
    distill_from_critic: bool = False
    distill_samples: int = 1
    n_update_iterations: int = 5
    loss_criterion: Literal["mse", "huber"] = "mse"


class ValueLearner:
    def __init__(
        self,
        model: nn.Module,
        group_name: str,
        episode_steps: int,
        gamma: float = 0.99,
        hyperparameters: ValueLearnerHyperparameters = ValueLearnerHyperparameters(),
        group_aggregation: Literal["mean", "sum"] = "mean",
        device: torch.device = torch.device("cpu"),
    ):
        hp = hyperparameters
        match hp.loss_criterion:
            case "huber":
                self.criterion: torch.nn.Module = torch.nn.HuberLoss()
            case "mse":
                self.criterion = torch.nn.MSELoss()
            case _:
                raise ValueError(
                    f"Unknown loss criterion: {hp.loss_criterion}. Use 'mse' or 'huber'. "
                )

        self.model = model
        self.model.to(device=device)
        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay
        )

        self.buffer_size = hp.buffer_size
        self.train_batch_size = hp.train_batch_size
        self.env_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=hp.buffer_size),
            sampler=SamplerWithoutReplacement(drop_last=True),
            batch_size=hp.train_batch_size,
        )
        self.n_update_iterations = hp.n_update_iterations
        self.device = device
        self.gamma = gamma
        self.train_early_start = hp.train_early_start
        self.clip_grad_norm = hp.clip_grad_norm

        # Need to override
        self.group_name = group_name
        self.episode_steps = episode_steps
        self.group_aggregation = group_aggregation

        # Critic distillation
        self.initialised_critic = False
        self.use_critic_distillation = hp.distill_from_critic
        self.distill_samples = hp.distill_samples

        self.is_training = False

    def initialise_critic_distillation(self, critic, ref_env):
        self.initialised_critic = True
        self.critic: nn.Module = critic  # Agent critic to distill from
        self.ref_env = ref_env  # Reference environment used to calculate the state distribution from layout

    def update(self, td: TensorDict):
        self.is_training = False

        # Update replay buffer
        X, y = self._get_env_from_td(td)
        X_post = self._eval_to_train(X)
        data = TensorDict(
            {
                "env": X,  # Raw environment
                "env_post": X_post,  # Post-processed environment
                "episode_reward": y,  # Episode reward (discounted)
            },
            batch_size=len(y),
        )
        self.env_buffer.extend(data)

        assert self.initialised_critic or not self.use_critic_distillation, (
            "Critic distillation not initialised"
        )

        # Train
        if len(self.env_buffer) >= max((self.train_batch_size, self.train_early_start)):
            self.is_training = True

            self.running_loss = 0.0
            train_y_batch_list = []
            self.model.train()
            for _ in range(self.n_update_iterations):
                self.optim.zero_grad()
                sample = self.env_buffer.sample(batch_size=self.train_batch_size)
                X_batch = sample.get("env_post").to(
                    dtype=torch.float32, device=self.device
                )

                # Get output
                if self.use_critic_distillation:
                    X = sample.get("env")
                    observations_tds_list: list[torch.Tensor] = []
                    for theta in X:
                        self.ref_env._env._reset_policy = TensorDictModule(
                            module=lambda: theta,
                            in_keys=[],
                            out_keys=[ENVIRONMENT_DESIGN_KEY],
                        )
                        observations_tds_list.append(
                            torch.stack(
                                [
                                    self.ref_env.reset()
                                    for _ in range(self.distill_samples)
                                ]
                            )
                        )
                    observations_tds = torch.stack(observations_tds_list)
                    self.critic.eval()
                    with torch.no_grad():
                        y_batch = self.critic(observations_tds).get(
                            (self.group_name, "state_value")
                        )
                        assert y_batch.shape == (
                            self.train_batch_size,
                            self.distill_samples,
                            y_batch.shape[-2],  # Number of agents
                            1,
                        ), y_batch.shape
                        y_batch = y_batch.mean(dim=-2)  # Mean instead of sum
                        y_batch = y_batch.mean(dim=-2)
                        y_batch = y_batch.squeeze(-1)
                        assert y_batch.shape == (self.train_batch_size,)
                else:
                    y_batch = sample.get("episode_reward").to(
                        dtype=torch.float32, device=self.device
                    )

                train_y_batch_list.append(y_batch)

                # Timesteps
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                if self.clip_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.clip_grad_norm
                    )
                    self.grad_norm = grad_norm.item()
                self.running_loss += loss.item()
                self.optim.step()

            self.running_loss = self.running_loss / self.n_update_iterations
            train_y_batch = torch.cat(train_y_batch_list)

            self.train_y_mean = train_y_batch.mean().item()
            self.train_y_max = train_y_batch.max().item()
            self.train_y_min = train_y_batch.min().item()

        sampling_y_pred = self.model(X_post)
        self.sampling_y_pred_mean = sampling_y_pred.mean().item()
        self.sampling_y_pred_max = sampling_y_pred.max().item()
        self.sampling_y_pred_min = sampling_y_pred.min().item()

    def _get_env_from_td(self, td: TensorDict):
        assert self.group_name is not None, "Group name must be set"
        assert self.episode_steps is not None, "Episode steps must be set"

        done = td.get(("next", "done"))
        X = td.get(("state", "layout"))[done.squeeze(-1)].to(dtype=torch.float32)
        reward = td.get(("next", self.group_name, "reward"))
        match self.group_aggregation:
            case "mean":
                reward = reward.mean(dim=-2)
            case "sum":
                reward = reward.sum(dim=-2)
            case _:
                raise ValueError(
                    f"Unknown group aggregation method: {self.group_aggregation}. Use 'mean' or 'sum'."
                )

        y = reward2go(reward, done=done, gamma=self.gamma, time_dim=-2)
        y = y.reshape(-1, self.episode_steps)
        y = y[:, 0]
        return X, y

    @abstractmethod
    def _eval_to_train(self, theta: TensorDict):
        raise NotImplementedError()


class ValueDesigner[SC](Designer[SC]):
    def __init__(
        self,
        designer_setting: DesignerParams[SC],
        value_learner: ValueLearner,
        random_generation_early_start: int = 0,
    ):
        super().__init__(designer_setting)
        self.value_learner = value_learner

        self.random_generation_early_start = random_generation_early_start
        self.ref_layouts = self._make_reference_layouts()

    def update(self, sampling_td):
        super().update(sampling_td)
        self.value_learner.update(sampling_td)

    def get_state(self):
        return {
            "model": self.model,
            "training_buffer": self.value_learner.env_buffer,
        }

    def get_logs(self):
        logs = super().get_logs()
        logs.update(
            {
                "sampling_y_pred_mean": self.value_learner.sampling_y_pred_mean,
                "sampling_y_pred_max": self.value_learner.sampling_y_pred_max,
                "sampling_y_pred_min": self.value_learner.sampling_y_pred_min,
            }
        )
        if self.ref_layouts is not None:
            self.model.eval()
            ref_y_pred = self.model(self.ref_layouts)
            logs.update(
                {
                    "ref_y_pred_mean": ref_y_pred.mean().item(),
                    "ref_y_pred_max": ref_y_pred.max().item(),
                    "ref_y_pred_min": ref_y_pred.min().item(),
                }
            )

        if self.value_learner.is_training:
            logs.update(
                {
                    "train_y_mean": self.value_learner.train_y_mean,
                    "train_y_max": self.value_learner.train_y_max,
                    "train_y_min": self.value_learner.train_y_min,
                    "designer_loss": self.value_learner.running_loss,
                }
            )
            if self.value_learner.clip_grad_norm is not None:
                logs["grad_norm"] = self.value_learner.grad_norm

        return logs

    @property
    def model(self):
        return self.value_learner.model

    def _make_reference_layouts(self):
        return None

    def generate_layout_batch(self, batch_size: int):
        if len(self.value_learner.env_buffer) < self.random_generation_early_start:
            return self._generate_random_layout_batch(batch_size)
        else:
            return self._generate_layout_batch(batch_size)

    @abstractmethod
    def _generate_layout_batch(self, batch_size: int):
        raise NotImplementedError()

    def _generate_random_layout_batch(self, batch_size: int):
        raise NotImplementedError()


class SamplingDesigner[SC](ValueDesigner[SC]):
    def __init__(
        self,
        designer_setting: DesignerParams[SC],
        value_learner: ValueLearner,
        random_generation_early_start: int = 0,
        n_samples: int = 5,
    ):
        super().__init__(
            designer_setting=designer_setting,
            value_learner=value_learner,
            random_generation_early_start=random_generation_early_start,
        )
        self.n_samples = n_samples

    def _generate_layout_batch(self, batch_size):
        self.model.eval()
        X = self.generate_random_layouts(batch_size * self.n_samples)
        X_post = self.value_learner._eval_to_train(X)
        y = self.model(X_post).squeeze()
        y = y.reshape(batch_size, self.n_samples)
        indices = y.argmax(dim=1).numpy(force=True)
        return [X[i * self.n_samples + j] for i, j in enumerate(indices)]

    @abstractmethod
    def generate_random_layouts(self, batch_size: int):
        raise NotImplementedError()


class DicodeDesigner[SC](ValueDesigner[SC]):
    def __init__(
        self,
        designer_setting: DesignerParams[SC],
        value_learner: ValueLearner,
        diffusion_generator: BaseGenerator,
        diffusion_setting: DiffusionOperation,
        random_generation_early_start: int = 0,
        total_annealing_iters: int = 1000,
    ):
        super().__init__(
            designer_setting=designer_setting,
            value_learner=value_learner,
            random_generation_early_start=random_generation_early_start,
        )
        self.generator = diffusion_generator
        self.diffusion = diffusion_setting
        self.total_iters = total_annealing_iters
        self.forward_guidance_weight = self.diffusion.forward_guidance_wt

    def _generate_layout_batch(self, batch_size: int):
        forward_enable = self.diffusion.forward_guidance_wt > 0

        operation = OptimizerDetails()

        # Annealing
        if self.diffusion.forward_guidance_annealing:
            mult = min(1.0, self.update_counter / self.total_iters)
            wt = self.diffusion.forward_guidance_wt * mult
            self.forward_guidance_weight = wt
            operation.forward_guidance_wt = wt
        else:
            operation.forward_guidance_wt = self.diffusion.forward_guidance_wt

        operation.num_recurrences = self.diffusion.num_recurrences
        operation.lr = self.diffusion.backward_lr
        operation.backward_steps = self.diffusion.backward_steps
        operation.use_forward = forward_enable
        operation.projection_constraint = self.projection_constraint

        self.model.eval()

        return list(
            self.generator.generate_batch(
                value=self.model,
                use_operation=True,
                operation_override=operation,
                batch_size=batch_size,
            )
        )

    def get_logs(self):
        logs = super().get_logs()
        logs["forward_guidance_weight"] = self.forward_guidance_weight
        return logs

    def projection_constraint(self, x):
        return x
