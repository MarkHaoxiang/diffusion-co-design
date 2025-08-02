from abc import abstractmethod
from dataclasses import dataclass
import math
from pathlib import Path
from multiprocessing.synchronize import Lock
from typing import Any, Literal
import tempfile

from tensordict import TensorDict
from tensordict.nn import TensorDictModule
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torchrl.envs.batched_envs import BatchedEnvBase
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torchrl.objectives.value.functional import reward2go

from diffusion_co_design.common.design.base import (
    DesignProducer,
    ENVIRONMENT_DESIGN_KEY,
)
from diffusion_co_design.common.env import ScenarioConfig
from diffusion_co_design.common.nn import EnvCritic
from diffusion_co_design.common.design.diffusion import (
    OptimizerDetails,
    DiffusionOperation,
    BaseGenerator,
)
from diffusion_co_design.common.pydra import Config


class DesignerConfig(Config):
    kind: Any
    environment_repeats: int = 1


@dataclass
class DesignerParams[SC: ScenarioConfig]:
    scenario: SC
    artifact_dir: Path
    lock: Lock
    environment_repeats: int = 1
    _temp_dir: tempfile.TemporaryDirectory | None = None

    @staticmethod
    def new(scenario: SC, artifact_dir: Path, environment_repeats: int = 1):
        return DesignerParams(
            scenario=scenario,
            artifact_dir=artifact_dir,
            lock=torch.multiprocessing.Lock(),
            environment_repeats=environment_repeats,
        )

    @staticmethod
    def placeholder(scenario: SC):
        temp_dir = tempfile.TemporaryDirectory()
        params = DesignerParams.new(scenario=scenario, artifact_dir=Path(temp_dir.name))
        params._temp_dir = temp_dir
        return params


class Designer[SC: ScenarioConfig](DesignProducer):
    def __init__(self, designer_setting: DesignerParams[SC]):
        super().__init__(
            designer_setting.artifact_dir,
            designer_setting.lock,
            designer_setting.environment_repeats,
        )
        self.scenario = designer_setting.scenario


class RandomDesigner[SC: ScenarioConfig](Designer[SC]):
    def generate_layout_batch(self, batch_size: int):
        return self.generate_random_layouts(batch_size)

    @abstractmethod
    def generate_random_layouts(self, batch_size: int):
        raise NotImplementedError()


class FixedDesigner[SC: ScenarioConfig](Designer[SC]):
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
    distill_embedding_hint: bool = False
    distill_embedding_hint_loss_weight: float = 0.1
    distill_synthetic_ratio: float = 0.0
    distill_synthetic_ood_ratio: float = 1.0
    n_update_iterations: int = 5
    loss_criterion: Literal["mse", "huber"] = "mse"


def get_training_pair_from_td(
    td: TensorDict,
    group_name: str,
    group_aggregation: Literal["mean", "sum"],
    episode_steps: int,
    gamma: float,
    get_layout_from_state,
):
    done = td.get(("next", "done"))
    state = td.get("state")
    X = get_layout_from_state(state)[done.squeeze(-1)].to(dtype=torch.float32)
    reward = td.get(("next", group_name, "reward"))
    match group_aggregation:
        case "mean":
            reward = reward.mean(dim=-2)
        case "sum":
            reward = reward.sum(dim=-2)
        case _:
            raise ValueError(
                f"Unknown group aggregation method: {group_aggregation}. Use 'mean' or 'sum'."
            )

    y = reward2go(reward, done=done, gamma=gamma, time_dim=-2)
    y = y.reshape(-1, episode_steps)
    y = y[:, 0]
    return X, y


class ValueLearner[SC: ScenarioConfig]:
    def __init__(
        self,
        model: EnvCritic,
        group_name: str,
        episode_steps: int,
        gamma: float = 0.99,
        hyperparameters: ValueLearnerHyperparameters = ValueLearnerHyperparameters(),
        group_aggregation: Literal["mean", "sum"] = "mean",
        random_designer: Designer[SC] | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.hp = hp = hyperparameters
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

        self.use_hint_loss = hp.distill_embedding_hint and hp.distill_from_critic
        if self.use_hint_loss:
            self.hint_loss_weight = hp.distill_embedding_hint_loss_weight

        else:
            self.optim = Adam(
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
        self.synthetic_ratio = hp.distill_synthetic_ratio
        self.synthetic_uniform_data_ratio = hp.distill_synthetic_ood_ratio

        self.is_training = False

        self.random_designer = random_designer
        self.distribution_designer: Designer[SC] | None = None

    def initialise_critic_distillation(self, critic, ref_env):
        self.initialised_critic = True
        self.critic: TensorDictModule = critic  # Agent critic to distill from
        self.ref_env = ref_env  # Reference environment used to calculate the state distribution from layout
        if self.use_hint_loss:
            self.hint_loss_fn = self._make_hint_loss(device=self.device)
            self.optim = Adam(
                list(self.model.parameters()) + list(self.hint_loss_fn.parameters()),
                lr=self.hp.lr,
                weight_decay=self.hp.weight_decay,
            )

    def update(self, td: TensorDict):
        self.is_training = False

        # Update replay buffer
        X, y = self._get_training_pair_from_td(td)
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

            self.running_prediction_loss = 0.0
            self.running_distillation_loss = 0.0

            train_y_batch_list = []
            self.model.train()
            for _ in range(self.n_update_iterations):
                self.optim.zero_grad()

                if self.use_critic_distillation:
                    n_synthetic = int(self.train_batch_size * self.synthetic_ratio)
                    n_sampling = self.train_batch_size - n_synthetic
                    n_u = int(n_synthetic * self.synthetic_uniform_data_ratio)
                    n_gen = n_synthetic - n_u

                    assert self.train_batch_size == n_u + n_gen + n_sampling

                    X_batch_list = []
                    X_eval_list = []
                    if n_sampling > 0:
                        sample = self.env_buffer.sample(batch_size=n_sampling)
                        X_batch_list.append(
                            sample.get("env_post").to(
                                dtype=torch.float32, device=self.device
                            )
                        )
                        X_eval_gen = sample.get("env").to(
                            dtype=torch.float32, device=self.device
                        )
                        X_eval_gen = [self._eval_to_gen(x) for x in X_eval_gen]
                        X_eval_list.append(torch.stack(X_eval_gen))
                    if n_u > 0:
                        assert self.random_designer is not None
                        X_uniform = torch.stack(
                            [
                                self._gen_to_torchable(x)
                                for x in self.random_designer.generate_layout_batch(n_u)
                            ]
                        ).to(dtype=torch.float32, device=self.device)
                        X_eval_list.append(X_uniform)
                        X_batch_list.append(self._gen_to_train(X_uniform))

                    if n_gen > 0:
                        assert self.distribution_designer is not None
                        X_gen = torch.stack(
                            [
                                self._gen_to_torchable(x)
                                for x in self.distribution_designer.generate_layout_batch(
                                    n_gen
                                )
                            ],
                        ).to(dtype=torch.float32, device=self.device)
                        X_eval_list.append(X_gen)
                        X_batch_list.append(self._gen_to_train(X_gen))

                    # Sample proportion from buffer
                    X_batch = torch.cat(X_batch_list, dim=0)
                    X_eval = torch.cat(X_eval_list, dim=0)
                    y_batch, hint_batch = self._get_critic_y_from_layout(X_eval)
                else:
                    sample = self.env_buffer.sample(batch_size=self.train_batch_size)
                    X_batch = sample.get("env_post").to(
                        dtype=torch.float32, device=self.device
                    )
                    y_batch = sample.get("episode_reward").to(
                        dtype=torch.float32, device=self.device
                    )

                # Get output
                train_y_batch_list.append(y_batch)

                # Timesteps
                y_pred, hint_pred = self.model.predict_theta_value_with_hint(X_batch)

                prediction_loss = self.criterion(y_pred, y_batch)
                self.running_prediction_loss += prediction_loss.item()
                loss = prediction_loss
                if self.use_hint_loss:
                    distil_hint_loss = self.hint_loss_fn(hint_batch, hint_pred)
                    loss += distil_hint_loss * self.hint_loss_weight
                    self.running_distillation_loss += distil_hint_loss.item()

                loss.backward()
                if self.clip_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.clip_grad_norm
                    )
                    self.grad_norm = grad_norm.item()
                self.optim.step()

            self.running_prediction_loss = (
                self.running_prediction_loss / self.n_update_iterations
            )
            train_y_batch = torch.cat(train_y_batch_list)

            self.train_y_mean = train_y_batch.mean().item()
            self.train_y_max = train_y_batch.max().item()
            self.train_y_min = train_y_batch.min().item()

        sampling_y_pred = self.model.predict_theta_value(X_post)
        self.sampling_y_pred_mean = sampling_y_pred.mean().item()
        self.sampling_y_pred_max = sampling_y_pred.max().item()
        self.sampling_y_pred_min = sampling_y_pred.min().item()

    def _get_training_pair_from_td(self, td: TensorDict):
        return get_training_pair_from_td(
            td=td,
            group_name=self.group_name,
            group_aggregation=self.group_aggregation,
            episode_steps=self.episode_steps,
            gamma=self.gamma,
            get_layout_from_state=self._get_layout_from_state,
        )

    def _get_critic_y_from_layout(self, X):
        observations_tds_list: list[torch.Tensor] = []
        for theta in X:
            self.ref_env._env._reset_policy = TensorDictModule(
                module=lambda: theta,
                in_keys=[],
                out_keys=[ENVIRONMENT_DESIGN_KEY],
            )
            observations_tds_list.append(
                torch.stack([self.ref_env.reset() for _ in range(self.distill_samples)])
            )
        observations_tds = torch.stack(observations_tds_list)
        self.critic.eval()
        with torch.no_grad():
            critic_td = self.critic(observations_tds)
            y_batch = critic_td.get((self.group_name, "state_value"))
            hint_batch = critic_td.get((self.group_name, "distillation_hint"), None)
            assert y_batch.shape == (
                self.train_batch_size,
                self.distill_samples,
                y_batch.shape[-2],  # Number of agents
                1,
            ), y_batch.shape
            match self.group_aggregation:
                case "mean":
                    y_batch = y_batch.mean(dim=-2)
                case "sum":
                    y_batch = y_batch.sum(dim=-2)

            if hint_batch is not None:
                # Reduce over samples
                hint_batch = hint_batch.mean(dim=1)

            y_batch = y_batch.mean(dim=-2)
            y_batch = y_batch.squeeze(-1)
            assert y_batch.shape == (self.train_batch_size,)
            return y_batch, hint_batch

    @abstractmethod
    def _get_layout_from_state(self, state: TensorDict):
        raise NotImplementedError()

    @abstractmethod
    def _eval_to_train(self, theta: TensorDict):
        raise NotImplementedError()

    def _gen_to_train(self, theta):
        return theta

    def _eval_to_gen(self, theta):
        return theta

    def _gen_to_torchable(self, theta):
        return torch.tensor(theta, dtype=torch.float32, device=self.device)

    def _make_hint_loss(self, device: torch.device) -> nn.Module:
        raise NotImplementedError()


class ValueDesigner[SC: ScenarioConfig](Designer[SC]):
    def __init__(
        self,
        designer_setting: DesignerParams[SC],
        value_learner: ValueLearner,
        random_generation_early_start: int = 0,
    ):
        super().__init__(designer_setting)
        self.value_learner = value_learner
        self.value_learner.distribution_designer = self

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
            ref_y_pred = self.model.predict_theta_value(self.ref_layouts)
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
                    "prediction_loss": self.value_learner.running_prediction_loss,
                }
            )
            if self.value_learner.clip_grad_norm is not None:
                logs["grad_norm"] = self.value_learner.grad_norm
            if self.value_learner.use_critic_distillation is not None:
                logs["distillation_loss"] = self.value_learner.running_distillation_loss

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


class SamplingDesigner[SC: ScenarioConfig](ValueDesigner[SC]):
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
        y = self.model.predict_theta_value(X_post).squeeze()
        y = y.reshape(batch_size, self.n_samples)
        indices = y.argmax(dim=1).numpy(force=True)
        return [X[i * self.n_samples + j] for i, j in enumerate(indices)]

    @abstractmethod
    def generate_random_layouts(self, batch_size: int):
        raise NotImplementedError()


class DicodeDesigner[SC: ScenarioConfig](ValueDesigner[SC]):
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


class GradientDescentDesigner[SC: ScenarioConfig](ValueDesigner[SC]):
    def __init__(
        self,
        designer_setting: DesignerParams[SC],
        value_learner: ValueLearner,
        random_generation_early_start: int = 0,
        lr: float = 0.03,
        n_epochs: int = 10,
        n_gradient_iterations: int = 10,
    ):
        super().__init__(
            designer_setting=designer_setting,
            value_learner=value_learner,
            random_generation_early_start=random_generation_early_start,
        )
        self.lr = lr
        self.epochs = n_epochs
        self.gradient_iterations = n_gradient_iterations

    def _generate_layout_batch(self, batch_size):
        env = self._generate_initial_env(batch_size=batch_size)
        optim = torch.optim.Adam([env], lr=self.lr)

        for _ in range(self.epochs):
            env.requires_grad = True
            for _ in range(self.gradient_iterations):
                optim.zero_grad()

                y_pred = self.model.predict_theta_value(env)
                loss = -y_pred.sum()
                loss.backward()
                optim.step()

            env = self.projection_constraint(env.detach())
        env = self._train_to_eval(env=env.detach())
        batch = list(env)

        return batch

    def projection_constraint(self, x):
        return x

    def _generate_initial_env(self, batch_size: int):
        env = torch.stack(self._generate_random_layout_batch(batch_size))
        env = self.value_learner._eval_to_train(env)

    @abstractmethod
    def _train_to_eval(self, env):
        raise NotImplementedError()

    @abstractmethod
    def _generate_random_layout_batch(self, batch_size):
        raise NotImplementedError()


class ReinforceDesigner[SC: ScenarioConfig](Designer[SC]):
    def __init__(
        self,
        designer_setting: DesignerParams[SC],
        group_name: str,
        group_aggregation: Literal["mean", "sum"] = "sum",
        lr: float = 3e-4,
        train_batch_size: int = 64,
        train_epochs: int = 5,
        gamma: float = 0.99,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(designer_setting)
        self.lr = lr
        self.train_batch_size = train_batch_size
        self.train_epochs = train_epochs
        self.gamma = gamma
        self.device = device

        self.policy = self._create_policy().to(device=device)
        self.optim = Adam(self.policy.parameters(), lr=self.lr)
        self.group_name = group_name
        self.group_aggregation = group_aggregation

        self.initialised = False

    def update(self, sampling_td):
        super().update(sampling_td)

        assert self.initialised

        self.reinforce_loss = 0.0
        for _ in range(self.train_epochs):
            # Collect episode (environment, reward) pairs
            self.policy.eval()
            envs, actions = self._generate_env_action_batch(
                batch_size=self.train_batch_size
            )

            chunk_number = math.ceil(self.train_batch_size / self.train_env_batch_size)
            env_chunks = torch.chunk(envs, chunk_number, dim=0)

            rewards_list = []
            for env_chunk in env_chunks:
                envs_list = list(env_chunk)
                n = len(envs_list)
                if n < self.train_env_batch_size:
                    envs_list += [envs_list[-1]] * (
                        self.train_env_batch_size - n
                    )  # Pad if needed

                td = self.train_env.reset(
                    list_of_kwargs=[{"layout_override": env.cpu()} for env in envs_list]
                )
                td = self.train_env.rollout(
                    max_steps=self.scenario.get_episode_steps(),
                    policy=self.agent_policy,
                    auto_reset=False,
                    tensordict=td,
                )

                done = td.get(("next", "done"))
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
                y = y.reshape(-1, self.scenario.get_episode_steps())
                y = y[:n, 0]

                rewards_list.append(y)

            rewards = torch.cat(rewards_list, dim=0)

            assert rewards.shape == (self.train_batch_size,), rewards.shape

            # Reinforce
            self.policy.train()
            self.reinforce_loss += self.reinforce(actions, rewards)

            self.reinforce_loss /= self.train_epochs
            self.train_env.reset()

    def reinforce(self, actions: torch.Tensor, rewards: torch.Tensor):
        B = rewards.shape[0]
        assert rewards.shape == (B,), rewards.shape

        action_log_probs = self._calculate_action_log_probs(
            actions
        )  # [B, Steps, Probability]

        loss = -(action_log_probs * rewards.unsqueeze(-1)).sum(dim=1).mean()
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optim.step()
        return loss.item()

    def get_logs(self):
        logs = {"reinforce_loss": self.reinforce_loss}
        return logs

    def initialise(
        self,
        train_env: BatchedEnvBase,
        train_env_batch_size: int,
        agent_policy: TensorDictModule,
    ):
        self.initialised = True
        self.train_env = train_env
        self.train_env_batch_size = train_env_batch_size
        self.agent_policy = agent_policy

    @abstractmethod
    def _create_policy(self) -> nn.Module:
        raise NotImplementedError()

    @abstractmethod
    def _generate_env_action_batch(self, batch_size: int):
        raise NotImplementedError()

    @abstractmethod
    def _calculate_action_log_probs(self, actions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def generate_layout_batch(self, batch_size: int):
        envs, _ = self._generate_env_action_batch(batch_size=batch_size)
        return list(envs)


type EnvReturn = tuple[
    Any,  # Environment
    float,  # Return
    int,  # Timestep
]


class ReplayDesigner[SC: ScenarioConfig](Designer[SC]):
    def __init__(
        self,
        group_name: str,
        designer_setting: DesignerParams[SC],
        buffer_size: int = 1000,
        infill_ratio: float = 0.25,
        replay_sample_ratio: float = 0.9,
        stale_sample_ratio: float = 0.3,
        return_smoothing_factor: float = 0.8,
        return_sample_temperature: float = 0.1,
        gamma: float = 0.99,
        group_aggregation: Literal["mean", "sum"] = "sum",
    ):
        super().__init__(designer_setting=designer_setting)
        self.buffer_size = buffer_size
        self.replay_sample_ratio = replay_sample_ratio
        self.return_smoothing_factor = return_smoothing_factor
        self.beta = return_sample_temperature
        self.group_name = group_name
        self.group_aggregation = group_aggregation
        self.gamma = gamma
        self.infill_ratio = infill_ratio
        self.stale_sample_ratio = stale_sample_ratio

        self.env_buffer: dict[Any, EnvReturn] = {
            self._hash(env): (env, 0.0, -1)  # Return, Timestep
            for env in self._generate_random_layout_batch(
                batch_size=int(buffer_size * self.infill_ratio)
            )
        }

        self.rng = np.random.default_rng()

    def update(self, sampling_td: TensorDict):
        super().update(sampling_td)

        X, y = self._get_training_pair_from_td(sampling_td)

        for i in range(len(X)):
            key = self._hash(X[i])

            env, env_return, timestep = self.env_buffer.get(key, (X[i], 0.0, 0))

            # Exponential average of return
            if timestep == -1:
                env_return = y[i].item()
            else:
                env_return = env_return * self.return_smoothing_factor + (
                    y[i].item() * (1 - self.return_smoothing_factor)
                )

            timestep = self.update_counter

            value = (env, env_return, timestep)
            if key not in self.env_buffer and len(self.env_buffer) >= self.buffer_size:
                # Discard the worst environment
                worst_key = min(
                    self.env_buffer.keys(), key=lambda k: self.env_buffer[k][1]
                )
                worst_env_return = self.env_buffer[worst_key][1]
                if worst_env_return < env_return:
                    self.env_buffer.pop(worst_key)

            if key in self.env_buffer or len(self.env_buffer) < self.buffer_size:
                self.env_buffer[key] = value

    def generate_layout_batch(self, batch_size):
        _, values = zip(*self.env_buffer.items())
        envs = [value[0] for value in values]

        # Replay from buffer
        # Prioritise stale samples
        timesteps = np.array([value[2] for value in values])
        current_timestep = self.update_counter
        scores = current_timestep - timesteps
        stale_p = scores / scores.sum()

        # Prioritise high returns
        returns = np.array([value[1] for value in values])
        returns[timesteps == -1] = returns.max()  # Encourage early exploration
        ranking = returns.argsort()[::-1].argsort()
        scores = ranking ** (1 / self.beta)
        return_p = scores / scores.sum()

        n_samples = self.rng.binomial(n=batch_size, p=self.replay_sample_ratio)
        n_stale = self.rng.binomial(n=n_samples, p=self.stale_sample_ratio)
        n_return = n_samples - n_stale

        env_idxs = np.arange(len(envs))
        sample_stale = self.rng.choice(env_idxs, size=n_stale, p=stale_p, replace=False)
        sample_stale = [envs[i] for i in sample_stale]
        sample_return = self.rng.choice(
            env_idxs, size=n_return, p=return_p, replace=False
        )
        sample_return = [envs[i] for i in sample_return]

        # Mutate new
        new_environments = []
        generate_base_envs = self.rng.choice(
            env_idxs, size=batch_size - n_samples, p=return_p, replace=False
        )
        generate_base_envs = [envs[i] for i in generate_base_envs]

        for base_env in generate_base_envs:
            new_environments.append(self._mutate(base_env))

        layouts = sample_stale + sample_return + new_environments
        return layouts

    @abstractmethod
    def _generate_random_layout_batch(self, batch_size: int):
        raise NotImplementedError()

    @abstractmethod
    def _get_layout_from_state(self, state: TensorDict):
        raise NotImplementedError()

    @abstractmethod
    def _mutate(self, env: Any):
        raise NotImplementedError()

    def _get_training_pair_from_td(self, td: TensorDict):
        return get_training_pair_from_td(
            td=td,
            group_name=self.group_name,
            group_aggregation=self.group_aggregation,
            episode_steps=self.scenario.get_episode_steps(),
            gamma=self.gamma,
            get_layout_from_state=self._get_layout_from_state,
        )

    @staticmethod
    @abstractmethod
    def _hash(env):
        raise NotImplementedError()
