import os
from abc import ABC, abstractmethod

import numpy as np
import torch
from torchrl.envs import EnvBase
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torchrl.objectives.value.functional import reward2go
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from diffusion_co_design.common.design import BaseDesigner, BaseDiskDesigner
from diffusion_co_design.wfcrl.schema import (
    ScenarioConfig,
    DesignerConfig,
    ClassifierConfig,
    PPOConfig,
    _Value,
    Fixed,
    Random,
    Diffusion,
)
from diffusion_co_design.wfcrl.model.classifier import EnvironmentCritic
from diffusion_co_design.wfcrl.diffusion.generate import Generate
from diffusion_co_design.wfcrl.diffusion.generator import (
    Generator,
    OptimizerDetails,
    eval_to_train,
    soft_projection_constraint,
)
from diffusion_co_design.common import DiffusionOperation, OUTPUT_DIR, get_latest_model


group_name = "turbine"


class Designer(BaseDesigner):
    def __init__(self, scenario: ScenarioConfig, environment_repeats: int = 1):
        super().__init__(environment_repeats=environment_repeats)
        self.scenario = scenario


class RandomDesigner(Designer):
    def __init__(
        self,
        scenario: ScenarioConfig,
        environment_repeats: int = 1,
        seed: int | None = None,
    ):
        super().__init__(scenario=scenario, environment_repeats=environment_repeats)

        self.generate = Generate(
            num_turbines=scenario.n_turbines,
            map_x_length=scenario.map_x_length,
            map_y_length=scenario.map_y_length,
            minimum_distance_between_turbines=scenario.min_distance_between_turbines,
            rng=seed,
        )

    def forward(self, objective=None):
        theta = torch.tensor(self.generate(n=1, training_dataset=False)).squeeze(0)
        return theta

    def _generate_environment_weights(self, objective):
        return self.forward(objective)


class FixedDesigner(Designer):
    def __init__(self, scenario: ScenarioConfig, seed: int | None = None):
        super().__init__(scenario)
        self.layout_image = torch.nn.Parameter(
            RandomDesigner(scenario, seed=seed)._generate_environment_weights(None),
            requires_grad=False,
        )

    def forward(self, objective):
        return self.layout_image.data

    def _generate_environment_weights(self, objective):
        return self.layout_image


def get_env_from_td(td, scenario: ScenarioConfig, gamma: float = 0.99):
    done = td.get(("next", "done"))
    X = td.get(("state", "layout"))[done.squeeze(-1)].to(dtype=torch.float32)
    reward = td.get(("next", group_name, "reward")).sum(dim=-2)
    y = reward2go(reward, done=done, gamma=gamma, time_dim=-2)
    y = y.reshape(-1, scenario.max_steps)
    y = y[:, 0]
    return X, y


class CentralisedDesigner(Designer):
    @abstractmethod
    def reset_env_buffer(self, batch_size: int) -> list:
        raise NotImplementedError

    def forward(self, objective):
        raise NotImplementedError("Use with disk designer")

    def _generate_environment_weights(self, objective):
        raise NotImplementedError("Use with disk designer")


class ValueDesigner(CentralisedDesigner, ABC):
    def __init__(
        self,
        scenario: ScenarioConfig,
        classifier: ClassifierConfig,
        gamma: float = 0.99,
        n_update_iterations: int = 5,
        train_batch_size: int = 64,
        buffer_size: int = 2048,
        lr: float = 3e-5,
        weight_decay: float = 0.0,
        environment_repeats: int = 1,
        distill_from_critic: bool = False,
        distill_samples: int = 1,
        early_start: int | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(scenario, environment_repeats=environment_repeats)
        self.model = EnvironmentCritic(
            cfg=scenario,
            embedding_size=classifier.embedding_size,
            depth=classifier.depth,
        )
        self.model = self.model.to(device)

        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = torch.nn.MSELoss()

        self.buffer_size = buffer_size
        self.train_batch_size = train_batch_size
        self.env_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=buffer_size),
            sampler=SamplerWithoutReplacement(drop_last=True),
            batch_size=self.train_batch_size,
        )

        self.n_update_iterations = n_update_iterations
        self.device = device
        self.gamma = gamma

        self.distill_from_critic = distill_from_critic
        self.distill_samples = distill_samples
        self.critic: TensorDictModule | None = None
        self.ref_env: EnvBase | None = None
        if self.distill_from_critic:
            self.manual_designer = FixedDesigner(scenario=scenario)

        self.generate = Generate(
            num_turbines=scenario.n_turbines,
            map_x_length=scenario.map_x_length,
            map_y_length=scenario.map_y_length,
            minimum_distance_between_turbines=scenario.min_distance_between_turbines,
            rng=0,
        )

        # Logging
        self.ref_random_designs = torch.tensor(
            np.array(self.generate(n=64, training_dataset=True)),
            dtype=torch.float32,
            device=self.device,
        )

        self.early_start = early_start

    @abstractmethod
    def _reset_env_buffer(self, batch_size):
        raise NotImplementedError()

    def reset_env_buffer(self, batch_size):
        if self.early_start and len(self.env_buffer) < self.early_start:
            return list(self.generate(n=batch_size))
        else:
            return self._reset_env_buffer(batch_size)

    def update(self, sampling_td):
        super().update(sampling_td)

        # Update replay buffer
        X, y = get_env_from_td(sampling_td, self.scenario, gamma=self.gamma)
        X_post = eval_to_train(X, self.scenario)

        data = TensorDict(
            {"env": X, "env_post": X_post, "episode_reward": y}, batch_size=len(y)
        )
        self.env_buffer.extend(data)

        if self.distill_from_critic:
            assert self.critic is not None
            assert self.ref_env is not None
            self.ref_env._env._reset_policy = self.manual_designer.to_td_module()

        # Train
        if len(self.env_buffer) >= self.train_batch_size:
            self.running_loss = 0
            train_y_batch = []
            self.model.train()
            for _ in range(self.n_update_iterations):
                self.optim.zero_grad()
                sample = self.env_buffer.sample(batch_size=self.train_batch_size)
                X_batch = sample.get("env_post").to(
                    dtype=torch.float32, device=self.device
                )

                if self.distill_from_critic:
                    assert self.critic is not None
                    assert self.ref_env is not None
                    X_img = sample.get("env")
                    observations_tds = []
                    for env_image in X_img:
                        observations_tds.append([])
                        self.manual_designer.layout_image.data.copy_(env_image)
                        for _ in range(self.distill_samples):
                            observations_tds[-1].append(self.ref_env.reset())
                        observations_tds[-1] = torch.stack(observations_tds[-1])
                    observations_tds = torch.stack(observations_tds)
                    self.critic.eval()
                    with torch.no_grad():
                        y_batch = self.critic(observations_tds).get(
                            (group_name, "state_value")
                        )
                        assert y_batch.shape == (
                            self.train_batch_size,
                            self.distill_samples,
                            self.scenario.n_agents,
                            1,
                        ), y_batch.shape
                        y_batch = y_batch.sum(dim=-2)
                        y_batch = y_batch.mean(dim=-2)
                        y_batch = y_batch.squeeze(-1)
                        assert y_batch.shape == (self.train_batch_size,)
                else:
                    y_batch = sample.get("episode_reward").to(
                        dtype=torch.float32, device=self.device
                    )
                train_y_batch.append(y_batch)

                # Timesteps
                y_pred = self.model.predict(X_batch)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                self.running_loss += loss.item()
                self.optim.step()
            # Logs
            self.running_loss = self.running_loss / self.n_update_iterations
            train_y_batch = torch.cat(train_y_batch)
            self.train_y_mean = train_y_batch.mean().item()
            self.train_y_max = train_y_batch.max().item()
            self.train_y_min = train_y_batch.min().item()

        classifier_prediction = self.model.predict(X_post)
        self.classifier_prediction_mean = classifier_prediction.mean().item()
        self.classifier_prediction_max = classifier_prediction.max().item()
        self.classifier_prediction_min = classifier_prediction.min().item()

        classifier_prediction_rand = self.model(self.ref_random_designs)
        self.classifier_prediction_rand_mean = classifier_prediction_rand.mean().item()
        self.classifier_prediction_rand_max = classifier_prediction_rand.max().item()
        self.classifier_prediction_rand_min = classifier_prediction_rand.min().item()

    def get_logs(self):
        logs = {
            "classifier_prediction_mean": self.classifier_prediction_mean,
            "classifier_prediction_max": self.classifier_prediction_max,
            "classifier_prediction_min": self.classifier_prediction_min,
            "classifier_prediction_rand_mean": self.classifier_prediction_rand_mean,
            "classifier_prediction_rand_max": self.classifier_prediction_rand_max,
            "classifier_prediction_rand_min": self.classifier_prediction_rand_min,
        }
        if len(self.env_buffer) >= self.train_batch_size:
            logs.update(
                {
                    "designer_loss": self.running_loss,
                    "design_y_mean": self.train_y_mean,
                    "design_y_max": self.train_y_max,
                    "design_y_min": self.train_y_min,
                }
            )
        return logs

    def get_model(self):
        return self.model

    def get_training_buffer(self):
        return self.env_buffer


class DiffusionDesigner(ValueDesigner):
    def __init__(
        self,
        scenario: ScenarioConfig,
        classifier: ClassifierConfig,
        diffusion: DiffusionOperation,
        gamma: float = 0.99,
        n_update_iterations: int = 5,
        train_batch_size: int = 64,
        buffer_size: int = 2048,
        lr: float = 3e-5,
        weight_decay: float = 0.0,
        environment_repeats: int = 1,
        distill_from_critic: bool = False,
        distill_samples: int = 1,
        early_start: int | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            scenario,
            classifier,
            gamma=gamma,
            n_update_iterations=n_update_iterations,
            train_batch_size=train_batch_size,
            buffer_size=buffer_size,
            lr=lr,
            weight_decay=weight_decay,
            environment_repeats=environment_repeats,
            distill_from_critic=distill_from_critic,
            distill_samples=distill_samples,
            early_start=early_start,
            device=device,
        )

        pretrain_dir = os.path.join(OUTPUT_DIR, "wfcrl", "diffusion", scenario.name)
        latest_checkpoint = get_latest_model(pretrain_dir, "model")

        self.generator = Generator(
            generator_model_path=latest_checkpoint,
            scenario=scenario,
            guidance_wt=diffusion.forward_guidance_wt,
            device=device,
        )

        self.diffusion = diffusion

    def _reset_env_buffer(self, batch_size: int):
        forward_enable = self.diffusion.forward_guidance_wt > 0
        operation = OptimizerDetails()
        operation.num_recurrences = self.diffusion.num_recurrences
        operation.lr = self.diffusion.backward_lr
        operation.backward_steps = self.diffusion.backward_steps
        operation.use_forward = forward_enable
        operation.projection_constraint = soft_projection_constraint(self.scenario)

        self.model.eval()

        return list(
            self.generator.generate_batch(
                value=self.model,
                use_operation=True,
                operation_override=operation,
                batch_size=batch_size,
            )
        )


class DiskDesigner(BaseDiskDesigner):
    def __init__(
        self, scenario, lock, artifact_dir, environment_repeats=1, master_designer=None
    ):
        super().__init__(lock, artifact_dir, environment_repeats, master_designer)
        self.scenario = scenario


class DesignerRegistry:
    @staticmethod
    def get(
        designer: DesignerConfig,
        scenario: ScenarioConfig,
        ppo_cfg: PPOConfig,
        artifact_dir: str,
        device: torch.device,
    ) -> tuple[Designer, Designer]:
        if isinstance(designer, Fixed):
            fixed = FixedDesigner(scenario)
            return fixed, fixed
        elif isinstance(designer, Random):
            random = RandomDesigner(
                scenario, environment_repeats=designer.environment_repeats
            )
            return random, random
        elif isinstance(designer, _Value):
            lock = torch.multiprocessing.Lock()
            master_designer: CentralisedDesigner = None  # type: ignore
            if isinstance(designer, Diffusion):
                master_designer = DiffusionDesigner(
                    scenario=scenario,
                    classifier=designer.model,
                    diffusion=designer.diffusion,
                    gamma=ppo_cfg.gamma,
                    n_update_iterations=designer.n_update_iterations,
                    train_batch_size=designer.batch_size,
                    buffer_size=designer.buffer_size,
                    lr=designer.lr,
                    weight_decay=designer.weight_decay,
                    environment_repeats=designer.environment_repeats,
                    distill_from_critic=designer.distill_enable,
                    distill_samples=designer.distill_samples,
                    early_start=designer.early_start,
                    device=device,
                )
            else:
                raise ValueError(f"Unknown designer type: {designer.type}. ")
            master = DiskDesigner(
                scenario=scenario,
                lock=lock,
                artifact_dir=artifact_dir,
                environment_repeats=designer.environment_repeats,
                master_designer=master_designer,
            )
            env = DiskDesigner(
                scenario=scenario,
                lock=lock,
                artifact_dir=artifact_dir,
                environment_repeats=designer.environment_repeats,
            )
            return master, env  # type: ignore
        else:
            raise ValueError(f"Unknown designer type: {designer.type}. ")
