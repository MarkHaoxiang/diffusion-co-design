from abc import abstractmethod
import os
import pickle as pkl
from typing import Literal

import numpy as np
import torch
from torchrl.envs import EnvBase
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torchrl.objectives.value.functional import reward2go
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from diffusion_co_design.common import OUTPUT_DIR, get_latest_model
from diffusion_co_design.rware.model.classifier import make_model, image_to_pos_colors
from diffusion_co_design.rware.diffusion.transform import (
    graph_projection_constraint,
    image_projection_constraint,
    storage_to_layout,
)
from diffusion_co_design.rware.diffusion.generate import generate
from diffusion_co_design.rware.diffusion.generator import Generator, OptimizerDetails
from diffusion_co_design.common.design import BaseDesigner
from diffusion_co_design.rware.schema import (
    ScenarioConfig,
    DesignerConfig,
    ClassifierConfig,
    DiffusionOperation,
    PPOConfig,
    Representation,
)


class Designer(BaseDesigner):
    def __init__(
        self,
        scenario: ScenarioConfig,
        environment_repeats: int = 1,
        representation: Representation = "image",
    ):
        super().__init__(environment_repeats=environment_repeats)
        self.scenario = scenario
        self.representation = representation

    def generate_environment(self, objective):
        return storage_to_layout(
            features=self.generate_environment_weights(objective),
            config=self.scenario,
            representation=self.representation,
        )


class RandomDesigner(Designer):
    def forward(self, objective):
        env = torch.tensor(
            generate(
                size=self.scenario.size,
                n_shelves=self.scenario.n_shelves,
                goal_idxs=self.scenario.goal_idxs,
                n_colors=self.scenario.n_colors,
            )[0]
        )
        return env

    def _generate_environment_weights(self, objective):
        return self.forward(objective)


class FixedDesigner(Designer):
    def __init__(self, scenario):
        super().__init__(scenario)
        self.layout_image = torch.nn.Parameter(
            RandomDesigner(scenario)._generate_environment_weights(None),
            requires_grad=False,
        )

    def forward(self, objective):
        return self.layout_image.data

    def _generate_environment_weights(self, objective):
        return self.layout_image


class CentralisedDesigner(Designer):
    @abstractmethod
    def reset_env_buffer(self, batch_size: int) -> list:
        raise NotImplementedError

    def forward(self, objective):
        raise NotImplementedError("Use with disk designer")

    def _generate_environment_weights(self, objective):
        raise NotImplementedError("Use with disk designer")


def get_env_from_td(td, scenario: ScenarioConfig, gamma: float = 0.99):
    done = td.get(("next", "done"))
    X = td.get("state")[done.squeeze(-1)].to(dtype=torch.float32)
    X = X[:, : scenario.n_colors]
    reward = td.get(("next", "agents", "reward")).sum(dim=-2)
    y = reward2go(reward, done=done, gamma=gamma, time_dim=-2)
    y = y.reshape(-1, scenario.max_steps)
    y = y[:, 0]
    return X, y


class ValueDesigner(CentralisedDesigner):
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
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(scenario, environment_repeats=environment_repeats)
        self.model = make_model(
            model=classifier.name,
            scenario=scenario,
            model_kwargs=classifier.model_kwargs,
            device=device,
        )
        self.representation = classifier.representation  # type: ignore

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
        self.running_loss = 0
        self.gamma = gamma

        self.distill_from_critic = distill_from_critic
        self.distill_samples = distill_samples
        self.critic: TensorDictModule | None = None
        self.ref_env: EnvBase | None = None
        if self.distill_from_critic:
            self.manual_designer = FixedDesigner(scenario=scenario)

    def update(self, sampling_td):
        super().update(sampling_td)

        # Update replay buffer
        X, y = get_env_from_td(sampling_td, self.scenario, gamma=self.gamma)
        match self.representation:
            case "graph":
                pos, colors = image_to_pos_colors(X, self.scenario.n_shelves)
                pos = (pos / (self.scenario.size - 1)) * 2 - 1
                X_post = {
                    "pos": pos.to(dtype=torch.float32, device=self.device),
                    "colors": colors.to(dtype=torch.float32, device=self.device),
                }
            case "image":
                X_post = X * 2 - 1

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
                            ("agents", "state_value")
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

                # Data Preprocessing
                if self.representation == "graph":
                    X_batch = (X_batch["pos"], X_batch["colors"])

                # Timesteps
                # t, _ = self.generator.schedule_sampler.sample(len(X_batch), self.device)
                # X_batch = self.generator.diffusion.q_sample(X_batch, t)
                # y_pred = self.model(X_batch, t).squeeze()

                y_pred = self.model.predict(X_batch)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                self.running_loss += loss.item()
                self.optim.step()
            # Logs
            self.running_loss = self.running_loss / self.n_update_iterations

        if self.representation == "graph":
            X_post = (X_post["pos"], X_post["colors"])
        classifier_prediction = self.model.predict(X_post)
        self.classifier_prediction_mean = classifier_prediction.mean().item()
        self.classifier_prediction_max = classifier_prediction.max().item()
        self.classifier_prediction_min = classifier_prediction.min().item()

    def get_logs(self):
        return {
            "designer_loss": self.running_loss,
            "classifier_prediction_mean": self.classifier_prediction_mean,
            "classifier_prediction_max": self.classifier_prediction_max,
            "classifier_prediction_min": self.classifier_prediction_min,
        }

    def get_model(self):
        return self.model

    def get_training_buffer(self):
        return self.env_buffer


class SamplingDesigner(ValueDesigner):
    def __init__(
        self,
        scenario: ScenarioConfig,
        classifier: ClassifierConfig,
        gamma: float = 0.99,
        n_sample: int = 5,
        n_update_iterations: int = 5,
        train_batch_size: int = 64,
        buffer_size: int = 2048,
        lr: float = 3e-5,
        weight_decay: float = 0.00,
        environment_repeats: int = 1,
        distill_from_critic: bool = False,
        distill_samples: int = 1,
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
            device=device,
        )
        self.n_sample = n_sample

    def reset_env_buffer(self, batch_size: int):
        B = batch_size
        self.model.eval()
        X_numpy: np.ndarray = np.array(
            generate(
                size=self.scenario.size,
                n_shelves=self.scenario.n_shelves,
                goal_idxs=self.scenario.goal_idxs,
                n_colors=self.scenario.n_colors,
                n=B * self.n_sample,
            )
        )
        X_torch = torch.tensor(X_numpy, dtype=torch.float32, device=self.device)
        X_torch = X_torch * 2 - 1
        y: torch.Tensor = self.model(X_torch).squeeze()

        y = y.reshape(B, self.n_sample)
        X_numpy = X_numpy.reshape((B, self.n_sample, *X_numpy.shape[1:]))
        indices = y.argmax(dim=1).numpy(force=True)
        batch = list(X_numpy[np.arange(B), indices])

        return batch


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
            device=device,
        )

        pretrain_dir = os.path.join(
            OUTPUT_DIR, "rware", "diffusion", self.representation, scenario.name
        )
        latest_checkpoint = get_latest_model(pretrain_dir, "model")

        self.generator = Generator(
            generator_model_path=latest_checkpoint,
            scenario=scenario,
            representation=self.representation,  # type: ignore
            guidance_wt=diffusion.forward_guidance_wt,
        )

        self.diffusion = diffusion
        self.early_start = diffusion.early_start

    def reset_env_buffer(self, batch_size: int):
        B = batch_size
        batch = []
        if len(self.env_buffer) < self.buffer_size and self.early_start:
            for _ in range(B):
                batch.append(
                    np.array(
                        generate(
                            size=self.scenario.size,
                            n_shelves=self.scenario.n_shelves,
                            goal_idxs=self.scenario.goal_idxs,
                            n_colors=self.scenario.n_colors,
                            representation=self.representation,
                        )[0]
                    )
                )
        else:
            forward_enable = self.diffusion.forward_guidance_wt > 0
            operation = OptimizerDetails()
            operation.num_recurrences = self.diffusion.num_recurrences
            operation.lr = self.diffusion.backward_lr
            operation.backward_steps = self.diffusion.backward_steps
            operation.use_forward = forward_enable

            match self.representation:
                case "graph":
                    operation.projection_constraint = graph_projection_constraint(
                        self.scenario
                    )
                case "image":
                    operation.projection_constraint = image_projection_constraint(
                        self.scenario
                    )

            self.model.eval()

            for env in self.generator.generate_batch(
                value=self.model,
                use_operation=True,
                operation_override=operation,
                batch_size=B,
            ):
                batch.append(env)
        return batch


class DiskDesigner(Designer):
    """Hack for parallel execution"""

    # So this is hacky
    # For some reason, normal mp queues break across multiple environments
    # So we just read the shared environment generation buffer from disk instead

    def __init__(
        self,
        scenario,
        lock,
        artifact_dir,
        representation: Representation,
        environment_repeats: int = 1,
        master_designer: CentralisedDesigner | None = None,
    ):
        super().__init__(
            scenario,
            environment_repeats=environment_repeats,
            representation=representation,
        )
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
                self.environment_repeat_counter = (
                    self.environment_repeat_counter + 1
                ) % self.environment_repeats
                if self.environment_repeat_counter == 0:
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


class DesignerRegistry:
    FIXED = "fixed"
    RANDOM = "random"
    RL = "rl"
    DIFFUSION = "diffusion"
    DIFFUSION_SHARED = "diffusion_shared"
    SAMPLING = "sampling"

    @staticmethod
    def get(
        designer: DesignerConfig,
        scenario: ScenarioConfig,
        ppo_cfg: PPOConfig,
        artifact_dir: str,
        device: torch.device,
    ) -> tuple[Designer, Designer]:
        match designer.type:
            case DesignerRegistry.FIXED:
                fixed = FixedDesigner(scenario)
                return fixed, fixed
            case DesignerRegistry.RANDOM:
                rand = RandomDesigner(
                    scenario, environment_repeats=designer.environment_repeats
                )
                return rand, rand
            case DesignerRegistry.RL:
                raise NotImplementedError()
            case DesignerRegistry.DIFFUSION:
                lock = torch.multiprocessing.Lock()
                assert designer.diffusion is not None
                assert designer.value_model is not None
                master_designer: CentralisedDesigner = DiffusionDesigner(
                    scenario,
                    classifier=designer.value_model,
                    diffusion=designer.diffusion,
                    gamma=ppo_cfg.gamma,
                    n_update_iterations=designer.value_n_update_iterations,
                    train_batch_size=designer.value_train_batch_size,
                    buffer_size=designer.value_buffer_size,
                    lr=designer.value_lr,
                    weight_decay=designer.value_weight_decay,
                    distill_from_critic=designer.value_distill_enable,
                    distill_samples=designer.value_distill_samples,
                    device=device,
                )
                master = DiskDesigner(
                    scenario,
                    lock,
                    artifact_dir,
                    environment_repeats=designer.environment_repeats,
                    representation=master_designer.representation,
                    master_designer=master_designer,
                )
                env = DiskDesigner(
                    scenario,
                    lock,
                    artifact_dir,
                    representation=master_designer.representation,
                    environment_repeats=designer.environment_repeats,
                )
                return master, env
            case DesignerRegistry.DIFFUSION_SHARED:
                raise NotImplementedError()
            case DesignerRegistry.SAMPLING:
                assert designer.value_model is not None
                lock = torch.multiprocessing.Lock()
                master_designer = SamplingDesigner(
                    scenario,
                    classifier=designer.value_model,
                    gamma=ppo_cfg.gamma,
                    n_update_iterations=designer.value_n_update_iterations,
                    train_batch_size=designer.value_train_batch_size,
                    buffer_size=designer.value_buffer_size,
                    lr=designer.value_lr,
                    weight_decay=designer.value_weight_decay,
                    distill_from_critic=designer.value_distill_enable,
                    distill_samples=designer.value_distill_samples,
                    device=device,
                )
                master = DiskDesigner(
                    scenario,
                    lock,
                    artifact_dir,
                    environment_repeats=designer.environment_repeats,
                    representation=master_designer.representation,
                    master_designer=master_designer,
                )
                env = DiskDesigner(
                    scenario,
                    lock,
                    artifact_dir,
                    representation=master_designer.representation,
                    environment_repeats=designer.environment_repeats,
                )
                return master, env
            case _:
                raise ValueError(f"Unknown designer type {designer}")
