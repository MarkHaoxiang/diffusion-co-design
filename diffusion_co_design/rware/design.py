from abc import abstractmethod
import math
import os
import pickle as pkl
from typing import Literal

import numpy as np
import torch
from torchrl.envs import EnvBase
from torchrl.envs.batched_envs import BatchedEnvBase
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torchrl.objectives.value.functional import reward2go
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from guided_diffusion.script_util import create_model

from diffusion_co_design.common import OUTPUT_DIR, get_latest_model
from diffusion_co_design.rware.model.classifier import make_model, image_to_pos_colors
from diffusion_co_design.rware.diffusion.transform import (
    graph_projection_constraint,
    image_projection_constraint,
    storage_to_layout,
    train_to_eval,
)
from diffusion_co_design.rware.diffusion.generate import generate, get_position
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


class PolicyDesigner(CentralisedDesigner):
    def __init__(
        self,
        scenario: ScenarioConfig,
        environment_repeats: int = 1,
        lr: float = 3e-4,
        train_batch_size: int = 64,
        train_epochs: int = 5,
        gamma: float = 0.99,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(scenario, environment_repeats, representation="image")

        self.device = device
        self.policy = create_model(
            image_size=self.scenario.size,
            image_channels=scenario.n_colors,
            num_channels=64,
            num_res_blocks=2,
            resblock_updown=True,
            use_new_attention_order=True,
            attention_resolutions="16,8,4",
            num_head_channels=64,
        ).to(device)

        self.optim = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.train_batch_size = train_batch_size
        self.train_epochs = train_epochs
        self.agent_policy: TensorDictModule | None = None
        self.train_env: BatchedEnvBase | None = None
        self.train_env_batch_size: int | None = None
        self.gamma = gamma

        self.reinforce_loss = 0.0

    def make_initial_env(self, batch_size: int):
        B = batch_size
        C = self.scenario.n_colors
        N = self.scenario.size
        initial_env = torch.zeros((B, C, N, N), device=self.device)
        for idx, color in zip(self.scenario.goal_idxs, self.scenario.goal_colors):
            initial_env[:, color, *get_position(idx, N)] = 1
        return initial_env

    def generate_environment_batch(self, batch_size: int):
        B = batch_size
        C = self.scenario.n_colors
        N = self.scenario.size
        initial_env = self.make_initial_env(B)

        with torch.no_grad():
            # Calculate logits
            logits = self.policy(initial_env).reshape(B, C, -1)  # (B, C, N * N)
            # Apply softmax channel-wise to get placement probabilities
            envs = torch.zeros((B, C, N * N), device=self.device)
            actions = torch.zeros(
                (B, self.scenario.n_shelves), device=self.device, dtype=torch.long
            )
            batch_idxs = torch.arange(B)
            for i in range(self.scenario.n_shelves):
                channel_selection = i % self.scenario.n_colors  # [B]
                mask = envs.sum(dim=1) > 0  # [B, N * N]
                logits_i = logits[batch_idxs, channel_selection]  # [B, N * N]
                logits_i = logits_i.masked_fill(mask, float("-inf"))  # [B, N * N]
                probs_i = torch.softmax(logits_i, dim=-1)  # [B, N * N]
                idxs = torch.multinomial(probs_i, 1).squeeze(1)
                envs[batch_idxs, channel_selection, idxs] = 1
                actions[batch_idxs, i] = idxs

            envs = envs.reshape(B, C, N, N)

        return envs, actions

    def reinforce(
        self, envs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor
    ):
        B = envs.shape[0]
        C = self.scenario.n_colors
        N = self.scenario.size
        assert envs.shape == (B, C, N, N), envs.shape
        assert actions.shape == (B, self.scenario.n_shelves), actions.shape
        assert rewards.shape == (B,), rewards.shape

        initial_env = self.make_initial_env(B)
        logits = self.policy(initial_env).reshape(B, C, -1)
        batch_idxs = torch.arange(B)
        constructed_envs = torch.zeros((B, C, N * N), device=self.device)

        loss = torch.tensor(0.0, device=self.device)

        for i in range(self.scenario.n_shelves):
            channel_selection = i % self.scenario.n_colors  # [B]
            mask = constructed_envs.sum(dim=1) > 0  # [B, N * N]
            logits_i = logits[batch_idxs, channel_selection]  # [B, N * N]
            logits_i = logits_i.masked_fill(mask, float("-inf"))  # [B, N * N]
            idxs = actions[batch_idxs, i]  # [B]

            log_probs_i = torch.nn.functional.log_softmax(logits_i, dim=-1)
            action_log_probs = log_probs_i[batch_idxs, idxs]  # [B]

            loss += -(action_log_probs * rewards).mean()  # [B]
            constructed_envs[batch_idxs, channel_selection, idxs] = 1

        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optim.step()
        return loss.item()

    def reset_env_buffer(self, batch_size):
        self.policy.eval()
        envs, _ = self.generate_environment_batch(batch_size=batch_size)
        batch = list(envs.numpy(force=True))
        return batch

    def get_model(self):
        return self.policy

    def update(self, sampling_td):
        super().update(sampling_td)

        assert self.train_env is not None
        assert self.agent_policy is not None
        self.reinforce_loss = 0.0
        for _ in range(self.train_epochs):
            # Generate envs
            self.policy.eval()
            envs, actions = self.generate_environment_batch(
                batch_size=self.train_batch_size
            )

            # Reset Collect rewards
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
                    list_of_kwargs=[{"layout_override": env} for env in envs_list]
                )
                td = self.train_env.rollout(
                    max_steps=self.scenario.max_steps,
                    policy=self.agent_policy,
                    auto_reset=False,
                    tensordict=td,
                )

                done = td.get(("next", "done"))
                reward = td.get(("next", "agents", "reward")).sum(dim=-2)
                y = reward2go(reward, done=done, gamma=self.gamma, time_dim=-2)
                y = y.reshape(-1, self.scenario.max_steps)
                y = y[:n, 0]

                rewards_list.append(y)

            rewards = torch.cat(rewards_list, dim=0)
            assert rewards.shape == (self.train_batch_size,), rewards.shape

            # Reinforce
            self.policy.train()
            self.reinforce_loss += self.reinforce(envs, actions, rewards)

        self.reinforce_loss /= self.train_epochs
        self.train_env.reset()

    def get_logs(self):
        logs = {"designer_loss": self.reinforce_loss}
        return logs


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
        self.gamma = gamma

        self.distill_from_critic = distill_from_critic
        self.distill_samples = distill_samples
        self.critic: TensorDictModule | None = None
        self.ref_env: EnvBase | None = None
        if self.distill_from_critic:
            self.manual_designer = FixedDesigner(scenario=scenario)

        # Logging
        self.ref_random_designs = torch.tensor(
            np.array(
                generate(
                    size=scenario.size,
                    n_shelves=scenario.n_shelves,
                    goal_idxs=scenario.goal_idxs,
                    n_colors=scenario.n_colors,
                    n=64,
                    training_dataset=True,
                    representation=self.representation,
                    rng=np.random.default_rng(seed=0),
                )
            ),
            dtype=torch.float32,
            device=self.device,
        )

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
                train_y_batch.append(y_batch)

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
            train_y_batch = torch.cat(train_y_batch)
            self.train_y_mean = train_y_batch.mean().item()
            self.train_y_max = train_y_batch.max().item()
            self.train_y_min = train_y_batch.min().item()

        if self.representation == "graph":
            X_post = (X_post["pos"], X_post["colors"])
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
        assert self.representation == "image"

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
                training_dataset=True,
            )
        )
        X_torch = torch.tensor(X_numpy, dtype=torch.float32, device=self.device)
        y: torch.Tensor = self.model(X_torch).squeeze()

        y = y.reshape(B, self.n_sample)
        X_numpy = X_numpy.reshape((B, self.n_sample, *X_numpy.shape[1:]))
        indices = y.argmax(dim=1).numpy(force=True)
        batch = list(X_numpy[np.arange(B), indices])

        return batch


class GradientDescentDesigner(ValueDesigner):
    def __init__(
        self,
        scenario: ScenarioConfig,
        classifier: ClassifierConfig,
        epochs: int = 10,
        gradient_iterations: int = 10,
        gradient_lr: float = 0.01,
        gamma: float = 0.99,
        n_update_iterations: int = 5,
        train_batch_size: int = 64,
        buffer_size: int = 2048,
        lr: float = 0.00003,
        weight_decay: float = 0,
        environment_repeats: int = 1,
        distill_from_critic: bool = False,
        distill_samples: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            scenario=scenario,
            classifier=classifier,
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

        self.epochs = epochs
        self.gradient_iterations = gradient_iterations
        self.gradient_lr = gradient_lr

    def reset_env_buffer(self, batch_size: int):
        B = batch_size

        match self.representation:
            case "graph":
                projection_constraint = graph_projection_constraint(self.scenario)
            case "image":
                projection_constraint = image_projection_constraint(self.scenario)

        # Generate random initial environments
        env = torch.tensor(
            generate(
                size=self.scenario.size,
                n_shelves=self.scenario.n_shelves,
                goal_idxs=self.scenario.goal_idxs,
                n_colors=self.scenario.n_colors,
                n=B,
                representation=self.representation,
                training_dataset=True,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        env_optim = torch.optim.Adam([env], lr=self.gradient_lr)

        for epoch in range(self.epochs):
            env.requires_grad = True
            for iteration in range(self.gradient_iterations):
                env_optim.zero_grad()

                y_pred = self.model(env)
                loss = -y_pred.sum()
                loss.backward()
                env_optim.step()
            env = projection_constraint(env.detach())

        env = train_to_eval(
            env=env.detach(), cfg=self.scenario, representation=self.representation
        )

        batch = list(env.numpy(force=True))

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
            device=device,
        )

        self.diffusion = diffusion
        self.early_start = diffusion.early_start

    def reset_env_buffer(self, batch_size: int):
        B = batch_size
        batch = []
        if len(self.env_buffer) < self.buffer_size and self.early_start:
            batch = generate(
                size=self.scenario.size,
                n_shelves=self.scenario.n_shelves,
                goal_idxs=self.scenario.goal_idxs,
                n_colors=self.scenario.n_colors,
                n=B,
                representation=self.representation,
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

            batch = list(
                self.generator.generate_batch(
                    value=self.model,
                    use_operation=True,
                    operation_override=operation,
                    batch_size=B,
                )
            )
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
            case (
                DesignerRegistry.RL
                | DesignerRegistry.DIFFUSION
                | DesignerRegistry.SAMPLING
            ):
                lock = torch.multiprocessing.Lock()
                master_designer: CentralisedDesigner = None  # type: ignore
                match designer.type:
                    case DesignerRegistry.RL:
                        master_designer = PolicyDesigner(
                            scenario=scenario,
                            lr=designer.value_lr,
                            train_batch_size=designer.value_train_batch_size,
                            train_epochs=designer.value_n_update_iterations,
                            gamma=ppo_cfg.gamma,
                            device=device,
                        )
                    case DesignerRegistry.DIFFUSION:
                        assert designer.diffusion is not None
                        assert designer.value_model is not None
                        master_designer = DiffusionDesigner(
                            scenario=scenario,
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
                    case DesignerRegistry.SAMPLING:
                        assert designer.value_model is not None
                        master_designer = SamplingDesigner(
                            scenario=scenario,
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
