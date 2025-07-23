import math
import os
from typing import Callable
from pathlib import Path

import numpy as np
import torch
from torchrl.envs import EnvBase
from torchrl.envs.batched_envs import BatchedEnvBase
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torchrl.objectives.value.functional import reward2go
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from guided_diffusion.script_util import create_model

from diffusion_co_design.common import (
    design,
    OUTPUT_DIR,
    get_latest_model,
    np_list_to_tensor_list,
)
from diffusion_co_design.common.rl.mappo.schema import PPOConfig
from diffusion_co_design.rware.model.classifier import make_model, image_to_pos_colors
from diffusion_co_design.rware.diffusion.transform import (
    graph_projection_constraint,
    image_projection_constraint,
    train_to_eval,
)
from diffusion_co_design.rware.diffusion.generate import generate, get_position
from diffusion_co_design.rware.diffusion.generator import Generator, OptimizerDetails
from diffusion_co_design.rware.static import GROUP_NAME, ENV_NAME
from diffusion_co_design.rware.schema import (
    ScenarioConfig as SC,
    DesignerConfig,
    EnvCriticConfig,
    DiffusionOperation,
    Representation,
)

from diffusion_co_design.rware.schema import Fixed, Random, _Value, Sampling, Diffusion


def make_generate_fn(scenario: SC, representation: Representation):
    def _generate(n: int):
        return generate(
            n=n,
            size=scenario.size,
            n_shelves=scenario.n_shelves,
            goal_idxs=scenario.goal_idxs,
            n_colors=scenario.n_colors,
            representation=representation,
        )

    return _generate


class RandomDesigner(design.RandomDesigner[SC]):
    def __init__(self, designer_setting: design.DesignerParams[SC]):
        super().__init__(designer_setting=designer_setting)
        self.generate = make_generate_fn(self.scenario, representation="image")

    def generate_random_layouts(self, batch_size: int):
        return np_list_to_tensor_list(self.generate(batch_size))


class FixedDesigner(design.FixedDesigner[SC]):
    def __init__(self, designer_setting: design.DesignerParams[SC]):
        super().__init__(
            designer_setting,
            layout=torch.tensor(
                make_generate_fn(designer_setting.scenario, representation="image")(n=1)
            ).squeeze(0),
        )


default_value_learner_hyperparameters = design.ValueLearnerHyperparameters(
    lr=3e-5,
    train_batch_size=64,
    buffer_size=2048,
    n_update_iterations=5,
    clip_grad_norm=None,
    distill_from_critic=False,
    distill_samples=1,
    train_early_start=0,
    loss_criterion="mse",
)


def make_reference_layouts(
    scenario: SC, representation: Representation, device: torch.device
):
    return torch.tensor(
        np.array(
            generate(
                size=scenario.size,
                n_shelves=scenario.n_shelves,
                goal_idxs=scenario.goal_idxs,
                n_colors=scenario.n_colors,
                n=64,
                training_dataset=True,
                representation=representation,
                rng=np.random.default_rng(seed=0),
            )
        ),
        dtype=torch.float32,
        device=device,
    )


class ValueLearner(design.ValueLearner):
    def __init__(
        self,
        scenario: SC,
        classifier: EnvCriticConfig,
        representation: Representation = "image",
        hyperparameters: design.ValueLearnerHyperparameters = default_value_learner_hyperparameters,
        gamma: float = 0.99,
        device: torch.device = torch.device("cpu"),
    ):
        self.scenario = scenario
        self.representation = representation
        super().__init__(
            model=make_model(
                model=classifier.name,
                scenario=scenario,
                model_kwargs=classifier.model_kwargs,
                device=device,
            ),
            group_name=GROUP_NAME,
            episode_steps=scenario.max_steps,
            hyperparameters=hyperparameters,
            gamma=gamma,
            group_aggregation="sum",
            device=device,
        )

    def _get_layout_from_state(self, state):
        return state[:, :, : self.scenario.n_colors]

    def _eval_to_train(self, theta):
        match self.representation:
            case "graph":
                pos, colors = image_to_pos_colors(theta, self.scenario.n_shelves)
                pos = (pos / (self.scenario.size - 1)) * 2 - 1
                return {
                    "pos": pos.to(dtype=torch.float32, device=self.device),
                    "colors": colors.to(dtype=torch.float32, device=self.device),
                }
            case "image":
                return theta * 2 - 1


class DicodeDesigner(design.DicodeDesigner[SC]):
    def __init__(
        self,
        designer_setting: design.DesignerParams[SC],
        classifier: EnvCriticConfig,
        diffusion: DiffusionOperation,
        representation: Representation,
        value_learner_hyperparameters: design.ValueLearnerHyperparameters = default_value_learner_hyperparameters,
        random_generation_early_start: int = 0,
        gamma: float = 0.99,
        total_annealing_iters: int = 2000,
        device: torch.device = torch.device("cpu"),
    ):
        self.representation = representation
        sc = designer_setting.scenario
        super().__init__(
            designer_setting=designer_setting,
            value_learner=ValueLearner(
                scenario=designer_setting.scenario,
                classifier=classifier,
                hyperparameters=value_learner_hyperparameters,
                representation=representation,
                gamma=gamma,
                device=device,
            ),
            diffusion_setting=diffusion,
            diffusion_generator=Generator(
                generator_model_path=get_latest_model(
                    os.path.join(
                        OUTPUT_DIR, ENV_NAME, "diffusion", representation, sc.name
                    ),
                    "model",
                ),
                scenario=sc,
                representation=representation,
                guidance_wt=diffusion.forward_guidance_wt,
                device=device,
            ),
            random_generation_early_start=random_generation_early_start,
            total_annealing_iters=total_annealing_iters,
        )

        self.generate = RandomDesigner(designer_setting)
        match self.representation:
            case "graph":
                self.pc = graph_projection_constraint(sc)
            case "image":
                self.pc = image_projection_constraint(sc)

    def projection_constraint(self, x):
        return self.pc(x)

    def _make_reference_layouts(self):
        return make_reference_layouts(
            self.scenario, self.representation, self.value_learner.device
        )

    def _generate_random_layout_batch(self, batch_size):
        return self.generate.generate_random_layouts(batch_size)


class SamplingDesigner(design.SamplingDesigner[SC]):
    def __init__(
        self,
        designer_setting: design.DesignerParams[SC],
        classifier: EnvCriticConfig,
        representation: Representation,
        value_learner_hyperparameters: design.ValueLearnerHyperparameters = default_value_learner_hyperparameters,
        n_samples: int = 16,
        random_generation_early_start: int = 0,
        gamma: float = 0.99,
        device: torch.device = torch.device("cpu"),
    ):
        self.representation = representation
        super().__init__(
            designer_setting=designer_setting,
            value_learner=ValueLearner(
                scenario=designer_setting.scenario,
                classifier=classifier,
                hyperparameters=value_learner_hyperparameters,
                representation=representation,
                gamma=gamma,
                device=device,
            ),
            n_samples=n_samples,
            random_generation_early_start=random_generation_early_start,
        )
        if representation != "image":
            raise NotImplementedError(
                "SamplingDesigner only supports 'image' representation, not yet implemented."
            )

    def _make_reference_layouts(self):
        return make_reference_layouts(
            self.scenario, self.representation, self.value_learner.device
        )

    def generate_random_layouts(self, batch_size):
        layouts = torch.tensor(
            np.array(
                generate(
                    size=self.scenario.size,
                    n_shelves=self.scenario.n_shelves,
                    goal_idxs=self.scenario.goal_idxs,
                    n_colors=self.scenario.n_colors,
                    n=batch_size,
                    training_dataset=False,
                )
            )
        )
        return layouts

    def _generate_random_layout_batch(self, batch_size):
        return list(self.generate_random_layouts(batch_size))


# class PolicyDesigner(CentralisedDesigner):
#     def __init__(
#         self,
#         scenario: ScenarioConfig,
#         environment_repeats: int = 1,
#         lr: float = 3e-4,
#         train_batch_size: int = 64,
#         train_epochs: int = 5,
#         gamma: float = 0.99,
#         device: torch.device = torch.device("cpu"),
#     ):
#         super().__init__(scenario, environment_repeats, representation="image")

#         self.device = device
#         self.policy = create_model(
#             image_size=self.scenario.size,
#             image_channels=scenario.n_colors,
#             num_channels=64,
#             num_res_blocks=2,
#             resblock_updown=True,
#             use_new_attention_order=True,
#             attention_resolutions="16,8,4",
#             num_head_channels=64,
#         ).to(device)

#         self.optim = torch.optim.Adam(self.policy.parameters(), lr=lr)

#         self.train_batch_size = train_batch_size
#         self.train_epochs = train_epochs
#         self.agent_policy: TensorDictModule | None = None
#         self.train_env: BatchedEnvBase | None = None
#         self.train_env_batch_size: int | None = None
#         self.gamma = gamma

#         self.reinforce_loss = 0.0

#     def make_initial_env(self, batch_size: int):
#         B = batch_size
#         C = self.scenario.n_colors
#         N = self.scenario.size
#         initial_env = torch.zeros((B, C, N, N), device=self.device)
#         for idx, color in zip(self.scenario.goal_idxs, self.scenario.goal_colors):
#             initial_env[:, color, *get_position(idx, N)] = 1
#         return initial_env

#     def generate_environment_batch(self, batch_size: int):
#         B = batch_size
#         C = self.scenario.n_colors
#         N = self.scenario.size
#         initial_env = self.make_initial_env(B)

#         with torch.no_grad():
#             # Calculate logits
#             logits = self.policy(initial_env).reshape(B, C, -1)  # (B, C, N * N)
#             # Apply softmax channel-wise to get placement probabilities
#             envs = torch.zeros((B, C, N * N), device=self.device)
#             actions = torch.zeros(
#                 (B, self.scenario.n_shelves), device=self.device, dtype=torch.long
#             )
#             batch_idxs = torch.arange(B)
#             for i in range(self.scenario.n_shelves):
#                 channel_selection = i % self.scenario.n_colors  # [B]
#                 mask = envs.sum(dim=1) > 0  # [B, N * N]
#                 logits_i = logits[batch_idxs, channel_selection]  # [B, N * N]
#                 logits_i = logits_i.masked_fill(mask, float("-inf"))  # [B, N * N]
#                 probs_i = torch.softmax(logits_i, dim=-1)  # [B, N * N]
#                 idxs = torch.multinomial(probs_i, 1).squeeze(1)
#                 envs[batch_idxs, channel_selection, idxs] = 1
#                 actions[batch_idxs, i] = idxs

#             envs = envs.reshape(B, C, N, N)

#         return envs, actions

#     def reinforce(
#         self, envs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor
#     ):
#         B = envs.shape[0]
#         C = self.scenario.n_colors
#         N = self.scenario.size
#         assert envs.shape == (B, C, N, N), envs.shape
#         assert actions.shape == (B, self.scenario.n_shelves), actions.shape
#         assert rewards.shape == (B,), rewards.shape

#         initial_env = self.make_initial_env(B)
#         logits = self.policy(initial_env).reshape(B, C, -1)
#         batch_idxs = torch.arange(B)
#         constructed_envs = torch.zeros((B, C, N * N), device=self.device)

#         loss = torch.tensor(0.0, device=self.device)

#         for i in range(self.scenario.n_shelves):
#             channel_selection = i % self.scenario.n_colors  # [B]
#             mask = constructed_envs.sum(dim=1) > 0  # [B, N * N]
#             logits_i = logits[batch_idxs, channel_selection]  # [B, N * N]
#             logits_i = logits_i.masked_fill(mask, float("-inf"))  # [B, N * N]
#             idxs = actions[batch_idxs, i]  # [B]

#             log_probs_i = torch.nn.functional.log_softmax(logits_i, dim=-1)
#             action_log_probs = log_probs_i[batch_idxs, idxs]  # [B]

#             loss += -(action_log_probs * rewards).mean()  # [B]
#             constructed_envs[batch_idxs, channel_selection, idxs] = 1

#         self.optim.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
#         self.optim.step()
#         return loss.item()

#     def reset_env_buffer(self, batch_size):
#         self.policy.eval()
#         envs, _ = self.generate_environment_batch(batch_size=batch_size)
#         batch = list(envs.numpy(force=True))
#         return batch

#     def get_model(self):
#         return self.policy

#     def update(self, sampling_td):
#         super().update(sampling_td)

#         assert self.train_env is not None
#         assert self.agent_policy is not None
#         self.reinforce_loss = 0.0
#         for _ in range(self.train_epochs):
#             # Generate envs
#             self.policy.eval()
#             envs, actions = self.generate_environment_batch(
#                 batch_size=self.train_batch_size
#             )

#             # Reset Collect rewards
#             chunk_number = math.ceil(self.train_batch_size / self.train_env_batch_size)
#             env_chunks = torch.chunk(envs, chunk_number, dim=0)

#             rewards_list = []
#             for env_chunk in env_chunks:
#                 envs_list = list(env_chunk)
#                 n = len(envs_list)
#                 if n < self.train_env_batch_size:
#                     envs_list += [envs_list[-1]] * (
#                         self.train_env_batch_size - n
#                     )  # Pad if needed

#                 td = self.train_env.reset(
#                     list_of_kwargs=[{"layout_override": env.cpu()} for env in envs_list]
#                 )
#                 td = self.train_env.rollout(
#                     max_steps=self.scenario.max_steps,
#                     policy=self.agent_policy,
#                     auto_reset=False,
#                     tensordict=td,
#                 )

#                 done = td.get(("next", "done"))
#                 reward = td.get(("next", "agents", "reward")).sum(dim=-2)
#                 y = reward2go(reward, done=done, gamma=self.gamma, time_dim=-2)
#                 y = y.reshape(-1, self.scenario.max_steps)
#                 y = y[:n, 0]

#                 rewards_list.append(y)

#             rewards = torch.cat(rewards_list, dim=0)
#             assert rewards.shape == (self.train_batch_size,), rewards.shape

#             # Reinforce
#             self.policy.train()
#             self.reinforce_loss += self.reinforce(envs, actions, rewards)

#         self.reinforce_loss /= self.train_epochs
#         self.train_env.reset()

#     def get_logs(self):
#         logs = {"designer_loss": self.reinforce_loss}
#         return logs


# class GradientDescentDesigner(ValueDesigner):
#     def __init__(
#         self,
#         scenario: ScenarioConfig,
#         classifier: EnvCriticConfig,
#         epochs: int = 20,
#         gradient_iterations: int = 10,
#         gradient_lr: float = 0.03,
#         gamma: float = 0.99,
#         n_update_iterations: int = 5,
#         train_batch_size: int = 64,
#         buffer_size: int = 2048,
#         lr: float = 0.00003,
#         weight_decay: float = 0,
#         environment_repeats: int = 1,
#         distill_from_critic: bool = False,
#         distill_samples: int = 1,
#         early_start: int | None = None,
#         device: torch.device = torch.device("cpu"),
#     ):
#         super().__init__(
#             scenario=scenario,
#             classifier=classifier,
#             gamma=gamma,
#             n_update_iterations=n_update_iterations,
#             train_batch_size=train_batch_size,
#             buffer_size=buffer_size,
#             lr=lr,
#             weight_decay=weight_decay,
#             environment_repeats=environment_repeats,
#             distill_from_critic=distill_from_critic,
#             distill_samples=distill_samples,
#             early_start=early_start,
#             device=device,
#         )

#         self.epochs = epochs
#         self.gradient_iterations = gradient_iterations
#         self.gradient_lr = gradient_lr

#     def _reset_env_buffer(self, batch_size: int):
#         match self.representation:
#             case "graph":
#                 projection_constraint = graph_projection_constraint(self.scenario)
#             case "image":
#                 projection_constraint = image_projection_constraint(self.scenario)

#         # Generate random initial environments
#         env = torch.tensor(
#             np.array(
#                 generate(
#                     size=self.scenario.size,
#                     n_shelves=self.scenario.n_shelves,
#                     goal_idxs=self.scenario.goal_idxs,
#                     n_colors=self.scenario.n_colors,
#                     n=batch_size,
#                     representation=self.representation,
#                     training_dataset=True,
#                 )
#             ),
#             dtype=torch.float32,
#             device=self.device,
#         )

#         env_optim = torch.optim.Adam([env], lr=self.gradient_lr)

#         for epoch in range(self.epochs):
#             env.requires_grad = True
#             for iteration in range(self.gradient_iterations):
#                 env_optim.zero_grad()

#                 y_pred = self.model(env)
#                 loss = -y_pred.sum()
#                 loss.backward()
#                 env_optim.step()

#             env = projection_constraint(env.detach())

#         env = train_to_eval(
#             env=env.detach(), cfg=self.scenario, representation=self.representation
#         )

#         batch = list(env.numpy(force=True))

#         return batch


# class DiffusionDesigner(ValueDesigner):
#     def __init__(
#         self,
#         scenario: ScenarioConfig,
#         classifier: EnvCriticConfig,
#         diffusion: DiffusionOperation,
#         gamma: float = 0.99,
#         n_update_iterations: int = 5,
#         train_batch_size: int = 64,
#         buffer_size: int = 2048,
#         lr: float = 3e-5,
#         weight_decay: float = 0.0,
#         environment_repeats: int = 1,
#         distill_from_critic: bool = False,
#         distill_samples: int = 1,
#         early_start: int | None = None,
#         device: torch.device = torch.device("cpu"),
#     ):
#         super().__init__(
#             scenario,
#             classifier,
#             gamma=gamma,
#             n_update_iterations=n_update_iterations,
#             train_batch_size=train_batch_size,
#             buffer_size=buffer_size,
#             lr=lr,
#             weight_decay=weight_decay,
#             environment_repeats=environment_repeats,
#             distill_from_critic=distill_from_critic,
#             distill_samples=distill_samples,
#             early_start=early_start,
#             device=device,
#         )

#         pretrain_dir = os.path.join(
#             OUTPUT_DIR, "rware", "diffusion", self.representation, scenario.name
#         )
#         latest_checkpoint = get_latest_model(pretrain_dir, "model")

#         self.generator = Generator(
#             generator_model_path=latest_checkpoint,
#             scenario=scenario,
#             representation=self.representation,  # type: ignore
#             guidance_wt=diffusion.forward_guidance_wt,
#             device=device,
#         )

#         self.diffusion = diffusion

#     def _reset_env_buffer(self, batch_size: int):
#         forward_enable = self.diffusion.forward_guidance_wt > 0
#         operation = OptimizerDetails()
#         operation.num_recurrences = self.diffusion.num_recurrences
#         operation.lr = self.diffusion.backward_lr
#         operation.backward_steps = self.diffusion.backward_steps
#         operation.use_forward = forward_enable

#         match self.representation:
#             case "graph":
#                 operation.projection_constraint = graph_projection_constraint(
#                     self.scenario
#                 )
#             case "image":
#                 operation.projection_constraint = image_projection_constraint(
#                     self.scenario
#                 )

#         self.model.eval()

#         return list(
#             self.generator.generate_batch(
#                 value=self.model,
#                 use_operation=True,
#                 operation_override=operation,
#                 batch_size=batch_size,
#             )
#         )


def create_designer(
    scenario: SC,
    designer: DesignerConfig,
    ppo_cfg: PPOConfig,
    artifact_dir: str | Path,
    device: torch.device,
) -> tuple[design.Designer[SC], Callable[[], design.DesignConsumer]]:
    lock = torch.multiprocessing.Lock()

    if isinstance(artifact_dir, str):
        artifact_dir_path = Path(artifact_dir)
    else:
        artifact_dir_path = artifact_dir
    designer_setting = design.DesignerParams(
        scenario=scenario,
        artifact_dir=artifact_dir_path,
        lock=lock,
        environment_repeats=designer.environment_repeats,
    )

    def design_consumer_fn():
        return design.DesignConsumer(artifact_dir_path, lock)

    if isinstance(designer, Fixed):
        designer_producer: design.Designer[SC] = FixedDesigner(designer_setting)
    elif isinstance(designer, Random):
        designer_producer = RandomDesigner(designer_setting)
    elif isinstance(designer, _Value):
        value_hyperparameters = design.ValueLearnerHyperparameters(
            lr=designer.lr,
            train_batch_size=designer.train_batch_size,
            buffer_size=designer.buffer_size,
            n_update_iterations=designer.n_update_iterations,
            clip_grad_norm=None,
            weight_decay=0.0,
            distill_from_critic=designer.distill_enable,
            distill_samples=designer.distill_samples,
            loss_criterion=designer.loss_criterion,
            train_early_start=designer.train_early_start,
        )
        if isinstance(designer, Sampling):
            designer_producer = SamplingDesigner(
                designer_setting=designer_setting,
                classifier=designer.model,
                value_learner_hyperparameters=value_hyperparameters,
                random_generation_early_start=designer.random_generation_early_start,
                gamma=ppo_cfg.gamma,
                n_samples=designer.n_samples,
                representation=designer.representation,
                device=device,
            )
        elif isinstance(designer, Diffusion):
            designer_producer = DicodeDesigner(
                diffusion=designer.diffusion,
                classifier=designer.model,
                designer_setting=designer_setting,
                value_learner_hyperparameters=value_hyperparameters,
                random_generation_early_start=designer.random_generation_early_start,
                gamma=ppo_cfg.gamma,
                total_annealing_iters=ppo_cfg.n_iters,
                representation=designer.representation,
                device=device,
            )

    return designer_producer, design_consumer_fn
