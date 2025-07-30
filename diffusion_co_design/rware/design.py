import os
from typing import Callable
from pathlib import Path

import numpy as np
import torch
from guided_diffusion.script_util import create_model

from diffusion_co_design.common import (
    design,
    OUTPUT_DIR,
    get_latest_model,
    np_list_to_tensor_list,
)
from diffusion_co_design.common.rl.mappo.schema import PPOConfig
from diffusion_co_design.rware.model.classifier import (
    make_model,
    image_to_pos_colors,
    make_hint_loss,
)
from diffusion_co_design.rware.diffusion.transform import (
    graph_projection_constraint,
    image_projection_constraint,
    train_to_eval,
    hashable_representation,
)
from diffusion_co_design.rware.diffusion.generate import generate, get_position
from diffusion_co_design.rware.diffusion.generator import Generator
from diffusion_co_design.rware.static import GROUP_NAME, ENV_NAME
from diffusion_co_design.rware.schema import (
    ScenarioConfig as SC,
    DesignerConfig,
    EnvCriticConfig,
    DiffusionOperation,
    Representation,
)

from diffusion_co_design.rware.schema import (
    Fixed,
    Random,
    _Value,
    Sampling,
    Diffusion,
    Descent,
    Reinforce,
    Replay,
)


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
        self.classifier_name = classifier.name
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

    def _make_hint_loss(self, device):
        return make_hint_loss(
            model=self.classifier_name,
            agent_critic=self.critic,
            env_critic=self.model,
            device=device,
        )


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

        self.generate = make_generate_fn(
            self.scenario, representation=self.representation
        )
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
        return np_list_to_tensor_list(self.generate(batch_size))


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


class DescentDesigner(design.GradientDescentDesigner[SC]):
    def __init__(
        self,
        designer_setting: design.DesignerParams[SC],
        classifier: EnvCriticConfig,
        representation: Representation,
        value_learner_hyperparameters: design.ValueLearnerHyperparameters = default_value_learner_hyperparameters,
        random_generation_early_start: int = 0,
        lr: float = 0.03,
        n_epochs: int = 10,
        n_gradient_iterations: int = 10,
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
            lr=lr,
            n_epochs=n_epochs,
            n_gradient_iterations=n_gradient_iterations,
            random_generation_early_start=random_generation_early_start,
        )
        self.generate = make_generate_fn(
            self.scenario, representation=self.representation
        )

    def _generate_initial_env(self, batch_size):
        return torch.stack(
            np_list_to_tensor_list(
                generate(
                    n=batch_size,
                    size=self.scenario.size,
                    n_shelves=self.scenario.n_shelves,
                    goal_idxs=self.scenario.goal_idxs,
                    n_colors=self.scenario.n_colors,
                    representation=self.representation,
                    training_dataset=True,
                ),
                device=self.value_learner.device,
            )
        )

    def _train_to_eval(self, env):
        return train_to_eval(
            env=env, cfg=self.scenario, representation=self.representation
        )

    def _generate_random_layout_batch(self, batch_size):
        return np_list_to_tensor_list(self.generate(batch_size))


class ReinforceDesigner(design.ReinforceDesigner[SC]):
    def __init__(
        self,
        designer_setting: design.DesignerParams[SC],
        lr: float = 1e-4,
        train_batch_size: int = 64,
        train_epochs: int = 5,
        gamma: float = 0.99,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            designer_setting=designer_setting,
            group_name=GROUP_NAME,
            group_aggregation="sum",
            lr=lr,
            train_batch_size=train_batch_size,
            train_epochs=train_epochs,
            gamma=gamma,
            device=device,
        )

    def _create_policy(self):
        return create_model(
            image_size=self.scenario.size,
            image_channels=self.scenario.n_colors,
            num_channels=64,
            num_res_blocks=2,
            resblock_updown=True,
            use_new_attention_order=True,
            attention_resolutions="16,8,4",
            num_head_channels=64,
        ).to(self.device)

    def _make_initial_env(self, batch_size: int):
        C = self.scenario.n_colors
        N = self.scenario.size
        initial_env = torch.zeros((batch_size, C, N, N), device=self.device)
        for idx, color in zip(self.scenario.goal_idxs, self.scenario.goal_colors):
            initial_env[:, color, *get_position(idx, N)] = 1
        return initial_env

    def _generate_env_action_batch(self, batch_size: int):
        B = batch_size
        C = self.scenario.n_colors
        N = self.scenario.size
        initial_env = self._make_initial_env(B)

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

    def _calculate_action_log_probs(self, actions):
        B = actions.shape[0]
        C = self.scenario.n_colors
        N = self.scenario.size
        assert actions.shape == (B, self.scenario.n_shelves), actions.shape

        action_log_prob_list = []

        initial_env = self._make_initial_env(B)
        logits = self.policy(initial_env).reshape(B, C, -1)
        batch_idxs = torch.arange(B)
        constructed_envs = torch.zeros((B, C, N * N), device=self.device)

        for i in range(self.scenario.n_shelves):
            channel_selection = i % self.scenario.n_colors  # [B]
            mask = constructed_envs.sum(dim=1) > 0  # [B, N * N]
            logits_i = logits[batch_idxs, channel_selection]  # [B, N * N]
            logits_i = logits_i.masked_fill(mask, float("-inf"))  # [B, N * N]
            idxs = actions[batch_idxs, i]  # [B]

            log_probs_i = torch.nn.functional.log_softmax(logits_i, dim=-1)
            action_log_probs = log_probs_i[batch_idxs, idxs]  # [B]
            action_log_prob_list.append(action_log_probs)
            constructed_envs[batch_idxs, channel_selection, idxs] = 1

        return torch.stack(action_log_prob_list).T


class ReplayDesigner(design.ReplayDesigner[SC]):
    def __init__(
        self,
        designer_setting: design.DesignerParams[SC],
        buffer_size: int = 1000,
        infill_ratio: float = 0.25,
        replay_sample_ratio: float = 0.9,
        stale_sample_ratio: float = 0.3,
        return_smoothing_factor: float = 0.8,
        return_sample_temperature: float = 0.1,
        ratio_of_shelves_moved_per_mutation: float = 0.1,
        gamma: float = 0.99,
    ):
        self.generate = make_generate_fn(
            designer_setting.scenario, representation="image"
        )
        super().__init__(
            group_name=GROUP_NAME,
            designer_setting=designer_setting,
            buffer_size=buffer_size,
            infill_ratio=infill_ratio,
            replay_sample_ratio=replay_sample_ratio,
            stale_sample_ratio=stale_sample_ratio,
            return_smoothing_factor=return_smoothing_factor,
            return_sample_temperature=return_sample_temperature,
            gamma=gamma,
            group_aggregation="sum",
        )
        self.shelves_moved_per_mutation = int(
            ratio_of_shelves_moved_per_mutation * self.scenario.n_shelves
        )

    def _generate_random_layout_batch(self, batch_size):
        return np_list_to_tensor_list(self.generate(batch_size))

    def _get_layout_from_state(self, state):
        return state[:, :, : self.scenario.n_colors]

    def _mutate(self, env: torch.Tensor):
        env = env.clone()
        empty_locations = []
        has_shelf_locations = []
        empty_locations_shelf = env.sum(dim=0) == 0

        for idx in range(self.scenario.size**2):
            if idx not in self.scenario.goal_idxs:
                if empty_locations_shelf[*get_position(idx, self.scenario.size)]:
                    empty_locations.append(idx)
                else:
                    has_shelf_locations.append(idx)
                empty_locations.append(idx)

        new_idxs = np.random.choice(
            empty_locations,
            size=self.shelves_moved_per_mutation,
            replace=False,
        )

        from_idxs = np.random.choice(
            has_shelf_locations,
            size=self.shelves_moved_per_mutation,
            replace=False,
        )

        for new_idx, from_idx in zip(new_idxs, from_idxs):
            pos = get_position(new_idx, self.scenario.size)
            color = env[:, *pos].argmax(dim=0)
            env[color, *pos] = 0
            env[color, *pos] = 1

        return env

    @staticmethod
    def _hash(env):
        return hashable_representation(env, representation="image")


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
            distill_embedding_hint=designer.distill_hint,
            distill_embedding_hint_loss_weight=designer.distill_hint_weight,
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
        elif isinstance(designer, Descent):
            designer_producer = DescentDesigner(
                designer_setting=designer_setting,
                classifier=designer.model,
                value_learner_hyperparameters=value_hyperparameters,
                random_generation_early_start=designer.random_generation_early_start,
                representation=designer.representation,
                lr=designer.gradient_lr,
                n_epochs=designer.gradient_epochs,
                n_gradient_iterations=designer.gradient_iterations,
                device=device,
            )
    elif isinstance(designer, Reinforce):
        designer_producer = ReinforceDesigner(
            designer_setting=designer_setting,
            lr=designer.lr,
            train_batch_size=designer.train_batch_size,
            train_epochs=designer.train_epochs,
            gamma=ppo_cfg.gamma,
            device=device,
        )
    elif isinstance(designer, Replay):
        designer_producer = ReplayDesigner(
            designer_setting=designer_setting,
            buffer_size=designer.buffer_size,
            infill_ratio=designer.infill_ratio,
            replay_sample_ratio=designer.replay_sample_ratio,
            stale_sample_ratio=designer.stale_sample_ratio,
            return_smoothing_factor=designer.return_smoothing_factor,
            return_sample_temperature=designer.return_sample_temperature,
            ratio_of_shelves_moved_per_mutation=designer.mutation_ratio,
            gamma=ppo_cfg.gamma,
        )

    return designer_producer, design_consumer_fn
