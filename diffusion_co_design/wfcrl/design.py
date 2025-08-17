import os
from typing import Callable
from pathlib import Path

import numpy as np
import torch

from diffusion_co_design.common import design
from diffusion_co_design.common.rl.mappo.schema import PPOConfig
from diffusion_co_design.wfcrl.static import GROUP_NAME, ENV_NAME
from diffusion_co_design.wfcrl.schema import (
    NormalisationStatistics,
    ScenarioConfig as SC,
    DesignerConfig,
    EnvCriticConfig,
    _Value,
    Fixed,
    Random,
    Diffusion,
    Sampling,
)
from diffusion_co_design.wfcrl.model.classifier import GNNCritic
from diffusion_co_design.wfcrl.model.rl import maybe_make_denormaliser
from diffusion_co_design.wfcrl.diffusion.generate import Generate
from diffusion_co_design.wfcrl.diffusion.generator import (
    Generator,
    eval_to_train,
    soft_projection_constraint,
)
from diffusion_co_design.common import (
    DiffusionOperation,
    OUTPUT_DIR,
    get_latest_model,
    np_list_to_tensor_list,
)


def make_generate_fn(scenario: SC, seed: int | None = None):
    return Generate(
        num_turbines=scenario.n_turbines,
        map_x_length=scenario.map_x_length,
        map_y_length=scenario.map_y_length,
        minimum_distance_between_turbines=scenario.min_distance_between_turbines,
        rng=seed,
    )


class RandomDesigner(design.RandomDesigner[SC]):
    def __init__(
        self,
        designer_setting: design.DesignerParams[SC],
        seed: int | None = None,
    ):
        super().__init__(designer_setting=designer_setting)
        self.generate = make_generate_fn(self.scenario, seed)

    def generate_random_layouts(self, batch_size):
        return np_list_to_tensor_list(
            self.generate(n=batch_size, training_dataset=False)
        )


class FixedDesigner(design.FixedDesigner[SC]):
    def __init__(
        self, designer_setting: design.DesignerParams, seed: int | None = None
    ):
        super().__init__(
            designer_setting=designer_setting,
            layout=torch.tensor(
                make_generate_fn(designer_setting.scenario, seed)(n=1)
            ).squeeze(0),
        )


default_value_learner_hyperparameters = design.ValueLearnerHyperparameters(
    lr=3e-5,
    train_batch_size=64,
    buffer_size=2048,
    n_update_iterations=10,
    clip_grad_norm=1.0,
    distill_from_critic=False,
    distill_samples=1,
    loss_criterion="huber",
)


class ValueLearner(design.ValueLearner):
    def __init__(
        self,
        scenario: SC,
        classifier: EnvCriticConfig,
        normalisation_statistics: NormalisationStatistics | None = None,
        hyperparameters: design.ValueLearnerHyperparameters = default_value_learner_hyperparameters,
        gamma: float = 0.99,
        device="cpu",
    ):
        self.scenario = scenario
        super().__init__(
            model=GNNCritic(
                cfg=scenario,
                node_emb_dim=classifier.node_emb_size,
                edge_emb_dim=classifier.edge_emb_size,
                n_layers=classifier.depth,
                post_hook=maybe_make_denormaliser(normalisation_statistics),
            ),
            group_name=GROUP_NAME,
            episode_steps=scenario.get_episode_steps(),
            hyperparameters=hyperparameters,
            gamma=gamma,
            group_aggregation="mean",
            device=device,
        )

    def _get_layout_from_state(self, state):
        return state["layout"]

    def _eval_to_train(self, theta):
        return eval_to_train(theta, self.scenario)


def make_reference_layouts(scenario: SC, device):
    generate = make_generate_fn(scenario, seed=0)
    return torch.tensor(
        np.array(generate(n=64, training_dataset=True)),
        dtype=torch.float32,
        device=device,
    )


class SamplingDesigner(design.SamplingDesigner[SC]):
    def __init__(
        self,
        designer_setting: design.DesignerParams[SC],
        classifier: EnvCriticConfig,
        normalisation_statistics: NormalisationStatistics | None = None,
        value_learner_hyperparameters: design.ValueLearnerHyperparameters = default_value_learner_hyperparameters,
        random_generation_early_start: int = 0,
        gamma: float = 0.99,
        device="cpu",
        n_samples: int = 16,
    ):
        super().__init__(
            designer_setting=designer_setting,
            value_learner=ValueLearner(
                scenario=designer_setting.scenario,
                classifier=classifier,
                normalisation_statistics=normalisation_statistics,
                hyperparameters=value_learner_hyperparameters,
                gamma=gamma,
                device=device,
            ),
            random_generation_early_start=random_generation_early_start,
            n_samples=n_samples,
        )
        self.generate = make_generate_fn(self.scenario)

    def generate_random_layouts(self, batch_size):
        layouts = torch.tensor(
            self.generate(
                n=batch_size,
                training_dataset=False,
            ),
            dtype=torch.float32,
            device=self.value_learner.device,
        )
        return layouts

    def _make_reference_layouts(self):
        return make_reference_layouts(self.scenario, self.value_learner.device)

    def _generate_random_layout_batch(self, batch_size: int):
        return list(self.generate_random_layouts(batch_size))


class DicodeDesigner(design.DicodeDesigner[SC]):
    def __init__(
        self,
        designer_setting: design.DesignerParams[SC],
        classifier: EnvCriticConfig,
        diffusion: DiffusionOperation,
        normalisation_statistics: NormalisationStatistics | None = None,
        value_learner_hyperparameters: design.ValueLearnerHyperparameters = default_value_learner_hyperparameters,
        random_generation_early_start: int = 0,
        gamma: float = 0.99,
        total_annealing_iters: int = 1000,
        device="cpu",
    ):
        sc = designer_setting.scenario
        super().__init__(
            designer_setting=designer_setting,
            value_learner=ValueLearner(
                scenario=sc,
                classifier=classifier,
                normalisation_statistics=normalisation_statistics,
                hyperparameters=value_learner_hyperparameters,
                gamma=gamma,
                device=device,
            ),
            diffusion_setting=diffusion,
            diffusion_generator=Generator(
                generator_model_path=get_latest_model(
                    os.path.join(OUTPUT_DIR, ENV_NAME, "diffusion", sc.name), "model"
                ),
                scenario=sc,
                default_guidance_wt=diffusion.forward_guidance_wt,
                device=device,
            ),
            random_generation_early_start=random_generation_early_start,
            total_annealing_iters=total_annealing_iters,
        )
        self.pc = soft_projection_constraint(self.scenario)
        self.generate = RandomDesigner(designer_setting)

    def projection_constraint(self, x):
        return self.pc(x)

    def _make_reference_layouts(self):
        return make_reference_layouts(self.scenario, self.value_learner.device)

    def _generate_random_layout_batch(self, batch_size: int):
        return self.generate.generate_random_layouts(batch_size)


def create_designer(
    scenario: SC,
    designer: DesignerConfig,
    ppo_cfg: PPOConfig,
    normalisation_statistics: NormalisationStatistics | None,
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
            train_batch_size=designer.batch_size,
            buffer_size=designer.buffer_size,
            n_update_iterations=designer.n_update_iterations,
            clip_grad_norm=designer.clip_grad_norm,
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
                normalisation_statistics=normalisation_statistics,
                value_learner_hyperparameters=value_hyperparameters,
                random_generation_early_start=designer.random_generation_early_start,
                gamma=ppo_cfg.gamma,
                n_samples=designer.n_samples,
                device=device,
            )
        elif isinstance(designer, Diffusion):
            designer_producer = DicodeDesigner(
                diffusion=designer.diffusion,
                classifier=designer.model,
                designer_setting=designer_setting,
                normalisation_statistics=normalisation_statistics,
                value_learner_hyperparameters=value_hyperparameters,
                random_generation_early_start=designer.random_generation_early_start,
                gamma=ppo_cfg.gamma,
                total_annealing_iters=ppo_cfg.n_iters,
                device=device,
            )

    return designer_producer, design_consumer_fn
