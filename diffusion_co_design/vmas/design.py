from typing import Callable
from pathlib import Path
import torch

from diffusion_co_design.common import design, np_list_to_tensor_list
from diffusion_co_design.common.rl.mappo.schema import PPOConfig
from diffusion_co_design.vmas.static import GROUP_NAME
from diffusion_co_design.vmas.diffusion.generate import Generate
from diffusion_co_design.vmas.schema import (
    ScenarioConfig as SC,
    Random,
    Fixed,
    DesignerConfig,
    EnvCriticConfig,
)
from diffusion_co_design.vmas.model.classifier import EnvCritic


class RandomDesigner(design.RandomDesigner[SC]):
    def __init__(
        self,
        designer_setting: design.DesignerParams[SC],
        seed: int | None = None,
    ):
        super().__init__(designer_setting=designer_setting)
        self.generate = Generate(scenario=self.scenario, rng=seed)

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
                Generate(scenario=designer_setting.scenario, rng=seed)(
                    n=1, training_dataset=False
                )[0]
            ),
        )


default_value_learner_hyperparameters = design.ValueLearnerHyperparameters(
    lr=3e-5,
    train_batch_size=64,
    buffer_size=2048,
    n_update_iterations=10,
    clip_grad_norm=1.0,
    distill_from_critic=False,
    distill_samples=1,
    loss_criterion="mse",
)


class ValueLearner(design.ValueLearner):
    def __init__(
        self,
        scenario: SC,
        classifier: EnvCriticConfig,
        gamma=0.99,
        hyperparameters=default_value_learner_hyperparameters,
        device=torch.device("cpu"),
    ):
        super().__init__(
            model=EnvCritic(
                scenario=scenario,
                node_emb_dim=classifier.hidden_size,
                num_layers=classifier.depth,
                k=classifier.k,
            ),
            group_name=GROUP_NAME,
            episode_steps=scenario.get_episode_steps(),
            gamma=gamma,
            hyperparameters=hyperparameters,
            group_aggregation="sum",
            random_designer=RandomDesigner(design.DesignerParams.placeholder(scenario)),
            device=device,
        )

    def _get_layout_from_state(self, state):
        return state

    def _eval_to_train(self, theta):
        pass


# class DicodeDesigner(design.DicodeDesigner[SC]):
#     def __init__(
#         self,
#         designer_setting,
#         value_learner,
#         diffusion_generator,
#         diffusion_setting,
#         random_generation_early_start=0,
#         total_annealing_iters=1000,
#     ):
#         super().__init__(
#             designer_setting=designer_setting,
#             value_learner,
#             diffusion_generator,
#             diffusion_setting,
#             random_generation_early_start,
#             total_annealing_iters,
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

    return designer_producer, design_consumer_fn
