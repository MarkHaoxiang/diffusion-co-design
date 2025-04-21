import os

from typing import Any, TypeAlias, Literal, Sequence
from diffusion_co_design.common import (
    Config,
    LoggingConfig,
    PPOConfig,
    MEMORY_MANAGEMENT,
    OUTPUT_DIR,
)


Representation: TypeAlias = Literal["image", "flat", "graph"]


class ScenarioConfig(Config):
    name: str
    size: int
    n_agents: int
    n_shelves: int
    n_goals: int
    agent_idxs: list[int]
    agent_colors: list[int]
    goal_idxs: list[int]
    goal_colors: list[int]
    n_colors: int
    max_steps: int


class DiffusionOperation(Config):
    num_recurrences: int = 4
    backward_lr: float = 0.01
    backward_steps: int = 0
    forward_guidance_wt: float = 5.0
    early_start: bool = (
        True  # Fill environment buffer with randomly generated environments
    )


class ClassifierConfig(Config):
    name: str
    representation: Literal["graph", "image", "graph_warehouse"]
    model_kwargs: dict[str, Any] = {}


class RLConfig(Config):
    version: Literal["v1", "v2"] = "v1"
    # V1
    kernel_sizes: int | Sequence[int] = 3
    num_cells: int | Sequence[int] = [16, 32, 64]
    strides: Sequence[int] | int = 1
    hidden_size: int = 128
    depth: int = 2
    share_params: bool = True
    # V2
    critic_kwargs: dict | None = None


class DesignerConfig(Config):
    type: str
    environment_repeats: int = 1
    value_model: ClassifierConfig | None = None
    value_n_update_iterations: int = 5
    value_train_batch_size: int = 64
    value_buffer_size: int = 4096
    value_weight_decay: float = 0.05
    value_lr: float = 3e-5
    value_distill_enable: bool = False
    value_distill_samples: int = 1
    diffusion: DiffusionOperation = DiffusionOperation()


class TrainingConfig(Config):
    # Problem definition: Built with diffusion.datasets.rware.generate
    memory_management: MEMORY_MANAGEMENT = "gpu"
    experiment_name: str
    designer: DesignerConfig
    scenario_name: str
    # Training
    ppo: PPOConfig
    # Policy
    policy: RLConfig = RLConfig()
    # Logging
    logging: LoggingConfig = LoggingConfig()
    start_from_checkpoint: str | None = None

    @property
    def scenario(self) -> ScenarioConfig:
        if not hasattr(self, "_scenario_cache"):
            file = os.path.join(
                OUTPUT_DIR, "rware", "scenario", self.scenario_name, "config.yaml"
            )
            self._scenario_cache = ScenarioConfig.from_file(file)
        return self._scenario_cache
