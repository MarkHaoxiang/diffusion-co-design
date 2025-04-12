import os

from typing import Any, TypeAlias, Literal, Sequence
from diffusion_co_design.common import (
    Config,
    LoggingConfig,
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
    # CNN
    kernel_sizes: int | Sequence[int] = 3
    num_cells: int | Sequence[int] = [16, 32, 64]
    strides: Sequence[int] | int = 1
    # MLP
    hidden_size: int = 128
    depth: int = 2
    # Both
    share_params: bool = True


class DesignerConfig(Config):
    type: str
    environment_repeats: int = 1
    value_model: ClassifierConfig | None = None
    value_n_update_iterations: int = 5
    value_train_batch_size: int = 64
    value_buffer_size: int = 4096
    value_weight_decay: float = 0.05
    value_lr: float = 3e-5
    diffusion: DiffusionOperation = DiffusionOperation()


class TrainingConfig(Config):
    # Problem definition: Built with diffusion.datasets.rware.generate
    experiment_name: str
    designer: DesignerConfig
    scenario_name: str
    # Sampling and training
    memory_management: MEMORY_MANAGEMENT = "gpu"
    n_iters: int = 500  # Number of training iterations
    n_epochs: int = 10  # Number of optimization steps per training iteration
    minibatch_size: int = 800  # Size of the mini-batches in each optimization step
    n_mini_batches: int = 20  # Number of mini-batches in each epoch
    ppo_lr: float = 5e-4  # Learning rate
    max_grad_norm: float = 1.0  # Maximum norm for the gradients
    # PPO
    clip_epsilon: float = 0.2  # clip value for PPO loss
    gamma: float = 0.99  # discount factor
    lmbda: float = 0.9  # lambda for generalised advantage estimation
    entropy_eps: float = 1e-4  # coefficient of the entropy term in the PPO loss
    # Policy
    policy: RLConfig = RLConfig()
    # Logging
    logging: LoggingConfig = LoggingConfig()
    start_from_checkpoint: str | None = None

    @property
    def frames_per_batch(self) -> int:
        return self.n_mini_batches * self.minibatch_size

    @property
    def total_frames(self) -> int:
        return self.frames_per_batch * self.n_iters

    @property
    def scenario(self) -> ScenarioConfig:
        if not hasattr(self, "_scenario_cache"):
            file = os.path.join(
                OUTPUT_DIR, "rware", "scenario", self.scenario_name, "config.yaml"
            )
            self._scenario_cache = ScenarioConfig.from_file(file)
        return self._scenario_cache
