import os
from typing import Annotated, Literal
from pydantic import Field

from diffusion_co_design.common import (
    Config,
    PPOConfig,
    LoggingConfig,
    DeviceConfig,
    DiffusionOperation,
    OUTPUT_DIR,
)


class ScenarioConfig(Config):
    name: str
    n_turbines: int
    max_steps: int
    map_x_length: int
    map_y_length: int
    min_distance_between_turbines: int


class NormalisationStatistics(Config):
    episode_return_mean: float
    episode_return_std: float
    reward_mean: float
    reward_std: float


class ClassifierConfig(Config):
    node_emb_size: int = 64
    edge_emb_size: int = 32
    depth: int = 2


# ====
# Designer registry


class _Designer(Config):
    environment_repeats: int = 1


class Random(_Designer):
    type: Literal["random"]


class Fixed(_Designer):
    type: Literal["fixed"]


class _Value(_Designer):
    model: ClassifierConfig
    batch_size: int = 64
    buffer_size: int = 2048
    lr: float = 3e-4
    n_update_iterations: int = 10
    clip_grad_norm: float | None = 1.0
    weight_decay: float = 0.0
    distill_enable: bool = False
    distill_samples: int = 5
    loss_criterion: Literal["mse", "huber"] = "huber"
    diffusion_early_start: int | None = None
    train_early_start: int = 0


class Sampling(_Value):
    type: Literal["sampling"]
    n_samples: int = 16


class Diffusion(_Value):
    type: Literal["diffusion"]
    diffusion: DiffusionOperation


DesignerConfig = Annotated[
    Random | Fixed | Diffusion | Sampling,
    Field(
        discriminator="type",
    ),
]

# ====


class RLConfig(Config):
    model_type: Literal["mlp", "gnn"]
    initial_std: float = 0.3
    # MLP
    mlp_hidden_size: int = 64
    mlp_depth: int = 2
    # GNN
    policy_node_hidden_size: int = 64
    policy_edge_hidden_size: int = 16
    policy_head_hidden_size: int = 64
    policy_gnn_depth: int = 3
    policy_head_depth: int = 2
    critic_node_hidden_size: int = 64
    critic_edge_hidden_size: int = 16
    critic_gnn_depth: int = 3


class TrainingConfig(Config):
    experiment_name: str
    device: DeviceConfig = DeviceConfig()
    scenario_name: str
    policy: RLConfig
    ppo: PPOConfig
    logging: LoggingConfig
    designer: DesignerConfig
    start_from_checkpoint: str | None = None

    @property
    def scenario(self) -> ScenarioConfig:
        if not hasattr(self, "_scenario_cache"):
            file = os.path.join(
                OUTPUT_DIR, "wfcrl", "scenario", self.scenario_name, "config.yaml"
            )
            self._scenario_cache = ScenarioConfig.from_file(file)
        return self._scenario_cache

    @property
    def normalisation(self) -> NormalisationStatistics | None:
        if not hasattr(self, "_normalisation_cache"):
            file = os.path.join(
                OUTPUT_DIR,
                "wfcrl",
                "scenario",
                self.scenario_name,
                "normalisation_statistics.yaml",
            )
            self._normalisation_cache = NormalisationStatistics.from_file(file)
        return self._normalisation_cache

    def dump(self) -> dict:
        out = super().model_dump()
        out["scenario"] = self.scenario.model_dump()
        if self.normalisation is not None:
            out["normalisation_statistics"] = self.normalisation.model_dump()
        else:
            out["normalisation_statistics"] = None
        return out
