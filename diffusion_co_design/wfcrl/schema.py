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
    early_start: int | None = None


class Diffusion(_Value):
    type: Literal["diffusion"]
    diffusion: DiffusionOperation = DiffusionOperation(
        num_recurrences=8, backward_lr=0.01, backward_steps=16, forward_guidance_wt=5.0
    )


DesignerConfig = Annotated[
    Random | Fixed | Diffusion,
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
    normalize_reward: bool
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

    def dump(self) -> dict:
        out = super().model_dump()
        out["scenario"] = self.scenario.model_dump()
        return out
