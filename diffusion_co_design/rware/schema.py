from typing import Any, Annotated, TypeAlias, Literal, Sequence

from pydantic import Field, model_validator

from diffusion_co_design.common import Config, DiffusionOperation
from diffusion_co_design.common.rl.mappo.schema import (
    TrainingConfig as _TrainingConfig,
    ScenarioConfig as _ScenarioConfig,
)
from diffusion_co_design.common.design import DesignerConfig as __Designer
from diffusion_co_design.rware.static import ENV_NAME

Representation: TypeAlias = Literal["image", "flat", "graph"]


class ScenarioConfig(_ScenarioConfig):
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

    def get_name(self) -> str:
        return self.name

    def get_episode_steps(self) -> int:
        return self.max_steps

    def get_num_agents(self) -> int:
        return self.n_agents


class EnvCriticConfig(Config):
    name: str
    representation: Literal["graph", "image", "graph_warehouse"]
    model_kwargs: dict[str, Any] = {}


# ====
# Actor critic registry


class ActorCriticConfigV1(Config):
    version: Literal["v1"] = "v1"
    kernel_sizes: int | Sequence[int] = 3
    num_cells: int | Sequence[int] = [16, 32, 64]
    strides: Sequence[int] | int = 1
    hidden_size: int = 128
    depth: int = 2
    share_params: bool = True


class ActorCRiticConfigV2(Config):
    version: Literal["v2"] = "v2"
    kernel_sizes: int | Sequence[int] = 3
    num_cells: int | Sequence[int] = [16, 32, 64]
    strides: Sequence[int] | int = 1
    hidden_size: int = 128
    depth: int = 2
    share_params: bool = True
    critic_kwargs: dict | None = None


ActorCriticConfig = Annotated[
    ActorCriticConfigV1 | ActorCRiticConfigV2,
    Field(
        discriminator="version",
    ),
]

# ====
# Designer registry


class _Designer(__Designer):
    representation: Representation


class Random(_Designer):
    kind: Literal["random"]
    representation: Representation = "image"

    @model_validator(mode="after")
    def validate_representation(cls, values):
        assert values.representation == "image"
        return values


class Fixed(_Designer):
    kind: Literal["fixed"]
    representation: Representation = "image"

    @model_validator(mode="after")
    def validate_representation(cls, values):
        assert values.representation == "image"
        return values


class _Value(_Designer):
    model: EnvCriticConfig
    n_update_iterations: int = 5
    train_batch_size: int = 64
    buffer_size: int = 4096
    weight_decay: float = 0.0
    lr: float = 3e-5
    distill_enable: bool = False
    distill_samples: int = 5
    distill_hint: bool = False
    distill_hint_weight: float = 0.1
    random_generation_early_start: int = 0
    loss_criterion: Literal["mse", "huber"] = "mse"
    train_early_start: int = 0

    @model_validator(mode="after")
    def validate_representation(cls, values):
        assert values.model.representation == values.representation
        return values


class Sampling(_Value):
    kind: Literal["sampling"]
    n_samples: int = 16


class Diffusion(_Value):
    kind: Literal["diffusion"]
    diffusion: DiffusionOperation


class Descent(_Value):
    kind: Literal["descent"]
    gradient_epochs: int = 20
    gradient_iterations: int = 10
    gradient_lr: float = 0.03


DesignerConfig = Annotated[
    Random | Fixed | Diffusion | Sampling | Descent, Field(discriminator="kind")
]


class TrainingConfig(
    _TrainingConfig[DesignerConfig, ScenarioConfig, ActorCriticConfig]
):
    @property
    def env_name(self) -> str:
        return ENV_NAME

    @property
    def _scenario_cfg_cls(self) -> type[ScenarioConfig]:
        return ScenarioConfig
