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


class Fixed(_Designer):
    kind: Literal["fixed"]
    representation: Representation = "image"


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
    distill_synthetic_ratio: float = 0.0
    distill_synthetic_ood_ratio: float = 1.0
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


class Reinforce(_Designer):
    kind: Literal["reinforce"]
    representation: Representation = "image"
    lr: float = 1e-4
    train_batch_size: int = 20
    train_epochs: int = 1

    @model_validator(mode="after")
    def validate_representation(cls, values):
        assert values.representation == "image"
        return values


class Replay(_Designer):
    kind: Literal["replay"]
    representation: Representation = "image"
    buffer_size: int = 1000
    infill_ratio: float = 0.25
    replay_sample_ratio: float = 0.9
    stale_sample_ratio: float = 0.3
    return_smoothing_factor: float = 0.8
    return_sample_temperature: float = 0.1
    mutation_ratio: float = 0.1

    @model_validator(mode="after")
    def validate_representation(cls, values):
        assert values.representation == "image"
        return values


DesignerConfig = Annotated[
    Random | Fixed | Diffusion | Sampling | Descent | Reinforce | Replay,
    Field(discriminator="kind"),
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
