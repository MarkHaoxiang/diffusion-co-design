from typing import Annotated, Literal
from abc import ABC, abstractmethod
from pydantic import model_validator, Field, TypeAdapter
from omegaconf import OmegaConf, DictConfig

import torch

from diffusion_co_design.common import Config, DiffusionOperation
from diffusion_co_design.common.design import DesignerConfig as _Designer
from diffusion_co_design.common.env import ScenarioConfig as BaseScenarioConfig
from diffusion_co_design.common.rl.mappo.schema import TrainingConfig as _TrainingConfig
from diffusion_co_design.vmas.static import ENV_NAME


class _ScenarioConfig(BaseScenarioConfig, ABC):
    name: str
    world_spawning_x: float
    world_spawning_y: float
    episode_steps: int
    agent_spawns: list[tuple[float, float]]
    agent_goals: list[tuple[float, float]]
    obstacle_sizes: list[float]

    def get_name(self) -> str:
        return self.name

    def get_episode_steps(self):
        return self.episode_steps

    def get_num_agents(self):
        return len(self.agent_spawns)

    @model_validator(mode="after")
    def check_agent_numbers(self):
        if len(self.agent_spawns) != len(self.agent_goals):
            raise ValueError("Number of agent spawns must match number of agent goals.")
        return self

    @property
    def n_obstacles(self):
        return len(self.obstacle_sizes)

    @property
    @abstractmethod
    def layout_space_high(self) -> torch.Tensor:
        raise NotImplementedError()

    @property
    def layout_space_low(self) -> torch.Tensor:
        return -self.layout_space_high

    @property
    def diffusion_shape(self):
        return self.layout_space_high.shape


class GlobalPlacementScenarioConfig(_ScenarioConfig):
    placement_area: Literal["global"] = "global"

    @property
    def layout_space_high(self) -> torch.Tensor:
        return (
            torch.tensor([self.world_spawning_x, self.world_spawning_y])
            .unsqueeze(0)
            .expand(self.n_obstacles, 2)
        )


class LocalPlacementScenarioConfig(_ScenarioConfig):
    placement_area: Literal["local"] = "local"
    obstacle_bounds: list[tuple[tuple[float, float], tuple[float, float]]]

    @property
    def layout_space_high(self) -> torch.Tensor:
        n = 0
        for (x_low, x_high), (y_low, y_high) in self.obstacle_bounds:
            n += 2 - (x_low == x_high) - (y_low == y_high)
        return torch.ones((n,))


ScenarioConfigType = Annotated[
    GlobalPlacementScenarioConfig | LocalPlacementScenarioConfig,
    Field(discriminator="placement_area"),
]


class ScenarioConfig:
    _adapter: TypeAdapter = TypeAdapter(ScenarioConfigType)

    @classmethod
    def parse(cls, data) -> ScenarioConfigType:
        return cls._adapter.validate_python(data)

    @classmethod
    def json_parse(cls, data: str) -> ScenarioConfigType:
        return cls._adapter.validate_json(data)

    @classmethod
    def from_raw(cls, config):
        return cls.parse(config)

    @classmethod
    def from_file(cls, path: str):
        with open(path, "r") as f:
            raw = OmegaConf.load(f)
            assert isinstance(raw, DictConfig)
            config = cls.from_raw(raw)
        return config


class ActorConfig(Config):
    depth: int = 2
    hidden_size: int = 128


class CriticConfig(Config):
    depth: int = 3
    hidden_size: int = 128
    k: int = 5


class ActorCriticConfig(Config):
    actor: ActorConfig
    critic: CriticConfig


class EnvCriticConfig(Config):
    depth: int = 3
    hidden_size: int = 128
    k: int = 5


# ====
# Designer registry
class Random(_Designer):
    kind: Literal["random"]


class Fixed(_Designer):
    kind: Literal["fixed"]


class _Value(_Designer):
    model: EnvCriticConfig
    batch_size: int = 64
    buffer_size: int = 2048
    lr: float = 3e-4
    n_update_iterations: int = 10
    clip_grad_norm: float | None = 1.0
    weight_decay: float = 0.0
    distill_enable: bool = False
    distill_samples: int = 5
    loss_criterion: Literal["mse", "huber"] = "huber"
    random_generation_early_start: int = 0
    train_early_start: int = 0


class Diffusion(_Value):
    kind: Literal["diffusion"]
    diffusion: DiffusionOperation


DesignerConfig = Annotated[Random | Fixed | Diffusion, Field(discriminator="kind")]


class TrainingConfig(
    _TrainingConfig[DesignerConfig, ScenarioConfigType, ActorCriticConfig]
):
    @property
    def env_name(self) -> str:
        return ENV_NAME

    @property
    def _scenario_cfg_cls(self) -> type[ScenarioConfig]:  # type: ignore
        return ScenarioConfig
