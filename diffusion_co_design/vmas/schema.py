from typing import Annotated, Literal
from pydantic import model_validator, Field

from diffusion_co_design.common import Config, DiffusionOperation
from diffusion_co_design.common.design import DesignerConfig as _Designer
from diffusion_co_design.common.env import ScenarioConfig as _ScenarioConfig
from diffusion_co_design.common.rl.mappo.schema import TrainingConfig as _TrainingConfig
from diffusion_co_design.vmas.static import ENV_NAME


class ScenarioConfig(_ScenarioConfig):
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

    @property
    def n_obstacles(self):
        return len(self.obstacle_sizes)

    @model_validator(mode="after")
    def check_agent_numbers(self):
        if len(self.agent_spawns) != len(self.agent_goals):
            raise ValueError("Number of agent spawns must match number of agent goals.")
        return self


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
    _TrainingConfig[DesignerConfig, ScenarioConfig, ActorCriticConfig]
):
    @property
    def env_name(self) -> str:
        return ENV_NAME

    @property
    def _scenario_cfg_cls(self) -> type[ScenarioConfig]:
        return ScenarioConfig
