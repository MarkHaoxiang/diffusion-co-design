from pydantic import model_validator

from diffusion_co_design.common import Config
from diffusion_co_design.common.env import ScenarioConfig as _ScenarioConfig


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

    @model_validator(mode="after")
    def check_agent_numbers(self):
        if len(self.agent_spawns) != len(self.agent_goals):
            raise ValueError("Number of agent spawns must match number of agent goals.")
        return self


class ActorConfig(Config):
    depth: int = 2
    hidden_size: int = 256


class CriticConfig(Config):
    pass


class ActorCriticConfig(Config):
    actor: ActorConfig
    critic: CriticConfig
