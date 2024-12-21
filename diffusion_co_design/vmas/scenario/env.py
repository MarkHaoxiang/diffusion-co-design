import torch

from diffusion_co_design.common.design.base import DesignConsumer
from diffusion_co_design.common.env import ENVIRONMENT_MODE
from diffusion_co_design.vmas.schema import (
    ScenarioConfigType,
    SimpleSpreadScenarioConfig,
)

from .obstacle_navigation_env import create_env as obstacle_navigation_create_env
from .simple_spread_env import create_env as simple_spread_create_env


def create_env(
    mode: ENVIRONMENT_MODE,
    scenario: ScenarioConfigType,
    designer: DesignConsumer,
    num_environments: int = 1,
    device: torch.device = torch.device("cpu"),
):
    if isinstance(scenario, SimpleSpreadScenarioConfig):
        return simple_spread_create_env(
            mode=mode,
            scenario=scenario,
            designer=designer,
            num_environments=num_environments,
            device=device,
        )
    else:
        return obstacle_navigation_create_env(
            mode=mode,
            scenario=scenario,
            designer=designer,
            num_environments=num_environments,
            device=device,
        )
