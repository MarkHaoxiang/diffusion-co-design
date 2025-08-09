import pytest
import torch

from diffusion_co_design.vmas.schema import (
    ActorCriticConfig,
    ActorConfig,
    CriticConfig,
    ScenarioConfig,
)

from diffusion_co_design.common.design import DesignerParams
from diffusion_co_design.vmas.design import FixedDesigner
from diffusion_co_design.vmas.scenario.env import create_env
from diffusion_co_design.vmas.model.rl import vmas_models

seed = 42
device = torch.device("cpu")

test_scenario = ScenarioConfig(
    name="test_scenario",
    world_spawning_x=1,
    world_spawning_y=1,
    episode_steps=100,
    agent_spawns=[
        (-0.9, -0.9),
        (-0.75, -0.9),
        (0.75, -0.9),
        (0.9, -0.9),
    ],
    agent_goals=[
        (-0.9, 0.9),
        (-0.75, 0.9),
        (0.75, 0.9),
        (0.9, 0.9),
    ],
    obstacle_sizes=[0.1, 0.2, 0.3],
)

actor_critic_config = ActorCriticConfig(actor=ActorConfig(), critic=CriticConfig())


@pytest.fixture
def designer():
    return FixedDesigner(
        designer_setting=DesignerParams.placeholder(scenario=test_scenario),
        seed=seed,
    ).get_placeholder()


@pytest.fixture
def env(designer):
    env = create_env(
        mode="reference", scenario=test_scenario, designer=designer, device=device
    )
    yield env
    env.close()


@pytest.fixture
def actor_critic(env):
    actor, critic = vmas_models(
        env=env,
        scenario=test_scenario,
        actor_critic_config=actor_critic_config,
        device=device,
    )
    return actor, critic


def test_execution(env):
    td = env.reset()
    print(td)
    assert False
