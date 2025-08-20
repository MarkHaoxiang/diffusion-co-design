import pytest
from expecttest import assert_expected_inline
import torch

from diffusion_co_design.vmas.schema import (
    ActorCriticConfig,
    ActorConfig,
    CriticConfig,
    GlobalPlacementScenarioConfig,
)

try:
    import vmas

    from diffusion_co_design.vmas.design import FixedDesigner
    from diffusion_co_design.vmas.scenario.env import create_env
    from diffusion_co_design.vmas.model.rl import vmas_models
    from diffusion_co_design.vmas.static import GROUP_NAME

    vmas_available = True
except ImportError:
    vmas_available = False

from diffusion_co_design.common.design import DesignerParams


seed = 42
device = torch.device("cpu")

test_scenario = GlobalPlacementScenarioConfig(
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
        actor_critic_cfg=actor_critic_config,
        device=device,
    )
    return actor, critic


@pytest.fixture
def critic(actor_critic):
    return actor_critic[1]


def test_construct_graph(critic):
    critic = critic.module
    agent_pos = torch.rand(4, 2, device=device)
    goal_pos = torch.rand(4, 2, device=device)
    obstacle_pos = torch.rand(3, 2, device=device)
    agent_vel = torch.rand(4, 2, device=device)

    graph_1 = critic.model.construct_graph(
        obstacle_pos=obstacle_pos,
        agent_pos=agent_pos,
        goal_pos=goal_pos,
        agent_vel=agent_vel,
    )

    # Apply rotation and translation
    theta = torch.rand(1) * 2 * torch.pi
    R = torch.tensor(
        [[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]
    )
    t = torch.rand(1, 2)
    agent_pos_rot = agent_pos @ R.T + t
    goal_pos_rot = goal_pos @ R.T + t
    obstacle_pos_rot = obstacle_pos @ R.T + t
    agent_vel_rot = agent_vel @ R.T

    graph_2 = critic.model.construct_graph(
        obstacle_pos=obstacle_pos_rot,
        agent_pos=agent_pos_rot,
        goal_pos=goal_pos_rot,
        agent_vel=agent_vel_rot,
    )

    assert torch.allclose(graph_1.x, graph_2.x, atol=1e-3)
    assert torch.allclose(graph_1.edge_index, graph_2.edge_index, atol=1e-3)
    assert torch.allclose(graph_1.edge_attr, graph_2.edge_attr, atol=1e-3)


def test_execution(env, actor_critic):
    actor, critic = actor_critic
    td = env.reset()

    td = actor(td)
    td = critic(td)

    assert_expected_inline(
        str(td[GROUP_NAME]),
        """\
TensorDict(
    fields={
        action: Tensor(shape=torch.Size([1, 4, 2]), device=cpu, dtype=torch.float32, is_shared=False),
        info: TensorDict(
            fields={
                agent_collisions: Tensor(shape=torch.Size([1, 4, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                final_rew: Tensor(shape=torch.Size([1, 4, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                pos_rew: Tensor(shape=torch.Size([1, 4, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([1, 4]),
            device=cpu,
            is_shared=False),
        loc: Tensor(shape=torch.Size([1, 4, 2]), device=cpu, dtype=torch.float32, is_shared=False),
        observation: Tensor(shape=torch.Size([1, 4, 18]), device=cpu, dtype=torch.float32, is_shared=False),
        sample_log_prob: Tensor(shape=torch.Size([1, 4]), device=cpu, dtype=torch.float32, is_shared=False),
        scale: Tensor(shape=torch.Size([1, 4, 2]), device=cpu, dtype=torch.float32, is_shared=False),
        state_value: Tensor(shape=torch.Size([1, 4, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
    batch_size=torch.Size([1, 4]),
    device=cpu,
    is_shared=False)""",
    )
