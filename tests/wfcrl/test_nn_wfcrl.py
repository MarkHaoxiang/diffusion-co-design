import pytest
from expecttest import assert_expected_inline
import torch

from diffusion_co_design.common.rl.util import create_batched_env
from diffusion_co_design.common.design import DesignerParams
from diffusion_co_design.wfcrl.design import FixedDesigner
from diffusion_co_design.wfcrl.env import create_env
from diffusion_co_design.wfcrl.model.rl import wfcrl_models_gnn
from diffusion_co_design.wfcrl.schema import GNNActorCriticConfig, ScenarioConfig

seed = 42
device = torch.device("cpu")

test_scenario = ScenarioConfig(
    name="test_scenario",
    n_turbines=7,
    max_steps=100,
    map_x_length=1000,
    map_y_length=1000,
    min_distance_between_turbines=200,
)


actor_critic_config = GNNActorCriticConfig(
    model_type="gnn",
    policy_node_hidden_size=16,
    policy_gnn_depth=2,
    policy_head_depth=2,
    critic_node_hidden_size=16,
    critic_gnn_depth=2,
)


@pytest.fixture
def designer():
    return FixedDesigner(
        designer_setting=DesignerParams.placeholder(scenario=test_scenario),
        seed=seed,
    ).get_placeholder()


@pytest.fixture
def env(designer):
    env = create_env(
        mode="reference",
        scenario=test_scenario,
        designer=designer,
        device=device,
    )
    yield env
    env.close()


@pytest.fixture
def actor_critic(env):
    # Create actor critic
    actor, critic = wfcrl_models_gnn(
        env=env,
        cfg=actor_critic_config,
        normalisation=None,
        device=device,
    )

    return actor, critic


def test_batched_execution(actor_critic, designer):
    actor, critic = actor_critic

    env = create_batched_env(
        create_env=create_env,
        mode="reference",
        num_environments=3,
        designer=designer,
        scenario=test_scenario,
        batch_mode="serial",
        device=device,
    )

    td = env.reset()
    critic(td)
    actor(td)

    assert_expected_inline(
        str(td["turbine"]),
        """\
TensorDict(
    fields={
        action: TensorDict(
            fields={
                yaw: Tensor(shape=torch.Size([3, 7, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([3, 7]),
            device=cpu,
            is_shared=False),
        done: Tensor(shape=torch.Size([3, 7, 1]), device=cpu, dtype=torch.bool, is_shared=False),
        episode_reward: Tensor(shape=torch.Size([3, 7, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        loc: Tensor(shape=torch.Size([3, 7, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        observation: TensorDict(
            fields={
                layout: Tensor(shape=torch.Size([3, 7, 2]), device=cpu, dtype=torch.float32, is_shared=False),
                wind_direction: Tensor(shape=torch.Size([3, 7, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                wind_speed: Tensor(shape=torch.Size([3, 7, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                yaw: Tensor(shape=torch.Size([3, 7, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([3, 7]),
            device=cpu,
            is_shared=False),
        sample_log_prob: Tensor(shape=torch.Size([3, 7]), device=cpu, dtype=torch.float32, is_shared=False),
        scale: Tensor(shape=torch.Size([3, 7, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        state_value: Tensor(shape=torch.Size([3, 7, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        terminated: Tensor(shape=torch.Size([3, 7, 1]), device=cpu, dtype=torch.bool, is_shared=False),
        truncated: Tensor(shape=torch.Size([3, 7, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
    batch_size=torch.Size([3, 7]),
    device=cpu,
    is_shared=False)""",
    )

    env.close()


def test_gnn_invariance(actor_critic, env):
    _, critic = actor_critic

    td = env.reset()
    layout = td["turbine", "observation", "layout"].clone()
    wind_direction = td["turbine", "observation", "wind_direction"].clone()  # Degrees

    initial_critic_out = critic(td)["turbine", "state_value"]

    # Rotate the layout
    theta = torch.rand(1) * 2 * torch.pi
    R = torch.tensor(
        [[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]
    )
    rotated_layout = layout @ R.T
    wind_direction = (torch.rad2deg(torch.deg2rad(wind_direction) - theta) + 360) % 360

    td["turbine", "observation", "layout"] = rotated_layout
    td["turbine", "observation", "wind_direction"] = wind_direction

    new_critic_out = critic(td)["turbine", "state_value"]

    assert torch.allclose(initial_critic_out[0], new_critic_out[0], atol=1e-2)
