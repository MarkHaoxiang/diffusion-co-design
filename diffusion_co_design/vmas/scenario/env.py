import torch
from torchrl.envs import TransformedEnv, RewardSum, StepCounter

from diffusion_co_design.common.design.base import DesignConsumer
from diffusion_co_design.common.env import ENVIRONMENT_MODE
from diffusion_co_design.vmas.schema import ScenarioConfig
from diffusion_co_design.vmas.diffusion.generate import Generate
from .obstacle_navigation import (
    Scenario as ObstacleNavigationScenario,
    DesignableVmasEnv,
)


def create_env(
    mode: ENVIRONMENT_MODE,
    scenario: ScenarioConfig,
    designer: DesignConsumer,
    num_environments: int = 1,
    device: torch.device = torch.device("cpu"),
):
    agent_spawns = torch.tensor(scenario.agent_spawns, device=device)
    agent_goals = torch.tensor(scenario.agent_goals, device=device)
    obstacle_sizes = torch.tensor(scenario.obstacle_sizes, device=device)

    # Use this to generate the initial environment
    # Because the VMAS generation logic is poor with many objects
    generate = Generate(scenario=scenario)
    obstacle_positions = torch.tensor(generate(), device=device).expand(
        (num_environments, scenario.n_obstacles, 2)
    )

    env = DesignableVmasEnv(
        scenario=ObstacleNavigationScenario(),
        scenario_cfg=scenario,
        reset_policy=designer,
        num_envs=num_environments,
        device=device,
        continuous_actions=True,
        max_steps=None,
        # Scenario kwargs
        shared_rew=False,
        world_spawning_x=scenario.world_spawning_x,
        world_spawning_y=scenario.world_spawning_y,
        agent_spawns=agent_spawns,
        agent_goals=agent_goals,
        obstacle_sizes=obstacle_sizes,
        obstacle_positions=obstacle_positions,
    )

    env = TransformedEnv(
        env,
        StepCounter(max_steps=scenario.get_episode_steps(), update_done=True),
    )

    if mode == "train":
        env = TransformedEnv(
            env,
            RewardSum(
                in_keys=env.reward_keys,
                out_keys=[(agent, "episode_reward") for (agent, _) in env.reward_keys],
            ),
        )

    return env
