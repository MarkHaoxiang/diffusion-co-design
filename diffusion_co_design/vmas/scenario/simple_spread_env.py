import torch
from torchrl.envs import TransformedEnv, RewardSum, StepCounter

from diffusion_co_design.common.design.base import DesignConsumer
from diffusion_co_design.common.env import ENVIRONMENT_MODE
from diffusion_co_design.vmas.schema import SimpleSpreadScenarioConfig
from .simple_spread import Scenario, DesignableVmasEnv


def create_env(
    mode: ENVIRONMENT_MODE,
    scenario: SimpleSpreadScenarioConfig,
    designer: DesignConsumer,
    num_environments: int = 1,
    device: torch.device = torch.device("cpu"),
):
    env = DesignableVmasEnv(
        scenario=Scenario(),
        scenario_cfg=scenario,
        reset_policy=designer,
        num_envs=num_environments,
        device=device,
        continuous_actions=True,
        max_steps=None,
        # Scenario kwargs
        n_agents=scenario.get_num_agents(),
        shared_rew=False,
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
