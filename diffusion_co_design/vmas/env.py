from torchrl.envs import VmasEnv, TransformedEnv, RewardSum
from vmas.simulator.scenario import BaseScenario

from pydantic import BaseModel


class DesignableScenario(BaseScenario):
    pass


class ScenarioConfig(BaseModel):
    max_steps: int = 100
    scenario_name: str = "navigation"
    n_agents: int = 3


def vmas_env(cfg: ScenarioConfig, device):
    env = VmasEnv(
        scenario=cfg.scenario_name,
        num_envs=1,
        continuous_actions=True,
        max_steps=cfg.max_steps,
        device=device,
        n_agents=3,
    )
    env = TransformedEnv(
        env,
        RewardSum(
            in_keys=env.reward_keys,
            out_keys=[(agent, "episode_reward") for (agent, _) in env.reward_keys],
        ),
    )

    return env
