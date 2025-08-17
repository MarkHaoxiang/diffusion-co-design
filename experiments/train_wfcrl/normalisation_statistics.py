import os
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf
import torch
from tensordict.nn import TensorDictModule

from diffusion_co_design.wfcrl.schema import (
    ScenarioConfig,
    NormalisationStatistics,
)
from diffusion_co_design.wfcrl.static import GROUP_NAME
from diffusion_co_design.wfcrl.design import RandomDesigner
from diffusion_co_design.wfcrl.env import create_env
from diffusion_co_design.common.design import DesignerParams, get_training_pair_from_td
from diffusion_co_design.common import OUTPUT_DIR


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: [*B, N, K]
        return torch.zeros((*x.shape[:-1], 1), device=x.device, dtype=x.dtype)


def main(scenario_name: str, n_episodes: int, gamma: float = 0.99):
    scenario = ScenarioConfig.from_file(
        path=os.path.join(OUTPUT_DIR, "wfcrl", "scenario", scenario_name, "config.yaml")
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = create_env(
        mode="train",
        scenario=scenario,
        designer=RandomDesigner(DesignerParams.placeholder(scenario)).get_placeholder(),
        device=device,
        render=False,
    )

    policy = TensorDictModule(
        Policy(),
        in_keys=[(GROUP_NAME, "observation", "layout")],
        out_keys=[env.action_key],
    )

    _episode_returns = []
    _rewards = []
    for i in tqdm(range(n_episodes)):
        td = env.rollout(
            policy=policy,
            max_steps=scenario.max_steps,
        )
        _, y = get_training_pair_from_td(
            td=td,
            group_name=GROUP_NAME,
            group_aggregation="mean",
            gamma=gamma,
            get_layout_from_state=lambda x: x["layout"],
            episode_steps=scenario.get_episode_steps(),
        )

        assert y.shape == (1,)
        _episode_returns.append(y)
        _rewards.append(td[("next", GROUP_NAME, "reward")])

    episode_returns = torch.stack(_episode_returns)
    rewards = torch.stack(_rewards)

    out = NormalisationStatistics(
        episode_return_mean=episode_returns.mean().item(),
        episode_return_std=episode_returns.std().item(),
        reward_mean=rewards.mean().item(),
        reward_std=rewards.std().item(),
    )
    out_path = os.path.join(
        OUTPUT_DIR, "wfcrl", "scenario", scenario_name, "normalisation_statistics.yaml"
    )

    OmegaConf.save(
        config=OmegaConf.structured(out.model_dump()),
        f=out_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
    )
    args = parser.parse_args()

    main(args.scenario, args.n_episodes, gamma=args.gamma)
