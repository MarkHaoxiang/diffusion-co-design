from typing import Sequence, Type

import torch
from torch.distributions import Categorical
from tensordict.nn import TensorDictModule, InteractionType
from torchrl.envs import VmasEnv
from torchrl.data.utils import DEVICE_TYPING
from torchrl.modules import ProbabilisticActor, MultiAgentMLP
from torchrl.modules import NormalParamExtractor, TanhNormal

from pydantic import BaseModel


class PolicyConfig(BaseModel):
    hidden_size: int = 128
    n_layers: int = 1
    dropout: float = 0
    bias: bool = True
    share_params: bool = True


def rware_models(env: VmasEnv, cfg: PolicyConfig, device: DEVICE_TYPING | None):

    policy_net = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[
                -1
            ],  # n_obs_per_agent
            n_agent_outputs=2 * env.action_spec.shape[-1],
            n_agents=env.n_agents,
            centralised=False,
            share_params=cfg.share_params,
            device=device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        ),
        NormalParamExtractor(),
    )

    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )

    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.unbatched_action_spec,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.unbatched_action_spec[env.action_key].space.low,
            "high": env.unbatched_action_spec[env.action_key].space.high,
        },
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
        default_interaction_type=InteractionType.RANDOM,
    )

    critic_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=1,
        n_agents=env.n_agents,
        centralised=True,
        share_params=True,
        device=device,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
    )

    critic = TensorDictModule(
        module=critic_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "state_value")],
    )

    return policy, critic, env
