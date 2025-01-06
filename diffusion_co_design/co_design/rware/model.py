from typing import Sequence, Type

import torch
from torch.distributions import Categorical
from torchrl.data import Composite, Unbounded
from tensordict.nn import TensorDictModule, InteractionType
from torchrl.envs import TransformedEnv, Compose, InitTracker, TensorDictPrimer
from torchrl.data.utils import DEVICE_TYPING
from torchrl.modules import ProbabilisticActor, MultiAgentMLP
from benchmarl.models.lstm import Lstm

from pydantic import BaseModel

from .env import RwareCoDesignWrapper


class PolicyConfig(BaseModel):
    hidden_size: int = 128
    n_layers: int = 1
    dropout: float = 0
    bias: bool = True
    share_params: bool = True


def rware_models(
    env: RwareCoDesignWrapper, cfg: PolicyConfig, device: DEVICE_TYPING | None
):

    # Construct Policy
    observation_spec = env.observation_spec.clone()
    group_obs_spec = observation_spec["agents"]
    for key in list(group_obs_spec.keys()):
        if key != "observation":
            del group_obs_spec[key]
    if "state" in observation_spec.keys():
        del observation_spec["state"]

    logits_shape = [
        *env.full_action_spec["agents", "action"].shape,
        env.full_action_spec["agents", "action"].space.n,
    ]

    actor_input_spec = Composite(
        {"agents": observation_spec["agents"].clone().to(device)}
    )
    actor_output_spec = Composite(
        {
            "agents": Composite(
                {"logits": Unbounded(shape=logits_shape)},
                shape=(env.num_agents,),
            )
        }
    )

    policy_net = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[
                -1
            ],  # n_obs_per_agent
            n_agent_outputs=env.action_spec.space.n,  # n_actions_per_agents
            n_agents=env.num_agents,
            centralised=False,  # the policies are decentralised (ie each agent will act from its observation)
            share_params=True,
            device=device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        ),
    )
    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "logits")],
    )

    # policy_module = Lstm(
    #     hidden_size=cfg.hidden_size,
    #     n_layers=cfg.n_layers,
    #     bias=cfg.bias,
    #     dropout=cfg.dropout,
    #     compile=False,
    #     share_params=cfg.share_params,
    #     device=device,
    #     is_critic=False,
    #     centralised=False,
    #     n_agents=env.num_agents,
    #     input_spec=actor_input_spec,
    #     output_spec=actor_output_spec,
    #     action_spec=env.full_action_spec,
    #     input_has_agent_dim=True,
    #     agent_group="agents",
    #     model_index=0,
    # )

    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=[("agents", "logits")],
        out_keys=[env.action_key],
        distribution_class=Categorical,
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
        default_interaction_type=InteractionType.RANDOM,
    )

    # rnn_spec = Composite(
    #     {
    #         f"_hidden_lstm_c_0": Unbounded(shape=(cfg.n_layers, cfg.hidden_size)),
    #         f"_hidden_lstm_h_0": Unbounded(shape=(cfg.n_layers, cfg.hidden_size)),
    #     }
    # )
    # rnn_spec = Composite(
    #     {
    #         group: Composite(
    #             rnn_spec.expand(len(agents), *rnn_spec.shape),
    #             shape=(len(agents),),
    #         )
    #         for group, agents in env.group_map.items()
    #     }
    # )

    # env = TransformedEnv(
    #     env, Compose(InitTracker(), TensorDictPrimer(rnn_spec, reset_key="_reset"))
    # )

    # TODO: We want to use a critic that directly accesses simulator state
    critic_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=1,  # 1 value per agent
        n_agents=env.num_agents,
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
