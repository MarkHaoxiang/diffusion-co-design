from typing import Sequence, Type

import torch
from torch import nn
from torch.distributions import Categorical
from torchrl.data import Composite, Unbounded
from tensordict.nn import TensorDictModule, InteractionType, TensorDictSequential
from torchrl.envs import TransformedEnv, Compose, InitTracker, TensorDictPrimer
from torchrl.data.utils import DEVICE_TYPING
from torchrl.modules import ProbabilisticActor, MultiAgentMLP, MultiAgentConvNet
from benchmarl.models.lstm import Lstm

from guided_diffusion.script_util import create_classifier, classifier_defaults

from pydantic import BaseModel

from .env import RwareCoDesignWrapper


class PolicyConfig(BaseModel):
    # CNN
    kernel_sizes: int | Sequence[int] = 3
    num_cells: int | Sequence[int] = [16, 32, 64]
    strides: Sequence[int] | int = 1
    # MLP
    hidden_size: int = 128
    depth: int = 2
    # Both
    share_params: bool = True


class Critic(nn.Module):
    # TODO
    def __init__(self, env: RwareCoDesignWrapper):
        super().__init__()
        critic_net_dict = classifier_defaults()  # This creates an internal unet

        critic_net_dict["image_size"] = env.layout.grid_size
        critic_net_dict["classifier_attention_resolutions"] = "16, 8, 4"
        critic_net_dict["output_dim"] = 128

        self.unet = create_classifier(**critic_net_dict)

    def forward(image, features):
        pass


def _rware_models_2(
    env: RwareCoDesignWrapper, cfg: PolicyConfig, device: DEVICE_TYPING | None
):
    # Policy
    policy_cnn_net_1 = MultiAgentConvNet(
        n_agents=env.num_agents,
        centralized=False,
        share_params=cfg.share_params,
        kernel_sizes=cfg.kernel_sizes,
        num_cells=cfg.num_cells,
        strides=cfg.strides,
        device=device,
    )
    policy_cnn_module_1 = TensorDictModule(
        module=policy_cnn_net_1,
        in_keys=[("agents", "observation", "global_image")],
        out_keys=[("agents", "observation", "global_image_features")],
    )
    policy_cnn_net_2 = MultiAgentConvNet(
        n_agents=env.num_agents,
        centralized=False,
        share_params=cfg.share_params,
        kernel_sizes=2,
        num_cells=cfg.num_cells,
        strides=cfg.strides,
        device=device,
    )
    policy_cnn_module_2 = TensorDictModule(
        module=policy_cnn_net_2,
        in_keys=[("agents", "observation", "local_image")],
        out_keys=[("agents", "observation", "local_image_features")],
    )
    policy_mlp_net = MultiAgentMLP(
        n_agent_inputs=None,  # Lazy instantiation
        n_agent_outputs=env.action_spec.space.n,  # n_actions_per_agents
        n_agents=env.num_agents,
        depth=cfg.depth,
        num_cells=cfg.hidden_size,
        activation_class=torch.nn.Tanh,
        centralised=False,
        share_params=cfg.share_params,
        device=device,
    )
    policy_mlp_module = TensorDictModule(
        module=policy_mlp_net,
        in_keys=[
            ("agents", "observation", "features"),
            ("agents", "observation", "global_image_features"),
            ("agents", "observation", "local_image_features"),
        ],
        out_keys=[("agents", "logits")],
    )
    policy_module = TensorDictSequential(
        policy_cnn_module_1, policy_cnn_module_2, policy_mlp_module
    )
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

    # Critic
    # TODO: A unet may be needed for diffusion
    # TODO: We need a custom module to collect the shared observations
    critic_cnn_net_1 = MultiAgentConvNet(
        n_agents=env.num_agents,
        centralized=False,
        share_params=cfg.share_params,
        kernel_sizes=cfg.kernel_sizes,
        num_cells=cfg.num_cells,
        strides=cfg.strides,
        device=device,
    )
    critic_cnn_module_1 = TensorDictModule(
        module=critic_cnn_net_1,
        in_keys=[("agents", "observation", "global_image")],
        out_keys=[("agents", "observation", "critic_global_image_features")],
    )
    critic_cnn_net_2 = MultiAgentConvNet(
        n_agents=env.num_agents,
        centralized=False,
        share_params=cfg.share_params,
        kernel_sizes=2,
        num_cells=cfg.num_cells,
        strides=cfg.strides,
        device=device,
    )
    critic_cnn_module_2 = TensorDictModule(
        module=critic_cnn_net_2,
        in_keys=[("agents", "observation", "local_image")],
        out_keys=[("agents", "observation", "critic_local_image_features")],
    )
    critic_mlp_net = MultiAgentMLP(
        n_agent_inputs=None,
        n_agent_outputs=1,
        n_agents=env.num_agents,
        depth=cfg.depth,
        num_cells=cfg.hidden_size,
        activation_class=torch.nn.Tanh,
        centralised=False,
        share_params=cfg.share_params,
        device=device,
    )
    critic_mlp_module = TensorDictModule(
        module=critic_mlp_net,
        in_keys=[
            ("agents", "observation", "features"),
            ("agents", "observation", "critic_local_image_features"),
            ("agents", "observation", "critic_global_image_features"),
        ],
        out_keys=[("agents", "state_value")],
    )
    critic = TensorDictSequential(
        critic_cnn_module_1,
        critic_cnn_module_2,
        critic_mlp_module,
        selected_out_keys=[("agents", "state_value")],
    )

    # Initialise
    td = env.reset()
    with torch.no_grad():
        policy(td)
        critic(td)

    return policy, critic


def _rware_models(
    env: RwareCoDesignWrapper, cfg: PolicyConfig, device: DEVICE_TYPING | None
):
    # Policy
    policy_cnn_net = MultiAgentConvNet(
        n_agents=env.num_agents,
        centralized=False,
        share_params=cfg.share_params,
        kernel_sizes=cfg.kernel_sizes,
        num_cells=cfg.num_cells,
        strides=cfg.strides,
        device=device,
    )
    policy_cnn_module = TensorDictModule(
        module=policy_cnn_net,
        in_keys=[("agents", "observation", "image")],
        out_keys=[("agents", "observation", "image_features")],
    )
    policy_mlp_net = MultiAgentMLP(
        n_agent_inputs=None,  # Lazy instantiation
        n_agent_outputs=env.action_spec.space.n,  # n_actions_per_agents
        n_agents=env.num_agents,
        depth=cfg.depth,
        num_cells=cfg.hidden_size,
        activation_class=torch.nn.Tanh,
        centralised=False,
        share_params=cfg.share_params,
        device=device,
    )
    policy_mlp_module = TensorDictModule(
        module=policy_mlp_net,
        in_keys=[
            ("agents", "observation", "features"),
            ("agents", "observation", "image_features"),
        ],
        out_keys=[("agents", "logits")],
    )
    policy_module = TensorDictSequential(policy_cnn_module, policy_mlp_module)
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

    # Critic
    # TODO: A unet may be needed for diffusion
    # TODO: We need a custom module to collect the shared observations
    critic_cnn_net = MultiAgentConvNet(
        n_agents=env.num_agents,
        centralized=False,
        share_params=cfg.share_params,
        kernel_sizes=cfg.kernel_sizes,
        num_cells=cfg.num_cells,
        strides=cfg.strides,
        device=device,
    )
    critic_cnn_module = TensorDictModule(
        module=critic_cnn_net,
        in_keys=[("agents", "observation", "image")],
        out_keys=[("agents", "observation", "critic_image_features")],
    )
    critic_mlp_net = MultiAgentMLP(
        n_agent_inputs=None,
        n_agent_outputs=1,
        n_agents=env.num_agents,
        depth=cfg.depth,
        num_cells=cfg.hidden_size,
        activation_class=torch.nn.Tanh,
        centralised=False,
        share_params=cfg.share_params,
        device=device,
    )
    critic_mlp_module = TensorDictModule(
        module=critic_mlp_net,
        in_keys=[
            ("agents", "observation", "features"),
            ("agents", "observation", "critic_image_features"),
        ],
        out_keys=[("agents", "state_value")],
    )
    critic = TensorDictSequential(
        critic_cnn_module,
        critic_mlp_module,
        selected_out_keys=[("agents", "state_value")],
    )

    # Initialise
    td = env.reset()
    with torch.no_grad():
        policy(td)
        critic(td)

    return policy, critic


def _rware_models_flattened(
    env: RwareCoDesignWrapper, cfg: PolicyConfig, device: DEVICE_TYPING | None
):
    # Deprecated, for the flattened observation space

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
            depth=cfg.depth,
            num_cells=cfg.hidden_size,
            activation_class=torch.nn.Tanh,
        ),
    )
    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "logits")],
    )

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

    # TODO: We want to use a critic that directly accesses simulator state
    critic_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=1,  # 1 value per agent
        n_agents=env.num_agents,
        centralised=True,
        share_params=True,
        device=device,
        depth=cfg.depth,
        num_cells=cfg.hidden_size,
        activation_class=torch.nn.Tanh,
    )

    critic = TensorDictModule(
        module=critic_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "state_value")],
    )

    # return policy, critic, env
    return policy, critic


# class PolicyConfig(BaseModel):
#     hidden_size: int = 128
#     n_layers: int = 1
#     dropout: float = 0
#     bias: bool = True
#     share_params: bool = True


# Construct Policy
# observation_spec = env.observation_spec.clone()
# group_obs_spec = observation_spec["agents"]
# for key in list(group_obs_spec.keys()):
#     if key != "observation":
#         del group_obs_spec[key]
# if "state" in observation_spec.keys():
#     del observation_spec["state"]

# logits_shape = [
#     *env.full_action_spec["agents", "action"].shape,
#     env.full_action_spec["agents", "action"].space.n,
# ]

# actor_input_spec = Composite(
#     {"agents": observation_spec["agents"].clone().to(device)}
# )
# actor_output_spec = Composite(
#     {
#         "agents": Composite(
#             {"logits": Unbounded(shape=logits_shape)},
#             shape=(env.num_agents,),
#         )
#     }
# )

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

rware_models = _rware_models_2
# rware_models = _rware_models_flattened
