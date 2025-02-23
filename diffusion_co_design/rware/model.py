from typing import Sequence

import torch
from torch import nn
from torch.distributions import Categorical

# from torchrl.data import Composite, Unbounded
from tensordict.nn import TensorDictModule, InteractionType, TensorDictSequential

# from torchrl.envs import TransformedEnv, Compose, InitTracker, TensorDictPrimer
from torchrl.data.utils import DEVICE_TYPING
from torchrl.modules import ProbabilisticActor, MultiAgentMLP, MultiAgentConvNet
from torchrl.envs import DTypeCastTransform
# from benchmarl.models.lstm import Lstm

# from guided_diffusion.script_util import create_classifier, classifier_defaults

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


def rware_models(
    env: RwareCoDesignWrapper, cfg: PolicyConfig, device: DEVICE_TYPING | None
):
    dtype_cast = TensorDictModule(
        DTypeCastTransform(
            dtype_in=torch.int,
            dtype_out=torch.float32,
            in_keys=[
                ("agents", "observation", "global_image"),
                ("agents", "observation", "local_image"),
            ],
            out_keys=[
                ("agents", "observation", "global_image_float"),
                ("agents", "observation", "local_image_float"),
            ],
        ),
        in_keys=[
            ("agents", "observation", "global_image"),
            ("agents", "observation", "local_image"),
        ],
        out_keys=[
            ("agents", "observation", "global_image_float"),
            ("agents", "observation", "local_image_float"),
        ],
    )

    # Policy
    policy_cnn_net = MultiAgentConvNet(
        n_agents=env.num_agents,
        centralized=False,
        share_params=cfg.share_params,
        kernel_sizes=2,
        num_cells=cfg.num_cells,
        strides=cfg.strides,
        device=device,
    )
    policy_cnn_module = TensorDictModule(
        module=policy_cnn_net,
        in_keys=[("agents", "observation", "local_image_float")],
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
            ("agents", "observation", "local_image_features"),
        ],
        out_keys=[("agents", "logits")],
    )
    policy_module = TensorDictSequential(
        dtype_cast,
        policy_cnn_module,
        policy_mlp_module,
        selected_out_keys=[("agents", "logits")],
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
    critic_cnn_net = MultiAgentConvNet(
        n_agents=env.num_agents,
        centralized=False,
        share_params=cfg.share_params,
        kernel_sizes=2,
        num_cells=cfg.num_cells,
        strides=cfg.strides,
        device=device,
    )
    critic_cnn_module = TensorDictModule(
        module=critic_cnn_net,
        in_keys=[("agents", "observation", "local_image_float")],
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
        ],
        out_keys=[("agents", "state_value")],
    )
    critic = TensorDictSequential(
        dtype_cast,
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
