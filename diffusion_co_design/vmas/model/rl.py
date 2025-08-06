import torch
from tensordict.nn import TensorDictModule, InteractionType
from torchrl.data.utils import DEVICE_TYPING
from torchrl.modules import ProbabilisticActor, MultiAgentMLP
from torchrl.modules import NormalParamExtractor, TanhNormal

from e3nn.o3 import Irreps
from segnn.balanced_irreps import BalancedIrreps
from segnn.segnn import SEGNN

from diffusion_co_design.vmas.schema import (
    ActorConfig,
    CriticConfig,
    ActorCriticConfig,
    ScenarioConfig,
)
from diffusion_co_design.vmas.scenario.obstacle_navigation import DesignableVmasEnv
from diffusion_co_design.vmas.static import GROUP_NAME


def create_policy(env: DesignableVmasEnv, cfg: ActorConfig, device: DEVICE_TYPING):
    policy_net = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=env.observation_spec[GROUP_NAME, "observation"].shape[
                -1
            ],  # n_obs_per_agent
            n_agent_outputs=2 * env.action_spec.shape[-1],
            n_agents=env.n_agents,
            centralised=False,
            share_params=True,
            device=device,
            depth=cfg.depth,
            num_cells=cfg.hidden_size,
            activation_class=torch.nn.Tanh,
        ),
        NormalParamExtractor(),
    )

    policy_module = TensorDictModule(
        policy_net,
        in_keys=[(GROUP_NAME, "observation")],
        out_keys=[(GROUP_NAME, "loc"), (GROUP_NAME, "scale")],
    )

    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.unbatched_action_spec,
        in_keys=[(GROUP_NAME, "loc"), (GROUP_NAME, "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.unbatched_action_spec[env.action_key].space.low,
            "high": env.unbatched_action_spec[env.action_key].space.high,
        },
        return_log_prob=True,
        log_prob_key=(GROUP_NAME, "sample_log_prob"),
        default_interaction_type=InteractionType.RANDOM,
    )

    return policy


class E3Critic(torch.nn.Module):
    def __init__(
        self,
        scenario: ScenarioConfig,
        hidden_features: int = 128,
        hidden_lmax: int = 2,
        num_layers: int = 3,
    ):
        super().__init__()
        # One-hot node type (agent, obstacle, goal)
        # Velocity
        # Relative position to center of map
        # Norm of velocity
        # Radius of entity
        irreps_node_input = Irreps("3x0e + 1o + 1o + 0e + 0e").simplify()

        # Absolute center distance, absolute collision distance
        additional_message_irreps = Irreps("2x0e")

        edge_attr_irreps = Irreps.spherical_harmonics(hidden_lmax)
        node_attr_irreps = Irreps.spherical_harmonics(hidden_lmax)

        hidden_irreps = BalancedIrreps(hidden_lmax, hidden_features, True)
        self.model = SEGNN(
            input_irreps=irreps_node_input,
            hidden_irreps=hidden_irreps,
            output_irreps=Irreps("1x0e"),
            edge_attr_irreps=edge_attr_irreps,
            node_attr_irreps=node_attr_irreps,
            num_layers=num_layers,
            norm=None,
            pool="avg",
            task="graph",
            additional_message_irreps=additional_message_irreps,
        )

    def forward(self, x):
        return self.model(x)


def create_critic(
    env: DesignableVmasEnv,
    scenario: ScenarioConfig,
    cfg: CriticConfig,
    device: DEVICE_TYPING,
):
    raise NotImplementedError()


def vmas_models(
    env: DesignableVmasEnv,
    scenario: ScenarioConfig,
    actor_critic_cfg: ActorCriticConfig,
    device: DEVICE_TYPING = torch.device("cpu"),
):
    policy = create_policy(env, actor_critic_cfg.actor, device)
    critic = create_critic(env, scenario, actor_critic_cfg.critic, device)

    td = env.reset()
    with torch.no_grad():
        policy(td)
        critic(td)

    return policy, critic
