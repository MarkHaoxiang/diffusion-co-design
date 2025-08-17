import torch
from tensordict.nn import TensorDictModule, InteractionType
from torchrl.data.utils import DEVICE_TYPING
from torchrl.modules import ProbabilisticActor, MultiAgentMLP
from torchrl.modules import NormalParamExtractor, TanhNormal

from diffusion_co_design.vmas.schema import (
    ActorConfig,
    CriticConfig,
    ActorCriticConfig,
    ScenarioConfig,
)
from diffusion_co_design.vmas.scenario.obstacle_navigation import DesignableVmasEnv
from diffusion_co_design.vmas.static import GROUP_NAME
from diffusion_co_design.vmas.model.shared import E3Critic


def create_policy(env: DesignableVmasEnv, cfg: ActorConfig, device: DEVICE_TYPING):
    policy_net = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=env.observation_spec[GROUP_NAME, "observation"].shape[
                -1
            ],  # n_obs_per_agent
            n_agent_outputs=2 * env.action_spec[env.action_key].shape[-1],
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


class E3RLCritic(E3Critic):
    def forward(self, obs, state):
        obstacle_pos = state
        agent_pos = obs[..., :2]
        goal_pos = obs[..., 4:6]
        agent_vel = obs[..., 2:4]

        return super().forward(obstacle_pos, agent_pos, goal_pos, agent_vel)


def create_critic(
    env: DesignableVmasEnv,
    scenario: ScenarioConfig,
    cfg: CriticConfig,
    device: DEVICE_TYPING,
):
    critic_net = E3Critic(
        scenario=scenario,
        node_emb_dim=cfg.hidden_size,
        num_layers=cfg.depth,
    ).to(device=device)

    critic = TensorDictModule(
        critic_net,
        in_keys=[(GROUP_NAME, "observation"), "state"],
        out_keys=[(GROUP_NAME, "state_value")],
    )
    return critic


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
