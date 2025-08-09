from math import prod

import torch
from tensordict.nn import TensorDictModule, InteractionType
from torchrl.data.utils import DEVICE_TYPING
from torchrl.modules import ProbabilisticActor, MultiAgentMLP
from torchrl.modules import NormalParamExtractor, TanhNormal
from torch_geometric.data import Data, Batch
from torch_geometric.nn import knn_graph

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
        node_emb_dim: int = 128,
        out_dim: int = 128,
        num_layers: int = 3,
        k: int = 5,
    ):
        super().__init__()
        self.scenario = scenario
        self.n_obstacles = len(scenario.obstacle_sizes)
        self.k = k

        # Observations
        # agent_pos (2) vel (2) goal_pos (2) lidar (12)

        # State
        # obstacle_pos (n_obstacles, 2)

    def forward(self, obs, state):
        B_all = obs.shape[:-2]
        B, N = prod(B_all), self.scenario.get_num_agents()

        # Shape checks
        obs = obs.reshape(-1, N, obs.shape[-1])
        assert obs.shape == (B, N, 18), obs.shape
        state = state.reshape(-1, N, state.shape[-1])
        assert state.shape == (B, self.n_obstacles, 2), state.shape

        # Construct graph
        data_list: list[Data] = []
        for i in range(B):
            obstacle_pos = state[i]
            agent_pos = obs[i, :, :2]
            goal_pos = obs[i, :, 4:6]

            # Note: discard lidar for now, position information should be sufficient
            agent_vel = obs[i, :, 2:4]

    def construct_graph(self, obstacle_pos, agent_pos, goal_pos, agent_vel):
        N = self.scenario.get_num_agents() * 2 + self.n_obstacles

        # Graph topology
        pos = torch.cat([agent_pos, goal_pos, obstacle_pos], dim=-1)
        assert pos.shape == (N, 2), pos.shape
        edge_index = knn_graph(pos, k=self.k, loop=False)
        # add direct edges between agents and their goals
        agent_goal_edges = torch.arange(self.n_obstacles, device=pos.device)
        agent_goal_edges = torch.stack(
            [agent_goal_edges, agent_goal_edges + self.scenario.get_num_agents()],
            dim=0,
        )
        edge_index = torch.cat([edge_index, agent_goal_edges], dim=-1)
        # remove duplicate edges
        edge_index = torch.unique(edge_index, dim=-1)

        # Node features:
        # One-hot encoding of node types
        # Velocity norm
        x = torch.zeros((N, 3 + 1), device=pos.device)
        x[: self.n_obstacles, 0] = 1.0
        x[self.n_obstacles : self.n_obstacles + self.scenario.get_num_agents(), 1] = 1.0
        x[self.n_obstacles + self.scenario.get_num_agents() :, 2] = 1.0


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
