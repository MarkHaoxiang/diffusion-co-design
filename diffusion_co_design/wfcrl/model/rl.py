from math import prod

import torch
import torch.nn as nn

from tensordict.nn import TensorDictModule, TensorDictSequential, InteractionType
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torch_geometric.data import Data, Batch
from torch_geometric.nn import knn_graph
from torch_geometric.nn.models import GAT

from diffusion_co_design.wfcrl.schema import (
    ScenarioConfig,
    ActorCriticConfig,
    NormalisationStatistics,
)


class OutputDenormaliser(nn.Module):
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x: torch.Tensor):
        return x * self.std + self.mean


class ID(nn.Module):
    def forward(self, x: torch.Tensor):
        return x


class WindFarmGNN(nn.Module):
    def __init__(
        self,
        scenario: ScenarioConfig,
        node_emb_dim: int = 64,
        out_dim: int = 64,
        n_layers: int = 3,
        wind_speed_low: float | torch.Tensor = 0,
        wind_speed_high: float | torch.Tensor = 28,
        k: int = 5,
    ):
        super().__init__()
        self.scenario = scenario
        self.wind_speed_low = wind_speed_low
        self.wind_speed_high = wind_speed_high
        self.k = k
        self.out_dim = out_dim

        self.model = GAT(
            in_channels=-1,
            edge_dim=5,
            hidden_channels=node_emb_dim,
            num_layers=n_layers,
            out_channels=out_dim,
            act="relu",
            add_self_loops=False,
            v2=True,
        )

    def forward(
        self,
        wind_direction: torch.Tensor,  # [*B, N, 1]
        wind_speed: torch.Tensor,  # [*B, N, 1]
        yaw: torch.Tensor,  # [*B, N, 1]
        layout: torch.Tensor,  # [*B, N, 2]
    ):
        has_batch = len(wind_direction.shape) > 2
        if not has_batch:
            wind_direction = wind_direction.unsqueeze(0)
            wind_speed = wind_speed.unsqueeze(0)
            yaw = yaw.unsqueeze(0)
            layout = layout.unsqueeze(0)

        B_all = wind_direction.shape[:-2]
        B, N = prod(B_all), self.scenario.get_num_agents()

        # Shape checks
        wind_direction = wind_direction.reshape(-1, N, 1)
        assert wind_direction.shape == (B, N, 1), wind_direction.shape
        wind_speed = wind_speed.reshape(-1, N, 1)
        assert wind_speed.shape == (B, N, 1), wind_speed.shape
        yaw = yaw.reshape(-1, N, 1)
        assert yaw.shape == (B, N, 1), yaw.shape
        layout = layout.reshape(-1, N, 2)
        assert layout.shape == (B, N, 2), layout.shape

        # Normalisation and feature engineering
        wind_direction = torch.deg2rad(wind_direction)
        wind_speed = (wind_speed - self.wind_speed_low) / (
            self.wind_speed_high - self.wind_speed_low
        )
        wind_x = wind_speed * torch.cos(wind_direction)
        wind_y = wind_speed * torch.sin(wind_direction)
        wind = torch.cat([wind_x, wind_y], dim=-1)  # [B, N, 2]
        yaw = torch.deg2rad(yaw)
        layout = layout.clone()
        layout[:, :, 0] = (layout[:, :, 0] / self.scenario.map_x_length) * 2 - 1
        layout[:, :, 1] = (layout[:, :, 1] / self.scenario.map_y_length) * 2 - 1

        # Construct graph
        data_list: list[Data] = []

        for i in range(B):
            # Build KNN graph
            pos = layout[i]  # [N, 2]
            edge_index = knn_graph(pos, k=self.k, loop=True)  # [2, E]

            # Node features
            x = torch.cat([wind_speed[i], yaw[i]], dim=-1)

            # Edge features
            src, dst = edge_index
            pos_diff = pos[dst] - pos[src]  # [E, 2]
            radial = torch.norm(pos_diff, dim=-1, keepdim=True)  # [E, 1]
            wind_src = wind[i][src]  # [E, 2]

            # Geometric interpretation of wind
            wind_dot_src = torch.sum(wind_src * pos_diff, dim=-1, keepdim=True)
            wind_cross_src = (
                wind_src[:, 0:1] * pos_diff[:, 1:2]
                - wind_src[:, 1:2] * pos_diff[:, 0:1]
            )
            wind_dst = wind[i][dst]
            wind_dot_dst = torch.sum(wind_dst * pos_diff, dim=-1, keepdim=True)
            wind_cross_dst = (
                wind_dst[:, 0:1] * pos_diff[:, 1:2]
                - wind_dst[:, 1:2] * pos_diff[:, 0:1]
            )

            edge_attr = torch.cat(
                [
                    radial,
                    wind_dot_src,
                    wind_cross_src,
                    wind_dot_dst,
                    wind_cross_dst,
                ],
                dim=-1,
            )  # [E, 5]

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
            data_list.append(data)

        data = Batch.from_data_list(data_list)

        out = self.model(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=data.batch,
        )

        out = out.view(*B_all, N, self.out_dim)
        if not has_batch:
            out = out.squeeze(0)

        return out


class CriticHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, 1)

    def forward(self, x):
        x = self.lin(x)
        shape = x.shape
        x = torch.mean(x, dim=-2, keepdim=True).expand(shape)
        return x


class MLPPolicy(nn.Module):
    def __init__(
        self,
        in_dim: int,
        depth,
        num_cells,
        action_dim: int,
        n_agents: int,
        share_params: bool = True,
    ):
        super().__init__()
        self.model = MultiAgentMLP(
            n_agent_inputs=in_dim,
            n_agent_outputs=action_dim,
            n_agents=n_agents,
            centralised=False,
            share_params=share_params,
            depth=depth,
            num_cells=num_cells,
            activation_class=nn.Tanh,
        )

        self.std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        mu = self.model(x)
        std = (
            torch.ones_like(mu) * self.std
        )  # NormalParamExtractor manages transformation
        return torch.cat((mu, std), dim=-1)


class MLPObservationNormalizer(nn.Module):
    def __init__(
        self,
        scenario: ScenarioConfig,
        wind_speed_low: int | torch.Tensor = 0,
        wind_speed_high: int | torch.Tensor = 28,
    ):
        super().__init__()
        self.wind_speed_low = wind_speed_low
        self.wind_speed_high = wind_speed_high
        self.scenario = scenario

    def forward(
        self,
        wind_direction: torch.Tensor,  # [*B, N, 1]
        wind_speed: torch.Tensor,  # [*B, N, 1]
        yaw: torch.Tensor,  # [*B, N, 1]
        layout: torch.Tensor,  # [*B, N, 2]
    ):
        has_batch = len(wind_direction.shape) > 2
        if not has_batch:
            wind_direction = wind_direction.unsqueeze(0)
            wind_speed = wind_speed.unsqueeze(0)
            yaw = yaw.unsqueeze(0)
            layout = layout.unsqueeze(0)
        # Max-min normalise wind speed
        wind_speed = (wind_speed - self.wind_speed_low) / (
            self.wind_speed_high - self.wind_speed_low
        )

        # Calculate Cartesian wind vector
        wind_direction = torch.deg2rad(wind_direction)
        wind_x = wind_speed * torch.cos(wind_direction)
        wind_y = wind_speed * torch.sin(wind_direction)
        wind = torch.cat([wind_x, wind_y], dim=-1)  # [B, N, 2]
        yaw = torch.deg2rad(yaw)

        # (-1|1) Normalise Layout
        layout = layout.clone()
        layout[:, :, 0] = (layout[:, :, 0] / self.scenario.map_x_length) * 2 - 1
        layout[:, :, 1] = (layout[:, :, 1] / self.scenario.map_y_length) * 2 - 1

        out = torch.cat((wind, yaw, layout), dim=-1)
        if not has_batch:
            out = out.squeeze(0)
        return out


def wfcrl_models_mlp(
    env,
    cfg: ActorCriticConfig,
    normalisation: NormalisationStatistics | None,
    device: str,
):
    observation_keys = [
        ("turbine", "observation", x)
        for x in ["wind_direction", "wind_speed", "yaw", "layout"]
    ]

    normaliser = MLPObservationNormalizer(
        scenario=env._env._scenario_cfg,
        wind_speed_low=env.observation_spec["turbine", "observation", "wind_speed"].low,
        wind_speed_high=env.observation_spec[
            "turbine", "observation", "wind_speed"
        ].high,
    ).to(device=device)

    normaliser_module = TensorDictModule(
        module=normaliser,
        in_keys=observation_keys,
        out_keys=[("turbine", "observation_vec")],
    )

    scale_mapping = "biased_softplus_" + str(cfg.initial_std)
    policy_mlp = nn.Sequential(
        MLPPolicy(
            in_dim=5,
            depth=cfg.mlp_depth,
            num_cells=cfg.mlp_hidden_size,
            action_dim=env.action_spec.shape[-1],
            n_agents=env.num_agents,
            share_params=False,
        ).to(device=device),
        NormalParamExtractor(scale_mapping=scale_mapping, scale_lb=0.01),
    )

    policy_mlp_module = TensorDictModule(
        module=policy_mlp,
        in_keys=[("turbine", "observation_vec")],
        out_keys=[("turbine", "loc"), ("turbine", "scale")],
    )

    policy_module = TensorDictSequential(
        normaliser_module,
        policy_mlp_module,
        selected_out_keys=[("turbine", "loc"), ("turbine", "scale")],
    )

    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec_unbatched,
        in_keys=[("turbine", "loc"), ("turbine", "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        default_interaction_type=InteractionType.RANDOM,
        distribution_kwargs={
            "low": env.full_action_spec_unbatched[env.action_key].space.low,
            "high": env.full_action_spec_unbatched[env.action_key].space.high,
        },
        return_log_prob=True,
        log_prob_key=("turbine", "sample_log_prob"),
    )
    critic_denormaliser = maybe_make_denormaliser(normalisation)
    critic_mlp = nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=5,
            n_agent_outputs=1,  # 1 value per agent
            n_agents=env.num_agents,
            centralised=True,
            share_params=True,
            device=device,
            depth=cfg.mlp_depth,
            num_cells=cfg.mlp_hidden_size,
            activation_class=torch.nn.Tanh,
        ),
        critic_denormaliser,
    )

    critic_mlp_module = TensorDictModule(
        module=critic_mlp,
        in_keys=[("turbine", "observation_vec")],
        out_keys=[("turbine", "state_value")],
    )

    critic_module = TensorDictSequential(
        normaliser_module,
        critic_mlp_module,
        selected_out_keys=[("turbine", "state_value")],
    )

    critic = TensorDictModule(
        module=critic_module,
        in_keys=observation_keys,
        out_keys=[("turbine", "state_value")],
    )

    # Initialise
    td = env.reset().to(device)
    with torch.no_grad():
        policy(td)
        critic(td)

    return policy, critic


def maybe_make_denormaliser(normalisation: NormalisationStatistics | None):
    if normalisation is not None:
        return OutputDenormaliser(
            mean=normalisation.episode_return_mean, std=normalisation.episode_return_std
        )
    else:
        return ID()


def wfcrl_models_gnn(
    env,
    cfg: ActorCriticConfig,
    normalisation: NormalisationStatistics | None,
    device: str,
):
    observation_keys = [
        ("turbine", "observation", x)
        for x in ["wind_direction", "wind_speed", "yaw", "layout"]
    ]

    policy_gnn = WindFarmGNN(
        scenario=env._env._scenario_cfg,
        node_emb_dim=cfg.policy_node_hidden_size,
        out_dim=cfg.policy_node_hidden_size,
        n_layers=cfg.policy_gnn_depth,
        wind_speed_low=env.observation_spec["turbine", "observation", "wind_speed"].low,
        wind_speed_high=env.observation_spec[
            "turbine", "observation", "wind_speed"
        ].high,
        k=cfg.policy_graph_k,
    ).to(device=device)
    policy_gnn_key = [("turbine", "observation", "policy_gnn_features")]
    policy_gnn_module = TensorDictModule(
        module=policy_gnn,
        in_keys=observation_keys,
        out_keys=policy_gnn_key,
    )

    scale_mapping = "biased_softplus_" + str(cfg.initial_std)
    policy_mlp = nn.Sequential(
        MLPPolicy(
            in_dim=cfg.policy_node_hidden_size,
            depth=cfg.policy_head_depth,
            num_cells=cfg.policy_head_hidden_size,
            action_dim=env.action_spec.shape[-1],
            n_agents=env.num_agents,
            share_params=True,
        ),
        NormalParamExtractor(scale_mapping=scale_mapping, scale_lb=0.01),
    )

    policy_mlp_module = TensorDictModule(
        module=policy_mlp,
        in_keys=policy_gnn_key,
        out_keys=[("turbine", "loc"), ("turbine", "scale")],
    )

    policy_module = TensorDictSequential(
        policy_gnn_module,
        policy_mlp_module,
        selected_out_keys=[("turbine", "loc"), ("turbine", "scale")],
    )

    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec_unbatched,
        default_interaction_type=InteractionType.RANDOM,
        in_keys=[("turbine", "loc"), ("turbine", "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.full_action_spec_unbatched[env.action_key].space.low,
            "high": env.full_action_spec_unbatched[env.action_key].space.high,
        },
        return_log_prob=True,
        log_prob_key=("turbine", "sample_log_prob"),
    )

    critic_gnn = WindFarmGNN(
        scenario=env._env._scenario_cfg,
        node_emb_dim=cfg.critic_node_hidden_size,
        out_dim=cfg.critic_node_hidden_size,
        n_layers=cfg.critic_gnn_depth,
        wind_speed_low=env.observation_spec["turbine", "observation", "wind_speed"].low,
        wind_speed_high=env.observation_spec[
            "turbine", "observation", "wind_speed"
        ].high,
        k=cfg.critic_graph_k,
    ).to(device=device)
    critic_gnn_key = [("turbine", "observation", "critic_gnn_features")]
    critic_gnn_module = TensorDictModule(
        module=critic_gnn,
        in_keys=observation_keys,
        out_keys=critic_gnn_key,
    )

    critic_denormaliser = maybe_make_denormaliser(normalisation)

    critic_head = nn.Sequential(
        CriticHead(in_dim=cfg.critic_node_hidden_size),
        critic_denormaliser,
    )

    critic_head_module = TensorDictModule(
        module=critic_head,
        in_keys=critic_gnn_key,
        out_keys=[("turbine", "state_value")],
    )

    critic = TensorDictSequential(
        critic_gnn_module,
        critic_head_module,
        selected_out_keys=[("turbine", "state_value")],
    )

    policy = policy.to(device)
    critic = critic.to(device)

    # Initialise
    td = env.reset().to(device)
    with torch.no_grad():
        policy(td)
        critic(td)

    return policy, critic


def wfcrl_models(
    env,
    cfg: ActorCriticConfig,
    normalisation: NormalisationStatistics | None,
    device: str,
):
    match cfg.model_type:
        case "mlp":
            return wfcrl_models_mlp(env, cfg, normalisation, device)
        case "gnn":
            return wfcrl_models_gnn(env, cfg, normalisation, device)
        case _:
            raise ValueError(f"Unknown model type: {cfg.model_type}")
