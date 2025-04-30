from typing import Literal
from math import prod

import torch
import torch.nn as nn

from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torch_geometric.utils import scatter

from diffusion_co_design.wfcrl.schema import ScenarioConfig, RLConfig
from diffusion_co_design.common.nn import fully_connected


class EquivariantModel(nn.Module):
    def __init__(
        self,
        scenario: ScenarioConfig,
        node_emb_dim: int = 64,
        edge_emb_dim: int = 16,
        n_layers: int = 4,
        wind_speed_low: int | torch.Tensor = 0,
        wind_speed_high: int | torch.Tensor = 28,
        aggr: Literal["add", "mean", "max"] = "add",
    ):
        super().__init__()
        self.scenario = scenario
        self.wind_speed_low = wind_speed_low
        self.wind_speed_high = wind_speed_high
        self.n_turbines = scenario.n_turbines  # N

        edge_index = fully_connected(self.n_turbines)  # [2, E = (N)(N-1)/2]
        # Remove self loops
        self.edge_index = edge_index[:, edge_index[0] != edge_index[1]]

        self.node_emb_dim = node_emb_dim
        self.node_in = nn.Linear(1 + 5, node_emb_dim)
        self.edge_in = nn.Linear(5, edge_emb_dim)

        self.message_mlp_list = nn.ModuleList()
        self.upd_mlp_list = nn.ModuleList()
        for _ in range(n_layers):
            self.message_mlp_list.append(
                nn.Sequential(
                    nn.Linear(node_emb_dim * 2 + edge_emb_dim, edge_emb_dim),
                    nn.SiLU(),
                    nn.Linear(edge_emb_dim, edge_emb_dim),
                    nn.SiLU(),
                )
            )

            self.upd_mlp_list.append(
                nn.Sequential(
                    nn.Linear(node_emb_dim + edge_emb_dim, node_emb_dim),
                    nn.SiLU(),
                    nn.Linear(node_emb_dim, node_emb_dim),
                    nn.SiLU(),
                )
            )

        self.att_mlp = nn.Sequential(nn.Linear(edge_emb_dim, 1), nn.Sigmoid())
        self.aggr = aggr

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

        self.edge_index = self.edge_index.to(device=wind_speed.device)
        B_all = wind_direction.shape[:-2]
        B = prod(B_all)
        N = self.n_turbines
        E = len(self.edge_index[0])

        # Shape checks
        wind_direction = wind_direction.reshape(-1, N, 1)
        assert wind_direction.shape == (B, N, 1), wind_direction.shape
        wind_speed = wind_speed.reshape(-1, N, 1)
        assert wind_speed.shape == (B, N, 1), wind_speed.shape
        yaw = yaw.reshape(-1, N, 1)
        assert yaw.shape == (B, N, 1), yaw.shape
        layout = layout.reshape(-1, N, 2)
        assert layout.shape == (B, N, 2), layout.shape

        # To radians
        wind_direction = torch.deg2rad(wind_direction)

        # Max-min normalise wind speed
        wind_speed = (wind_speed - self.wind_speed_low) / (
            self.wind_speed_high - self.wind_speed_low
        )

        # Calculate Cartesian wind vector
        wind_x = wind_speed * torch.cos(wind_direction)
        wind_y = wind_speed * torch.sin(wind_direction)
        wind = torch.cat([wind_x, wind_y], dim=-1)  # [B, N, 2]

        # Normalise yaw
        yaw = torch.deg2rad(yaw)

        # (-1|1) Normalise Layout
        layout = layout.clone()
        layout[:, :, 0] = (layout[:, :, 0] / self.scenario.map_x_length) * 2 - 1
        layout[:, :, 1] = (layout[:, :, 1] / self.scenario.map_y_length) * 2 - 1

        # Fully connected graph
        src, dst = self.edge_index[0], self.edge_index[1]
        pos_diff = layout[:, dst] - layout[:, src]  # [B, E, 2]
        radial = torch.norm(pos_diff, dim=-1, keepdim=True)  # [B, E, 1]

        # Feature engineering
        wind_src = wind[:, src]
        wind_dot = torch.sum(wind_src * pos_diff, dim=-1, keepdim=True)  # [B, E, 1]
        wind_cross = (
            wind_src[..., 0:1] * pos_diff[..., 1:2]
            - wind_src[..., 1:2] * pos_diff[..., 0:1]
        )  # [B, E, 1]

        assert radial.shape == (B, E, 1), radial.shape
        assert wind_dot.shape == (B, E, 1), wind_dot.shape
        assert wind_cross.shape == (B, E, 1), wind_cross.shape

        # In layers
        edge_features = torch.cat(
            [radial, wind_speed[:, src], wind_dot, wind_cross, yaw[:, src]], dim=-1
        )  # [B, E, 5]
        assert edge_features.shape == (B, E, 5), edge_features.shape
        e = self.edge_in(edge_features)  # [B, E, edge_emb_dim]
        node_features = torch.cat(
            [
                wind_speed,
                scatter(edge_features, src, dim=1, dim_size=N, reduce=self.aggr),
            ],
            dim=-1,
        )  # [B, N, 1 + 5]
        assert node_features.shape == (B, N, 6), node_features.shape
        h = self.node_in(node_features)  # [B, N, node_emb_dim]

        # Message passing
        for i, (message_mlp, upd_mlp) in enumerate(
            zip(self.message_mlp_list, self.upd_mlp_list)
        ):
            is_final_layer = i == len(self.message_mlp_list) - 1

            # Message
            msg = message_mlp(
                torch.cat([h[:, src], h[:, dst], e], dim=-1)
            )  # [B, E, edge_emb_dim]

            # Attention
            msg = msg * self.att_mlp(msg)

            h_aggr = scatter(
                msg, src, dim=1, dim_size=N, reduce=self.aggr
            )  # [B, N, edge_emb_dim]
            h_aggr = upd_mlp(torch.cat([h, h_aggr], dim=-1))  # [B, N, node_emb_dim]

            # Residual connections
            h = h + h_aggr
            if not is_final_layer:
                e = e + msg

        assert h.shape == (B, N, self.node_emb_dim)
        h = h.reshape(*B_all, N, self.node_emb_dim)
        if not has_batch:
            h = h.squeeze(0)

        return h


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
            activation_class=nn.SiLU,
        )

        self.std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        mu = self.model(x)
        std = (
            torch.ones_like(mu) * self.std
        )  # NormalParamExtractor manages transformation
        return torch.cat((mu, std), dim=-1)


def wfcrl_models(env, cfg: RLConfig, device: str):
    observation_keys = [
        ("turbine", "observation", x)
        for x in ["wind_direction", "wind_speed", "yaw", "layout"]
    ]

    gnn = EquivariantModel(
        scenario=env._env._scenario_cfg,
        node_emb_dim=cfg.node_hidden_size,
        edge_emb_dim=cfg.edge_hidden_size,
        n_layers=cfg.backbone_depth,
        wind_speed_low=env.observation_spec["turbine", "observation", "wind_speed"].low,
        wind_speed_high=env.observation_spec[
            "turbine", "observation", "wind_speed"
        ].high,
    ).to(device=device)
    gnn_key = [("turbine", "observation", "gnn_features")]
    gnn_module = TensorDictModule(
        module=gnn,
        in_keys=observation_keys,
        out_keys=gnn_key,
    )

    policy_mlp = nn.Sequential(
        MLPPolicy(
            in_dim=cfg.node_hidden_size,
            depth=cfg.head_depth,
            num_cells=cfg.mlp_hidden_size,
            action_dim=env.action_spec.shape[-1],
            n_agents=env.num_agents,
            share_params=True,
        ),
        NormalParamExtractor(scale_lb=0.01),
    )

    policy_mlp_module = TensorDictModule(
        module=policy_mlp,
        in_keys=gnn_key,
        out_keys=[("turbine", "loc"), ("turbine", "scale")],
    )

    policy_module = TensorDictSequential(
        gnn_module,
        policy_mlp_module,
        selected_out_keys=[("turbine", "loc"), ("turbine", "scale")],
    )

    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec_unbatched,
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

    critic_mlp = MultiAgentMLP(
        n_agent_inputs=cfg.node_hidden_size,
        n_agent_outputs=1,  # 1 value per agent
        n_agents=env.num_agents,
        centralised=False,
        share_params=True,
        device=device,
        depth=cfg.head_depth,
        num_cells=cfg.mlp_hidden_size,
        activation_class=torch.nn.SiLU,
    )

    critic_mlp_module = TensorDictModule(
        module=critic_mlp,
        in_keys=gnn_key,
        out_keys=[("turbine", "state_value")],
    )

    critic_module = TensorDictSequential(
        gnn_module,
        critic_mlp_module,
        selected_out_keys=[("turbine", "state_value")] + gnn_key,
    )

    critic = TensorDictModule(
        module=critic_module,
        in_keys=observation_keys,
        out_keys=[("turbine", "state_value")],
    )

    policy = policy.to(device)
    critic = critic.to(device)

    # Initialise
    td = env.reset().to(device)
    with torch.no_grad():
        policy(td)
        critic(td)

    return policy, critic
