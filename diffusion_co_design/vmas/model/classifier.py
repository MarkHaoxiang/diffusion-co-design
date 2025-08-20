import torch
from diffusion_co_design.common.nn import EnvCritic as _EnvCritic
from diffusion_co_design.vmas.schema import (
    ScenarioConfigType,
    LocalPlacementScenarioConfig,
)
from diffusion_co_design.vmas.model.shared import E3Critic


def expand_batch(tensor, batch_dims):
    return tensor.view((1,) * len(batch_dims) + tensor.shape).expand(
        batch_dims + tensor.shape
    )


class GNNEnvCritic(_EnvCritic):
    def __init__(
        self,
        scenario: ScenarioConfigType,
        node_emb_dim: int = 128,
        num_layers: int = 3,
        k: int = 5,
    ):
        super().__init__()
        self.agent_pos = torch.tensor(scenario.agent_spawns)
        self.goal_pos = torch.tensor(scenario.agent_goals)
        self.agent_vel = torch.zeros_like(self.agent_pos)

        self.model = E3Critic(
            scenario=scenario, node_emb_dim=node_emb_dim, num_layers=num_layers, k=k
        )

    def forward(self, x: torch.Tensor):
        B_all = x.shape[:-2]
        obstacle_pos = x

        agent_pos = expand_batch(self.agent_pos.clone(), B_all).to(x.device)
        goal_pos = expand_batch(self.goal_pos.clone(), B_all).to(x.device)
        agent_vel = expand_batch(self.agent_vel.clone(), B_all).to(x.device)
        res = self.model(obstacle_pos, agent_pos, goal_pos, agent_vel)
        res = res.sum(dim=-2).squeeze(-1)
        return res


class MLPEnvCritic(_EnvCritic):
    def __init__(
        self, scenario: LocalPlacementScenarioConfig, hidden_dim: int, num_layers: int
    ):
        super().__init__()
        self.scenario = scenario
        self.n_obstacles = len(scenario.obstacle_sizes)

        layers: list[torch.nn.Module] = []
        dim_in = scenario.get_num_agents() * 6 + scenario.diffusion_shape[0]
        for i in range(num_layers - 1):
            layers.append(torch.nn.Linear(dim_in, hidden_dim))
            layers.append(torch.nn.ReLU())
            dim_in = hidden_dim
        layers.append(torch.nn.Linear(dim_in, scenario.get_num_agents()))

        self.agent_pos = torch.tensor(scenario.agent_spawns)
        self.goal_pos = torch.tensor(scenario.agent_goals)
        self.agent_vel = torch.zeros_like(self.agent_pos)
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        # Share agent observations
        B_all = x.shape[:-1]
        layout = x
        agent_pos = expand_batch(self.agent_pos.clone(), B_all).to(x.device)
        goal_pos = expand_batch(self.goal_pos.clone(), B_all).to(x.device)
        agent_vel = expand_batch(self.agent_vel.clone(), B_all).to(x.device)

        shared_obs = torch.cat((agent_pos, goal_pos, agent_vel), dim=-1).flatten(
            start_dim=-2, end_dim=-1
        )
        x_in = torch.cat([shared_obs, layout], dim=-1)

        res = self.model(x_in).unsqueeze(-1)
        res = res.sum(dim=-2).squeeze(-1)
        return res
