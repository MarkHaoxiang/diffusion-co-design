import torch
from diffusion_co_design.common.nn import EnvCritic as _EnvCritic
from diffusion_co_design.vmas.schema import ScenarioConfig
from diffusion_co_design.vmas.model.shared import E3Critic


class EnvCritic(_EnvCritic):
    def __init__(
        self,
        scenario: ScenarioConfig,
        node_emb_dim: int = 128,
        num_layers: int = 3,
        k: int = 5,
    ):
        super().__init__(scenario=scenario)
        self.agent_pos = torch.tensor(scenario.agent_spawns)
        self.goal_pos = torch.tensor(scenario.agent_goals)
        self.agent_vel = torch.zeros_like(self.agent_pos)

        self.model = E3Critic(
            scenario=scenario, node_emb_dim=node_emb_dim, num_layers=num_layers, k=k
        )

    def forward(self, x: torch.Tensor):
        B_all = x.shape[:-2]
        obstacle_pos = x

        def expand_batch(tensor, batch_dims):
            return tensor.view((1,) * len(batch_dims) + tensor.shape).expand(
                batch_dims + tensor.shape
            )

        agent_pos = expand_batch(self.agent_pos.clone(), B_all).to(x.device)
        goal_pos = expand_batch(self.goal_pos.clone(), B_all).to(x.device)
        agent_vel = expand_batch(self.agent_vel.clone(), B_all).to(x.device)
        return self.model(obstacle_pos, agent_pos, goal_pos, agent_vel)
