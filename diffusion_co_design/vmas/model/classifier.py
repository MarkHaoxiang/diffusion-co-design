import torch
from diffusion_co_design.common.nn import EnvCritic as _EnvCritic
from diffusion_co_design.vmas.schema import ScenarioConfig
from diffusion_co_design.vmas.model.shared import E3Critic


class EnvCritic(_EnvCritic, E3Critic):
    def __init__(self, scenario: ScenarioConfig):
        _EnvCritic.__init__(self)
        E3Critic.__init__(self, scenario)

    def forward(self, x: torch.Tensor):
        pass
