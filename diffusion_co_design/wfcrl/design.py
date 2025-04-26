import torch

from diffusion_co_design.common.design import BaseDesigner
from diffusion_co_design.wfcrl.schema import ScenarioConfig
from diffusion_co_design.wfcrl.diffusion.generate import Generate


class Designer(BaseDesigner):
    def __init__(self, scenario: ScenarioConfig, environment_repeats: int = 1):
        super().__init__(environment_repeats=environment_repeats)
        self.scenario = scenario


class FixedDesigner(Designer):
    def __init__(self, scenario):
        super().__init__(scenario)
        self.layout_image = torch.nn.Parameter(
            RandomDesigner(scenario)._generate_environment_weights(None),
            requires_grad=False,
        )

    def forward(self, objective):
        return self.layout_image.data

    def _generate_environment_weights(self, objective):
        return self.layout_image


class RandomDesigner(Designer):
    def __init__(self, scenario: ScenarioConfig, environment_repeats: int = 1):
        super().__init__(scenario=scenario, environment_repeats=environment_repeats)

        self.generate = Generate(
            num_turbines=scenario.n_turbines,
            map_x_length=scenario.map_x_length,
            map_y_length=scenario.map_y_length,
            minimum_distance_between_turbines=scenario.min_distance_between_turbines,
        )

    def forward(self, objective=None):
        theta = torch.tensor(self.generate(n=1, training_dataset=False)).squeeze(0)
        return theta

    def _generate_environment_weights(self, objective):
        return self.forward(objective)
