import numpy as np
from diffusion_co_design.common.design.generate import Generate as _Generate
from diffusion_co_design.vmas.schema import ScenarioConfig


class Generate(_Generate):
    def __init__(
        self,
        scenario: ScenarioConfig,
        rng: np.random.Generator | int | None = None,
    ):
        super().__init__(
            radius=scenario.obstacle_sizes,
            occupied_locations=scenario.agent_spawns + scenario.agent_goals,
            occupied_radius=[0.1] * len(scenario.agent_spawns) * 2,
            map_x_length=scenario.world_spawning_x,
            map_y_length=scenario.world_spawning_y,
            additional_minimum_distance=0.005,
            rng=rng,
        )
