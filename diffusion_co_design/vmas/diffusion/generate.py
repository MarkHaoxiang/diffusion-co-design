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
            map_x_length=scenario.world_spawning_x * 2,
            map_y_length=scenario.world_spawning_y * 2,
            additional_minimum_distance=0.005,
            rng=rng,
        )

    def __call__(
        self,
        n=1,
        training_dataset=False,
        max_attempts_per_environment=100,
        max_backtrack_attempts=1,
        disable_tqdm=True,
    ):
        res = super().__call__(
            n,
            training_dataset,
            max_attempts_per_environment,
            max_backtrack_attempts,
            disable_tqdm,
        )

        if not training_dataset:
            res[:, :, 0] -= self.w / 2
            res[:, :, 1] -= self.h / 2

        return res
