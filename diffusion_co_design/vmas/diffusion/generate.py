from typing import Literal
import numpy as np
from diffusion_co_design.common.design.generate import Generate as _Generate
from diffusion_co_design.vmas.schema import (
    ScenarioConfigType,
    GlobalPlacementScenarioConfig,
    LocalPlacementScenarioConfig,
)


class GlobalGenerate(_Generate):
    def __init__(
        self,
        scenario: GlobalPlacementScenarioConfig,
        rng: np.random.Generator | int | None = None,
        sampling_method: Literal["projection", "sampling"] = "projection",
        **kwargs,
    ):
        occupied_locations = scenario.agent_spawns + scenario.agent_goals
        occupied_locations = [
            (x + scenario.world_spawning_x, y + scenario.world_spawning_y)
            for (x, y) in occupied_locations
        ]

        super().__init__(
            radius=scenario.obstacle_sizes,
            occupied_locations=occupied_locations,
            occupied_radius=[0.05] * len(scenario.agent_spawns) * 2,
            map_x_length=scenario.world_spawning_x * 2,
            map_y_length=scenario.world_spawning_y * 2,
            additional_minimum_distance=0.005,
            rng=rng,
            sampling_method=sampling_method,
            **kwargs,
        )

    def __call__(
        self,
        n=1,
        training_dataset=False,
        disable_tqdm=True,
    ):
        res = super().__call__(n, training_dataset, disable_tqdm)

        if not training_dataset:
            res[:, :, 0] -= self.w / 2
            res[:, :, 1] -= self.h / 2

        return res


class LocalGenerate:
    def __init__(
        self,
        scenario: LocalPlacementScenarioConfig,
        rng: np.random.Generator | int | None = None,
    ):
        self.scenario = scenario
        if isinstance(rng, np.random.Generator):
            self._rng = rng
        else:
            self._rng = np.random.default_rng(rng)

    def __call__(
        self,
        n=1,
        training_dataset=False,
        disable_tqdm=True,
    ):
        return self._rng.random(
            size=(n, *self.scenario.diffusion_shape), dtype=np.float32
        )


def create_generate(
    scenario: ScenarioConfigType, rng: np.random.Generator | int | None = None, **kwargs
):
    if isinstance(scenario, GlobalPlacementScenarioConfig):
        return GlobalGenerate(scenario, rng, **kwargs)
    elif isinstance(scenario, LocalPlacementScenarioConfig):
        return LocalGenerate(scenario, rng)
