import numpy as np
from diffusion_co_design.common.design.generate import Generate as _Generate


class Generate(_Generate):
    def __init__(
        self,
        num_turbines: int,
        map_x_length: int,
        map_y_length: int,
        minimum_distance_between_turbines: int,
        rng: np.random.Generator | int | None = None,
    ):
        super().__init__(
            radius=[minimum_distance_between_turbines / 2] * num_turbines,
            occupied_locations=[],
            occupied_radius=[],
            map_x_length=map_x_length,
            map_y_length=map_y_length,
            additional_minimum_distance=0.0,
            sampling_method="projection",
            projection_steps=20,
            penalty_lr=0.02,
            rng=rng,
        )
