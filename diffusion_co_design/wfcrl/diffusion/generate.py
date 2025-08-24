import numpy as np
from diffusion_co_design.common.design.generate import Generate as _Generate
from diffusion_co_design.wfcrl.schema import ScenarioConfig
from diffusion_co_design.wfcrl.diffusion.constraints import slsqp_projection_constraint


class Generate(_Generate):
    def __init__(
        self,
        num_turbines: int,
        map_x_length: int,
        map_y_length: int,
        minimum_distance_between_turbines: int,
        rng: np.random.Generator | int | None = None,
        post_slsqp_projection: bool = False,  # Apply a secondary projection which has better constraint satisfaction, but costs more compute.
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

        self.post_slsqp_projection = post_slsqp_projection
        # TODO: Should refactor WFCRL to avoid using a dummy config
        self.slsqp = slsqp_projection_constraint(
            ScenarioConfig(
                name="placeholder",
                n_turbines=num_turbines,
                map_x_length=map_x_length,
                map_y_length=map_y_length,
                min_distance_between_turbines=minimum_distance_between_turbines,
                max_steps=1,
            )
        )

    def __call__(self, n=1, training_dataset=False, disable_tqdm=True):
        x = super().__call__(n, training_dataset, disable_tqdm)

        if self.post_slsqp_projection:
            if not training_dataset:
                # Normalise to [-1, 1]
                x[:, :, 0] = x[:, :, 0] / self.w * 2 - 1
                x[:, :, 1] = x[:, :, 1] / self.h * 2 - 1

            x = self.slsqp(x).numpy(force=True)

            if not training_dataset:
                x[:, :, 0] = (x[:, :, 0] + 1) / 2 * self.w
                x[:, :, 1] = (x[:, :, 1] + 1) / 2 * self.h

        return x
