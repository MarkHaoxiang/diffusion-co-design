import numpy as np
import warnings


class Generate:
    def __init__(
        self,
        num_turbines: int,
        map_x_length: int,
        map_y_length: int,
        minimum_distance_between_turbines: int,
        rng: np.random.Generator | int | None = None,
    ):
        super().__init__()
        self.num_turbines = num_turbines
        self.w = map_x_length
        self.h = map_y_length
        self.d = minimum_distance_between_turbines

        self._rng_uninitialised = rng
        self._lazy_initialise = False

    def _get_random_point(self):
        if self._rng_i >= len(self._random_number_queue):
            self._reset_random_number_queue()
        random_number = self._random_number_queue[self._rng_i]
        self._rng_i += 1
        return random_number

    def _reset_random_number_queue(self):
        self._rng_i = 0
        self._random_number_queue = self._rng.uniform(
            low=0.0,
            high=1.0,
            size=(2048, 2),
        )

    def __call__(
        self,
        n: int = 1,
        training_dataset: bool = False,
        max_attempts_per_environment: int = 10,
    ):
        if not self._lazy_initialise:
            self._lazy_initialise = True
            rng = self._rng_uninitialised
            if isinstance(rng, np.random.Generator):
                self._rng = rng
            else:
                self._rng = np.random.default_rng(rng)
            self._reset_random_number_queue()  # Batched for speed

        environments = np.zeros((n, self.num_turbines, 2), dtype=np.float32)
        i = 0
        while i < n:
            attempts = 0
            points = []
            j = 0
            while j < self.num_turbines:
                candidate = self._get_random_point() * self.w
                if (
                    len(points) == 0
                    or (
                        np.linalg.norm(np.stack(points) - candidate, axis=-1) >= self.d
                    ).all()
                ):
                    points.append(candidate)
                    environments[i, j] = candidate
                    j += 1
                else:
                    attempts += 1
                    if attempts >= max_attempts_per_environment:
                        warnings.warn(
                            f"Could not generate a valid environment after {max_attempts_per_environment} attempts. "
                        )
                        break
            if j == self.num_turbines:
                i += 1

        if training_dataset:
            # Normalise to [-1, 1]
            environments[:, :, 0] = (environments[:, :, 0] / (self.w - 1)) * 2 - 1
            environments[:, :, 1] = (environments[:, :, 1] / (self.h - 1)) * 2 - 1

        return environments
