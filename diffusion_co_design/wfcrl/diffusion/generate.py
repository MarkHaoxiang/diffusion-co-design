import numpy as np
from tqdm import tqdm


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
            size=(128, 2),
        )

    def __call__(
        self,
        n: int = 1,
        training_dataset: bool = False,
        max_attempts_per_environment: int = 100,
        max_backtrack_attempts: int = 1,
        disable_tqdm: bool = True,
    ):
        rng = self._rng_uninitialised
        if isinstance(rng, np.random.Generator):
            self._rng = rng
        else:
            self._rng = np.random.default_rng(rng)
        self._reset_random_number_queue()  # Batched for speed

        environments = np.zeros((n, self.num_turbines, 2), dtype=np.float32)
        for i in tqdm(range(n), disable=disable_tqdm):
            attempts = 0
            backtrack_attempts = 0
            points = []
            j = 0
            while j < self.num_turbines:
                candidate = self._get_random_point()
                candidate[0] = candidate[0] * self.w
                candidate[1] = candidate[1] * self.h
                if (
                    len(points) == 0
                    or (
                        np.linalg.norm(np.stack(points) - candidate, axis=-1) >= self.d
                    ).all()
                ):
                    points.append(candidate)
                    environments[i, j] = candidate
                    j += 1
                    attempts = 0
                else:
                    attempts += 1
                    if attempts >= max_attempts_per_environment:
                        if backtrack_attempts < max_backtrack_attempts:
                            j -= 1
                            backtrack_attempts += 1
                            points.pop()
                        else:
                            j = 0
                            backtrack_attempts = 0
                            points = []
                        attempts = 0

        if training_dataset:
            # Normalise to [-1, 1]
            environments[:, :, 0] = environments[:, :, 0] / self.w * 2 - 1
            environments[:, :, 1] = environments[:, :, 1] / self.h * 2 - 1

        return environments
