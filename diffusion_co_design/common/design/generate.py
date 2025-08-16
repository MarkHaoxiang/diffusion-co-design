import numpy as np
from tqdm import tqdm


class Generate:
    def __init__(
        self,
        radius: list[float],
        occupied_locations: list[tuple[float, float]],
        occupied_radius: list[float],
        map_x_length: float,
        map_y_length: float,
        additional_minimum_distance: float = 0.0,
        rng: np.random.Generator | int | None = None,
    ):
        super().__init__()
        self.n_placements = len(radius)  # Number of placements
        self.w = map_x_length
        self.h = map_y_length
        self._rng_uninitialised = rng

        self.placement_radius = (
            np.array(radius, dtype=np.float32) + additional_minimum_distance / 2
        )
        self.occupied_locations = occupied_locations
        self.occupied_radius = (
            np.array(occupied_radius, dtype=np.float32)
            + additional_minimum_distance / 2
        )
        self.occupied_radius = np.concatenate(
            [self.occupied_radius, self.placement_radius], axis=0
        )

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

        placements = np.zeros((n, self.n_placements, 2), dtype=np.float32)
        for i in tqdm(range(n), disable=disable_tqdm):
            attempts = 0
            backtrack_attempts = 0
            points: list = self.occupied_locations.copy()
            j = 0
            backtrack_amount = 3
            while j < self.n_placements:
                candidate = self._get_random_point()
                candidate[0] = candidate[0] * self.w
                candidate[1] = candidate[1] * self.h

                if len(points) > 0:
                    dist = np.linalg.norm(np.stack(points) - candidate, axis=-1)
                    min_dist = (
                        self.placement_radius[j] + self.occupied_radius[: dist.shape[0]]
                    )

                if len(points) == 0 or np.all(dist >= min_dist):
                    points.append(candidate)
                    placements[i, j] = candidate
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
                            j = max(0, j - backtrack_amount)
                            backtrack_attempts = 0
                            points = []
                        attempts = 0

        if training_dataset:
            # Normalise to [-1, 1]
            placements[:, :, 0] = placements[:, :, 0] / self.w * 2 - 1
            placements[:, :, 1] = placements[:, :, 1] / self.h * 2 - 1

        return placements
