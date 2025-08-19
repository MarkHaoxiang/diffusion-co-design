from abc import ABC, abstractmethod
from typing import Literal
import numpy as np
from tqdm import tqdm
import torch


class GenerationMethod(ABC):
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
        self.n_placements = len(radius)  # Number of placements
        self.w = map_x_length
        self.h = map_y_length
        if isinstance(rng, np.random.Generator):
            self._rng = rng
        else:
            self._rng = np.random.default_rng(rng)

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

    @abstractmethod
    def generate(self, n: int = 1, disable_tqdm: bool = True):
        raise NotImplementedError()


class SamplingGeneration(GenerationMethod):
    def __init__(
        self,
        radius: list[float],
        occupied_locations: list[tuple[float, float]],
        occupied_radius: list[float],
        map_x_length: float,
        map_y_length: float,
        additional_minimum_distance: float = 0.0,
        max_attempts_per_environment: int = 100,
        max_backtrack_attempts: int = 1,
        rng: np.random.Generator | int | None = None,
    ):
        super().__init__(
            radius=radius,
            occupied_locations=occupied_locations,
            occupied_radius=occupied_radius,
            map_x_length=map_x_length,
            map_y_length=map_y_length,
            additional_minimum_distance=additional_minimum_distance,
            rng=rng,
        )

        self.max_attempts_per_environment = max_attempts_per_environment
        self.max_backtrack_attempts = max_backtrack_attempts

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

    def generate(
        self,
        n: int = 1,
        disable_tqdm: bool = True,
    ):
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
                    if attempts >= self.max_attempts_per_environment:
                        if backtrack_attempts < self.max_backtrack_attempts:
                            j -= 1
                            backtrack_attempts += 1
                            points.pop()
                        else:
                            j = max(0, j - backtrack_amount)
                            points = points[:j]
                            backtrack_attempts = 0
                        attempts = 0

        return placements


# Pos between [-1, 1], pre-scaling
def soft_penalty(
    existing_pos: torch.Tensor,  # [N, 2]
    existing_radius: torch.Tensor,  # [N]
    pos: torch.Tensor,  # [B, M, 2]
    radii: torch.Tensor,  # [B, M]
    additional_collision_distance: float,
    original_pos: torch.Tensor,  # [B, M, 2]
    x_scale: float = 1.0,
    y_scale: float = 1.0,
):
    B, M, _ = pos.shape
    N = existing_pos.shape[0]

    scaled_pos = pos.clone()
    scaled_pos[:, :, 0] = scaled_pos[:, :, 0] * x_scale
    scaled_pos[:, :, 1] = scaled_pos[:, :, 1] * y_scale

    existing_pos_exp = existing_pos.unsqueeze(0).expand(B, N, 2)  # [B, N, 2]
    existing_rad_exp = existing_radius.unsqueeze(0).expand(B, N)  # [B, N]
    scale_factor = max(x_scale, y_scale)

    all_pos = torch.cat([scaled_pos, existing_pos_exp], dim=1)  # [B, M+N, 2]
    all_rad = (
        torch.cat([radii, existing_rad_exp], dim=1) + additional_collision_distance / 2
    )

    total_count = M + N
    dist = torch.cdist(all_pos, all_pos, p=2)  # [B, total, total]
    min_dist = all_rad.unsqueeze(2) + all_rad.unsqueeze(1)  # [B, total, total]

    violation = (min_dist - dist).clamp(min=0.0)

    mask_upper = torch.triu(
        torch.ones(total_count, total_count, device=pos.device), diagonal=1
    ).bool()
    penalty = violation[:, mask_upper].sum(dim=-1) / scale_factor

    collision_mask_new = (violation[:, :M, :] > 0).any(dim=2)  # [B, M]
    non_colliding_mask = ~collision_mask_new  # [B, M]

    deviation = (pos - original_pos).norm(dim=-1)  # [B, M]
    penalty_deviation = (deviation * non_colliding_mask.float()).sum(dim=-1) * 0.05
    penalty += penalty_deviation

    return penalty


def soft_projection_constraint(
    existing_pos: torch.Tensor,
    existing_radius: torch.Tensor,
    pos: torch.Tensor,
    radii: torch.Tensor,
    additional_collision_distance: float = 0.0,
    x_scale: float = 1.0,
    y_scale: float = 1.0,
    projection_steps: int = 30,
    penalty_lr: float = 0.02,
    rng: torch.Generator | None = None,
):
    B, M, _ = pos.shape
    N = existing_pos.shape[0]
    assert existing_pos.shape == (N, 2)
    assert existing_radius.shape == (N,)
    assert pos.shape == (B, M, 2)
    assert radii.shape == (B, M)

    pos_0 = pos.clone()
    noise = (
        torch.randn(
            size=(B, M, 2), dtype=torch.float32, generator=rng, device=pos.device
        )
        * 0.01
    )
    pos = pos + noise

    with torch.enable_grad():
        pos.requires_grad_(True)
        optim = torch.optim.SGD([pos], lr=penalty_lr)
        for _ in range(projection_steps):
            optim.zero_grad()
            penalty = soft_penalty(
                existing_pos=existing_pos,
                existing_radius=existing_radius,
                pos=pos,
                radii=radii,
                original_pos=pos_0,
                additional_collision_distance=additional_collision_distance,
                x_scale=x_scale,
                y_scale=y_scale,
            ).sum()
            penalty.backward()
            optim.step()
            pos.data[..., 0] = pos.data[..., 0].clamp(min=-1, max=1)
            pos.data[..., 1] = pos.data[..., 1].clamp(min=-1, max=1)
        pos = pos.detach()

    return pos


class ProjectionGenerator(GenerationMethod):
    def __init__(
        self,
        radius: list[float],
        occupied_locations: list[tuple[float, float]],
        occupied_radius: list[float],
        map_x_length: float,
        map_y_length: float,
        additional_minimum_distance: float = 0.0,
        projection_steps: int = 30,
        penalty_lr: float = 0.02,
        rng: np.random.Generator | int | None = None,
    ):
        super().__init__(
            radius=radius,
            occupied_locations=occupied_locations,
            occupied_radius=occupied_radius,
            map_x_length=map_x_length,
            map_y_length=map_y_length,
            additional_minimum_distance=additional_minimum_distance,
            rng=rng,
        )
        self.projection_steps = projection_steps
        self.penalty_lr = penalty_lr
        self.torch_rng = torch.Generator().manual_seed(int(self._rng.integers(2**32)))

    def generate(self, n=1, disable_tqdm=True):
        batch_size = 64
        placements = np.zeros((n, self.n_placements, 2), dtype=np.float32)

        if len(self.occupied_locations) == 0:
            existing_pos = torch.zeros((0, 2), dtype=torch.float32)
        else:
            existing_pos = torch.tensor(self.occupied_locations)
        existing_pos = existing_pos / torch.tensor([[self.w, self.h]]) * 2 - 1
        existing_radius = torch.tensor(self.occupied_radius[: existing_pos.shape[0]])

        i = 0
        with tqdm(total=n, disable=disable_tqdm) as pbar:
            while i < n:
                B = min(batch_size, n - i)
                candidates = (
                    torch.rand(
                        size=(B, self.n_placements, 2),
                        dtype=torch.float32,
                        generator=self.torch_rng,
                    )
                    * 2
                    - 1
                )
                candidates_radii = (
                    torch.tensor(self.placement_radius)
                    .unsqueeze(0)
                    .expand(B, len(self.placement_radius))
                )

                pos = soft_projection_constraint(
                    existing_pos=existing_pos,
                    existing_radius=existing_radius,
                    pos=candidates,
                    radii=candidates_radii,
                    x_scale=self.w / 2,
                    y_scale=self.h / 2,
                    projection_steps=self.projection_steps,
                    penalty_lr=self.penalty_lr,
                    rng=self.torch_rng,
                ).numpy()

                pos[:, :, 0] = self.w * (pos[:, :, 0] + 1) / 2
                pos[:, :, 1] = self.h * (pos[:, :, 1] + 1) / 2
                placements[i : i + B] = pos

                pbar.update(B)
                i += B

        return placements


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
        sampling_method: Literal["sampling", "projection"] = "sampling",
        **kwargs,
    ):
        super().__init__()

        self.sampling_method = sampling_method
        self.w = map_x_length
        self.h = map_y_length

        match sampling_method:
            case "sampling":
                self._generate: GenerationMethod = SamplingGeneration(
                    radius=radius,
                    occupied_locations=occupied_locations,
                    occupied_radius=occupied_radius,
                    map_x_length=map_x_length,
                    map_y_length=map_y_length,
                    additional_minimum_distance=additional_minimum_distance,
                    rng=rng,
                    **kwargs,
                )
            case "projection":
                self._generate = ProjectionGenerator(
                    radius=radius,
                    occupied_locations=occupied_locations,
                    occupied_radius=occupied_radius,
                    map_x_length=map_x_length,
                    map_y_length=map_y_length,
                    additional_minimum_distance=additional_minimum_distance,
                    rng=rng,
                    **kwargs,
                )
            case _:
                raise NotImplementedError()

    def __call__(
        self,
        n: int = 1,
        training_dataset: bool = False,
        disable_tqdm: bool = True,
    ):
        placements = self._generate.generate(n=n, disable_tqdm=disable_tqdm)

        if training_dataset:
            # Normalise to [-1, 1]
            placements[:, :, 0] = placements[:, :, 0] / self.w * 2 - 1
            placements[:, :, 1] = placements[:, :, 1] / self.h * 2 - 1

        return placements
