import warnings

import torch
import numpy as np

from scipy.optimize import minimize
from diffusion_co_design.wfcrl.schema import ScenarioConfig


def soft_penalty(pos: torch.Tensor, min_d: float, original_pos: torch.Tensor):
    _, N, _ = pos.shape
    dist = torch.cdist(pos, pos, p=2)  # [B, N, N]
    mask = torch.triu(torch.ones(N, N, device=pos.device), diagonal=1).bool()
    violation = (min_d - dist).clamp(min=0.0)
    penalty = violation[:, mask].sum(dim=-1)  # [B]

    # For original position, add a penalty for deviation
    original_diff = pos - original_pos
    original_dist = original_diff.norm(dim=-1)  # [B, N]
    penalty += original_dist.sum(dim=-1) * 0.05
    return penalty


def soft_projection_constraint(cfg: ScenarioConfig):
    min_d = 2 * cfg.min_distance_between_turbines / cfg.map_x_length

    def _projection_constraint(pos):
        B, N, _ = pos.shape
        pos_0 = pos.clone()

        y_mult_factor = cfg.map_y_length / cfg.map_x_length
        pos_0[:, :, 1] = pos_0[:, :, 1] * y_mult_factor
        pos = pos_0.clone()

        pos = (
            pos + torch.randn_like(pos) * 0.01
        )  # Add some noise to avoid corners problem, where positions exactly equal from clamping

        # Apply soft penalty
        with torch.enable_grad():
            pos.requires_grad_(True)
            optim = torch.optim.SGD([pos], lr=0.02)
            for _ in range(20):
                optim.zero_grad()
                penalty = soft_penalty(pos, min_d, pos_0).sum()
                penalty.backward()
                optim.step()
                pos.data[..., 0] = pos.data[..., 0].clamp(min=-1, max=1)
                pos.data[..., 1] = pos.data[..., 1].clamp(
                    min=-y_mult_factor, max=y_mult_factor
                )
            pos = pos.detach()

        pos[:, :, 1] = pos[:, :, 1] / y_mult_factor
        return pos

    return _projection_constraint


def slsqp_projection_constraint(cfg: ScenarioConfig):
    min_d = 2 * cfg.min_distance_between_turbines / cfg.map_x_length

    def dist_constraint(i, j):
        def constr(x):
            xi = x[2 * i : 2 * i + 2]
            xj = x[2 * j : 2 * j + 2]
            return np.linalg.norm(xi - xj) - min_d

        return constr

    def _projection_constraint(pos):
        B, N, _ = pos.shape
        if isinstance(pos, torch.Tensor):
            dtype = pos.dtype
            device = pos.device
            pos = pos.numpy(force=True)
        else:
            device = torch.device("cpu")
            dtype = torch.float32
        pos_out = []

        y_mult_factor = cfg.map_y_length / cfg.map_x_length
        pos[:, :, 1] = pos[:, :, 1] * y_mult_factor

        for b in range(B):
            x0 = pos[b].flatten()

            def objective(x):
                return np.sum((x - x0) ** 2)

            constraints = [
                {"type": "ineq", "fun": dist_constraint(i, j)}
                # ineq is GEQ 0 constraint
                for i in range(N)
                for j in range(i + 1, N)
            ]

            bounds = [(-1, 1), (-y_mult_factor, y_mult_factor)] * N

            res = minimize(
                objective,
                x0=x0,
                method="SLSQP",
                constraints=constraints,
                bounds=bounds,
                options={"disp": False, "ftol": 1e-3, "maxiter": 50},
            )

            if not res.success:
                warnings.warn(
                    "Projection constraint failed. Using original position. Suggestion: increase map bounds."
                )
                pos_out.append(torch.tensor(pos[b], dtype=dtype, device=device))
            else:
                pos_proj = torch.tensor(res.x, dtype=dtype, device=device).reshape(N, 2)
                pos_out.append(pos_proj)

        pos_proj = torch.stack(pos_out, dim=0)
        pos_proj[:, :, 1] = pos_proj[:, :, 1] / y_mult_factor

        assert pos_proj.shape == pos.shape, (
            f"Expected {pos.shape}, got {pos_proj.shape}"
        )
        return pos_proj

    return _projection_constraint
