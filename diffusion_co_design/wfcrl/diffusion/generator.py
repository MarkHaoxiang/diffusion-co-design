import warnings

import torch
from torch import nn
import numpy as np

from scipy.optimize import minimize
from diffusion_co_design.common.design import OptimizerDetails, BaseGenerator
from diffusion_co_design.wfcrl.schema import ScenarioConfig
from diffusion_co_design.wfcrl.model.diffusion import diffusion_setup


def train_to_eval(env: torch.Tensor, cfg: ScenarioConfig, constraint_fn=None):
    if constraint_fn:
        env = constraint_fn(env)
    env[:, :, 0] = (env[:, :, 0] + 1) / 2 * (cfg.map_x_length - 1)
    env[:, :, 1] = (env[:, :, 1] + 1) / 2 * (cfg.map_y_length - 1)
    return env


def eval_to_train(env: torch.Tensor, cfg: ScenarioConfig):
    env[:, :, 0] = env[:, :, 0] / (cfg.map_x_length - 1) * 2 - 1
    env[:, :, 1] = env[:, :, 1] / (cfg.map_y_length - 1) * 2 - 1
    return env


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
    min_d = 2 * cfg.min_distance_between_turbines / (cfg.map_x_length - 1)

    def _projection_constraint(pos):
        B, N, _ = pos.shape
        pos_0 = pos.clone()

        # Apply soft penalty
        with torch.enable_grad():
            pos.requires_grad_(True)
            optim = torch.optim.SGD([pos], lr=0.02)
            for _ in range(20):
                optim.zero_grad()
                penalty = soft_penalty(pos, min_d, pos_0).sum()
                penalty.backward()
                optim.step()
                pos.data = pos.data.clamp(-1, 1)
            pos = pos.detach()

        return pos

    return _projection_constraint


def slsqp_projection_constraint(cfg: ScenarioConfig):
    min_d = 2 * cfg.min_distance_between_turbines / (cfg.map_x_length - 1)

    def _projection_constraint(pos):
        B, N, _ = pos.shape
        dtype = pos.dtype
        device = pos.device
        pos = pos.numpy(force=True)
        pos_out = []

        for b in range(B):
            x0 = pos[b].flatten()

            # Minimize distance to original position
            def objective(x):
                return np.sum((x - x0) ** 2)

            # Subject to bounding box and minimum distance constraints
            def dist_constraint(i, j):
                def constr(x):
                    xi = x[2 * i : 2 * i + 2]
                    xj = x[2 * j : 2 * j + 2]
                    return np.linalg.norm(xi - xj) - min_d

                return constr

            constraints = [
                {"type": "ineq", "fun": dist_constraint(i, j)}
                # ineq is GEQ 0 constraint
                for i in range(N)
                for j in range(i + 1, N)
            ]

            bounds = [(-1, 1)] * (2 * N)

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
        assert pos_proj.shape == pos.shape, (
            f"Expected {pos.shape}, got {pos_proj.shape}"
        )
        return pos_proj

    return _projection_constraint


class Generator(BaseGenerator):
    def __init__(
        self,
        generator_model_path: str,
        scenario: ScenarioConfig,
        batch_size: int = 32,
        rng: torch.Generator | None = None,
        guidance_wt: float = 50.0,
        device: torch.device = torch.device("cpu"),
    ):
        self.model, self.diffusion = diffusion_setup(scenario)
        super().__init__(
            generator_model_path=generator_model_path,
            model=self.model,
            diffusion=self.diffusion,
            batch_size=batch_size,
            rng=rng,
            guidance_wt=guidance_wt,
            device=device,
        )

        self.scenario = scenario
        self.post_process_fn = slsqp_projection_constraint(scenario)

    def generate_batch(
        self,
        batch_size: int | None = None,
        value: nn.Module | None = None,
        use_operation: bool = False,
        operation_override: OptimizerDetails | None = None,
    ):
        sample = super().generate_batch(
            batch_size=batch_size,
            value=value,
            use_operation=use_operation,
            operation_override=operation_override,
        )

        # Storage
        sample = train_to_eval(
            env=sample, cfg=self.scenario, constraint_fn=self.post_process_fn
        )

        return sample.numpy(force=True)

    def shape(self, batch_size: int | None = None):
        B = batch_size or self.batch_size
        return (B, self.scenario.n_turbines, 2)
