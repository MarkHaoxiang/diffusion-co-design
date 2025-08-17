import torch

from diffusion_co_design.common.design import OptimizerDetails, BaseGenerator
from diffusion_co_design.vmas.model.diffusion import diffusion_setup
from diffusion_co_design.vmas.schema import ScenarioConfig


def train_to_eval(env: torch.Tensor, cfg: ScenarioConfig):
    env[:, :, 0] = env[:, :, 0] * cfg.world_spawning_x
    env[:, :, 1] = env[:, :, 1] * cfg.world_spawning_y
    return env


def eval_to_train(env: torch.Tensor, cfg: ScenarioConfig):
    env[:, :, 0] = env[:, :, 0] / cfg.world_spawning_x
    env[:, :, 1] = env[:, :, 1] / cfg.world_spawning_y
    return env


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

    pos[:, :, 0] = pos[:, :, 0] * x_scale
    pos[:, :, 1] = pos[:, :, 1] * y_scale

    existing_pos_exp = existing_pos.unsqueeze(0).expand(B, N, 2)  # [B, N, 2]
    existing_rad_exp = existing_radius.unsqueeze(0).expand(B, N)  # [B, N]

    all_pos = torch.cat([pos, existing_pos_exp], dim=1)  # [B, M+N, 2]
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
    penalty = violation[:, mask_upper].sum(dim=-1)

    collision_mask_new = (violation[:, :M, :] > 0).any(dim=2)  # [B, M]
    non_colliding_mask = ~collision_mask_new  # [B, M]
    deviation = (pos - original_pos).norm(dim=-1)  # [B, M]
    penalty_deviation = (deviation * non_colliding_mask.float()).sum(dim=-1) * 0.05
    penalty += penalty_deviation

    return penalty


def soft_projection_constraint(cfg: ScenarioConfig):
    agent_pos = torch.tensor(cfg.agent_spawns)
    goal_pos = torch.tensor(cfg.agent_goals)
    agent_radius = 0.05  # For both agents and goals
    existing_pos = torch.cat([agent_pos, goal_pos], dim=0)  # [N, 2]
    existing_radius = torch.tensor([agent_radius] * len(existing_pos))
    obstacle_sizes = torch.tensor(cfg.obstacle_sizes).unsqueeze(0)

    def _projection_constraint(pos):
        B, M, _ = pos.shape
        pos_0 = pos.clone()

        pos = (
            pos + torch.randn_like(pos) * 0.01
        )  # Add some noise to avoid corners problem, where positions exactly equal from clamping

        # Apply soft penalty
        with torch.enable_grad():
            pos.requires_grad_(True)
            optim = torch.optim.SGD([pos], lr=0.02)
            for _ in range(20):
                optim.zero_grad()
                penalty = soft_penalty(
                    existing_pos=existing_pos.to(pos.device),
                    existing_radius=existing_radius.to(pos.device),
                    pos=pos,
                    radii=obstacle_sizes.expand(B, M).to(pos.device),
                    original_pos=pos_0,
                    additional_collision_distance=0.005,
                    x_scale=cfg.world_spawning_x,
                    y_scale=cfg.world_spawning_y,
                ).sum()
                penalty.backward()
                optim.step()
                pos.data[..., 0] = pos.data[..., 0].clamp(min=-1, max=1)
                pos.data[..., 1] = pos.data[..., 1].clamp(min=-1, max=1)
            pos = pos.detach()

        return pos

    return _projection_constraint


class Generator(BaseGenerator):
    def __init__(
        self,
        generator_model_path: str,
        scenario: ScenarioConfig,
        batch_size: int = 32,
        rng: torch.Generator | None = None,
        default_guidance_wt: float = 50.0,
        device: torch.device = torch.device("cpu"),
    ):
        self.model, self.diffusion = diffusion_setup(scenario)
        super().__init__(
            generator_model_path=generator_model_path,
            model=self.model,
            diffusion=self.diffusion,
            batch_size=batch_size,
            rng=rng,
            default_guidance_wt=default_guidance_wt,
            device=device,
        )

        self.scenario = scenario

    def generate_batch(
        self,
        batch_size: int | None = None,
        value=None,
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
        sample = train_to_eval(env=sample, cfg=self.scenario)
        return sample

    def shape(self, batch_size: int | None = None):
        B = batch_size or self.batch_size
        return (B, self.scenario.get_num_agents(), 2)
