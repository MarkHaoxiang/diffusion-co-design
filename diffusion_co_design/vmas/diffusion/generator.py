import torch

from diffusion_co_design.vmas.model.diffusion import diffusion_setup
from diffusion_co_design.vmas.schema import (
    ScenarioConfigType,
    LocalPlacementScenarioConfig,
    GlobalPlacementScenarioConfig,
)
from diffusion_co_design.common.design import OptimizerDetails, BaseGenerator
from diffusion_co_design.common.design.generate import (
    soft_projection_constraint as _soft_projection_constraint,
)


def train_to_eval(env: torch.Tensor, cfg: ScenarioConfigType):
    env = env.clone()
    if isinstance(cfg, LocalPlacementScenarioConfig):
        return env
    elif isinstance(cfg, GlobalPlacementScenarioConfig):
        env[:, :, 0] = env[:, :, 0] * cfg.world_spawning_x
        env[:, :, 1] = env[:, :, 1] * cfg.world_spawning_y
        return env
    else:
        assert False


def eval_to_train(env: torch.Tensor, cfg: ScenarioConfigType):
    env = env.clone()
    if isinstance(cfg, LocalPlacementScenarioConfig):
        return env
    elif isinstance(cfg, GlobalPlacementScenarioConfig):
        env[:, :, 0] = env[:, :, 0] / cfg.world_spawning_x
        env[:, :, 1] = env[:, :, 1] / cfg.world_spawning_y
        return env
    else:
        assert False


def soft_projection_constraint(
    cfg: ScenarioConfigType, projection_steps: int = 30, penalty_lr: float = 0.02
):
    agent_pos = torch.tensor(cfg.agent_spawns)
    goal_pos = torch.tensor(cfg.agent_goals)
    agent_radius = 0.05  # For both agents and goals
    existing_pos = torch.cat([agent_pos, goal_pos], dim=0)  # [N, 2]
    existing_radius = torch.tensor([agent_radius] * len(existing_pos))
    obstacle_sizes = torch.tensor(cfg.obstacle_sizes).unsqueeze(0)

    def _projection_constraint(pos):
        B, M, _ = pos.shape
        return _soft_projection_constraint(
            existing_pos=existing_pos.to(pos.device),
            existing_radius=existing_radius.to(pos.device),
            pos=pos,
            radii=obstacle_sizes.expand(B, M).to(pos.device),
            x_scale=cfg.world_spawning_x,
            y_scale=cfg.world_spawning_y,
            projection_steps=projection_steps,
            penalty_lr=penalty_lr,
        )

    return _projection_constraint


class Generator(BaseGenerator):
    def __init__(
        self,
        generator_model_path: str,
        scenario: ScenarioConfigType,
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
        return (B, *self.scenario.diffusion_shape)
