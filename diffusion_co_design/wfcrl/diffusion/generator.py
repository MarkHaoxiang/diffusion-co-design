import torch

from diffusion_co_design.common.design import OptimizerDetails, BaseGenerator
from diffusion_co_design.common.nn import EnvCritic
from diffusion_co_design.wfcrl.schema import ScenarioConfig
from diffusion_co_design.wfcrl.model.diffusion import diffusion_setup
from diffusion_co_design.wfcrl.diffusion.constraints import slsqp_projection_constraint


def train_to_eval(env: torch.Tensor, cfg: ScenarioConfig, constraint_fn=None):
    env = env.clone()
    if constraint_fn:
        env = constraint_fn(env)
    env[:, :, 0] = (env[:, :, 0] + 1) / 2 * cfg.map_x_length
    env[:, :, 1] = (env[:, :, 1] + 1) / 2 * cfg.map_y_length
    return env


def eval_to_train(env: torch.Tensor, cfg: ScenarioConfig):
    env = env.clone()
    env[:, :, 0] = env[:, :, 0] / cfg.map_x_length * 2 - 1
    env[:, :, 1] = env[:, :, 1] / cfg.map_y_length * 2 - 1
    return env


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
        self.post_process_fn = slsqp_projection_constraint(scenario)

    def generate_batch(
        self,
        batch_size: int | None = None,
        value: EnvCritic | None = None,
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
