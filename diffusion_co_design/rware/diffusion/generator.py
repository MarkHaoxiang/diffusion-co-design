import torch

from diffusion_co_design.common.design import OptimizerDetails, BaseGenerator
from diffusion_co_design.common.nn import EnvCritic

from diffusion_co_design.rware.schema import ScenarioConfig, Representation
from diffusion_co_design.rware.model.diffusion import diffusion_setup
from diffusion_co_design.rware.diffusion.transform import train_to_eval


class Generator(BaseGenerator):
    def __init__(
        self,
        generator_model_path: str,
        scenario: ScenarioConfig,
        representation: Representation,
        batch_size: int = 32,
        rng: torch.Generator | None = None,
        guidance_wt: float = 50.0,
        device: torch.device = torch.device("cpu"),
    ):
        self.model, self.diffusion = diffusion_setup(scenario, representation)
        super().__init__(
            generator_model_path=generator_model_path,
            model=self.model,
            diffusion=self.diffusion,
            batch_size=batch_size,
            rng=rng,
            default_guidance_wt=guidance_wt,
            device=device,
        )

        self.size = scenario.size
        self.n_colors = scenario.n_colors
        self.n_shelves = scenario.n_shelves
        self.representation = representation
        self.scenario = scenario

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
            env=sample, cfg=self.scenario, representation=self.representation
        )

        return sample.numpy(force=True)

    def shape(self, batch_size: int | None = None):
        B = batch_size or self.batch_size
        if self.representation == "image":
            return (B, self.n_colors, self.size, self.size)
        elif self.representation == "flat":
            return (B, 2 * self.n_shelves, 1)
            # return (self.batch_size, (2 + self.n_colors) * self.n_shelves, 1)
        elif self.representation == "graph":
            return (B, self.n_shelves, 2)
