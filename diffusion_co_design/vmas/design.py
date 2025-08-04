from diffusion_co_design.common import design, np_list_to_tensor_list
from diffusion_co_design.vmas.diffusion.generate import Generate
from diffusion_co_design.vmas.schema import ScenarioConfig as SC


class RandomDesigner(design.RandomDesigner[SC]):
    def __init__(
        self,
        designer_setting: design.DesignerParams[SC],
        seed: int | None = None,
    ):
        super().__init__(designer_setting=designer_setting)
        self.generate = Generate(scenario=self.scenario, rng=seed)

    def generate_random_layouts(self, batch_size):
        return np_list_to_tensor_list(
            self.generate(n=batch_size, training_dataset=False)
        )
