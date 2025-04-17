from diffusion_co_design.common.design import BaseDesigner
from diffusion_co_design.wfcrl.schema import ScenarioConfig


class Designer(BaseDesigner):
    def __init__(self, scenario: ScenarioConfig, environment_repeats: int = 1):
        super().__init__(environment_repeats=environment_repeats)
        self.scenario = scenario
