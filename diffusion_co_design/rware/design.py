from abc import ABC, abstractmethod
import os

import torch
from torch import nn
from tensordict.nn import TensorDictModule
from pydantic import BaseModel
from guided_diffusion.script_util import create_classifier, classifier_defaults

from diffusion_co_design.utils import OUTPUT_DIR
from diffusion_co_design.pretrain.rware.transform import image_to_layout
from diffusion_co_design.pretrain.rware.generate import generate
from diffusion_co_design.pretrain.rware.generator import Generator, GeneratorConfig


class ScenarioConfig(BaseModel):
    name: str
    size: int
    n_shelves: int
    agent_idxs: list[int]
    goal_idxs: list[int]


class Designer(nn.Module, ABC):
    def __init__(self, scenario: ScenarioConfig):
        super().__init__()
        self.scenario = scenario

    # TODO: Batch size support
    @abstractmethod
    def generate_environment_image(self, objective):
        raise NotImplementedError()

    def generate_environment(self, objective):
        return image_to_layout(self.generate_environment_image(objective))

    def to_td_module(self):
        return TensorDictModule(
            self,
            in_keys=[("environment_design", "objective")],
            out_keys=[("environment_design", "layout_image")],
        )


class RandomDesigner(Designer):
    def __init__(self, scenario: ScenarioConfig):
        super().__init__(scenario)
        self._generate_args = (
            scenario.size,
            scenario.n_shelves,
            scenario.agent_idxs,
            scenario.goal_idxs,
        )

    def forward(self, objective):
        env = torch.tensor(generate(*self._generate_args)[0])
        return env

    def generate_environment_image(self, objective):
        return self.forward(objective)


class DiffusionDesigner(Designer):
    def __init__(self, scenario: ScenarioConfig, device):
        super().__init__(scenario)
        model_dict = classifier_defaults()
        model_dict["image_size"] = scenario.size
        model_dict["image_channels"] = 3
        model_dict["classifier_width"] = 128
        model_dict["classifier_depth"] = 2
        model_dict["classifier_attention_resolutions"] = "16, 8, 4"
        model_dict["output_dim"] = 1
        self.model = create_classifier(**model_dict).to(device)
        gen_cfg = GeneratorConfig(
            generator_model_path=os.path.join(
                OUTPUT_DIR, "diffusion_pretrain", scenario.name, "model100000.pt"
            ),
            size=scenario.size,
        )
        self.generator = Generator(gen_cfg, guidance_wt=10.0)

    def generate_environment(self, objective):
        return self.generator.generate_batch()


class FixedDesigner(Designer):
    def __init__(self, scenario):
        super().__init__(scenario)
        self.layout_image = RandomDesigner(scenario).generate_environment_image(None)

    def forward(self, objective):
        return self.layout_image

    def generate_environment_image(self, objective):
        return self.layout_image


class DesignerRegistry:
    FIXED = "none"  # Good for testing
    RANDOM = "random"
    RL = "rl"
    DIFFUSION = "diffusion"
    DIFFUSION_SHARED = "diffusion_shared"

    @staticmethod
    def get(designer: str, scenario: ScenarioConfig):
        match designer:
            case DesignerRegistry.FIXED:
                return FixedDesigner(scenario)
            case DesignerRegistry.RANDOM:
                return RandomDesigner(scenario)
            case DesignerRegistry.RL:
                raise NotImplementedError()
            case DesignerRegistry.DIFFUSION(scenario):
                return DiffusionDesigner
            case DesignerRegistry.DIFFUSION_SHARED:
                raise NotImplementedError()
            case _:
                raise ValueError(f"Unknown designer type {designer}")
