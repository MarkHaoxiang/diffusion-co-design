from abc import ABC, abstractmethod

import torch
from torch import nn
from tensordict.nn import TensorDictModule
from pydantic import BaseModel

from diffusion_co_design.pretrain.rware.transform import image_to_layout

from diffusion_co_design.pretrain.rware.generate import generate


class ScenarioConfig(BaseModel):
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

    @staticmethod
    def get(designer: str):
        match designer:
            case DesignerRegistry.FIXED:
                return FixedDesigner
            case DesignerRegistry.RANDOM:
                return RandomDesigner
            case DesignerRegistry.RL:
                raise NotImplementedError()
            case DesignerRegistry.DIFFUSION:
                raise NotImplementedError()
            case _:
                raise ValueError(f"Unknown designer type {designer}")
