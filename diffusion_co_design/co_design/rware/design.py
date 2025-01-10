from enum import Enum

import torch
from torch import nn

from diffusion_co_design.diffusion.datasets.rware.generate import generate


class RandomDesigner(nn.Module):
    def __init__(
        self, size: int, n_shelves: int, agent_idxs: list[int], goal_idxs: list[int]
    ):
        super().__init__()
        self._generate_args = (size, n_shelves, agent_idxs, goal_idxs)

    def forward(self, _):
        env = torch.tensor(generate(*self._generate_args)[0])
        return env


class Designer(Enum):
    RANDOM = "random"
    RL = "rl"
    DIFFUSION = "diffusion"

    @staticmethod
    def get(designer: str):
        match designer:
            case Designer.RANDOM:
                return RandomDesigner
            case Designer.RL:
                raise NotImplementedError()
            case Designer.DIFFUSION:
                raise NotImplementedError()
            case _:
                raise ValueError("Unknown designer type")
