from abc import ABC, abstractmethod

from tensordict import TensorDict
from tensordict.nn import TensorDictModule
import torch.nn as nn
from torchrl.data import ReplayBuffer


class BaseDesigner(nn.Module, ABC):
    def __init__(self, environment_repeats: int = 1):
        super().__init__()
        self.update_counter = 0
        self.environment_repeats = environment_repeats
        self.environment_repeat_counter = 0
        self.previous_environment = None

    @abstractmethod
    def _generate_environment_weights(self, objective):
        raise NotImplementedError()

    def generate_environment_weights(self, objective=None):
        self.environment_repeat_counter += 1
        if self.environment_repeat_counter >= self.environment_repeats:
            self.environment_repeat_counter = 0
            self.previous_environment = None
        if self.previous_environment is not None:
            return self.previous_environment
        else:
            self.previous_environment = self._generate_environment_weights(objective)
            return self.previous_environment

    def to_td_module(self):
        return TensorDictModule(
            self,
            in_keys=[("environment_design", "objective")],
            out_keys=[("environment_design", "layout_weights")],
        )

    def update(self, sampling_td: TensorDict):
        self.update_counter += 1

    def reset(self, **kwargs):
        self.environment_repeat_counter = 0
        self.previous_environment = None

    def get_logs(self) -> dict:
        return {}

    def get_model(self) -> None | nn.Module:
        return None

    def get_training_buffer(self) -> None | ReplayBuffer:
        return None
