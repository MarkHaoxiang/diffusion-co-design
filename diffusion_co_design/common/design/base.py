from abc import ABC, abstractmethod
from multiprocessing.synchronize import Lock
from pathlib import Path
import pickle as pkl

from tensordict import TensorDict
from tensordict.nn import TensorDictModule
import torch.nn as nn

ENVIRONMENT_DESIGN_KEY = ("environment_design", "layout_weights")
BUFFER_FILENAME = "layout_buffer.pkl"


class _Designer(nn.Module):  # nn module to operate with tensordict
    def __init__(self, artifact_dir: Path, lock: Lock):
        super().__init__()
        self.lock = lock
        self.artifact_dir = artifact_dir
        if not self.artifact_dir.exists():
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.buffer_path = artifact_dir.joinpath(BUFFER_FILENAME)

    def pop_layout(self):
        with self.lock:
            with open(self.buffer_path, "rb") as f:
                buffer: list = pkl.load(f)
            if len(buffer) == 0:
                raise ValueError("Layout buffer is empty")
            layout = buffer.pop()
            with open(self.buffer_path, "wb") as f:
                pkl.dump(buffer, f)
        return layout

    def to_td_module(self):
        return TensorDictModule(
            self,
            in_keys=[],
            out_keys=[ENVIRONMENT_DESIGN_KEY],
        )

    def forward(self):
        return self.pop_layout()

    def _read_buffer_unsafe(self) -> list:
        with open(self.buffer_path, "rb") as f:
            buffer: list = pkl.load(f)
        return buffer

    def _write_buffer_unsafe(self, buffer: list):
        with open(self.buffer_path, "wb") as f:
            pkl.dump(buffer, f)


class DesignProducer(_Designer, ABC):
    def __init__(
        self,
        artifact_dir: Path,
        lock: Lock,
        environment_repeats: int = 1,
    ):
        super().__init__(artifact_dir, lock)

        self.environment_repeats = environment_repeats
        if not self.buffer_path.exists():
            self._write_buffer_unsafe([])
        self._write_buffer_unsafe([])

        self.update_counter = 0
        self.training_environment_buffer: list = []

    def update(self, sampling_td: TensorDict):
        self.update_counter += 1

    def reset(self):
        self.update_counter = 0
        self._clear_layout_buffer()

    def replenish_layout_buffer(self, batch_size: int, clear_buffer: bool = False):
        if clear_buffer:
            self._clear_layout_buffer()

        with self.lock:
            buffer = self._read_buffer_unsafe()

            generated_layouts = self.generate_layout_batch(batch_size)
            buffer.extend(generated_layouts)
            self._write_buffer_unsafe(buffer)

    def replenish_training_set(
        self, training_episodes_per_batch: int, num_different_envs_in_parallel: int
    ):
        while len(self.training_environment_buffer) < training_episodes_per_batch:
            # Replenish the training buffer
            generated_layouts = (
                self.generate_layout_batch(num_different_envs_in_parallel)
                * self.environment_repeats
            )
            self.training_environment_buffer.extend(generated_layouts)

        with self.lock:
            buffer = self._read_buffer_unsafe()
            assert len(buffer) == 0
            self._write_buffer_unsafe(
                self.training_environment_buffer[:training_episodes_per_batch]
            )
            self.training_environment_buffer = self.training_environment_buffer[
                training_episodes_per_batch:
            ]

    def replenish_evaluation_set(self, n_eval_episodes: int):
        self.replenish_layout_buffer(batch_size=n_eval_episodes)

    def _clear_layout_buffer(self):
        with self.lock:
            self._write_buffer_unsafe([])

    def get_logs(self) -> dict:
        return {}

    def get_state(self) -> dict:
        return {}

    def set_environment_repeats(self, n: int):
        self.environment_repeats = n

    @property
    def buffer_len(self) -> int:
        with self.lock:
            return len(self._read_buffer_unsafe())

    @abstractmethod
    def generate_layout_batch(self, batch_size: int) -> list:
        raise NotImplementedError()


class DesignConsumer(_Designer):
    pass


class LiveDesignConsumer(DesignConsumer):
    """ "A design consumer that uses the design producer to get layouts on demand."""

    def __init__(self, design_producer: DesignProducer):
        super().__init__(design_producer.artifact_dir, design_producer.lock)
        self.design_producer = design_producer

    def forward(self):
        return self.design_producer.generate_layout_batch(1)[0]
