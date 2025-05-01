from dataclasses import dataclass
from typing import Literal

import torch

MEMORY_MANAGEMENT = Literal[
    "gpu",  # Use GPU, except in data collection
    "cpu",  # All
    "mixed",  # Use CPU as the storing device
]

cuda = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")


@dataclass
class Device:
    env_device: torch.device
    storage_device: torch.device
    train_device: torch.device


def memory_management(mem: MEMORY_MANAGEMENT, gpu_id: int = 0):
    gpu = (
        torch.device(f"cuda:{gpu_id}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    cpu = torch.device("cpu")
    match mem:
        case "gpu":
            return Device(
                env_device=gpu,
                storage_device=gpu,
                train_device=gpu,
            )
        case "cpu":
            return Device(env_device=cpu, storage_device=cpu, train_device=cpu)
        case "mixed":
            return Device(env_device=gpu, storage_device=cpu, train_device=gpu)
        case _:
            raise ValueError(f"Unknown option {mem}")
