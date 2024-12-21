from dataclasses import dataclass
from typing import Literal

import torch

from diffusion_co_design.common.pydra import Config

DEVICE_MANAGEMENT = Literal[
    "gpu",  # Use GPU, except in data collection
    "cpu",  # All
    "mixed",  # Use CPU as the storing device
]


class DeviceConfig(Config):
    device_management: DEVICE_MANAGEMENT = "gpu"
    gpu_id: int = 0
    max_gpu_memory: float = 1.0


cuda = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")


@dataclass
class Device:
    env_device: torch.device
    storage_device: torch.device
    train_device: torch.device


def memory_management(device: DeviceConfig):
    if torch.cuda.is_available():
        gpu = torch.device(f"cuda:{device.gpu_id}")
        torch.cuda.set_per_process_memory_fraction(
            fraction=device.max_gpu_memory, device=gpu
        )
    else:
        gpu = torch.device("cpu")

    cpu = torch.device("cpu")
    match device.device_management:
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
            raise ValueError("Unknown option.")
