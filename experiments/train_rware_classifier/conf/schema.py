from typing import Literal

from diffusion_co_design.common.pydra import Config as _Config
from diffusion_co_design.rware.schema import ClassifierConfig


class Config(_Config):
    training_dir: str
    model: ClassifierConfig
    batch_size: int
    train_epochs: int
    lr: float
    weight_decay: float
    logging_mode: Literal["online", "offline", "disabled"]
