from typing import Literal

from diffusion_co_design.common.pydra import Config as _Config
from diffusion_co_design.rware.schema import EnvCriticConfig


class Config(_Config):
    training_dir: str
    model: EnvCriticConfig
    batch_size: int
    train_epochs: int
    lr: float
    weight_decay: float
    train_target: Literal["sampling", "critic"]
    logging_mode: Literal["online", "offline", "disabled"]
    experiment_suffix: str = ""
    enable_hint: bool
    hint_loss_weight: float
