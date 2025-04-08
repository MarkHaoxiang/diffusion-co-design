from typing import Literal
from pydantic import BaseModel

from diffusion_co_design.rware.classifier import Model


class Config(BaseModel):
    training_dir: str
    model: Model
    batch_size: int
    train_epochs: int
    lr: float
    weight_decay: float
    logging_mode: Literal["online", "offline", "disabled"]
