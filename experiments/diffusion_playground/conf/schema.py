from typing import Any, Literal
from pydantic import BaseModel


class Model(BaseModel):
    name: str
    representation: Literal["graph", "image"]
    model_kwargs: dict[str, Any] = {}


class Config(BaseModel):
    training_dir: str
    model: Model
    batch_size: int
    train_epochs: int
    lr: float
    weight_decay: float
    logging_mode: Literal["online", "offline", "disabled"]
