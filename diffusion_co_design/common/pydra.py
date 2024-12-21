from collections.abc import MutableMapping

from omegaconf import OmegaConf, DictConfig
from pydantic import BaseModel


class Config(BaseModel):
    @classmethod
    def from_raw(cls, config: MutableMapping):
        """
        Convert a dictionary to a Config object.
        """
        return cls(**config)

    @classmethod
    def from_file(cls, path: str, overrides: dict | None = None):
        """
        Convert a file to a Config object.
        """
        with open(path, "r") as f:
            raw = OmegaConf.load(f)
            if overrides is not None:
                raw = OmegaConf.merge(raw, overrides)
            assert isinstance(raw, DictConfig)
            config = cls.from_raw(raw)
        return config
