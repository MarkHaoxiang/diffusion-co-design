from __future__ import annotations

from omegaconf import DictConfig, OmegaConf


# https://www.gaohongnan.com/software_engineering/config_management/01-pydra.html#pydra
def omega_to_pydantic(config: DictConfig, config_cls):
    """Converts Hydra config to Pydantic config."""
    # use to_container to resolve
    config_dict = OmegaConf.to_object(config)  # type: ignore[assignment]
    return config_cls(**config_dict)
