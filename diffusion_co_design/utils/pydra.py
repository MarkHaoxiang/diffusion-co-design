from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel


# https://www.gaohongnan.com/software_engineering/config_management/01-pydra.html#pydra
def hydra_to_pydantic[C: BaseModel](config: DictConfig, config_cls: type[C]) -> C:
    """Converts Hydra config to Pydantic config."""
    # use to_container to resolve
    config_dict = OmegaConf.to_object(config)  # type: ignore[assignment]
    return config_cls(**config_dict)
