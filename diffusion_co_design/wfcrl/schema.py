import os
from typing import Annotated, Literal
from pydantic import Field

from diffusion_co_design.common import Config, DiffusionOperation
from diffusion_co_design.common.rl.mappo.schema import TrainingConfig as _TrainingConfig
from diffusion_co_design.common.rl.mappo.schema import ScenarioConfig as _ScenarioConfig
from diffusion_co_design.common.design import DesignerConfig as _Designer
from diffusion_co_design.common.nn.geometric import Connectivity, KNN

from diffusion_co_design.wfcrl.static import ENV_NAME


class ScenarioConfig(_ScenarioConfig):
    name: str
    n_turbines: int
    max_steps: int
    map_x_length: int
    map_y_length: int
    min_distance_between_turbines: int

    def get_name(self) -> str:
        return self.name

    def get_episode_steps(self) -> int:
        return self.max_steps

    def get_num_agents(self) -> int:
        return self.n_turbines


class NormalisationStatistics(Config):
    episode_return_mean: float
    episode_return_std: float
    reward_mean: float
    reward_std: float


class EnvCriticConfig(Config):
    node_emb_size: int = 64
    depth: int = 2
    connectivity: Connectivity = KNN(k=5)


# ====
# Designer registry


class Random(_Designer):
    kind: Literal["random"]


class Fixed(_Designer):
    kind: Literal["fixed"]


class _Value(_Designer):
    model: EnvCriticConfig
    batch_size: int = 64
    buffer_size: int = 2048
    lr: float = 3e-4
    n_update_iterations: int = 10
    clip_grad_norm: float | None = 1.0
    weight_decay: float = 0.0
    distill_enable: bool = False
    distill_samples: int = 5
    loss_criterion: Literal["mse", "huber"] = "huber"
    random_generation_early_start: int = 0
    train_early_start: int = 0


class Sampling(_Value):
    kind: Literal["sampling"]
    n_samples: int = 16


class Diffusion(_Value):
    kind: Literal["diffusion"]
    diffusion: DiffusionOperation


class Reinforce(_Designer):
    kind: Literal["reinforce"] = "reinforce"
    lr: float = 1e-4
    train_batch_size: int = 20
    train_epochs: int = 3


DesignerConfig = Annotated[
    Random | Fixed | Diffusion | Sampling | Reinforce, Field(discriminator="kind")
]

# ====


class _ActorCritic(Config):
    initial_std: float = 0.3


class MLPActorCriticConfig(_ActorCritic):
    model_type: Literal["mlp"] = "mlp"
    hidden_size: int = 64
    depth: int = 2


class GNNActorCriticConfig(_ActorCritic):
    model_type: Literal["gnn"] = "gnn"
    policy_node_hidden_size: int = 64
    policy_head_hidden_size: int = 64
    policy_gnn_depth: int = 3
    policy_head_depth: int = 2
    policy_connectivity: Connectivity = KNN(k=5)
    critic_node_hidden_size: int = 64
    critic_gnn_depth: int = 3
    critic_connectivity: Connectivity = KNN(k=5)


ActorCriticConfig = Annotated[
    MLPActorCriticConfig | GNNActorCriticConfig, Field(discriminator="model_type")
]


class TrainingConfig(
    _TrainingConfig[DesignerConfig, ScenarioConfig, ActorCriticConfig]
):
    @property
    def normalisation(self) -> NormalisationStatistics | None:
        if not hasattr(self, "_normalisation_cache"):
            file = os.path.join(self.scenario_folder, "normalisation_statistics.yaml")
            self._normalisation_cache = NormalisationStatistics.from_file(file)
        return self._normalisation_cache

    def dump(self) -> dict:
        out = super().dump()
        if self.normalisation is not None:
            out["normalisation_statistics"] = self.normalisation.model_dump()
        else:
            out["normalisation_statistics"] = None
        return out

    @property
    def env_name(self) -> str:
        return ENV_NAME

    @property
    def _scenario_cfg_cls(self) -> type[ScenarioConfig]:
        return ScenarioConfig
