import os
from abc import abstractmethod

from diffusion_co_design.common.constants import OUTPUT_DIR
from diffusion_co_design.common.pydra import Config
from diffusion_co_design.common.env import ScenarioConfig
from diffusion_co_design.common.device import DeviceConfig
from diffusion_co_design.common.design import DesignerConfig
from diffusion_co_design.common.logging import LoggingConfig

MAX_TRAIN_ENVS = 20


class PPOConfig(Config):
    n_iters: int  # Number of training iterations
    n_epochs: int  # Number of optimization steps per training iteration
    minibatch_size: int  # Size of the mini-batches in each optimization step
    n_mini_batches: int  # Number of mini-batches in each epoch
    clip_epsilon: float  # clip value for PPO loss
    gamma: float  # discount factor
    lmbda: float  # lambda for generalised advantage estimation
    actor_lr: float  # Learning rate for the actor
    min_actor_lr: float = 0  # Prevent decay past this LR for the actor
    critic_lr: float  # Learning rate for the critic
    min_critic_lr: float = 0  # Prevent decay past this LR for the critic
    lr_scheduler_enabled: bool  # Whether to use a learning rate scheduler

    max_grad_norm: float  # Maximum norm for the gradients
    entropy_eps: float  # coefficient of the entropy term in the PPO loss
    normalise_advantage: bool  # Whether to normalise the advantage estimates

    @property
    def frames_per_batch(self) -> int:
        return self.n_mini_batches * self.minibatch_size

    @property
    def total_frames(self) -> int:
        return self.frames_per_batch * self.n_iters


class TrainingConfig[
    DC: DesignerConfig,
    SC: ScenarioConfig,
    ACC: Config,
](Config):
    experiment_name: str
    device: DeviceConfig = DeviceConfig()
    scenario_name: str
    policy: ACC
    ppo: PPOConfig
    logging: LoggingConfig
    designer: DC
    start_from_checkpoint: str | None = None

    @property
    def scenario_folder(self) -> str:
        return os.path.join(OUTPUT_DIR, self.env_name, "scenario", self.scenario_name)

    @property
    def scenario(self) -> SC:
        if not hasattr(self, "_scenario_cache"):
            file = os.path.join(self.scenario_folder, "config.yaml")
            self._scenario_cache = self._scenario_cfg_cls.from_file(file)
        return self._scenario_cache

    def dump(self) -> dict:
        out = super().model_dump()
        out["scenario"] = self.scenario.model_dump()
        return out

    @property
    def n_train_envs(self) -> int:
        n_train_envs = min(
            self.ppo.frames_per_batch // self.scenario.get_episode_steps(),
            MAX_TRAIN_ENVS,
        )
        assert self.ppo.frames_per_batch % n_train_envs == 0
        assert (
            self.ppo.frames_per_batch // n_train_envs
        ) % self.scenario.get_episode_steps() == 0
        return n_train_envs

    @property
    @abstractmethod
    def env_name(self) -> str:
        raise NotImplementedError()

    @property
    @abstractmethod
    def _scenario_cfg_cls(self) -> type[ScenarioConfig]:
        raise NotImplementedError()
