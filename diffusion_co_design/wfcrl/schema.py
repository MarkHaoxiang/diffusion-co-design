from diffusion_co_design.common import (
    Config,
    PPOConfig,
    LoggingConfig,
    MEMORY_MANAGEMENT,
)


class ScenarioConfig(Config):
    n_turbines: int
    max_steps: int


class RLConfig(Config):
    # MLP
    policy_hidden_size: int = 64
    policy_depth: int = 2
    critic_hidden_size: int = 128
    critic_depth: int = 2
    mappo: bool = True


class TrainingConfig(Config):
    experiment_name: str
    memory_management: MEMORY_MANAGEMENT
    scenario: ScenarioConfig
    policy: RLConfig
    ppo: PPOConfig
    logging: LoggingConfig
