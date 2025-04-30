from diffusion_co_design.common import (
    Config,
    PPOConfig,
    LoggingConfig,
    MEMORY_MANAGEMENT,
)


class ScenarioConfig(Config):
    n_turbines: int
    max_steps: int
    map_x_length: int
    map_y_length: int
    min_distance_between_turbines: int


class RLConfig(Config):
    # MLP
    node_hidden_size: int = 64
    edge_hidden_size: int = 16
    mlp_hidden_size: int = 64
    backbone_depth: int = 3
    head_depth: int = 2


class TrainingConfig(Config):
    experiment_name: str
    memory_management: MEMORY_MANAGEMENT
    normalize_reward: bool
    scenario: ScenarioConfig
    policy: RLConfig
    ppo: PPOConfig
    logging: LoggingConfig
