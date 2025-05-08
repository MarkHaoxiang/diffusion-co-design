from typing import Literal

from diffusion_co_design.common import Config, PPOConfig, LoggingConfig, DeviceConfig


class ScenarioConfig(Config):
    n_turbines: int
    max_steps: int
    map_x_length: int
    map_y_length: int
    min_distance_between_turbines: int


class RLConfig(Config):
    model_type: Literal["mlp", "gnn"]
    # MLP
    mlp_hidden_size: int = 64
    mlp_depth: int = 2
    # GNN
    policy_node_hidden_size: int = 64
    policy_edge_hidden_size: int = 16
    policy_head_hidden_size: int = 64
    policy_gnn_depth: int = 3
    policy_head_depth: int = 2
    critic_node_hidden_size: int = 64
    critic_edge_hidden_size: int = 16
    critic_gnn_depth: int = 3


class TrainingConfig(Config):
    experiment_name: str
    device: DeviceConfig = DeviceConfig()
    normalize_reward: bool
    scenario: ScenarioConfig
    policy: RLConfig
    ppo: PPOConfig
    logging: LoggingConfig
