from pydantic import BaseModel


class PolicyConfig(BaseModel):
    # MLP
    policy_hidden_size: int = 64
    policy_depth: int = 2
    critic_hidden_size: int = 128
    critic_depth: int = 2
    mappo: bool = True


class TrainingConfig(BaseModel):
    experiment_name: str
