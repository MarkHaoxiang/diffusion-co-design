import torch
from torchrl.objectives import ClipPPOLoss
from tensordict import TensorDict

from diffusion_co_design.common.pydra import Config


class PPOConfig(Config):
    n_iters: int  # Number of training iterations
    n_epochs: int  # Number of optimization steps per training iteration
    minibatch_size: int  # Size of the mini-batches in each optimization step
    n_mini_batches: int  # Number of mini-batches in each epoch
    clip_epsilon: float  # clip value for PPO loss
    gamma: float  # discount factor
    lmbda: float  # lambda for generalised advantage estimation
    actor_lr: float  # Learning rate for the actor
    critic_lr: float  # Learning rate for the critic

    max_grad_norm: float  # Maximum norm for the gradients
    entropy_eps: float  # coefficient of the entropy term in the PPO loss
    normalise_advantage: bool  # Whether to normalise the advantage estimates

    @property
    def frames_per_batch(self) -> int:
        return self.n_mini_batches * self.minibatch_size

    @property
    def total_frames(self) -> int:
        return self.frames_per_batch * self.n_iters


# https://github.com/pytorch/rl/blob/main/torchrl/objectives/utils.py#L602
def group_optimizers(*optimizers: torch.optim.Optimizer) -> torch.optim.Optimizer:
    """Groups multiple optimizers into a single one.

    All optimizers are expected to have the same type.
    """
    cls = None
    params = []
    for optimizer in optimizers:
        if optimizer is None:
            continue
        if cls is None:
            cls = type(optimizer)
        if cls is not type(optimizer):
            raise ValueError("Cannot group optimizers of different type.")
        params.extend(optimizer.param_groups)
    return cls(params)


def minibatch_advantage_calculation(
    sampling_td: TensorDict,
    loss_module: ClipPPOLoss,
    group_name: str,
    batch_size: int = 1,
    batch_dim: int = 0,
):
    shape = sampling_td.get(("next", group_name, "reward")).shape

    keys_list = [
        (group_name, "state_value"),
        ("next", group_name, "state_value"),
        "advantage",
        "value_target",
    ]

    buffer = [
        torch.zeros(shape, device=sampling_td.device) for _ in range(len(keys_list))
    ]

    with torch.no_grad():
        batch_size = 1
        for i, eb in enumerate(sampling_td.split(batch_size, dim=batch_dim)):
            loss_module.value_estimator(
                eb,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )

            start = i * batch_size
            end = min((i + 1) * batch_size, sampling_td.shape[0])
            for j, key in enumerate(keys_list):
                buffer[j][start:end] = eb.get(key)

    sampling_td.update({key: value for key, value in zip(keys_list, buffer)})
    return sampling_td
