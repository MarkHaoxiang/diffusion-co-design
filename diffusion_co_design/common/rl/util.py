from typing import Callable, Literal

import torch
import torch.optim as o
from torchrl.envs import ParallelEnv, SerialEnv
from torchrl.envs.batched_envs import BatchedEnvBase

from diffusion_co_design.common.env import ScenarioConfig
from diffusion_co_design.common.design import DesignConsumer


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
    assert cls is not None, "No optimizers provided."
    return cls(params)  # type: ignore


def make_optimiser_and_lr_scheduler(
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    start_actor_lr,
    start_critic_lr,
    n_train_iters: int,
    min_actor_lr: float = 0,
    min_critic_lr: float = 0,
    enable_lr_scheduler: bool = True,
):
    actor_optim = o.Adam(actor.parameters(), start_actor_lr)
    critic_optim = o.Adam(critic.parameters(), start_critic_lr)

    if enable_lr_scheduler:
        actor_scheduler = o.lr_scheduler.CosineAnnealingLR(
            optimizer=actor_optim, T_max=n_train_iters, eta_min=min_actor_lr
        )
        critic_scheduler = o.lr_scheduler.CosineAnnealingLR(
            optimizer=critic_optim, T_max=n_train_iters, eta_min=min_critic_lr
        )

        def scheduler_step():
            actor_scheduler.step()
            critic_scheduler.step()
            return actor_scheduler.get_last_lr() + critic_scheduler.get_last_lr()

    else:

        def scheduler_step():
            return list((start_actor_lr, start_critic_lr))

    def optim_step():
        actor_optim.step()
        critic_optim.step()
        actor_optim.zero_grad()
        critic_optim.zero_grad()

    return optim_step, scheduler_step


def create_batched_env[SC: ScenarioConfig](
    create_env: Callable[..., ParallelEnv],
    mode: Literal["train", "eval", "reference"],
    num_environments: int,
    scenario: SC,
    designer: DesignConsumer,
    batch_mode: Literal["serial", "parallel"] = "parallel",
    device: str | None = None,
) -> BatchedEnvBase:
    def create_env_fn(render: bool = False):
        return create_env(
            mode,
            scenario=scenario,
            designer=designer,
            render=render,
            device=torch.device("cpu"),
        )

    eval_kwargs = [{"render": True}]
    for _ in range(num_environments - 1):
        eval_kwargs.append({})

    if batch_mode == "parallel":
        return ParallelEnv(
            num_workers=num_environments,
            create_env_fn=create_env_fn,
            create_env_kwargs=eval_kwargs if mode == "eval" else {},
            device=device,
        )
    elif batch_mode == "serial":
        return SerialEnv(
            num_workers=num_environments,
            create_env_fn=create_env_fn,
            create_env_kwargs=eval_kwargs if mode == "eval" else {},
            device=device,
        )
