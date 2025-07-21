import torch
import torch.optim as o


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
