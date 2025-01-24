import os
from os.path import join
import time

# Torch, TorchRL, TensorDict
import hydra.core
import hydra.core.hydra_config
import torch
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, SamplerWithoutReplacement, LazyTensorStorage
from torchrl.objectives import ClipPPOLoss, ValueEstimators

# Config Management
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pydantic import BaseModel

# Rware
from tqdm import tqdm

from diffusion_co_design.utils import (
    LoggingConfig,
    omega_to_pydantic,
    init_logging,
    log_training,
    log_evaluation,
)
from diffusion_co_design.rware.env import ScenarioConfig, create_batched_env, create_env
from diffusion_co_design.rware.model import rware_models, PolicyConfig


# Devices
device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")
print(f"Training on device {device}")


class TrainingConfig(BaseModel):
    # Problem definition: Built with diffusion.datasets.rware.generate
    experiment_name: str
    designer: str
    scenario: ScenarioConfig
    # Sampling and training
    n_iters: int = 500  # Number of training iterations
    n_epochs: int = 10  # Number of optimization steps per training iteration
    #    minibatch_size: int = 400  # Size of the mini-batches in each optimization step
    minibatch_size: int = 400  # Size of the mini-batches in each optimization step
    n_mini_batches: int = 20  # Number of mini-batches in each epoch
    ppo_lr: float = 5e-4  # Learning rate
    max_grad_norm: float = 1.0  # Maximum norm for the gradients
    # PPO
    clip_epsilon: float = 0.2  # clip value for PPO loss
    gamma: float = 0.99  # discount factor
    lmbda: float = 0.9  # lambda for generalised advantage estimation
    entropy_eps: float = 1e-4  # coefficient of the entropy term in the PPO loss
    # Policy
    policy: PolicyConfig = PolicyConfig()
    # Logging
    logging: LoggingConfig = LoggingConfig()

    @property
    def frames_per_batch(self) -> int:
        return self.n_mini_batches * self.minibatch_size

    @property
    def total_frames(self) -> int:
        return self.frames_per_batch * self.n_iters


def train(cfg: TrainingConfig):
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Load scenario config
    # train_env = create_env(cfg.scenario, is_eval=False, device=device)
    placeholder_env = create_env(cfg.scenario, is_eval=False, device=device)
    n_envs = cfg.frames_per_batch // placeholder_env.max_steps
    assert n_envs * placeholder_env.max_steps == cfg.frames_per_batch
    train_env = create_batched_env(
        num_environments=n_envs,
        scenario_cfg=cfg.scenario,
        is_eval=False,
        device=device,
    )
    eval_env = create_env(cfg.scenario, is_eval=True, device=device)
    policy, critic = rware_models(placeholder_env, cfg.policy, device)

    collector = SyncDataCollector(
        train_env,
        policy,
        device=device,
        storing_device=device,
        frames_per_batch=cfg.frames_per_batch,
        total_frames=cfg.total_frames,
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(cfg.frames_per_batch, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.minibatch_size,
    )

    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=critic,
        clip_epsilon=cfg.clip_epsilon,
        entropy_coef=cfg.entropy_eps,
        normalize_advantage=False,
    )

    loss_module.set_keys(
        reward=train_env.reward_key,
        action=train_env.action_key,
        sample_log_prob=("agents", "sample_log_prob"),
        value=("agents", "state_value"),
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )

    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=cfg.gamma, lmbda=cfg.lmbda
    )

    optim = torch.optim.Adam(loss_module.parameters(), cfg.ppo_lr)

    # Logging
    pbar = tqdm(total=cfg.n_iters)
    logger = init_logging(
        experiment_name=cfg.experiment_name, log_dir=output_dir, cfg=cfg.logging
    )

    # Main Training Loop

    try:
        total_time, total_frames = 0, 0
        sampling_start = time.time()
        for iteration, sampling_td in enumerate(collector):
            sampling_time = time.time() - sampling_start
            training_tds, training_start = [], time.time()

            # Compute GAE
            with torch.no_grad():
                loss_module.value_estimator(
                    sampling_td,
                    params=loss_module.critic_network_params,
                    target_params=loss_module.target_critic_network_params,
                )

            # Add to the replay buffer (shuffling)
            current_frames = sampling_td.numel()
            total_frames += current_frames
            replay_buffer.extend(sampling_td.reshape(-1))

            # PPO Update
            for _ in range(cfg.n_epochs):
                for _ in range(cfg.n_mini_batches):
                    loss_vals = loss_module(replay_buffer.sample())
                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )
                    loss_value.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        loss_module.parameters(), cfg.max_grad_norm
                    )
                    optim.step()
                    optim.zero_grad()

                    training_log_td = loss_vals.detach()
                    training_log_td.set("grad_norm", grad_norm.mean())
                    training_tds.append(loss_vals.detach())

            collector.update_policy_weights_()

            training_time = time.time() - training_start
            total_time += sampling_time + training_time

            # Logging
            if logger:
                log_training(
                    logger=logger,
                    training_td=training_log_td,
                    sampling_td=sampling_td,
                    sampling_time=sampling_time,
                    training_time=training_time,
                    total_time=total_time,
                    iteration=iteration,
                    current_frames=current_frames,
                    total_frames=total_frames,
                    step=iteration,
                )

                if (
                    cfg.logging.evaluation_episodes > 0
                    and iteration % cfg.logging.evaluation_interval == 0
                ):
                    evaluation_start = time.time()
                    with (
                        torch.no_grad(),
                    ):  # TODO: I don't think we want determinism for rware.
                        frames = []
                        rollouts = []
                        for i in range(cfg.logging.evaluation_episodes):
                            callback = lambda env, td: (
                                frames.append(env.render()) if i == 0 else None
                            )
                            rollout = eval_env.rollout(
                                max_steps=(
                                    1000
                                    if placeholder_env.max_steps is None
                                    else placeholder_env.max_steps
                                ),
                                policy=policy,
                                callback=callback,
                                auto_cast_to_device=True,
                            )
                            rollouts.append(rollout)

                        evaluation_time = time.time() - evaluation_start

                        log_evaluation(
                            logger,
                            frames,
                            rollouts,
                            evaluation_time,
                            step=iteration,
                        )

            if cfg.logging.type == "wandb":
                logger.experiment.log({}, commit=True)

            if iteration % cfg.logging.checkpoint_interval == 0:
                checkpoint_dir = join(output_dir, "checkpoints")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                torch.save(
                    policy.state_dict(),
                    join(checkpoint_dir, f"policy_{iteration}.pt"),
                )
                torch.save(
                    critic.state_dict(),
                    join(checkpoint_dir, f"critic_{iteration}.pt"),
                )

            pbar.update()
            sampling_start = time.time()
    finally:
        # Cleaup
        collector.shutdown()
        for env in (train_env, eval_env):
            if not env.is_closed:
                env.close()


@hydra.main(version_base=None, config_path="conf", config_name="default")
def run(config: DictConfig):
    print(f"Running job {HydraConfig.get().job.name}")
    config = omega_to_pydantic(config, TrainingConfig)
    train(config)


if __name__ == "__main__":
    run()
