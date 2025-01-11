import os
import time

# Torch, TorchRL, TensorDict
import torch
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, SamplerWithoutReplacement, LazyTensorStorage
from torchrl.objectives import ClipPPOLoss, ValueEstimators

# Config Management
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

# Rware
from tqdm import tqdm

from diffusion_co_design.utils import (
    omega_to_pydantic,
    BASE_DIR,
    LoggingConfig,
    init_logging,
    log_training,
    log_evaluation,
)
from diffusion_co_design.co_design.rware.env import (
    ScenarioConfig,
    rware_env,
)
from diffusion_co_design.co_design.rware.model import rware_models, PolicyConfig


# Devices
device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")


class TrainingConfig(BaseModel):
    # Problem definition: Built with diffusion.datasets.rware.generate
    designer: str
    experiment_name: (
        str  # TODO: Rethink the config linkup between generation and co_design
    )
    # Sampling and training
    n_iters: int = 100  # Number of training iterations
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
    policy_cfg: PolicyConfig = PolicyConfig()
    # Logging
    logging_cfg: LoggingConfig = LoggingConfig()

    @property
    def frames_per_batch(self) -> int:
        return self.n_mini_batches * self.minibatch_size

    @property
    def total_frames(self) -> int:
        return self.frames_per_batch * self.n_iters


def train(cfg: TrainingConfig):
    # Load scenario config
    scenario = OmegaConf.load(
        os.path.join(BASE_DIR, "diffusion_datasets", cfg.experiment_name, "config.yaml")
    )
    scenario: ScenarioConfig = omega_to_pydantic(scenario, ScenarioConfig)

    train_env = rware_env(scenario, is_eval=False, device=device)
    eval_env = rware_env(scenario, is_eval=True, device=device)
    policy, critic = rware_models(train_env, cfg.policy_cfg, device)

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
    logger = init_logging(cfg.experiment_name, cfg.logging_cfg)

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
                    cfg.logging_cfg.evaluation_episodes > 0
                    and iteration % cfg.logging_cfg.evaluation_interval == 0
                ):
                    evaluation_start = time.time()
                    with (
                        torch.no_grad(),
                    ):  # TODO: I don't think we want determinism for rware - discuss.
                        frames = []
                        rollouts = []
                        for i in range(cfg.logging_cfg.evaluation_episodes):
                            callback = lambda env, td: (
                                frames.append(env.render()) if i == 0 else None
                            )
                            rollout = eval_env.rollout(
                                max_steps=(
                                    1000
                                    if train_env.max_steps is None
                                    else train_env.max_steps
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

            pbar.update()
            sampling_start = time.time()
    finally:
        # Cleaup
        collector.shutdown()
        for env in (train_env, eval_env):
            if not env.is_closed:
                env.close()


@hydra.main(version_base=None, config_path="configs", config_name="testing")
def run(config: DictConfig):
    print(f"Running job {HydraConfig.get().job.name}")
    config = omega_to_pydantic(config, TrainingConfig)
    train(config)


if __name__ == "__main__":
    run()
