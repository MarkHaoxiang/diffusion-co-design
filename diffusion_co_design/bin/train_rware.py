import os
from os.path import join
import time

# Torch, TorchRL, TensorDict
import hydra.core
import hydra.core.hydra_config
from tensordict import TensorDict
import torch
import torch.multiprocessing as mp
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
    LogTraining,
    omega_to_pydantic,
    init_logging,
    log_evaluation,
    MEMORY_MANAGEMENT,
    memory_management,
)
from diffusion_co_design.rware.env import ScenarioConfig, create_batched_env, create_env
from diffusion_co_design.rware.model import rware_models, PolicyConfig
from diffusion_co_design.rware.design import DesignerRegistry


class TrainingConfig(BaseModel):
    # Problem definition: Built with diffusion.datasets.rware.generate
    experiment_name: str
    designer: str
    scenario: ScenarioConfig
    # Sampling and training
    memory_management: MEMORY_MANAGEMENT = "gpu"
    n_iters: int = 500  # Number of training iterations
    n_epochs: int = 10  # Number of optimization steps per training iteration
    minibatch_size: int = 800  # Size of the mini-batches in each optimization step
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
    device = memory_management(cfg.memory_management)
    n_train_envs = cfg.frames_per_batch // cfg.scenario.max_steps
    master_designer, env_designer = DesignerRegistry.get(
        cfg.designer,
        cfg.scenario,
        output_dir,
        environment_batch_size=max(
            n_train_envs + cfg.logging.evaluation_episodes, n_train_envs * 2 + 1
        ),
        device=device.train_device,
    )
    # designer.share_memory()

    placeholder_env = create_env(
        cfg.scenario, env_designer, is_eval=False, device=device.env_device
    )
    assert n_train_envs * cfg.scenario.max_steps == cfg.frames_per_batch
    train_env = create_batched_env(
        num_environments=n_train_envs,
        designer=env_designer,
        scenario=cfg.scenario,
        is_eval=False,
        device=device.env_device,
    )
    eval_env = create_batched_env(
        num_environments=cfg.logging.evaluation_episodes,
        scenario=cfg.scenario,
        designer=env_designer,
        is_eval=True,
        device=device.env_device,
    )
    master_designer.reset()
    policy, critic = rware_models(
        placeholder_env, cfg.policy, device=device.train_device
    )

    collector = SyncDataCollector(
        train_env,
        policy,
        device=device.train_device,
        storing_device=device.storage_device,
        frames_per_batch=cfg.frames_per_batch,
        total_frames=cfg.total_frames,
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(cfg.frames_per_batch, device=device.storage_device),
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
        experiment_name=cfg.experiment_name,
        log_dir=output_dir,
        cfg=cfg.logging,
        full_config=cfg,
    )
    log_training = LogTraining()

    # Main Training Loop
    master_designer.reset()
    try:
        total_time, total_frames = 0.0, 0
        sampling_start = time.time()
        for iteration, sampling_td in enumerate(collector):
            sampling_time = time.time() - sampling_start
            training_tds, training_start = [], time.time()

            # Compute GAE
            loss_module.to(device=device.storage_device)
            with torch.no_grad():
                loss_module.value_estimator(
                    sampling_td,
                    params=loss_module.critic_network_params,
                    target_params=loss_module.target_critic_network_params,
                )
            loss_module.to(device=device.train_device)

            # Add to the replay buffer (shuffling)
            current_frames = sampling_td.numel()
            total_frames += current_frames
            replay_buffer.extend(sampling_td.reshape(-1))

            if logger:
                log_training.collect_sampling_td(sampling_td, sampling_time)

            # del sampling_td  # Clear now to reduce memory
            # torch.cuda.empty_cache()

            # PPO Update
            for _ in range(cfg.n_epochs):
                for _ in range(cfg.n_mini_batches):
                    minibatch: TensorDict = replay_buffer.sample()
                    loss_vals = loss_module(minibatch.to(device=device.train_device))
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

            # Designer (aka diffusion policy) update
            master_designer.update(sampling_td)

            training_time = time.time() - training_start
            total_time += sampling_time + training_time

            # Logging
            if logger:
                log_training.collect_training_td(
                    training_log_td, training_time, total_time
                )
                log_training.log(master_designer.get_logs())
                log_training.commit(logger, iteration, total_frames, current_frames)

                if (
                    cfg.logging.evaluation_episodes > 0
                    and iteration % cfg.logging.evaluation_interval == 0
                ):
                    evaluation_start = time.time()
                    with (
                        torch.no_grad(),
                    ):
                        frames = []

                        def callback(env, td):
                            return frames.append(env.render()[0])

                        rollouts = eval_env.rollout(
                            max_steps=cfg.scenario.max_steps,
                            policy=policy,
                            callback=callback,
                            auto_cast_to_device=True,
                        )

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
    train(omega_to_pydantic(config, TrainingConfig))


if __name__ == "__main__":
    run()
