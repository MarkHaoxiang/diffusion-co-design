import os
from os.path import join
import time

import hydra.core
import hydra.core.hydra_config
from tensordict import TensorDict
import torch
from torchrl.trainers import RewardNormalizer
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, SamplerWithoutReplacement, LazyTensorStorage
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl.envs.utils import set_exploration_type, ExplorationType

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from tqdm import tqdm

from diffusion_co_design.common import RLExperimentLogger, memory_management
from diffusion_co_design.common.ppo import make_optimiser_and_lr_scheduler
from diffusion_co_design.wfcrl.schema import TrainingConfig
from diffusion_co_design.wfcrl.design import FixedDesigner, RandomDesigner
from diffusion_co_design.wfcrl.env import create_batched_env, create_env
from diffusion_co_design.wfcrl.model.rl import wfcrl_models

group_name = "turbine"


def train(cfg: TrainingConfig):
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    device = memory_management(cfg.device)
    n_train_envs = min(cfg.ppo.frames_per_batch // cfg.scenario.max_steps, 20)
    assert cfg.ppo.frames_per_batch % n_train_envs == 0
    assert (cfg.ppo.frames_per_batch // n_train_envs) % cfg.scenario.max_steps == 0

    # designer = RandomDesigner(cfg.scenario)
    designer = FixedDesigner(cfg.scenario, seed=0)

    reference_env = create_env(
        mode="reference",
        scenario=cfg.scenario,
        designer=designer,
        device=device.env_device,
    )
    train_env = create_batched_env(
        mode="train",
        designer=designer,
        num_environments=n_train_envs,
        scenario=cfg.scenario,
        device=device.env_device,
    )
    eval_env = create_batched_env(
        mode="eval",
        designer=designer,
        num_environments=cfg.logging.evaluation_episodes,
        scenario=cfg.scenario,
        device=device.env_device,
    )

    policy, critic = wfcrl_models(reference_env, cfg.policy, device=device.train_device)

    collector = SyncDataCollector(
        train_env,
        policy,
        device=device.train_device,
        storing_device=device.storage_device,
        frames_per_batch=cfg.ppo.frames_per_batch,
        total_frames=cfg.ppo.total_frames,
        exploration_type=ExplorationType.RANDOM,
        reset_at_each_iter=True,
    )
    if cfg.normalize_reward:
        reward_normalizer = RewardNormalizer(reward_key=reference_env.reward_key)

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(
            cfg.ppo.frames_per_batch, device=device.storage_device
        ),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.ppo.minibatch_size,
    )

    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=critic,
        clip_epsilon=cfg.ppo.clip_epsilon,
        entropy_coef=cfg.ppo.entropy_eps,
        normalize_advantage=cfg.ppo.normalise_advantage,
        normalize_advantage_exclude_dims=(-2,),
    )

    loss_module.set_keys(
        reward=train_env.reward_key,
        action=train_env.action_key,
        sample_log_prob=(group_name, "sample_log_prob"),
        value=(group_name, "state_value"),
        done=(group_name, "done"),
        terminated=(group_name, "terminated"),
    )

    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=cfg.ppo.gamma, lmbda=cfg.ppo.lmbda
    )
    optim, scheduler_step = make_optimiser_and_lr_scheduler(
        actor=policy, critic=critic, cfg=cfg.ppo
    )

    # Logging
    pbar = tqdm(total=cfg.ppo.n_iters)

    logger = RLExperimentLogger(
        directory=output_dir,
        experiment_name=cfg.experiment_name,
        project_name="diffusion-co-design-wfcrl",
        group_name=group_name,
        config=cfg.model_dump(),
        mode=cfg.logging.mode,
    )

    try:
        logger.begin()
        total_time, total_frames = 0.0, 0
        sampling_start = time.time()
        for iteration, sampling_td in enumerate(collector):
            sampling_time = time.time() - sampling_start
            training_tds, training_start = [], time.time()
            logger.collect_sampling_td(sampling_td)

            if cfg.normalize_reward:
                reward_normalizer.update_reward_stats(sampling_td["next"])
                sampling_td["next"] = reward_normalizer.normalize_reward(
                    sampling_td["next"]
                )
                logger.log(
                    {
                        "normalizer_mean": reward_normalizer._reward_stats[
                            "mean"
                        ].item(),
                        "normalizer_std": reward_normalizer._reward_stats["std"].item(),
                    }
                )

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

            # PPO Update
            for _ in range(cfg.ppo.n_epochs):
                for _ in range(cfg.ppo.n_mini_batches):
                    minibatch: TensorDict = replay_buffer.sample()
                    loss_vals = loss_module(minibatch.to(device=device.train_device))
                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )
                    loss_value.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        loss_module.parameters(), cfg.ppo.max_grad_norm
                    )
                    optim.step()
                    optim.zero_grad()

                    training_log_td = loss_vals.detach()
                    training_log_td.set("grad_norm", grad_norm.mean())
                    training_tds.append(loss_vals.detach())

            collector.update_policy_weights_()
            logger.collect_training_td(training_log_td)
            del minibatch, training_log_td
            torch.cuda.empty_cache()

            training_time = time.time() - training_start
            total_time += sampling_time + training_time

            # Logging
            logger.collect_times(sampling_time, training_time, 0, total_time)

            if (
                cfg.logging.evaluation_episodes > 0
                and iteration % cfg.logging.evaluation_interval == 0
            ):
                evaluation_start = time.time()
                with (
                    torch.no_grad(),
                    set_exploration_type(ExplorationType.DETERMINISTIC),
                ):
                    frames = []

                    def callback(env, td):
                        return frames.append(env.render()[0])

                    rollouts = eval_env.rollout(
                        max_steps=cfg.scenario.max_steps,
                        policy=policy,
                        callback=callback,
                        auto_cast_to_device=True,
                        break_when_all_done=True,
                    )

                    evaluation_time = time.time() - evaluation_start

                    logger.collect_evaluation_td(rollouts, evaluation_time, frames)

            lr = scheduler_step()
            logger.log({"lr_actor": lr[0], "lr_critic": lr[1]})
            logger.commit(total_frames, current_frames)

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
        logger.close()
        collector.shutdown()
        for env in (train_env, eval_env):
            if not env.is_closed:
                env.close()


@hydra.main(version_base=None, config_path="conf", config_name="default")
def run(config: DictConfig):
    print(f"Running job {HydraConfig.get().job.name}")
    train(TrainingConfig.from_raw(config))


if __name__ == "__main__":
    run()
