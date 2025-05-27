import os
import copy
import time
import warnings

import hydra.core
import hydra.core.hydra_config
from tensordict import TensorDict
import torch
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, SamplerWithoutReplacement, LazyTensorStorage
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl.envs.utils import set_exploration_type, ExplorationType

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from tqdm import tqdm

from diffusion_co_design.common import (
    RLExperimentLogger,
    memory_management,
    start_from_checkpoint,
)
from diffusion_co_design.common.ppo import make_optimiser_and_lr_scheduler
from diffusion_co_design.wfcrl.schema import TrainingConfig
import diffusion_co_design.wfcrl.design as design
from diffusion_co_design.wfcrl.env import create_batched_env, create_env
from diffusion_co_design.wfcrl.model.rl import wfcrl_models

group_name = "turbine"


def train(cfg: TrainingConfig):
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    device = memory_management(cfg.device)
    n_train_envs = min(cfg.ppo.frames_per_batch // cfg.scenario.max_steps, 20)
    assert cfg.ppo.frames_per_batch % n_train_envs == 0
    assert (cfg.ppo.frames_per_batch // n_train_envs) % cfg.scenario.max_steps == 0

    master_designer, env_designer = design.DesignerRegistry.get(
        designer=cfg.designer,
        scenario=cfg.scenario,
        ppo_cfg=cfg.ppo,
        normalisation_statistics=cfg.normalisation,
        artifact_dir=output_dir,
        device=device.train_device,
    )
    environment_reset_num = n_train_envs + cfg.logging.evaluation_episodes + 2
    if cfg.designer.environment_repeats == 1:
        environment_reset_num += n_train_envs  # Collector construction reset
    master_designer.reset(batch_size=environment_reset_num)

    reference_env = create_env(
        mode="reference",
        scenario=cfg.scenario,
        designer=env_designer,
        device=device.env_device,
    )
    train_env = create_batched_env(
        mode="train",
        designer=env_designer,
        num_environments=n_train_envs,
        scenario=cfg.scenario,
        device=device.env_device,
    )
    env_designer = copy.copy(env_designer)
    env_designer.environment_repeats = 1
    eval_env = create_batched_env(
        mode="eval",
        designer=env_designer,
        num_environments=cfg.logging.evaluation_episodes,
        scenario=cfg.scenario,
        device=device.env_device,
    )

    policy, critic = wfcrl_models(
        reference_env,
        cfg.policy,
        cfg.normalisation,
        device=device.train_device,
    )

    if isinstance(master_designer, design.DiskDesigner):
        core = master_designer.master_designer
        if isinstance(core, design.ValueDesigner):
            core.critic = critic
            core.ref_env = create_env(
                mode="reference",
                scenario=cfg.scenario,
                designer=design.FixedDesigner(cfg.scenario),
                device=device.env_device,
            )

    collector = SyncDataCollector(
        train_env,
        policy,
        device=device.train_device,
        storing_device=device.storage_device,
        frames_per_batch=cfg.ppo.frames_per_batch,
        total_frames=cfg.ppo.total_frames,
        exploration_type=ExplorationType.RANDOM,
    )

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
        entropy_bonus=True if cfg.ppo.entropy_eps > 0 else False,
        entropy_coef=cfg.ppo.entropy_eps,
        normalize_advantage=cfg.ppo.normalise_advantage,
        normalize_advantage_exclude_dims=(-2,),
    )

    loss_module.set_keys(
        reward=train_env.reward_key,
        action=train_env.action_key,
        sample_log_prob=(group_name, "sample_log_prob"),
        value=(group_name, "state_value"),
        terminated=(group_name, "terminated"),
        done=(group_name, "done"),
    )

    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=cfg.ppo.gamma, lmbda=cfg.ppo.lmbda
    )
    optim_step, scheduler_step = make_optimiser_and_lr_scheduler(
        actor=policy, critic=critic, cfg=cfg.ppo
    )

    # Logging
    pbar = tqdm(total=cfg.ppo.n_iters)

    logger = RLExperimentLogger(
        directory=output_dir,
        experiment_name=cfg.experiment_name,
        project_name="diffusion-co-design-wfcrl",
        group_name=group_name,
        config=cfg.dump(),
        mode=cfg.logging.mode,
    )

    start_from_checkpoint(
        training_dir=cfg.start_from_checkpoint,
        models=[
            (policy, "policy_"),
            (critic, "critic_"),
        ],
    )

    master_designer.reset(batch_size=n_train_envs)
    try:
        logger.begin()
        total_time, total_frames = 0.0, 0
        sampling_start = time.time()
        for iteration, sampling_td in enumerate(collector):
            sampling_time = time.time() - sampling_start
            training_tds, training_start = [], time.time()
            logger.collect_sampling_td(sampling_td)

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
            policy.train()
            critic.train()
            for _ in range(cfg.ppo.n_epochs):
                for _ in range(cfg.ppo.n_mini_batches):
                    minibatch: TensorDict = replay_buffer.sample()

                    loss_vals = loss_module(minibatch.to(device=device.train_device))
                    loss_value = loss_vals["loss_objective"] + loss_vals["loss_critic"]
                    if cfg.ppo.entropy_eps > 0:
                        loss_value += loss_vals["loss_entropy"]

                    # NaN check
                    if torch.isnan(loss_value) or torch.isinf(loss_value):
                        warnings.warn("NaN or Inf detected in loss value")
                        torch.save(sampling_td, "nan_inf_sampling_td.pt")
                        torch.save(minibatch, "nan_inf_minibatch.pt")
                        assert False

                    loss_value.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        loss_module.parameters(), cfg.ppo.max_grad_norm
                    )
                    optim_step()

                    training_log_td = loss_vals.detach()
                    training_log_td.set("grad_norm", grad_norm.mean())
                    training_tds.append(loss_vals.detach())

            collector.update_policy_weights_()
            logger.collect_training_td(training_log_td)

            policy.eval()
            critic.eval()

            design_start = time.time()
            master_designer.update(sampling_td)
            design_time = time.time() - design_start

            training_time = time.time() - training_start
            total_time += sampling_time + training_time

            # Logging
            logger.collect_times(sampling_time, training_time, design_time, total_time)
            logger.log(master_designer.get_logs())

            if (
                cfg.logging.evaluation_episodes > 0
                and iteration % cfg.logging.evaluation_interval == 0
            ):
                evaluation_start = time.time()
                if isinstance(master_designer, design.DiskDesigner):
                    master_designer.force_regenerate(
                        batch_size=cfg.logging.evaluation_episodes * 2,
                        mode="eval",
                    )
                with (
                    torch.no_grad(),
                    set_exploration_type(ExplorationType.RANDOM),
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

            is_final_iteration = iteration == cfg.ppo.n_iters - 1
            if iteration % cfg.logging.checkpoint_interval == 0 or is_final_iteration:
                logger.checkpoint_state_dict(policy, f"policy_{iteration}")
                logger.checkpoint_state_dict(critic, f"critic_{iteration}")
                model = master_designer.get_model()
                buffer = master_designer.get_training_buffer()
                if model is not None:
                    logger.checkpoint_state_dict(model, f"designer_{iteration}")
                if buffer is not None:
                    buffer.dumps(
                        os.path.join(logger.checkpoint_dir, f"env-buffer_{iteration}")
                    )

            pbar.update()
            sampling_start = time.time()
            if isinstance(master_designer, design.DiskDesigner):
                master_designer.force_regenerate(batch_size=n_train_envs, mode="train")
    finally:
        # Cleaup
        logger.close()
        collector.shutdown()
        for env in (train_env, eval_env):
            if not env.is_closed:
                env.close()


@hydra.main(version_base=None, config_path="conf", config_name="random")
def run(config: DictConfig):
    print(f"Running job {HydraConfig.get().job.name}")
    train(TrainingConfig.from_raw(config))


if __name__ == "__main__":
    run()
