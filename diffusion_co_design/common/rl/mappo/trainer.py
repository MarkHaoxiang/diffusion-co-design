import os
import time
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Literal

import hydra
import hydra.core.hydra_config
from tensordict import TensorDict
import torch
from torchrl.envs import EnvBase, ParallelEnv
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, SamplerWithoutReplacement, LazyTensorStorage
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl.envs.utils import set_exploration_type, ExplorationType

from tqdm import tqdm

from diffusion_co_design.common.design import (
    DesignerConfig,
    Designer,
    DesignConsumer,
    ValueDesigner,
)

from diffusion_co_design.common.pydra import Config
from diffusion_co_design.common.env import ScenarioConfig
from diffusion_co_design.common.device import memory_management
from diffusion_co_design.common.rl.util import make_optimiser_and_lr_scheduler
from diffusion_co_design.common.rl.mappo.schema import TrainingConfig, PPOConfig
from diffusion_co_design.common.logging import RLExperimentLogger
from diffusion_co_design.common.misc import start_from_checkpoint


class MAPPOCoDesign[
    DC: DesignerConfig,
    SC: ScenarioConfig,
    ACC: Config,
    TC: TrainingConfig,
](ABC):
    def __init__(self, cfg: TC, project_name: str):
        super().__init__()
        self.cfg = cfg
        self.device = memory_management(cfg.device)
        self.project_name = project_name

    def run(self):
        cfg, device = self.cfg, self.device
        n_train_envs = cfg.n_train_envs
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        designer, make_design_consumer = self.create_designer(
            scenario=cfg.scenario,
            designer=cfg.designer,
            ppo=cfg.ppo,
            artifact_dir=Path(output_dir),
            device=self.device.train_device,
        )

        environment_reset_num = n_train_envs + cfg.logging.evaluation_episodes + 3

        if self.cfg.designer.environment_repeats == 1:
            environment_reset_num += n_train_envs + 1
        designer.replenish_layout_buffer(batch_size=environment_reset_num)

        def create_reference_env():
            return self.create_env(
                mode="reference",
                scenario=self.cfg.scenario,
                designer=make_design_consumer(),
                device=self.device.env_device,
            )

        train_env = self.create_batched_env(
            mode="train",
            designer=make_design_consumer(),
            num_environments=n_train_envs,
            scenario=cfg.scenario,
            device=device.env_device,
        )
        eval_env = self.create_batched_env(
            mode="eval",
            designer=make_design_consumer(),
            num_environments=cfg.logging.evaluation_episodes,
            scenario=cfg.scenario,
            device=device.env_device,
        )

        policy, critic = self.create_actor_critic_models(
            reference_env=create_reference_env(),
            actor_critic_config=cfg.policy,
            device=device.train_device,
        )

        if isinstance(designer, ValueDesigner):
            designer.value_learner.initialise_critic_distillation(
                critic=critic, ref_env=create_reference_env()
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
            sample_log_prob=(self.group_name, "sample_log_prob"),
            value=(self.group_name, "state_value"),
            terminated=(self.group_name, "terminated"),
            done=(self.group_name, "done"),
        )

        loss_module.make_value_estimator(
            ValueEstimators.GAE, gamma=cfg.ppo.gamma, lmbda=cfg.ppo.lmbda
        )
        optim_step, scheduler_step = make_optimiser_and_lr_scheduler(
            actor=policy,
            critic=critic,
            start_actor_lr=cfg.ppo.actor_lr,
            start_critic_lr=cfg.ppo.critic_lr,
            min_actor_lr=cfg.ppo.min_actor_lr,
            min_critic_lr=cfg.ppo.min_critic_lr,
            enable_lr_scheduler=cfg.ppo.lr_scheduler_enabled,
            n_train_iters=cfg.ppo.n_iters,
        )

        # Logging
        pbar = tqdm(total=cfg.ppo.n_iters)

        logger = RLExperimentLogger(
            directory=output_dir,
            experiment_name=cfg.experiment_name,
            project_name=self.project_name,
            group_name=self.group_name,
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

        designer.reset()
        designer.replenish_layout_buffer(n_train_envs)
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

                        loss_vals = loss_module(
                            minibatch.to(device=device.train_device)
                        )
                        loss_value = (
                            loss_vals["loss_objective"] + loss_vals["loss_critic"]
                        )
                        if cfg.ppo.entropy_eps > 0:
                            loss_value += loss_vals["loss_entropy"]

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
                logger.log(
                    {
                        "state_value_mean": sampling_td.get(
                            (self.group_name, "state_value")
                        )
                        .mean()
                        .item()
                    }
                )

                policy.eval()
                critic.eval()

                design_start = time.time()
                designer.update(sampling_td)
                design_time = time.time() - design_start

                training_time = time.time() - training_start
                total_time += sampling_time + training_time

                # Logging
                logger.collect_times(
                    sampling_time, training_time, design_time, total_time
                )
                logger.log(designer.get_logs())

                if (
                    cfg.logging.evaluation_episodes > 0
                    and iteration % cfg.logging.evaluation_interval == 0
                ):
                    evaluation_start = time.time()
                    designer.replenish_layout_buffer(
                        batch_size=cfg.logging.evaluation_episodes * 2
                    )
                    designer.set_environment_repeats(1)
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
                    designer.set_environment_repeats(cfg.designer.environment_repeats)

                lr = scheduler_step()
                logger.log({"lr_actor": lr[0], "lr_critic": lr[1]})
                logger.commit(total_frames, current_frames)

                is_final_iteration = iteration == cfg.ppo.n_iters - 1
                if (
                    iteration % cfg.logging.checkpoint_interval == 0
                    or is_final_iteration
                ):
                    policy_model_path = logger.checkpoint_state_dict(
                        policy, f"policy_{iteration}"
                    )

                    critic_model_path = logger.checkpoint_state_dict(
                        critic, f"critic_{iteration}"
                    )
                    if is_final_iteration:
                        logger.upload_model(
                            model_path=policy_model_path, name="policy_final"
                        )
                        logger.upload_model(
                            model_path=critic_model_path, name="critic_final"
                        )
                    designer_state = designer.get_state()
                    if "model" in designer_state:
                        designer_model_path = logger.checkpoint_state_dict(
                            designer_state.get("model"), f"designer_{iteration}"
                        )
                        if is_final_iteration:
                            logger.upload_model(
                                model_path=designer_model_path, name="designer_final"
                            )

                    if "buffer" in designer_state:
                        designer_model_path.get("buffer").dumps(
                            os.path.join(
                                logger.checkpoint_dir, f"env-buffer_{iteration}"
                            )
                        )

                pbar.update()
                sampling_start = time.time()
                designer.replenish_layout_buffer(batch_size=n_train_envs)
        finally:
            # Cleaup
            logger.close()
            collector.shutdown()
            for env in (train_env, eval_env):
                if not env.is_closed:
                    env.close()

    @property
    @abstractmethod
    def group_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def create_designer(
        self,
        scenario: SC,
        designer: DC,
        ppo: PPOConfig,
        artifact_dir: Path,
        device: torch.device,
    ) -> tuple[Designer[SC], Callable[[], DesignConsumer]]:
        raise NotImplementedError()

    def create_batched_env(
        self,
        num_environments: int,
        scenario: SC,
        designer: DesignConsumer,
        mode: Literal["train", "eval", "reference"],
        device: str | None = None,
    ) -> ParallelEnv:
        def create_env_fn(render: bool = False):
            return self.create_env(
                mode,
                scenario=scenario,
                designer=designer,
                render=render,
                device=torch.device("cpu"),
            )

        eval_kwargs = [{"render": True}]
        for _ in range(num_environments - 1):
            eval_kwargs.append({})

        return ParallelEnv(
            num_workers=num_environments,
            create_env_fn=create_env_fn,
            create_env_kwargs=eval_kwargs if mode == "eval" else {},
            device=device,
        )

    @abstractmethod
    def create_env(
        self,
        mode: Literal["train", "eval", "reference"],
        scenario: SC,
        designer: DesignConsumer,
        device: torch.device,
        render: bool = False,
    ) -> EnvBase:
        raise NotImplementedError()

    @abstractmethod
    def create_actor_critic_models(
        self,
        reference_env: EnvBase,
        actor_critic_config: ACC,
        device: torch.device,
    ):
        raise NotImplementedError()
