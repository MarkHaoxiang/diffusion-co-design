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
from torchrl.envs import EnvBase
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
    ReinforceDesigner,
)

from diffusion_co_design.common.pydra import Config
from diffusion_co_design.common.env import ScenarioConfig
from diffusion_co_design.common.device import memory_management
from diffusion_co_design.common.rl.util import (
    make_optimiser_and_lr_scheduler,
    create_batched_env,
)
from diffusion_co_design.common.design.base import LiveDesignConsumer
from diffusion_co_design.common.rl.mappo.schema import TrainingConfig, PPOConfig
from diffusion_co_design.common.logging import RLExperimentLogger
from diffusion_co_design.common.misc import start_from_checkpoint


class MAPPOCoDesign[
    DC: DesignerConfig,
    SC: ScenarioConfig,
    ACC: Config,
    TC: TrainingConfig,
](ABC):
    support_vmap = True

    def __init__(self, cfg: TC, project_name: str):
        super().__init__()
        self.cfg = cfg
        self.device = memory_management(cfg.device)
        self.project_name = project_name

    def run(self):
        cfg, device = self.cfg, self.device
        n_train_envs = cfg.n_train_envs
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        self.artifact_dir = Path(output_dir)
        designer, make_design_consumer = self.create_designer(
            scenario=cfg.scenario,
            designer=cfg.designer,
            ppo=cfg.ppo,
            artifact_dir=self.artifact_dir,
            device=self.device.train_device,
        )

        environment_reset_num = n_train_envs + cfg.logging.evaluation_episodes + 3

        if self.cfg.designer.environment_repeats == 1:
            environment_reset_num += n_train_envs + 1
        designer.replenish_layout_buffer(batch_size=environment_reset_num)

        placeholder_designer = LiveDesignConsumer(
            self.create_placeholder_designer(scenario=cfg.scenario)
        )

        def create_reference_env():
            return self.create_env(
                mode="reference",
                scenario=self.cfg.scenario,
                designer=placeholder_designer,
                device=self.device.env_device,
            )

        train_env = self.create_batched_env(
            make_design_consumer=make_design_consumer, n_envs=n_train_envs, mode="train"
        )

        eval_env = self.create_batched_env(
            make_design_consumer=make_design_consumer,
            n_envs=cfg.logging.evaluation_episodes,
            mode="eval",
        )

        ref_env = create_reference_env()

        policy, critic = self.create_actor_critic_models(
            reference_env=ref_env,
            actor_critic_config=cfg.policy,
            device=device.train_device,
        )

        if isinstance(designer, ValueDesigner):
            designer.value_learner.initialise_critic_distillation(
                critic=critic, ref_env=create_reference_env()
            )
        elif isinstance(designer, ReinforceDesigner):
            designer.initialise(
                train_env=self.create_batched_env(
                    make_design_consumer=make_design_consumer,
                    n_envs=n_train_envs,
                    mode="train",
                ),
                train_env_batch_size=n_train_envs,
                agent_policy=policy,
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
            ValueEstimators.GAE,
            gamma=cfg.ppo.gamma,
            lmbda=cfg.ppo.lmbda,
            deactivate_vmap=not self.support_vmap,
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
        training_episodes_per_batch = (
            cfg.ppo.frames_per_batch // cfg.scenario.get_episode_steps()
        )

        designer.replenish_layout_buffer(training_episodes_per_batch)
        try:
            logger.begin()
            total_time, total_frames = 0.0, 0
            sampling_start = time.time()
            for iteration, sampling_td in enumerate(collector):
                sampling_time = time.time() - sampling_start
                training_tds, training_start = [], time.time()

                # Compute GAE
                sampling_td = sampling_td.reshape(
                    -1, self.cfg.scenario.get_episode_steps()
                )
                assert sampling_td["next"][ref_env.done_keys[0]][:, -1].all()
                sampling_td = self.post_sample_hook(sampling_td)

                loss_module.to(device=device.storage_device)
                self.minibatch_advantage_calculation(
                    sampling_td=sampling_td, loss_module=loss_module
                )
                loss_module.to(device=device.train_device)

                # Add to the replay buffer (shuffling)
                current_frames = sampling_td.numel()
                total_frames += current_frames
                replay_buffer.extend(sampling_td.reshape(-1))

                logger.collect_sampling_td(sampling_td)

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
                logger.log(designer.get_logs(), path_prefix="train/designer")

                if (
                    cfg.logging.evaluation_episodes > 0
                    and iteration % cfg.logging.evaluation_interval == 0
                ):
                    evaluation_start = time.time()
                    designer.replenish_evaluation_set(cfg.logging.evaluation_episodes)
                    with (
                        torch.no_grad(),
                        set_exploration_type(ExplorationType.RANDOM),
                    ):
                        frames = []

                        def callback(env, td):
                            return frames.append(env.render()[0])

                        rollouts = eval_env.rollout(
                            max_steps=self.cfg.scenario.get_episode_steps(),
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
                designer.replenish_training_set(
                    training_episodes_per_batch=training_episodes_per_batch,
                    num_different_envs_in_parallel=n_train_envs,
                )
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

    def minibatch_advantage_calculation(
        self,
        sampling_td: TensorDict,
        loss_module: ClipPPOLoss,
        batch_size: int = 1,
        batch_dim: int = 0,
    ):
        group_name = self.group_name
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

    def post_sample_hook(self, sampling_td: TensorDict) -> TensorDict:
        return sampling_td

    def create_batched_env(self, make_design_consumer, n_envs, mode):
        return create_batched_env(
            create_env=self.create_env,
            mode=mode,
            designer=make_design_consumer(),
            num_environments=n_envs,
            scenario=self.cfg.scenario,
            device=self.device.env_device,
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
    ) -> tuple[torch.nn.Module, torch.nn.Module]:
        raise NotImplementedError()

    @abstractmethod
    def create_placeholder_designer(self, scenario: SC) -> Designer[SC]:
        """
        Create a placeholder designer for the environment. This is used for reference environments.
        """
        raise NotImplementedError()
