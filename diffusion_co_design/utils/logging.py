# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted from
# https://github.com/pytorch/rl/blob/main/sota-implementations/multiagent/utils/logging.py#L32
#
from typing import Literal, Any

from pydantic import BaseModel
from tensordict import TensorDictBase
import torch
import numpy as np
from torchrl.record.loggers import Logger, WandbLogger, CSVLogger


LoggerTypes = Literal["wandb", "csv"] | None


class LoggingConfig(BaseModel):
    type: LoggerTypes = "wandb"
    wandb_mode: str = "online"
    evaluation_interval: int = 20
    evaluation_episodes: int = 5
    checkpoint_interval: int = 50


def init_logging(
    experiment_name: str,
    log_dir: str,
    cfg: LoggingConfig,
    full_config: Any | None = None,
):
    full_config = dict(full_config) if full_config is not None else {}

    match cfg.type:
        case "csv":
            logger = CSVLogger(
                exp_name=experiment_name,
                log_dir=log_dir,
            )
            logger.log_hparams(full_config)
        case "wandb":
            logger = WandbLogger(
                exp_name=experiment_name,
                project="diffusion-co-design",
                mode=cfg.wandb_mode,
                dir=log_dir,
                config=full_config,
            )
        case _:
            logger = None

    return logger


class LogTraining:
    def __init__(self):
        self.metrics_to_log = {}

    def collect_sampling_td(self, sampling_td: TensorDictBase, sampling_time: float):
        if "info" in sampling_td.get("agents").keys():
            self.metrics_to_log.update(
                {
                    f"train/info/{key}": value.mean().item()
                    for key, value in sampling_td.get(("agents", "info")).items()
                }
            )

        reward = sampling_td.get(("next", "agents", "reward")).mean(
            -2
        )  # Mean over agents
        done = sampling_td.get(("next", "done"))
        if done.ndim > reward.ndim:
            done = done[..., 0, :]  # Remove expanded agent dim
        episode_reward = sampling_td.get(("next", "agents", "episode_reward")).mean(-2)[
            done
        ]

        if episode_reward.numel() == 0:  # Prevent crash if no dones
            episode_reward = torch.zeros(())

        self.metrics_to_log.update(
            {
                "train/sampling_time": sampling_time,
                "train/reward/reward_min": reward.min().item(),
                "train/reward/reward_mean": reward.mean().item(),
                "train/reward/reward_max": reward.max().item(),
                "train/reward/episode_reward_min": episode_reward.min().item(),
                "train/reward/episode_reward_mean": episode_reward.mean().item(),
                "train/reward/episode_reward_max": episode_reward.max().item(),
            }
        )

    def collect_training_td(
        self, training_td: TensorDictBase, training_time: float, total_time: float
    ):
        self.metrics_to_log.update(
            {
                f"train/learner/{key}": value.mean().item()
                for key, value in training_td.items()
            }
        )
        self.metrics_to_log.update(
            {
                "train/training_time": training_time,
                "train/total_time": total_time,
            }
        )

    def commit(self, logger, step, total_frames, current_frames):
        metrics_to_log = self.metrics_to_log
        metrics_to_log.update(
            {
                "train/current_frames": current_frames,
                "train/total_frames": total_frames,
            }
        )
        if isinstance(logger, WandbLogger):
            logger.experiment.log(metrics_to_log, commit=False)
        else:
            for key, value in metrics_to_log.items():
                logger.log_scalar(key.replace("/", "_"), value, step=step)
        self.metrics_to_log.clear()


def log_evaluation(
    logger: WandbLogger,
    frames,
    rollouts: TensorDictBase,
    evaluation_time: float,
    step: int,
):
    rewards = rollouts.get(("next", "agents", "reward")).sum(dim=1).mean(dim=1)
    metrics_to_log = {
        "eval/episode_reward_min": min(rewards),
        "eval/episode_reward_max": max(rewards),
        "eval/episode_reward_mean": sum(rewards) / len(rollouts),
        "eval/episode_len_mean": sum([td.batch_size[0] for td in rollouts])
        / len(rollouts),
        "eval/evaluation_time": evaluation_time,
    }

    vid = torch.tensor(
        np.transpose(frames[: rollouts[0].batch_size[0]], (0, 3, 1, 2)),
        dtype=torch.uint8,
    ).unsqueeze(0)

    if isinstance(logger, WandbLogger):
        import wandb

        logger.experiment.log(metrics_to_log, commit=False)
        logger.experiment.log(
            {
                "eval/video": wandb.Video(vid, fps=10, format="mp4"),
            },
            commit=False,
        )
    else:
        for key, value in metrics_to_log.items():
            logger.log_scalar(key.replace("/", "_"), value, step=step)
        logger.log_video("eval_video", vid, step=step)

    # if ("next", "agents", "reward") not in sampling_td.keys(True, True):
    #     sampling_td.set(
    #         ("next", "agents", "reward"),
    #         sampling_td.get(("next", "reward"))
    #         .expand(sampling_td.get("agents").shape)
    #         .unsqueeze(-1),
    #     )
    # if ("next", "agents", "episode_reward") not in sampling_td.keys(True, True):
    #     sampling_td.set(
    #         ("next", "agents", "episode_reward"),
    #         sampling_td.get(("next", "episode_reward"))
    #         .expand(sampling_td.get("agents").shape)
    #         .unsqueeze(-1),
    #     )
