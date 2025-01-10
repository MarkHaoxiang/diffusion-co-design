# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted from
# https://github.com/pytorch/rl/blob/main/sota-implementations/multiagent/utils/logging.py#L32
#

from pydantic import BaseModel
from tensordict import TensorDictBase
from torchrl.record.loggers import Logger, WandbLogger


class LoggingConfig(BaseModel):
    enable: bool = False
    offline: bool = True


def init_logging(experiment_name: str, cfg: LoggingConfig):
    if not cfg.enable:
        return None
    config = dict(cfg)
    logger = WandbLogger(
        exp_name=experiment_name,
        project="diffusion-co-design",
        offline=cfg.offline,
        config=config,
    )
    return logger


def log_training(
    logger: Logger,
    training_td: TensorDictBase,
    sampling_td: TensorDictBase,
    sampling_time: float,
    training_time: float,
    total_time: float,
    iteration: int,
    current_frames: int,
    total_frames: int,
    step: int,
):
    if ("next", "agents", "reward") not in sampling_td.keys(True, True):
        sampling_td.set(
            ("next", "agents", "reward"),
            sampling_td.get(("next", "reward"))
            .expand(sampling_td.get("agents").shape)
            .unsqueeze(-1),
        )
    if ("next", "agents", "episode_reward") not in sampling_td.keys(True, True):
        sampling_td.set(
            ("next", "agents", "episode_reward"),
            sampling_td.get(("next", "episode_reward"))
            .expand(sampling_td.get("agents").shape)
            .unsqueeze(-1),
        )

    metrics_to_log = {
        f"train/learner/{key}": value.mean().item()
        for key, value in training_td.items()
    }

    if "info" in sampling_td.get("agents").keys():
        metrics_to_log.update(
            {
                f"train/info/{key}": value.mean().item()
                for key, value in sampling_td.get(("agents", "info")).items()
            }
        )

    reward = sampling_td.get(("next", "agents", "reward")).mean(-2)  # Mean over agents
    done = sampling_td.get(("next", "done"))
    if done.ndim > reward.ndim:
        done = done[..., 0, :]  # Remove expanded agent dim
    episode_reward = sampling_td.get(("next", "agents", "episode_reward")).mean(-2)[
        done
    ]
    metrics_to_log.update(
        {
            "train/reward/reward_min": reward.min().item(),
            "train/reward/reward_mean": reward.mean().item(),
            "train/reward/reward_max": reward.max().item(),
            "train/reward/episode_reward_min": episode_reward.min().item(),
            "train/reward/episode_reward_mean": episode_reward.mean().item(),
            "train/reward/episode_reward_max": episode_reward.max().item(),
            "train/sampling_time": sampling_time,
            "train/training_time": training_time,
            "train/iteration_time": training_time + sampling_time,
            "train/total_time": total_time,
            "train/training_iteration": iteration,
            "train/current_frames": current_frames,
            "train/total_frames": total_frames,
        }
    )
    if isinstance(logger, WandbLogger):
        logger.experiment.log(metrics_to_log, commit=False)
    else:
        for key, value in metrics_to_log.items():
            logger.log_scalar(key.replace("/", "_"), value, step=step)

    return metrics_to_log
