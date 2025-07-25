# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted from
# https://github.com/pytorch/rl/blob/main/sota-implementations/multiagent/utils/logging.py#L32
#

import os
from typing import Literal, Any

from tensordict import TensorDictBase
import wandb
import torch
import numpy as np

from diffusion_co_design.common.pydra import Config


class LoggingConfig(Config):
    mode: Literal["online", "offline", "disabled"] = "online"
    evaluation_interval: int = 20
    evaluation_episodes: int = 5
    checkpoint_interval: int = 50


class ExperimentLogger:
    def __init__(
        self,
        directory: str,
        experiment_name: str,
        config: dict | None = None,
        project_name: str = "diffusion-co-design",
        mode: Literal["online", "offline", "disabled"] = "online",
    ):
        super().__init__()

        self.experiment_name = experiment_name
        self.directory = directory
        self.checkpoint_dir = os.path.join(directory, "checkpoints")
        self.config = config
        self.mode = mode
        self.project_name = project_name

        os.makedirs(directory, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def log(self, data: dict[str, Any]):
        wandb.log(data, commit=False)

    def commit(self):
        wandb.log({}, commit=True)

    def begin(self):
        wandb.init(
            project=self.project_name,
            name=self.experiment_name,
            dir=self.directory,
            config=self.config,
            mode=self.mode,
        )

    def checkpoint_state_dict(self, model, name: str):
        model_path = os.path.join(self.checkpoint_dir, name + ".pt")
        torch.save(model.state_dict(), model_path)
        return model_path

    def checkpoint_torch(self, object, name: str):
        torch.save(object, os.path.join(self.checkpoint_dir, name))

    @property
    def summary(self):
        return wandb.run.summary

    def close(self):
        wandb.finish()

    def __enter__(self):
        self.begin()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class RLExperimentLogger(ExperimentLogger):
    def __init__(
        self,
        directory: str,
        experiment_name: str,
        group_name: str,
        config: dict | None = None,
        project_name: str = "diffusion-co-design",
        mode: Literal["online", "offline", "disabled"] = "disabled",
    ):
        super().__init__(
            directory=directory,
            experiment_name=experiment_name,
            config=config,
            project_name=project_name,
            mode=mode,
        )
        self.group_name = group_name
        self.metrics_to_log: dict[str, Any] = {}

    def log(self, log: dict[str, Any], path_prefix: str | None = "train"):
        if path_prefix is not None:
            log = {f"{path_prefix}/{k}": v for k, v in log.items()}
        self.metrics_to_log.update(log)

    def collect_sampling_td(self, td: TensorDictBase):
        if "info" in td.get(self.group_name).keys():
            self.metrics_to_log.update(
                {
                    f"train/info/{key}": value.mean().item()
                    for key, value in td.get((self.group_name, "info")).items()
                }
            )

        reward = td.get(("next", self.group_name, "reward")).mean(
            -2
        )  # Mean over agents
        done = td.get(("next", "done"))
        if done.ndim > reward.ndim:
            done = done[..., 0, :]  # Remove expanded agent dim
        episode_reward = td.get(("next", self.group_name, "episode_reward")).mean(-2)[
            done
        ]

        if episode_reward.numel() == 0:  # Prevent crash if no dones
            episode_reward = torch.zeros(())

        self.metrics_to_log.update(
            {
                "train/reward/reward_min": reward.min().item(),
                "train/reward/reward_mean": reward.mean().item(),
                "train/reward/reward_max": reward.max().item(),
                "train/reward/episode_reward_min": episode_reward.min().item(),
                "train/reward/episode_reward_mean": episode_reward.mean().item(),
                "train/reward/episode_reward_max": episode_reward.max().item(),
            }
        )

    def collect_training_td(self, td: TensorDictBase):
        self.metrics_to_log.update(
            {f"train/learner/{key}": value.mean().item() for key, value in td.items()}
        )

    def collect_evaluation_td(
        self, td: TensorDictBase, evaluation_time: float, frames=None
    ):
        rewards = td.get(("next", self.group_name, "reward")).sum(dim=1).mean(dim=1)
        metrics_to_log = {
            "eval/episode_reward_min": min(rewards),
            "eval/episode_reward_max": max(rewards),
            "eval/episode_reward_mean": sum(rewards) / len(td),
            "eval/episode_len_mean": sum([td.batch_size[0] for td in td]) / len(td),
            "eval/evaluation_time": evaluation_time,
        }
        self.metrics_to_log.update(metrics_to_log)

        if frames is not None:
            vid = torch.tensor(
                np.transpose(frames[: td[0].batch_size[0]], (0, 3, 1, 2)),
                dtype=torch.uint8,
            ).unsqueeze(0)

            self.metrics_to_log["eval/video"] = wandb.Video(vid, fps=10, format="mp4")  # type: ignore

    def collect_times(
        self,
        sampling_time: float,
        rl_training_time: float,
        designer_time: float,
        total_time: float,
    ):
        self.metrics_to_log.update(
            {
                "train/sampling_time": sampling_time,
                "train/training_time": rl_training_time,
                "train/designer_time": designer_time,
                "train/total_time": total_time,
            }
        )

    def commit(
        self, total_frames: int | None = None, current_frames: int | None = None
    ):
        metrics_to_log = self.metrics_to_log
        if total_frames is not None:
            metrics_to_log["train/total_frames"] = total_frames
        if current_frames is not None:
            metrics_to_log["train/current_frames"] = current_frames

        wandb.log(metrics_to_log, commit=True)
        self.metrics_to_log.clear()

    def upload_model(self, model_path: str, name: str):
        artifact = wandb.Artifact(name=name, type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact, aliases=[name, "latest"])
        wandb.run.summary[f"{name}_artifact"] = f"{name}:latest"
