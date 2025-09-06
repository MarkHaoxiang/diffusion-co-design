"""
Train a diffusion model on images.

Modified from https://github.com/rllab-snu/ADD
"""

import argparse
import os
import shutil
import numpy as np
import torch
from omegaconf import OmegaConf
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

from diffusion_co_design.common import OUTPUT_DIR
from diffusion_co_design.wfcrl.schema import ScenarioConfig
from diffusion_co_design.wfcrl.diffusion.generate import Generate
from diffusion_co_design.wfcrl.model.diffusion import diffusion_setup


def main():
    args = create_argparser().parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx

    # Setup scenario
    data_dir = os.path.join(OUTPUT_DIR, "wfcrl", "scenario", args.experiment_name)
    scenario = ScenarioConfig.from_file(os.path.join(data_dir, "config.yaml"))

    log_dir = os.path.join(OUTPUT_DIR, "wfcrl", "diffusion", args.experiment_name)

    dist_util.setup_dist()
    logger.configure(dir=log_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = diffusion_setup(scenario=scenario)
    model.to(dist_util.dev())

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")

    dataset = np.load(os.path.join(data_dir, "dataset.npy"))
    print(f"Load dataset with shape {dataset.shape}")

    def data_iterator():
        while True:
            idxs = np.random.choice(dataset.shape[0], args.batch_size, replace=True)
            batch = torch.from_numpy(np.stack(dataset[idxs]))
            yield batch, {}

    # data = data_iterator(data)
    data = data_iterator()

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=0,  # Disabled
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        anneal_lr_enable=True,  # New addition
    ).run_loop()


def create_argparser():
    defaults = dict(
        experiment_name="wmr",
        schedule_sampler="uniform",
        lr=3e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=256,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        save_interval=50000,
        resume_checkpoint="",
        gpu_idx="0",
    )

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
