"""
Train a diffusion model on images.

Modified from https://github.com/rllab-snu/ADD
"""

import argparse
import os
import numpy as np
from omegaconf import OmegaConf
import torch
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

from diffusion_co_design.utils import OUTPUT_DIR
from diffusion_co_design.pretrain.rware.generate import generate
from diffusion_co_design.pretrain.rware.generator import (
    create_model_and_diffusion_rware,
)


def main():
    args = create_argparser().parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx

    data_dir = os.path.join(
        OUTPUT_DIR, "diffusion_datasets", args.representation, args.experiment_name
    )
    data_config = OmegaConf.load(os.path.join(data_dir, "config.yaml"))
    args.image_channels = data_config.n_colors

    log_dir = os.path.join(
        OUTPUT_DIR, "diffusion_pretrain", args.representation, args.experiment_name
    )

    dist_util.setup_dist()
    logger.configure(dir=log_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion_rware(
        scenario=data_config,
        representation=args.representation,
    )
    model.to(dist_util.dev())

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")

    # Training on the true underlying distribution
    # By generation samples on the fly

    def data_iterator():
        while True:
            batch = torch.from_numpy(
                np.stack(
                    generate(
                        size=data_config.size,
                        n_shelves=data_config.n_shelves,
                        goal_idxs=data_config.goal_idxs,
                        agent_idxs=data_config.agent_idxs,
                        n_colors=data_config.n_colors,
                        n=args.batch_size,
                        representation=args.representation,
                        training_dataset=True,
                    )
                )
            )
            if args.representation == "flat":
                batch = batch.unsqueeze(-1)
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
        lr_anneal_steps=args.lr_anneal_steps,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
    ).run_loop()


def create_argparser():
    defaults = dict(
        experiment_name="rware_16_50_5_4_corners",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=256,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        save_interval=10000,
        resume_checkpoint="",
        gpu_idx="0",
        representation="graph",
    )

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
