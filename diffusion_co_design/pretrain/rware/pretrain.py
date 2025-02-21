"""
Train a diffusion model on images.

Modified from https://github.com/rllab-snu/ADD
"""

import argparse
import os
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, TensorDataset
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

from diffusion_co_design.utils import OUTPUT_DIR
from diffusion_co_design.pretrain.rware.generate import generate
from diffusion_co_design.pretrain.rware.generator import (
    get_model_and_diffusion_defaults,
)


def main():
    args = create_argparser().parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx

    data_dir = os.path.join(OUTPUT_DIR, "diffusion_datasets", args.experiment_name)
    log_dir = os.path.join(OUTPUT_DIR, "diffusion_pretrain", args.experiment_name)

    dist_util.setup_dist()
    logger.configure(dir=log_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")

    # For RGB training
    # data = load_data(
    #     data_dir=data_dir,
    #     batch_size=args.batch_size,
    #     image_size=args.image_size,
    #     class_cond=args.class_cond,
    #     random_flip=args.random_flip,
    #     rgb=args.rgb,
    # )

    # For storage channel training
    # data = np.load(data_dir + "/environments.npy")
    # data = torch.from_numpy(data).to(torch.float32)
    # data = TensorDataset(data)
    # data = DataLoader(data, batch_size=args.batch_size, shuffle=True)

    # def data_iterator(data):
    #     while True:
    #         for batch in data:
    #             yield batch[0], {}

    # Training on the true underlying distribution
    # By generation samples on the fly
    data_config = OmegaConf.load(os.path.join(data_dir, "config.yaml"))

    def data_iterator():
        while True:
            batch = torch.from_numpy(
                np.stack(
                    generate(
                        size=data_config.size,
                        n_shelves=data_config.n_shelves,
                        goal_idxs=data_config.goal_idxs,
                        agent_idxs=data_config.agent_idxs,
                        n=args.batch_size,
                        rgb=False,
                        training_dataset=True,
                    )
                )
            )
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
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        experiment_name="default",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=256,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        random_flip=False,
        rgb=True,
        gpu_idx="0",
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(get_model_and_diffusion_defaults(size=16, image_channels=1))

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
