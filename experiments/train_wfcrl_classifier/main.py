import os
from functools import partial

import hydra
import torch
from tqdm import tqdm
from experiments.train_wfcrl.main import TrainingConfig
from diffusion_co_design.common import (
    ExperimentLogger,
    cuda as device,
)

from dataset import load_dataset, make_dataloader
from diffusion_co_design.wfcrl.model.classifier import GNNCritic
from diffusion_co_design.wfcrl.model.rl import maybe_make_denormaliser

from conf.schema import Config


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    cfg = Config.from_raw(cfg)
    training_cfg = TrainingConfig.from_file(
        os.path.join(cfg.training_dir, ".hydra", "config.yaml")
    )

    # Load dataset
    train_dataset, eval_dataset = load_dataset(
        scenario=training_cfg.scenario,
        training_dir=cfg.training_dir,
        dataset_size=10_000,
        num_workers=25,
        test_proportion=0.2,
        recompute=False,
        device=device,
    )

    make_dataloader_fn = partial(
        make_dataloader,
        scenario=training_cfg.scenario,
        batch_size=cfg.batch_size,
        device=device,
    )
    train_dataloader = make_dataloader_fn(train_dataset)
    eval_dataloader = make_dataloader_fn(eval_dataset)

    # Load model
    model = torch.nn.Sequential(
        GNNCritic(
            cfg=training_cfg.scenario,
            node_emb_dim=cfg.model.node_emb_size,
            edge_emb_dim=cfg.model.edge_emb_size,
            n_layers=cfg.model.depth,
        ),
        maybe_make_denormaliser(training_cfg.normalisation),
    ).to(device=device)

    print(f"Num parameters: {sum([p.numel() for p in model.parameters()])}")

    optim = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    criterion = torch.nn.HuberLoss()

    train_losses = []
    eval_losses = []

    with (
        ExperimentLogger(
            directory=output_dir,
            experiment_name="wfcrl_env_critic_" + cfg.train_target,
            config=cfg.model_dump(),
            project_name="diffusion-co-design-wfcrl-classifier",
            mode=cfg.logging_mode,
        ) as logger,
        tqdm(range(cfg.train_epochs)) as pbar,
    ):
        for epoch in range(cfg.train_epochs):
            running_train_loss = 0
            model.train()
            for x, y, y_critic in train_dataloader:
                optim.zero_grad()

                match cfg.train_target:
                    case "sampling":
                        y_target = y
                    case "critic":
                        y_target = y_critic

                y_pred = model(x)
                loss = criterion(y_pred, y_target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()
                running_train_loss += loss.item()
            running_train_loss = running_train_loss / len(train_dataloader)

            # Evaluate
            model.eval()
            running_eval_loss_sampling = 0
            running_eval_loss_critic = 0
            with torch.no_grad():
                for x, y, y_critic in eval_dataloader:
                    y_pred = model(x)
                    loss_sampling = criterion(y_pred, y)
                    loss_critic = criterion(y_pred, y_critic)
                    running_eval_loss_sampling += loss_sampling.item()
                    running_eval_loss_critic += loss_critic.item()
            n = len(eval_dataloader)
            running_eval_loss_sampling = running_eval_loss_sampling / n
            running_eval_loss_critic = running_eval_loss_critic / n

            train_losses.append(running_train_loss)
            eval_losses.append(running_eval_loss_sampling)
            pbar.set_description(f" Train Loss {running_train_loss}")

            logger.log(
                {
                    "train/loss": running_train_loss,
                    "eval/loss_sampling": running_eval_loss_sampling,
                    "eval/loss_critic": running_eval_loss_critic,
                },
            )
            logger.commit()
            pbar.update()

        torch.save(
            model.state_dict(), os.path.join(logger.checkpoint_dir, "classifier.pt")
        )


if __name__ == "__main__":
    main()
