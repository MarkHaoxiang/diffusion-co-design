import os

import hydra
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from experiments.train_rware.main import (
    TrainingConfig,
)
from diffusion_co_design.common import (
    ExperimentLogger,
    omega_to_pydantic,
    cuda as device,
)

from dataset import (
    load_dataset,
    make_dataloader,
    working_dir,
)
from diffusion_co_design.rware.model.classifier import make_model

from conf.schema import Config


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    cfg = omega_to_pydantic(cfg, Config)
    hydra_dir = os.path.join(cfg.training_dir, ".hydra")
    training_cfg = omega_to_pydantic(
        OmegaConf.load(os.path.join(hydra_dir, "config.yaml")), TrainingConfig
    )
    training_cfg.scenario.representation = cfg.model.representation.split("_")[0]

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

    train_dataloader = make_dataloader(
        train_dataset,
        scenario=training_cfg.scenario,
        batch_size=128,
        representation=cfg.model.representation,
        device=device,
    )

    eval_dataloader = make_dataloader(
        eval_dataset,
        scenario=training_cfg.scenario,
        batch_size=128,
        representation=cfg.model.representation,
        device=device,
    )

    # Load model
    model = make_model(
        cfg.model.name,
        scenario=training_cfg.scenario,
        model_kwargs=cfg.model.model_kwargs,
        device=device,
    )

    optim = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    criterion = torch.nn.MSELoss()

    train_losses = []
    eval_losses = []

    experiment_name = f"{cfg.model.name}_{cfg.model.representation}"
    with (
        ExperimentLogger(
            base_dir=working_dir,
            experiment_name=experiment_name,
            config=cfg,
            mode=cfg.logging_mode,
        ) as logger,
        tqdm(range(cfg.train_epochs)) as pbar,
    ):
        for epoch in range(cfg.train_epochs):
            running_train_loss = 0
            model.train()
            for x, y in train_dataloader:
                optim.zero_grad()
                y_pred = model.predict(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optim.step()
                running_train_loss += loss.item()
            running_train_loss = running_train_loss / len(train_dataloader)

            # Evaluate
            model.eval()
            running_eval_loss = 0
            with torch.no_grad():
                for x, y in eval_dataloader:
                    y_pred = model.predict(x)
                    loss = criterion(y_pred, y)
                    running_eval_loss += loss.item()
            running_eval_loss = running_eval_loss / len(eval_dataloader)

            train_losses.append(running_train_loss)
            eval_losses.append(running_eval_loss)
            pbar.set_description(
                f" Train Loss {running_train_loss} Eval Loss {running_eval_loss}"
            )

            logger.log(
                {
                    "train/loss": running_train_loss,
                    "eval/loss": running_eval_loss,
                },
            )
            logger.commit()
            pbar.update()

        torch.save(
            model.state_dict(), os.path.join(logger.checkpoint_dir, "classifier.pt")
        )


if __name__ == "__main__":
    main()
