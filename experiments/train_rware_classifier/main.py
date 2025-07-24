import os
from functools import partial

import hydra
import hydra.core.hydra_config
import torch
from tqdm import tqdm
from diffusion_co_design.common import ExperimentLogger, cuda as device
from diffusion_co_design.rware.schema import TrainingConfig

from dataset import load_dataset, make_dataloader
from diffusion_co_design.rware.model.classifier import make_model, make_hint_loss

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
        representation=cfg.model.representation,
        device=device,
    )
    train_dataloader = make_dataloader_fn(train_dataset)
    eval_dataloader = make_dataloader_fn(eval_dataset)

    # Load model
    model = make_model(
        cfg.model.name,
        scenario=training_cfg.scenario,
        model_kwargs=cfg.model.model_kwargs,
        device=device,
    )

    if cfg.enable_hint:
        hint_loss_fn = make_hint_loss(
            model=cfg.model.name, env_critic=model, device=device
        )
        optim = torch.optim.AdamW(
            list(model.parameters()) + list(hint_loss_fn.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
    else:
        optim = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

    criterion = torch.nn.MSELoss()

    train_losses = []
    eval_losses = []

    experiment_name = f"{cfg.model.name}_{cfg.model.representation}_{cfg.train_target}_{cfg.experiment_suffix}"
    with (
        ExperimentLogger(
            directory=output_dir,
            experiment_name=experiment_name,
            config=cfg.model_dump(),
            project_name="diffusion-co-design-rware-classifier",
            mode=cfg.logging_mode,
        ) as logger,
        tqdm(range(cfg.train_epochs)) as pbar,
    ):
        for epoch in range(cfg.train_epochs):
            running_train_loss = 0
            running_train_hint_loss = 0
            model.train()
            for x, y, y_critic, distillation_hint in train_dataloader:
                optim.zero_grad()

                match cfg.train_target:
                    case "sampling":
                        y_target = y
                    case "critic":
                        y_target = y_critic

                y_pred, student_features = model.predict(x)

                loss = 0
                if cfg.enable_hint:
                    hint_loss = hint_loss_fn(distillation_hint, student_features)
                    running_train_hint_loss += hint_loss.item()
                    loss += hint_loss * cfg.hint_loss_weight

                prediction_loss = criterion(y_pred, y_target)
                loss += prediction_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                running_train_loss += prediction_loss.item()
            running_train_loss = running_train_loss / len(train_dataloader)
            running_train_hint_loss = running_train_hint_loss / len(train_dataloader)

            # Evaluate
            model.eval()
            running_eval_loss_sampling = 0
            running_eval_loss_critic = 0
            with torch.no_grad():
                for x, y, y_critic, distillation_hint in eval_dataloader:
                    y_pred, student_features = model.predict(x)
                    loss_sampling = criterion(y_pred, y)
                    loss_critic = criterion(y_pred, y_critic)
                    running_eval_loss_sampling += loss_sampling.item()
                    running_eval_loss_critic += loss_critic.item()
            n = len(eval_dataloader)
            running_eval_loss_sampling = running_eval_loss_sampling / n
            running_eval_loss_critic = running_eval_loss_critic / n

            train_losses.append(running_train_loss)
            eval_losses.append(running_eval_loss_sampling)
            pbar.set_description(f"Train Loss {running_train_loss}")

            logger.log(
                {
                    "train/loss": running_train_loss,
                    "eval/loss_sampling": running_eval_loss_sampling,
                    "eval/loss_critic": running_eval_loss_critic,
                },
            )
            if cfg.enable_hint:
                logger.log({"train/hint_loss": running_train_hint_loss})
            logger.commit()
            pbar.update()

        torch.save(
            model.state_dict(), os.path.join(logger.checkpoint_dir, "classifier.pt")
        )


if __name__ == "__main__":
    main()
