import os
import re

import torch


def get_latest_model(dir: str, prefix: str) -> str:
    files = os.listdir(dir)
    target_files = [f for f in files if re.match(rf"{prefix}\d+\.pt", f)]
    latest = max(target_files, key=lambda x: int(re.search(r"\d+", x).group()))
    return os.path.join(dir, latest)


def start_from_checkpoint(
    training_dir: str | None,
    models: list[tuple[torch.nn.Module | None, str]],
):
    if training_dir is None:
        return

    training_dir = os.path.join(training_dir, "checkpoints")

    for model, name in models:
        if model is None:
            continue

        checkpoint = get_latest_model(training_dir, name)
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict)


def np_list_to_tensor_list(np_list: list) -> list[torch.Tensor]:
    return [torch.tensor(x) for x in np_list]
