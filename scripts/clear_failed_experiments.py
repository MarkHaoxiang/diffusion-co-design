import os
import yaml
import shutil
import re
from diffusion_co_design.common import OUTPUT_DIR

rware_limit = 3000
wfcrl_limit = 200

if __name__ == "__main__":
    remove = []

    for experiment_folder, limit in (
        ("train_rware", rware_limit),
        ("train_wfcrl", wfcrl_limit),
    ):
        experiments = os.path.join(OUTPUT_DIR, "experiments", experiment_folder)

        all_experiments = []
        for day in os.listdir(experiments):
            day_folder = os.path.join(experiments, day)
            seconds = os.listdir(day_folder)
            if len(seconds) == 0:
                remove.append(day_folder)
            for second in os.listdir(day_folder):
                all_experiments.append(os.path.join(day_folder, second))

        for experiment in all_experiments:
            try:
                with open(os.path.join(experiment, ".hydra", "config.yaml"), "r") as f:
                    config = yaml.safe_load(f)
                    experiment_name = config["experiment_name"]

                    checkpoint_dir = os.path.join(experiment, "checkpoints")
                    if not os.path.exists(checkpoint_dir):
                        remove.append(experiment)
                        continue

                    files = os.listdir(checkpoint_dir)
                    target_files = [f for f in files if re.match(r".*\d+\.pt", f)]
                    numbers = [int(re.search(r"\d+", x).group()) for x in target_files]
                    if len(numbers) == 0 or max(numbers) < limit:
                        remove.append(experiment)
            except:  # noqa: E722
                pass  # Probably a multirun

    print(remove)
    for folder in remove:
        shutil.rmtree(folder)
