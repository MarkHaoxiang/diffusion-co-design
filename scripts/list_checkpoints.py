import os
import yaml
from diffusion_co_design.common import OUTPUT_DIR

if __name__ == "__main__":
    experiments = os.path.join(OUTPUT_DIR, "experiments", "train_rware")

    all_experiments = []
    for day in os.listdir(experiments):
        for second in os.listdir(os.path.join(experiments, day)):
            all_experiments.append(os.path.join(experiments, day, second))

    for experiment in all_experiments:
        with open(os.path.join(experiment, ".hydra", "config.yaml"), "r") as f:
            config = yaml.safe_load(f)
            experiment_name = config["experiment_name"]

            checkpoint_dir = os.path.join(experiment, "checkpoints")

            if os.path.exists(os.path.join(checkpoint_dir, "policy_3999.pt")):
                print(f"({repr(experiment_name)}, {repr(checkpoint_dir)}),")
