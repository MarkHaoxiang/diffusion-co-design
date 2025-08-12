import os
import argparse
import shutil
from omegaconf import OmegaConf
from multiprocessing.pool import Pool
import numpy as np

from diffusion_co_design.common import OUTPUT_DIR
from diffusion_co_design.vmas.schema import ScenarioConfig
from diffusion_co_design.vmas.diffusion.generate import Generate


def generate_env(n_and_scenario):
    n, scenario = n_and_scenario
    generate = Generate(scenario=scenario, rng=np.random.default_rng())
    dataset = generate(n=n, training_dataset=True, disable_tqdm=False)
    return dataset


def setup_scenario(scenario_name: str, n: int):
    scenario = ScenarioConfig.from_file(os.path.join("conf", f"{scenario_name}.yaml"))
    data_dir = os.path.join(OUTPUT_DIR, "vmas", "scenario", scenario_name)

    # Remove existing
    if os.path.exists(data_dir):
        shutil.rmtree(path=data_dir)

    # Save Config
    os.makedirs(data_dir)
    with open(os.path.join(data_dir, "config.yaml"), "w") as f:
        out = scenario.model_dump()
        yaml = OmegaConf.create(out)
        OmegaConf.save(yaml, f)

    # Generate n samples

    P = min(n, 20)
    with Pool(P) as p:
        input_arr = [((n // P) + (1 if i < (n % P) else 0), scenario) for i in range(P)]
        dataset_all = p.map(generate_env, input_arr)  # type: ignore
        dataset = np.concatenate(dataset_all, axis=0)

    np.save(os.path.join(data_dir, "dataset.npy"), dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario_name", required=True, type=str)
    args = parser.parse_args()

    setup_scenario(scenario_name=args.scenario_name, n=1_000_000)
