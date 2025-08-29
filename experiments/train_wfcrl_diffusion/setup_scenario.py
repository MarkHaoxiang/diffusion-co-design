import os
import argparse
import shutil
from omegaconf import OmegaConf
from multiprocessing.pool import Pool
import numpy as np

from diffusion_co_design.common import OUTPUT_DIR
from diffusion_co_design.wfcrl.schema import ScenarioConfig
from diffusion_co_design.wfcrl.diffusion.generate import Generate
from diffusion_co_design.wfcrl.design import manual_design_cases


def generate_env(n_and_scenario):
    n, scenario = n_and_scenario
    generate = Generate(
        num_turbines=scenario.n_turbines,
        map_x_length=scenario.map_x_length,
        map_y_length=scenario.map_y_length,
        minimum_distance_between_turbines=scenario.min_distance_between_turbines,
        rng=np.random.default_rng(),
    )
    dataset = generate(n=n, training_dataset=True, disable_tqdm=False)
    return dataset


def setup_scenario(scenario_name: str, n: int):
    if os.path.exists(os.path.join("conf", f"{scenario_name}.yaml")):
        scenario = ScenarioConfig.from_file(
            os.path.join("conf", f"{scenario_name}.yaml")
        )
    else:
        # Try to generate scenario from pre-defined WFCRL windfarms
        datacase = manual_design_cases[scenario_name]
        xcoords = datacase.xcoords
        ycoords = datacase.ycoords
        min_x = min(xcoords)
        max_x = max(xcoords)
        min_y = min(ycoords)
        max_y = max(ycoords)
        map_x_length = max_x - min_x + 10
        map_y_length = max_y - min_y + 10
        scenario = ScenarioConfig(
            name=scenario_name,
            max_steps=150,
            n_turbines=len(xcoords),
            map_x_length=map_x_length,
            map_y_length=map_y_length,
            min_distance_between_turbines=250,
        )
    data_dir = os.path.join(OUTPUT_DIR, "wfcrl", "scenario", scenario_name)

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

    setup_scenario(scenario_name=args.scenario_name, n=100_000)
