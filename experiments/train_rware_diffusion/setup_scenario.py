import os
import shutil

import hydra
from omegaconf import DictConfig, OmegaConf
from diffusion_co_design.rware.diffusion.generate import generate_scenario
from diffusion_co_design.common import OUTPUT_DIR


@hydra.main(
    version_base=None, config_path="conf", config_name="rware_16_50_5_4_corners"
)
def main(cfg: DictConfig) -> None:
    scenario = generate_scenario(**cfg)

    data_dir = os.path.join(OUTPUT_DIR, "rware", "scenario", scenario.name)
    if os.path.exists(data_dir):
        shutil.rmtree(path=data_dir)
    os.makedirs(data_dir)
    print(f"Generating dataset at {data_dir}")

    with open(os.path.join(data_dir, "config.yaml"), "w") as f:
        out = scenario.model_dump()
        out["agent_idxs"] = scenario.agent_idxs
        out["goal_idxs"] = scenario.goal_idxs
        out["goal_colors"] = scenario.goal_colors
        yaml = OmegaConf.create(out)
        OmegaConf.save(yaml, f)


if __name__ == "__main__":
    main()
