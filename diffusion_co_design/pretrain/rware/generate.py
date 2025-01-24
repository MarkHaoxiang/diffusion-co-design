import os
import shutil
import random

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

import numpy as np
from matplotlib import cm
from tqdm import tqdm
from PIL import Image

from rware.warehouse import ImageLayer

from diffusion_co_design.utils import omega_to_pydantic, OUTPUT_DIR


COLOR_ORDER = [ImageLayer.SHELVES, ImageLayer.AGENTS, ImageLayer.GOALS, "Highway"]


class WarehouseRandomGeneratorConfig(BaseModel):
    experiment_name: str = "default"
    seed: int = 0
    size: int = 16
    n_agents: int = 5
    n_shelves: int = 50
    n_goals: int = 5
    n_samples: int = 10
    agent_idxs: list[int] | None = None
    goal_idxs: list[int] | None = None


# Shelves, Agents, Goals
n_colors = len(COLOR_ORDER)
colors = cm.get_cmap("rainbow", n_colors)
colors = (np.array([colors(i)[:3] for i in range(n_colors)]) * 255).round()


def generate(
    size: int,
    n_shelves: int,
    agent_idxs: list[int],
    goal_idxs: list[int],
    n: int = 1,
    disable_tqdm=True,
) -> list[np.ndarray]:

    # Possible positions
    remaining_idxs = []
    for idx in range(size**2):
        if idx not in agent_idxs and idx not in goal_idxs:
            remaining_idxs.append(idx)

    environments = []
    for _ in tqdm(range(n), disable=disable_tqdm):
        shelf_idxs = random.sample(remaining_idxs, n_shelves)

        # Environment placement
        env = np.zeros((size, size, 3), dtype=np.uint8)
        env[:, :] = colors[-1]
        for idx in shelf_idxs:
            env[idx // size, idx % size] = colors[0]
        for idx in agent_idxs:
            env[idx // size, idx % size] = colors[1]
        for idx in goal_idxs:
            env[idx // size, idx % size] = colors[2]

        environments.append(env)

    return environments


def generate_run(cfg: WarehouseRandomGeneratorConfig):
    size = cfg.size

    # Create empty data directory
    data_dir = os.path.join(OUTPUT_DIR, "diffusion_datasets", cfg.experiment_name)
    if os.path.exists(data_dir):
        shutil.rmtree(path=data_dir)
    os.makedirs(data_dir)

    # Seeding
    random.seed(cfg.seed)
    np.random.seed(seed=cfg.seed)

    # Blocks
    block_idxs = list(range(size**2))
    random.shuffle(block_idxs)

    if cfg.agent_idxs is None and cfg.goal_idxs is None:
        agent_idxs = block_idxs[: cfg.n_agents]
        goal_idxs = block_idxs[cfg.n_agents : cfg.n_agents + cfg.n_goals]
    elif cfg.agent_idxs is not None and cfg.goal_idxs is not None:
        agent_idxs, goal_idxs = cfg.agent_idxs, cfg.goal_idxs
        assert len(agent_idxs) == cfg.n_agents
        assert len(goal_idxs) == cfg.n_goals
    else:
        raise ValueError("Not yet supported, pass in both agent and goals or none")

    # Save config
    with open(os.path.join(data_dir, "config.yaml"), "w") as f:
        out = cfg.model_dump()
        out["agent_idxs"] = agent_idxs
        out["goal_idxs"] = goal_idxs
        yaml = OmegaConf.create(out)
        OmegaConf.save(yaml, f)

    # Generate uniform environments for exploration
    environments = generate(
        size, cfg.n_shelves, agent_idxs, goal_idxs, n=cfg.n_samples, disable_tqdm=False
    )
    for i, env in tqdm(enumerate(environments)):
        # Save images
        im = Image.fromarray(env)
        im.save(data_dir + "/%07d" % i + ".png")


@hydra.main(version_base=None, config_path="configs", config_name="default")
def run(config: DictConfig):
    print(f"Running job {HydraConfig.get().job.name}")
    config = omega_to_pydantic(config, WarehouseRandomGeneratorConfig)
    generate_run(config)


if __name__ == "__main__":
    run()
