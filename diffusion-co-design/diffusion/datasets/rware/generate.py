import os
import shutil
import random
import json

import numpy as np
from matplotlib import cm
from tqdm import tqdm
from pydantic_settings import BaseSettings, SettingsConfigDict
from PIL import Image

from rware.layout import Layout, Direction

DATA_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../rware")


class WarehouseRandomGeneratorConfig(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True)
    experiment_name: str = "default"
    seed: int = 0
    size: int = 16
    n_agents: int = 5
    n_shelves: int = 50
    n_goals: int = 5
    n_samples: int = 10


# Shelves, Agents, Goals
n_colors = 3
colors = cm.get_cmap("rainbow", n_colors)
colors = (np.array([colors(i)[:3] for i in range(n_colors)]) * 255).round()


def generate(cfg: WarehouseRandomGeneratorConfig):
    size = cfg.size

    # Create empty data directory
    data_dir = os.path.join(DATA_BASE_DIR, cfg.experiment_name)
    if os.path.exists(data_dir):
        shutil.rmtree(path=data_dir)
    os.makedirs(data_dir)

    # Seeding
    random.seed(cfg.seed)
    np.random.seed(seed=cfg.seed)

    # Blocks
    block_idxs = list(range(size**2))

    # Fixed agent and goal positions (randomly generated)
    # TODO markli: allow this to be passed in by the user.
    random.shuffle(block_idxs)
    agent_idxs = block_idxs[: cfg.n_agents]
    goal_idxs = block_idxs[cfg.n_agents : cfg.n_agents + cfg.n_goals]
    remaining_idxs = block_idxs[cfg.n_agents + cfg.n_goals :]

    # Save config
    with open(os.path.join(data_dir, "config.json"), "w") as f:
        out = cfg.model_dump()
        out["agent_idxs"] = agent_idxs
        out["goal_idxs"] = goal_idxs
        json.dump(out, fp=f)

    # Generate uniform environments for exploration
    for sample_idx in tqdm(range(cfg.n_samples)):
        shelf_idxs = random.sample(remaining_idxs, cfg.n_shelves)

        # Environment placement
        env = np.zeros((size, size, 3), dtype=np.uint8)
        for idx in shelf_idxs:
            env[idx // size, idx % size] = colors[0]
        for idx in agent_idxs:
            env[idx // size, idx % size] = colors[1]
        for idx in goal_idxs:
            env[idx // size, idx % size] = colors[2]

        # Save images
        im = Image.fromarray(env)
        im.save(data_dir + "/%07d" % sample_idx + ".png")


def image_to_layout(im: np.ndarray):
    # Round to nearest colors
    h, w, _ = im.shape


if __name__ == "__main__":
    generate(WarehouseRandomGeneratorConfig())
