import os
import shutil
import random
from typing import Literal, TypeAlias

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

import numpy as np
from tqdm import tqdm


from diffusion_co_design.utils import omega_to_pydantic, OUTPUT_DIR

Representation: TypeAlias = Literal["image", "flat"]


class WarehouseRandomGeneratorConfig(BaseModel):
    name: str = "default"
    seed: int = 0
    size: int = 16
    n_agents: int = 5
    n_shelves: int = 50
    n_goals: int = 5
    n_samples: int = 10
    agent_idxs: list[int] | None = None
    agent_colors: list[int] | None = None
    goal_idxs: list[int] | None = None
    goal_colors: list[int] | None = None
    n_colors: int = 1
    max_steps: int = 500
    representation: Representation = "flat"


def generate(
    size: int,
    n_shelves: int,
    agent_idxs: list[int],
    goal_idxs: list[int],
    n_colors: int,
    n: int = 1,
    disable_tqdm: bool = True,
    training_dataset: bool = False,
    representation: Representation = "image",
) -> list[np.ndarray]:
    # Possible positions
    remaining_idxs = []
    for idx in range(size**2):
        if idx not in agent_idxs and idx not in goal_idxs:
            remaining_idxs.append(idx)

    environments = []
    for _ in tqdm(range(n), disable=disable_tqdm):
        shelf_idxs = random.sample(remaining_idxs, n_shelves)

        if representation == "image":
            # Shelf placement
            dtype = np.float32 if training_dataset else np.uint8
            env = np.zeros((n_colors, size, size), dtype=dtype)
            for i, idx in enumerate(shelf_idxs):
                color = i % n_colors
                env[color, idx // size, idx % size] = 1.0
            if training_dataset:
                env = env * 2 - 1  # type: ignore
        elif representation == "flat":
            # Shelf placement
            # features_dim_shelf = 2 + n_colors
            features_dim_shelf = 2
            env = np.zeros((n_shelves * features_dim_shelf), dtype=np.float32)  # type: ignore
            for i, idx in enumerate(shelf_idxs):
                # color = i % n_colors
                x = float(idx // size)
                y = float(idx % size)
                if training_dataset:
                    x = x / (size - 1)
                    y = y / (size - 1)
                start = i * features_dim_shelf
                env[start] = x
                env[start + 1] = y
                # env[start + 2 + color] = 1.0

            if training_dataset:
                env = env * 2 - 1

        environments.append(env)

    return environments


def generate_run(cfg: WarehouseRandomGeneratorConfig):
    size = cfg.size

    # Create empty data directory
    data_dir = os.path.join(
        OUTPUT_DIR, "diffusion_datasets", cfg.representation, cfg.name
    )
    if os.path.exists(data_dir):
        shutil.rmtree(path=data_dir)
    os.makedirs(data_dir)
    print(f"Generating dataset at {data_dir}")

    # Seeding
    random.seed(cfg.seed)
    np.random.seed(seed=cfg.seed)

    # Blocks
    block_idxs = list(range(size**2))
    random.shuffle(block_idxs)

    if cfg.agent_idxs is None and cfg.goal_idxs is None and cfg.goal_colors is None:
        agent_idxs = block_idxs[: cfg.n_agents]
        goal_idxs = block_idxs[cfg.n_agents : cfg.n_agents + cfg.n_goals]
    elif cfg.agent_idxs is not None and cfg.goal_idxs is not None:
        agent_idxs, goal_idxs = cfg.agent_idxs, cfg.goal_idxs
        assert len(agent_idxs) == cfg.n_agents
        assert len(goal_idxs) == cfg.n_goals
    else:
        raise ValueError("Not yet supported, pass in both agent and goals or none")

    if cfg.goal_colors is None:
        goal_colors = [i % cfg.n_colors for i in range(cfg.n_goals)]
    else:
        goal_colors = cfg.goal_colors
        assert len(goal_colors) == cfg.n_goals
        assert max(goal_colors) == cfg.n_colors - 1

    if cfg.agent_colors is None:
        agent_colors = [-1 for _ in range(cfg.n_agents)]
    else:
        agent_colors = cfg.agent_colors
        assert len(agent_colors) == cfg.n_agents
        assert max(agent_colors) == cfg.n_colors - 1 or -1 in agent_colors

    # Save config
    with open(os.path.join(data_dir, "config.yaml"), "w") as f:
        out = cfg.model_dump()
        out["agent_idxs"] = agent_idxs
        out["goal_idxs"] = goal_idxs
        out["goal_colors"] = goal_colors
        yaml = OmegaConf.create(out)
        OmegaConf.save(yaml, f)

    # Generate uniform environments for exploration
    environments = generate(
        size,
        cfg.n_shelves,
        agent_idxs,
        goal_idxs,
        n_colors=cfg.n_colors,
        n=cfg.n_samples,
        representation=cfg.representation,
        disable_tqdm=False,
        training_dataset=True,
    )

    env_buffer = np.stack(environments)
    np.save(data_dir + "/environments.npy", env_buffer)


@hydra.main(
    version_base=None,
    config_path="../../bin/conf/scenario",
    config_name="rware_16_50_5_5_random",
)
def run(config: DictConfig):
    print(f"Running job {HydraConfig.get().job.name}")
    generate_config: WarehouseRandomGeneratorConfig = omega_to_pydantic(
        config, WarehouseRandomGeneratorConfig
    )

    generate_run(generate_config)


if __name__ == "__main__":
    run()
