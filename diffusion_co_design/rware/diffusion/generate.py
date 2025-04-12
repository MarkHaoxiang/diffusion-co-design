import random

import numpy as np
from tqdm import tqdm

from diffusion_co_design.rware.schema import Representation, ScenarioConfig


def get_position(idx: int, size: int):
    return (idx // size, idx % size)


def generate(
    size: int,
    n_shelves: int,
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
        if idx not in goal_idxs:
            # if idx not in agent_idxs and idx not in goal_idxs:
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
                env[color, *get_position(idx, size)] = 1.0
            if training_dataset:
                env = env * 2 - 1  # type: ignore
        elif representation == "flat" or representation == "graph":
            # Shelf placement
            # features_dim_shelf = 2 + n_colors
            features_dim_shelf = 2
            env = np.zeros((n_shelves * features_dim_shelf), dtype=np.float32)  # type: ignore
            for i, idx in enumerate(shelf_idxs):
                # color = i % n_colors
                x, y = get_position(idx, size)
                if training_dataset:
                    x = x / (size - 1)
                    y = y / (size - 1)
                start = i * features_dim_shelf
                env[start] = x
                env[start + 1] = y
                # env[start + 2 + color] = 1.0

            if training_dataset:
                env = env * 2 - 1

            if representation == "graph":
                # Reshape
                env = env.reshape((n_shelves, features_dim_shelf))  # type: ignore

        environments.append(env)

    return environments


def generate_scenario(
    name: str,
    n_agents: int,
    n_shelves: int,
    n_goals: int,
    n_colors: int,
    agent_idxs: list[int] | None,
    agent_colors: list[int] | None,
    goal_idxs: list[int] | None,
    goal_colors: list[int] | None,
    size: int = 16,
    max_steps: int = 500,
    seed: int = 0,
):
    """Generate random scenario"""

    # Seeding
    random.seed(seed)
    np.random.seed(seed=seed)

    # Blocks
    block_idxs = list(range(size**2))
    random.shuffle(block_idxs)

    if agent_idxs is None and goal_idxs is None and goal_colors is None:
        agent_idxs = block_idxs[:n_agents]
        goal_idxs = block_idxs[n_agents : n_agents + n_goals]
    elif agent_idxs is not None and goal_idxs is not None:
        agent_idxs, goal_idxs = agent_idxs, goal_idxs
        assert len(agent_idxs) == n_agents
        assert len(goal_idxs) == n_goals
    else:
        raise ValueError("Not yet supported, pass in both agent and goals or none")

    if goal_colors is None:
        goal_colors = [i % n_colors for i in range(n_goals)]
    else:
        goal_colors = goal_colors
        assert len(goal_colors) == n_goals
        assert max(goal_colors) == n_colors - 1

    if agent_colors is None:
        agent_colors = [-1 for _ in range(n_agents)]
    else:
        agent_colors = agent_colors
        assert len(agent_colors) == n_agents
        assert max(agent_colors) == n_colors - 1 or -1 in agent_colors

    return ScenarioConfig(
        name=name,
        size=size,
        n_agents=n_agents,
        n_shelves=n_shelves,
        n_goals=n_goals,
        agent_idxs=agent_idxs,
        agent_colors=agent_colors,
        goal_idxs=goal_idxs,
        goal_colors=goal_colors,
        n_colors=n_colors,
        max_steps=max_steps,
    )
