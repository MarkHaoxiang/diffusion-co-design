import numpy as np
import warnings

from rware.layout import Layout, ImageLayer
from diffusion_co_design.pretrain.rware.generate import (
    WarehouseRandomGeneratorConfig,
)


def storage_to_layout_image(
    shelf_im: np.ndarray,
    agent_idxs: list[int],
    agent_colors: list[int],
    goal_idxs: list[int],
    goal_colors: list[int],
):
    n_colors = shelf_im.shape[0]
    size = shelf_im.shape[1]

    # Prevent duplicate shelves
    fc = np.argmax(shelf_im, axis=0)
    mask = np.zeros_like(shelf_im)
    mask[fc, np.arange(shelf_im.shape[1])[:, None], np.arange(shelf_im.shape[2])] = 1
    shelf_im = shelf_im * mask

    agent_im = np.zeros((1, size, size))
    agent_colors_im = np.zeros((1, size, size))
    goal_im = np.zeros((1, size, size))

    for idx, color in zip(agent_idxs, agent_colors):
        agent_im[0, idx // size, idx % size] = 1
        agent_colors_im[0, idx // size, idx % size] = color + 1
    for idx, color in zip(goal_idxs, goal_colors):
        goal_im[0, idx // size, idx % size] = color + 1
        shelf_im[:, idx // size, idx % size] = 0

    im = np.concat((shelf_im, agent_im, agent_colors_im, goal_im), axis=0)
    layout = Layout.from_image(
        image=im,
        image_layers=[ImageLayer.STORAGE] * n_colors
        + [ImageLayer.AGENTS, ImageLayer.AGENT_COLOR, ImageLayer.GOALS],
    )
    return layout


def storage_to_layout_flat(
    features: np.ndarray,
    size: int,
    n_colors: int,
    n_shelves: int,
    agent_idxs: list[int],
    agent_colors: list[int],
    goal_idxs: list[int],
    goal_colors: list[int],
    auto_add_colors: bool = True,
):
    if len(features.shape) > 1:
        features = features.squeeze(-1)
    shelf_im = np.zeros((n_colors, size, size))

    n_colors = shelf_im.shape[0]
    size = shelf_im.shape[1]

    if auto_add_colors:
        feature_dim_shelf = 2
    else:
        feature_dim_shelf = 2 + n_colors
    for i in range(n_shelves):
        start = i * feature_dim_shelf
        x = np.clip(round(features[start]), 0, size - 1)
        y = np.clip(0, round(features[start + 1]), size - 1)

        if auto_add_colors:
            color = i % n_colors
        else:
            color = int(np.argmax(features[start + 2 : start + 2 + n_colors]))

        shelf_im[color, x, y] = 1

    return storage_to_layout_image(
        shelf_im=shelf_im,
        agent_idxs=agent_idxs,
        agent_colors=agent_colors,
        goal_idxs=goal_idxs,
        goal_colors=goal_colors,
    )


def storage_to_layout(
    features: np.ndarray,
    config: WarehouseRandomGeneratorConfig,
    representation_override: str | None = None,
):
    assert config.agent_idxs is not None
    if config.agent_colors is None:
        warnings.warn("agent_colors is None, using -1 as colors")
        config.agent_colors = [-1] * len(config.agent_idxs)

    assert config.goal_idxs is not None
    assert config.goal_colors is not None

    representation = representation_override
    if representation is None:
        representation = config.representation

    if representation == "image":
        return storage_to_layout_image(
            shelf_im=features,
            agent_idxs=config.agent_idxs,
            agent_colors=config.agent_colors,
            goal_idxs=config.goal_idxs,
            goal_colors=config.goal_colors,
        )
    elif representation == "flat":
        return storage_to_layout_flat(
            features=features,
            size=config.size,
            n_colors=config.n_colors,
            n_shelves=config.n_shelves,
            agent_idxs=config.agent_idxs,
            agent_colors=config.agent_colors,
            goal_idxs=config.goal_idxs,
            goal_colors=config.goal_colors,
        )
    elif representation == "graph":
        features = features.flatten()
        return storage_to_layout_flat(
            features=features,
            size=config.size,
            n_colors=config.n_colors,
            n_shelves=config.n_shelves,
            agent_idxs=config.agent_idxs,
            agent_colors=config.agent_colors,
            goal_idxs=config.goal_idxs,
            goal_colors=config.goal_colors,
        )
