import torch
import numpy as np
import warnings
from scipy.optimize import linear_sum_assignment

from rware.layout import Layout, ImageLayer
from diffusion_co_design.pretrain.rware.generate import (
    WarehouseRandomGeneratorConfig,
    get_position,
)


def graph_projection_constraint(cfg: WarehouseRandomGeneratorConfig):
    def _graph_projection_constraint(pos):
        B, N, _ = pos.shape
        G = cfg.size
        device = pos.device

        # Move points to the discrete grid
        lin = torch.linspace(-1, 1, steps=G, device=device)
        yy, xx = torch.meshgrid(lin, lin, indexing="ij")

        grid = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # [G^2, 2]
        # Remove goal positions from grid
        exclude_list = []
        assert cfg.goal_idxs is not None
        for idx in cfg.goal_idxs:
            x, y = get_position(idx, cfg.size)
            exclude_list.append([x, y])
        exclude = (
            2 * (torch.tensor(exclude_list, device=device) / (cfg.size - 1)) - 1
        )  # [M, 2]
        mask = torch.any(torch.cdist(grid, exclude, p=1) < 1e-5, dim=-1)
        grid = grid[~mask]

        assert N <= grid.shape[0], "More points than available grid cells"
        assert grid.shape[0] == cfg.size**2 - cfg.n_goals

        distance_limit = 1 / (G - 1) - 1e-5

        target = []
        for b in range(B):
            x_b = pos[b]

            cost = (
                torch.cdist(
                    x_b.unsqueeze(0), grid.unsqueeze(0), p=1
                )  # Taxicab distance
                .squeeze(0)
                .cpu()
                .numpy()
            )

            _, col_ind = linear_sum_assignment(cost[:, :])

            matched = grid[col_ind]
            target.append(matched)

        # Minimise movement
        target = torch.stack(target, dim=0).to(device)
        delta = target - pos
        original_sign = torch.sign(delta)
        delta = delta - distance_limit * original_sign
        new_sign = torch.sign(delta)
        delta = delta * (original_sign == new_sign).float()
        target = pos + delta

        return target

    return _graph_projection_constraint


def image_projection_constraint(cfg: WarehouseRandomGeneratorConfig):
    def _image_projection_constraint(x):
        target = [cfg.n_shelves // cfg.n_colors for _ in range(cfg.n_colors)]
        remainder = cfg.n_shelves % cfg.n_colors
        for i in range(remainder):
            target[i] += 1

        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)
        mask = torch.full_like(x_flat, -1)

        for c, k in enumerate(target):
            _, indices = torch.topk(x_flat[:, c, :], k=k, dim=1)
            mask[:, c, :].scatter_(1, indices, 1)

        return mask.view(B, C, H, W)

    return _image_projection_constraint


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
        agent_im[0, *get_position(idx, size)] = 1
        agent_colors_im[0, *get_position(idx, size)] = color + 1
    for idx, color in zip(goal_idxs, goal_colors):
        goal_im[0, *get_position(idx, size)] = color + 1
        shelf_im[:, *get_position(idx, size)] = 0

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

    if auto_add_colors:
        colors = []
        shelves_per_color = n_shelves // n_colors
        remainder = n_shelves % n_colors
        for i in range(n_colors):
            if remainder > 0:
                n = shelves_per_color + 1
                remainder -= 1
            else:
                n = shelves_per_color

            colors += [i] * n

    for i in range(n_shelves):
        start = i * feature_dim_shelf
        x = np.clip(round(features[start]), 0, size - 1)
        y = np.clip(0, round(features[start + 1]), size - 1)

        if auto_add_colors:
            color = colors[i]
            # color = i % n_colors
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
