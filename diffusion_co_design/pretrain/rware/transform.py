import numpy as np

from rware.layout import Layout, ImageLayer


def storage_to_layout(
    shelf_im: np.ndarray,
    agent_idxs: list[int],
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
    goal_im = np.zeros((1, size, size))

    for idx in agent_idxs:
        agent_im[0, idx // size, idx % size] = 1
    for idx, color in zip(goal_idxs, goal_colors):
        goal_im[0, idx // size, idx % size] = color + 1
        shelf_im[:, idx // size, idx % size] = 0

    im = np.concat((shelf_im, agent_im, goal_im), axis=0)
    layout = Layout.from_image(
        image=im,
        image_layers=[ImageLayer.STORAGE] * n_colors
        + [ImageLayer.AGENTS, ImageLayer.GOALS],
    )
    return layout
