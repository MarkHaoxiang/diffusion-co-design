import os
import numpy as np
import PIL.Image

from diffusion_co_design.pretrain.rware.generate import (
    colors,
    COLOR_ORDER,
)
from diffusion_co_design.utils import OUTPUT_DIR
from rware.layout import Layout


def rgb_to_layout(im: np.ndarray, agent_idxs=None, goal_idxs=None) -> Layout:
    # Round to nearest colors
    w, h, _ = im.shape
    flattened = im.reshape(-1, 3)
    distances = flattened[:, None, :] - colors[None, :, :]
    distances = np.linalg.norm(distances, axis=2)
    closest = np.argmin(distances, axis=1).reshape(w, h)
    n = len(COLOR_ORDER)
    image = (np.arange(n) == closest[..., None]).astype(int)
    assert np.isclose(image.sum(axis=-1).max(), 1)
    assert np.isclose(image.sum(axis=-1).min(), 1)
    image = image.transpose(2, 0, 1)[:-1, :, :]
    shelf_im = image[0]
    size = shelf_im.shape[0]

    # Force override using agent_idxs and goal_idxs
    if agent_idxs is not None:
        agent_im = np.zeros_like(shelf_im)
        for idx in agent_idxs:
            agent_im[idx // size, idx % size] = 1
            shelf_im[idx // size, idx % size] = 0
    else:
        agent_im = image[1]
    if goal_idxs is not None:
        goal_im = np.zeros_like(shelf_im)
        for idx in goal_idxs:
            goal_im[idx // size, idx % size] = 1
            shelf_im[idx // size, idx % size] = 0
        im = np.stack((shelf_im, agent_im, goal_im))
    else:
        goal_im = image[2]

    im = np.stack((shelf_im, agent_im, goal_im))
    layout = Layout.from_image(im, image_layers=COLOR_ORDER[:-1])
    return layout


def storage_to_layout(shelf_im: np.ndarray, agent_idxs, goal_idxs):
    # Construct agent and goal
    shelf_im = shelf_im.squeeze()
    size = shelf_im.shape[0]
    agent_im = np.zeros_like(shelf_im)
    for idx in agent_idxs:
        agent_im[idx // size, idx % size] = 1
    goal_im = np.zeros_like(shelf_im)
    for idx in goal_idxs:
        goal_im[idx // size, idx % size] = 1
    im = np.stack((shelf_im, agent_im, goal_im))
    layout = Layout.from_image(im, image_layers=COLOR_ORDER[:-1])
    return layout


def storage_to_rgb(im: np.ndarray, agent_idxs, goal_idxs, channel_first: bool = True):
    # [batch_size, 1, w, h]
    unbatched = False
    if len(im.shape) == 3:
        unbatched = True
        im = np.expand_dims(im, axis=0)

    size = im.shape[2]
    im = im.squeeze(axis=1)
    rgb = np.zeros((im.shape[0], size, size, 3), dtype=np.uint8)
    # Shelves
    rgb[im == 1] = colors[0]
    # Agents, Goals
    for idx in agent_idxs:
        rgb[:, idx // size, idx % size] = colors[1]
    for idx in goal_idxs:
        rgb[:, idx // size, idx % size] = colors[2]
    if channel_first:
        rgb = rgb.transpose(0, 3, 1, 2)
    if unbatched:
        rgb = rgb.squeeze(axis=0)
    return rgb


if __name__ == "__main__":
    # Run a test on the generated images

    # path = os.path.join(OUTPUT_DIR, "diffusion_datasets/default/0000000.png")
    # with open(path, "rb") as f:
    #     im = PIL.Image.open(f)
    #     im = np.array(im)

    # layout = rgb_to_layout(im)

    # print(f"Number of agents: {len(layout._agents)}")
    # print(f"Number of goals: {len(layout.goals)}")
    # print(
    #     f"Number of shelves: {(layout.grid_size[0] * layout.grid_size[1]) - layout.highways.sum()}"
    # )

    path = os.path.join(
        OUTPUT_DIR, "diffusion_datasets/rware_8_20_3_2_corners/environments.npy"
    )
    im = np.load(path)[0]

    layout = storage_to_layout(im, [42, 37, 8], [0, 63])

    print(f"Number of agents: {len(layout._agents)}")
    print(f"Number of goals: {len(layout.goals)}")
    print(layout.goals)
    print(
        f"Number of shelves: {(layout.grid_size[0] * layout.grid_size[1]) - layout.highways.sum()}"
    )
