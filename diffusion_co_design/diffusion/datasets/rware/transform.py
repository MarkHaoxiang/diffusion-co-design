import os
import numpy as np
import PIL.Image

from diffusion_co_design.diffusion.datasets.rware.generate import (
    colors,
    COLOR_ORDER,
)
from diffusion_co_design.utils import BASE_DIR
from rware.layout import Layout


def image_to_layout(im: np.ndarray) -> Layout:

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
    layout = Layout.from_image(image, image_layers=COLOR_ORDER[:-1])
    # TODO: Preprocessing to ensure image fits layout constraints.
    return layout


if __name__ == "__main__":
    # Run a test on the generated images

    path = os.path.join(BASE_DIR, "diffusion_datasets/default/0000000.png")
    with open(path, "rb") as f:
        im = PIL.Image.open(f)
        im = np.array(im)

    layout = image_to_layout(im)

    print(f"Number of agents: {len(layout._agents)}")
    print(f"Number of goals: {len(layout.goals)}")
    print(
        f"Number of shelves: {(layout.grid_size[0] * layout.grid_size[1]) -layout.highways.sum()}"
    )
