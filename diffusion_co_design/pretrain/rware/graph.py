import os
import shutil
import pickle as pkl

import torch
from tqdm import tqdm
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from rware.layout import Layout
from rware.entity import Entity

from diffusion_co_design.pretrain.rware.generate import (
    generate,
    WarehouseRandomGeneratorConfig,
)
from diffusion_co_design.pretrain.rware.transform import storage_to_layout
from diffusion_co_design.utils import omega_to_pydantic, OUTPUT_DIR


def generate_layout_graph(layout: Layout) -> Data:
    agents = layout.reset_agents(0)
    shelves = layout.reset_shelves()
    goals = layout.goals

    num_types = 3
    num_colors = layout.num_colors

    # Node feature matrix
    # one_hot feature of type
    # one_hot feature of color
    num_nodes = len(agents) + len(shelves) + len(goals)
    x = torch.zeros(size=(num_nodes, 3 + num_colors))

    index = 0
    for agent in agents:
        x[index, 0] = 1  # Agent
        if agent.color == -1:
            x[index, num_types:] = 1
        else:
            x[index, num_types + agent.color] = 1

        index += 1

    for shelf in shelves:
        x[index, 1] = 1  # Shelf
        x[index, num_types + shelf.color] = 1

        index += 1

    for goal in goals:
        x[index, 2] = 1  # Goal
        x[index, num_types + goal.color] = 1

        index += 1

    # Positions
    pos = torch.zeros((num_nodes, 2))
    entities: list[Entity] = agents + shelves + goals
    for i, entity in enumerate(entities):
        pos[i, 0] = entity.pos.x
        pos[i, 1] = entity.pos.y

    # Connectivity
    def fully_connected(n):
        row, col = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
        edge_index = torch.stack([row.flatten(), col.flatten()], dim=0)
        return edge_index

    # Connectivity
    edge_index = fully_connected(num_nodes)

    return Data(x=x, edge_index=edge_index, pos=pos)


def generate_run_graph(cfg: WarehouseRandomGeneratorConfig):
    size = cfg.size

    data_dir = os.path.join(OUTPUT_DIR, "diffusion_datasets_graph", cfg.name)
    if os.path.exists(data_dir):
        shutil.rmtree(path=data_dir)
    os.makedirs(data_dir)
    print(f"Generating dataset at {data_dir}")

    with open(os.path.join(data_dir, "config.yaml"), "w") as f:
        out = cfg.model_dump()
        yaml = OmegaConf.create(out)
        OmegaConf.save(yaml, f)

        assert cfg.agent_idxs is not None
        assert cfg.goal_idxs is not None
        assert cfg.agent_colors is not None
        assert cfg.goal_colors is not None

        environments = generate(
            size,
            cfg.n_shelves,
            cfg.agent_idxs,
            cfg.goal_idxs,
            n_colors=cfg.n_colors,
            n=cfg.n_samples,
            disable_tqdm=False,
            training_dataset=False,
        )

        dataset = []
        for env in tqdm(environments):
            layout = storage_to_layout(
                shelf_im=env,
                agent_idxs=cfg.agent_idxs,
                agent_colors=cfg.agent_colors,
                goal_idxs=cfg.goal_idxs,
                goal_colors=cfg.goal_colors,
            )

            data = generate_layout_graph(layout)

            dataset.append(data)

    with open(data_dir + "/environments.pkl", "wb") as fp:
        pkl.dump(environments, fp)


@hydra.main(
    version_base=None,
    config_path="../../bin/conf/scenario",
    config_name="rware_16_50_5_4_corners",
)
def run(config: DictConfig):
    print(f"Running job {HydraConfig.get().job.name}")
    generate_config: WarehouseRandomGeneratorConfig = omega_to_pydantic(
        config, WarehouseRandomGeneratorConfig
    )

    generate_run_graph(generate_config)


if __name__ == "__main__":
    run()
