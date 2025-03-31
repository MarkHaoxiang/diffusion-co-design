import os
import shutil
import pickle as pkl

import torch
import torch.nn as nn
from tqdm import tqdm
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_scatter import scatter
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


class WarehouseDiffusionLayer(MessagePassing):
    def __init__(
        self,
        embedding_dim: int = 32,
        timestep_embedding_size: int = 32,
        normalise_pos: bool = True,
        pos_aggr: str = "add",
    ):
        super().__init__(aggr="add")

        self.normalise_pos = normalise_pos

        self.message_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2 + timestep_embedding_size, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
        )

        self.pos_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
        )
        self.pos_aggr = pos_aggr

    def forward(self, x, edge_index, pos, timestep_embedding):
        out = self.propagate(
            edge_index, x=x, pos=pos, timestep_embedding=timestep_embedding
        )

        return out

    def message(self, x_i, x_j, pos_i, pos_j, timestep_embedding):
        # Position encodings
        pos_diff = pos_i - pos_j
        radial = torch.sum(pos_diff**2, dim=-1, keepdim=True)
        if self.normalise_pos:
            norm = torch.sqrt(radial).detach() + 1e-6
            pos_diff /= norm

        # Node encodings
        msg = torch.cat([x_i, x_j, radial, timestep_embedding], dim=-1)
        msg = self.message_mlp(msg)
        return (msg, pos_diff)

    def aggregate(self, inputs, index, ptr=None, aim_size=None):
        msg, pos_diff = inputs
        pos_vec = pos_diff * self.pos_mlp(msg)
        aggr_h = scatter(msg, index, dim=0, dim_size=aim_size, reduce=self.aggr)
        aggr_pos = scatter(
            pos_vec, index, dim=0, dim_size=aim_size, reduce=self.pos_aggr
        )
        return aggr_h, aggr_pos

    def update(self, inputs, x, pos):
        aggr_h, aggr_pos = inputs
        upd_out = self.node_mlp(torch.cat([x, aggr_h], dim=-1))
        return upd_out + x, aggr_pos + pos


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
