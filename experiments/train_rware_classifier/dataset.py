import os

from tqdm import tqdm
from omegaconf import OmegaConf
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from tensordict import TensorDict

from diffusion_co_design.common import get_latest_model, omega_to_pydantic
from diffusion_co_design.common.constants import EXPERIMENT_DIR
from diffusion_co_design.rware.model.classifier import image_to_pos_colors
from diffusion_co_design.rware.model import rware_models
from diffusion_co_design.rware.design import ScenarioConfig
from experiments.train_rware.main import (
    TrainingConfig,
    DesignerRegistry,
    DesignerConfig,
)
from diffusion_co_design.rware.diffusion.graph import WarehouseGNNBase
from diffusion_co_design.rware.env import create_env, create_batched_env

working_dir = os.path.join(EXPERIMENT_DIR, "diffusion_playground")
if not os.path.exists(working_dir):
    os.makedirs(working_dir)


class EnvReturnsDataset(Dataset):
    def __init__(self, env_returns, device):
        self.env_returns = env_returns
        self.device = device

    def __len__(self):
        return len(self.env_returns)

    def __getitem__(self, idx):
        sample = self.env_returns[idx]
        X = sample.get("env").to(dtype=torch.float32, device=self.device)
        y = sample.get("episode_reward").to(dtype=torch.float32, device=self.device)
        return X, y


def rware_policy_return_dataset(
    scenario: ScenarioConfig,
    training_dir: str,
    dataset_size: int,
    num_parallel_collection: int,
    device: str,
):
    # Create policy
    checkpoint_dir = os.path.join(training_dir, "checkpoints")
    latest_policy = get_latest_model(checkpoint_dir, "policy_")
    hydra_dir = os.path.join(training_dir, ".hydra")
    training_cfg = omega_to_pydantic(
        OmegaConf.load(os.path.join(hydra_dir, "config.yaml")), TrainingConfig
    )

    _, env_designer = DesignerRegistry.get(
        DesignerConfig(type="random"),
        scenario,
        working_dir,
        environment_batch_size=32,
        device=device,
    )

    ref_env = create_env(scenario, env_designer, render=True, device=device)
    policy, _ = rware_models(ref_env, training_cfg.policy, device=device)
    policy.load_state_dict(torch.load(latest_policy))
    ref_env.close()

    # Collection
    collection_env = create_batched_env(
        num_environments=num_parallel_collection,
        scenario=scenario,
        designer=env_designer,
        is_eval=False,
        device="cpu",
    )

    env_returns = ReplayBuffer(
        storage=LazyTensorStorage(max_size=dataset_size),
        sampler=SamplerWithoutReplacement(),
        batch_size=1,
    )

    for _ in tqdm(range(dataset_size // num_parallel_collection)):
        rollout = collection_env.rollout(
            max_steps=scenario.max_steps, policy=policy, auto_cast_to_device=True
        )
        done = rollout.get(("next", "done"))
        X = rollout.get("state")[done.squeeze()]
        y = rollout.get(("next", "agents", "episode_reward")).mean(-2)[done]
        data = TensorDict({"env": X, "episode_reward": y}, batch_size=len(y))
        env_returns.extend(data)
    del rollout, done
    collection_env.close()

    return env_returns


def load_dataset(
    scenario: ScenarioConfig,
    training_dir: str,
    dataset_size: int,
    num_workers: int,
    test_proportion: float,
    device: str,
    recompute: bool = False,
):
    dataset_file = os.path.join(working_dir, f"env_returns_{dataset_size}")

    if os.path.exists(dataset_file) and not recompute:
        env_returns = ReplayBuffer(
            storage=LazyTensorStorage(max_size=dataset_size),
            sampler=SamplerWithoutReplacement(),
            batch_size=1,
        )
        env_returns.loads(dataset_file)
    else:
        env_returns = rware_policy_return_dataset(
            scenario=scenario,
            training_dir=training_dir,
            dataset_size=dataset_size,
            num_parallel_collection=num_workers,
            device=device,
        )
        env_returns.dumps(dataset_file)
    dataset = EnvReturnsDataset(env_returns, device=device)
    train_size = int((1 - test_proportion) * len(env_returns))
    eval_size = len(env_returns) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size]
    )

    return train_dataset, eval_dataset


class CollateFn:
    def __init__(self, cfg, device):
        self.n_shelves = cfg.n_shelves
        self.size = cfg.size
        self.device = device

    def __call__(self, batch):
        images = torch.stack([x[0] for x in batch])
        labels = torch.stack([x[1] for x in batch])

        pos, colors = image_to_pos_colors(images, self.n_shelves)
        pos = (pos / (self.size - 1)) * 2 - 1

        return (
            (
                pos.to(dtype=torch.float32, device=self.device),
                colors.to(dtype=torch.float32, device=self.device),
            ),
            labels.to(dtype=torch.float32, device=self.device),
        )


class CollateFnWarehouseGraph:
    def __init__(self, cfg, device):
        super().__init__()
        self._collate_fn = CollateFn(cfg, device)
        self._gnn = WarehouseGNNBase(scenario=cfg, include_color_features=True)

    def __call__(self, batch):
        data, labels = self._collate_fn(batch)
        graph = self._gnn.make_graph_batch_from_data(pos=data[0], color=data[1])[0]
        return graph, labels


class ImageCollateFn:
    def __init__(self, cfg, device):
        self.n_shelves = cfg.n_shelves
        self.size = cfg.size
        self.device = device

    def __call__(self, batch):
        X, y = default_collate(batch)

        X = X.to(dtype=torch.float32, device=self.device)
        y = y.to(dtype=torch.float32, device=self.device)

        X = X * 2 - 1

        return X, y


def make_dataloader(
    dataset,
    scenario: ScenarioConfig,
    batch_size: int,
    representation: str,
    device: str,
    **kwargs,
):
    if representation == "image":
        cf = ImageCollateFn(cfg=scenario, device=device, **kwargs)
    elif representation == "graph":
        cf = CollateFn(cfg=scenario, device=device, **kwargs)  # type: ignore
    elif representation == "graph_warehouse":
        cf = CollateFnWarehouseGraph(cfg=scenario, device=device, **kwargs)  # type: ignore

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=cf,
        persistent_workers=True,
    )
