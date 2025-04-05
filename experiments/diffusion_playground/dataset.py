import os

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from tensordict import TensorDict

from diffusion_co_design.utils.constants import EXPERIMENT_DIR
from diffusion_co_design.rware.design import ScenarioConfig
from diffusion_co_design.bin.train_rware import (
    DesignerRegistry,
    DesignerConfig,
)
from diffusion_co_design.rware.env import create_batched_env

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
    policy,
    dataset_size: int,
    num_parallel_collection: int,
    device: str,
):
    _, env_designer = DesignerRegistry.get(
        DesignerConfig(type="random"),
        scenario,
        working_dir,
        environment_batch_size=32,
        device=device,
    )

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
    policy,
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
            policy=policy,
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


def convert_to_pos_colors(data: torch.Tensor, n_shelves: int):
    # image of shape (batch_size, color, x, y)
    batch_size, n_colors, x, y = data.shape

    pos = torch.zeros(batch_size, n_shelves, 2)
    colors = torch.zeros(batch_size, n_shelves, n_colors)

    for i in range(batch_size):
        image = data[i]
        shelf_exists = torch.nonzero(image)
        for j in range(shelf_exists.shape[0]):
            color, x, y = shelf_exists[j]
            pos[i, j] = torch.tensor([x, y])
            colors[i, j] = torch.eye(n_colors)[color]

    return pos.to(data.device), colors.to(data.device)


class CollateFn:
    def __init__(self, cfg, device):
        self.n_shelves = cfg.n_shelves
        self.size = cfg.size
        self.device = device

    def __call__(self, batch):
        images = torch.stack([x[0] for x in batch])
        labels = torch.stack([x[1] for x in batch])

        pos, colors = convert_to_pos_colors(images, self.n_shelves)
        pos = (pos / self.size) * 2 - 1

        return (
            (
                pos.to(dtype=torch.float32, device=self.device),
                colors.to(dtype=torch.float32, device=self.device),
            ),
            labels.to(dtype=torch.float32, device=self.device),
        )
