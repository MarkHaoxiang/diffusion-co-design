import os
from pathlib import Path

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from tensordict import TensorDict

from diffusion_co_design.common import get_latest_model
from diffusion_co_design.common.constants import EXPERIMENT_DIR
from diffusion_co_design.common.rl.util import create_batched_env
from diffusion_co_design.common.design.base import LiveDesignConsumer
from diffusion_co_design.common.design import DesignerParams
from diffusion_co_design.rware.design import RandomDesigner
from diffusion_co_design.rware.model.classifier import image_to_pos_colors
from diffusion_co_design.rware.model.rl import rware_models
from diffusion_co_design.rware.model.graph import WarehouseGNNBase
from diffusion_co_design.rware.env import create_env
from diffusion_co_design.rware.schema import TrainingConfig, ScenarioConfig

working_dir = os.path.join(EXPERIMENT_DIR, "train_rware_classifier")
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
        y_pred = sample.get("expected_reward").to(
            dtype=torch.float32, device=self.device
        )
        distillation_hint = sample.get("distillation_hint", None)
        if distillation_hint is not None:
            distillation_hint = distillation_hint.to(
                dtype=torch.float32, device=self.device
            )
            return X, y, y_pred, distillation_hint
        else:
            return X, y, y_pred


def rware_policy_return_dataset(
    scenario: ScenarioConfig,
    training_dir: str,
    dataset_size: int,
    num_parallel_collection: int,
    device: torch.device,
):
    # Create policy
    checkpoint_dir = os.path.join(training_dir, "checkpoints")
    latest_policy = get_latest_model(checkpoint_dir, "policy_")
    latest_critic = get_latest_model(checkpoint_dir, "critic_")
    training_cfg = TrainingConfig.from_file(
        os.path.join(training_dir, ".hydra", "config.yaml")
    )
    gamma = training_cfg.ppo.gamma

    env_designer = LiveDesignConsumer(
        RandomDesigner(
            designer_setting=DesignerParams(
                scenario=scenario,
                artifact_dir=Path(working_dir),
                lock=torch.multiprocessing.Lock(),
                environment_repeats=1,
            )
        )
    )

    ref_env = create_env(
        mode="reference",
        scenario=scenario,
        designer=env_designer,
        representation="image",
        device=device,
    )

    policy, critic = rware_models(ref_env, training_cfg.policy, device=device)
    policy.load_state_dict(torch.load(latest_policy, map_location=device))
    critic.load_state_dict(torch.load(latest_critic, map_location=device))
    ref_env.close()

    # Collection
    collection_env = create_batched_env(
        mode="train",
        create_env=create_env,
        num_environments=num_parallel_collection,
        scenario=scenario,
        designer=env_designer,
        device="cpu",
    )

    env_returns = ReplayBuffer(
        storage=LazyTensorStorage(max_size=dataset_size),
        sampler=SamplerWithoutReplacement(),
        batch_size=1,
    )

    discount = (gamma ** torch.linspace(0, 499, 500)).view(1, 500, 1, 1)
    with torch.no_grad():
        for _ in tqdm(range(dataset_size // num_parallel_collection)):
            rollout = collection_env.rollout(
                max_steps=scenario.max_steps, policy=policy, auto_cast_to_device=True
            )
            X = rollout.get("state")[:, 0, : scenario.n_colors]
            ep_reward = rollout.get(("next", "agents", "reward"))
            ep_reward = ep_reward * discount
            ep_reward = ep_reward.sum(dim=(1, 2, 3))

            first_obs = rollout[:, 0]
            critic_out = critic(first_obs.to(device=device))
            expected_reward = critic_out["agents", "state_value"].sum(dim=-2).squeeze()
            distillation_hint = critic_out.get(("agents", "distillation_hint"), None)

            data_dict = {
                "env": X,
                "episode_reward": ep_reward.detach().cpu(),
                "expected_reward": expected_reward.detach().cpu(),
            }
            if distillation_hint is not None:
                data_dict["distillation_hint"] = distillation_hint.detach().cpu()

            data = TensorDict(data_dict, batch_size=len(ep_reward))

            env_returns.extend(data)
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


def _batch_has_distillation_hint(batch):
    if len(batch[0]) < 4:
        return False
    return True


class CollateFn:
    def __init__(self, cfg, device):
        self.n_shelves = cfg.n_shelves
        self.size = cfg.size
        self.device = device

    def __call__(self, batch):
        images = torch.stack([x[0] for x in batch])
        labels_1 = torch.stack([x[1] for x in batch])
        labels_2 = torch.stack([x[2] for x in batch])
        if _batch_has_distillation_hint(batch):
            distillation_hint = torch.stack([x[3] for x in batch])
        else:
            distillation_hint = None

        pos, colors = image_to_pos_colors(images, self.n_shelves)
        pos = (pos / (self.size - 1)) * 2 - 1

        return (
            (
                pos.to(dtype=torch.float32, device=self.device),
                colors.to(dtype=torch.float32, device=self.device),
            ),
            labels_1.to(dtype=torch.float32, device=self.device),
            labels_2.to(dtype=torch.float32, device=self.device),
            distillation_hint.to(dtype=torch.float32, device=self.device)
            if distillation_hint
            else None,
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
        if _batch_has_distillation_hint(batch):
            X, y_1, y_2, distillation_hint = default_collate(batch)
        else:
            X, y_1, y_2 = default_collate(batch)
            distillation_hint = None

        X = X.to(dtype=torch.float32, device=self.device)
        y_1 = y_1.to(dtype=torch.float32, device=self.device)
        y_2 = y_2.to(dtype=torch.float32, device=self.device)

        X = X * 2 - 1

        return X, y_1, y_2, distillation_hint


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
