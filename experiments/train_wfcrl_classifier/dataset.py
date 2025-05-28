import os

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from tensordict import TensorDict

from diffusion_co_design.common import get_latest_model
from diffusion_co_design.common.constants import EXPERIMENT_DIR
from diffusion_co_design.wfcrl.model.rl import wfcrl_models
from diffusion_co_design.wfcrl.diffusion.generator import eval_to_train
from diffusion_co_design.wfcrl.design import ScenarioConfig, Random
from diffusion_co_design.wfcrl.env import create_env, create_batched_env
from diffusion_co_design.wfcrl.schema import TrainingConfig
from diffusion_co_design.wfcrl.design import DesignerRegistry

working_dir = os.path.join(EXPERIMENT_DIR, "train_wfcrl_classifier")
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
        return X, y, y_pred


def wfcrl_policy_return_dataset(
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

    _, env_designer = DesignerRegistry.get(
        designer=Random(type="random"),
        scenario=scenario,
        artifact_dir=working_dir,
        ppo_cfg=training_cfg.ppo,
        normalisation_statistics=training_cfg.normalisation,
        device=device,
    )
    ref_env = create_env(
        mode="reference",
        scenario=scenario,
        designer=env_designer,
        render=False,
        device=device,
    )
    policy, critic = wfcrl_models(
        env=ref_env,
        cfg=training_cfg.policy,
        normalisation=training_cfg.normalisation,
        device=device,
    )
    policy.load_state_dict(torch.load(latest_policy))
    critic.load_state_dict(torch.load(latest_critic))
    ref_env.close()

    # Collection
    collection_env = create_batched_env(
        mode="train",
        num_environments=num_parallel_collection,
        scenario=scenario,
        designer=env_designer,
        device=device,
    )

    env_returns = ReplayBuffer(
        storage=LazyTensorStorage(max_size=dataset_size),
        sampler=SamplerWithoutReplacement(),
        batch_size=1,
    )
    n = training_cfg.scenario.max_steps

    discount = (gamma ** torch.linspace(0, n - 1, n)).view(1, n, 1, 1)
    with torch.no_grad():
        for _ in tqdm(range(dataset_size // num_parallel_collection)):
            rollout = collection_env.rollout(
                max_steps=scenario.max_steps, policy=policy, auto_cast_to_device=True
            )
            X = rollout.get(("state", "layout"))[:, 0].cpu()
            ep_reward = rollout.get(("next", "turbine", "reward")).cpu()
            ep_reward = ep_reward * discount
            ep_reward = ep_reward.sum(dim=(1, 2, 3))

            first_obs = rollout[:, 0]
            expected_reward = (
                critic(first_obs)["turbine", "state_value"].sum(dim=-2).squeeze()
            ).cpu()
            data = TensorDict(
                {
                    "env": X,
                    "episode_reward": ep_reward,
                    "expected_reward": expected_reward,
                },
                batch_size=len(ep_reward),
            )

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
        env_returns = wfcrl_policy_return_dataset(
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
    def __init__(self, cfg: ScenarioConfig, device):
        self.device = device
        self.cfg = cfg

    def __call__(self, batch):
        layout = torch.stack([x[0] for x in batch])
        labels_1 = torch.stack([x[1] for x in batch])
        labels_2 = torch.stack([x[2] for x in batch])

        layout = eval_to_train(layout, self.cfg)

        return (
            layout.to(dtype=torch.float32, device=self.device),
            labels_1.to(dtype=torch.float32, device=self.device),
            labels_2.to(dtype=torch.float32, device=self.device),
        )


def make_dataloader(
    dataset,
    scenario: ScenarioConfig,
    batch_size: int,
    device: str,
    **kwargs,
):
    cf = CollateFn(cfg=scenario, device=device, **kwargs)  # type: ignore

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=cf,
        persistent_workers=True,
    )
