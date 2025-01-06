# Adapted from https://pytorch.org/rl/stable/tutorials/multiagent_ppo.html
import os

# Torch, TorchRL, TensorDict
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, SamplerWithoutReplacement, LazyTensorStorage
from torchrl.modules import MultiAgentMLP
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl.envs.utils import check_env_specs
from torchrl.record import WandbLogger

# Config Management
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

# Rware
from tqdm import tqdm

from diffusion_co_design.utils.pydra import omega_to_pydantic
from diffusion_co_design.co_design.rware.env import (
    ScenarioConfig,
    rware_env,
)
from diffusion_co_design.co_design.rware.model import rware_models, PolicyConfig


# Devices
device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")


class LoggingConfig(BaseModel):
    offline: bool = False


class TrainingConfig(BaseModel):
    # Problem definition: Built with diffusion.datasets.rware.generate
    designer: str
    scenario_dir: str
    # Sampling and training
    n_iters: int = 50  # Number of training iterations
    n_epochs: int = 30  # Number of optimization steps per training iteration
    #    minibatch_size: int = 400  # Size of the mini-batches in each optimization step
    minibatch_size: int = 400  # Size of the mini-batches in each optimization step
    n_mini_batches: int = 15  # Number of mini-batches in each epoch
    ppo_lr: float = 3e-4  # Learning rate
    max_grad_norm: float = 1.0  # Maximum norm for the gradients
    # PPO
    clip_epsilon: float = 0.2  # clip value for PPO loss
    gamma: float = 0.99  # discount factor
    lmbda: float = 0.9  # lambda for generalised advantage estimation
    entropy_eps: float = 1e-4  # coefficient of the entropy term in the PPO loss
    # Policy
    policy_cfg: PolicyConfig = PolicyConfig()
    # Logging
    logging_enable: bool = True
    logging_cfg: LoggingConfig = LoggingConfig()

    @property
    def frames_per_batch(self) -> int:
        return self.n_mini_batches * self.minibatch_size

    @property
    def total_frames(self) -> int:
        return self.frames_per_batch * self.n_iters


def train(cfg: TrainingConfig):
    # Load scenario config
    scenario = OmegaConf.load(os.path.join(cfg.scenario_dir, "config.yaml"))
    scenario: ScenarioConfig = omega_to_pydantic(scenario, ScenarioConfig)

    env = rware_env(scenario, device)
    policy, critic, env = rware_models(
        env, cfg.policy_cfg, device
    )  # Env is updated to track RNN information
    check_env_specs(env)

    collector = SyncDataCollector(
        env,
        policy,
        device=device,
        storing_device=device,
        frames_per_batch=cfg.frames_per_batch,
        total_frames=cfg.total_frames,
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(
            cfg.frames_per_batch, device=device
        ),  # We store the frames_per_batch collected at each iteration
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.minibatch_size,  # We will sample minibatches of this size
    )

    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=critic,
        clip_epsilon=cfg.clip_epsilon,
        entropy_coef=cfg.entropy_eps,
        normalize_advantage=False,  # Important to avoid normalizing across the agent dimension
    )

    loss_module.set_keys(  # We have to tell the loss where to find the keys
        reward=env.reward_key,
        action=env.action_key,
        sample_log_prob=("agents", "sample_log_prob"),
        value=("agents", "state_value"),
        # These last 2 keys will be expanded to match the reward shape
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )

    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=cfg.gamma, lmbda=cfg.lmbda
    )  # We build GAE
    GAE = loss_module.value_estimator

    optim = torch.optim.Adam(loss_module.parameters(), cfg.ppo_lr)

    # Logging
    pbar = tqdm(total=cfg.n_iters, desc="episode_reward_mean = 0")
    losses_objective = torch.zeros((cfg.n_epochs, cfg.n_mini_batches))
    losses_critic = torch.zeros((cfg.n_epochs, cfg.n_mini_batches))
    losses_entropy = torch.zeros((cfg.n_epochs, cfg.n_mini_batches))
    log, collected_frames = {}, 0
    if cfg.logging_enable:
        config = dict(cfg)
        config["scenario"] = dict(scenario)
        logger = WandbLogger(
            exp_name=cfg.scenario_dir,
            project="diffusion-co-design",
            offline=cfg.logging_cfg.offline,
            config=config,
        )
    else:
        logger = None

    # Main Training Loop
    for td in collector:
        # Reshape to match value estimator
        batch_shape = td.get_item_shape(("next", env.reward_key))
        for key in ("done", "terminated"):
            td.set(
                ("next", "agents", key),
                td.get(("next", key)).unsqueeze(-1).expand(batch_shape),
            )

        with torch.no_grad():
            GAE(
                td,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )  # Compute GAE and add it to the data

        data_view = td.reshape(-1)  # Flatten the batch size to shuffle data
        collected_frames += data_view.batch_size[0]
        replay_buffer.extend(data_view)

        for i in range(cfg.n_epochs):
            for j in range(cfg.n_mini_batches):
                subdata = replay_buffer.sample()

                loss_vals = loss_module(subdata)

                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                loss_value.backward()

                torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), cfg.max_grad_norm
                )

                optim.step()
                optim.zero_grad()

                losses_objective[i, j] = loss_vals["loss_objective"].item()
                losses_critic[i, j] = loss_vals["loss_critic"].item()
                losses_entropy[i, j] = loss_vals["loss_entropy"].item()

        collector.update_policy_weights_()

        # Logging
        log["frame"] = collected_frames
        log["training_rewards"] = (
            td.get(("next", "agents", "episode_reward"))[
                td.get(("next", "agents", "done"))
            ]
            .mean()
            .item()
        )
        log["losses_objective"] = losses_objective.mean().item()
        log["losses_critic"] = losses_critic.mean().item()
        log["losses_entropy"] = losses_entropy.mean().item()
        if logger:
            for key, value in log.items():
                logger.log_scalar(key, value)

            pbar.update()


@hydra.main(version_base=None, config_path="configs", config_name="testing")
def run(config: DictConfig):
    print(f"Running job {HydraConfig.get().job.name}")
    config = omega_to_pydantic(config, TrainingConfig)
    train(config)


if __name__ == "__main__":
    run()
