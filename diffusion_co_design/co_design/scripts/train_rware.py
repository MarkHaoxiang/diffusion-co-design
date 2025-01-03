# Adapted from https://pytorch.org/rl/stable/tutorials/multiagent_ppo.html
import os
from enum import Enum

# Torch, TorchRL, TensorDict
import torch
from torch.distributions import Categorical
from tensordict.nn import TensorDictModule
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv, MarlGroupMapType
from torchrl.envs.utils import check_env_specs
from torchrl.modules import MultiAgentMLP, ProbabilisticActor
from torchrl.objectives import ClipPPOLoss, ValueEstimators

# Config Management
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

# Rware
from rware.pettingzoo import PettingZooWrapper as RwarePZW
from rware.warehouse import Warehouse

from tqdm import tqdm

from diffusion_co_design.diffusion.datasets.rware.transform import image_to_layout
from diffusion_co_design.utils.pydra import omega_to_pydantic
from diffusion_co_design.co_design.rware.design import RandomDesigner
from diffusion_co_design.co_design.rware.env import RwareCoDesignWrapper

# Devices
device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")


class ScenarioConfig(BaseModel):
    size: int
    n_shelves: int
    agent_idxs: list[int]
    goal_idxs: list[int]


class TrainingConfig(BaseModel):
    # Problem definition: Built with diffusion.datasets.rware.generate
    designer: str
    scenario_dir: str
    # Sampling and training
    frames_per_batch: int = 1_000  # Number of frames per training iteration
    n_iters: int = 10  # Number of training iterations
    num_epochs: int = 30  # Number of optimization steps per training iteration
    minibatch_size: int = 400  # Size of the mini-batches in each optimization step
    ppo_lr: float = 3e-4  # Learning rate
    max_grad_norm: float = 1.0  # Maximum norm for the gradients
    # PPO
    clip_epsilon: float = 0.2  # clip value for PPO loss
    gamma: float = 0.99  # discount factor
    lmbda: float = 0.9  # lambda for generalised advantage estimation
    entropy_eps: float = 1e-4  # coefficient of the entropy term in the PPO loss

    @property
    def total_frames(self) -> int:
        return self.frames_per_batch * self.n_iters


def train(cfg: TrainingConfig):
    # Load scenario config
    scenario = OmegaConf.load(os.path.join(cfg.scenario_dir, "config.yaml"))
    scenario: ScenarioConfig = omega_to_pydantic(scenario, ScenarioConfig)

    # Define environment design policy
    # TODO: We probably want feature engineering.
    # Or does this even matter if everything is fixed for initial experiments?
    scenario_objective = {
        "agent_positions": torch.tensor(scenario.agent_idxs),
        "goal_idxs": torch.tensor(scenario.goal_idxs),
    }
    design_policy = None  # TODO

    # Build environment
    initial_layout = image_to_layout(
        RandomDesigner(
            size=scenario.size,
            n_shelves=scenario.n_shelves,
            agent_idxs=scenario.agent_idxs,
            goal_idxs=scenario.goal_idxs,
        )(None)
    )
    env = RwarePZW(Warehouse(layout=initial_layout))
    env.reset()
    env = RwareCoDesignWrapper(
        env,
        reset_policy=design_policy,
        environment_objective=scenario_objective,
        group_map=MarlGroupMapType.ALL_IN_ONE_GROUP,
    )
    env = TransformedEnv(
        env,
        RewardSum(
            in_keys=env.reward_keys,
            out_keys=[(agent, "episode_reward") for (agent, _) in env.reward_keys],
        ),
    )
    check_env_specs(env)

    # Policy, critic
    policy_net = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[
                -1
            ],  # n_obs_per_agent
            n_agent_outputs=env.action_spec.space.n,  # n_actions_per_agents
            n_agents=env.num_agents,
            centralised=False,  # the policies are decentralised (ie each agent will act from its observation)
            share_params=True,
            device=device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        ),
    )

    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "logits")],
    )

    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=[("agents", "logits")],
        out_keys=[env.action_key],
        distribution_class=Categorical,
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
    )

    critic_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=1,  # 1 value per agent
        n_agents=env.num_agents,
        centralised=True,
        share_params=True,
        device=device,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
    )

    critic = TensorDictModule(
        module=critic_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "state_value")],
    )

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

    pbar = tqdm(total=cfg.n_iters, desc="episode_reward_mean = 0")

    episode_reward_mean_list = []
    for tensordict_data in collector:
        tensordict_data.set(
            ("next", "agents", "done"),
            tensordict_data.get(("next", "done"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
        )
        tensordict_data.set(
            ("next", "agents", "terminated"),
            tensordict_data.get(("next", "terminated"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
        )
        # We need to expand the done and terminated to match the reward shape (this is expected by the value estimator)

        with torch.no_grad():
            GAE(
                tensordict_data,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )  # Compute GAE and add it to the data

        data_view = tensordict_data.reshape(
            -1
        )  # Flatten the batch size to shuffle data
        replay_buffer.extend(data_view)

        for _ in range(cfg.num_epochs):
            for _ in range(cfg.frames_per_batch // cfg.minibatch_size):
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
                )  # Optional

                optim.step()
                optim.zero_grad()

        collector.update_policy_weights_()

        # Logging
        done = tensordict_data.get(("next", "agents", "done"))
        episode_reward_mean = (
            tensordict_data.get(("next", "agents", "episode_reward"))[done]
            .mean()
            .item()
        )
        episode_reward_mean_list.append(episode_reward_mean)
        pbar.set_description(
            f"episode_reward_mean = {episode_reward_mean}", refresh=False
        )
        pbar.update()


@hydra.main(version_base=None, config_path="configs", config_name="testing")
def run(config: DictConfig):
    print(f"Running job {HydraConfig.get().job.name}")
    config = omega_to_pydantic(config, TrainingConfig)
    train(config)


if __name__ == "__main__":
    run()
