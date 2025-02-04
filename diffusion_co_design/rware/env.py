import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.envs import (
    PettingZooWrapper,
    MarlGroupMapType,
    TransformedEnv,
    RewardSum,
    SerialEnv,
    ParallelEnv,
)
from rware.pettingzoo import PettingZooWrapper as RwarePZW
from rware.warehouse import Warehouse, ObservationRegistry, RewardRegistry

from diffusion_co_design.pretrain.rware.transform import image_to_layout
from diffusion_co_design.rware.design import ScenarioConfig, Designer, FixedDesigner


class RwareCoDesignWrapper(PettingZooWrapper):
    def __init__(
        self,
        env=None,
        reset_policy: TensorDictModule | None = None,
        environment_objective=None,
        return_state=False,
        group_map=None,
        use_mask=False,
        categorical_actions=True,
        seed=None,
        done_on_any=None,
        **kwargs,
    ):
        super().__init__(
            env,
            return_state,
            group_map,
            use_mask,
            categorical_actions,
            seed,
            done_on_any,
            **kwargs,
        )
        # Hack: TorchRL messes with object attributes, so need to set in inner environment
        # Also, it's difficult to rewrite sync
        self._env._reset_policy = reset_policy
        self._env._environment_objective = environment_objective

    def _reset(self, tensordict: TensorDictBase | None = None, **kwargs):
        """Extract the layout from tensordict and pass to env"""
        reset_policy_output = None

        if tensordict is not None and self._env._environment_objective is not None:
            tensordict[("environment_design", "objective")] = (
                self._env._environment_objective
            )

        if (
            tensordict is not None
            and "environment_design" in tensordict
            and self._env._reset_policy is not None
        ):
            reset_policy_output = self._env._reset_policy(tensordict)
            tensordict.update(
                reset_policy_output, keys_to_update=reset_policy_output.keys()
            )
            layout = image_to_layout(
                tensordict.get(("environment_design", "layout_image"))
                .detach()
                .cpu()
                .numpy()
            )
            options = {"layout": layout}
        else:
            options = None

        # Maybe add the newest layout to tensordict out
        tensordict_out = super()._reset(tensordict, options=options, **kwargs)
        return tensordict_out


def create_env(
    scenario: ScenarioConfig,
    designer: Designer,
    is_eval: bool = False,
    render: bool = False,
    device: str | None = None,
):
    is_train = not is_eval
    # Define environment design policy
    # TODO: We probably want feature engineering.
    # Or does this even matter if everything is fixed for initial experiments?
    # if is_eval:  # Temp generalisation_experiment
    # design_policy = FixedDesigner(scenario)

    scenario_objective = {
        "agent_positions": torch.tensor(scenario.agent_idxs),
        "goal_idxs": torch.tensor(scenario.goal_idxs),
    }
    initial_layout = designer.generate_environment(scenario_objective)
    design_policy = designer.to_td_module()

    env = RwarePZW(
        Warehouse(
            layout=initial_layout,
            request_queue_size=5,
            render_mode="rgb_array" if render else None,
            sensor_range=3,
            max_steps=500,
            reward_type=RewardRegistry.SHAPED,
            observation_type=ObservationRegistry.IMAGE_LAYOUT,
        )
    )
    env.reset()
    env = RwareCoDesignWrapper(
        env,
        reset_policy=design_policy,
        environment_objective=scenario_objective,
        group_map=MarlGroupMapType.ALL_IN_ONE_GROUP,
        device=device,
    )
    if is_train:
        env = TransformedEnv(
            env,
            RewardSum(
                in_keys=env.reward_keys,
                out_keys=[(agent, "episode_reward") for (agent, _) in env.reward_keys],
            ),
        )
    return env


def create_batched_env(
    num_environments: int,
    scenario: ScenarioConfig,
    designer: Designer,
    is_eval: bool = False,
    device=None,
):
    def create_env_fn(render: bool = False):
        return create_env(
            scenario, designer, is_eval=is_eval, render=render, device="cpu"
        )

    eval_kwargs = [{"render": True}]
    for _ in range(num_environments - 1):
        eval_kwargs.append({})

    return ParallelEnv(
        num_workers=num_environments,
        create_env_fn=create_env_fn,
        create_env_kwargs=eval_kwargs if is_eval else {},
        device=device,
    )
    # return SerialEnv(
    #     num_workers=num_environments, create_env_fn=create_env_fn, device=device
    # )
