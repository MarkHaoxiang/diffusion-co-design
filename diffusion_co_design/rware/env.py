import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.envs import (
    PettingZooWrapper,
    MarlGroupMapType,
    TransformedEnv,
    RewardSum,
)

from pydantic import BaseModel

from diffusion_co_design.pretrain.rware.transform import image_to_layout
from diffusion_co_design.co_design.rware.design import RandomDesigner
from rware.pettingzoo import PettingZooWrapper as RwarePZW
from rware.warehouse import Warehouse, ObservationType


class ScenarioConfig(BaseModel):
    size: int
    n_shelves: int
    agent_idxs: list[int]
    goal_idxs: list[int]


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


def rware_env(scenario: ScenarioConfig, is_eval: bool = False, device: str = None):
    design_policy = None

    # Define environment design policy
    # TODO: We probably want feature engineering.
    # Or does this even matter if everything is fixed for initial experiments?
    scenario_objective = {
        "agent_positions": torch.tensor(scenario.agent_idxs),
        "goal_idxs": torch.tensor(scenario.goal_idxs),
    }

    initial_layout = image_to_layout(
        RandomDesigner(
            size=scenario.size,
            n_shelves=scenario.n_shelves,
            agent_idxs=scenario.agent_idxs,
            goal_idxs=scenario.goal_idxs,
        )(None)
    )
    env = RwarePZW(
        Warehouse(
            layout=initial_layout,
            request_queue_size=5,
            render_mode="rgb_array" if is_eval else None,
            observation_type=ObservationType.IMAGE_LAYOUT,
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
    if not is_eval:
        env = TransformedEnv(
            env,
            RewardSum(
                in_keys=env.reward_keys,
                out_keys=[(agent, "episode_reward") for (agent, _) in env.reward_keys],
            ),
        )
    return env
