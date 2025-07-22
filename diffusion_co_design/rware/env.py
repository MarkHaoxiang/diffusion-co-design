from typing import Literal

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs import (
    PettingZooWrapper,
    MarlGroupMapType,
    TransformedEnv,
    RewardSum,
)
from rware.pettingzoo import PettingZooWrapper as RwarePZW
from rware.warehouse import Warehouse, ObservationRegistry, RewardRegistry, ImageLayer

from diffusion_co_design.common.design import DesignConsumer
from diffusion_co_design.rware.diffusion.generate import generate
from diffusion_co_design.rware.diffusion.transform import storage_to_layout
from diffusion_co_design.rware.schema import ScenarioConfig


class RwareCoDesignWrapper(PettingZooWrapper):
    def __init__(
        self,
        env=None,
        reset_policy: TensorDictModule | None = None,
        scenario_cfg: ScenarioConfig | None = None,
        representation=None,
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
        self._env.representation = representation
        assert scenario_cfg is not None
        self._env._scenario_cfg = scenario_cfg

    def _reset(self, tensordict: TensorDict | None = None, **kwargs):
        """Extract the layout from tensordict and pass to env"""

        if "layout_override" in kwargs and kwargs["layout_override"] is not None:
            layout_weights = kwargs.pop("layout_override").numpy(force=True)
            layout = storage_to_layout(
                features=layout_weights,
                config=self._env._scenario_cfg,
                representation=self._env.representation,
            )
            options = {"layout": layout}

        elif self._env._reset_policy is not None:
            # Should recompute layout
            td = (
                tensordict
                if tensordict is not None
                else TensorDict({}, device=self.device)
            )
            reset_policy_output = self._env._reset_policy(td)
            td.update(reset_policy_output, keys_to_update=reset_policy_output.keys())

            layout = storage_to_layout(
                features=reset_policy_output.get(
                    ("environment_design", "layout_weights")
                ).numpy(force=True),
                config=self._env._scenario_cfg,
                representation=self._env.representation,
            )
            options = {"layout": layout}
        else:
            options = None

        tensordict_out = super()._reset(tensordict, options=options, **kwargs)

        if self.return_state:
            tensordict_out["state"] = self._env.state()

        return tensordict_out

    def _step(self, tensordict):
        tensordict_out = super()._step(tensordict)
        if self.return_state:
            tensordict_out["state"] = self._env.state()
        return tensordict_out


def create_env(
    mode: Literal["train", "eval", "reference"],
    scenario: ScenarioConfig,
    designer: DesignConsumer,
    representation: Literal["image", "graph"] = "image",
    render: bool = False,
    device: torch.device = torch.device("cpu"),
):
    initial_layout = storage_to_layout(
        generate(
            size=scenario.size,
            n_shelves=scenario.n_shelves,
            goal_idxs=scenario.goal_idxs,
            n_colors=scenario.n_colors,
            training_dataset=False,
            representation=representation,
        )[0],
        config=scenario,
        representation=representation,
    )

    env = RwarePZW(
        Warehouse(
            layout=initial_layout,
            request_queue_size=5,
            render_mode="rgb_array" if render else None,
            sensor_range=3,
            max_steps=scenario.max_steps,
            max_inactivity_steps=None,
            reward_type=RewardRegistry.SHAPED,
            observation_type=ObservationRegistry.SHAPED,
            image_observation_layers=[
                ImageLayer.STORAGE,
                ImageLayer.GOALS_COLOR_ONE_HOT,
                ImageLayer.ACCESSIBLE,
                ImageLayer.REQUESTS,
            ],
            return_info=[],
        )
    )

    env.reset()
    env = RwareCoDesignWrapper(
        env,
        reset_policy=designer.to_td_module(),
        representation=representation,
        group_map=MarlGroupMapType.ALL_IN_ONE_GROUP,
        scenario_cfg=scenario,
        return_state=True,
        device=device,
    )
    if mode == "train":
        env = TransformedEnv(
            env,
            RewardSum(
                in_keys=env.reward_keys,
                out_keys=[(agent, "episode_reward") for (agent, _) in env.reward_keys],
            ),
        )
    return env


def render_env(theta, scenario, representation):
    layout = storage_to_layout(theta, scenario, representation=representation)
    warehouse = Warehouse(layout=layout, render_mode="rgb_array")
    im = warehouse.render()
    warehouse.close()
    return im
