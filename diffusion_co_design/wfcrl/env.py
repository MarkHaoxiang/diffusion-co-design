import os
import shutil
from typing import Literal
import math
import copy

import torch
import numpy as np
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs import (
    PettingZooWrapper,
    TransformedEnv,
    RewardSum,
    Compose,
    RemoveEmptySpecs,
)
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pettingzoo.utils.conversions import aec_to_parallel
import gymnasium.spaces as spaces

from wfcrl.environments.registration import get_default_control, validate_case
from wfcrl.environments.data_cases import FlorisCase
from wfcrl.interface import FlorisInterface
from wfcrl.multiagent_env import MAWindFarmEnv
from wfcrl.rewards import DoNothingReward
from wfcrl.mdp import WindFarmMDP

from diffusion_co_design.common.design import DesignConsumer
from diffusion_co_design.wfcrl.static import GROUP_NAME
from diffusion_co_design.wfcrl.design import make_generate_fn
from diffusion_co_design.wfcrl.schema import ScenarioConfig


class DesignableMAWindFarmEnv(MAWindFarmEnv):
    def __init__(
        self,
        interface,
        farm_case,
        controls,
        continuous_control=True,
        reward_shaper=None,
        start_iter=0,
        max_num_steps=500,
        load_coef=0.1,
        scenario: ScenarioConfig | None = None,
        render_mode: Literal["rgb_array"] | None = None,
    ):
        if reward_shaper is None:
            reward_shaper = DoNothingReward()
        max_num_steps = max_num_steps + 1  # Offset by 1 bug fix
        super().__init__(
            interface=interface,
            farm_case=farm_case,
            controls=controls,
            continuous_control=continuous_control,
            reward_shaper=reward_shaper,
            start_iter=start_iter,
            max_num_steps=max_num_steps,
            load_coef=load_coef,
        )
        self.interface_cls = interface
        self.start_iter = start_iter

        self.state_space["layout"] = spaces.Box(
            low=0,
            high=np.inf,
            shape=(self.num_turbines, 2),
            dtype=np.float32,
        )

        # Used for rendering only
        self.scenario = scenario
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        # Possibly override xcoords and ycoords
        if options is not None:
            old_farm_case = self.mdp.farm_case
            new_farm_case = copy.copy(old_farm_case)
            if "xcoords" in options:
                coords = options.get("xcoords")
                if isinstance(coords, torch.Tensor):
                    coords = coords.numpy(force=True)
                if isinstance(coords, np.ndarray):
                    coords = coords.tolist()
                assert len(coords) == old_farm_case.num_turbines
                new_farm_case.xcoords = coords
            if "ycoords" in options:
                coords = options.get("ycoords")
                if isinstance(coords, torch.Tensor):
                    coords = coords.numpy(force=True)
                if isinstance(coords, np.ndarray):
                    coords = coords.tolist()
                assert len(coords) == old_farm_case.num_turbines
                new_farm_case.ycoords = coords

            if isinstance(self.mdp.interface, FlorisInterface):
                # Remove cached file
                os.remove(self.mdp.interface.simul_file)
                shutil.rmtree(os.path.dirname(self.mdp.interface.simul_file))

            # Override MDP
            self.mdp = WindFarmMDP(
                interface=self.interface_cls,
                farm_case=new_farm_case,
                controls=self.controls,
                continuous_control=self.continuous_control,
                start_iter=self.start_iter,
                horizon=self.start_iter + self.max_num_steps,
            )
            self.farm_case = new_farm_case
        return super().reset(seed, options)

    def state(self):
        state = super().state()
        # Add layout to state
        state["layout"] = np.stack(
            [self.mdp.farm_case.xcoords, self.mdp.farm_case.ycoords], axis=-1
        ).astype(np.float32)
        return state

    def render(self):
        if self.render_mode is None:
            return None
        assert self.render_mode == "rgb_array"

        state = self.state()

        xcoords = self.mdp.farm_case.xcoords
        ycoords = self.mdp.farm_case.ycoords
        if self.scenario is not None:
            map_width = self.scenario.map_x_length
            map_height = self.scenario.map_y_length
        else:
            map_width = max(xcoords)
            map_height = max(ycoords)

        y_mult = max(1, map_height / map_height)
        x_mult = max(1, map_width / map_height)
        fig, ax = plt.subplots(figsize=(5 * x_mult, 5 * y_mult))
        canvas = FigureCanvas(fig)

        # Turbines
        if self.scenario is not None:
            radius = self.scenario.min_distance_between_turbines / 2
        else:
            radius = 50

        coords = np.stack([xcoords, ycoords], axis=-1)
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.7, edgecolors="k", zorder=2)

        for x, y in coords:
            circle = Circle(
                (x, y),
                radius=radius,
                edgecolor="green",
                facecolor="none",
                linewidth=1.2,
                zorder=1,
            )
            ax.add_patch(circle)

        # Wind
        wind_speed = state["wind_speed"] / 28 * radius
        wind_direction = state["wind_direction"]
        wind_direction = np.deg2rad(wind_direction)
        dx = wind_speed * np.sin(wind_direction) * -1
        dy = wind_speed * np.cos(wind_direction) * -1

        ax.quiver(coords[:, 0], coords[:, 1], dx, dy, color="blue", width=0.005)

        # Yaw
        yaw = state["yaw"] + state["wind_direction"]
        yaw = np.deg2rad(yaw)
        dx = radius * np.sin(yaw) * -1
        dy = radius * np.cos(yaw) * -1
        ax.quiver(coords[:, 0], coords[:, 1], dx, dy, color="red", width=0.005)

        ax.grid(True)
        ax.set_xlim(0, map_width)
        ax.set_ylim(0, map_height)

        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(
            height, width, 4
        )
        img = img[:, :, :3]
        plt.close(fig)

        return img

    def _build_agent_spaces(self):
        super()._build_agent_spaces()
        for i, agent in enumerate(self.possible_agents):
            self._obs_spaces[agent]["layout"] = spaces.Box(
                low=0, high=np.inf, shape=(2,), dtype=np.float32
            )


class WfcrlCoDesignWrapper(PettingZooWrapper):
    def __init__(
        self,
        env=None,
        reset_policy: TensorDictModule | None = None,
        scenario_cfg=None,
        group_map=None,
        use_mask=False,
        categorical_actions=True,
        seed=None,
        done_on_any=None,
        **kwargs,
    ):
        super().__init__(
            env,
            return_state=True,
            group_map=group_map,
            use_mask=use_mask,
            categorical_actions=categorical_actions,
            seed=seed,
            done_on_any=done_on_any,
            **kwargs,
        )
        self._env._reset_policy = reset_policy
        assert scenario_cfg is not None
        self._env._scenario_cfg = scenario_cfg

    def _reset(self, tensordict: TensorDict | None = None, **kwargs):
        """Extract the layout from tensordict and pass to env"""

        if "layout_override" in kwargs and kwargs["layout_override"] is not None:
            theta = kwargs.pop("layout_override").numpy(force=True)
            xcoords, ycoords = theta[:, 0], theta[:, 1]
            options = {"xcoords": xcoords, "ycoords": ycoords}
        elif self._env._reset_policy is not None:
            # Should recompute layout
            td = (
                tensordict
                if tensordict is not None
                else TensorDict({}, device=self.device)
            )
            reset_policy_output = self._env._reset_policy(td)
            td.update(reset_policy_output, keys_to_update=reset_policy_output.keys())
            theta = reset_policy_output.get(
                ("environment_design", "layout_weights")
            ).numpy(force=True)
            xcoords = theta[:, 0]
            ycoords = theta[:, 1]
            options = {"xcoords": xcoords, "ycoords": ycoords}
        else:
            options = None

        tensordict_out = super()._reset(tensordict, options=options, **kwargs)
        tensordict_out["state"] = self._env.state()

        return tensordict_out

    def _step(self, tensordict):
        tensordict_out = super()._step(tensordict)
        tensordict_out["state"] = self._env.state()
        return tensordict_out

    def state(self):
        # Placeholder, overrides super.state() because it doesn't support dict state spaces yet.
        return torch.zeros(1)


def _create_designable_windfarm(
    scenario: ScenarioConfig,
    initial_xcoords,
    initial_ycoords,
    render: bool = False,
):
    if isinstance(initial_xcoords, np.ndarray):
        initial_xcoords = initial_xcoords.tolist()
    if isinstance(initial_ycoords, np.ndarray):
        initial_ycoords = initial_ycoords.tolist()
    case = FlorisCase(
        num_turbines=scenario.n_turbines,
        xcoords=initial_xcoords,
        ycoords=initial_ycoords,
        dt=60,
        buffer_window=1,
        t_init=0,
    )
    validate_case("", case)
    return DesignableMAWindFarmEnv(
        interface=FlorisInterface,
        farm_case=case,
        controls=get_default_control(["yaw"]),
        start_iter=math.ceil(case.t_init / case.dt),
        max_num_steps=scenario.max_steps,
        scenario=scenario,
        render_mode="rgb_array" if render else None,
    )


def create_env(
    mode: Literal["train", "eval", "reference"],
    scenario: ScenarioConfig,
    designer: DesignConsumer,
    device: str | None = None,
    render: bool = False,
):
    theta = make_generate_fn(scenario)()[0]  # Placeholder initial environment
    assert isinstance(theta, np.ndarray)
    env = _create_designable_windfarm(
        scenario=scenario,
        initial_xcoords=theta[:, 0],
        initial_ycoords=theta[:, 1],
        render=render,
    )
    env = aec_to_parallel(env)

    env = WfcrlCoDesignWrapper(
        env=env,
        reset_policy=designer.to_td_module(),
        scenario_cfg=scenario,
        device=device,
    )

    env = TransformedEnv(
        env=env,
        transform=Compose(
            RewardSum(
                in_keys=[env.reward_key], out_keys=[(GROUP_NAME, "episode_reward")]
            ),
            RemoveEmptySpecs(),
        ),
    )
    return env


def render_layout(x, scenario):
    env = _create_designable_windfarm(
        scenario=scenario,
        initial_xcoords=x[:, 0].tolist(),
        initial_ycoords=x[:, 1].tolist(),
        render=True,
    )

    env.reset()
    im = env.render()
    env.close()
    return im


def hashable_representation(env: torch.Tensor):
    # Round to 4dp
    env = torch.round(env, decimals=4)
    np_repr = np.ascontiguousarray(env.detach().cpu().to(torch.uint8).numpy())
    return np_repr.tobytes()
