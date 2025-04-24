from typing import Literal
import math
import copy

import numpy as np
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs import (
    ParallelEnv,
    PettingZooWrapper,
    TransformedEnv,
    CatTensors,
    RewardSum,
    Compose,
    RemoveEmptySpecs,
)
from pettingzoo.utils.conversions import aec_to_parallel
import gymnasium.spaces as spaces

from wfcrl.environments.registration import get_default_control, validate_case
from wfcrl.environments.data_cases import FlorisCase
from wfcrl.interface import FlorisInterface
from wfcrl.multiagent_env import MAWindFarmEnv
from wfcrl.rewards import DoNothingReward
from wfcrl.mdp import WindFarmMDP

from diffusion_co_design.wfcrl.design import Designer
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
    ):
        if reward_shaper is None:
            reward_shaper = DoNothingReward()
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
            dtype=np.float64,
        )

    def reset(self, seed=None, options=None):
        # Possibly override xcoords and ycoords
        if options is not None:
            old_farm_case = self.mdp.farm_case
            new_farm_case = copy.copy(old_farm_case)
            if "xcoords" in options:
                coords = options.get("xcoords")
                assert len(coords) == old_farm_case.num_turbines
                new_farm_case.xcoords = coords
            if "ycoords" in options:
                coords = options.get("ycoords")
                assert len(coords) == old_farm_case.num_turbines
                new_farm_case.ycoords = coords

            # Override MDP
            self.mdp = WindFarmMDP(
                interface=self.interface_cls,
                farm_case=self.farm_case,
                controls=self.controls,
                continuous_control=self.continuous_control,
                start_iter=self.start_iter,
                horizon=self.start_iter + self.max_num_steps,
            )
        return super().reset(seed, options)

    def state(self):
        state = super().state()
        # Add layout to state
        state["layout"] = np.stack(
            [self.mdp.farm_case.xcoords, self.mdp.farm_case.ycoords], axis=-1
        )
        return state

    def _build_agent_spaces(self):
        super()._build_agent_spaces()
        for i, agent in enumerate(self.possible_agents):
            self._obs_spaces[agent]["layout"] = spaces.Box(
                low=0, high=np.inf, shape=(2,), dtype=np.float64
            )


class WfcrlCoDesignWrapper(PettingZooWrapper):
    def __init__(
        self,
        env=None,
        reset_policy: TensorDictModule | None = None,
        environment_objective=None,
        scenario_cfg=None,
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
        self._env._reset_policy = reset_policy
        self._env._environment_objective = environment_objective
        assert scenario_cfg is not None
        self._env._scenario_cfg = scenario_cfg

    def _reset(self, tensordict: TensorDict | None = None, **kwargs):
        """Extract the layout from tensordict and pass to env"""

        if (
            self._env._environment_objective is not None
            and self._env._reset_policy is not None
        ):
            # Should recompute layout
            if tensordict is not None:
                tensordict[("environment_design", "objective")] = (
                    self._env._environment_objective
                )
                reset_policy_output = self._env._reset_policy(tensordict)
                tensordict.update(
                    reset_policy_output, keys_to_update=reset_policy_output.keys()
                )

            else:
                td = TensorDict(
                    {
                        (
                            "environment_design",
                            "objective",
                        ): self._env._environment_objective
                    }
                )
                reset_policy_output = self._env._reset_policy(td)

            theta = reset_policy_output.get(
                ("environment_design", "layout_weights")
            ).numpy(force=True)
            xcoords = theta[:, 0]
            ycoords = theta[:, 1]
            options = {"xcoords": xcoords, "ycoords": ycoords}
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


def _create_designable_windfarm(n_turbines: int, initial_xcoords, initial_ycoords):
    if isinstance(initial_xcoords, np.ndarray):
        initial_xcoords = initial_xcoords.tolist()
    if isinstance(initial_ycoords, np.ndarray):
        initial_ycoords = initial_ycoords.tolist()
    case = FlorisCase(
        num_turbines=n_turbines,
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
    )


def create_env(
    mode: Literal["train", "eval", "reference"],
    scenario: ScenarioConfig,
    designer: Designer,
    device: str | None = None,
):
    theta = designer.generate_environment_weights().numpy(force=True)
    env = _create_designable_windfarm(
        n_turbines=scenario.n_turbines,
        initial_xcoords=theta[:, 0],
        initial_ycoords=theta[:, 1],
    )
    env = aec_to_parallel(env)

    env = WfcrlCoDesignWrapper(
        env=env,
        reset_policy=designer.to_td_module(),
        environment_objective=None,
        scenario_cfg=scenario,
        return_state=True,
        device=device,
    )

    observation_keys = [
        ("turbine", "observation", x) for x in ["wind_direction", "wind_speed", "yaw"]
    ]
    env = TransformedEnv(
        env=env,
        transform=Compose(
            CatTensors(
                in_keys=observation_keys,
                out_key=("turbine", "observation_vec"),
                del_keys=False,
            ),
            RewardSum(
                in_keys=[env.reward_key], out_keys=[("turbine", "episode_reward")]
            ),
            RemoveEmptySpecs(),
        ),
    )
    env.reset()
    return env


def create_batched_env(
    num_environments: int,
    scenario: ScenarioConfig,
    mode: Literal["train", "eval", "reference"],
    device: str | None = None,
):
    def create_env_fn():
        return create_env(mode, scenario=scenario, device="cpu")

    return ParallelEnv(
        num_workers=num_environments,
        create_env_fn=create_env_fn,
        create_env_kwargs={},
        device=device,
    )
