from typing import Literal
import math
import copy

import numpy as np
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
from wfcrl import environments as envs
from wfcrl.environments.registration import get_default_control, validate_case
from wfcrl.environments.data_cases import FlorisCase
from wfcrl.interface import FlorisInterface
from wfcrl.multiagent_env import MAWindFarmEnv
from wfcrl.mdp import WindFarmMDP

from diffusion_co_design.wfcrl.schema import ScenarioConfig


class DesignableMAWindFarmEnv(MAWindFarmEnv):
    def __init__(
        self,
        interface,
        farm_case,
        controls,
        continuous_control=True,
        reward_shaper=...,
        start_iter=0,
        max_num_steps=500,
        load_coef=0.1,
    ):
        super().__init__(
            interface,
            farm_case,
            controls,
            continuous_control,
            reward_shaper,
            start_iter,
            max_num_steps,
            load_coef,
        )
        self.interface_cls = interface
        self.start_iter = start_iter

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
                new_farm_case.xcoords = coords

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
    device: str | None = None,
):
    env = PettingZooWrapper(
        aec_to_parallel(
            envs.make(
                "Dec_Turb3_Row1_Floris", max_num_steps=scenario.max_steps, load_coef=1.0
            ),
        ),
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
