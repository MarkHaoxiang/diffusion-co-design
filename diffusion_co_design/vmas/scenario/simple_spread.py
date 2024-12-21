#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import numpy as np
import torch
from torch import Tensor
from torchrl.envs import VmasEnv
from torchrl.data.tensor_specs import BoundedContinuous
import vmas
from vmas import render_interactively

from torchrl.data.tensor_specs import (
    Categorical,
    Composite,
    StackedComposite,
)
from torchrl.envs.utils import (
    check_marl_grouping,
    MarlGroupMapType,
)

from diffusion_co_design.vmas.schema import SimpleSpreadScenarioConfig

from vmas.scenarios.mpe import simple_spread


class Scenario(simple_spread.Scenario):
    def random_map_positions(self, env_index=None, world=None):
        if world is None:
            world = self.world
        return torch.zeros(
            (
                (1, world.dim_p)
                if env_index is not None
                else (world.batch_dim, world.dim_p)
            ),
            device=world.device,
            dtype=torch.float32,
        ).uniform_(-1.0, 1.0)

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        agent_goals = kwargs.pop("agent_goals", None)

        world = super().make_world(batch_dim, device, **kwargs)

        if agent_goals is None:
            agent_goals = torch.stack(
                [self.random_map_positions(world=world) for _ in world.agents], dim=1
            )

        assert isinstance(agent_goals, torch.Tensor)
        assert agent_goals.shape == (
            batch_dim,
            len(world.agents),
            world.dim_p,
        )
        self.agent_goals = agent_goals

        return world

    def reset_world_at(self, env_index=None):
        for agent in self.world.agents:
            agent.set_pos(self.random_map_positions(env_index), batch_index=env_index)

        for i, landmark in enumerate(self.world.landmarks):
            landmark.set_pos(
                self.agent_goals[
                    (env_index if env_index is not None else slice(None)), i, :
                ],
                batch_index=env_index,
            )

    def reward(self, agent):
        return super().reward(agent) / 10.0


class DesignableVmasEnv(VmasEnv):
    def __init__(
        self,
        scenario: Scenario,
        scenario_cfg: SimpleSpreadScenarioConfig,
        reset_policy,
        num_envs=1,
        device="cpu",
        continuous_actions=True,
        max_steps=None,
        seed=None,
        dict_spaces=False,
        multidiscrete_actions=False,
        clamp_actions=False,
        grad_enabled=False,
        terminated_truncated=False,
        **kwargs,
    ):
        scenario._scenario_cfg = scenario_cfg
        super().__init__(
            scenario=scenario,
            num_envs=num_envs,
            device=device,
            continuous_actions=continuous_actions,
            max_steps=max_steps,
            seed=seed,
            dict_spaces=dict_spaces,
            multidiscrete_actions=multidiscrete_actions,
            clamp_actions=clamp_actions,
            grad_enabled=grad_enabled,
            terminated_truncated=terminated_truncated,
            **kwargs,
        )

        self._env._reset_policy = reset_policy

    def _make_specs(
        self,
        env: vmas.simulator.environment.environment.Environment,  # noqa
    ) -> None:
        # Create and check group map
        self.agent_names = [agent.name for agent in self.agents]
        self.agent_names_to_indices_map = {
            agent.name: i for i, agent in enumerate(self.agents)
        }
        if self.group_map is None:
            self.group_map = self._get_default_group_map(self.agent_names)
        elif isinstance(self.group_map, MarlGroupMapType):
            self.group_map = self.group_map.get_group_map(self.agent_names)
        check_marl_grouping(self.group_map, self.agent_names)

        full_action_spec_unbatched = Composite(device=self.device)
        full_observation_spec_unbatched = Composite(device=self.device)
        full_reward_spec_unbatched = Composite(device=self.device)

        self.het_specs = False
        self.het_specs_map = {}
        for group in self.group_map.keys():
            (
                group_observation_spec,
                group_action_spec,
                group_reward_spec,
                group_info_spec,
            ) = self._make_unbatched_group_specs(group)
            full_action_spec_unbatched[group] = group_action_spec
            full_observation_spec_unbatched[group] = group_observation_spec
            full_reward_spec_unbatched[group] = group_reward_spec
            if group_info_spec is not None:
                full_observation_spec_unbatched[(group, "info")] = group_info_spec
            group_het_specs = isinstance(
                group_observation_spec, StackedComposite
            ) or isinstance(group_action_spec, StackedComposite)
            self.het_specs_map[group] = group_het_specs
            self.het_specs = self.het_specs or group_het_specs

        full_done_spec_unbatched = Composite(
            {
                "done": Categorical(
                    n=2,
                    shape=torch.Size((1,)),
                    dtype=torch.bool,
                    device=self.device,
                ),
            },
        )

        # ===
        # Add state

        sc: SimpleSpreadScenarioConfig = env.scenario._scenario_cfg
        full_observation_spec_unbatched["state"] = BoundedContinuous(
            low=sc.layout_space_low,
            high=sc.layout_space_high,
            device=self.device,
            dtype=torch.float32,
        )

        # ===

        self.full_action_spec_unbatched = full_action_spec_unbatched
        self.full_observation_spec_unbatched = full_observation_spec_unbatched
        self.full_reward_spec_unbatched = full_reward_spec_unbatched
        self.full_done_spec_unbatched = full_done_spec_unbatched

    def _reset(self, tensordict=None, **kwargs):
        scenario: Scenario = self._env.scenario  # vmas.simulator.environment

        if self._env._reset_policy is not None or "layout_override" in kwargs:
            if "layout_override" in kwargs and kwargs["layout_override"] is not None:
                new_layouts = list(kwargs.pop("layout_override"))
            else:
                new_layouts = [
                    self._env._reset_policy() for _ in range(self._env.num_envs)
                ]

            theta = torch.stack(new_layouts, dim=0)

        else:
            theta = None

        if theta is not None:
            assert isinstance(theta, Tensor)
            theta = theta.view(scenario.agent_goals.shape)
            theta = theta.to(scenario.agent_goals.device)
            scenario.agent_goals = theta

        tensordict_out = super()._reset(tensordict, **kwargs)
        tensordict_out["state"] = self._get_scenario_state()

        return tensordict_out

    def _step(self, tensordict):
        tensordict_out = super()._step(tensordict)
        tensordict_out["state"] = self._get_scenario_state()
        return tensordict_out

    def render(self):
        # Transformation to meet the convention of pettingzoo rendering with Parallel collection
        return np.expand_dims(self._env.render(mode="rgb_array"), axis=0)

    def _get_scenario_state(self):
        layout = self._env.scenario.agent_goals.clone()
        return layout


if __name__ == "__main__":
    render_interactively(
        scenario=Scenario(),
        control_two_agents=True,
    )
