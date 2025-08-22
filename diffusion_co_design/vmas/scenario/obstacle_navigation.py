#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import warnings
import typing

import numpy as np
import torch
from torch import Tensor
from torchrl.envs import VmasEnv
from torchrl.data.tensor_specs import BoundedContinuous
import vmas
from vmas import render_interactively
from vmas.simulator.core import Agent, Entity, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils

from torchrl.data.tensor_specs import (
    Categorical,
    Composite,
    StackedComposite,
)
from torchrl.envs.utils import (
    check_marl_grouping,
    MarlGroupMapType,
)

from diffusion_co_design.vmas.schema import (
    ScenarioConfigType,
    LocalPlacementScenarioConfig,
    GlobalPlacementScenarioConfig,
)

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


def setup_entity_locations(
    entity_locations: Tensor | None,
    entity_radius: Tensor,
    n_entities: int,
    occupied_positions: Tensor | None,
    occupied_radius: Tensor,
    x_bounds: tuple[float, float] = (-1, 1),
    y_bounds: tuple[float, float] = (-1, 1),
    device: torch.device = torch.device("cpu"),
    constraint_violation: typing.Literal["warn", "error"] = "error",
):
    occupied_positions = (
        occupied_positions
        if occupied_positions is not None
        else torch.zeros((0, 2), device=device)
    )

    assert entity_radius.shape == (n_entities,)
    assert occupied_radius.shape == (occupied_positions.shape[0],)

    occupied_radius = torch.cat([occupied_radius, entity_radius], dim=0)

    if entity_locations is not None:
        assert isinstance(entity_locations, torch.Tensor)
        assert entity_locations.shape == (n_entities, 2)
        assert entity_locations.device == device
        occupied_positions = torch.cat([occupied_positions, entity_locations], dim=0)
    else:
        entity_locations = torch.zeros((n_entities, 2), device=device)

        for i in range(n_entities):
            # Search for a valid position
            tries = 0
            min_dist = occupied_radius + entity_radius[i]
            while True:
                pos = torch.rand(2, device=device)
                pos[0] = pos[0] * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
                pos[1] = pos[1] * (y_bounds[1] - y_bounds[0]) + y_bounds[0]
                dist = torch.cdist(occupied_positions, pos.unsqueeze(0))
                if torch.all(dist >= min_dist[: occupied_positions.shape[0] + 1]):
                    break
                tries += 1
                if tries > 50_000:
                    warnings.warn(
                        "Could not find a valid position for entity after 50,000 tries."
                    )
            entity_locations[i] = pos
            occupied_positions = torch.cat(
                [occupied_positions, pos.unsqueeze(0)], dim=0
            )

    dist = (
        torch.cdist(occupied_positions, occupied_positions)
        + torch.eye(occupied_positions.shape[0], device=device)
        * 1e6  # Prevent self-distance
    )
    min_dist = occupied_radius.unsqueeze(1) + occupied_radius.unsqueeze(0)
    if torch.any(dist < min_dist):
        error_message = f"Entity locations are too close to each other. Please adjust the entity radius or the occupied positons. Violation amt: {torch.max(min_dist - dist)}"
        if constraint_violation == "warn":
            warnings.warn(error_message)
        else:
            raise ValueError(error_message)

    return entity_locations, occupied_positions


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.plot_grid = False
        self.n_agents = kwargs.pop("n_agents", 4)
        self.collisions = kwargs.pop("collisions", True)

        # Make world
        self.world_spawning_x = kwargs.pop("world_spawning_x", 1)
        self.world_spawning_y = kwargs.pop("world_spawning_y", 1)
        self.enforce_bounds = kwargs.pop("enforce_bounds", False)
        self.x_semidim = self.world_spawning_x if self.enforce_bounds else None
        self.y_semidim = self.world_spawning_y if self.enforce_bounds else None
        world = World(
            batch_dim,
            device,
            substeps=2,
            x_semidim=self.x_semidim,
            y_semidim=self.y_semidim,
        )

        # Make agents
        self.lidar_range = kwargs.pop("lidar_range", 0.35)
        self.agent_radius = kwargs.pop("agent_radius", 0.05)
        self.comms_range = kwargs.pop("comms_range", 0)
        self.n_lidar_rays = kwargs.pop("n_lidar_rays", 12)
        self.min_collision_distance = 0.005

        agent_goals = kwargs.pop("agent_goals", None)
        self.agent_goals, occupied_locations = setup_entity_locations(
            entity_locations=agent_goals,
            entity_radius=torch.tensor(
                [self.agent_radius] * self.n_agents, device=device
            )
            + self.min_collision_distance / 2,
            n_entities=self.n_agents,
            occupied_positions=None,
            occupied_radius=torch.zeros((0,), device=device),
            x_bounds=(-self.world_spawning_x, self.world_spawning_x),
            y_bounds=(-self.world_spawning_y, self.world_spawning_y),
            device=device,
        )

        agent_spawns = kwargs.pop("agent_spawns", None)
        self.agent_spawns, occupied_locations = setup_entity_locations(
            entity_locations=agent_spawns,
            entity_radius=torch.tensor(
                [self.agent_radius] * self.n_agents, device=device
            )
            + self.min_collision_distance / 2,
            n_entities=self.n_agents,
            occupied_positions=occupied_locations,
            occupied_radius=torch.fill(
                torch.zeros((occupied_locations.shape[0],), device=device),
                self.agent_radius,
            )
            + self.min_collision_distance / 2,
            x_bounds=(-self.world_spawning_x, self.world_spawning_x),
            y_bounds=(-self.world_spawning_y, self.world_spawning_y),
            device=device,
        )

        # Make obstacles
        self.obstacle_sizes = kwargs.pop(
            "obstacle_sizes", torch.tensor([0.3, 0.2, 0.1], device=device)
        )
        self.n_obstacles = len(self.obstacle_sizes)
        obstacle_positions = kwargs.pop("obstacle_positions", None)
        self.obstacle_locations = torch.zeros(
            (batch_dim, self.n_obstacles, 2), device=device
        )

        for i in range(batch_dim):
            self.obstacle_locations[i] = setup_entity_locations(
                entity_locations=obstacle_positions[i]
                if obstacle_positions is not None
                else None,
                entity_radius=self.obstacle_sizes + self.min_collision_distance / 2,
                n_entities=self.n_obstacles,
                occupied_positions=occupied_locations,
                occupied_radius=torch.fill(
                    torch.zeros((occupied_locations.shape[0],), device=device),
                    self.agent_radius,
                )
                + self.min_collision_distance / 2,
                x_bounds=(-self.world_spawning_x, self.world_spawning_x),
                y_bounds=(-self.world_spawning_y, self.world_spawning_y),
                device=device,
                constraint_violation="warn",
            )[0]

        # Reward settings
        self.shared_rew = kwargs.pop("shared_rew", True)
        self.pos_shaping_factor = kwargs.pop("pos_shaping_factor", 1)
        self.final_reward = kwargs.pop("final_reward", 0.01)

        self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", -1)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        known_colors = [
            (0.22, 0.49, 0.72),
            (1.00, 0.50, 0),
            (0.30, 0.69, 0.29),
            (0.97, 0.51, 0.75),
            (0.60, 0.31, 0.64),
            (0.89, 0.10, 0.11),
            (0.87, 0.87, 0),
        ]
        colors = torch.randn(
            (max(self.n_agents - len(known_colors), 0), 3), device=device
        )

        def entity_filter_agents(e: Entity) -> bool:
            return isinstance(e, Agent)

        # Add agents
        for i in range(self.n_agents):
            color = (
                known_colors[i]
                if i < len(known_colors)
                else colors[i - len(known_colors)]
            )

            # Constraint: all agents have same action range and multiplier
            agent = Agent(
                name=f"agent_{i}",
                collide=self.collisions,
                color=color,
                shape=Sphere(radius=self.agent_radius),
                render_action=True,
                sensors=(
                    [
                        Lidar(
                            world,
                            n_rays=self.n_lidar_rays,
                            max_range=self.lidar_range,
                            entity_filter=entity_filter_agents,
                        ),
                    ]
                    if self.collisions
                    else None
                ),
            )
            agent.pos_rew = torch.zeros(batch_dim, device=device)
            agent.agent_collision_rew = agent.pos_rew.clone()
            world.add_agent(agent)

            # Add goals
            goal = Landmark(
                name=f"goal {i}",
                collide=False,
                color=color,
            )
            world.add_landmark(goal)
            agent.goal = goal

        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = self.pos_rew.clone()

        # Add obstacles
        self.obstacles: list[Landmark] = []
        for i in range(self.n_obstacles):
            obstacle = Landmark(
                name=f"obstacle {i}",
                collide=True,
                color=Color.GRAY,
                shape=Sphere(radius=self.obstacle_sizes[i]),
            )
            world.add_landmark(obstacle)
            self.obstacles.append(obstacle)

        return world

    def reset_world_at(self, env_index: int | None = None):
        # Spawn agents
        for i, agent in enumerate(self.world.agents):
            pos = self.agent_spawns[i].unsqueeze(0)
            agent.set_pos(pos, batch_index=env_index)

        # Spawn goals
        for i, agent in enumerate(self.world.agents):
            pos = self.agent_goals[i].unsqueeze(0)
            agent.goal.set_pos(pos, batch_index=env_index)

        # Spawn obstacles
        for i, obstacle in enumerate(self.obstacles):
            if env_index is None:
                pos = self.obstacle_locations[:, i, :]
            else:
                pos = self.obstacle_locations[env_index, i, :]
            obstacle.set_pos(pos, batch_index=env_index)

        for i, agent in enumerate(self.world.agents):
            if env_index is None:
                agent.pos_shaping = (
                    torch.linalg.vector_norm(
                        agent.state.pos - agent.goal.state.pos, dim=1
                    )
                    * self.pos_shaping_factor
                )
            else:
                agent.pos_shaping[env_index] = (
                    torch.linalg.vector_norm(
                        agent.state.pos[env_index] - agent.goal.state.pos[env_index]
                    )
                    * self.pos_shaping_factor
                )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.pos_rew[:] = 0
            self.final_rew[:] = 0

            for a in self.world.agents:
                self.pos_rew += self.agent_reward(a)
                a.agent_collision_rew[:] = 0

            self.all_goal_reached = torch.all(
                torch.stack([a.on_goal for a in self.world.agents], dim=-1),
                dim=-1,
            )

            self.final_rew[self.all_goal_reached] = self.final_reward

            for i, a in enumerate(self.world.agents):
                for j, b in enumerate(self.world.agents):
                    if i <= j:
                        continue
                    if self.world.collides(a, b):
                        distance = self.world.get_distance(a, b)
                        a.agent_collision_rew[
                            distance <= self.min_collision_distance
                        ] += self.agent_collision_penalty
                        b.agent_collision_rew[
                            distance <= self.min_collision_distance
                        ] += self.agent_collision_penalty

        pos_reward = self.pos_rew if self.shared_rew else agent.pos_rew
        return pos_reward + self.final_rew + agent.agent_collision_rew

    def agent_reward(self, agent: Agent):
        agent.distance_to_goal = torch.linalg.vector_norm(
            agent.state.pos - agent.goal.state.pos,
            dim=-1,
        )
        agent.on_goal = agent.distance_to_goal < agent.goal.shape.radius

        pos_shaping = agent.distance_to_goal * self.pos_shaping_factor
        agent.pos_rew = agent.pos_shaping - pos_shaping
        agent.pos_shaping = pos_shaping

        return agent.pos_rew - torch.where(agent.on_goal, 0.0, 0.01)

    def observation(self, agent: Agent):
        goal_poses = []
        goal_poses.append(agent.state.pos - agent.goal.state.pos)
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
            ]
            + goal_poses
            + (
                [agent.sensors[0]._max_range - agent.sensors[0].measure()]
                if self.collisions
                else []
            ),
            dim=-1,
        )

    def done(self):
        return torch.zeros(
            (self.world.batch_dim, 1), dtype=torch.bool, device=self.world.device
        )

    def info(self, agent: Agent) -> dict[str, Tensor]:
        return {
            "pos_rew": self.pos_rew if self.shared_rew else agent.pos_rew,
            "final_rew": self.final_rew,
            "agent_collisions": agent.agent_collision_rew,
        }

    def extra_render(self, env_index: int = 0) -> "list[Geom]":
        from vmas.simulator import rendering

        geoms: list[Geom] = []

        # Communication lines
        for i, agent1 in enumerate(self.world.agents):
            for j, agent2 in enumerate(self.world.agents):
                if j <= i:
                    continue
                agent_dist = torch.linalg.vector_norm(
                    agent1.state.pos - agent2.state.pos, dim=-1
                )
                if agent_dist[env_index] <= self.comms_range:
                    color = Color.BLACK.value
                    line = rendering.Line(
                        (agent1.state.pos[env_index]),
                        (agent2.state.pos[env_index]),
                        width=1,
                    )
                    xform = rendering.Transform()
                    line.add_attr(xform)
                    line.set_color(*color)
                    geoms.append(line)

        return geoms


class DesignableVmasEnv(VmasEnv):
    def __init__(
        self,
        scenario: Scenario,
        scenario_cfg: ScenarioConfigType,
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
        if isinstance(scenario_cfg, LocalPlacementScenarioConfig):
            is_theta_mask = torch.zeros(
                (scenario_cfg.n_obstacles, 2), dtype=torch.bool, device=device
            )
            for i, ((x_low, x_high), (y_low, y_high)) in enumerate(
                scenario_cfg.obstacle_bounds
            ):
                if x_low != x_high:
                    is_theta_mask[i, 0] = 1
                if y_low != y_high:
                    is_theta_mask[i, 1] = 1
            self._env._is_theta_mask = is_theta_mask
            self._env.bounds_tensor = torch.tensor(
                scenario_cfg.obstacle_bounds, device=device
            )  # [M, 2, 2]

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

        sc: ScenarioConfigType = env.scenario._scenario_cfg
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
        sc: ScenarioConfigType = scenario._scenario_cfg

        if "layout_override" in kwargs and kwargs["layout_override"] is not None:
            theta = kwargs.pop("layout_override")
        elif self._env._reset_policy is not None:
            new_layouts = [self._env._reset_policy() for _ in range(self._env.num_envs)]
            if isinstance(sc, LocalPlacementScenarioConfig):
                B, M, _ = scenario.obstacle_locations.shape
                theta = torch.zeros_like(scenario.obstacle_locations)  # [B, M, 2]
                bounds_tensor = self._env.bounds_tensor

                idx = []
                for i, ((x_low, x_high), (y_low, y_high)) in enumerate(
                    sc.obstacle_bounds
                ):
                    if x_low != x_high:
                        idx.append((i, 0))
                    if y_low != y_high:
                        idx.append((i, 1))
                idx = torch.tensor(idx, device=theta.device, dtype=torch.int64)

                for i, layout in enumerate(new_layouts):
                    z = torch.zeros((M, 2), device=theta.device)  # [M, 2]
                    z[idx[:, 0], idx[:, 1]] = torch.tensor(
                        (layout + 1) / 2, device=theta.device
                    )
                    theta[i] = (
                        bounds_tensor[:, :, 0]
                        + (bounds_tensor[:, :, 1] - bounds_tensor[:, :, 0]) * z
                    )
            else:
                theta = torch.stack(new_layouts, dim=0)
        else:
            theta = None

        if theta is not None:
            assert isinstance(theta, Tensor)
            assert theta.shape == scenario.obstacle_locations.shape, (
                f"Expected theta shape {scenario.obstacle_locations.shape}, got {theta.shape}"
            )
            theta = theta.to(scenario.obstacle_locations.device)
            scenario.obstacle_locations = theta

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
        layout = self._env.scenario.obstacle_locations.clone()
        if isinstance(self._env.scenario._scenario_cfg, GlobalPlacementScenarioConfig):
            return layout
        elif isinstance(self._env.scenario._scenario_cfg, LocalPlacementScenarioConfig):
            layouts_of_interest = layout[:, self._env._is_theta_mask]  # [B, M, 2]
            # Normalise
            bounds_tensor = self._env.bounds_tensor[self._env._is_theta_mask].unsqueeze(
                0
            )
            layouts_of_interest = (layouts_of_interest - bounds_tensor[:, :, 0]) / (
                bounds_tensor[:, :, 1] - bounds_tensor[:, :, 0]
            ) * 2 - 1
            return layouts_of_interest


if __name__ == "__main__":
    render_interactively(
        scenario=Scenario(),
        control_two_agents=True,
    )
