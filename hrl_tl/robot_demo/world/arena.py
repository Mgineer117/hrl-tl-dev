"""A fixed Zone layout shared by observations and robot demonstration."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal, Self, cast

import pydantic
import yaml
from contgrid.envs.zone import env as zone_env
from contgrid.envs.zone import scenario as zone_scenario

from hrl_tl.robot_demo import config, pose

# The reference room spans 10 native units between the inner wall faces (10 ft = 3.048 m).
DEFAULT_SCALE: float = 10.0 / 3.048


class Zone(config.Settings):
    """A colored disk landmark in Zone coordinates."""

    color: Literal["yellow", "red", "white", "black"]
    center: pose.Point2D
    radius: float = pydantic.Field(gt=0)


class Arena(config.Settings):
    """A resolved native environment and its matching robot configuration.

    Attributes:
        environment: Native ZoneEnv keyword arguments with fixed spawns.
        zones: Actual post-swap landmarks.
        walls: Native wall bounds ordered min_x, max_x, min_y, max_y.
        start: Initial agent position in Zone coordinates.
        robot: World transform, control limits, and ROS topics.
        max_steps: Maximum number of issued low-level movements.
        seed: Layout generation seed for provenance.
        physical_walls: Whether solid walls and inset target limits are active.
    """

    environment: dict[str, Any]
    zones: tuple[Zone, ...]
    walls: tuple[tuple[float, float, float, float], ...]
    start: pose.Point2D
    robot: config.DemoConfig
    max_steps: int = pydantic.Field(default=250, gt=0)
    seed: int = 0
    physical_walls: bool = True

    @pydantic.model_validator(mode="after")
    def validate_geometry(self) -> Self:
        """Reject snapshots whose rendered geometry differs from native state."""
        spawn = self.environment["scenario_config"]["spawn_config"]
        if spawn["spawn_method"]["mode"] != "fixed":
            raise ValueError("Arena must contain resolved fixed spawns")
        if spawn["agent"] != [self.start.x, self.start.y]:
            raise ValueError("Rendered robot start differs from native spawn")
        expected = sorted(
            (color, *item["pos"], spawn["zone_size"][color])
            for color in ("yellow", "red", "white", "black")
            for item in spawn[f"{color}_zone"]
        )
        actual = sorted(
            (z.color, z.center.x, z.center.y, z.radius) for z in self.zones
        )
        if actual != expected:
            raise ValueError("Rendered zones differ from native spawns")
        initial = pose.zone_to_world(self.start, self.robot.frame)
        if not self.robot.bounds.contains(
            initial.x, initial.y, self.robot.motion.wall_stop_margin
        ):
            raise ValueError("Robot start is too close to a wall")
        clearance = spawn["agent_size"]
        if any(
            math.hypot(self.start.x - z.center.x, self.start.y - z.center.y)
            < z.radius + clearance
            for z in self.zones
        ):
            raise ValueError("Robot start overlaps a zone")
        grid = self.environment["world_config"]["grid"]
        cell, rows = grid["cell_size"], grid["layout"]
        walls = sorted(
            (
                (c - 0.5) * cell,
                (c + 0.5) * cell,
                (len(rows) - r - 1.5) * cell,
                (len(rows) - r - 0.5) * cell,
            )
            for r, row in enumerate(rows)
            for c, value in enumerate(row)
            if value == "#"
        )
        if sorted(self.walls) != walls:
            raise ValueError("Rendered walls differ from native grid")
        return self

    @property
    def identity(self) -> str:
        """Content identity sha256 hash shared by controller and tools."""
        values = self.model_dump()
        if self.robot.motion.turn_clearance == 0:
            values["robot"]["motion"].pop("turn_clearance", None)
        if self.physical_walls:
            values.pop("physical_walls", None)
        data = json.dumps(values, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    def save(self, path: Path) -> None:
        """Write a new snapshot without overwriting an earlier run."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(self.model_dump_json(indent=2) + "\n")


def load(path: Path) -> Arena:
    """Load the arena configuration snapshot from a JSON file."""
    return Arena.model_validate_json(path.read_text(encoding="utf-8"))


def prepare(path: Path, seed: int, scale: float = DEFAULT_SCALE) -> Arena:
    """Resolve random spawns once using the existing ContGrid environment.

    Args:
        path: Existing native environment YAML path.
        seed: Seed applied to the native reset.
        scale: Zone units per world metre (defaults to 10/3.048).

    Returns:
        The generated Arena instance.

    Raises:
        ValueError: If configuration or layout is invalid.
    """
    with path.open("r", encoding="utf-8") as stream:
        source = yaml.safe_load(stream)
    if source["id"] != "contgrid/Zone-v0":
        raise ValueError("Robot arena requires contgrid/Zone-v0")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Scale must be finite and positive")
    native = zone_env.ZoneEnv(**source["env_kwargs"])
    try:
        native.reset(seed=seed)
        scenario = cast(zone_scenario.ZoneScenario, native.scenario)
        world = native.env.world
        grid = world.grid
        top_bottom = list(grid.layout[0]) + list(grid.layout[-1])
        if (
            any("#" in row[1:-1] for row in grid.layout[1:-1])
            or any(value != "#" for value in top_bottom)
            or any(row[0] != "#" or row[-1] != "#" for row in grid.layout)
        ):
            raise ValueError("Initial robot demo supports an open Zone room")
        cell = grid.cell_size
        min_x = min_y = cell / 2
        max_x = (grid.width_cells - 1.5) * cell
        max_y = (grid.height_cells - 1.5) * cell
        frame = config.FrameConfig(
            origin_x_m=-(min_x + max_x) / (2 * scale),
            origin_y_m=-(min_y + max_y) / (2 * scale),
            sim_units_per_meter=scale,
        )
        lower = pose.zone_to_world(pose.Point2D(x=min_x, y=min_y), frame)
        upper = pose.zone_to_world(pose.Point2D(x=max_x, y=max_y), frame)
        robot = config.DemoConfig(
            frame=frame,
            motion=config.MotionConfig(
                position_tolerance=(
                    config.MotionConfig().position_tolerance / max(1.0, scale)
                ),
            ),
            bounds=config.Bounds(
                min_x=lower.x,
                min_y=lower.y,
                max_x=upper.x,
                max_y=upper.y,
            ),
        )
        for _ in range(1000):
            agent = world.agents[0]
            start = pose.Point2D(
                x=float(agent.state.pos[0]), y=float(agent.state.pos[1])
            )
            initial = pose.zone_to_world(start, frame)
            clearance = max(agent.size, robot.bounds.margin * scale)
            landmarks = (
                scenario.yellow + scenario.red + scenario.white + scenario.black
            )
            if robot.bounds.contains(
                initial.x, initial.y, robot.motion.wall_stop_margin
            ) and all(
                math.hypot(start.x - lm.state.pos[0], start.y - lm.state.pos[1])
                >= lm.size + clearance
                for lm in landmarks
            ):
                break
            if scenario.config.spawn_config.agent is not None:
                raise ValueError("Fixed start lacks robot clearance")
            native.reset()
        else:
            raise ValueError("Could not sample a start with robot clearance")
        scenario_config = scenario.config.model_dump(mode="json")
        spawn = scenario_config["spawn_config"]
        spawn["agent"] = [start.x, start.y]
        spawn["spawn_method"] = {"mode": "fixed"}
        zones: list[Zone] = []
        groups = (
            ("yellow", scenario.yellow),
            ("red", scenario.red),
            ("white", scenario.white),
            ("black", scenario.black),
        )
        for color, landmark_list in groups:
            spawn[f"{color}_zone"] = []
            for landmark in landmark_list:
                x, y = map(float, landmark.state.pos)
                spawn[f"{color}_zone"].append({"pos": [x, y]})
                zones.append(
                    Zone(
                        color=color,
                        center=pose.Point2D(x=x, y=y),
                        radius=float(landmark.size),
                    )
                )
        environment = dict(source["env_kwargs"])
        environment["scenario_config"] = scenario_config
        return Arena(
            environment=environment,
            zones=tuple(zones),
            walls=tuple(map(tuple, scenario.wall_bounds.tolist())),
            start=start,
            robot=robot,
            seed=seed,
            max_steps=source["max_episode_steps"],
        )
    finally:
        native.close()
