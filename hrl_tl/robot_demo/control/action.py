"""Conversion of the final Zone movement action to a fixed short waypoint."""

from __future__ import annotations

import math

import pydantic

from hrl_tl.robot_demo import config, pose
from hrl_tl.robot_demo.world import arena


class MovementAction(config.Settings):
    """Final low-level action, not the five-element high-level PPO output."""

    direction_index: int = pydantic.Field(ge=0, lt=8, strict=True)
    magnitude_index: int = pydantic.Field(ge=0, lt=5, strict=True)


class NativeMovementAction(MovementAction):
    """A primitive action with its native and physically reachable endpoints."""

    target: pose.Point2D
    native_target: pose.Point2D


def target_from_action(
    movement: MovementAction, start: pose.Pose2D, settings: config.DemoConfig
) -> pose.Point2D:
    """Freeze a world waypoint relative to the measured action-start pose.

    Learned commands carry an endpoint computed from the discrete action.
    Sequence-test commands use the configured distance mapping instead.
    Neither index is a physical velocity command.
    """
    if isinstance(movement, NativeMovementAction):
        return movement.target
    heading = (
        movement.direction_index * math.tau / 8 - settings.frame.rotation_rad
    )
    distance = (
        settings.step_lengths_sim[movement.magnitude_index]
        / settings.frame.sim_units_per_meter
    )
    return pose.Point2D(
        x=start.x + distance * math.cos(heading),
        y=start.y + distance * math.sin(heading),
    )


def interpret_manual_action(
    layout: arena.Arena, angle_deg: float, distance_m: float
) -> tuple[MovementAction, float, float]:
    """Maps one supported Zone-frame angle and stride to discrete action bins.

    Args:
        layout: Arena containing robot frame scale and step length bins.
        angle_deg: Desired Zone-frame angle in degrees (multiple of 45).
        distance_m: Desired stride distance in meters.

    Returns:
        Tuple of (MovementAction, interpreted_angle_deg, actual_distance_m).

    Raises:
        ValueError: If angle or distance are non-finite, not a multiple of 45
            degrees, or not matching a configured step bin within 1 mm.
    """
    if not math.isfinite(angle_deg) or not math.isfinite(distance_m):
        raise ValueError("angle and distance must be finite numbers.")
    direction = angle_deg % 360.0
    direction_index = round(direction / 45.0) % 8
    interpreted_angle = direction_index * 45.0
    angle_error = abs((direction - interpreted_angle + 180.0) % 360.0 - 180.0)
    if angle_error > 1e-6:
        raise ValueError("angle must be a multiple of 45 degrees.")
    available_steps = [
        value / layout.robot.frame.sim_units_per_meter
        for value in layout.robot.step_lengths_sim
    ]
    magnitude_index = min(
        range(len(available_steps)),
        key=lambda index: abs(available_steps[index] - distance_m),
    )
    if abs(available_steps[magnitude_index] - distance_m) > 0.001:
        choices = ", ".join(f"{step:.5f}" for step in available_steps)
        raise ValueError(
            f"distance must match a configured step within 1 mm; available: {choices}"
        )
    return (
        MovementAction(
            direction_index=direction_index,
            magnitude_index=magnitude_index,
        ),
        interpreted_angle,
        available_steps[magnitude_index],
    )
