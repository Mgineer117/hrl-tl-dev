"""Kinematic unicycle simulation and offline policy rollout for robot demos."""

from __future__ import annotations

import io
import math
from typing import Any

from stable_baselines3.common import base_class

from hrl_tl.robot_demo import pose
from hrl_tl.robot_demo.control import motion, smoothing
from hrl_tl.robot_demo.inference.hierarchy import RobotHierarchy
from hrl_tl.robot_demo.world import arena


def advance_unicycle(
    current: pose.Pose2D, command: motion.Velocity, dt: float, stamp: float
) -> pose.Pose2D:
    """Integrate the commanded unicycle velocity for one fixed time step.

    Args:
        current: Current 2D pose of the robot.
        command: Commanded linear and angular velocity.
        dt: Integration time step in seconds.
        stamp: Current simulation timestamp in seconds.

    Returns:
        The updated 2D pose.
    """
    yaw = current.yaw
    turn = command.angular * dt
    if abs(command.angular) < 1e-10:
        x = current.x + command.linear * math.cos(yaw) * dt
        y = current.y + command.linear * math.sin(yaw) * dt
    else:
        radius = command.linear / command.angular
        x = current.x + radius * (math.sin(yaw + turn) - math.sin(yaw))
        y = current.y - radius * (math.cos(yaw + turn) - math.cos(yaw))
    return pose.Pose2D(
        x=x,
        y=y,
        yaw=yaw + turn,
        stamp=stamp,
        received_at=stamp,
    )


def append_position(
    trajectory: list[pose.Point2D], point: pose.Point2D
) -> None:
    """Append a position waypoint to the trajectory if it moved significantly.

    Args:
        trajectory: Accumulated list of 2D waypoints.
        point: Candidate waypoint to append.
    """
    if (
        not trajectory
        or math.hypot(trajectory[-1].x - point.x, trajectory[-1].y - point.y)
        >= 0.0001
    ):
        trajectory.append(point)


def simulate(
    layout: arena.Arena,
    upper: base_class.BaseAlgorithm,
    wrapper_kwargs: dict[str, Any],
    start: pose.Pose2D,
    max_actions: int,
    *,
    continuous: bool = False,
    smoothing_filter: smoothing.ActionSmoothingFilter | None = None,
) -> dict[str, Any]:
    """Run model waypoints through the repository controller and ideal motion.

    Args:
        layout: Arena layout specification.
        upper: Loaded high-level meta controller algorithm.
        wrapper_kwargs: Wrapper configuration kwargs used to construct the hierarchy.
        start: Initial starting pose in arena world coordinates.
        max_actions: Maximum number of high-level or primitive actions.
        continuous: Whether to allow continuous evaluation past timeouts.
        smoothing_filter: Optional real-time action smoothing filter.

    Returns:
        A dictionary containing the simulation result, trajectory, and actions.

    Raises:
        RuntimeError: If initial pose is rejected or controller faults unexpectedly.
    """
    motion_controller = motion.MotionExecutor(
        layout.robot.motion, layout.robot.bounds
    )
    if not motion_controller.update_pose(start):
        raise RuntimeError("Could not initialize the simulated pose")

    hierarchy = RobotHierarchy(
        layout,
        upper,
        wrapper_kwargs,
        io.StringIO(),
        max_actions,
        continuous=continuous,
        smoothing_filter=smoothing_filter,
    )
    sample_every = max(1, round(layout.robot.ros.control_hz / 10))
    trajectory = [pose.Point2D(x=start.x, y=start.y)]
    actions: list[dict[str, Any]] = []
    dt = 1.0 / layout.robot.ros.control_hz
    sim_time = start.stamp
    current = start
    command_id = 0
    reason = ""
    try:
        command = hierarchy.next_action(current)
        while command is not None:
            command_id += 1
            decision = hierarchy.decision or {}
            if not motion_controller.submit(
                command_id, command.target, sim_time
            ):
                raise RuntimeError(f"Controller rejected action {command_id}")

            ticks = 0
            while True:
                velocity = motion_controller.tick(sim_time)
                sim_time += dt
                current = advance_unicycle(current, velocity, dt, sim_time)
                motion_controller.update_pose(current)
                ticks += 1
                if ticks % sample_every == 0:
                    append_position(
                        trajectory, pose.Point2D(x=current.x, y=current.y)
                    )
                result = motion_controller.take_result()
                if result is not None:
                    if result.outcome != "target_reached" and not (
                        continuous
                        and result.reason in ("motion_timeout", "pose_timeout")
                    ):
                        reason = f"motion_{result.outcome}:{result.reason}"
                    break
                if motion_controller.state in (
                    motion.State.FAULT,
                    motion.State.STOPPED,
                ):
                    reason = (
                        f"motion_{motion_controller.state.value}:"
                        f"{motion_controller.reason}"
                    )
                    break
                if (
                    ticks
                    > math.ceil(layout.robot.motion.motion_timeout / dt) + 2
                ):
                    reason = "simulation_timeout"
                    break

            append_position(trajectory, pose.Point2D(x=current.x, y=current.y))
            actions.append(
                {
                    "command_id": command_id,
                    "policy_decision": decision,
                    "action": command.model_dump(),
                    "target_world": command.target.model_dump(),
                    "actual_end_world": {"x": current.x, "y": current.y},
                    "motion_result": (
                        result.model_dump() if result is not None else None
                    ),
                }
            )
            if reason:
                break
            if continuous and command_id >= max_actions:
                reason = "action_limit"
                break
            command = hierarchy.next_action(current)
            if command is None:
                reason = hierarchy.reason or "policy_ended"
        if not reason:
            reason = hierarchy.reason or "policy_ended"
    finally:
        hierarchy.close()

    return {
        "reason": reason,
        "actions": actions,
        "trajectory_world_m": [point.model_dump() for point in trajectory],
        "start_pose_world": start.model_dump(),
        "end_pose_world": current.model_dump(),
        "simulated_seconds": sim_time - start.stamp,
        "controller_dt_seconds": dt,
    }
