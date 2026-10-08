"""Network and ROS 2 communication preflight checks for physical robot runs."""

from __future__ import annotations

import logging
import socket
import time

import rclpy  # ty: ignore[unresolved-import]
from geometry_msgs.msg import PoseStamped
from rclpy.qos import qos_profile_sensor_data  # ty: ignore[unresolved-import]

from hrl_tl.robot_demo import pose
from hrl_tl.robot_demo.control.ros_adapter import decode_pose
from hrl_tl.robot_demo.world import arena

_LOGGER: logging.Logger = logging.getLogger(__name__)


def preflight(robot_ip: str, layout: arena.Arena) -> None:
    """Verify robot SSH connectivity and ROS 2 topic subscriptions before motion.

    Args:
        robot_ip: Target robot IP address.
        layout: Loaded arena specification.

    Raises:
        RuntimeError: If SSH is unreachable, topics are missing, or poses are invalid.
    """
    try:
        with socket.create_connection((robot_ip, 22), timeout=2):
            pass
    except OSError as error:
        raise RuntimeError(
            f"Robot {robot_ip}: SSH port 22 is unreachable: {error}"
        ) from error

    settings = layout.robot.ros
    pose_topic = settings.pose_topic
    cmd_topic = settings.cmd_vel_topic
    cmd_type = settings.cmd_vel_type

    rclpy.init(args=[])
    node = rclpy.create_node("zone_robot_preflight")
    received: list[PoseStamped] = []
    subscription = node.create_subscription(
        PoseStamped, pose_topic, received.append, qos_profile_sensor_data
    )
    try:
        deadline = time.monotonic() + 10.0
        topics = dict(node.get_topic_names_and_types())
        expected = (
            "geometry_msgs/msg/Twist"
            if cmd_type == "twist"
            else "geometry_msgs/msg/TwistStamped"
        )
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
            topics = dict(node.get_topic_names_and_types())
            if (
                pose_topic in topics
                and "geometry_msgs/msg/PoseStamped" in topics[pose_topic]
                and len(received) >= 2
                and (
                    received[-1].header.stamp.sec,
                    received[-1].header.stamp.nanosec,
                )
                > (
                    received[-2].header.stamp.sec,
                    received[-2].header.stamp.nanosec,
                )
                and expected in topics.get(cmd_topic, [])
                and node.count_subscribers(cmd_topic) > 0
            ):
                try:
                    corrected = decode_pose(
                        received[-1], settings, time.monotonic()
                    )
                except ValueError as error:
                    raise RuntimeError(
                        f"Invalid mocap pose: {error}"
                    ) from error
                if not layout.robot.bounds.contains(
                    corrected.x,
                    corrected.y,
                    layout.robot.motion.wall_stop_margin,
                ):
                    raise RuntimeError(
                        f"Corrected mocap pose ({corrected.x:.3f},"
                        f" {corrected.y:.3f}) is outside the calibrated wall"
                        " buffer"
                    )
                zone = pose.world_to_zone(corrected, layout.robot.frame)
                _LOGGER.info(
                    "Preflight OK: %s:22, %s, %s (%s)",
                    robot_ip,
                    pose_topic,
                    cmd_topic,
                    expected,
                )
                _LOGGER.info(
                    "Raw QTM: (%.3f, %.3f) m; corrected world: (%.3f, %.3f) m;"
                    " Zone: (%.3f, %.3f)",
                    received[-1].pose.position.x,
                    received[-1].pose.position.y,
                    corrected.x,
                    corrected.y,
                    zone.x,
                    zone.y,
                )
                return

        pose_types = topics.get(pose_topic, [])
        cmd_types = topics.get(cmd_topic, [])
        pose_publishers = node.count_publishers(pose_topic)
        cmd_subscribers = node.count_subscribers(cmd_topic)
        problems: list[str] = []
        if "geometry_msgs/msg/PoseStamped" not in pose_types:
            problems.append(
                f"{pose_topic} type missing (observed {pose_types or 'no topic'})"
            )
        elif len(received) < 2:
            problems.append(
                "fewer than two PoseStamped messages arrived on"
                f" {pose_topic} during the 10 s wait"
            )
        elif received[-1].header.stamp == received[-2].header.stamp:
            problems.append(f"{pose_topic} source timestamps did not advance")
        if expected not in cmd_types:
            problems.append(
                f"{cmd_topic} type must be {expected} (observed"
                f" {cmd_types or 'no topic'})"
            )
        elif cmd_subscribers == 0:
            problems.append(f"{cmd_topic} has no subscribers")
        raise RuntimeError(
            "ROS preflight failed: "
            + "; ".join(problems)
            + f". Diagnostics: pose publishers={pose_publishers}, "
            f"pose messages received={len(received)}, "
            f"{cmd_topic} subscribers={cmd_subscribers}, visible topics={topics}"
        )
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()
