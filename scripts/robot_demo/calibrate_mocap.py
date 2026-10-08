"""Measure the QTM-to-world translation at world (0, 0) for every arena."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

from hrl_tl.robot_demo import config, pose

ARENAS = Path("configs/robot_demo/arenas")
SEEDS = (0, 295, 383)


def correction(
    samples: list[pose.Pose2D], ros: config.RosConfig
) -> tuple[float, float]:
    """Find translation that places the stationary robot reference at (0, 0)."""
    if not samples:
        raise ValueError("No Qualisys samples received")
    mean_x = sum(sample.x for sample in samples) / len(samples)
    mean_y = sum(sample.y for sample in samples) / len(samples)
    if any(
        math.hypot(sample.x - mean_x, sample.y - mean_y) > 0.02
        for sample in samples
    ):
        raise ValueError("Robot/marker moved more than 2 cm during calibration")
    if any(
        abs(pose.angle_difference(sample.yaw, samples[0].yaw)) > 0.1
        for sample in samples
    ):
        raise ValueError("Robot rotated during calibration")
    no_offset = ros.model_copy(
        update={"mocap_offset_x_m": 0.0, "mocap_offset_y_m": 0.0}
    )
    world = [pose.mocap_to_world(sample, no_offset) for sample in samples]
    return (
        -sum(p.x for p in world) / len(world),
        -sum(p.y for p in world) / len(world),
    )


def collect(
    topic: str, frame: str, encoding: str, count: int, timeout: float
) -> list[pose.Pose2D]:
    """Collect advancing PoseStamped samples from the Qualisys ROS topic."""
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.qos import qos_profile_sensor_data

    samples: list[pose.Pose2D] = []
    last_stamp = -1.0
    rclpy.init()
    node = rclpy.create_node("calibrate_mocap")

    def on_pose(message: PoseStamped) -> None:
        nonlocal last_stamp
        if message.header.frame_id != frame:
            raise ValueError(
                f"Expected frame {frame!r}, got {message.header.frame_id!r}"
            )
        stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
        if stamp <= last_stamp:
            return
        rotation = message.pose.orientation
        yaw = rotation.z if encoding == "legacy_euler" else pose.quaternion_yaw(
            rotation.x, rotation.y, rotation.z, rotation.w
        )
        samples.append(
            pose.Pose2D(
                x=message.pose.position.x,
                y=message.pose.position.y,
                yaw=yaw,
                stamp=stamp,
                received_at=time.monotonic(),
            )
        )
        last_stamp = stamp

    node.create_subscription(
        PoseStamped, topic, on_pose, qos_profile_sensor_data
    )
    try:
        deadline = time.monotonic() + timeout
        while len(samples) < count and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if len(samples) < count:
            raise TimeoutError(
                f"Received {len(samples)}/{count} fresh poses from {topic}"
            )
        return samples
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--arenas-dir", type=Path, default=ARENAS)
    args = parser.parse_args()
    if args.samples < 2 or args.timeout <= 0:
        parser.error("--samples must be >= 2 and --timeout must be positive")

    paths = [args.arenas_dir / f"seed_{seed}.json" for seed in SEEDS]
    arenas = [json.loads(path.read_text()) for path in paths]
    base = arenas[0]["robot"]["ros"]
    for arena in arenas[1:]:
        if arena["robot"]["frame"] != arenas[0]["robot"]["frame"]:
            raise ValueError("Arena world frames differ")
        other = arena["robot"]["ros"]
        if any(
            other.get(key) != value
            for key, value in base.items()
            if key not in ("mocap_offset_x_m", "mocap_offset_y_m")
        ):
            raise ValueError("Arena ROS settings differ")
    ros = config.RosConfig.model_validate(base)
    print(
        f"Keep the robot stationary at arena-world (0, 0) m; "
        f"reading {ros.pose_topic}..."
    )
    samples = collect(
        ros.pose_topic,
        ros.pose_frame,
        ros.pose_encoding,
        args.samples,
        args.timeout,
    )
    offset_x, offset_y = correction(samples, ros)
    for arena in arenas:
        arena["robot"]["ros"]["mocap_offset_x_m"] = offset_x
        arena["robot"]["ros"]["mocap_offset_y_m"] = offset_y
    for path, arena in zip(paths, arenas):
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(arena, indent=2, allow_nan=False) + "\n"
        )
        os.replace(temporary, path)
    print(
        f"Saved correction ({offset_x:.6f}, {offset_y:.6f}) m to "
        f"{', '.join(map(str, paths))}"
    )


if __name__ == "__main__":
    main()
