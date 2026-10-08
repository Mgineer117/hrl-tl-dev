"""Run the trained Zone hierarchy from Qualisys poses on a physical TurtleBot3."""

from __future__ import annotations

import collections.abc
import io
import json
import math
import os
import time
from pathlib import Path
from typing import Literal

import torch
import yaml
from absl import app, flags, logging
from sb3_hrl.option.policies.primitive_step_ppo import PrimitiveStepPPO

from hrl_tl.config.meta_option import TLMetaOptionWrapperConfigReader
from hrl_tl.robot_demo import pose
from hrl_tl.robot_demo.control import action, smoothing
from hrl_tl.robot_demo.inference.hierarchy import RobotHierarchy
from hrl_tl.robot_demo.inference.learned_policy import LearnedPolicyProvider
from hrl_tl.robot_demo.inference.manual import SingleActionProvider
from hrl_tl.robot_demo.world import arena

FLAGS = flags.FLAGS
flags.DEFINE_enum(
    "mode",
    "check",
    ["check", "preflight", "move", "run"],
    "Execution mode.",
)
flags.DEFINE_string(
    "arena", "configs/arena.json", "Path to arena layout JSON specification."
)
flags.DEFINE_integer(
    "max_actions",
    250,
    "Maximum actions (for compatibility in check and policy modes).",
    lower_bound=1,
)
flags.DEFINE_string(
    "robot_ip", "192.168.0.77", "Robot IP address for preflight."
)
flags.DEFINE_float(
    "angle_deg",
    None,
    "Zone-frame direction (0, 45, ..., 315 degrees) for move mode.",
)
flags.DEFINE_float(
    "distance_m",
    None,
    "Configured primitive step distance in meters for move mode.",
)
flags.DEFINE_string(
    "wrapper_config",
    "configs/wrapper.yaml",
    "Path to wrapper YAML configuration.",
)
flags.DEFINE_string(
    "primitive_model",
    "models/best_model.zip",
    "Path to primitive policy checkpoint.",
)
flags.DEFINE_string(
    "upper_model",
    "models/final_model_8.j.b_30.0M_rep_2.zip",
    "Path to high-level PPO checkpoint.",
)
flags.DEFINE_string(
    "logs_dir", "logs", "Directory where run logs will be recorded."
)
_SMOOTHING = flags.DEFINE_enum(
    "smoothing",
    "butterworth",
    ["none", "ema", "butterworth"],
    "Action smoothing filter to apply.",
)
_EMA_ALPHA = flags.DEFINE_float(
    "ema_alpha", 0.4, "EMA alpha (0, 1].", lower_bound=0.0, upper_bound=1.0
)
_BUTTER_CUTOFF = flags.DEFINE_float(
    "butter_cutoff", 1.0, "Cutoff frequency in Hz.", lower_bound=0.0
)
_BUTTER_ORDER = flags.DEFINE_integer(
    "butter_order", 2, "Butterworth filter order.", lower_bound=1
)


def _resolve_mode(
    argv: collections.abc.Sequence[str],
) -> Literal["check", "preflight", "move", "run"]:
    """Resolves mode from positional argument or flag."""
    if len(argv) > 1:
        candidate = argv[1].lower()
        if candidate in ("check", "preflight", "move", "run"):
            return candidate  # type: ignore[return-value]
        raise app.UsageError(
            f"Unknown mode '{candidate}'. Allowed modes: check, preflight, move, run."
        )
    return FLAGS.mode  # type: ignore[return-value]


def main(argv: collections.abc.Sequence[str]) -> None:
    """CLI entry point for TurtleBot3 robot demo execution."""
    mode = _resolve_mode(argv)
    if mode == "move":
        if FLAGS.angle_deg is None or FLAGS.distance_m is None:
            raise app.UsageError(
                "move mode requires both --angle_deg and --distance_m."
            )
        if not math.isfinite(FLAGS.angle_deg) or not math.isfinite(
            FLAGS.distance_m
        ):
            raise app.UsageError(
                "move angle and distance must be finite numbers."
            )
    elif FLAGS.angle_deg is not None or FLAGS.distance_m is not None:
        raise app.UsageError(
            "--angle_deg and --distance_m are only valid in move mode."
        )

    if (
        mode in ("preflight", "move", "run")
        and os.environ.get("ROS_DOMAIN_ID") != "40"
    ):
        raise app.UsageError(
            "Set ROS_DOMAIN_ID=40 before connecting to the robot."
        )

    arena_path = Path(FLAGS.arena).resolve()
    layout = arena.load(arena_path)

    manual_movement = None
    manual_metadata = None
    interpreted_angle: float | None = None
    actual_distance: float | None = None
    if mode == "move":
        if FLAGS.angle_deg is None or FLAGS.distance_m is None:
            raise app.UsageError(
                "move mode requires both --angle_deg and --distance_m."
            )
        manual_movement, interpreted_angle, actual_distance = (
            action.interpret_manual_action(
                layout, FLAGS.angle_deg, FLAGS.distance_m
            )
        )
        manual_metadata = {
            "event": "manual_action",
            "zone_direction_deg": interpreted_angle,
            "requested_distance_m": FLAGS.distance_m,
            "interpreted_distance_m": actual_distance,
            "movement": manual_movement.model_dump(),
        }

    primitive_path = Path(FLAGS.primitive_model).resolve()
    upper_path = Path(FLAGS.upper_model).resolve()
    if mode in ("check", "run"):
        for path in (primitive_path, upper_path):
            if not path.is_file():
                raise app.UsageError(f"Missing robot policy checkpoint: {path}")

    if mode in ("preflight", "move", "run"):
        from hrl_tl.robot_demo.control import preflight

        preflight.preflight(FLAGS.robot_ip, layout)
    if mode == "preflight":
        logging.info("Preflight checks passed.")
        return

    logs_root = Path(FLAGS.logs_dir).resolve()
    if mode == "move":
        from hrl_tl.robot_demo.control import ros_adapter

        if (
            manual_movement is None
            or manual_metadata is None
            or interpreted_angle is None
            or actual_distance is None
        ):
            raise app.UsageError(
                "Failed to initialize manual movement parameters."
            )
        log_dir = logs_root / str(time.time_ns())
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "manual_action.json").write_text(
            json.dumps(manual_metadata, indent=2, allow_nan=False) + "\n"
        )
        provider = SingleActionProvider(
            manual_movement,
            angle_deg=interpreted_angle,
            distance_m=actual_distance,
        )

        logging.info(
            "One action: Zone direction %g deg, step %.5f m",
            manual_metadata["zone_direction_deg"],
            manual_metadata["interpreted_distance_m"],
        )
        logging.info("Run data: %s", log_dir)
        exit_code = ros_adapter.run(
            layout.robot, provider, log_dir / "motion.jsonl"
        )
        if exit_code != 0:
            raise SystemExit(exit_code)
        return

    wrapper_yaml_path = Path(FLAGS.wrapper_config).resolve()
    reader = TLMetaOptionWrapperConfigReader.model_validate(
        yaml.safe_load(wrapper_yaml_path.read_text())
    )
    reader.low_level_policy_args["model_path"] = str(primitive_path)
    reader.max_episode_steps = layout.max_steps
    wrapper = reader.to_config()

    torch.set_num_threads(1)
    upper = PrimitiveStepPPO.load(upper_path, device="cpu")

    smoothing_filter = smoothing.create_smoothing_filter(
        _SMOOTHING.value,
        alpha=_EMA_ALPHA.value,
        cutoff_hz=_BUTTER_CUTOFF.value,
        order=_BUTTER_ORDER.value,
    )

    if mode == "check":
        with io.StringIO() as log:
            hierarchy = RobotHierarchy(
                layout,
                upper,
                wrapper.wrapper_kwargs,
                log,
                FLAGS.max_actions,
                smoothing_filter=smoothing_filter,
            )
            try:
                start = pose.zone_to_world(layout.start, layout.robot.frame)
                now = time.monotonic()
                command = hierarchy.next_action(
                    pose.Pose2D(
                        x=start.x,
                        y=start.y,
                        yaw=-layout.robot.frame.rotation_rad,
                        stamp=now,
                        received_at=now,
                    )
                )
                print(
                    json.dumps(
                        {
                            "decision": hierarchy.decision,
                            "primitive_action": (
                                command.model_dump() if command else None
                            ),
                            "observation": json.loads(log.getvalue())[
                                "observation"
                            ],
                            "cmd_vel_topic": layout.robot.ros.cmd_vel_topic,
                            "cmd_vel_type": layout.robot.ros.cmd_vel_type,
                        },
                        indent=2,
                    )
                )
                if command is None:
                    raise SystemExit(1)
                return
            finally:
                hierarchy.close()

    log_dir = logs_root / str(time.time_ns())
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "policy.jsonl").open("x") as policy_log:
        from hrl_tl.robot_demo.control import ros_adapter

        learned_provider = LearnedPolicyProvider(
            layout,
            upper,
            wrapper.wrapper_kwargs,
            policy_log,
            max_actions=FLAGS.max_actions,
            continuous=True,
            require_arena=False,
            smoothing_filter=smoothing_filter,
        )
        try:
            logging.info("Policy log: %s", log_dir / "policy.jsonl")
            logging.info("Motion log: %s", log_dir / "motion.jsonl")
            exit_code = ros_adapter.run(
                layout.robot, learned_provider, log_dir / "motion.jsonl"
            )
            if exit_code != 0:
                raise SystemExit(exit_code)
        finally:
            learned_provider.close()


if __name__ == "__main__":
    app.run(main)
