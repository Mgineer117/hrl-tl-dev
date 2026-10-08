"""CLI entry point to simulate the learned Zone policy from an initial pose."""

from __future__ import annotations

import collections.abc
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from absl import app, flags, logging
from sb3_hrl.option.policies.primitive_step_ppo import PrimitiveStepPPO

from hrl_tl.config.meta_option import TLMetaOptionWrapperConfigReader
from hrl_tl.robot_demo import pose
from hrl_tl.robot_demo.control import smoothing
from hrl_tl.robot_demo.simulation import trajectory_sim
from hrl_tl.robot_demo.world import arena, arena_figure

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "arena",
    "configs/robot_demo/arenas/seed_383.json",
    "Path to the arena layout JSON.",
)
flags.DEFINE_float("raw_x", -2.5447392578125, "Raw mocap x position in meters.")
flags.DEFINE_float("raw_y", 0.023682369, "Raw mocap y position in meters.")
flags.DEFINE_float("yaw", -1.53085881, "Raw mocap yaw in radians.")
flags.DEFINE_boolean(
    "arena_start",
    True,
    "Start at fixed spawn with world yaw 0 if True, else from raw QTM coordinates.",
)
flags.DEFINE_integer("seed", None, "Policy RNG seed (defaults to arena seed).")
flags.DEFINE_integer(
    "max_actions", 250, "Maximum actions to simulate.", lower_bound=1
)
flags.DEFINE_string(
    "output",
    "artifacts/fixed_seed/seed_383_trajectory.png",
    "Output trajectory image path.",
)
flags.DEFINE_string(
    "wrapper_config",
    "configs/wrapper.yaml",
    "Path to wrapper YAML configuration.",
)
flags.DEFINE_string(
    "primitive_model",
    "models/best_model.zip",
    "Path to primitive model checkpoint.",
)
flags.DEFINE_string(
    "upper_model",
    "models/final_model_8.j.b_30.0M_rep_2.zip",
    "Path to high-level PPO checkpoint.",
)
_SMOOTHING = flags.DEFINE_enum(
    "smoothing",
    "butterworth",
    ["none", "ema", "butterworth"],
    "Action smoothing filter to apply.",
)
_EMA_ALPHA = flags.DEFINE_float(
    "ema_alpha",
    0.4,
    "Smoothing factor for exponential moving average (0, 1].",
    lower_bound=0.0,
    upper_bound=1.0,
)
_BUTTER_CUTOFF = flags.DEFINE_float(
    "butter_cutoff",
    1.0,
    "Cutoff frequency in Hz for Butterworth filter.",
    lower_bound=0.0,
)
_BUTTER_ORDER = flags.DEFINE_integer(
    "butter_order",
    2,
    "Order of the Butterworth filter.",
    lower_bound=1,
)


def main(argv: collections.abc.Sequence[str]) -> None:
    """CLI entry point for policy simulation."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    arena_path = Path(FLAGS.arena).resolve()
    layout = arena.load(arena_path)

    primitive_path = Path(FLAGS.primitive_model).resolve()
    upper_path = Path(FLAGS.upper_model).resolve()
    for path in (primitive_path, upper_path):
        if not path.is_file():
            raise app.UsageError(f"Missing policy checkpoint: {path}")

    if FLAGS.arena_start:
        point = pose.zone_to_world(layout.start, layout.robot.frame)
        start = pose.Pose2D(
            x=point.x, y=point.y, yaw=0.0, stamp=0.0, received_at=0.0
        )
    else:
        raw_start = pose.Pose2D(
            x=FLAGS.raw_x,
            y=FLAGS.raw_y,
            yaw=FLAGS.yaw,
            stamp=0.0,
            received_at=0.0,
        )
        start = pose.mocap_to_world(raw_start, layout.robot.ros)

    if not layout.robot.bounds.contains(
        start.x, start.y, layout.robot.motion.wall_stop_margin
    ):
        raise app.UsageError(
            "Corrected start pose is outside the controller wall buffer."
        )

    wrapper_yaml_path = Path(FLAGS.wrapper_config).resolve()
    reader = TLMetaOptionWrapperConfigReader.model_validate(
        yaml.safe_load(wrapper_yaml_path.read_text())
    )
    reader.low_level_policy_args["model_path"] = str(primitive_path)
    reader.max_episode_steps = layout.max_steps
    wrapper = reader.to_config()

    torch.set_num_threads(1)
    upper = PrimitiveStepPPO.load(upper_path, device="cpu")

    seed = layout.seed if FLAGS.seed is None else FLAGS.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    smoothing_filter = smoothing.create_smoothing_filter(
        _SMOOTHING.value,
        alpha=_EMA_ALPHA.value,
        cutoff_hz=_BUTTER_CUTOFF.value,
        order=_BUTTER_ORDER.value,
    )

    result = trajectory_sim.simulate(
        layout,
        upper,
        wrapper.wrapper_kwargs,
        start,
        FLAGS.max_actions,
        smoothing_filter=smoothing_filter,
    )
    result["smoothing"] = {
        "method": _SMOOTHING.value,
        "ema_alpha": _EMA_ALPHA.value,
        "butter_cutoff": _BUTTER_CUTOFF.value,
        "butter_order": _BUTTER_ORDER.value,
    }

    output = Path(FLAGS.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    start_zone = pose.world_to_zone(start, layout.robot.frame)
    end = result["end_pose_world"]
    end_zone = pose.world_to_zone(
        pose.Point2D(x=end["x"], y=end["y"]), layout.robot.frame
    )
    summary = {
        "reason": result["reason"],
        "actions": len(result["actions"]),
        "start_world": [start.x, start.y],
        "end_world": [end["x"], end["y"]],
    }
    sampled = [
        pose.Point2D.model_validate(p) for p in result["trajectory_world_m"]
    ]
    arena_figure.save(
        layout, output, trajectory=sampled, summary=summary, overwrite=True
    )
    result["source_qtm_pose"] = {
        "source": "arena_start" if FLAGS.arena_start else "raw_qtm",
        "raw_x_m": None if FLAGS.arena_start else FLAGS.raw_x,
        "raw_y_m": None if FLAGS.arena_start else FLAGS.raw_y,
        "yaw_rad": 0.0 if FLAGS.arena_start else FLAGS.yaw,
        "corrected_zone_position": {"x": start_zone.x, "y": start_zone.y},
        "final_zone_position": {"x": end_zone.x, "y": end_zone.y},
    }
    result["arena_identity"] = layout.identity
    result["random_seed"] = seed
    result["image"] = str(output)
    trace_path = output.with_suffix(".json")
    trace_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    logging.info(
        "Rollout reason: %s; actions: %d",
        result["reason"],
        len(result["actions"]),
    )
    logging.info("Saved arena trajectory plot: %s", output)
    logging.info("Saved rollout trace data: %s", trace_path)


if __name__ == "__main__":
    app.run(main)
