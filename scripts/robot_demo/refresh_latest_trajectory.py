"""Refresh bundled trajectories and plot from the newest robot run."""

from __future__ import annotations

import collections.abc
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pydantic
import torch
import yaml
from absl import app, flags, logging
from sb3_hrl.option.policies.primitive_step_ppo import PrimitiveStepPPO

from hrl_tl.config.meta_option import TLMetaOptionWrapperConfigReader
from hrl_tl.robot_demo import pose
from hrl_tl.robot_demo.rendering import trajectory_data, trajectory_plotter
from hrl_tl.robot_demo.simulation import trajectory_sim
from hrl_tl.robot_demo.world import arena

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "run_id", None, "Specific logs/<run-id> folder name to process."
)
flags.DEFINE_string(
    "logs_dir", "logs", "Path to logs directory containing robot runs."
)
flags.DEFINE_integer("seed", None, "Arena seed: 0, 295, or 383 (required).")
flags.DEFINE_string(
    "wrapper_config",
    "configs/robot_demo/wrapper.yaml",
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
    "output",
    "latest_trajectory/trajectories.json",
    "Path to write trajectory JSON data.",
)


class RunRecord(pydantic.BaseModel):
    """Execution log metadata and measured trajectory from a robot demonstration."""

    settings: dict[str, Any]
    first_policy: dict[str, Any]
    measured_samples: list[dict[str, Any]]
    commands: int
    stop_reason: str


def latest_run(logs_dir: Path, run_id: str | None) -> Path:
    """Finds the latest or requested run folder containing motion and policy logs.

    Args:
        logs_dir: Root directory containing run subfolders.
        run_id: Optional specific folder name.

    Returns:
        Path to the selected run directory.

    Raises:
        FileNotFoundError: If no valid run directory with logs exists.
    """
    if run_id is not None:
        candidates = [logs_dir / run_id]
    else:
        candidates = sorted(
            (p for p in logs_dir.iterdir() if p.is_dir() and p.name.isdigit()),
            key=lambda p: int(p.name),
            reverse=True,
        )
    for folder in candidates:
        if (folder / "motion.jsonl").is_file() and (
            folder / "policy.jsonl"
        ).is_file():
            return folder
    raise FileNotFoundError(
        "No run found with both motion.jsonl and policy.jsonl."
    )


def read_run(folder: Path) -> RunRecord:
    """Reads execution logs and extracts trajectory and configuration data.

    Args:
        folder: Path to run folder containing jsonl logs.

    Returns:
        A RunRecord containing settings, first policy event, samples, commands, and stop reason.

    Raises:
        ValueError: If logs are missing or incomplete.
    """
    first_policy = None
    with (folder / "policy.jsonl").open() as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("event") == "policy":
                first_policy = event
                break
    if first_policy is None:
        raise ValueError(f"No policy action in {folder}.")

    settings = None
    measured: list[dict[str, Any]] = [first_policy["pose"]]
    commands = 0
    stop_reason = "snapshot_running"
    first_stamp = first_policy["pose"]["stamp"]
    last_stamp = first_stamp
    with (folder / "motion.jsonl").open() as stream:
        for line in stream:
            event = json.loads(line)
            kind = event.get("event")
            if kind == "config":
                settings = event["settings"]
            elif kind == "command":
                commands += 1
            elif kind == "control" and event.get("pose"):
                sample = event["pose"]
                if sample["stamp"] > last_stamp:
                    measured.append(sample)
                    last_stamp = sample["stamp"]
            elif kind == "state" and event.get("state") in ("stopped", "fault"):
                stop_reason = event.get("reason", "")
            elif kind == "shutdown":
                stop_reason = event.get("reason", stop_reason)
    if settings is None or commands == 0:
        raise ValueError(f"Incomplete motion log in {folder}.")
    return RunRecord(
        settings=settings,
        first_policy=first_policy,
        measured_samples=measured,
        commands=commands,
        stop_reason=stop_reason,
    )


def main(argv: collections.abc.Sequence[str]) -> None:
    """CLI entry point for refreshing the latest trajectory."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    if FLAGS.seed not in (0, 295, 383):
        raise app.UsageError("Set --seed to 0, 295, or 383.")

    logs_dir = Path(FLAGS.logs_dir).resolve()
    folder = latest_run(logs_dir, FLAGS.run_id)
    run_record = read_run(folder)

    arena_path = Path(f"configs/robot_demo/arenas/seed_{FLAGS.seed}.json").resolve()
    layout = arena.load(arena_path)
    if run_record.first_policy[
        "arena_id"
    ] != layout.identity or run_record.settings != json.loads(
        layout.robot.model_dump_json()
    ):
        raise ValueError(
            "Current arena/config differs from the logged run; restore its configuration first."
        )

    start = pose.Pose2D.model_validate(
        run_record.first_policy["pose"]
    ).model_copy(update={"stamp": 0.0, "received_at": 0.0})

    primitive_path = Path(FLAGS.primitive_model).resolve()
    upper_path = Path(FLAGS.upper_model).resolve()
    for path in (primitive_path, upper_path):
        if not path.is_file():
            raise app.UsageError(f"Missing policy checkpoint: {path}")

    wrapper_yaml_path = Path(FLAGS.wrapper_config).resolve()
    reader = TLMetaOptionWrapperConfigReader.model_validate(
        yaml.safe_load(wrapper_yaml_path.read_text())
    )
    reader.low_level_policy_args["model_path"] = str(primitive_path)
    reader.max_episode_steps = layout.max_steps
    wrapper = reader.to_config()

    torch.set_num_threads(1)
    upper = PrimitiveStepPPO.load(upper_path, device="cpu")

    random.seed(layout.seed)
    np.random.seed(layout.seed)
    torch.manual_seed(layout.seed)

    result = trajectory_sim.simulate(
        layout,
        upper,
        wrapper.wrapper_kwargs,
        start,
        run_record.commands,
        continuous=True,
    )
    ideal = result["trajectory_world_m"]

    summary = {
        "run_id": folder.name,
        "seed": layout.seed,
        "measured_commands": run_record.commands,
        "measured_samples": len(run_record.measured_samples),
        "stop_reason": run_record.stop_reason,
        "open_loop_actions": len(result["actions"]),
        "open_loop_reason": result["reason"],
        "start_world_m": {"x": start.x, "y": start.y, "yaw": start.yaw},
        "measured_end_world_m": {
            key: run_record.measured_samples[-1][key] for key in ("x", "y")
        },
        "open_loop_end_world_m": {key: ideal[-1][key] for key in ("x", "y")},
    }
    data = {
        **summary,
        "arena_world_m": {
            "bounds": [
                layout.robot.bounds.min_x,
                layout.robot.bounds.max_x,
                layout.robot.bounds.min_y,
                layout.robot.bounds.max_y,
            ],
            "zones": [
                {
                    "color": zone.color,
                    "center": pose.zone_to_world(
                        zone.center, layout.robot.frame
                    ).model_dump(),
                    "radius": zone.radius
                    / layout.robot.frame.sim_units_per_meter,
                }
                for zone in layout.zones
            ],
        },
        "measured_trajectory_world_m": [
            {"x": point["x"], "y": point["y"]}
            for point in run_record.measured_samples
        ],
        "open_loop_trajectory_world_m": ideal,
    }

    output_path = Path(FLAGS.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n"
    )

    traj_data = trajectory_data.load_trajectory_data(output_path)
    png_path = trajectory_plotter.generate_trajectory_plot(
        traj_data, output_path.with_suffix("")
    )

    logging.info("Saved trajectories JSON: %s", output_path)
    logging.info("Saved trajectory plot: %s", png_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app.run(main)
