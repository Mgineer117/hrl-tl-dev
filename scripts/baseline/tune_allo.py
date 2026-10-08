"""Hyperparameter tuning runner for Allo meta-policy baseline."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import contgrid  # noqa: F401 - Register gym envs
import yaml
from absl import app, flags
from immutabledict import immutabledict
from rl_pipeline.sb3 import SB3Pipeline

import hrl_tl.envs.fetch  # noqa: F401 - Register fetch gym envs
from hrl_tl.config.wrapper import AlloMetaPipelineConfigReader
from hrl_tl.tuning import (
    TuningEnvPaths,
    TuningWorkflowConfig,
    build_optuna_config,
    export_replicate_yaml,
    normalize_device,
    run_tuning_stage,
)

CONFIG_BY_ENV: immutabledict[str, TuningEnvPaths] = immutabledict(
    {
        "fr_cont": TuningEnvPaths(
            base_pipeline=Path(
                "configs/fr_cont/train/allo/7.f.a_allo_meta_rep0.yaml"
            ),
            tuning_config=Path("configs/tuning/allo_hl_fr_cont.yaml"),
            best_export=Path("configs/tuning/best/allo_hl_fr_cont_best.yaml"),
            replicate_export=Path(
                "configs/fr_cont/train/allo/allo_meta_tuned_rep.yaml"
            ),
        ),
        "zone": TuningEnvPaths(
            base_pipeline=Path(
                "configs/zone/train/allo/7.i.a_allo_meta_rep0.yaml"
            ),
            tuning_config=Path("configs/tuning/allo_hl_zone.yaml"),
            best_export=Path("configs/tuning/best/allo_hl_zone_best.yaml"),
            replicate_export=Path(
                "configs/zone/train/allo/allo_meta_tuned_rep.yaml"
            ),
        ),
        "fetch": TuningEnvPaths(
            base_pipeline=Path(
                "configs/fetch/train/allo/7.l.a_allo_meta_rep0.yaml"
            ),
            tuning_config=Path("configs/tuning/allo_hl_fetch.yaml"),
            best_export=Path("configs/tuning/best/allo_hl_fetch_best.yaml"),
            replicate_export=Path(
                "configs/fetch/train/allo/allo_meta_tuned_rep.yaml"
            ),
        ),
    }
)

_ENV = flags.DEFINE_enum(
    "env",
    default="fr_cont",
    enum_values=["fr_cont", "zone", "fetch"],
    help="Target environment name ('fr_cont', 'zone', or 'fetch').",
)
_TRIALS = flags.DEFINE_integer(
    "trials",
    default=None,
    lower_bound=1,
    help="Override total number of tuning trials.",
)
_TIMESTEPS = flags.DEFINE_integer(
    "timesteps",
    default=None,
    lower_bound=1,
    help="Override training timesteps per trial.",
)
_DASHBOARD = flags.DEFINE_boolean(
    "dashboard",
    default=False,
    help="Launch Optuna dashboard automatically.",
)
_DASHBOARD_PORT = flags.DEFINE_integer(
    "dashboard_port",
    default=8080,
    lower_bound=1,
    help="Port for Optuna dashboard web server.",
)
_DEVICE = flags.DEFINE_string(
    "device",
    default=None,
    help="Override GPU device (e.g. '0', '1', '2', 'cuda:0', 'cpu').",
)
_TRAIN_REPLICATES = flags.DEFINE_boolean(
    "train_replicates",
    default=False,
    help="Immediately train 5 replicates with best hyperparams after tuning.",
)


def main(argv: Sequence[str]) -> None:
    """Execute Allo meta-policy hyperparameter tuning workflow.

    Args:
      argv: Command-line argument vector.

    Raises:
      app.UsageError: If unexpected positional arguments are supplied.
    """
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    env_name: str = _ENV.value
    env_info = CONFIG_BY_ENV[env_name]

    with env_info.tuning_config.open("r", encoding="utf-8") as f:
        workflow_data = yaml.safe_load(f)

    if _TRIALS.value is not None:
        workflow_data["n_trials"] = _TRIALS.value
    if _TIMESTEPS.value is not None:
        workflow_data["total_timesteps"] = _TIMESTEPS.value
    if _DASHBOARD.value:
        workflow_data["launch_dashboard"] = True
    if _DASHBOARD_PORT.value:
        workflow_data["dashboard_port"] = _DASHBOARD_PORT.value

    workflow_config = TuningWorkflowConfig(**workflow_data)
    optuna_config = build_optuna_config(workflow_config)

    with env_info.base_pipeline.open("r", encoding="utf-8") as f:
        rep_template = yaml.safe_load(f)
    single_cfg = rep_template["single_pipeline_config"]

    single_reader = AlloMetaPipelineConfigReader(**single_cfg)
    pipeline_config = single_reader.to_config()
    if _DEVICE.value is not None:
        pipeline_config.device = normalize_device(_DEVICE.value)
    pipeline = SB3Pipeline(config=pipeline_config)

    study = run_tuning_stage(
        pipeline=pipeline,
        optuna_config=optuna_config,
        export_yaml_path=env_info.best_export,
    )

    replicate_path = export_replicate_yaml(
        source_yaml_path=env_info.base_pipeline,
        target_yaml_path=env_info.replicate_export,
        best_params=study.best_params,
        num_replicates=5,
        replicate_signature="rep_{rep_id}",
    )

    if _TRAIN_REPLICATES.value:
        print(f"\nTraining 5 replicates using {replicate_path}...")
        from rl_pipeline.sb3 import (
            SB3ReplicatePipeline,
            SB3ReplicatePipelineConfigReader,
        )

        rep_reader = SB3ReplicatePipelineConfigReader[
            AlloMetaPipelineConfigReader
        ].from_yaml(str(replicate_path))
        rep_pipeline = SB3ReplicatePipeline(config=rep_reader.to_config())
        rep_pipeline.train_on_unsaved_model()
        models = rep_pipeline.load_models("best")
        rep_pipeline.record_replays(models)


if __name__ == "__main__":
    app.run(main)
