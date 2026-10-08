"""Hyperparameter tuning runner for ALLO representation pretraining."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import contgrid  # noqa: F401 - Register gym envs
import yaml
from absl import app, flags
from immutabledict import immutabledict
from rl_pipeline.sb3 import SB3Pipeline, SB3PipelineConfigReader

import hrl_tl.envs.fetch  # noqa: F401 - Register fetch gym envs
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
                "configs/fr_cont/train/allo/7.d.a_allo_ext.yaml"
            ),
            tuning_config=Path("configs/tuning/allo_pr_fr_cont.yaml"),
            best_export=Path(
                "configs/tuning/best/allo_pretrain_fr_cont_best.yaml"
            ),
            replicate_export=Path(
                "configs/fr_cont/train/allo/allo_ext_tuned_rep.yaml"
            ),
        ),
        "zone": TuningEnvPaths(
            base_pipeline=Path("configs/zone/train/allo/7.g.a_allo_ext.yaml"),
            tuning_config=Path("configs/tuning/allo_pr_zone.yaml"),
            best_export=Path(
                "configs/tuning/best/allo_pretrain_zone_best.yaml"
            ),
            replicate_export=Path(
                "configs/zone/train/allo/allo_ext_tuned_rep.yaml"
            ),
        ),
        "fetch": TuningEnvPaths(
            base_pipeline=Path("configs/fetch/train/allo/7.j.c_allo_ext.yaml"),
            tuning_config=Path("configs/tuning/allo_pr_fetch.yaml"),
            best_export=Path(
                "configs/tuning/best/allo_pretrain_fetch_best.yaml"
            ),
            replicate_export=Path(
                "configs/fetch/train/allo/allo_ext_tuned_rep.yaml"
            ),
        ),
    }
)

_ENV = flags.DEFINE_enum(
    "env",
    default="fr_cont",
    enum_values=["fr_cont", "zone", "fetch"],
    help="Target environment name.",
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
    help="Override per-trial optimization timesteps.",
)
_BUFFER_SIZE = flags.DEFINE_integer(
    "buffer_size",
    default=None,
    lower_bound=1000,
    help="Override replay buffer size (transitions collected before training).",
)
_DASHBOARD = flags.DEFINE_boolean(
    "dashboard",
    default=False,
    help="Launch Optuna dashboard automatically.",
)
_DASHBOARD_PORT = flags.DEFINE_integer(
    "dashboard_port",
    default=8082,
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
    """Execute ALLO representation pretraining hyperparameter optimization.

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
        tuning_raw = yaml.safe_load(f)

    tune_workflow_cfg = TuningWorkflowConfig(**tuning_raw)
    if _TRIALS.value is not None:
        tune_workflow_cfg.n_trials = _TRIALS.value
    if _TIMESTEPS.value is not None:
        tune_workflow_cfg.total_timesteps = _TIMESTEPS.value
    if _DASHBOARD.value:
        tune_workflow_cfg.launch_dashboard = True
        tune_workflow_cfg.dashboard_port = _DASHBOARD_PORT.value

    optuna_config = build_optuna_config(tune_workflow_cfg)

    with env_info.base_pipeline.open("r", encoding="utf-8") as f:
        raw_base = yaml.safe_load(f)

    if "single_pipeline_config" in raw_base:
        single_cfg = raw_base["single_pipeline_config"]
    else:
        single_cfg = raw_base

    single_pipeline_reader = SB3PipelineConfigReader(**single_cfg)
    pipeline_config = single_pipeline_reader.to_config()
    if _DEVICE.value is not None:
        pipeline_config.device = normalize_device(_DEVICE.value)
    if _BUFFER_SIZE.value is not None:
        pipeline_config.algo_config.algo_kwargs["buffer_size"] = (
            _BUFFER_SIZE.value
        )
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
            SB3PipelineConfigReader
        ].from_yaml(str(replicate_path))
        rep_pipeline = SB3ReplicatePipeline(config=rep_reader.to_config())
        rep_pipeline.train_on_unsaved_model()
        models = rep_pipeline.load_models("best")
        rep_pipeline.record_replays(models)


if __name__ == "__main__":
    app.run(main)
