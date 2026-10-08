"""Hyperparameter tuning runner for PPO meta-policy with learned LTL subpolicy."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import contgrid  # noqa: F401
import yaml
from absl import app, flags
from immutabledict import immutabledict
from rl_pipeline.sb3 import SB3Pipeline

import hrl_tl.envs.fetch  # noqa: F401
from hrl_tl.config.meta_option import TLMetaOptionPipelineConfigReader
from hrl_tl.tuning import (
    TuningEnvPaths,
    TuningWorkflowConfig,
    build_optuna_config,
    export_replicate_yaml,
    normalize_device,
    run_tuning_stage,
)

os.environ.setdefault("MUJOCO_GL", "egl")

CONFIG_BY_ENV: immutabledict[str, TuningEnvPaths] = immutabledict({
    "fetch": TuningEnvPaths(
        base_pipeline=Path("configs/fetch/train/15.b/15.b.a_tune_meta.yaml"),
        tuning_config=Path("configs/tuning/spec_cond_hl_fetch.yaml"),
        best_export=Path("configs/tuning/best/spec_cond_hl_fetch_best.yaml"),
        replicate_export=Path(
            "configs/fetch/train/15.b/spec_cond_meta_tuned_rep.yaml"
        ),
    ),
})

_ENV = flags.DEFINE_enum(
    "env",
    default="fetch",
    enum_values=["fetch"],
    help="Target environment name for meta-policy tuning.",
)
_TRIALS = flags.DEFINE_integer(
    "trials",
    default=None,
    lower_bound=1,
    help="Override total number of Optuna tuning trials.",
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
    help="Override computing device (e.g. '0', '1', 'cuda:0', 'cpu').",
)


def main(argv: Sequence[str]) -> None:
    """Executes the PPO meta-policy hyperparameter tuning workflow.

    Args:
        argv: Command-line arguments.

    Raises:
        app.UsageError: If unexpected positional arguments are supplied.
    """
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    env_info = CONFIG_BY_ENV[_ENV.value]
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

    reader = TLMetaOptionPipelineConfigReader(**single_cfg)
    pipeline_config = reader.to_config()

    if _DEVICE.value is not None:
        pipeline_config.device = normalize_device(_DEVICE.value)

    pipeline = SB3Pipeline(config=pipeline_config, verbose=True)

    study = run_tuning_stage(
        pipeline=pipeline,
        optuna_config=optuna_config,
        export_yaml_path=env_info.best_export,
    )

    export_replicate_yaml(
        source_yaml_path=env_info.base_pipeline,
        target_yaml_path=env_info.replicate_export,
        best_params=study.best_params,
        tune_params=optuna_config.tune_params,
        num_replicates=5,
        replicate_signature="rep_{rep_id}",
        replicate_start_id=0,
    )


if __name__ == "__main__":
    app.run(main)
