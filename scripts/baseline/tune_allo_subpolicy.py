"""Hyperparameter tuning runner for ALLO subpolicies."""

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
from hrl_tl.config.wrapper import AlloSB3PipelineConfigReader
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
                "configs/fr_cont/train/allo/7.e.a_allo_subpolicy_rep0.yaml"
            ),
            tuning_config=Path("configs/tuning/allo_sub_fr_cont.yaml"),
            best_export=Path(
                "configs/tuning/best/allo_subpolicy_fr_cont_best.yaml"
            ),
            replicate_export=Path(
                "configs/fr_cont/train/allo/allo_subpolicy_tuned_rep.yaml"
            ),
        ),
        "zone": TuningEnvPaths(
            base_pipeline=Path(
                "configs/zone/train/allo/7.h.a_allo_subpolicy_rep0.yaml"
            ),
            tuning_config=Path("configs/tuning/allo_sub_zone.yaml"),
            best_export=Path(
                "configs/tuning/best/allo_subpolicy_zone_best.yaml"
            ),
            replicate_export=Path(
                "configs/zone/train/allo/allo_subpolicy_tuned_rep.yaml"
            ),
        ),
        "fetch": TuningEnvPaths(
            base_pipeline=Path(
                "configs/fetch/train/allo/7.k.a_allo_subpolicy_rep0.yaml"
            ),
            tuning_config=Path("configs/tuning/allo_sub_fetch.yaml"),
            best_export=Path(
                "configs/tuning/best/allo_subpolicy_fetch_best.yaml"
            ),
            replicate_export=Path(
                "configs/fetch/train/allo/allo_subpolicy_tuned_rep.yaml"
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
_EIG_IDX = flags.DEFINE_integer(
    "eig_idx",
    default=0,
    lower_bound=0,
    help="Eigenvalue index of the subpolicy to tune.",
)
_REVERSE_REWARD = flags.DEFINE_boolean(
    "reverse_reward",
    default=False,
    help="Whether to reverse the intrinsic reward direction.",
)
_REP_IDX = flags.DEFINE_integer(
    "rep_idx",
    default=0,
    lower_bound=0,
    help="Extractor replicate index used to compute intrinsic rewards.",
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
    default=8081,
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
    help="Immediately train tuned subpolicy after tuning.",
)


def main(argv: Sequence[str]) -> None:
    """Execute ALLO subpolicy hyperparameter tuning workflow.

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

    eig_idx: int = _EIG_IDX.value
    reverse_reward: bool = _REVERSE_REWARD.value
    rep_idx: int = _REP_IDX.value

    training_config_reader = AlloSB3PipelineConfigReader.from_yaml(
        str(env_info.base_pipeline)
    )
    training_config_reader.eig_idx = eig_idx
    training_config_reader.reverse_reward = reverse_reward
    training_config_reader.rep_idx = rep_idx
    if _DEVICE.value is not None:
        norm_device = normalize_device(_DEVICE.value)
        training_config_reader.device = norm_device
        training_config_reader.pipeline_config.device = norm_device

    pipeline_config = training_config_reader.to_config()
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
    print(f"\nGenerated replicate configuration at {replicate_path}")

    if _TRAIN_REPLICATES.value:
        print("\nTraining tuned subpolicy with best hyperparameters...")
        tuned_reader = AlloSB3PipelineConfigReader.from_yaml(
            str(env_info.base_pipeline)
        )
        tuned_reader.eig_idx = eig_idx
        tuned_reader.reverse_reward = reverse_reward
        tuned_reader.rep_idx = rep_idx
        if _DEVICE.value is not None:
            norm_device = normalize_device(_DEVICE.value)
            tuned_reader.device = norm_device
            tuned_reader.pipeline_config.device = norm_device
        tuned_reader.algo_kwargs = dict(study.best_params)
        tuned_pipeline = SB3Pipeline(
            config=tuned_reader.to_config(), verbose=True
        )
        tuned_pipeline.train_on_unsaved_model()
        model = tuned_pipeline.load_model("best")
        tuned_pipeline.record_replay(model, verbose=True)
        tuned_pipeline.evaluate(
            n_eval_episodes=100,
            checkpoint="best",
            eval_file_name="best_model_eval.yaml",
        )


if __name__ == "__main__":
    app.run(main)
