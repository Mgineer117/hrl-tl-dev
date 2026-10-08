"""Hyperparameter tuning runner for flat PPO baseline in Zone."""

from collections.abc import Sequence
from pathlib import Path

import contgrid  # noqa: F401 - Register gym envs
import yaml
from absl import app, flags
from immutabledict import immutabledict
from rl_pipeline.sb3 import SB3Pipeline, SB3PipelineConfigReader

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
            base_pipeline=Path("configs/fr_cont/train/6.d/6.d.a_ppo.yaml"),
            tuning_config=Path("configs/tuning/ppo_fr_cont.yaml"),
            best_export=Path("configs/tuning/best/ppo_fr_cont_best.yaml"),
            replicate_export=Path(
                "configs/fr_cont/train/6.d/ppo_tuned_rep.yaml"
            ),
        ),
        "zone": TuningEnvPaths(
            base_pipeline=Path("configs/zone/train/6.e/6.e.a_ppo.yaml"),
            tuning_config=Path("configs/tuning/ppo_zone.yaml"),
            best_export=Path("configs/tuning/best/ppo_zone_best.yaml"),
            replicate_export=Path("configs/zone/train/6.e/ppo_tuned_rep.yaml"),
        ),
        "fetch": TuningEnvPaths(
            base_pipeline=Path("configs/fetch/train/6.f/6.f.a_ppo.yaml"),
            tuning_config=Path("configs/tuning/ppo_fetch.yaml"),
            best_export=Path("configs/tuning/best/ppo_fetch_best.yaml"),
            replicate_export=Path("configs/fetch/train/6.f/ppo_tuned_rep.yaml"),
        ),
    }
)

_ENV = flags.DEFINE_enum(
    "env",
    default="fr_cont",
    enum_values=["zone", "fr_cont", "fetch"],
    help="Target environment name ('zone' or 'fr_cont').",
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
    """Execute flat PPO hyperparameter tuning workflow.

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
    single_cfg = rep_template.get("single_pipeline_config", rep_template)

    single_reader = SB3PipelineConfigReader(**single_cfg)
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
            SB3PipelineConfigReader
        ].from_yaml(str(replicate_path))
        rep_pipeline = SB3ReplicatePipeline(config=rep_reader.to_config())
        rep_pipeline.train_on_unsaved_model()
        models = rep_pipeline.load_models("best")
        rep_pipeline.record_replays(models)


if __name__ == "__main__":
    app.run(main)
