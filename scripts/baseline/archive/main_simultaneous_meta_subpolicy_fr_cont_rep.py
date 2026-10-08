"""Script for simultaneous training of meta-policy (PPO) and subpolicies (SDSAC)."""

from collections.abc import Sequence

from absl import app
from absl import flags
import contgrid  # noqa: F401
from rl_pipeline.sb3 import (
    SB3ReplicatePipeline,
    SB3ReplicatePipelineConfig,
    SB3ReplicatePipelineConfigReader,
)

from hrl_tl.config.wrapper import TLHighLevelPipelineConfigReader

_REP_IDX = flags.DEFINE_integer(
    "rep_idx",
    default=0,
    help="Replicate index for simultaneous meta-policy training.",
    lower_bound=0,
)
_EXP_CONFIG_PATH = flags.DEFINE_string(
    "exp_config_path",
    default="",
    help="Path to the meta-policy experiment config YAML file. If empty, defaults to configs/fr_cont/train/baseline/9.c.a_meta_rep{rep_idx}.yaml.",
)
_GPU_IDS = flags.DEFINE_string(
    "gpu_ids",
    default="",
    help="Comma-separated list of GPU IDs to distribute training and caching (e.g. '0,1,2' or '0'). If empty, uses config default.",
)
_GPU_ID = flags.DEFINE_integer(
    "gpu_id",
    default=-1,
    help="Single GPU ID override (if >= 0). Defaults to -1.",
)
_RETRAIN_MODEL = flags.DEFINE_boolean(
    "retrain_model",
    default=False,
    help="Whether to retrain existing meta-policy models.",
)
_RECORD_REPLAYS = flags.DEFINE_boolean(
    "record_replays",
    default=True,
    help="Whether to record animations/replays of best models after training.",
)
_EVAL_EPISODES = flags.DEFINE_integer(
    "eval_episodes",
    default=100,
    help="Number of evaluation episodes after training. If 0, skips evaluation.",
    lower_bound=0,
)


def main(argv: Sequence[str]) -> None:
    """Trains a meta-policy while simultaneously optimizing selected subpolicies.

    Args:
        argv: Command-line arguments.
    """
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    rep_idx: int = _REP_IDX.value
    exp_config_path: str = (
        _EXP_CONFIG_PATH.value
        if _EXP_CONFIG_PATH.value
        else f"configs/fr_cont/train/baseline/9.c.a_meta_rep{rep_idx}.yaml"
    )
    gpu_ids_str: str = _GPU_IDS.value
    gpu_id: int = _GPU_ID.value
    retrain_model: bool = _RETRAIN_MODEL.value
    record_replays: bool = _RECORD_REPLAYS.value
    eval_episodes: int = _EVAL_EPISODES.value

    available_devices: list[str] = []
    if gpu_ids_str:
        available_devices = [
            f"cuda:{gid.strip()}"
            for gid in gpu_ids_str.split(",")
            if gid.strip()
        ]
    elif gpu_id >= 0:
        available_devices = [f"cuda:{gpu_id}"]

    print(f"Loading simultaneous experiment config from: {exp_config_path}")

    exp_config: SB3ReplicatePipelineConfig = (
        SB3ReplicatePipelineConfigReader[TLHighLevelPipelineConfigReader]
        .from_yaml(exp_config_path)
        .to_config()
    )

    for ind_cfg in exp_config.ind_pipeline_configs:
        ind_cfg.retrain_model = retrain_model
        if available_devices:
            ind_cfg.device = available_devices[0]
            if (
                ind_cfg.wrapper_config is not None
                and "low_level_policy_args"
                in ind_cfg.wrapper_config.wrapper_kwargs
            ):
                ll_args = ind_cfg.wrapper_config.wrapper_kwargs[
                    "low_level_policy_args"
                ]
                ll_args["available_devices"] = available_devices
                ll_args["device"] = available_devices[0]

    pipeline = SB3ReplicatePipeline(config=exp_config, verbose=True)
    pipeline.train_on_unsaved_model()

    models = pipeline.load_models("best")

    if record_replays:
        pipeline.record_replays(models, verbose=True)

    if eval_episodes > 0:
        pipeline.evaluate(n_eval_episodes=eval_episodes, checkpoint="best")


if __name__ == "__main__":
    app.run(main)
