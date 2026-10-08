"""Training script for TL HRL high-level policy on Zone environment."""

from __future__ import annotations

from collections.abc import Sequence

import contgrid  # noqa: F401
from absl import app, flags
from rl_pipeline.sb3 import (
    SB3ReplicatePipeline,
    SB3ReplicatePipelineConfig,
    SB3ReplicatePipelineConfigReader,
)

from hrl_tl.config.wrapper import TLHighLevelPipelineConfigReader
from hrl_tl.tuning import normalize_device

_EXP_CONFIG_PATH = flags.DEFINE_string(
    "exp_config_path",
    default="configs/zone/train/8.b/8.b.k_hrl_train_rep0.yaml",
    help="Path to the experiment configuration YAML file.",
)
_RETRAIN_MODEL = flags.DEFINE_boolean(
    "retrain_model",
    default=True,
    help="Whether to retrain an existing model.",
)
_RECORD_REPLAYS = flags.DEFINE_boolean(
    "record_replays",
    default=True,
    help="Whether to record replays of the best model.",
)
_DEVICE = flags.DEFINE_string(
    "device",
    default=None,
    help="Override GPU device (e.g. '0', '1', 'cuda:0', 'cpu').",
)


def main(argv: Sequence[str]) -> None:
    """Executes the replicate training pipeline.

    Args:
        argv: Command-line arguments.

    Raises:
        app.UsageError: If unexpected positional arguments are supplied.
    """
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    exp_config: SB3ReplicatePipelineConfig = (
        SB3ReplicatePipelineConfigReader[TLHighLevelPipelineConfigReader]
        .from_yaml(_EXP_CONFIG_PATH.value)
        .to_config()
    )
    exp_config.ind_pipeline_configs[0].retrain_model = _RETRAIN_MODEL.value

    if _DEVICE.value is not None:
        normalized_device = normalize_device(_DEVICE.value)
        for ind_config in exp_config.ind_pipeline_configs:
            ind_config.device = normalized_device

    pipeline = SB3ReplicatePipeline(config=exp_config, verbose=True)
    pipeline.train_on_unsaved_model()

    models = pipeline.load_models("best")
    if _RECORD_REPLAYS.value:
        pipeline.record_replays(models, verbose=True)


if __name__ == "__main__":
    app.run(main)
