"""Training script for predicate-conditioned primitive policy on Continuous Four-Room."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import contgrid  # noqa: F401
from absl import app, flags

from hrl_tl.training import run_cpc_primitive_training
from hrl_tl.training.types import ReplayModel

_EXP_CONFIG_PATH = flags.DEFINE_string(
    "exp_config_path",
    default="configs/fr_cont/cpc/pr_soft/3.l.b_train_primitive_rep.yaml",
    help="Path to the replicate experiment configuration YAML file.",
)
_RETRAIN_MODEL = flags.DEFINE_boolean(
    "retrain_model",
    default=False,
    help="Whether to retrain an existing model.",
)
_RECORD_REPLAYS = flags.DEFINE_boolean(
    "record_replays",
    default=True,
    help="Whether to record replays after training.",
)
_REPLAY_MODEL = flags.DEFINE_enum(
    "replay_model",
    default="best",
    enum_values=["best", "final", "latest"],
    help="Which model checkpoint to use for replays ('best', 'final', or 'latest').",
)
_DEVICE = flags.DEFINE_string(
    "device",
    default=None,
    help="Override GPU device (e.g. '0', '1', 'cuda:0', 'cpu').",
)
_NUM_REPLICATES = flags.DEFINE_integer(
    "num_replicates",
    default=None,
    lower_bound=1,
    help="Override number of replicates to train.",
)
_REPLICATE_START_ID = flags.DEFINE_integer(
    "replicate_start_id",
    default=None,
    lower_bound=0,
    help="Override starting replicate ID.",
)


def main(argv: Sequence[str]) -> None:
    """Executes replicate training for the Continuous Four-Room primitive policy.

    Args:
        argv: Command-line arguments.

    Raises:
        app.UsageError: If unexpected positional arguments are supplied.
    """
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    run_cpc_primitive_training(
        config_path=_EXP_CONFIG_PATH.value,
        retrain_model=_RETRAIN_MODEL.value,
        record_replays=_RECORD_REPLAYS.value,
        replay_model=cast(ReplayModel, _REPLAY_MODEL.value),
        device=_DEVICE.value,
        num_replicates=_NUM_REPLICATES.value,
        replicate_start_id=_REPLICATE_START_ID.value,
    )


if __name__ == "__main__":
    app.run(main)
