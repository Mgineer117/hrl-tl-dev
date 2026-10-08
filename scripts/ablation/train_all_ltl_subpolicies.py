"""Training runner for reach-avoid LTL subpolicies with SD-SAC on Fetch."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import contgrid  # noqa: F401 - Registers gym environments.
import tqdm
from absl import app, flags, logging
from rl_pipeline.sb3 import SB3Pipeline, SB3PipelineConfig

import hrl_tl.envs.fetch  # noqa: F401 - Registers fetch robotics environments.
from hrl_tl.config.wrapper import TLSB3PipelineConfigReader
from hrl_tl.training.types import ReplayModel
from hrl_tl.tuning import normalize_device

os.environ.setdefault("MUJOCO_GL", "egl")

_EXP_CONFIG_PATH = flags.DEFINE_string(
    "exp_config_path",
    default="configs/fetch/train/ablation/12.a.b_subpolicy_rep{rep_idx}.yaml",
    help="Path template to the subpolicy pipeline YAML config with {rep_idx}.",
)
_SPEC_FILE_PATH = flags.DEFINE_string(
    "spec_file_path",
    default="assets/formulae/fetch/cpc/all_formulae_1_cla_1_max_pred.json",
    help="Path to the JSON file containing candidate LTL specifications.",
)
_TASK_ID = flags.DEFINE_integer(
    "task_id",
    default=None,
    lower_bound=0,
    help=(
        "Specific subpolicy task index to train (0 to num_specs - 1). If None,"
        " checks SLURM_ARRAY_TASK_ID. If still None, trains all specifications."
    ),
)
_TL_SPEC = flags.DEFINE_string(
    "tl_spec",
    default=None,
    help="Explicit temporal logic specification string override.",
)
_REP_IDX = flags.DEFINE_integer(
    "rep_idx",
    default=0,
    lower_bound=0,
    help="Replicate index for subpolicy pretraining (0 to 4).",
)
_DEVICE = flags.DEFINE_string(
    "device",
    default="0",
    help="Computing device (e.g. '0', '1', 'cuda:0', 'cpu').",
)
_RETRAIN_MODEL = flags.DEFINE_boolean(
    "retrain_model",
    default=False,
    help="Whether to retrain an existing model checkpoint.",
)
_EVAL_ONLY = flags.DEFINE_boolean(
    "eval_only",
    default=False,
    help="Skip training and only evaluate saved checkpoints.",
)
_RECORD_REPLAYS = flags.DEFINE_boolean(
    "record_replays",
    default=True,
    help="Whether to save replay animations after training.",
)
_REPLAY_MODEL = flags.DEFINE_enum(
    "replay_model",
    default="best",
    enum_values=["best", "final", "latest"],
    help="Which model checkpoint to use for replays ('best', 'final', or 'latest').",
)
_N_EVAL_EPISODES = flags.DEFINE_integer(
    "n_eval_episodes",
    default=100,
    lower_bound=1,
    help="Number of evaluation episodes per subpolicy.",
)


def load_specifications(spec_file_path: Path) -> list[str]:
    """Loads candidate LTL specifications from a JSON file.

    Args:
        spec_file_path: Path to the JSON specification dictionary file.

    Returns:
        List of specification strings.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If the 'specifications' key is missing.
    """
    try:
        with spec_file_path.open("r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        return list(data["specifications"])
    except FileNotFoundError:
        logging.exception("Specification file not found: %s", spec_file_path)
        raise


def train_single_specification(
    config_path: Path,
    tl_spec: str,
    device: str,
    retrain_model: bool,
    eval_only: bool,
    record_replays: bool,
    replay_model: ReplayModel,
    n_eval_episodes: int,
) -> None:
    """Trains and evaluates an SD-SAC subpolicy for a single LTL formula.

    Args:
        config_path: Path to the resolved pipeline config YAML.
        tl_spec: Temporal logic formula to optimize.
        device: Normalised compute device identifier.
        retrain_model: Whether to retrain an existing checkpoint.
        eval_only: Whether to skip training and only evaluate.
        record_replays: Whether to record replay animation.
        replay_model: Checkpoint selection for replays.
        n_eval_episodes: Evaluation episode count.
    """
    logging.info("Training subpolicy for spec='%s' on device=%s", tl_spec, device)
    reader = TLSB3PipelineConfigReader.from_yaml(str(config_path))
    reader.tl_spec = tl_spec
    reader.pipeline_config.device = device
    reader.pipeline_config.retrain_model = retrain_model
    training_config: SB3PipelineConfig = reader.to_config()

    pipeline = SB3Pipeline(config=training_config, verbose=True)

    if not eval_only:
        pipeline.train_on_unsaved_model()

    model = pipeline.load_model(replay_model)
    if record_replays:
        pipeline.record_replay(model, verbose=True)

    pipeline.evaluate(
        n_eval_episodes=n_eval_episodes,
        checkpoint="best",
        eval_file_name="best_model_eval.yaml",
    )


def main(argv: Sequence[str]) -> None:
    """Executes the subpolicy training workflow for Fetch Reach-Avoid.

    Args:
        argv: Command-line arguments.

    Raises:
        app.UsageError: If unexpected positional arguments are supplied.
    """
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    rep_idx: int = _REP_IDX.value
    raw_config_path = _EXP_CONFIG_PATH.value.format(rep_idx=rep_idx)
    config_path = Path(raw_config_path)

    spec_file_path = Path(_SPEC_FILE_PATH.value)
    all_specs = load_specifications(spec_file_path)

    # Determine target specifications to train.
    if _TL_SPEC.value is not None:
        target_specs = [_TL_SPEC.value]
    else:
        task_id: int | None = _TASK_ID.value
        if task_id is None and "SLURM_ARRAY_TASK_ID" in os.environ:
            task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

        if task_id is not None:
            if task_id >= len(all_specs):
                logging.info(
                    "Task ID %d exceeds total specs count (%d). Exiting.",
                    task_id,
                    len(all_specs),
                )
                return
            target_specs = [all_specs[task_id]]
        else:
            target_specs = all_specs

    device: str = normalize_device(_DEVICE.value)
    logging.info(
        "Processing %d subpolicies (rep_idx=%d) on %s",
        len(target_specs),
        rep_idx,
        device,
    )

    for tl_spec in tqdm.tqdm(target_specs, desc="Training subpolicies"):
        train_single_specification(
            config_path=config_path,
            tl_spec=tl_spec,
            device=device,
            retrain_model=_RETRAIN_MODEL.value,
            eval_only=_EVAL_ONLY.value,
            record_replays=_RECORD_REPLAYS.value,
            replay_model=cast(ReplayModel, _REPLAY_MODEL.value),
            n_eval_episodes=_N_EVAL_EPISODES.value,
        )


if __name__ == "__main__":
    app.run(main)
