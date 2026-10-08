"""Training runner for ALLO intrinsic reward subpolicies."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import cast

os.environ.setdefault("MUJOCO_GL", "egl")

import contgrid  # noqa: F401 - Registers gym environments.
import tqdm
import yaml
from absl import app, flags, logging
from rl_pipeline.sb3 import (
    SB3Pipeline,
    SB3PipelineConfig,
)

import hrl_tl.envs.fetch  # noqa: F401
from hrl_tl.config.wrapper import AlloSB3PipelineConfigReader
from hrl_tl.training.types import ReplayModel
from hrl_tl.tuning import normalize_device

_EXP_CONFIG_PATH = flags.DEFINE_string(
    "exp_config_path",
    default="configs/zone/train/allo/7.h.b_allo_subpolicy_rep0.yaml",
    help="Path to the subpolicy experiment configuration YAML file.",
)
_TASK_ID = flags.DEFINE_integer(
    "task_id",
    default=None,
    lower_bound=0,
    help=(
        "Specific subpolicy task index to train (0 to 2*num_eigs - 1). If"
        " None, checks SLURM_ARRAY_TASK_ID. If still None, trains all"
        " subpolicies sequentially."
    ),
)
_NUM_EIGS = flags.DEFINE_integer(
    "num_eigs",
    default=None,
    lower_bound=1,
    help=(
        "Number of eigenvalues. If None, automatically detected from the"
        " extractor model configuration."
    ),
)
_REP_IDX = flags.DEFINE_integer(
    "rep_idx",
    default=None,
    lower_bound=0,
    help=(
        "Extractor replicate index used for intrinsic reward. If None, inferred"
        " from the config file name or defaults to 0."
    ),
)
_DEVICE = flags.DEFINE_string(
    "device",
    default="0",
    help="Override GPU device (e.g. '0', '1', 'cuda:0', 'cpu').",
)
_RETRAIN_MODEL = flags.DEFINE_boolean(
    "retrain_model",
    default=False,
    help="Whether to retrain an existing model.",
)
_EVAL_ONLY = flags.DEFINE_boolean(
    "eval_only",
    default=False,
    help="Skip training and only evaluate saved checkpoint.",
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
_N_EVAL_EPISODES = flags.DEFINE_integer(
    "n_eval_episodes",
    default=100,
    lower_bound=1,
    help="Number of evaluation episodes.",
)


def resolve_num_eigs(config_path: Path) -> int:
    """Discovers the number of eigenvalues from the extractor model config.

    Args:
        config_path: Path to the subpolicy pipeline YAML file.

    Returns:
        Number of eigenvalues (representation dimension) of the extractor.
    """
    try:
        pipe_cfg = AlloSB3PipelineConfigReader.from_yaml(
            str(config_path)
        ).to_config()
        wrapper_cfg = pipe_cfg.wrapper_config

        if wrapper_cfg is None:
            raise ValueError(f"Wrapper config not found in {config_path}")

        extractor_file: str = wrapper_cfg.wrapper_kwargs[
            "intrinsic_reward_args"
        ]["pipeline_config_path"]

        extractor_path = Path(extractor_file)
        with extractor_path.open("r", encoding="utf-8") as f:
            extractor_cfg = yaml.safe_load(f)

        single_pipe = extractor_cfg["single_pipeline_config"]
        ext_config_dir = Path(single_pipe["config_dir"])
        model_file = single_pipe["model_config_file"]
        if not model_file:
            raise ValueError("Model config file not found")

        model_path = ext_config_dir / model_file
        with model_path.open("r", encoding="utf-8") as f:
            model_cfg = yaml.safe_load(f)

        rep_dim = model_cfg["algo_config"]["algo_kwargs"]["representation_dim"]
        return int(rep_dim)

    except (OSError, yaml.YAMLError, KeyError) as e:
        logging.error(
            "Failed to resolve representation dimension automatically: %s.", e
        )
        raise


def resolve_rep_idx(config_path: Path) -> int:
    """Extracts replicate index from config file name or defaults to 0.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Integer replicate index.
    """
    match = re.search(r"rep(\d+)", config_path.stem)
    if match is not None:
        return int(match.group(1))
    raise ValueError(f"Failed to resolve replicate index from {config_path}")


def train_subpolicy(
    training_config_path: str,
    eig_idx: int,
    reverse_reward: bool,
    rep_idx: int,
    device: str,
    retrain_model: bool,
    eval_only: bool,
    record_replays: bool,
    replay_model: ReplayModel,
    n_eval_episodes: int,
) -> None:
    """Trains and evaluates a single ALLO subpolicy.

    Args:
        training_config_path: Path to the pipeline config YAML.
        eig_idx: Eigenvalue index.
        reverse_reward: Whether reward direction is inverted.
        rep_idx: Replicate index.
        device: Computing device for training and model evaluation.
        retrain_model: Whether to retrain an existing model.
        eval_only: Skip training and only evaluate.
        record_replays: Whether to save replay animation after training.
        replay_model: Which model checkpoint to use for replays ('best',
            'final', or 'latest').
        n_eval_episodes: Number of evaluation episodes.
    """
    logging.info(
        "Subpolicy eig_idx=%d, reverse_reward=%s, rep_idx=%d on device=%s",
        eig_idx,
        reverse_reward,
        rep_idx,
        device,
    )
    training_config_reader = AlloSB3PipelineConfigReader.from_yaml(
        training_config_path
    )
    training_config_reader.eig_idx = eig_idx
    training_config_reader.reverse_reward = reverse_reward
    training_config_reader.rep_idx = rep_idx
    training_config_reader.device = device
    training_config_reader.pipeline_config.device = device
    training_config_reader.pipeline_config.retrain_model = retrain_model
    training_config: SB3PipelineConfig = training_config_reader.to_config()

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
    """Executes the subpolicy training workflow.

    Args:
        argv: Command-line arguments.

    Raises:
        app.UsageError: If unexpected positional arguments are supplied.
    """
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    config_path = Path(_EXP_CONFIG_PATH.value)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    num_eigs: int = (
        _NUM_EIGS.value
        if _NUM_EIGS.value is not None
        else resolve_num_eigs(config_path)
    )
    rep_idx: int = (
        _REP_IDX.value
        if _REP_IDX.value is not None
        else resolve_rep_idx(config_path)
    )

    device: str = (
        normalize_device(_DEVICE.value)
        if _DEVICE.value is not None
        else "cuda:0"
    )

    all_eig_pairs: list[tuple[int, bool]] = [
        (eig_idx, reverse)
        for eig_idx in range(num_eigs)
        for reverse in (False, True)
    ]

    # Resolve task index from flag or Slurm environment variable.
    task_id: int | None = _TASK_ID.value
    if task_id is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    if task_id is not None:
        if task_id >= len(all_eig_pairs):
            logging.info(
                "Task ID %d exceeds total subpolicies (%d). Exiting.",
                task_id,
                len(all_eig_pairs),
            )
            return
        target_pairs = [all_eig_pairs[task_id]]
    else:
        target_pairs = all_eig_pairs

    logging.info(
        "Training %d subpolicy(ies) (num_eigs=%d, total=%d)",
        len(target_pairs),
        num_eigs,
        len(all_eig_pairs),
    )

    for eig_idx, reverse_reward in tqdm.tqdm(
        target_pairs, desc="Processing subpolicies"
    ):
        train_subpolicy(
            training_config_path=str(config_path),
            eig_idx=eig_idx,
            reverse_reward=reverse_reward,
            rep_idx=rep_idx,
            device=device,
            retrain_model=_RETRAIN_MODEL.value,
            eval_only=_EVAL_ONLY.value,
            record_replays=_RECORD_REPLAYS.value,
            replay_model=cast(ReplayModel, _REPLAY_MODEL.value),
            n_eval_episodes=_N_EVAL_EPISODES.value,
        )


if __name__ == "__main__":
    app.run(main)
