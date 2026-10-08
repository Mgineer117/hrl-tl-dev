"""Script for pretraining all subpolicies for reach-avoid TL specifications."""

from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

from absl import app
from absl import flags
import contgrid  # noqa: F401
from rl_pipeline.sb3 import SB3Pipeline, SB3PipelineConfig
import tqdm

from hrl_tl.config.wrapper import TLSB3PipelineConfigReader

_REP_IDX = flags.DEFINE_integer(
    "rep_idx",
    default=0,
    help="Replicate index for subpolicy pretraining.",
    lower_bound=0,
)
_TRAINING_CONFIG_PATH = flags.DEFINE_string(
    "training_config_path",
    default="",
    help="Path to the training config YAML file. If empty, defaults to configs/fr_cont/train/baseline/9.a.a_subpolicy_rep{rep_idx}.yaml.",
)
_SPEC_FILE_PATH = flags.DEFINE_string(
    "spec_file_path",
    default="assets/formulae/fourroom/all_formulae_1_cla_1_max_pred_hard.json",
    help="Path to the JSON file containing all reach-avoid TL specifications.",
)
_PROCESS_ID = flags.DEFINE_integer(
    "process_id",
    default=0,
    help="Process ID for sharding specifications across parallel jobs.",
    lower_bound=0,
)
_TOTAL_PROCESSES = flags.DEFINE_integer(
    "total_processes",
    default=4,
    help="Total number of parallel processes for sharding specifications.",
    lower_bound=1,
)
_GPU_ID = flags.DEFINE_integer(
    "gpu_id",
    default=0,
    help="GPU ID to use for training (cuda:{gpu_id}).",
    lower_bound=0,
)
_RETRAIN_MODEL = flags.DEFINE_boolean(
    "retrain_model",
    default=False,
    help="Whether to retrain models that already have saved checkpoints.",
)
_EVAL_ONLY = flags.DEFINE_boolean(
    "eval_only",
    default=False,
    help="Whether to only evaluate existing models without training.",
)
_N_EVAL_EPISODES = flags.DEFINE_integer(
    "n_eval_episodes",
    default=100,
    help="Number of evaluation episodes per subpolicy.",
    lower_bound=1,
)


def get_sharded_specs(
    specifications: Sequence[str], process_id: int, total_processes: int
) -> list[str]:
    """Distributes specifications evenly across processes.

    Args:
        specifications: Full sequence of TL specification strings.
        process_id: Rank index of the current process.
        total_processes: Total count of parallel processes.

    Returns:
        The slice of specifications assigned to this process.
    """
    total_specs = len(specifications)
    base_specs_per_process = total_specs // total_processes
    extra_specs = total_specs % total_processes

    if process_id < extra_specs:
        start_index = process_id * (base_specs_per_process + 1)
        end_index = start_index + base_specs_per_process + 1
    else:
        start_index = (
            extra_specs * (base_specs_per_process + 1)
            + (process_id - extra_specs) * base_specs_per_process
        )
        end_index = start_index + base_specs_per_process

    return list(specifications[start_index:end_index])


def main(argv: Sequence[str]) -> None:
    """Pretrains subpolicies for the assigned temporal logic specifications.

    Args:
        argv: Command-line arguments.
    """
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    rep_idx: int = _REP_IDX.value
    training_config_path: str = (
        _TRAINING_CONFIG_PATH.value
        if _TRAINING_CONFIG_PATH.value
        else f"configs/fr_cont/train/baseline/9.a.a_subpolicy_rep{rep_idx}.yaml"
    )
    spec_file_path: Path = Path(_SPEC_FILE_PATH.value)
    process_id: int = _PROCESS_ID.value
    total_processes: int = _TOTAL_PROCESSES.value
    gpu_id: int = _GPU_ID.value
    retrain_model: bool = _RETRAIN_MODEL.value
    eval_only: bool = _EVAL_ONLY.value
    n_eval_episodes: int = _N_EVAL_EPISODES.value

    print(
        f"Process ID: {process_id}/{total_processes}, GPU ID: {gpu_id}, Rep Index: {rep_idx}"
    )

    with spec_file_path.open("r", encoding="utf-8") as f:
        spec_data: dict[str, Any] = json.load(f)

    all_specs: list[str] = spec_data["specifications"]
    tl_specs: list[str] = get_sharded_specs(
        all_specs, process_id=process_id, total_processes=total_processes
    )
    print(
        f"Assigned {len(tl_specs)} of {len(all_specs)} specifications to process {process_id}."
    )

    for tl_spec in tqdm.tqdm(
        tl_specs, desc=f"Process {process_id} - Pretraining subpolicies"
    ):
        training_config_reader = TLSB3PipelineConfigReader.from_yaml(
            training_config_path
        )
        training_config_reader.tl_spec = tl_spec
        training_config_reader.pipeline_config.device = f"cuda:{gpu_id}"
        training_config_reader.pipeline_config.retrain_model = retrain_model
        training_config: SB3PipelineConfig = training_config_reader.to_config()

        pipeline: SB3Pipeline = SB3Pipeline(
            config=training_config, verbose=True
        )

        if not eval_only:
            pipeline.train_on_unsaved_model()

        model = pipeline.load_model("best")
        pipeline.record_replay(model, verbose=True)

        pipeline.evaluate(
            n_eval_episodes=n_eval_episodes,
            checkpoint="best",
            eval_file_name="best_model_eval.yaml",
        )


if __name__ == "__main__":
    app.run(main)
