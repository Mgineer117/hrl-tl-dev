"""Script for training a single temporal logic subpolicy with SDSAC on demand."""

from collections.abc import Sequence

from absl import app
from absl import flags
import contgrid  # noqa: F401
from rl_pipeline.sb3 import SB3Pipeline, SB3PipelineConfig

from hrl_tl.config.wrapper import TLSB3PipelineConfigReader

_TL_SPEC = flags.DEFINE_string(
    "tl_spec",
    default="",
    help="Temporal logic specification string to train (e.g. 'Fpsi_ld & G!psi_lv').",
    required=True,
)
_TRAINING_CONFIG_PATH = flags.DEFINE_string(
    "training_config_path",
    default="configs/fr_cont/train/baseline/9.a.a_subpolicy_rep0.yaml",
    help="Path to the subpolicy training configuration YAML file.",
)
_DEVICE = flags.DEFINE_string(
    "device",
    default="cuda:0",
    help="Target computation device (e.g. 'cuda:0', 'cuda:1', 'cpu').",
)
_RETRAIN_MODEL = flags.DEFINE_boolean(
    "retrain_model",
    default=False,
    help="Whether to force retraining if a saved checkpoint already exists.",
)
_RECORD_REPLAY = flags.DEFINE_boolean(
    "record_replay",
    default=False,
    help="Whether to record a replay animation after training.",
)
_N_EVAL_EPISODES = flags.DEFINE_integer(
    "n_eval_episodes",
    default=0,
    help="Number of evaluation episodes after training. If 0, evaluation is skipped.",
    lower_bound=0,
)


def main(argv: Sequence[str]) -> None:
    """Trains a single subpolicy for the specified temporal logic formula.

    Args:
        argv: Command-line arguments.
    """
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    tl_spec: str = _TL_SPEC.value
    training_config_path: str = _TRAINING_CONFIG_PATH.value
    device: str = _DEVICE.value
    retrain_model: bool = _RETRAIN_MODEL.value
    record_replay: bool = _RECORD_REPLAY.value
    n_eval_episodes: int = _N_EVAL_EPISODES.value

    print(f"Training subpolicy for spec: '{tl_spec}' on device '{device}'...")

    training_config_reader = TLSB3PipelineConfigReader.from_yaml(
        training_config_path
    )
    training_config_reader.tl_spec = tl_spec
    training_config_reader.pipeline_config.device = device
    training_config_reader.pipeline_config.retrain_model = retrain_model

    training_config: SB3PipelineConfig = training_config_reader.to_config()
    pipeline: SB3Pipeline = SB3Pipeline(config=training_config, verbose=True)

    pipeline.train_on_unsaved_model()

    if record_replay or n_eval_episodes > 0:
        model = pipeline.load_model("best", device=device)
        if record_replay:
            pipeline.record_replay(model, verbose=True)
        if n_eval_episodes > 0:
            pipeline.evaluate(
                n_eval_episodes=n_eval_episodes,
                checkpoint="best",
                eval_file_name="best_model_eval.yaml",
            )

    print(f"Subpolicy training finished for spec: '{tl_spec}'.")


if __name__ == "__main__":
    app.run(main)
