"""Plot reward curves with 95% confidence intervals for Fetch benchmarks."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from absl import app, flags

from hrl_tl.eval.plot import CurvePlotConfig, run_curve_pipeline
from hrl_tl.eval.reward import (
    EvalDataConfig,
    SB3EvalFileReader,
    SB3FileReaderConfig,
)

_PLOT_PATH = flags.DEFINE_string(
    "plot_path",
    default="out/plots/fetch/baseline/fetch_reward_conv_baselines.png",
    help="Path to save the generated plot image.",
)
_CSV_PATH = flags.DEFINE_string(
    "csv_path",
    default=None,
    help="Path to save the CI summary CSV. Defaults to plot_path with .csv suffix.",
)
_METHODS = flags.DEFINE_multi_string(
    "methods",
    default=[],
    help="Subset of methods to plot. Defaults to all configured methods if omitted.",
)
_XLIM_LEFT = flags.DEFINE_float(
    "xlim_left",
    default=9.0,
    help="Left limit for x-axis in millions.",
)
_XLIM_RIGHT = flags.DEFINE_float(
    "xlim_right",
    default=12.0,
    help="Right limit for x-axis in millions.",
)

# Default evaluation data configurations for archival reproducibility.
_DEFAULT_CONFIGS: tuple[EvalDataConfig, ...] = (
    EvalDataConfig(
        name="Ours",
        num_replicates=5,
        max_timesteps=2_000_000,
        data_points=400,
        smooth_window=4,
        start_timestep=10_000_000,
        file_reader_class=SB3EvalFileReader,
        eval_file_config=SB3FileReaderConfig(
            models_dir="out/fetch/ltl_hl/11.b.b/hl_policy_fetch_2.0M",
            ind_eval_file="rep_{rep_id}/eval/evaluations.npz",
        ),
    ),
    EvalDataConfig(
        name="GCLTL-RL",
        num_replicates=5,
        max_timesteps=2_000_000,
        data_points=400,
        smooth_window=4,
        start_timestep=10_000_000,
        file_reader_class=SB3EvalFileReader,
        eval_file_config=SB3FileReaderConfig(
            models_dir="out/fetch/ltl_hl/5.l.b/hl_policy_fetch_2.0M",
            ind_eval_file="rep_{rep_id}/eval/evaluations.npz",
        ),
    ),
    EvalDataConfig(
        name="ALLO",
        num_replicates=5,
        max_timesteps=3_000_000,
        data_points=400,
        smooth_window=4,
        start_timestep=5_000_000 + 8 * 500_000,
        file_reader_class=SB3EvalFileReader,
        eval_file_config=SB3FileReaderConfig(
            models_dir="out/fetch/allo/meta/7.l.b/allo_meta_fetch_3.0M",
            ind_eval_file="rep_{rep_id}/eval/evaluations.npz",
        ),
    ),
    EvalDataConfig(
        name="HIRO",
        num_replicates=5,
        max_timesteps=40_000_000,
        data_points=200,
        smooth_window=4,
        file_reader_class=SB3EvalFileReader,
        eval_file_config=SB3FileReaderConfig(
            models_dir="out/fetch/hiro/4.g.b/hiro_fetch_12.0M",
            ind_eval_file="rep_{rep_id}/eval/evaluations.npz",
        ),
    ),
    EvalDataConfig(
        name="PPO",
        num_replicates=5,
        max_timesteps=12_000_000,
        data_points=200,
        smooth_window=2,
        file_reader_class=SB3EvalFileReader,
        eval_file_config=SB3FileReaderConfig(
            models_dir="out/fetch/ppo/6.f.b/ppo_fetch_12.0M",
            ind_eval_file="rep_{rep_id}/eval/evaluations.npz",
        ),
    ),
)


def main(argv: Sequence[str]) -> None:
    """Loads evaluation data, generates the reward plot, and prints statistics."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    plot_path = Path(_PLOT_PATH.value)
    csv_path = Path(_CSV_PATH.value) if _CSV_PATH.value is not None else None

    run_curve_pipeline(
        configs=_DEFAULT_CONFIGS,
        plot_path=plot_path,
        csv_path=csv_path,
        selected_methods=_METHODS.value,
        plot_config=CurvePlotConfig(
            xlim_left=_XLIM_LEFT.value,
            xlim_right=_XLIM_RIGHT.value,
        ),
    )


if __name__ == "__main__":
    app.run(main)
