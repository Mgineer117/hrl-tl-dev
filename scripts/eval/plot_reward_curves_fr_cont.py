"""Plot reward curves with 95% confidence intervals for Four Rooms Continuous benchmarks."""

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
    default="out/plots/fr_cont/baseline/fr_cont_reward_conv_baselines.png",
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
    default=5,
    help="Left limit for x-axis in millions.",
)
_XLIM_RIGHT = flags.DEFINE_float(
    "xlim_right",
    default=20,
    help="Right limit for x-axis in millions.",
)

# Default evaluation data configurations for archival reproducibility.
_DEFAULT_CONFIGS: tuple[EvalDataConfig, ...] = (
    EvalDataConfig(
        name="Ours",
        num_replicates=5,
        max_timesteps=10_000_000,
        data_points=400,
        smooth_window=4,
        start_timestep=10_000_000,
        file_reader_class=SB3EvalFileReader,
        eval_file_config=SB3FileReaderConfig(
            models_dir="out/fr_cont/ltl_hl/3.j.e/hl_policy_fr_cont_10.0M",
            ind_eval_file="rep_{rep_id}/eval/evaluations.npz",
        ),
    ),
    # EvalDataConfig(
    #     name="TL-HRL (PPO)",
    #     num_replicates=4,
    #     max_timesteps=10_000_000,
    #     data_points=400,
    #     smooth_window=4,
    #     start_timestep=10_000_000,
    #     file_reader_class=SB3EvalFileReader,
    #     eval_file_config=SB3FileReaderConfig(
    #         models_dir="out/fr_cont/ltl_hl/3.l.a/hl_policy_fr_cont_20.0M",
    #         ind_eval_file="rep_{rep_id}/eval/evaluations.npz",
    #     ),
    # ),
    EvalDataConfig(
        name="GCLTL-RL",
        num_replicates=5,
        max_timesteps=10_000_000,
        data_points=400,
        smooth_window=4,
        start_timestep=10_000_000,
        file_reader_class=SB3EvalFileReader,
        eval_file_config=SB3FileReaderConfig(
            models_dir="out/fr_cont/ltl_hl/5.b.b/hl_policy_fr_cont_10.0M",
            ind_eval_file="rep_{rep_id}/eval/evaluations.npz",
        ),
    ),
    EvalDataConfig(
        name="ALLO",
        num_replicates=5,
        max_timesteps=5_800_000,
        data_points=400,
        smooth_window=4,
        start_timestep=10_000_000 + 42 * 100_000,
        file_reader_class=SB3EvalFileReader,
        eval_file_config=SB3FileReaderConfig(
            models_dir="out/fr_cont/allo/meta/7.c.b/allo_meta_fr_cont_6.0M",
            ind_eval_file="rep_{rep_id}/eval/evaluations.npz",
        ),
    ),
    EvalDataConfig(
        name="HIRO",
        num_replicates=5,
        max_timesteps=1_500_000,
        data_points=200,
        smooth_window=2,
        file_reader_class=SB3EvalFileReader,
        eval_file_config=SB3FileReaderConfig(
            models_dir="out/fr_cont/hiro/4.b.a/hiro_fr_cont_15.0M",
            ind_eval_file="rep_{rep_id}/eval/evaluations.npz",
        ),
    ),
    EvalDataConfig(
        name="PPO",
        num_replicates=5,
        max_timesteps=20_000_000,
        data_points=200,
        smooth_window=2,
        file_reader_class=SB3EvalFileReader,
        eval_file_config=SB3FileReaderConfig(
            models_dir="out/fr_cont/ppo/6.a.a/ppo_fr_cont_20.0M",
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
