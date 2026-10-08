"""Plot reward curves and compute 95% bootstrap CIs for Fetch ablation studies."""

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
    default="out/plots/fetch/ablation/fetch_reward_conv_ablation_indiv_param.png",
    help="Path to save the generated ablation plot image.",
)
_CSV_PATH = flags.DEFINE_string(
    "csv_path",
    default=None,
    help="Path to save the CI summary CSV. Defaults to plot_path with .csv suffix.",
)
_METHODS = flags.DEFINE_multi_string(
    "methods",
    default=[],
    help="Subset of ablation variants to plot. Defaults to all configured variants.",
)
_XLIM_LEFT = flags.DEFINE_float(
    "xlim_left",
    default=10.0,
    help="Left limit for x-axis in millions.",
)
_XLIM_RIGHT = flags.DEFINE_float(
    "xlim_right",
    default=12.0,
    help="Right limit for x-axis in millions.",
)
_HEIGHT = flags.DEFINE_float(
    "height",
    default=3.0,
    help="Height of the plot in inches.",
    lower_bound=0.1,
)
_ASPECT = flags.DEFINE_float(
    "aspect",
    default=0.8,
    help="Aspect ratio of each facet (width = height * aspect).",
    lower_bound=0.1,
)

_ABLATION_CONFIGS: tuple[EvalDataConfig, ...] = (
    EvalDataConfig(
        name="DFO (Ours)",
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
        name=r"Fixed $L$ ($L=7.47$)",
        num_replicates=5,
        max_timesteps=2_000_000,
        data_points=400,
        smooth_window=4,
        start_timestep=10_000_000,
        file_reader_class=SB3EvalFileReader,
        eval_file_config=SB3FileReaderConfig(
            models_dir="out/fetch/abl/14.a/meta/14.a.a/hl_policy_fetch_2.0M",
            ind_eval_file="rep_{rep_id}/eval/evaluations.npz",
        ),
    ),
    EvalDataConfig(
        name=r"Fixed $k$ ($k=0.286$)",
        num_replicates=5,
        max_timesteps=2_000_000,
        data_points=400,
        smooth_window=4,
        start_timestep=10_000_000,
        file_reader_class=SB3EvalFileReader,
        eval_file_config=SB3FileReaderConfig(
            models_dir="out/fetch/abl/14.b/meta/14.b.a/hl_policy_fetch_2.0M",
            ind_eval_file="rep_{rep_id}/eval/evaluations.npz",
        ),
    ),
    EvalDataConfig(
        name=r"Fixed $ϵ$ ($ϵ=2.65$)",
        num_replicates=5,
        max_timesteps=2_000_000,
        data_points=400,
        smooth_window=4,
        start_timestep=10_000_000,
        file_reader_class=SB3EvalFileReader,
        eval_file_config=SB3FileReaderConfig(
            models_dir="out/fetch/abl/14.c/meta/14.c.a/hl_policy_fetch_2.0M",
            ind_eval_file="rep_{rep_id}/eval/evaluations.npz",
        ),
    ),
)


def main(argv: Sequence[str]) -> None:
    """Loads evaluation data, generates the ablation plot, and saves the CI summary."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    plot_path = Path(_PLOT_PATH.value)
    csv_path = Path(_CSV_PATH.value) if _CSV_PATH.value is not None else None

    run_curve_pipeline(
        configs=_ABLATION_CONFIGS,
        plot_path=plot_path,
        csv_path=csv_path,
        selected_methods=_METHODS.value,
        plot_config=CurvePlotConfig(
            height=_HEIGHT.value,
            aspect=_ASPECT.value,
            xlim_left=_XLIM_LEFT.value,
            xlim_right=_XLIM_RIGHT.value,
        ),
    )


if __name__ == "__main__":
    app.run(main)
