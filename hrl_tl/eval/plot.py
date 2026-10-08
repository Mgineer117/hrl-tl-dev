"""Plotting and evaluation statistics utilities for reward curves."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from pydantic import BaseModel
from scipy import stats

if TYPE_CHECKING:
    from hrl_tl.eval.reward import EvalDataConfig

_LIBERTINE_CANDIDATE_FONTS: tuple[str, ...] = (
    "Linux Libertine O",
    "Linux Libertine",
    "Libertine",
)
_SERIF_FALLBACK_FONTS: tuple[str, ...] = (
    "Linux Libertine O",
    "Linux Libertine",
    "DejaVu Serif",
)


class RewardSummary(BaseModel):
    """Summary statistics for reward evaluations including confidence intervals."""

    mean: float
    ci_lower: float
    ci_upper: float


class CurvePlotConfig(BaseModel):
    """Configuration options for evaluation curve plotting and styling."""

    x_col: str = "timestep"
    y_col: str = "reward"
    hue_col: str = "method"
    x_label: str = "Training Timesteps"
    y_label: str = "Mean Return"
    title: str | None = None
    label_fontsize: int = 14
    height: float = 3.0
    aspect: float = 1.2
    linewidth: float = 1.0
    confidence_level: int = 95
    font_family: str = "Linux Libertine O"
    dpi: int = 300
    xlim_left: float | None = None
    xlim_right: float | None = None
    ylim_bottom: float | None = None
    ylim_top: float | None = None


class SuccessPlotConfig(CurvePlotConfig):
    """Configuration options for success rate curve plotting and styling."""

    y_col: str = "success_rate"
    y_label: str = "Evaluation Success Rate"
    ylim_bottom: float | None = 0.0
    ylim_top: float | None = 1.05


def resolve_font_family(
    preferred_font: str = "Linux Libertine O",
    candidates: Sequence[str] = _LIBERTINE_CANDIDATE_FONTS,
) -> str:
    """Finds the best available font family name registered in matplotlib.

    Args:
        preferred_font: Preferred font family name.
        candidates: Sequence of font names to check if preferred is unavailable.

    Returns:
        The matched font name if available in matplotlib, otherwise preferred_font.
    """
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    if preferred_font in available_fonts:
        return preferred_font
    for candidate in candidates:
        if candidate in available_fonts:
            return candidate
    return preferred_font


def set_plot_theme(
    font_family: str = "Linux Libertine O",
    serif_fonts: Sequence[str] = _SERIF_FALLBACK_FONTS,
    style: str = "whitegrid",
) -> None:
    """Sets seaborn and matplotlib themes with Libertine serif font family.

    Args:
        font_family: Name of the primary font family.
        serif_fonts: Sequence of fallback serif font families.
        style: Seaborn style theme name.
    """
    resolved_font = resolve_font_family(font_family)
    sns.set_theme(
        style=style,
        font=resolved_font,
        rc={
            "font.family": "serif",
            "font.serif": list(serif_fonts),
            "mathtext.fontset": "custom",
            "mathtext.rm": resolved_font,
            "mathtext.it": f"{resolved_font}:italic",
            "mathtext.bf": f"{resolved_font}:bold",
        },
    )


def compute_bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    confidence_level: float = 0.95,
    n_resamples: int = 10_000,
    random_state: int = 0,
) -> RewardSummary:
    """Computes the mean and percentile bootstrap confidence interval.

    Args:
        values: Sequence or array of metric values across replicates.
        confidence_level: Desired confidence level between 0 and 1.
        n_resamples: Number of bootstrap iterations.
        random_state: Seed for reproducible resampling.

    Returns:
        RewardSummary containing sample mean and bootstrap CI bounds.
    """
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return RewardSummary(
            mean=float("nan"), ci_lower=float("nan"), ci_upper=float("nan")
        )
    if len(arr) == 1:
        val = float(arr[0])
        return RewardSummary(mean=val, ci_lower=val, ci_upper=val)

    res = stats.bootstrap(
        (arr,),
        np.mean,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        method="percentile",
        random_state=random_state,
    )
    return RewardSummary(
        mean=float(np.mean(arr)),
        ci_lower=float(res.confidence_interval.low),
        ci_upper=float(res.confidence_interval.high),
    )


def millions_formatter(x: float, pos: object) -> str:
    """Formats tick values in millions (e.g. 10M, 10.5M).

    Args:
        x: Tick value.
        pos: Tick position.

    Returns:
        Formatted string representing value in millions.
    """
    return f"{x:g}M"


def filter_configs_by_method(
    configs: Sequence[EvalDataConfig],
    selected_methods: Sequence[str] | set[str] | None = None,
) -> list[EvalDataConfig]:
    """Filters evaluation data configurations by method names.

    Args:
        configs: Sequence of evaluation data configurations.
        selected_methods: Optional sequence or set of method names to retain.
            If empty or None, all configurations are retained.

    Returns:
        List of matching evaluation data configurations.
    """
    selected_set = set(selected_methods) if selected_methods else set()
    return [
        config
        for config in configs
        if not selected_set or config.name in selected_set
    ]


def load_eval_data_df(
    configs: Sequence[EvalDataConfig],
    timestep_divisor: float = 1e6,
) -> pd.DataFrame:
    """Loads evaluation data from configurations into a single pandas DataFrame.

    Args:
        configs: Sequence of evaluation configurations to load.
        timestep_divisor: Divisor to scale the timestep values (e.g. 1e6 to
            convert timesteps to millions). If 1.0, timesteps are unchanged.

    Returns:
        DataFrame containing evaluation records with timestep, reward,
        replicate, and method columns.
    """
    plot_data: list[dict[str, object]] = []

    for config in configs:
        print(f"\nProcessing configuration: {config.name}")

        reward_data, timestep_data = config.file_reader_class.read_eval_files(
            config.eval_file_config, config.num_replicates
        )

        reward_data, timestep_data = (
            config.file_reader_class.format_data_length(
                reward_data,
                timestep_data,
                config.max_timesteps,
                config.data_points,
            )
        )

        plot_data.extend(
            config.file_reader_class.get_plot_data(
                reward_data,
                timestep_data,
                config.smooth_window,
                config.name,
                config.start_timestep,
            )
        )

    df = pd.DataFrame(plot_data)
    if not df.empty and timestep_divisor != 1.0:
        df["timestep"] = df["timestep"] / timestep_divisor

    return df


def plot_curves(
    df: pd.DataFrame,
    plot_path: Path,
    config: CurvePlotConfig | None = None,
) -> sns.FacetGrid:
    """Renders and saves evaluation curves with confidence intervals.

    Args:
        df: DataFrame containing evaluation data.
        plot_path: Destination path for the saved plot image.
        config: Plotting and styling configuration options.

    Returns:
        Seaborn FacetGrid object for the generated plot.
    """
    cfg = config if config is not None else CurvePlotConfig()

    set_plot_theme(font_family=cfg.font_family)

    g = sns.relplot(
        data=df,
        x=cfg.x_col,
        y=cfg.y_col,
        hue=cfg.hue_col,
        kind="line",
        estimator="mean",
        linewidth=cfg.linewidth,
        height=cfg.height,
        aspect=cfg.aspect,
        errorbar=("ci", cfg.confidence_level),
    )

    g.set_axis_labels(cfg.x_label, cfg.y_label, fontsize=cfg.label_fontsize)

    if cfg.title is not None:
        g.ax.set_title(cfg.title, fontsize=cfg.label_fontsize)

    if cfg.xlim_left is not None or cfg.xlim_right is not None:
        g.ax.set_xlim(left=cfg.xlim_left, right=cfg.xlim_right)

    if cfg.ylim_bottom is not None or cfg.ylim_top is not None:
        g.ax.set_ylim(bottom=cfg.ylim_bottom, top=cfg.ylim_top)

    if g.legend:
        g.legend.remove()
    handles, labels = g.ax.get_legend_handles_labels()
    g.ax.legend(
        handles,
        labels,
        loc="best",
        title="",
        frameon=True,
        facecolor="white",
        framealpha=0.8,
    )

    g.ax.xaxis.set_major_formatter(FuncFormatter(millions_formatter))

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    g.figure.tight_layout()
    g.savefig(plot_path, dpi=cfg.dpi, bbox_inches="tight")
    print(f"\nPlot saved to: {plot_path}")

    return g


def save_metric_summary_csv(
    df: pd.DataFrame,
    configs: Sequence[EvalDataConfig],
    csv_path: Path,
    metric_col: str = "reward",
    metric_title: str | None = None,
) -> pd.DataFrame:
    """Computes final evaluation metric summary statistics, prints, and saves to CSV.

    Args:
        df: DataFrame containing evaluation data.
        configs: Sequence of evaluation configurations to summarize.
        csv_path: Filesystem path to save the summary CSV.
        metric_col: Column name in df containing metric values.
        metric_title: Human-readable metric title for logging (auto-inferred if omitted).

    Returns:
        DataFrame containing summary statistics for each method.
    """
    effective_title = (
        metric_title
        if metric_title is not None
        else metric_col.replace("_", " ").title()
    )
    records: list[dict[str, object]] = []
    print("\n" + "=" * 80)
    print(f"Final Evaluation {effective_title} Summary (95% Bootstrap CI)")
    print("=" * 80)
    for config in configs:
        method_data = df[df["method"] == config.name]
        min_ts = float(method_data["timestep"].min())
        max_ts = float(method_data["timestep"].max())
        final_values = method_data[method_data["timestep"] == max_ts][
            metric_col
        ].to_numpy()
        summary = compute_bootstrap_ci(final_values)
        records.append(
            {
                "method": config.name,
                "min_timestep": min_ts,
                "max_timestep": max_ts,
                f"mean_{metric_col}": summary.mean,
                "ci_lower": summary.ci_lower,
                "ci_upper": summary.ci_upper,
                "num_replicates": len(final_values),
            }
        )
        print(
            f"{config.name:10s} | Timestep: {min_ts:.0f}M - {max_ts:.0f}M | "
            f"Final Mean {effective_title}: {summary.mean:7.4f} "
            f"(95% CI: [{summary.ci_lower:7.4f}, {summary.ci_upper:7.4f}])"
        )
    print("=" * 80)

    summary_df = pd.DataFrame(records)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(csv_path, index=False)
    print(f"Summary statistics saved to: {csv_path}")
    return summary_df


def run_curve_pipeline(
    configs: Sequence[EvalDataConfig],
    plot_path: Path,
    csv_path: Path | None = None,
    selected_methods: Sequence[str] | set[str] | None = None,
    plot_config: CurvePlotConfig | None = None,
    metric_col: str | None = None,
    metric_title: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs the end-to-end evaluation curve loading, plotting, and CI summary pipeline.

    Args:
        configs: Sequence of evaluation data configurations.
        plot_path: Filesystem path to save the generated plot image.
        csv_path: Optional filesystem path for the summary CSV. If omitted,
            defaults to plot_path with a '.csv' extension.
        selected_methods: Optional sequence or set of method names to filter.
        plot_config: Optional plotting configuration options. Defaults to CurvePlotConfig().
        metric_col: Optional column name for metric values. Inferred from plot_config if omitted.
        metric_title: Optional human-readable title for logging.

    Returns:
        A tuple of (eval_df, summary_df) containing the full evaluation records
        and the bootstrap CI summary statistics.
    """
    configs_to_plot = filter_configs_by_method(configs, selected_methods)
    df = load_eval_data_df(configs_to_plot)

    cfg = plot_config if plot_config is not None else CurvePlotConfig()
    plot_curves(df, plot_path, config=cfg)

    col = metric_col if metric_col is not None else cfg.y_col
    effective_csv_path = (
        csv_path if csv_path is not None else plot_path.with_suffix(".csv")
    )
    summary_df = save_metric_summary_csv(
        df=df,
        configs=configs_to_plot,
        csv_path=effective_csv_path,
        metric_col=col,
        metric_title=metric_title,
    )

    return df, summary_df
