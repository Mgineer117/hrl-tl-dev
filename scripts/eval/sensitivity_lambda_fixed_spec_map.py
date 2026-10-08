import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, cast

import contgrid  # noqa: F401
import matplotlib._mathtext_data as md
import matplotlib.pyplot as plt
import numpy as np
import yaml
from rl_pipeline.sb3 import (
    SB3Pipeline,
    SB3PipelineConfig,
    SB3PipelineConfigReader,
)
from sb3_soft import SDSAC
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from tqdm import tqdm

from hrl_tl.config.wrapper import (
    GCLTLWrapperConfigReader,
    TLSB3PipelineConfigReader,
)
from hrl_tl.envs.tl_fourroom import ContRoomsObsVarValueInfoGenerator
from hrl_tl.wrappers.low_level_policies.cpc import (
    CPCCompositePolicy,
    LambdaConfig,
)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 25,
        "axes.labelpad": 34,
        "mathtext.fontset": "stix",
    }
)
md.tex2uni["epsilon"] = 0x03F5


def build_values(min_val: float, max_val: float, bins: int) -> list[float]:
    if bins < 2:
        raise ValueError(f"bins must be >= 2, got {bins}")
    return [float(v) for v in np.linspace(min_val, max_val, bins)]


def serialize_for_json(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def resolve_model_path(model_path: str) -> Path:
    """Resolve model path and handle common .zip suffix variations."""
    path = Path(model_path)

    if path.exists():
        return path

    if path.suffix != ".zip":
        zip_path = path.with_suffix(".zip")
        if zip_path.exists():
            return zip_path
    else:
        no_suffix_path = path.with_suffix("")
        if no_suffix_path.exists():
            return no_suffix_path

    raise FileNotFoundError(
        "Model file not found. Tried paths: "
        f"'{path}'"
        + (f", '{path.with_suffix('.zip')}'" if path.suffix != ".zip" else "")
        + (f", '{path.with_suffix('')}'" if path.suffix == ".zip" else "")
    )


def evaluate_single_config(
    fixed_spec: str,
    fixed_scenario_config: dict[str, Any],
    model: OnPolicyAlgorithm | OffPolicyAlgorithm,
    var_info_generator: ContRoomsObsVarValueInfoGenerator,
    eval_pipeline_config_reader: SB3PipelineConfigReader,
    gcltl_config_reader: GCLTLWrapperConfigReader,
    lambda_config: LambdaConfig,
    action_space_type: Literal["discrete", "continuous"],
    goal_rep: Literal["index", "one_hot"],
    normalize_lambdas: bool,
    n_eval_episodes: int,
    deterministic: bool,
) -> dict[str, Any]:
    policy = CPCCompositePolicy(
        tl_spec=fixed_spec,
        predicates=gcltl_config_reader.predicates,
        model=model,
        var_value_info_generator=var_info_generator,
        lambda_config=lambda_config,
        goal_rep=goal_rep,
        action_space_type=action_space_type,
        normalize_lambdas=normalize_lambdas,
        verbose=False,
    )

    pipeline_config: SB3PipelineConfig = TLSB3PipelineConfigReader(
        tl_spec=fixed_spec, pipeline_config=eval_pipeline_config_reader
    ).to_config()

    pipeline_config.env_config.env_kwargs["scenario_config"] = deepcopy(
        fixed_scenario_config
    )

    pipeline = SB3Pipeline(config=pipeline_config, verbose=False)
    eval_stats = pipeline.evaluate(
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        checkpoint=policy,
        save_to_file=False,
        env="single",
    )

    episode_reward = float(eval_stats.mean_reward)
    if eval_stats.episode_rewards:
        episode_reward = float(eval_stats.episode_rewards[0])

    episode_success: bool | None = None
    if eval_stats.episode_successes:
        episode_success = bool(eval_stats.episode_successes[0])

    episode_failure: bool | None = None
    if eval_stats.episode_failures:
        episode_failure = bool(eval_stats.episode_failures[0])

    return {
        "success_rate": (
            float(eval_stats.success_rate)
            if eval_stats.success_rate is not None
            else float("nan")
        ),
        "failure_rate": (
            float(eval_stats.failure_rate)
            if eval_stats.failure_rate is not None
            else float("nan")
        ),
        "mean_reward": float(eval_stats.mean_reward),
        "episode_reward": episode_reward,
        "mean_episode_length": float(eval_stats.mean_episode_length),
        "episode_success": episode_success,
        "episode_failure": episode_failure,
    }


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def row_sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
        success = row["success_rate"]
        failure = row["failure_rate"]
        reward = row["episode_reward"]

        success_sort = -1e9 if np.isnan(success) else float(success)
        failure_sort = 1e9 if np.isnan(failure) else float(failure)
        reward_sort = -1e9 if np.isnan(reward) else float(reward)

        return (success_sort, -failure_sort, reward_sort)

    sorted_rows = sorted(rows, key=row_sort_key, reverse=True)

    ranked_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(sorted_rows, start=1):
        row_with_rank = dict(row)
        row_with_rank["rank_overall"] = rank
        ranked_rows.append(row_with_rank)

    return ranked_rows


def compute_metric_stats(
    rows: list[dict[str, Any]], key: str
) -> dict[str, float | None]:
    values = np.array([float(row[key]) for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def compute_successful_combination_stats(
    rows: list[dict[str, Any]],
    success_threshold: float = 0.0,
) -> dict[str, float | int]:
    """Count combinations with success_rate above threshold and report ratio."""
    total = len(rows)
    if total == 0:
        return {
            "success_threshold": float(success_threshold),
            "num_successful_combinations": 0,
            "num_total_combinations": 0,
            "successful_ratio": 0.0,
        }

    num_successful = sum(
        1
        for row in rows
        if np.isfinite(float(row["success_rate"]))
        and float(row["success_rate"]) > success_threshold
    )

    return {
        "success_threshold": float(success_threshold),
        "num_successful_combinations": int(num_successful),
        "num_total_combinations": int(total),
        "successful_ratio": float(num_successful / total),
    }


def write_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = [
        "rank_overall",
        "L_gain",
        "k_steepness",
        "eps_margin",
        "success_rate",
        "failure_rate",
        "mean_reward",
        "episode_reward",
        "mean_episode_length",
        "episode_success",
        "episode_failure",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Load evaluation rows from a sensitivity_results.csv file."""
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_summary_json(
    rows: list[dict[str, Any]],
    summary_path: Path,
    run_config: dict[str, Any],
    total_planned: int,
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    best = rows[0] if rows else None

    summary = {
        "fixed_spec": run_config["fixed_spec"],
        "saved_scenario_path": run_config["saved_scenario_path"],
        "n_eval_episodes": run_config["n_eval_episodes"],
        "deterministic": run_config["deterministic"],
        "normalize_lambdas": run_config["normalize_lambdas"],
        "parameter_ranges": {
            "L_gain": {
                "min": run_config["L_gain_min"],
                "max": run_config["L_gain_max"],
                "bins": run_config["L_gain_bins"],
            },
            "k_steepness": {
                "min": run_config["k_steepness_min"],
                "max": run_config["k_steepness_max"],
                "bins": run_config["k_steepness_bins"],
            },
            "eps_margin": {
                "min": run_config["eps_margin_min"],
                "max": run_config["eps_margin_max"],
                "bins": run_config["eps_margin_bins"],
            },
        },
        "total_evaluated": len(rows),
        "total_planned": total_planned,
        "best_config": {
            "L_gain": serialize_for_json(best["L_gain"])
            if best is not None
            else None,
            "k_steepness": (
                serialize_for_json(best["k_steepness"])
                if best is not None
                else None
            ),
            "eps_margin": (
                serialize_for_json(best["eps_margin"])
                if best is not None
                else None
            ),
            "success_rate": (
                serialize_for_json(best["success_rate"])
                if best is not None
                else None
            ),
            "failure_rate": (
                serialize_for_json(best["failure_rate"])
                if best is not None
                else None
            ),
            "episode_reward": (
                serialize_for_json(best["episode_reward"])
                if best is not None
                else None
            ),
        },
        "aggregates": {
            "success_rate": compute_metric_stats(rows, "success_rate"),
            "failure_rate": compute_metric_stats(rows, "failure_rate"),
            "episode_reward": compute_metric_stats(rows, "episode_reward"),
        },
        "successful_combinations": compute_successful_combination_stats(
            rows,
            success_threshold=float(run_config.get("success_threshold", 0.0)),
        ),
        "top_10": rows[:10],
    }

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, default=serialize_for_json)


def _rounded(v: float, digits: int = 10) -> float:
    return round(float(v), digits)


def build_max_projection_matrix(
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    metric_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_vals = np.array(
        sorted({_rounded(float(r[x_key])) for r in rows}), dtype=np.float64
    )
    y_vals = np.array(
        sorted({_rounded(float(r[y_key])) for r in rows}), dtype=np.float64
    )

    x_index = {_rounded(v): i for i, v in enumerate(x_vals.tolist())}
    y_index = {_rounded(v): i for i, v in enumerate(y_vals.tolist())}

    matrix = np.full((len(y_vals), len(x_vals)), np.nan, dtype=np.float64)

    for row in rows:
        x = _rounded(float(row[x_key]))
        y = _rounded(float(row[y_key]))
        metric = float(row[metric_key])
        if not np.isfinite(metric):
            continue

        yi = y_index[y]
        xi = x_index[x]

        current = matrix[yi, xi]
        if np.isnan(current) or metric > current:
            matrix[yi, xi] = metric

    return x_vals, y_vals, matrix


def save_heatmaps(
    rows: list[dict[str, Any]],
    metric_key: str,
    output_path: Path,
) -> None:
    projections = [
        ("L_gain", "k_steepness", "eps_margin"),
        ("L_gain", "eps_margin", "k_steepness"),
        ("k_steepness", "eps_margin", "L_gain"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), constrained_layout=True)

    for ax, (x_key, y_key, z_key) in zip(axes, projections):
        x_vals, y_vals, matrix = build_max_projection_matrix(
            rows=rows,
            x_key=x_key,
            y_key=y_key,
            metric_key=metric_key,
        )

        x_min = float(x_vals.min())
        x_max = float(x_vals.max())
        y_min = float(y_vals.min())
        y_max = float(y_vals.max())

        # Avoid identical axis bounds in smoke runs with a single grid point.
        if np.isclose(x_min, x_max):
            x_min -= 0.5
            x_max += 0.5
        if np.isclose(y_min, y_max):
            y_min -= 0.5
            y_max += 0.5

        image = ax.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            extent=[x_min, x_max, y_min, y_max],
            interpolation="nearest",
            cmap="viridis",
        )
        fig.colorbar(image, ax=ax)

        ax.set_xlabel(x_key, labelpad=10)
        ax.set_ylabel(y_key, labelpad=10)
        ax.set_title(f"{metric_key}: max over {z_key}", pad=12)

        # Force ticks to coincide exactly with the sampled bin values.
        ax.set_xticks(x_vals)
        ax.set_yticks(y_vals)
        ax.tick_params(axis="x", rotation=45, pad=6)
        ax.tick_params(axis="y", pad=6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_3d_success_scatter(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save a 3D scatter of hyperparameters, keeping only non-zero success points."""
    success_rows = [
        row
        for row in rows
        if np.isfinite(float(row["success_rate"]))
        and float(row["success_rate"]) > 0.0
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    if not success_rows:
        ax.text2D(
            0.5,
            0.5,
            "No points with success_rate > 0",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_axis_off()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return

    x = np.array(
        [float(row["L_gain"]) for row in success_rows], dtype=np.float64
    )
    y = np.array(
        [float(row["k_steepness"]) for row in success_rows], dtype=np.float64
    )
    z = np.array(
        [float(row["eps_margin"]) for row in success_rows], dtype=np.float64
    )
    success = np.array(
        [float(row["success_rate"]) for row in success_rows], dtype=np.float64
    )
    reward = np.array(
        [float(row["episode_reward"]) for row in success_rows], dtype=np.float64
    )

    reward_min = float(np.min(reward))
    reward_max = float(np.max(reward))
    if np.isclose(reward_min, reward_max):
        sizes = np.full_like(reward, 90.0)
    else:
        # Map reward to marker area in [40, 220].
        sizes = 40.0 + 180.0 * (
            (reward - reward_min) / (reward_max - reward_min)
        )

    ax3d = cast(Any, ax)
    scatter = ax3d.scatter(
        x,
        y,
        z,
        c=success,
        s=sizes,
        cmap="plasma",
        alpha=0.85,
        edgecolors="black",
        linewidths=0.3,
    )
    # colorbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.05)
    # colorbar.set_label("success_rate")

    ax.tick_params(axis="both", which="major", pad=6)
    ax.set_xlabel(r"Maximum Gain $L$", labelpad=14)
    ax.set_ylabel(r"Transition Steepness $k$", labelpad=14)
    ax.set_zlabel(r"Safety Margin $\epsilon$", labelpad=14)
    # ax.set_title("3D Hyperparameter Scatter (success_rate > 0)")

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_all_artifacts(
    rows: list[dict[str, Any]],
    run_config: dict[str, Any],
    total_planned: int,
    output_dir: Path,
    partial: bool,
) -> None:
    suffix = ".partial" if partial else ""
    ranked_rows = rank_rows(rows)

    write_csv(ranked_rows, output_dir / f"sensitivity_results{suffix}.csv")
    write_summary_json(
        ranked_rows,
        output_dir / f"sensitivity_summary{suffix}.json",
        run_config=run_config,
        total_planned=total_planned,
    )

    for metric in ["success_rate", "failure_rate", "episode_reward"]:
        save_heatmaps(
            ranked_rows,
            metric_key=metric,
            output_path=output_dir
            / f"heatmap_{metric}_max_projection{suffix}.png",
        )

    save_3d_success_scatter(
        ranked_rows,
        output_path=output_dir / f"scatter_3d_success_filtered{suffix}.png",
    )


def regenerate_plots_from_csv(
    csv_path: Path,
    output_dir: Path | None = None,
) -> None:
    """Regenerate heatmaps and 3D scatter plot from an existing CSV."""
    if output_dir is None:
        output_dir = csv_path.parent

    print(f"Loading existing data from {csv_path}...")
    rows = load_csv(csv_path)
    if not rows:
        print(f"No rows found in {csv_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in ["success_rate", "failure_rate", "episode_reward"]:
        save_heatmaps(
            rows,
            metric_key=metric,
            output_path=output_dir / f"heatmap_{metric}_max_projection.png",
        )

    save_3d_success_scatter(
        rows,
        output_path=output_dir / "scatter_3d_success_filtered.png",
    )
    print(f"Figures successfully regenerated in {output_dir}")


def cleanup_partial_artifacts(output_dir: Path) -> int:
    """Remove checkpoint artifacts with '.partial' in the filename."""
    removed = 0
    for partial_path in output_dir.glob("*.partial.*"):
        if partial_path.is_file():
            partial_path.unlink()
            removed += 1
    return removed


def main() -> None:
    # ----------------------------
    # Figure regeneration options
    # ----------------------------
    # Set replot_only to True to regenerate figures without running evaluation.
    replot_only: bool = True
    replot_csv_path: Path | None = None
    custom_output_dir: Path | None = None

    # ----------------------------
    # Fixed run configuration
    # ----------------------------
    exp_name: str = "3.k.m"  # "3.f.f"  # "3.k.m"
    scenario_name: str = "3.k.g"
    rep: int = 0
    output_dir = custom_output_dir or Path(
        f"out/fr_cont/cpc/sensitivity/{exp_name}_fixed_spec_map_{scenario_name}_rep{rep}"
    )

    if replot_only:
        target_csv = replot_csv_path or (output_dir / "sensitivity_results.csv")
        if not target_csv.exists():
            raise FileNotFoundError(
                f"Results CSV not found for replotting: {target_csv}"
            )
        regenerate_plots_from_csv(csv_path=target_csv, output_dir=output_dir)
        return

    model_path = (
        f"out/fr_cont/cpc/pr/{exp_name}/cpc_20.0M/rep_{rep}/best_model.zip"
    )

    is_soft = "3.k" in exp_name
    pr_name: str = "pr_soft" if is_soft else "pr"

    gcltl_config_path = (
        "configs/fr_cont/gcltl/wrapper/gc_ltl_dense_oh_0.35.yaml"
    )
    eval_pipeline_config_path = f"configs/fr_cont/cpc/{pr_name}/{exp_name}_train_primitive_eval_cpc_fix.yaml"
    saved_scenario_path = Path(
        f"configs/fr_cont/env/saved/{scenario_name}_env_kwargs_Fpsi_td & G!psi_hl.yaml"
    )
    fixed_spec = "Fpsi_td & G!psi_hl"

    model_class_name = "SDSAC" if is_soft else "PPO"  # "SDSAC" | "PPO"
    action_space_type: Literal["discrete", "continuous"] = "discrete"
    device = "cuda:2"
    goal_rep: Literal["index", "one_hot"] = "one_hot"
    normalize_lambdas = False

    n_eval_episodes = 1
    deterministic = False

    L_gain_min = 2.0
    L_gain_max = 100.0
    L_gain_bins = 50

    k_steepness_min = 0.1
    k_steepness_max = 1.0
    k_steepness_bins = 10

    eps_margin_min = 0.5
    eps_margin_max = 5.0
    eps_margin_bins = 10

    checkpoint_every = 100
    max_combinations: int | None = None
    success_threshold = 0.0

    run_config: dict[str, Any] = {
        "fixed_spec": fixed_spec,
        "saved_scenario_path": str(saved_scenario_path),
        "n_eval_episodes": n_eval_episodes,
        "deterministic": deterministic,
        "normalize_lambdas": normalize_lambdas,
        "L_gain_min": L_gain_min,
        "L_gain_max": L_gain_max,
        "L_gain_bins": L_gain_bins,
        "k_steepness_min": k_steepness_min,
        "k_steepness_max": k_steepness_max,
        "k_steepness_bins": k_steepness_bins,
        "eps_margin_min": eps_margin_min,
        "eps_margin_max": eps_margin_max,
        "eps_margin_bins": eps_margin_bins,
        "success_threshold": success_threshold,
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    if not saved_scenario_path.exists():
        raise FileNotFoundError(
            f"Saved scenario file not found: {saved_scenario_path}"
        )

    print("Loading fixed scenario config and model resources...")
    with saved_scenario_path.open("r") as f:
        fixed_scenario_config: dict[str, Any] = yaml.safe_load(f)

    model_class: type[BaseAlgorithm]
    if model_class_name == "SDSAC":
        model_class = SDSAC
    else:
        model_class = PPO

    eval_pipeline_config_reader = SB3PipelineConfigReader.from_yaml(
        eval_pipeline_config_path
    )
    gcltl_config_reader = GCLTLWrapperConfigReader.from_yaml(gcltl_config_path)
    var_info_generator = ContRoomsObsVarValueInfoGenerator()
    resolved_model_path = resolve_model_path(model_path)
    model = cast(
        OnPolicyAlgorithm | OffPolicyAlgorithm,
        model_class.load(str(resolved_model_path), device=device),
    )

    L_gain_values = build_values(L_gain_min, L_gain_max, L_gain_bins)
    k_steepness_values = build_values(
        k_steepness_min,
        k_steepness_max,
        k_steepness_bins,
    )
    eps_margin_values = build_values(
        eps_margin_min,
        eps_margin_max,
        eps_margin_bins,
    )

    combinations: list[tuple[float, float, float]] = [
        (L_gain, k_steepness, eps_margin)
        for L_gain in L_gain_values
        for k_steepness in k_steepness_values
        for eps_margin in eps_margin_values
    ]

    if max_combinations is not None:
        combinations = combinations[:max_combinations]

    total_planned = len(combinations)
    print(f"Starting sensitivity sweep: {total_planned} combinations")

    rows: list[dict[str, Any]] = []

    try:
        with tqdm(total=total_planned, desc="Lambda sensitivity") as pbar:
            for idx, (L_gain, k_steepness, eps_margin) in enumerate(
                combinations, start=1
            ):
                lambda_config = LambdaConfig(
                    L_gain=L_gain,
                    k_steepness=k_steepness,
                    eps_margin=eps_margin,
                )

                metric_row = evaluate_single_config(
                    fixed_spec=fixed_spec,
                    fixed_scenario_config=fixed_scenario_config,
                    model=model,
                    var_info_generator=var_info_generator,
                    eval_pipeline_config_reader=eval_pipeline_config_reader,
                    gcltl_config_reader=gcltl_config_reader,
                    lambda_config=lambda_config,
                    action_space_type=action_space_type,
                    goal_rep=goal_rep,
                    normalize_lambdas=normalize_lambdas,
                    n_eval_episodes=n_eval_episodes,
                    deterministic=deterministic,
                )

                metric_row["L_gain"] = float(L_gain)
                metric_row["k_steepness"] = float(k_steepness)
                metric_row["eps_margin"] = float(eps_margin)

                rows.append(metric_row)
                pbar.update(1)

                pbar.set_postfix(
                    {
                        "sr": f"{metric_row['success_rate']:.3f}",
                        "fr": f"{metric_row['failure_rate']:.3f}",
                        "R": f"{metric_row['episode_reward']:.2f}",
                    }
                )

                if checkpoint_every > 0 and idx % checkpoint_every == 0:
                    write_all_artifacts(
                        rows=rows,
                        run_config=run_config,
                        total_planned=total_planned,
                        output_dir=output_dir,
                        partial=True,
                    )

    except KeyboardInterrupt:
        print("\nInterrupted. Writing partial artifacts...")
        write_all_artifacts(
            rows=rows,
            run_config=run_config,
            total_planned=total_planned,
            output_dir=output_dir,
            partial=True,
        )
        raise

    write_all_artifacts(
        rows=rows,
        run_config=run_config,
        total_planned=total_planned,
        output_dir=output_dir,
        partial=False,
    )

    removed_partial_count = cleanup_partial_artifacts(output_dir)

    print("Sensitivity analysis complete.")
    success_stats = compute_successful_combination_stats(
        rows,
        success_threshold=success_threshold,
    )
    print(
        "Successful combinations: "
        f"{success_stats['num_successful_combinations']}/"
        f"{success_stats['num_total_combinations']} "
        f"({success_stats['successful_ratio']:.3f})"
    )
    print(f"Removed partial artifacts: {removed_partial_count}")
    print(f"Results directory: {output_dir}")


if __name__ == "__main__":
    main()
