from __future__ import annotations

import csv
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any, Literal, cast

import contgrid  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from gymnasium import spaces
from rl_pipeline.sb3 import SB3Pipeline, SB3PipelineConfigReader
from sb3_soft import SDSAC
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.distributions import (
    CategoricalDistribution,
    MultiCategoricalDistribution,
)
from tqdm import tqdm

CheckpointType = int | Literal["latest", "final", "best"]


def serialize_for_json(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def parse_checkpoint(checkpoint: str) -> CheckpointType:
    value = checkpoint.strip().lower()
    if value == "latest":
        return cast(Literal["latest"], value)
    if value == "final":
        return cast(Literal["final"], value)
    if value == "best":
        return cast(Literal["best"], value)

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            "Checkpoint must be one of: latest, final, best, or an integer timestep."
        ) from exc


def parse_resolution(resolution: str) -> float | tuple[float, float]:
    value = resolution.strip()
    if "," in value:
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 2:
            raise ValueError(
                "Resolution tuple must have exactly two comma-separated values, e.g. 0.5,1.0"
            )
        x_step = float(parts[0])
        y_step = float(parts[1])
        if x_step <= 0 or y_step <= 0:
            raise ValueError("Resolution values must be positive.")
        return (x_step, y_step)

    scalar = float(value)
    if scalar <= 0:
        raise ValueError("Resolution must be positive.")
    return scalar


def try_register_contgrid() -> None:
    try:
        __import__("contgrid")
    except ModuleNotFoundError:
        pass


def find_state_enumerator_env(env: Any) -> Any:
    if hasattr(env, "all_possible_states_at_resolution"):
        return env

    visited = set()
    stack = [env]

    while stack:
        current = stack.pop()
        current_id = id(current)
        if current_id in visited:
            continue
        visited.add(current_id)

        if hasattr(current, "all_possible_states_at_resolution"):
            return current

        for attr in ("env", "unwrapped"):
            inner = getattr(current, attr, None)
            if inner is not None and id(inner) not in visited:
                stack.append(inner)

    raise AttributeError(
        "Could not find `all_possible_states_at_resolution` on the evaluation environment or wrapped envs."
    )


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


def resolve_wrapper_config_path(train_config_path: str) -> Path:
    """Resolve wrapper config path from a SB3 pipeline YAML.

    We only use the wrapper YAML to recover the predicate list (for goal_pred
    encoding). We do *not* rely on rl_pipeline's wrapper parsing here.
    """

    reader = SB3PipelineConfigReader.from_yaml(train_config_path)
    wrapper_rel = getattr(reader, "wrapper_config_file", "")
    if not wrapper_rel:
        raise ValueError(
            "The pipeline config does not define `wrapper_config_file`, so the script "
            "cannot infer the predicate order for `goal_pred`."
        )

    config_dir_raw = getattr(reader, "config_dir", "")
    candidate_roots: list[Path] = []
    if config_dir_raw:
        candidate_roots.append(Path(config_dir_raw))
    candidate_roots.append(Path(train_config_path).parent)

    for root in candidate_roots:
        candidate = (root / wrapper_rel).resolve()
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Wrapper config file not found. Tried: "
        + ", ".join(
            str((root / wrapper_rel).resolve()) for root in candidate_roots
        )
    )


def load_predicate_names(wrapper_config_path: Path) -> list[str]:
    """Load predicate names from either a GCLTL or TL wrapper yaml."""

    with wrapper_config_path.open("r") as f:
        wrapper_cfg = yaml.safe_load(f)

    if not isinstance(wrapper_cfg, dict):
        raise ValueError(
            f"Wrapper config must be a dict, got {type(wrapper_cfg)} from {wrapper_config_path}"
        )

    predicates_raw: Any
    if "predicates" in wrapper_cfg:
        predicates_raw = wrapper_cfg["predicates"]
    elif "atomic_predicates" in wrapper_cfg:
        predicates_raw = wrapper_cfg["atomic_predicates"]
    else:
        raise KeyError(
            "Wrapper config must define either `predicates` or `atomic_predicates`. "
            f"Found keys: {sorted(wrapper_cfg.keys())}"
        )

    if not isinstance(predicates_raw, list) or not predicates_raw:
        raise ValueError(
            f"Predicate list is empty or invalid in {wrapper_config_path}: {predicates_raw!r}"
        )

    names: list[str] = []
    for item in predicates_raw:
        if not isinstance(item, dict) or "name" not in item:
            raise ValueError(
                f"Each predicate must be a mapping with a 'name'. Got: {item!r}"
            )
        names.append(str(item["name"]))

    # Match GoalRep behavior: predicates sorted by name.
    names_sorted = sorted(names)
    if len(set(names_sorted)) != len(names_sorted):
        raise ValueError(
            f"Duplicate predicate names found in {wrapper_config_path}: {names_sorted}"
        )
    return names_sorted


def build_goal_pred_value_for_model(
    model: BaseAlgorithm,
    predicate_names_sorted: list[str],
    goal_pred_name: str,
) -> np.ndarray | int:
    if not isinstance(model.observation_space, spaces.Dict):
        raise TypeError(
            f"Model observation_space must be a Dict to inject goal_pred; got {type(model.observation_space)}"
        )
    if "goal_pred" not in model.observation_space.spaces:
        raise KeyError(
            "Model observation_space does not include 'goal_pred'. "
            f"Keys: {list(model.observation_space.spaces.keys())}"
        )

    goal_space = model.observation_space.spaces["goal_pred"]
    goal_index = predicate_names_sorted.index(goal_pred_name)

    if isinstance(goal_space, spaces.Discrete):
        return int(goal_index)

    if isinstance(goal_space, spaces.MultiDiscrete):
        n = int(np.asarray(goal_space.nvec).size)
        if n != len(predicate_names_sorted):
            raise ValueError(
                "Model goal_pred size does not match predicate list. "
                f"goal_space.nvec has size {n}, predicates has size {len(predicate_names_sorted)}"
            )
        one_hot = np.zeros(n, dtype=np.int64)
        one_hot[goal_index] = 1
        return one_hot

    if isinstance(goal_space, spaces.Box):
        # Some models may represent goal conditioning as a continuous vector.
        n = int(np.prod(goal_space.shape))
        if n != len(predicate_names_sorted):
            raise ValueError(
                "Model goal_pred Box shape does not match predicate list. "
                f"goal_space.shape={goal_space.shape}, predicates has size {len(predicate_names_sorted)}"
            )
        vec = np.zeros(goal_space.shape, dtype=np.float32)
        vec.reshape(-1)[goal_index] = 1.0
        return vec

    raise TypeError(f"Unsupported goal_pred space type: {type(goal_space)}")


def prepare_observation_for_policy(
    model: BaseAlgorithm,
    observation: Any,
    goal_pred_value: np.ndarray | int,
) -> dict[str, Any]:
    """Make a state-enumerator observation compatible with the primitive policy.

    - Ensures the observation is a dict matching model.observation_space keys.
    - Injects the required `goal_pred` value.
    - Sanitizes Box entries (replace inf/nan, clip to bounds) so `get_distribution`
      doesn't produce NaNs.
    """

    if not isinstance(observation, dict):
        raise TypeError(
            f"Expected dict observation for Dict policy, got {type(observation)}"
        )

    if not isinstance(model.observation_space, spaces.Dict):
        raise TypeError(
            f"Model observation_space must be a Dict; got {type(model.observation_space)}"
        )

    expected_spaces = model.observation_space.spaces

    prepared: dict[str, Any] = {}
    for key in expected_spaces.keys():
        if key == "goal_pred":
            prepared[key] = goal_pred_value
            continue
        if key not in observation:
            raise KeyError(
                f"State enumerator observation is missing key {key!r}. "
                f"Available keys: {list(observation.keys())}"
            )
        prepared[key] = observation[key]

    # Sanitize numeric values for Box spaces.
    for key, space in expected_spaces.items():
        if key not in prepared:
            continue

        if isinstance(space, spaces.Box):
            arr = np.asarray(prepared[key], dtype=np.float64)
            if np.isposinf(arr).any():
                arr = np.where(np.isposinf(arr), np.asarray(space.high), arr)
            if np.isneginf(arr).any():
                arr = np.where(np.isneginf(arr), np.asarray(space.low), arr)
            if np.isnan(arr).any():
                arr = np.where(np.isnan(arr), 0.0, arr)

            # Clip to the space bounds (broadcast-safe for matching shapes).
            arr = np.clip(arr, np.asarray(space.low), np.asarray(space.high))
            prepared[key] = arr

        elif isinstance(space, spaces.MultiDiscrete):
            if key == "goal_pred":
                prepared[key] = np.asarray(prepared[key], dtype=np.int64)

    return prepared


def load_model_and_env(
    train_config_path: str,
    saved_scenario_path: str,
    model_path: str,
    model_class_name: Literal["PPO", "SDSAC"],
    device: str | None,
) -> tuple[BaseAlgorithm, Any]:
    try_register_contgrid()

    pipeline_config_reader = SB3PipelineConfigReader.from_yaml(
        train_config_path
    )
    if hasattr(pipeline_config_reader, "wrapper_config_file"):
        pipeline_config_reader.wrapper_config_file = ""

    config: Any = pipeline_config_reader.to_config()
    saved_scenario_file = Path(saved_scenario_path)
    if not saved_scenario_file.exists():
        raise FileNotFoundError(
            f"Saved scenario file not found: {saved_scenario_file}"
        )

    with saved_scenario_file.open("r") as f:
        fixed_scenario_config: dict[str, Any] = yaml.safe_load(f)

    config.env_config.env_kwargs["scenario_config"] = deepcopy(
        fixed_scenario_config
    )
    pipeline = SB3Pipeline(config=config, verbose=True)
    env = pipeline.env_loader.env()

    resolved_model_path = resolve_model_path(model_path)
    model_class: type[BaseAlgorithm] = (
        PPO if model_class_name == "PPO" else SDSAC
    )
    load_kwargs: dict[str, Any] = {}
    if device is not None:
        load_kwargs["device"] = device

    loaded_model = model_class.load(str(resolved_model_path), **load_kwargs)
    return loaded_model, env


def compute_basic_stats(values: list[float]) -> dict[str, float | None]:
    finite_values = np.asarray(
        [v for v in values if np.isfinite(v)], dtype=np.float64
    )

    if finite_values.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None}

    return {
        "mean": float(np.mean(finite_values)),
        "std": float(np.std(finite_values)),
        "min": float(np.min(finite_values)),
        "max": float(np.max(finite_values)),
    }


def compute_distribution_stats(
    model: BaseAlgorithm, observation: Any
) -> dict[str, Any]:
    obs_tensor, _ = model.policy.obs_to_tensor(observation)
    dist = cast(Any, model.policy).get_distribution(obs_tensor)

    if isinstance(dist, CategoricalDistribution):
        probs_tensor = dist.distribution.probs
        entropy_tensor = dist.distribution.entropy()

        probs = probs_tensor.detach().cpu().numpy()
        entropy = entropy_tensor.detach().cpu().numpy()
        action_probs = (
            np.squeeze(probs, axis=0) if probs.ndim == 2 else np.asarray(probs)
        )
        greedy_action = int(np.argmax(action_probs))

        return {
            "distribution_type": "categorical",
            "entropy": float(np.squeeze(entropy)),
            "entropy_per_dim": [float(np.squeeze(entropy))],
            "max_prob": float(np.max(action_probs)),
            "greedy_action": greedy_action,
            "action_probs": action_probs.tolist(),
        }

    if isinstance(dist, MultiCategoricalDistribution):
        entropy_per_dim = [
            d.entropy().detach().cpu().numpy() for d in dist.distribution
        ]
        probs_per_dim = [
            d.probs.detach().cpu().numpy() for d in dist.distribution
        ]

        entropy_values = [float(np.squeeze(value)) for value in entropy_per_dim]
        probs_list: list[list[float]] = []
        greedy_action: int | list[int]
        greedy_action = []
        max_probs: list[float] = []

        for probs in probs_per_dim:
            squeezed_probs = (
                np.squeeze(probs, axis=0)
                if probs.ndim == 2
                else np.asarray(probs)
            )
            probs_list.append([float(v) for v in squeezed_probs.tolist()])
            greedy_action.append(int(np.argmax(squeezed_probs)))
            max_probs.append(float(np.max(squeezed_probs)))

        return {
            "distribution_type": "multi_categorical",
            "entropy": float(np.sum(entropy_values)),
            "entropy_per_dim": entropy_values,
            "max_prob": float(np.mean(max_probs))
            if max_probs
            else float("nan"),
            "greedy_action": greedy_action,
            "action_probs": probs_list,
        }

    raise TypeError(f"Unsupported distribution type: {type(dist)}")


def analyze_states(
    model: BaseAlgorithm,
    state_map: dict[tuple[float, float], Any],
    goal_pred_name: str,
    goal_pred_value: np.ndarray | int,
) -> list[dict[str, Any]]:
    if len(state_map) == 0:
        raise ValueError(
            "No states were returned by all_possible_states_at_resolution()."
        )

    rows: list[dict[str, Any]] = []
    for (x, y), observation in tqdm(
        state_map.items(), desc=f"Analyzing states (goal={goal_pred_name})"
    ):
        prepared_obs = prepare_observation_for_policy(
            model=model,
            observation=observation,
            goal_pred_value=goal_pred_value,
        )
        stats = compute_distribution_stats(model, prepared_obs)
        rows.append(
            {
                "goal_pred_name": goal_pred_name,
                "x": float(x),
                "y": float(y),
                "distribution_type": stats["distribution_type"],
                "entropy": float(stats["entropy"]),
                "max_prob": float(stats["max_prob"]),
                "greedy_action": stats["greedy_action"],
                "entropy_per_dim": stats["entropy_per_dim"],
                "action_probs": stats["action_probs"],
            }
        )

    return rows


def write_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "goal_pred_name",
        "x",
        "y",
        "distribution_type",
        "entropy",
        "max_prob",
        "greedy_action",
        "entropy_per_dim",
        "action_probs",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "entropy_per_dim": json.dumps(
                        serialize_for_json(row["entropy_per_dim"])
                    ),
                    "action_probs": json.dumps(
                        serialize_for_json(row["action_probs"])
                    ),
                }
            )


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    entropy_values = [float(row["entropy"]) for row in rows]
    max_prob_values = [float(row["max_prob"]) for row in rows]
    distribution_types = Counter(str(row["distribution_type"]) for row in rows)

    greedy_actions_raw = [row["greedy_action"] for row in rows]

    if greedy_actions_raw and isinstance(greedy_actions_raw[0], list):
        greedy_action_counts = {
            f"dim_{dim}": dict(
                Counter(actions[dim] for actions in greedy_actions_raw)
            )
            for dim in range(len(greedy_actions_raw[0]))
        }
    else:
        greedy_action_counts = dict(
            Counter(int(action) for action in greedy_actions_raw)
        )

    return {
        "num_states": len(rows),
        "distribution_types": dict(distribution_types),
        "entropy": compute_basic_stats(entropy_values),
        "max_prob": compute_basic_stats(max_prob_values),
        "greedy_action_counts": greedy_action_counts,
        "sample_action_probs": [
            serialize_for_json(row["action_probs"]) for row in rows[:5]
        ],
        "sample_entropy_per_dim": [
            serialize_for_json(row["entropy_per_dim"]) for row in rows[:5]
        ],
    }


def write_summary_json(
    rows: list[dict[str, Any]],
    summary_path: Path,
    run_config: dict[str, Any],
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows_by_goal: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        goal_name = str(row.get("goal_pred_name", ""))
        rows_by_goal.setdefault(goal_name, []).append(row)

    summary_by_goal = {
        goal_name: summarize_rows(goal_rows)
        for goal_name, goal_rows in rows_by_goal.items()
        if goal_name
    }

    top_10_by_entropy_by_goal = {
        goal_name: sorted(
            goal_rows,
            key=lambda row: float(row["entropy"]),
            reverse=True,
        )[:10]
        for goal_name, goal_rows in rows_by_goal.items()
        if goal_name
    }

    summary = {
        "train_config": run_config["train_config"],
        "model_path": run_config["model_path"],
        "model_class_name": run_config["model_class_name"],
        "resolution": run_config["resolution"],
        "agent_name": run_config["agent_name"],
        "device": run_config["device"],
        "goal_pred_names": run_config.get("goal_pred_names"),
        "num_rows": len(rows),
        "summary": summarize_rows(rows),
        "summary_by_goal_pred": summary_by_goal,
        "top_10_by_entropy": sorted(
            rows, key=lambda row: float(row["entropy"]), reverse=True
        )[:10],
        "top_10_by_entropy_by_goal_pred": top_10_by_entropy_by_goal,
    }

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, default=serialize_for_json)


def load_rows_from_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Load rows back from the CSV produced by `write_csv`.

    This expects `entropy_per_dim` and `action_probs` to be JSON-encoded strings
    in the CSV and will decode them accordingly.
    """
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # parse numeric fields
            greedy_raw = r.get("greedy_action", "")
            try:
                greedy = json.loads(greedy_raw)
            except Exception:
                try:
                    greedy = int(greedy_raw)
                except Exception:
                    greedy = greedy_raw

            row: dict[str, Any] = {
                "goal_pred_name": r["goal_pred_name"],
                "x": float(r["x"]),
                "y": float(r["y"]),
                "distribution_type": r["distribution_type"],
                "entropy": float(r["entropy"]),
                "max_prob": float(r["max_prob"]),
                "greedy_action": greedy,
                "entropy_per_dim": json.loads(r["entropy_per_dim"]),
                "action_probs": json.loads(r["action_probs"]),
            }
            rows.append(row)
    return rows


def plot_aggregated_entropy_distribution(
    rows: list[dict[str, Any]], output_dir: Path, policy_name: str
) -> None:
    if not rows:
        return
    entropies = [float(r["entropy"]) for r in rows if np.isfinite(r["entropy"])]
    plt.figure(figsize=(8, 5))
    sns.histplot(entropies, kde=True, stat="percent", bins=100)
    plt.xlabel("Entropy")
    plt.title(f"Aggregated Entropy Distribution (all goals) - {policy_name}")
    plt.tight_layout()
    out = (
        output_dir
        / "plots"
        / f"entropy_aggregated_{policy_name.split()[0].lower()}.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


def plot_entropy_distribution_by_goal(
    rows: list[dict[str, Any]], output_dir: Path, policy_name: str
) -> None:
    if not rows:
        return
    # Group rows by goal
    by_goal: dict[str, list[float]] = {}
    for r in rows:
        key = str(r.get("goal_pred_name", ""))
        by_goal.setdefault(key, []).append(float(r["entropy"]))

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    # Create a figure with a subplot per goal if small, else save per-goal files
    goals = list(by_goal.keys())
    if len(goals) <= 6:
        cols = min(3, len(goals))
        rows_n = int(np.ceil(len(goals) / cols))
        fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 3 * rows_n))
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for ax in axes_flat[len(goals) :]:
            ax.set_visible(False)
        for i, g in enumerate(goals):
            sns.histplot(
                by_goal[g], kde=True, stat="percent", bins=100, ax=axes_flat[i]
            )
            axes_flat[i].set_title(f"Goal: {g}")
            axes_flat[i].set_xlabel("Entropy")
        plt.tight_layout()
        plt.savefig(
            plot_dir
            / f"entropy_by_goal_grid_{policy_name.split()[0].lower()}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        for g in goals:
            plt.figure(figsize=(6, 4))
            sns.histplot(by_goal[g], kde=True, stat="percent", bins=100)
            plt.title(f"{policy_name} - Entropy Distribution - Goal: {g}")
            plt.xlabel("Entropy")
            plt.tight_layout()
            # sanitize filename
            fname = (
                f"entropy_goal_{g}".replace("/", "_")
                + f"_{policy_name.split()[0].lower()}"
            )
            plt.savefig(plot_dir / f"{fname}.png", dpi=300, bbox_inches="tight")
            plt.close()


def plot_entropy_boxplot_by_goal(
    rows: list[dict[str, Any]], output_dir: Path, policy_name: str
) -> None:

    if not rows:
        return
    # Prepare data for boxplot
    data = {}
    for r in rows:
        g = str(r.get("goal_pred_name", ""))
        data.setdefault(g, []).append(float(r["entropy"]))

    # Convert to lists for seaborn
    labels = list(data.keys())
    values = [data[k] for k in labels]
    plt.figure(figsize=(max(6, 0.6 * len(labels)), 5))
    sns.boxplot(data=values)
    plt.xticks(
        ticks=np.arange(len(labels)), labels=labels, rotation=45, ha="right"
    )
    plt.ylabel("Entropy")
    plt.title(f"{policy_name} - Entropy by Goal (boxplot)")
    plt.tight_layout()
    plt.savefig(
        output_dir / "entropy_boxplot_by_goal.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def main() -> None:
    # Configure all runtime parameters directly here.Mik915ko821!

    exp_name: str = "3.f.f"
    scenario_name: str = "3.k.g"
    rep: int = 0

    is_soft = "3.k" in exp_name
    pr_name: str = "pr_soft" if is_soft else "pr"
    policy_name: str = (
        "SDSAC Primitive Policy" if is_soft else "PPO Primitive Policy"
    )

    eval_pipeline_config_path = f"configs/fr_cont/cpc/{pr_name}/{exp_name}_train_primitive_eval_cpc_fix.yaml"
    saved_scenario_path = Path(
        f"configs/fr_cont/env/saved/{scenario_name}_env_kwargs_Fpsi_td & G!psi_hl.yaml"
    )

    model_path = (
        f"out/fr_cont/cpc/pr/{exp_name}/cpc_20.0M/rep_{rep}/best_model.zip"
    )
    model_class_name: Literal["PPO", "SDSAC"] = "SDSAC" if is_soft else "PPO"
    resolution_input = "0.25"
    agent_name_input: str | None = None
    device_override: str | None = "cuda:2"
    output_dir = Path(
        f"out/eval/policy_entropy/{exp_name}_all_goal_preds_{scenario_name}_rep{rep}"
    )

    resolution = parse_resolution(resolution_input)
    # CSV plotting option: if an existing CSV is present and the user wants to
    # reuse it, load rows from CSV and skip running the environment/model.
    csv_path = output_dir / "policy_entropy_stats.csv"
    use_existing_csv: bool = True

    rows: list[dict[str, Any]] = []

    if use_existing_csv and csv_path.exists():
        rows = load_rows_from_csv(csv_path)
    else:
        # Load model and environment only when we need to recompute rows.
        model, env = load_model_and_env(
            train_config_path=eval_pipeline_config_path,
            saved_scenario_path=str(saved_scenario_path),
            model_path=model_path,
            model_class_name=model_class_name,
            device=device_override,
        )

        wrapper_config_path = resolve_wrapper_config_path(
            eval_pipeline_config_path
        )
        predicate_names_sorted = load_predicate_names(wrapper_config_path)
        goal_pred_names = predicate_names_sorted

        enum_env = find_state_enumerator_env(env)
        enum_env.reset()
        all_states = enum_env.all_possible_states_at_resolution(resolution)
        available_agent_names = list(all_states.keys())
        if len(available_agent_names) == 0:
            raise RuntimeError(
                "No agent entries returned by all_possible_states_at_resolution()."
            )

        agent_name = agent_name_input or available_agent_names[0]
        if agent_name not in all_states:
            raise KeyError(
                f"Agent '{agent_name}' not found. Available agent names: {available_agent_names}"
            )

        state_map = all_states[agent_name]
        rows = []
        for goal_pred_name in goal_pred_names:
            goal_pred_value = build_goal_pred_value_for_model(
                model=model,
                predicate_names_sorted=predicate_names_sorted,
                goal_pred_name=goal_pred_name,
            )
            rows.extend(
                analyze_states(
                    model=model,
                    state_map=state_map,
                    goal_pred_name=goal_pred_name,
                    goal_pred_value=goal_pred_value,
                )
            )
        run_config: dict[str, Any] = {
            "train_config": eval_pipeline_config_path,
            "saved_scenario_path": str(saved_scenario_path),
            "eval_pipeline_config_path": eval_pipeline_config_path,
            "wrapper_config_path": str(wrapper_config_path),
            "predicate_order": predicate_names_sorted,
            "goal_pred_names": goal_pred_names,
            "model_path": model_path,
            "model_class_name": model_class_name,
            "resolution": resolution_input,
            "agent_name": agent_name,
            "device": device_override,
        }
        # Write CSV/summary only when rows were recomputed. If we're plotting from
        # an existing CSV, avoid overwriting it.
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "policy_entropy_summary.json"
        write_csv(rows, csv_path)
        write_summary_json(rows, summary_path, run_config=run_config)
        summary = summarize_rows(rows)
        positions = len(state_map)
        goals_count = len(goal_pred_names)
        print("Policy entropy analysis complete.")
        print(
            "State-goal pairs analyzed: "
            f"{summary['num_states']} (positions={positions}, goals={goals_count})"
        )
        print(f"Entropy mean: {summary['entropy']['mean']:.6f}")
        print(f"Entropy std: {summary['entropy']['std']:.6f}")
        print(f"Max prob mean: {summary['max_prob']['mean']:.6f}")
        print(f"Outputs written to: {output_dir}")

    # Generate plots for aggregated and per-goal entropy distributions.
    try:
        # Create the plot using seaborn relplot
        sns.set_theme(style="whitegrid", font="Times New Roman")
        plot_aggregated_entropy_distribution(rows, output_dir, policy_name)
        plot_entropy_distribution_by_goal(rows, output_dir, policy_name)
        plot_entropy_boxplot_by_goal(rows, output_dir, policy_name)
    except Exception as exc:
        print(f"Plotting failed: {exc}")


if __name__ == "__main__":
    main()
