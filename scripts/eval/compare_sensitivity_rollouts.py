"""Compare rollout animations between sensitivity cases.

Finds hyperparameter configs (L_gain, k_steepness, eps_margin) that succeeded
with the 3.f.f model (less sensitive) but failed with the 3.k.h model (more
sensitive) on the same 3.k.g scenario.  For a selection of those configs,
generates rollout GIF/MP4 animations for both models side-by-side.

Usage:
    python scripts/eval/compare_sensitivity_rollouts.py
"""

import csv
import json
import os
from copy import deepcopy
from pathlib import Path
from pprint import pprint
from typing import Any, Literal, cast

import contgrid  # noqa: F401
import imageio
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

from hrl_tl.config.wrapper import (
    GCLTLWrapperConfigReader,
    TLSB3PipelineConfigReader,
)
from hrl_tl.envs.tl_fourroom import ContRoomsObsVarValueInfoGenerator
from hrl_tl.utils.render import DiscreteActionVectorRenderer
from hrl_tl.wrappers.low_level_policies.cpc import (
    CPCCompositePolicy,
    LambdaConfig,
)

# ── helpers ──────────────────────────────────────────────────────────────────


def load_sensitivity_csv(csv_path: Path) -> list[dict[str, str]]:
    """Load a sensitivity results CSV into a list of row dicts."""
    with csv_path.open("r") as f:
        return list(csv.DictReader(f))


def find_divergent_configs(
    less_sensitive_csv: Path,
    more_sensitive_csv: Path,
) -> tuple[
    list[dict[str, Any]],  # succeeded in less_sensitive only
    list[dict[str, Any]],  # succeeded in both
    list[dict[str, Any]],  # succeeded in more_sensitive only
]:
    """Cross-reference two sensitivity CSVs.

    Returns three lists of hyperparameter dicts sorted by the less-sensitive
    model's reward (best first).
    """
    less_rows = {
        (r["L_gain"], r["k_steepness"], r["eps_margin"]): r
        for r in load_sensitivity_csv(less_sensitive_csv)
    }
    more_rows = {
        (r["L_gain"], r["k_steepness"], r["eps_margin"]): r
        for r in load_sensitivity_csv(more_sensitive_csv)
    }

    only_less: list[dict[str, Any]] = []
    both: list[dict[str, Any]] = []
    only_more: list[dict[str, Any]] = []

    common_keys = set(less_rows.keys()) & set(more_rows.keys())
    for key in common_keys:
        lr = less_rows[key]
        mr = more_rows[key]
        entry = {
            "L_gain": float(key[0]),
            "k_steepness": float(key[1]),
            "eps_margin": float(key[2]),
            "less_reward": float(lr["episode_reward"]),
            "more_reward": float(mr["episode_reward"]),
            "less_ep_len": float(lr["mean_episode_length"]),
            "more_ep_len": float(mr["mean_episode_length"]),
            "less_success": lr["episode_success"] == "True",
            "more_success": mr["episode_success"] == "True",
        }
        if entry["less_success"] and not entry["more_success"]:
            only_less.append(entry)
        elif entry["less_success"] and entry["more_success"]:
            both.append(entry)
        elif not entry["less_success"] and entry["more_success"]:
            only_more.append(entry)

    only_less.sort(key=lambda x: x["less_reward"], reverse=True)
    both.sort(key=lambda x: x["less_reward"], reverse=True)
    only_more.sort(key=lambda x: x["more_reward"], reverse=True)

    return only_less, both, only_more


def select_representative_configs(
    configs: list[dict[str, Any]],
    n: int = 5,
) -> list[dict[str, Any]]:
    """Pick *n* configs spread across the list (best, worst, and evenly spaced)."""
    if len(configs) <= n:
        return configs
    indices = np.linspace(0, len(configs) - 1, n, dtype=int).tolist()
    return [configs[i] for i in indices]


def record_rollout(
    *,
    model: OnPolicyAlgorithm | OffPolicyAlgorithm,
    eval_pipeline_config_reader: SB3PipelineConfigReader,
    gcltl_config_reader: GCLTLWrapperConfigReader,
    var_info_generator: ContRoomsObsVarValueInfoGenerator,
    fixed_spec: str,
    fixed_scenario_config: dict[str, Any],
    lambda_config: LambdaConfig,
    action_space_type: Literal["discrete", "continuous"],
    goal_rep: Literal["index", "one_hot"],
    normalize_lambdas: bool,
    deterministic: bool,
    save_path: str,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run one rollout episode and save animation frames as GIF + MP4.

    Returns a dict with rollout stats (reward, success, length).
    """
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
    demo_env = pipeline.env_loader.env()

    # Set up action renderers
    composite_action_renderer = DiscreteActionVectorRenderer(
        arrow_width=0.05,
        arrow_color="green",
        num_directions=demo_env.action_space.n,  # type: ignore
    )
    goal_action_renderer = DiscreteActionVectorRenderer(
        arrow_color="blue",
        num_directions=demo_env.action_space.n,  # type: ignore
    )
    constraint_action_renderer = DiscreteActionVectorRenderer(
        arrow_color="red",
        num_directions=demo_env.action_space.n,  # type: ignore
    )

    obs, _ = demo_env.reset()
    terminated: bool = False
    truncated: bool = False
    frames = [demo_env.render()]
    rewards: list[float] = []
    success: bool = False

    while not (terminated or truncated):
        action, _ = policy.predict(obs, deterministic=deterministic)  # type: ignore

        # Update renderers with latest CPC action probabilities
        if (
            policy.action_combinations is not None
            and policy.last_joint_prob is not None
        ):
            if policy.action_combinations.shape[1] == 1:
                action_combinations = np.hstack(
                    [
                        policy.action_combinations,
                        np.ones((policy.action_combinations.shape[0], 1)) * 5,
                    ]
                )
            else:
                action_combinations = policy.action_combinations

            composite_action_renderer.set_multi_discrete_probabilities(
                action_combinations, policy.last_joint_prob
            )
            goal_action_renderer.set_multi_discrete_probabilities(
                action_combinations, policy.last_goal_prob
            )
            constraint_action_renderer.set_multi_discrete_probabilities(
                action_combinations, policy.last_constraint_prob
            )

        obs, reward, terminated, truncated, info = demo_env.step(action)
        if verbose:
            print(
                f"  step {len(rewards) + 1}: R={reward:.2f}, term={terminated}, "
                f"trunc={truncated}, success={info.get('is_success', 'N/A')}"
            )
        rewards.append(float(reward))
        _ = demo_env.render()

        # Post-render action arrows onto the frame
        frame_with_actions = constraint_action_renderer.render(
            demo_env.unwrapped.env  # type: ignore
        )
        frame_with_actions = goal_action_renderer.render(
            demo_env.unwrapped.env  # type: ignore
        )
        frame_with_actions = composite_action_renderer.render(
            demo_env.unwrapped.env  # type: ignore
        )
        frames.append(frame_with_actions)

        if info.get("is_success", False):
            success = True

    demo_env.close()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.mimsave(save_path, frames, fps=2, loop=10)  # type: ignore
    mp4_path = save_path.replace(".gif", ".mp4")
    imageio.mimsave(mp4_path, frames, fps=2)  # type: ignore
    if verbose:
        total_r = sum(rewards)
        print(
            f"  → total_reward={total_r:.3f}, success={success}, "
            f"length={len(rewards)}, saved={save_path}"
        )

    return {
        "total_reward": sum(rewards),
        "success": success,
        "episode_length": len(rewards),
        "save_path": save_path,
    }


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    # ── Experiment identifiers ───────────────────────────────────────────
    less_sensitive_exp: str = "3.f.f"  # PPO model (less sensitive)
    more_sensitive_exp: str = "3.k.m"  # SDSAC model (more sensitive)
    scenario_name: str = "3.k.g"  # common evaluation scenario
    rep: int = 0

    fixed_spec: str = "Fpsi_td & G!psi_hl"
    device: str = "cuda:2"
    action_space_type: Literal["discrete", "continuous"] = "discrete"
    goal_rep: Literal["index", "one_hot"] = "one_hot"
    normalize_lambdas: bool = False
    deterministic: bool = False

    # How many divergent configs to animate
    n_configs: int = 5
    # Also record some configs that succeeded in both (for reference)
    n_both_configs: int = 2

    verbose_steps: bool = False  # step-by-step printing

    # ── Paths ────────────────────────────────────────────────────────────
    less_csv = Path(
        f"out/fr_cont/cpc/sensitivity/"
        f"{less_sensitive_exp}_fixed_spec_map_{scenario_name}_rep{rep}/"
        f"sensitivity_results.csv"
    )
    more_csv = Path(
        f"out/fr_cont/cpc/sensitivity/"
        f"{more_sensitive_exp}_fixed_spec_map_{scenario_name}_rep{rep}/"
        f"sensitivity_results.csv"
    )

    less_model_path = f"out/fr_cont/cpc/pr/{less_sensitive_exp}/cpc_20.0M/rep_{rep}/best_model.zip"
    more_model_path = f"out/fr_cont/cpc/pr/{more_sensitive_exp}/cpc_20.0M/rep_{rep}/best_model.zip"

    less_eval_config_path = f"configs/fr_cont/cpc/pr/{less_sensitive_exp}_train_primitive_eval_cpc_fix.yaml"
    more_eval_config_path = (
        f"configs/fr_cont/cpc/pr_soft/"
        f"{more_sensitive_exp}_train_primitive_eval_cpc_fix.yaml"
    )

    gcltl_config_path = (
        "configs/fr_cont/gcltl/wrapper/gc_ltl_dense_oh_0.35.yaml"
    )
    saved_scenario_path = Path(
        f"configs/fr_cont/env/saved/{scenario_name}_env_kwargs_{fixed_spec}.yaml"
    )

    output_dir = Path(
        f"out/fr_cont/cpc/sensitivity/rollout_comparison/"
        f"{less_sensitive_exp}_vs_{more_sensitive_exp}_{scenario_name}_rep{rep}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Validate paths ───────────────────────────────────────────────────
    for p, label in [
        (less_csv, "less-sensitive CSV"),
        (more_csv, "more-sensitive CSV"),
        (Path(less_model_path), "less-sensitive model"),
        (Path(more_model_path), "more-sensitive model"),
        (saved_scenario_path, "saved scenario"),
    ]:
        # Allow .zip suffix mismatch
        if not p.exists() and not p.with_suffix(".zip").exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    # ── Load sensitivity CSVs and find divergent configs ──────────────
    print("=" * 72)
    print("Loading sensitivity CSVs and cross-referencing configs...")
    print("=" * 72)

    only_less, both_success, only_more = find_divergent_configs(
        less_csv, more_csv
    )

    print(f"  Succeeded in {less_sensitive_exp} only: {len(only_less)}")
    print(f"  Succeeded in both:                      {len(both_success)}")
    print(f"  Succeeded in {more_sensitive_exp} only: {len(only_more)}")

    # Pick representative divergent configs
    selected_divergent = select_representative_configs(only_less, n=n_configs)
    selected_both = select_representative_configs(
        both_success, n=n_both_configs
    )

    print(
        f"\nSelected {len(selected_divergent)} divergent configs for rollout:"
    )
    for i, c in enumerate(selected_divergent):
        print(
            f"  [{i + 1}] L={c['L_gain']:.1f}, k={c['k_steepness']:.2f}, "
            f"eps={c['eps_margin']:.1f} | "
            f"{less_sensitive_exp}: R={c['less_reward']:.3f} (✓) | "
            f"{more_sensitive_exp}: R={c['more_reward']:.3f} (✗)"
        )

    if selected_both:
        print(
            f"\nSelected {len(selected_both)} both-success configs for reference:"
        )
        for i, c in enumerate(selected_both):
            print(
                f"  [{i + 1}] L={c['L_gain']:.1f}, k={c['k_steepness']:.2f}, "
                f"eps={c['eps_margin']:.1f} | "
                f"{less_sensitive_exp}: R={c['less_reward']:.3f} (✓) | "
                f"{more_sensitive_exp}: R={c['more_reward']:.3f} (✓)"
            )

    # ── Load scenario config ─────────────────────────────────────────────
    print("\nLoading scenario config and models...")
    with saved_scenario_path.open("r") as f:
        fixed_scenario_config: dict[str, Any] = yaml.safe_load(f)

    # ── Load models ──────────────────────────────────────────────────────
    gcltl_config_reader = GCLTLWrapperConfigReader.from_yaml(gcltl_config_path)
    var_info_generator = ContRoomsObsVarValueInfoGenerator()

    less_eval_reader = SB3PipelineConfigReader.from_yaml(less_eval_config_path)
    more_eval_reader = SB3PipelineConfigReader.from_yaml(more_eval_config_path)

    print(f"  Loading {less_sensitive_exp} (PPO) from {less_model_path}")
    less_model = cast(
        OnPolicyAlgorithm | OffPolicyAlgorithm,
        PPO.load(less_model_path, device=device),
    )

    print(f"  Loading {more_sensitive_exp} (SDSAC) from {more_model_path}")
    more_model = cast(
        OnPolicyAlgorithm | OffPolicyAlgorithm,
        SDSAC.load(more_model_path, device=device),
    )

    # ── Generate rollouts ────────────────────────────────────────────────
    all_rollout_results: list[dict[str, Any]] = []

    def run_pair(
        config: dict[str, Any],
        category: str,
        idx: int,
    ) -> None:
        L = config["L_gain"]
        k = config["k_steepness"]
        eps = config["eps_margin"]
        lambda_config = LambdaConfig(L_gain=L, k_steepness=k, eps_margin=eps)

        tag = f"{category}_{idx + 1}_L{L:.1f}_k{k:.2f}_eps{eps:.1f}"
        print(f"\n{'─' * 60}")
        print(f"Config: {tag}")
        print(f"  Lambda: L_gain={L}, k_steepness={k}, eps_margin={eps}")
        print(f"{'─' * 60}")

        # Less sensitive model rollout
        print(f"\n  [{less_sensitive_exp}] Rolling out...")
        less_path = str(output_dir / f"{tag}_{less_sensitive_exp}.gif")
        less_stats = record_rollout(
            model=less_model,
            eval_pipeline_config_reader=less_eval_reader,
            gcltl_config_reader=gcltl_config_reader,
            var_info_generator=var_info_generator,
            fixed_spec=fixed_spec,
            fixed_scenario_config=fixed_scenario_config,
            lambda_config=lambda_config,
            action_space_type=action_space_type,
            goal_rep=goal_rep,
            normalize_lambdas=normalize_lambdas,
            deterministic=deterministic,
            save_path=less_path,
            verbose=verbose_steps,
        )

        # More sensitive model rollout
        print(f"\n  [{more_sensitive_exp}] Rolling out...")
        more_path = str(output_dir / f"{tag}_{more_sensitive_exp}.gif")
        more_stats = record_rollout(
            model=more_model,
            eval_pipeline_config_reader=more_eval_reader,
            gcltl_config_reader=gcltl_config_reader,
            var_info_generator=var_info_generator,
            fixed_spec=fixed_spec,
            fixed_scenario_config=fixed_scenario_config,
            lambda_config=lambda_config,
            action_space_type=action_space_type,
            goal_rep=goal_rep,
            normalize_lambdas=normalize_lambdas,
            deterministic=deterministic,
            save_path=more_path,
            verbose=verbose_steps,
        )

        result = {
            "category": category,
            "index": idx + 1,
            "L_gain": L,
            "k_steepness": k,
            "eps_margin": eps,
            f"{less_sensitive_exp}_reward": less_stats["total_reward"],
            f"{less_sensitive_exp}_success": less_stats["success"],
            f"{less_sensitive_exp}_length": less_stats["episode_length"],
            f"{less_sensitive_exp}_gif": less_stats["save_path"],
            f"{more_sensitive_exp}_reward": more_stats["total_reward"],
            f"{more_sensitive_exp}_success": more_stats["success"],
            f"{more_sensitive_exp}_length": more_stats["episode_length"],
            f"{more_sensitive_exp}_gif": more_stats["save_path"],
        }
        all_rollout_results.append(result)

        print(
            f"\n  Summary: {less_sensitive_exp} "
            f"R={less_stats['total_reward']:.3f} "
            f"{'✓' if less_stats['success'] else '✗'} "
            f"(len={less_stats['episode_length']}) | "
            f"{more_sensitive_exp} "
            f"R={more_stats['total_reward']:.3f} "
            f"{'✓' if more_stats['success'] else '✗'} "
            f"(len={more_stats['episode_length']})"
        )

    # Run divergent configs (succeeded in less-sensitive only)
    for idx, config in enumerate(selected_divergent):
        run_pair(config, category="divergent", idx=idx)

    # Run both-success configs (reference)
    for idx, config in enumerate(selected_both):
        run_pair(config, category="both_success", idx=idx)

    # ── Save comparison summary ──────────────────────────────────────────
    summary = {
        "less_sensitive_model": less_sensitive_exp,
        "more_sensitive_model": more_sensitive_exp,
        "scenario": scenario_name,
        "fixed_spec": fixed_spec,
        "rep": rep,
        "deterministic": deterministic,
        "total_divergent_configs": len(only_less),
        "total_both_success_configs": len(both_success),
        "total_only_more_configs": len(only_more),
        "rollouts": all_rollout_results,
    }

    summary_path = output_dir / "comparison_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{'=' * 72}")
    print(f"Comparison summary saved to: {summary_path}")
    print(f"Animations saved to: {output_dir}")
    print(f"{'=' * 72}")

    # Print final table
    print(
        f"\n{'Category':<15} {'#':<3} {'L':>5} {'k':>6} {'eps':>5} | "
        f"{'3.f.f R':>8} {'ok':>3} {'len':>4} | "
        f"{'3.k.h R':>8} {'ok':>3} {'len':>4}"
    )
    print("─" * 80)
    for r in all_rollout_results:
        ok_l = "✓" if r[f"{less_sensitive_exp}_success"] else "✗"
        ok_m = "✓" if r[f"{more_sensitive_exp}_success"] else "✗"
        print(
            f"{r['category']:<15} {r['index']:<3} "
            f"{r['L_gain']:>5.1f} {r['k_steepness']:>6.2f} {r['eps_margin']:>5.1f} | "
            f"{r[f'{less_sensitive_exp}_reward']:>8.3f} {ok_l:>3} "
            f"{r[f'{less_sensitive_exp}_length']:>4} | "
            f"{r[f'{more_sensitive_exp}_reward']:>8.3f} {ok_m:>3} "
            f"{r[f'{more_sensitive_exp}_length']:>4}"
        )


if __name__ == "__main__":
    main()
