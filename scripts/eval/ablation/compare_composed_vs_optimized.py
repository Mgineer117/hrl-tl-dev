"""Hierarchical option-slicing evaluation comparing CPC vs pretrained subpolicies on Fetch."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import contgrid  # noqa: F401 - Registers gym environments.
import numpy as np
from absl import app, flags, logging
from rl_pipeline.sb3 import (
    SB3Pipeline,
    SB3PipelineConfig,
    SB3ReplicatePipelineConfigReader,
)
from stable_baselines3 import PPO

import hrl_tl.envs.fetch  # noqa: F401 - Registers fetch robotics environments.
from hrl_tl.config.meta_option import TLMetaOptionPipelineConfigReader
from hrl_tl.tuning import normalize_device

os.environ.setdefault("MUJOCO_GL", "egl")

_CPC_CONFIG_PATH = flags.DEFINE_string(
    "cpc_config_path",
    default="configs/fetch/train/11.b/11.b.b_hrl_train_rep0.yaml",
    help="Path to CPC meta-policy replicate YAML config.",
)
_CPC_MODEL_PATH = flags.DEFINE_string(
    "cpc_model_path",
    default="out/fetch/ltl_hl/11.b.b/hl_policy_fetch_2.0M/rep_0/best_model.zip",
    help="Path to trained CPC meta-policy model checkpoint.",
)
_ABLATION_CONFIG_PATH = flags.DEFINE_string(
    "ablation_config_path",
    default="configs/fetch/train/ablation/12.b.a_meta_rep0.yaml",
    help="Path to Ablation meta-policy replicate YAML config.",
)
_ABLATION_MODEL_PATH = flags.DEFINE_string(
    "ablation_model_path",
    default="out/fetch/ablation/meta/rep_0/best_model.zip",
    help="Path to trained Ablation meta-policy model checkpoint.",
)
_SPEC_PATH = flags.DEFINE_string(
    "spec_path",
    default="assets/formulae/fetch/cpc/all_formulae_1_cla_1_max_pred.json",
    help="Path to candidate LTL specifications JSON file.",
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    default="out/fetch/ablation/eval",
    help="Directory to save evaluation summaries and LaTeX tables.",
)
_DEVICE = flags.DEFINE_string(
    "device",
    default="cpu",
    help="Device for policy inference (e.g. 'cpu', 'cuda:0').",
)
_N_EVAL_EPISODES = flags.DEFINE_integer(
    "n_eval_episodes",
    default=50,
    lower_bound=1,
    help="Number of evaluation episodes per agent.",
)
_SEED_BASE = flags.DEFINE_integer(
    "seed_base",
    default=42,
    help="Base random seed for reproducible episode evaluations.",
)


@dataclass
class OptionInvocation:
    """Record of a single macro-option step within a hierarchical episode."""

    spec: str
    reward: float
    steps: int
    is_success: bool
    lambda_config: dict[str, float] = field(default_factory=dict)


@dataclass
class AgentEvalResult:
    """Evaluation summary for a hierarchical agent across multiple episodes."""

    agent_name: str
    episode_returns: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    episode_successes: list[bool] = field(default_factory=list)
    option_invocations: list[OptionInvocation] = field(default_factory=list)

    @property
    def mean_return(self) -> float:
        return float(np.mean(self.episode_returns)) if self.episode_returns else 0.0

    @property
    def std_return(self) -> float:
        return float(np.std(self.episode_returns)) if self.episode_returns else 0.0

    @property
    def success_rate(self) -> float:
        return (
            float(np.mean(self.episode_successes) * 100.0)
            if self.episode_successes
            else 0.0
        )

    @property
    def mean_length(self) -> float:
        return float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0


def evaluate_hierarchical_agent(
    agent_name: str,
    config_path: Path,
    model_path: Path,
    n_episodes: int,
    seed_base: int,
    device: str,
) -> AgentEvalResult:
    """Evaluates a hierarchical meta-policy and records option slices.

    Args:
        agent_name: Human-readable agent name.
        config_path: Replicate YAML config path.
        model_path: Trained PPO model zip checkpoint.
        n_episodes: Number of evaluation episodes.
        seed_base: Base random seed for environment resets.
        device: Device to load policy on.

    Returns:
        AgentEvalResult containing task metrics and sliced option invocations.
    """
    logging.info("Loading %s meta-policy from %s...", agent_name, model_path)
    reader = SB3ReplicatePipelineConfigReader[
        TLMetaOptionPipelineConfigReader
    ].from_yaml(str(config_path))
    pipeline_config: SB3PipelineConfig = reader.single_pipeline_config.to_config()

    pipeline = SB3Pipeline(config=pipeline_config, verbose=False)
    env = pipeline.env_loader.env()
    model = PPO.load(str(model_path), device=device)

    result = AgentEvalResult(agent_name=agent_name)

    for ep in range(n_episodes):
        ep_seed = seed_base + ep
        obs, _ = env.reset(seed=ep_seed)
        ep_reward = 0.0
        ep_steps = 0
        terminated = False
        truncated = False
        last_info: dict[str, Any] = {}

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, macro_reward, terminated, truncated, info = env.step(action)
            last_info = info
            ep_reward += float(macro_reward)

            spec = str(info.get("current_tl_spec", ""))
            option_steps = int(info.get("meta_option_steps", 1))
            ep_steps += option_steps

            if spec and spec not in ("0", "1", ""):
                lambda_cfg = info.get("lambda_config", {})
                converted_lambda = {
                    k: float(v) for k, v in lambda_cfg.items()
                } if isinstance(lambda_cfg, dict) else {}

                result.option_invocations.append(
                    OptionInvocation(
                        spec=spec,
                        reward=float(macro_reward),
                        steps=option_steps,
                        is_success=bool(info.get("is_success", False)),
                        lambda_config=converted_lambda,
                    )
                )

        result.episode_returns.append(ep_reward)
        result.episode_lengths.append(ep_steps)
        result.episode_successes.append(bool(last_info.get("is_success", False)))

    env.close()
    return result


def compute_option_statistics(
    specs: list[str],
    invocations: list[OptionInvocation],
) -> dict[str, dict[str, Any]]:
    """Groups option invocations by specification and computes summary statistics."""
    grouped: dict[str, list[OptionInvocation]] = {spec: [] for spec in specs}
    for inv in invocations:
        if inv.spec in grouped:
            grouped[inv.spec].append(inv)

    stats: dict[str, dict[str, Any]] = {}
    for spec, records in grouped.items():
        if records:
            rewards = [r.reward for r in records]
            lengths = [r.steps for r in records]
            successes = [r.is_success for r in records]
            l_gains = [
                r.lambda_config["L_gain"]
                for r in records
                if "L_gain" in r.lambda_config
            ]
            k_steeps = [
                r.lambda_config["k_steepness"]
                for r in records
                if "k_steepness" in r.lambda_config
            ]
            eps_margins = [
                r.lambda_config["eps_margin"]
                for r in records
                if "eps_margin" in r.lambda_config
            ]

            stats[spec] = {
                "count": len(records),
                "mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
                "success_rate": float(np.mean(successes) * 100.0),
                "mean_steps": float(np.mean(lengths)),
                "mean_L": float(np.mean(l_gains)) if l_gains else None,
                "mean_k": float(np.mean(k_steeps)) if k_steeps else None,
                "mean_eps": float(np.mean(eps_margins)) if eps_margins else None,
            }
        else:
            stats[spec] = {
                "count": 0,
                "mean_reward": None,
                "std_reward": None,
                "success_rate": None,
                "mean_steps": None,
                "mean_L": None,
                "mean_k": None,
                "mean_eps": None,
            }
    return stats


def format_latex_table(
    specs: list[str],
    cpc_stats: dict[str, dict[str, Any]],
    abl_stats: dict[str, dict[str, Any]] | None,
) -> str:
    """Formats comparison metrics into a LaTeX tabular string."""
    rows: list[str] = [
        r"\begin{table}[tb]",
        r"    \centering",
        r"    \caption{Empirical Suboptimality Gap and Parameter Selection Across Reach-Avoid Option Slices on Fetch.}",
        r"    \label{tab:ablation_fetch}",
        r"    \rowcolors{2}{gray!25}{white}",
        r"    \resizebox{\columnwidth}{!}{%",
        r"    \begin{tabular}{l|cc|cc|c|c}",
        r"        \toprule",
        r"        \multirow{2}{*}{Option Specification $\phi$} & \multicolumn{2}{c|}{Option Return} & \multicolumn{2}{c|}{Success Rate (\%)} & Subopt. Gap & CPC Params \\",
        r"        & Optimized $V^*$ & Composed $V^{\compol}$ & Optimized & Composed & $\hat{D}(\phi)$ & $\langle \bar{L}, \bar{k}, \bar{\epsilon} \rangle$ \\",
        r"        \midrule",
    ]

    for spec in specs:
        spec_clean = spec.replace("&", r"\wedge").replace("!", r"\neg ")
        c_stat = cpc_stats.get(spec, {})
        c_ret = (
            f"{c_stat['mean_reward']:.2f} $\\pm$ {c_stat['std_reward']:.2f}"
            if c_stat.get("mean_reward") is not None
            else "--"
        )
        c_sr = f"{c_stat['success_rate']:.1f}" if c_stat.get("success_rate") is not None else "--"

        params_str = "--"
        if c_stat.get("mean_L") is not None:
            params_str = (
                f"$\\langle {c_stat['mean_L']:.1f}, "
                f"{c_stat['mean_k']:.1f}, "
                f"{c_stat['mean_eps']:.2f} \\rangle$"
            )

        if abl_stats and spec in abl_stats and abl_stats[spec].get("mean_reward") is not None:
            a_stat = abl_stats[spec]
            a_ret = f"{a_stat['mean_reward']:.2f} $\\pm$ {a_stat['std_reward']:.2f}"
            a_sr = f"{a_stat['success_rate']:.1f}"
            gap_val = float(a_stat["mean_reward"]) - float(c_stat["mean_reward"])
            gap_str = f"{gap_val:+.2f}"
        else:
            a_ret = "--"
            a_sr = "--"
            gap_str = "--"

        rows.append(
            f"        ${spec_clean}$ & {a_ret} & {c_ret} & {a_sr} & {c_sr} & {gap_str} & {params_str} \\\\"
        )

    rows.extend(
        [
            r"        \midrule",
            r"        \textbf{Pretraining Cost} & \multicolumn{2}{c|}{$90\,\mathrm{M}$ transitions} & \multicolumn{2}{c|}{$\mathbf{10\,\mathrm{M}}$ transitions} & \multicolumn{2}{c}{$\mathbf{9\times}$ Sample Reduction} \\",
            r"        \bottomrule",
            r"    \end{tabular}%",
            r"    }",
            r"\end{table}",
        ]
    )
    return "\n".join(rows)


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    output_dir = Path(_OUTPUT_DIR.value)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = normalize_device(_DEVICE.value)

    with Path(_SPEC_PATH.value).open("r", encoding="utf-8") as f:
        specs_data = json.load(f)
    specs: list[str] = specs_data["specifications"]

    logging.info("Candidate specifications loaded: %d specs", len(specs))

    # 1. Evaluate CPC Hierarchical Agent
    cpc_model_path = Path(_CPC_MODEL_PATH.value)
    if not cpc_model_path.exists():
        raise FileNotFoundError(f"CPC model checkpoint not found at: {cpc_model_path}")

    cpc_result = evaluate_hierarchical_agent(
        agent_name="CPC_Hierarchical",
        config_path=Path(_CPC_CONFIG_PATH.value),
        model_path=cpc_model_path,
        n_episodes=_N_EVAL_EPISODES.value,
        seed_base=_SEED_BASE.value,
        device=device,
    )

    cpc_option_stats = compute_option_statistics(specs, cpc_result.option_invocations)

    logging.info("==================================================")
    logging.info("CPC Hierarchical Agent Evaluation Summary:")
    logging.info("  Mean Return:   %.2f +/- %.2f", cpc_result.mean_return, cpc_result.std_return)
    logging.info("  Success Rate:  %.1f%%", cpc_result.success_rate)
    logging.info("  Mean Steps:    %.1f primitive steps", cpc_result.mean_length)
    logging.info("  Option Invocations Logged: %d", len(cpc_result.option_invocations))
    logging.info("==================================================")

    # 2. Evaluate Ablation (Pretrained Subpolicies) Agent if model checkpoint exists
    ablation_model_path = Path(_ABLATION_MODEL_PATH.value)
    ablation_result: AgentEvalResult | None = None
    ablation_option_stats: dict[str, dict[str, Any]] | None = None

    if ablation_model_path.exists():
        logging.info("Evaluating Ablation Agent from %s...", ablation_model_path)
        ablation_result = evaluate_hierarchical_agent(
            agent_name="Ablation_Pretrained",
            config_path=Path(_ABLATION_CONFIG_PATH.value),
            model_path=ablation_model_path,
            n_episodes=_N_EVAL_EPISODES.value,
            seed_base=_SEED_BASE.value,
            device=device,
        )
        ablation_option_stats = compute_option_statistics(
            specs, ablation_result.option_invocations
        )
        logging.info("==================================================")
        logging.info("Ablation Agent Evaluation Summary:")
        logging.info(
            "  Mean Return:   %.2f +/- %.2f",
            ablation_result.mean_return,
            ablation_result.std_return,
        )
        logging.info("  Success Rate:  %.1f%%", ablation_result.success_rate)
        logging.info(
            "  Mean Steps:    %.1f primitive steps", ablation_result.mean_length
        )
        logging.info(
            "  Option Invocations Logged: %d", len(ablation_result.option_invocations)
        )
        logging.info("==================================================")
    else:
        logging.warning(
            "Ablation model checkpoint not found at %s.\n"
            "Launch training on ICC with:\n"
            "  ./cluster/queue_experiment.sh campaign cluster/manifests/ablation_fetch_meta.txt",
            ablation_model_path,
        )

    # 3. Generate and export LaTeX comparison table
    latex_table = format_latex_table(specs, cpc_option_stats, ablation_option_stats)
    table_path = output_dir / "ablation_table.tex"
    table_path.write_text(latex_table, encoding="utf-8")
    logging.info("Exported LaTeX table to: %s", table_path)

    # 4. Save JSON summary metrics
    summary_data = {
        "cpc": {
            "mean_return": cpc_result.mean_return,
            "std_return": cpc_result.std_return,
            "success_rate": cpc_result.success_rate,
            "mean_length": cpc_result.mean_length,
            "option_stats": cpc_option_stats,
        },
        "ablation": {
            "mean_return": ablation_result.mean_return if ablation_result else None,
            "std_return": ablation_result.std_return if ablation_result else None,
            "success_rate": ablation_result.success_rate if ablation_result else None,
            "mean_length": ablation_result.mean_length if ablation_result else None,
            "option_stats": ablation_option_stats,
        },
    }
    summary_path = output_dir / "ablation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logging.info("Exported summary JSON to: %s", summary_path)


if __name__ == "__main__":
    app.run(main)
