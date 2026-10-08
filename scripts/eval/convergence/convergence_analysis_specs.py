import json
import os
from typing import Any

from gym_tl_tools import replace_special_characters
from pydantic import BaseModel, Field

from hrl_tl.eval.conv import analyze_convergence


class MissingEvaluation(BaseModel):
    """Model for tracking missing evaluation files."""

    spec_id: int
    specification: str
    model_name: str
    eval_path: str
    error: str | None = None


class SpecificationResult(BaseModel):
    """Model for individual specification analysis results."""

    spec_id: int
    specification: str
    model_name: str
    has_converged: bool
    convergence_point: int | None = None
    total_episodes: int
    final_mean: float
    final_std: float
    final_success_rate: float
    recent_cv: float
    trend_slope: float
    trend_r_squared: float
    variance_ratio: float


class ConvergenceSummary(BaseModel):
    """Model for convergence analysis summary statistics."""

    total_specifications: int
    analyzed: int = 0
    converged: int = 0
    not_converged: int = 0
    missing_evaluations: int = 0
    convergence_threshold: float
    convergence_window: int


class ConvergenceResults(BaseModel):
    """Model for complete convergence analysis results."""

    summary: ConvergenceSummary
    specifications: dict[str, SpecificationResult] = Field(default_factory=dict)
    missing_evaluations: list[MissingEvaluation] = Field(default_factory=list)
    converged_specs: list[SpecificationResult] = Field(default_factory=list)
    not_converged_specs: list[SpecificationResult] = Field(default_factory=list)


def check_convergence_for_all_specs(
    formulae_file_path: str = "assets/formulae/fourroom/all_formulae_1_cla_1_max_pred.json",
    base_eval_path: str = "out/fourroom/ltl_ll/ll_policies",
    model_prefix: str = "fourroom_tl_ppo_rand",
    output_file: str = "convergence_summary.json",
    convergence_threshold: float = 0.1,
    convergence_window: int = 50,
) -> None:
    """
    Check convergence for all specifications from the formulae file.

    Args:
        formulae_file_path: Path to JSON file containing specifications
        base_eval_path: Base path where model evaluation files are stored
        model_prefix: Prefix used for model directory names
        output_file: Output file to save convergence summary
        convergence_threshold: CV threshold for convergence detection
        convergence_window: Window size for convergence detection
    """

    # Load specifications from JSON file
    print(f"Loading specifications from: {formulae_file_path}")
    try:
        with open(formulae_file_path, "r") as f:
            spec_data: dict[str, Any] = json.load(f)

        specifications: list[str] = spec_data["specifications"]
        num_specs: int = spec_data["num_specifications"]
        predicates: list[str] = spec_data["predicates"]

        print(f"Found {num_specs} specifications")
        print(f"Predicates: {predicates}")

    except FileNotFoundError:
        print(f"Error: Could not find formulae file at {formulae_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in formulae file {formulae_file_path}")
        return

    # Results storage using Pydantic models
    convergence_results = ConvergenceResults(
        summary=ConvergenceSummary(
            total_specifications=num_specs,
            convergence_threshold=convergence_threshold,
            convergence_window=convergence_window,
        )
    )

    print(f"\nAnalyzing convergence for {num_specs} specifications...")
    print("=" * 60)

    for spec_id, tl_spec in enumerate(specifications):
        # Create model name using replace_special_characters
        sanitized_spec = replace_special_characters(tl_spec)
        model_name = f"{model_prefix}_{sanitized_spec}"

        # Construct path to evaluation file
        eval_path = os.path.join(
            base_eval_path, model_name, "eval", "evaluations.npz"
        )

        print(f"[{spec_id + 1}/{num_specs}] Checking: {tl_spec}")
        print(f"  Model: {model_name}")
        print(f"  Eval path: {eval_path}")

        # Check if evaluation file exists
        if not os.path.exists(eval_path):
            print("  ❌ Evaluation file not found")
            missing_eval = MissingEvaluation(
                spec_id=spec_id,
                specification=tl_spec,
                model_name=model_name,
                eval_path=eval_path,
            )
            convergence_results.missing_evaluations.append(missing_eval)
            convergence_results.summary.missing_evaluations += 1
            continue

        try:
            # Analyze convergence using the existing function
            print("  📊 Analyzing convergence...")
            results = analyze_convergence(
                file_path=eval_path,
                convergence_threshold=convergence_threshold,
                convergence_window=convergence_window,
                plot=False,  # Disable plotting for batch analysis
                save_plot=False,
            )

            # Extract key metrics
            has_converged = results.has_converged
            convergence_point = results.convergence_point
            total_episodes = results.total_episodes
            metrics = results.metrics

            # Store results
            spec_result = SpecificationResult(
                spec_id=spec_id,
                specification=tl_spec,
                model_name=model_name,
                has_converged=has_converged,
                convergence_point=convergence_point,
                total_episodes=total_episodes,
                final_mean=metrics.final_mean,
                final_std=metrics.final_std,
                final_success_rate=results.final_success_rate,
                recent_cv=metrics.recent_cv,
                trend_slope=metrics.trend_slope,
                trend_r_squared=metrics.trend_r_squared,
                variance_ratio=metrics.variance_ratio,
            )

            convergence_results.specifications[tl_spec] = spec_result
            convergence_results.summary.analyzed += 1

            if has_converged:
                convergence_results.summary.converged += 1
                convergence_results.converged_specs.append(spec_result)
                print(
                    f"  ✅ CONVERGED at episode {convergence_point}/{total_episodes}"
                )
                print(
                    f"     Final performance: {metrics.final_mean:.3f} ± {metrics.final_std:.3f}"
                )
            else:
                convergence_results.summary.not_converged += 1
                convergence_results.not_converged_specs.append(spec_result)
                print(f"  ❌ NOT CONVERGED ({total_episodes} episodes)")
                print(
                    f"     Final performance: {metrics.final_mean:.3f} ± {metrics.final_std:.3f}"
                )
                print(f"     Recent CV: {metrics.recent_cv:.3f}")

        except Exception as e:
            print(f"  ⚠️  Error analyzing {eval_path}: {str(e)}")
            missing_eval = MissingEvaluation(
                spec_id=spec_id,
                specification=tl_spec,
                model_name=model_name,
                eval_path=eval_path,
                error=str(e),
            )
            convergence_results.missing_evaluations.append(missing_eval)
            convergence_results.summary.missing_evaluations += 1

        print()

    # Save results to JSON file
    print(f"Saving results to: {output_file}")
    with open(output_file, "w") as f:
        json.dump(convergence_results.model_dump(), f, indent=2, default=str)

    # Print summary
    print("=" * 60)
    print("CONVERGENCE ANALYSIS SUMMARY")
    print("=" * 60)
    summary = convergence_results.summary
    print(f"Total specifications: {summary.total_specifications}")
    print(f"Successfully analyzed: {summary.analyzed}")
    print(f"Missing evaluation files: {summary.missing_evaluations}")
    print()
    print(f"Converged models: {summary.converged}")
    print(f"Not converged models: {summary.not_converged}")

    if summary.analyzed > 0:
        convergence_rate = summary.converged / summary.analyzed * 100
        print(f"Convergence rate: {convergence_rate:.1f}%")

    print()

    # Show some examples
    if convergence_results.converged_specs:
        print("Examples of CONVERGED specifications:")
        for spec in convergence_results.converged_specs[:3]:
            print(
                f"  • {spec.specification} (converged at episode {spec.convergence_point})"
            )

    if convergence_results.not_converged_specs:
        print("\nExamples of NOT CONVERGED specifications:")
        for spec in convergence_results.not_converged_specs:
            print(f"  • {spec.specification} (CV: {spec.recent_cv:.3f})")

    if convergence_results.missing_evaluations:
        print("\nFirst few missing evaluation files:")
        for missing in convergence_results.missing_evaluations:
            print(f"  • {missing.specification} -> {missing.model_name}")

    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    # Default paths based on the user's request
    check_convergence_for_all_specs(
        formulae_file_path="assets/formulae/fourroom/all_formulae_1_cla_1_max_pred.json",
        base_eval_path="out/fourroom/ltl_ll/ll_policies",
        model_prefix="fourroom_tl_ppo_pos_2_rand_obs_cleared",
        output_file="out/fourroom/ltl_ll/eval/convergence_summary_cleared_2_obs.json",
        convergence_threshold=0.3,
        convergence_window=50,
    )
