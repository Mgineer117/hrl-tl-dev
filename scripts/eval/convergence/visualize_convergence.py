#!/usr/bin/env python3
"""
Script to visualize convergence analysis results.
"""

import json
import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd


def visualize_convergence_results(results_file: str) -> None:
    """
    Create visualizations from convergence analysis results.

    Args:
        results_file: Path to the JSON file containing convergence results
    """

    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return

    # Load results
    with open(results_file, "r") as f:
        results: Dict[str, Any] = json.load(f)

    print(f"Visualizing results from: {results_file}")

    # Extract summary
    summary = results["summary"]
    print(f"Total analyzed: {summary['analyzed']}")
    print(f"Converged: {summary['converged']}")
    print(f"Not converged: {summary['not_converged']}")
    print(f"Missing evaluations: {summary['missing_evaluations']}")

    if summary["analyzed"] == 0:
        print("No analyzed specifications to visualize.")
        return

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        f"Convergence Analysis Results\\n({results_file})", fontsize=16
    )

    # 1. Convergence summary pie chart
    ax1 = axes[0, 0]
    labels = ["Converged", "Not Converged", "Missing Eval"]
    sizes = [
        summary["converged"],
        summary["not_converged"],
        summary["missing_evaluations"],
    ]
    colors = ["lightgreen", "lightcoral", "lightgray"]

    ax1.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90
    )
    ax1.set_title("Convergence Summary")

    # 2. Performance distribution for converged models
    ax2 = axes[0, 1]
    if results["converged_specs"]:
        final_means = [
            spec["final_mean"] for spec in results["converged_specs"]
        ]
        ax2.hist(
            final_means,
            bins=min(len(final_means), 20),
            alpha=0.7,
            color="green",
        )
        ax2.set_title("Final Performance Distribution\\n(Converged Models)")
        ax2.set_xlabel("Final Mean Return")
        ax2.set_ylabel("Count")
    else:
        ax2.text(
            0.5,
            0.5,
            "No converged\\nmodels",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.set_title("Final Performance Distribution\\n(Converged Models)")

    # 3. Convergence point distribution
    ax3 = axes[1, 0]
    if results["converged_specs"]:
        convergence_points = [
            spec["convergence_point"] for spec in results["converged_specs"]
        ]
        ax3.hist(
            convergence_points,
            bins=min(len(convergence_points), 20),
            alpha=0.7,
            color="blue",
        )
        ax3.set_title("Convergence Point Distribution")
        ax3.set_xlabel("Episode Number")
        ax3.set_ylabel("Count")
    else:
        ax3.text(
            0.5,
            0.5,
            "No converged\\nmodels",
            ha="center",
            va="center",
            transform=ax3.transAxes,
        )
        ax3.set_title("Convergence Point Distribution")

    # 4. CV distribution for non-converged models
    ax4 = axes[1, 1]
    if results["not_converged_specs"]:
        cvs = [
            spec["recent_cv"]
            for spec in results["not_converged_specs"]
            if spec["recent_cv"] != float("inf")
        ]
        if cvs:
            ax4.hist(cvs, bins=min(len(cvs), 20), alpha=0.7, color="red")
            ax4.set_title("Coefficient of Variation\\n(Non-Converged Models)")
            ax4.set_xlabel("Recent CV")
            ax4.set_ylabel("Count")
            ax4.axvline(
                x=summary["convergence_threshold"],
                color="black",
                linestyle="--",
                label=f"Threshold: {summary['convergence_threshold']}",
            )
            ax4.legend()
        else:
            ax4.text(
                0.5,
                0.5,
                "No valid CV\\nvalues",
                ha="center",
                va="center",
                transform=ax4.transAxes,
            )
            ax4.set_title("Coefficient of Variation\\n(Non-Converged Models)")
    else:
        ax4.text(
            0.5,
            0.5,
            "All models\\nconverged!",
            ha="center",
            va="center",
            transform=ax4.transAxes,
        )
        ax4.set_title("Coefficient of Variation\\n(Non-Converged Models)")

    plt.tight_layout()

    # Save the plot
    plot_filename = results_file.replace(".json", "_visualization.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to: {plot_filename}")
    plt.show()

    # Create a detailed summary table
    if results["specifications"]:
        create_summary_table(
            results, results_file.replace(".json", "_table.csv")
        )


def create_summary_table(results: Dict[str, Any], output_file: str) -> None:
    """Create a CSV table with detailed results."""

    specs_data = []
    for spec_name, spec_data in results["specifications"].items():
        specs_data.append(
            {
                "Specification": spec_name,
                "Model_Name": spec_data["model_name"],
                "Converged": spec_data["has_converged"],
                "Convergence_Point": spec_data["convergence_point"]
                if spec_data["has_converged"]
                else None,
                "Total_Episodes": spec_data["total_episodes"],
                "Final_Mean": round(spec_data["final_mean"], 4),
                "Final_Std": round(spec_data["final_std"], 4),
                "Recent_CV": round(spec_data["recent_cv"], 4)
                if spec_data["recent_cv"] != float("inf")
                else "inf",
                "Trend_Slope": round(spec_data["trend_slope"], 6),
                "R_Squared": round(spec_data["trend_r_squared"], 4),
                "Variance_Ratio": round(spec_data["variance_ratio"], 4)
                if spec_data["variance_ratio"] != float("inf")
                else "inf",
            }
        )

    df = pd.DataFrame(specs_data)
    df = df.sort_values(["Converged", "Final_Mean"], ascending=[False, False])
    df.to_csv(output_file, index=False)
    print(f"Detailed table saved to: {output_file}")

    # Print top performers
    print("\\nTop 5 performing converged models:")
    converged_df = df[df["Converged"]]
    if not converged_df.empty:
        for idx, row in converged_df.head().iterrows():
            print(
                f"  {row['Specification']}: {row['Final_Mean']:.3f} ± {row['Final_Std']:.3f}"
            )
    else:
        print("  No converged models found.")


def main():
    """Visualize convergence results for all available result files."""

    result_files = [
        "out/fourroom/ltl_ll/convergence_summary.json",
        # "convergence_summary_stay.json",
        # "convergence_summary.json",
    ]

    for result_file in result_files:
        if os.path.exists(result_file):
            print(f"\\n{'=' * 60}")
            visualize_convergence_results(result_file)
        else:
            print(f"Skipping {result_file} (not found)")


if __name__ == "__main__":
    main()
