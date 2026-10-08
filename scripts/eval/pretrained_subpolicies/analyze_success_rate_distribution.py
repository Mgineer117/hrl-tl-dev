#!/usr/bin/env python3
"""
Script to analyze and plot the distribution of final_success_rate from convergence summary files.
"""

import json
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import BaseModel

# Set style for better plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class SuccessRateAnalysis(BaseModel):
    """Model for success rate analysis results."""

    total_specs: int
    analyzed_specs: int
    converged_specs: int
    success_rates: List[float]
    converged_success_rates: List[float]
    not_converged_success_rates: List[float]
    mean_success_rate: float
    median_success_rate: float
    std_success_rate: float
    min_success_rate: float
    max_success_rate: float
    mean_converged_success_rate: Optional[float] = None
    mean_not_converged_success_rate: Optional[float] = None


def load_convergence_summary(file_path: str) -> Dict[str, Any]:
    """
    Load convergence summary from JSON file.

    Args:
        file_path: Path to the convergence summary JSON file

    Returns:
        Dictionary containing the convergence summary data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Convergence summary file not found: {file_path}"
        )

    with open(file_path, "r") as f:
        data = json.load(f)

    return data


def extract_success_rates(
    convergence_data: Dict[str, Any],
) -> SuccessRateAnalysis:
    """
    Extract success rates from convergence data and compute statistics.

    Args:
        convergence_data: Dictionary containing convergence analysis results

    Returns:
        SuccessRateAnalysis object containing extracted data and statistics
    """
    specifications = convergence_data.get("specifications", {})
    summary = convergence_data.get("summary", {})

    # Extract success rates for all analyzed specifications
    success_rates = []
    converged_success_rates = []
    not_converged_success_rates = []

    for spec_name, spec_data in specifications.items():
        success_rate = spec_data.get("final_success_rate", 0.0)
        success_rates.append(success_rate)

        if spec_data.get("has_converged", False):
            converged_success_rates.append(success_rate)
        else:
            not_converged_success_rates.append(success_rate)

    # Compute statistics
    success_rates_array = np.array(success_rates)

    analysis = SuccessRateAnalysis(
        total_specs=summary.get("total_specifications", 0),
        analyzed_specs=len(success_rates),
        converged_specs=len(converged_success_rates),
        success_rates=success_rates,
        converged_success_rates=converged_success_rates,
        not_converged_success_rates=not_converged_success_rates,
        mean_success_rate=float(np.mean(success_rates_array))
        if len(success_rates) > 0
        else 0.0,
        median_success_rate=float(np.median(success_rates_array))
        if len(success_rates) > 0
        else 0.0,
        std_success_rate=float(np.std(success_rates_array))
        if len(success_rates) > 0
        else 0.0,
        min_success_rate=float(np.min(success_rates_array))
        if len(success_rates) > 0
        else 0.0,
        max_success_rate=float(np.max(success_rates_array))
        if len(success_rates) > 0
        else 0.0,
    )

    # Add means for converged and not converged groups
    if len(converged_success_rates) > 0:
        analysis.mean_converged_success_rate = float(
            np.mean(converged_success_rates)
        )

    if len(not_converged_success_rates) > 0:
        analysis.mean_not_converged_success_rate = float(
            np.mean(not_converged_success_rates)
        )

    return analysis


def create_success_rate_plots(
    analysis: SuccessRateAnalysis, save_path: Optional[str] = None
) -> None:
    """
    Create comprehensive plots for success rate distribution analysis.

    Args:
        analysis: SuccessRateAnalysis object containing the data
        save_path: Optional path to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Overall distribution histogram
    n, bins, patches = ax.hist(
        analysis.success_rates,
        bins=20,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )

    # Add count labels on top of each bin
    max_count = float(np.max(n))  # Convert to float for type safety
    for i, count in enumerate(n):
        if count > 0:  # Only show labels for non-empty bins
            bin_center = (bins[i] + bins[i + 1]) / 2
            ax.text(
                bin_center,
                count + max_count * 0.01,  # Slightly above the bar
                str(int(count)),
                ha="center",
                va="bottom",
                fontsize=12,
            )

    ax.axvline(
        analysis.mean_success_rate,
        color="red",
        linestyle="--",
        label=f"Mean: {analysis.mean_success_rate:.3f}",
    )
    ax.axvline(
        analysis.median_success_rate,
        color="orange",
        linestyle="--",
        label=f"Median: {analysis.median_success_rate:.3f}",
    )
    ax.set_xlabel("Final Success Rate", fontsize=16)
    ax.set_ylabel("Number of Specifications", fontsize=16)
    ax.set_title("Overall Success Rate Distribution", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Success rate distribution plots saved to: {save_path}")

    plt.show()


def print_analysis_summary(analysis: SuccessRateAnalysis) -> None:
    """
    Print a detailed summary of the success rate analysis.

    Args:
        analysis: SuccessRateAnalysis object containing the data
    """
    print("=" * 60)
    print("SUCCESS RATE DISTRIBUTION ANALYSIS")
    print("=" * 60)

    print(f"Total specifications: {analysis.total_specs}")
    print(f"Analyzed specifications: {analysis.analyzed_specs}")
    print(f"Converged specifications: {analysis.converged_specs}")
    print(
        f"Not converged specifications: {analysis.analyzed_specs - analysis.converged_specs}"
    )
    print()

    print("Overall Success Rate Statistics:")
    print(f"  Mean: {analysis.mean_success_rate:.4f}")
    print(f"  Median: {analysis.median_success_rate:.4f}")
    print(f"  Standard Deviation: {analysis.std_success_rate:.4f}")
    print(
        f"  Range: [{analysis.min_success_rate:.4f}, {analysis.max_success_rate:.4f}]"
    )
    print()

    if analysis.mean_converged_success_rate is not None:
        print(
            f"Converged specs mean success rate: {analysis.mean_converged_success_rate:.4f}"
        )

    if analysis.mean_not_converged_success_rate is not None:
        print(
            f"Not converged specs mean success rate: {analysis.mean_not_converged_success_rate:.4f}"
        )

    if (
        analysis.mean_converged_success_rate is not None
        and analysis.mean_not_converged_success_rate is not None
    ):
        diff = (
            analysis.mean_converged_success_rate
            - analysis.mean_not_converged_success_rate
        )
        print(f"Difference (converged - not converged): {diff:.4f}")
    print()

    # Percentiles
    if len(analysis.success_rates) > 0:
        percentiles = [10, 25, 50, 75, 90]
        print("Success Rate Percentiles:")
        for p in percentiles:
            value = np.percentile(analysis.success_rates, p)
            print(f"  {p}th percentile: {value:.4f}")
        print()

    # Count by success rate ranges
    ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    print("Specifications by Success Rate Range:")
    for low, high in ranges:
        if low == 0.8:  # Last range includes 1.0
            count = sum(
                1 for rate in analysis.success_rates if low <= rate <= high
            )
        else:
            count = sum(
                1 for rate in analysis.success_rates if low <= rate < high
            )
        percentage = (
            (count / analysis.analyzed_specs * 100)
            if analysis.analyzed_specs > 0
            else 0
        )
        print(f"  {low:.1f}-{high:.1f}: {count} specs ({percentage:.1f}%)")


def save_detailed_results(
    analysis: SuccessRateAnalysis,
    convergence_data: Dict[str, Any],
    output_file: str,
) -> None:
    """
    Save detailed results to a CSV file.

    Args:
        analysis: SuccessRateAnalysis object
        convergence_data: Original convergence data
        output_file: Path to save the CSV file
    """
    specifications = convergence_data.get("specifications", {})

    # Create detailed data
    detailed_data = []
    for spec_name, spec_data in specifications.items():
        detailed_data.append(
            {
                "Specification": spec_name,
                "Final_Success_Rate": spec_data.get("final_success_rate", 0.0),
                "Has_Converged": spec_data.get("has_converged", False),
                "Convergence_Point": spec_data.get("convergence_point", None),
                "Total_Episodes": spec_data.get("total_episodes", 0),
                "Final_Mean": spec_data.get("final_mean", 0.0),
                "Recent_CV": spec_data.get("recent_cv", 0.0),
            }
        )

    # Create DataFrame and sort by success rate
    df = pd.DataFrame(detailed_data)
    df = df.sort_values("Final_Success_Rate", ascending=False)

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Detailed results saved to: {output_file}")


def save_specs_by_success_rate_range(
    analysis: SuccessRateAnalysis,
    convergence_data: Dict[str, Any],
    output_file: str,
) -> None:
    """
    Save specifications grouped by success rate ranges to a CSV file.

    Args:
        analysis: SuccessRateAnalysis object
        convergence_data: Original convergence data
        output_file: Path to save the CSV file
    """
    specifications = convergence_data.get("specifications", {})

    # Define success rate ranges
    ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    range_labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]

    # Group specifications by success rate ranges
    range_data = []
    for i, (low, high) in enumerate(ranges):
        range_label = range_labels[i]
        specs_in_range = []

        for spec_name, spec_data in specifications.items():
            success_rate = spec_data.get("final_success_rate", 0.0)

            # Check if specification falls in this range
            if i == len(ranges) - 1:  # Last range includes 1.0
                in_range = low <= success_rate <= high
            else:
                in_range = low <= success_rate < high

            if in_range:
                specs_in_range.append(
                    {
                        "Success_Rate_Range": range_label,
                        "Specification": spec_name,
                        "Final_Success_Rate": success_rate,
                        "Has_Converged": spec_data.get("has_converged", False),
                        "Convergence_Point": spec_data.get(
                            "convergence_point", None
                        ),
                        "Total_Episodes": spec_data.get("total_episodes", 0),
                        "Final_Mean": spec_data.get("final_mean", 0.0),
                        "Recent_CV": spec_data.get("recent_cv", 0.0),
                    }
                )

        range_data.extend(specs_in_range)

    # Create DataFrame and sort by range and success rate
    df = pd.DataFrame(range_data)
    if not df.empty:
        df = df.sort_values(
            ["Success_Rate_Range", "Final_Success_Rate"],
            ascending=[True, False],
        )

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Specifications by success rate range saved to: {output_file}")

    # Print summary of each range
    print("\nSpecifications per success rate range:")
    for range_label in range_labels:
        count = (
            len(df[df["Success_Rate_Range"] == range_label])
            if not df.empty
            else 0
        )
        print(f"  {range_label}: {count} specifications")


def main():
    """Main function to run the success rate distribution analysis."""

    # Configuration
    convergence_summary_files = [
        "out/fourroom/ltl_ll/convergence_summary_cleared.json",
        # Add more files here if needed
    ]

    for summary_file in convergence_summary_files:
        if not os.path.exists(summary_file):
            print(f"Skipping {summary_file} (not found)")
            continue

        print(f"\n{'=' * 80}")
        print(f"Analyzing: {summary_file}")
        print(f"{'=' * 80}")

        try:
            # Load and analyze data
            convergence_data = load_convergence_summary(summary_file)
            analysis = extract_success_rates(convergence_data)

            # Print summary
            print_analysis_summary(analysis)

            # Create plots
            base_name = os.path.splitext(os.path.basename(summary_file))[0]
            plot_save_path = f"out/plots/fourroom/ltl_ll/{base_name}_success_rate_analysis.png"

            # Ensure output directory exists
            os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)

            create_success_rate_plots(analysis, plot_save_path)

            # Save detailed CSV
            csv_save_path = (
                f"out/fourroom/ltl_ll/{base_name}_success_rate_details.csv"
            )
            save_detailed_results(analysis, convergence_data, csv_save_path)

            # Save specifications by success rate range
            range_csv_save_path = (
                f"out/fourroom/ltl_ll/{base_name}_success_rate_ranges.csv"
            )
            save_specs_by_success_rate_range(
                analysis, convergence_data, range_csv_save_path
            )

        except Exception as e:
            print(f"Error analyzing {summary_file}: {str(e)}")


if __name__ == "__main__":
    main()
