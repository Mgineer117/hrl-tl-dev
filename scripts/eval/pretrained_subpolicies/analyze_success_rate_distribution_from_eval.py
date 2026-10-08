#!/usr/bin/env python3
"""
Script to analyze and plot the distribution of final_success_rate from convergence summary files.
"""

import os

import matplotlib.pyplot as plt
import seaborn as sns

from hrl_tl.eval.success import (
    create_success_rate_plots,
    extract_success_rates,
    generate_all_training_configs,
    print_analysis_summary,
    save_detailed_results,
)

# Set style for better plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def main():
    """Main function to run the success rate distribution analysis."""

    # Configuration - paths to specification and training config files
    # spec_file_path = "assets/formulae/fourroom/all_formulae_1_cla_1_max_pred.json"
    spec_file_path = (
        "assets/formulae/fourroom/all_formulae_1_cla_1_max_pred_core.json"
    )

    training_config_path = "configs/fourroom/train/ll_policies/2.c.n_pretrain_tl_ppo_tensor_2_obs.yaml"
    filename: str = "2.c.n_core_success_rate_details"
    # spec_file_path = "assets/formulae/fourroom/all_primitives.json"
    # training_config_path = (
    #     "configs/fourroom/train/ll_policies/2.c.g_pretrain_tl_ppo_rand_2_obs.yaml"
    # )
    # base_name: str = "2.c.g"

    # Output directory configuration
    output_base_dir = "out/fourroom/ltl_ll/eval"
    # plots_base_dir = "out/plots/fourroom/ltl_ll/eval"
    # output_base_dir = "out/fourroom/primitives/eval"
    plots_base_dir = output_base_dir

    print(f"\n{'=' * 80}")
    print("SUCCESS RATE DISTRIBUTION ANALYSIS")
    print(f"{'=' * 80}")
    print(f"Specification file: {spec_file_path}")
    print(f"Training config: {training_config_path}")
    print(f"{'=' * 80}")

    try:
        # Check if required files exist
        if not os.path.exists(spec_file_path):
            print(f"Error: Specification file not found: {spec_file_path}")
            return

        if not os.path.exists(training_config_path):
            print(
                f"Error: Training config file not found: {training_config_path}"
            )
            return

        # Generate all training configurations
        print("Generating training configurations...")
        training_configs = generate_all_training_configs(
            spec_file_path, training_config_path
        )
        print(f"Generated {len(training_configs)} training configurations")

        # Extract success rates and detailed metrics
        print("Extracting success rates from evaluation results...")
        analysis, detailed_metrics = extract_success_rates(training_configs)

        # Print analysis summary
        print_analysis_summary(analysis)

        # Create plots
        plot_save_path = os.path.join(plots_base_dir, f"{filename}.png")

        # Ensure output directories exist
        os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
        os.makedirs(output_base_dir, exist_ok=True)

        # Create and save success rate plots
        create_success_rate_plots(analysis, plot_save_path)

        # Save detailed results to CSV
        csv_save_path = os.path.join(output_base_dir, f"{filename}.csv")
        save_detailed_results(detailed_metrics, csv_save_path)

        print(f"\n{'=' * 80}")
        print("ANALYSIS COMPLETE")
        print(f"{'=' * 80}")
        print(f"Results saved to: {csv_save_path}")
        print(f"Plots saved to: {plot_save_path}")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
