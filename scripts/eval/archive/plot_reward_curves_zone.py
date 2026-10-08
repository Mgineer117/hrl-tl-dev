## Plot a band plot filled between the convolved max and min curves of the given set of reward
## time-series with the convolved mean curve

import os

import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from hrl_tl.eval.reward import (
    EvalDataConfig,
    SB3EvalFileReader,
    SB3FileReaderConfig,
)

if __name__ == "__main__":
    # Configuration parameters
    num_reps: int = 10
    convolve_window: int = 20
    plot_title: str | None = "2 Subtasks"
    # Use the first config's directory for saving the plot
    plot_path: str = (
        "out/plots/zone/baseline/zone_reward_conv_baselines_st2.png"
    )

    # List of evaluation data configurations to plot
    eval_data_configs = [
        EvalDataConfig(
            name="TL-HRL (SDSAC)",
            num_replicates=1,
            max_timesteps=2_200_000,
            data_points=400,
            smooth_window=4,
            start_timestep=10_000_000,
            file_reader_class=SB3EvalFileReader,
            eval_file_config=SB3FileReaderConfig(
                models_dir="out/zone/ltl_hl/8.d.a/hl_policy_zone_20.0M",
                ind_eval_file="rep_{rep_id}/eval/evaluations.npz",
            ),
        ),
        EvalDataConfig(
            name="TL-HRL (PPO)",
            num_replicates=1,
            max_timesteps=10_000_000,
            data_points=400,
            smooth_window=4,
            start_timestep=10_000_000,
            file_reader_class=SB3EvalFileReader,
            eval_file_config=SB3FileReaderConfig(
                models_dir="out/zone/ltl_hl/8.b.i/hl_policy_zone_20.0M",
                ind_eval_file="rep_{rep_id}/eval/evaluations.npz",
            ),
        ),
        EvalDataConfig(
            name="GCLTL-RL",
            num_replicates=1,
            max_timesteps=10_000_000,
            data_points=400,
            smooth_window=4,
            start_timestep=10_000_000,
            file_reader_class=SB3EvalFileReader,
            eval_file_config=SB3FileReaderConfig(
                models_dir="out/zone/ltl_hl/5.d.i/hl_policy_zone_40.0M",
                ind_eval_file="rep_{rep_id}/eval/evaluations.npz",
            ),
        ),
        # EvalDataConfig(
        #     name="ALLO Option HRL",
        #     num_replicates=5,
        #     max_timesteps=10_000_000,
        #     data_points=400,
        #     smooth_window=4,
        #     start_timestep=10_000_000 + 42 * 100_000,
        #     file_reader_class=SB3EvalFileReader,
        #     eval_file_config=SB3FileReaderConfig(
        #         models_dir="out/zone/allo/meta/7.c.a/allo_meta_zone_10.0M",
        #         ind_eval_file="rep_{rep_id}/eval/evaluations.npz",
        #     ),
        # ),
        # EvalDataConfig(
        #     name="HIRO",
        #     num_replicates=5,
        #     max_timesteps=15_00_000,
        #     data_points=200,
        #     smooth_window=2,
        #     file_reader_class=SB3EvalFileReader,
        #     eval_file_config=SB3FileReaderConfig(
        #         models_dir="out/zone/hiro/4.b.a/hiro_zone_15.0M",
        #         ind_eval_file="rep_{rep_id}/eval/evaluations.npz",
        #     ),
        # ),
        # EvalDataConfig(
        #     name="PPO",
        #     num_replicates=1,
        #     max_timesteps=16_000_000,
        #     data_points=200,
        #     smooth_window=2,
        #     file_reader_class=SB3EvalFileReader,
        #     eval_file_config=SB3FileReaderConfig(
        #         models_dir="out/zone/ppo/6.b.a/ppo_zone_20.0M",
        #         ind_eval_file="rep_{rep_id}/eval/evaluations.npz",
        #     ),
        # ),
    ]

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    # Prepare data for seaborn relplot
    # Create a DataFrame with all individual runs from all configurations
    plot_data = []

    for config in eval_data_configs:
        print(f"\nProcessing configuration: {config.name}")

        # Load evaluation data from multiple replicates
        reward_data, timestep_data = config.file_reader_class.read_eval_files(
            config.eval_file_config, config.num_replicates
        )

        # Ensure all data have the same length
        reward_data, timestep_data = (
            config.file_reader_class.format_data_length(
                reward_data,
                timestep_data,
                config.max_timesteps,
                config.data_points,
            )
        )

        # Add data from this configuration to the plot data
        plot_data += config.file_reader_class.get_plot_data(
            reward_data,
            timestep_data,
            config.smooth_window,
            config.name,
            config.start_timestep,
        )

    df = pd.DataFrame(plot_data)

    # Convert timesteps to millions
    df["timestep"] = df["timestep"] / 1e6

    # Create the plot using seaborn relplot
    sns.set_theme(style="whitegrid", font="Times New Roman")

    # Use relplot to create the plot with confidence intervals
    g = sns.relplot(
        data=df,
        x="timestep",
        y="reward",
        hue="method",  # Use method name for different colors/legend
        kind="line",
        estimator="mean",
        linewidth=1,
        height=3,
        aspect=1.5,
    )

    # Customize the plot
    g.set_axis_labels("Training Timesteps", "Evaluation Reward", fontsize=12)

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

    # Format x-axis to show large numbers in millions
    def millions_formatter(x, pos):
        return f"{x:.0f}M"

    g.ax.xaxis.set_major_formatter(FuncFormatter(millions_formatter))

    # Save the plot
    g.figure.tight_layout()
    g.savefig(plot_path, dpi=300, bbox_inches="tight")

    print(f"Plot saved to: {plot_path}")

    # Print summary statistics for each method
    for config in eval_data_configs:
        method_data = df[df["method"] == config.name]
        timestep_range = f"{method_data['timestep'].min():.0f} - {method_data['timestep'].max():.0f}"
        print(f"{config.name} - Timestep range: {timestep_range}")
