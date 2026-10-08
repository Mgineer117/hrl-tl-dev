## Plot a band plot filled between the convolved max and min curves of the given set of reward
## time-series with the convolved mean curve

import os

import pandas as pd
import seaborn as sns

from hrl_tl.eval.reward import (
    EvalDataConfig,
    SB3EvalFileReader,
    SB3FileReaderConfig,
    WandbCSVEvalFileReader,
    WandbCSVReaderConfig,
)

if __name__ == "__main__":
    # Configuration parameters
    num_reps: int = 10
    convolve_window: int = 20

    # List of evaluation data configurations to plot
    eval_data_configs = [
        EvalDataConfig(
            name="TL-HRL (pretrained subpolicies)",
            num_replicates=5,
            max_timesteps=500_000,
            data_points=400,
            smooth_window=5,
            start_timestep=4_900_000,
            file_reader_class=SB3EvalFileReader,
            eval_file_config=SB3FileReaderConfig(
                models_dir="out/fourroom/ltl_hl/3.a.q/hl_policy_fourroom_500.0K",
                ind_eval_file="rep_{rep_id}/eval/evaluations.npz",
            ),
        ),
        EvalDataConfig(
            name="PPO",
            num_replicates=10,
            max_timesteps=5_500_000,
            data_points=200,
            smooth_window=2,
            file_reader_class=SB3EvalFileReader,
            eval_file_config=SB3FileReaderConfig(
                models_dir="out/poc/test/2.b/2.b.q/fourroom_ppo_test_5.0M",
                ind_eval_file="rep_{rep_id}/eval/evaluations.npz",
            ),
        ),
        EvalDataConfig(
            name="HIRO",
            num_replicates=7,
            max_timesteps=5_500_000,
            data_points=200,
            smooth_window=2,
            file_reader_class=WandbCSVEvalFileReader,
            eval_file_config=WandbCSVReaderConfig(
                file_path="out/fourroom/hiro/4.a/data/hiro_rewards.csv"
            ),
        ),
    ]

    # Use the first config's directory for saving the plot
    plot_path: str = (
        "out/plots/fourroom/baseline/fourroom_reward_conv_baselines.png"
    )

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
        linewidth=2,
        height=3,
        aspect=1.5,
    )

    # Customize the plot
    g.set_axis_labels("Training Timesteps", "Evaluation Reward", fontsize=12)

    # Remove legend title if multiple methods are plotted
    # Access the legend through matplotlib and remove the title
    sns.move_legend(g, loc="upper center", frameon=True)
    if g.legend:
        g.legend.set_title("")
        # Add translucent background to the legend
        g.legend.get_frame().set_facecolor("white")
        g.legend.get_frame().set_alpha(0.8)
    # sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))

    # Format x-axis to show large numbers nicely
    g.ax.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))

    # Save the plot
    g.figure.tight_layout()
    g.savefig(plot_path, dpi=300, bbox_inches="tight")

    print(f"Plot saved to: {plot_path}")

    # Print summary statistics for each method
    for config in eval_data_configs:
        method_data = df[df["method"] == config.name]
        timestep_range = f"{method_data['timestep'].min():.0f} - {method_data['timestep'].max():.0f}"
        print(f"{config.name} - Timestep range: {timestep_range}")
