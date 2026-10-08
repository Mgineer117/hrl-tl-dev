"""
Compare CPC Composed Policy with Optimal Reach-Avoid Policy

This script compares:
1. CPC composed policy (combining goal-reaching policies)
2. Optimal reach-avoid policy (trained directly on the task)

Comparison includes:
- Action probabilities for each state
- Value estimates for each state
- Visual comparison of policies
- Quantitative metrics (KL divergence, value differences)

Supports multiple obstacle scenarios (indices 0, 2, 3) with fixed goal at index 1.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from hrl_tl.envs.simple_grid import SimpleGridEnv
from hrl_tl.rl.policy.cpc import CPCTabularPolicy
from hrl_tl.rl.policy.tabular_policy import TabularGoalConditionedPolicy
from hrl_tl.wrappers.reach_avoid import ReachAvoidWrapper


def compute_kl_divergence(p, q, epsilon=1e-10):
    """
    Compute KL divergence KL(p||q) = sum(p * log(p/q))

    Args:
        p: Target distribution
        q: Approximating distribution
        epsilon: Small constant for numerical stability
    """
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    return np.sum(p * np.log(p / q))


def compare_policies_statewise(cpc_policy: CPCTabularPolicy, optimal_policy):
    """
    Compare CPC and optimal policies for all states.

    Returns:
        comparison_data: Dictionary containing comparison metrics
    """
    grid_size = 3
    goal_idx = 1  # Top-right

    comparison_data = {
        "states": [],
        "cpc_probs": [],
        "optimal_probs": [],
        "cpc_values": [],
        "optimal_values": [],
        "kl_divergence": [],
        "value_diff": [],
    }

    for x in range(grid_size):
        for y in range(grid_size):
            state = np.array([x, y])

            if x == 0 and y == 0:
                pass

            # Get CPC action probabilities
            cpc_probs = cpc_policy.get_action_probs(state)

            # Get optimal policy action probabilities
            optimal_probs = optimal_policy.get_action_probs(state, goal_idx)

            # Get values
            cpc_value = cpc_policy.base_policy.get_value(state, goal_idx)
            optimal_value = optimal_policy.get_value(state, goal_idx)

            # Compute metrics
            kl_div = compute_kl_divergence(optimal_probs, cpc_probs)
            value_diff = abs(optimal_value - cpc_value)

            comparison_data["states"].append(state)
            comparison_data["cpc_probs"].append(cpc_probs)
            comparison_data["optimal_probs"].append(optimal_probs)
            comparison_data["cpc_values"].append(cpc_value)
            comparison_data["optimal_values"].append(optimal_value)
            comparison_data["kl_divergence"].append(kl_div)
            comparison_data["value_diff"].append(value_diff)

    return comparison_data


def print_comparison_table(comparison_data):
    """Print detailed comparison table."""
    action_names = ["Up", "Down", "Left", "Right"]

    print("\n" + "=" * 120)
    print("POLICY COMPARISON: CPC vs Optimal Reach-Avoid")
    print("=" * 120)

    for i, state in enumerate(comparison_data["states"]):
        x, y = state[0], state[1]
        print(f"\nState ({x}, {y}):")
        print("-" * 120)

        # Action probabilities
        print("Action Probabilities:")
        print(f"  {'Action':<10} {'CPC':<12} {'Optimal':<12} {'Diff':<12}")
        print("  " + "-" * 48)

        cpc_probs = comparison_data["cpc_probs"][i]
        optimal_probs = comparison_data["optimal_probs"][i]

        for j, action_name in enumerate(action_names):
            diff = abs(cpc_probs[j] - optimal_probs[j])
            print(
                f"  {action_name:<10} {cpc_probs[j]:.6f}     {optimal_probs[j]:.6f}     {diff:.6f}"
            )

        # Values and metrics
        print(f"\nValue Functions:")
        print(f"  CPC Value:     {comparison_data['cpc_values'][i]:.4f}")
        print(f"  Optimal Value: {comparison_data['optimal_values'][i]:.4f}")
        print(f"  Value Diff:    {comparison_data['value_diff'][i]:.4f}")

        print(f"\nKL Divergence: {comparison_data['kl_divergence'][i]:.6f}")

    print("\n" + "=" * 120)
    print("SUMMARY STATISTICS")
    print("=" * 120)
    print(
        f"Average KL Divergence: {np.mean(comparison_data['kl_divergence']):.6f}"
    )
    print(
        f"Max KL Divergence:     {np.max(comparison_data['kl_divergence']):.6f}"
    )
    print(
        f"Average Value Diff:    {np.mean(comparison_data['value_diff']):.4f}"
    )
    print(f"Max Value Diff:        {np.max(comparison_data['value_diff']):.4f}")
    print("=" * 120)


def visualize_value_comparison(comparison_data, save_path):
    """
    Visualize value functions side-by-side as heatmaps.
    """
    grid_size = 3

    # Create value grids
    cpc_values_grid = np.zeros((grid_size, grid_size))
    optimal_values_grid = np.zeros((grid_size, grid_size))
    value_diff_grid = np.zeros((grid_size, grid_size))

    for i, state in enumerate(comparison_data["states"]):
        x, y = state[0], state[1]
        cpc_values_grid[y, x] = comparison_data["cpc_values"][i]
        optimal_values_grid[y, x] = comparison_data["optimal_values"][i]
        value_diff_grid[y, x] = comparison_data["value_diff"][i]

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # CPC values
    im1 = axes[0].imshow(cpc_values_grid, cmap="RdYlGn", origin="lower")
    axes[0].set_title(
        "CPC Composed Policy\nValue Function", fontsize=12, fontweight="bold"
    )
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")

    # Add value text
    for i in range(grid_size):
        for j in range(grid_size):
            text = axes[0].text(
                j,
                i,
                f"{cpc_values_grid[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )

    plt.colorbar(im1, ax=axes[0])

    # Optimal values
    im2 = axes[1].imshow(optimal_values_grid, cmap="RdYlGn", origin="lower")
    axes[1].set_title(
        "Optimal Reach-Avoid Policy\nValue Function",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")

    # Add value text
    for i in range(grid_size):
        for j in range(grid_size):
            text = axes[1].text(
                j,
                i,
                f"{optimal_values_grid[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )

    plt.colorbar(im2, ax=axes[1])

    # Value difference
    im3 = axes[2].imshow(value_diff_grid, cmap="Reds", origin="lower")
    axes[2].set_title(
        "Absolute Value Difference\n|CPC - Optimal|",
        fontsize=12,
        fontweight="bold",
    )
    axes[2].set_xlabel("X")
    axes[2].set_ylabel("Y")

    # Add value text
    for i in range(grid_size):
        for j in range(grid_size):
            text = axes[2].text(
                j,
                i,
                f"{value_diff_grid[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )

    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Value comparison visualization saved to {save_path}")


def visualize_policy_comparison(comparison_data, save_path, cpc_policy=None):
    """
    Visualize action probability differences as heatmaps for each action.

    Args:
        comparison_data: Dictionary containing comparison metrics
        save_path: Path to save visualization
        cpc_policy: CPC policy to get goal/obstacle positions from
    """
    grid_size = 3
    action_names = ["Up", "Down", "Left", "Right"]

    # Get goal and obstacle positions from environment or use defaults
    env = SimpleGridEnv()
    goal_idx = cpc_policy.goal_idx if cpc_policy else 1
    obstacle_idx = cpc_policy.obstacle_idx if cpc_policy else 0
    goal_pos = tuple(env.goal_positions[goal_idx])
    obstacle_pos = tuple(env.goal_positions[obstacle_idx])

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    for action_idx, action_name in enumerate(action_names):
        # CPC probabilities
        cpc_probs_grid = np.full((grid_size, grid_size), np.nan)
        optimal_probs_grid = np.full((grid_size, grid_size), np.nan)
        diff_probs_grid = np.full((grid_size, grid_size), np.nan)

        for i, state in enumerate(comparison_data["states"]):
            x, y = state[0], state[1]

            # Skip goal and obstacle states
            if (x, y) == goal_pos or (x, y) == obstacle_pos:
                continue

            cpc_probs_grid[y, x] = comparison_data["cpc_probs"][i][action_idx]
            optimal_probs_grid[y, x] = comparison_data["optimal_probs"][i][
                action_idx
            ]
            diff_probs_grid[y, x] = abs(
                cpc_probs_grid[y, x] - optimal_probs_grid[y, x]
            )

        # CPC
        im1 = axes[0, action_idx].imshow(
            cpc_probs_grid, cmap="Blues", vmin=0, vmax=1, origin="lower"
        )
        axes[0, action_idx].set_title(
            f"{action_name}\nCPC", fontsize=11, fontweight="bold"
        )
        axes[0, action_idx].set_xlabel("X")
        if action_idx == 0:
            axes[0, action_idx].set_ylabel("Y")

        # Add probability text
        for i in range(grid_size):
            for j in range(grid_size):
                if not np.isnan(cpc_probs_grid[i, j]):
                    text = axes[0, action_idx].text(
                        j,
                        i,
                        f"{cpc_probs_grid[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="white"
                        if cpc_probs_grid[i, j] > 0.5
                        else "black",
                        fontsize=9,
                    )

        plt.colorbar(im1, ax=axes[0, action_idx])

        # Optimal
        im2 = axes[1, action_idx].imshow(
            optimal_probs_grid, cmap="Blues", vmin=0, vmax=1, origin="lower"
        )
        axes[1, action_idx].set_title(
            f"{action_name}\nOptimal", fontsize=11, fontweight="bold"
        )
        axes[1, action_idx].set_xlabel("X")
        if action_idx == 0:
            axes[1, action_idx].set_ylabel("Y")

        # Add probability text
        for i in range(grid_size):
            for j in range(grid_size):
                if not np.isnan(optimal_probs_grid[i, j]):
                    text = axes[1, action_idx].text(
                        j,
                        i,
                        f"{optimal_probs_grid[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="white"
                        if optimal_probs_grid[i, j] > 0.5
                        else "black",
                        fontsize=9,
                    )

        plt.colorbar(im2, ax=axes[1, action_idx])

        # Difference (|CPC - Optimal|)
        im3 = axes[2, action_idx].imshow(
            diff_probs_grid, cmap="Reds", vmin=0, vmax=1, origin="lower"
        )
        axes[2, action_idx].set_title(
            f"{action_name}\n|CPC - Optimal|", fontsize=11, fontweight="bold"
        )
        axes[2, action_idx].set_xlabel("X")
        if action_idx == 0:
            axes[2, action_idx].set_ylabel("Y")

        # Add difference text
        for i in range(grid_size):
            for j in range(grid_size):
                if not np.isnan(diff_probs_grid[i, j]):
                    text = axes[2, action_idx].text(
                        j,
                        i,
                        f"{diff_probs_grid[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="white"
                        if diff_probs_grid[i, j] > 0.5
                        else "black",
                        fontsize=9,
                    )

        plt.colorbar(im3, ax=axes[2, action_idx])

    plt.suptitle(
        "Action Probability Comparison: CPC vs Optimal",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Policy comparison visualization saved to {save_path}")


def visualize_argmax_actions(comparison_data, save_path, cpc_policy=None):
    """
    Visualize argmax actions as arrows on the grid for CPC and optimal policies.

    Args:
        comparison_data: Dictionary containing comparison metrics
        save_path: Path to save visualization
        cpc_policy: CPC policy to get goal/obstacle positions from
    """
    grid_size = 3
    action_names = ["Up", "Down", "Left", "Right"]

    # Action to direction vector mapping (in grid coordinates)
    action_to_arrow = {
        0: (0, 1),  # Up: increase y
        1: (0, -1),  # Down: decrease y
        2: (-1, 0),  # Left: decrease x
        3: (1, 0),  # Right: increase x
    }

    # Create figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Get goal and obstacle positions from environment or use defaults
    env = SimpleGridEnv()
    goal_idx = cpc_policy.goal_idx if cpc_policy else 1
    obstacle_idx = cpc_policy.obstacle_idx if cpc_policy else 0
    goal_pos = tuple(env.goal_positions[goal_idx])
    obstacle_pos = tuple(env.goal_positions[obstacle_idx])

    for ax_idx, (ax, policy_name) in enumerate(zip(axes, ["CPC", "Optimal"])):
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_aspect("equal")
        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)
        ax.set_title(
            f"{policy_name} Policy\nArgmax Actions",
            fontsize=13,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))

        # Draw grid cells
        for i in range(grid_size):
            for j in range(grid_size):
                # Draw cell boundary
                rect = plt.Rectangle(
                    (i - 0.5, j - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="black",
                    linewidth=1,
                )
                ax.add_patch(rect)

        # Mark goal and obstacle
        ax.plot(
            goal_pos[0],
            goal_pos[1],
            "g*",
            markersize=25,
            label="Goal",
            zorder=3,
        )
        ax.plot(
            obstacle_pos[0],
            obstacle_pos[1],
            "rX",
            markersize=20,
            label="Obstacle",
            zorder=3,
        )

        # Plot arrows for each state
        for i, state in enumerate(comparison_data["states"]):
            x, y = state[0], state[1]

            # Skip goal and obstacle states
            if (x, y) == goal_pos or (x, y) == obstacle_pos:
                continue

            # Get argmax action
            if ax_idx == 0:  # CPC
                probs = comparison_data["cpc_probs"][i]
            else:  # Optimal
                probs = comparison_data["optimal_probs"][i]

            argmax_action = np.argmax(probs)
            dx, dy = action_to_arrow[argmax_action]

            # Draw arrow
            ax.arrow(
                x,
                y,
                dx * 0.35,
                dy * 0.35,
                head_width=0.15,
                head_length=0.1,
                fc="blue",
                ec="blue",
                linewidth=2,
                zorder=2,
            )

            # Add action label
            action_label = action_names[argmax_action][0]  # First letter
            ax.text(
                x - 0.35,
                y - 0.35,
                action_label,
                fontsize=8,
                color="darkblue",
                fontweight="bold",
            )

        ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Argmax action comparison visualization saved to {save_path}")


def save_side_by_side_animation(
    cpc_policy, optimal_policy, obstacle_idx, save_path, max_steps=50
):
    """
    Create side-by-side animation comparing CPC and optimal policies.

    Args:
        cpc_policy: CPC composed policy
        optimal_policy: Optimal reach-avoid policy
        obstacle_idx: Index of the obstacle
        save_path: Path to save animation
        max_steps: Maximum steps per episode
    """
    # Get goal and obstacle positions
    env_base = SimpleGridEnv()
    goal_pos = env_base.goal_positions[1]  # Fixed goal at index 1
    obstacle_pos = env_base.goal_positions[obstacle_idx]

    # Create wrapped environment for rendering
    class SimpleGridTopRightEnv(SimpleGridEnv):
        def reset(self, **kwargs):
            obs, info = super().reset(**kwargs)
            self.goal_index = 1
            obs[2] = 1
            return obs, info

    env = ReachAvoidWrapper(
        SimpleGridTopRightEnv(render_mode="rgb_array"),
        goal_pos=goal_pos,
        obstacle_pos=obstacle_pos,
    )

    # Run both policies from the same starting position
    start_pos = np.array([0, 0])

    # Collect trajectories for both policies
    def collect_trajectory(policy, is_cpc=False):
        obs, info = env.reset()
        # Force starting position (environment already has goal fixed at 1)
        env.unwrapped.agent_pos = start_pos.copy()
        obs[:2] = start_pos.copy()

        frames = []
        probs_list = []
        actions_taken = []
        states_list = []

        state = start_pos.copy()
        goal_idx = 1

        frames.append(env.render())
        if is_cpc:
            probs = policy.get_action_probs(state)
        else:
            probs = policy.get_action_probs(state, goal_idx)
        probs_list.append(probs)
        states_list.append(state.copy())

        for step in range(max_steps):
            if is_cpc:
                action = policy.sample_action(state, deterministic=True)
                probs = policy.get_action_probs(state)
            else:
                probs = policy.get_action_probs(state, goal_idx)
                action = np.argmax(probs)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            frames.append(env.render())
            state = next_obs[:2]

            if is_cpc:
                probs = policy.get_action_probs(state)
            else:
                probs = policy.get_action_probs(state, goal_idx)

            probs_list.append(probs)
            actions_taken.append(action)
            states_list.append(state.copy())

            if done:
                break

        actions_taken.append(None)
        return frames, probs_list, actions_taken, states_list

    cpc_frames, cpc_probs, cpc_actions, cpc_states = collect_trajectory(
        cpc_policy, is_cpc=True
    )
    optimal_frames, optimal_probs, optimal_actions, optimal_states = (
        collect_trajectory(optimal_policy, is_cpc=False)
    )

    env.close()

    # Pad shorter trajectory
    max_len = max(len(cpc_frames), len(optimal_frames))

    while len(cpc_frames) < max_len:
        cpc_frames.append(cpc_frames[-1])
        cpc_probs.append(cpc_probs[-1])
        cpc_actions.append(cpc_actions[-1])
        cpc_states.append(cpc_states[-1])

    while len(optimal_frames) < max_len:
        optimal_frames.append(optimal_frames[-1])
        optimal_probs.append(optimal_probs[-1])
        optimal_actions.append(optimal_actions[-1])
        optimal_states.append(optimal_states[-1])

    # Create animation
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.3)

    ax_cpc_env = fig.add_subplot(gs[0])
    ax_cpc_probs = fig.add_subplot(gs[1])
    ax_optimal_env = fig.add_subplot(gs[2])
    ax_optimal_probs = fig.add_subplot(gs[3])

    ax_cpc_env.axis("off")
    ax_optimal_env.axis("off")

    im_cpc = ax_cpc_env.imshow(cpc_frames[0])
    im_optimal = ax_optimal_env.imshow(optimal_frames[0])

    action_names = ["Up", "Down", "Left", "Right"]

    def update(frame_idx):
        # CPC environment
        im_cpc.set_array(cpc_frames[frame_idx])
        cpc_state = cpc_states[frame_idx]
        ax_cpc_env.set_title(
            f"CPC Policy\nPos: ({cpc_state[0]}, {cpc_state[1]})",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )

        # CPC probabilities
        ax_cpc_probs.clear()
        cpc_prob = cpc_probs[frame_idx]
        cpc_action = cpc_actions[frame_idx]

        colors_cpc = [
            "#ff7f0e" if i == cpc_action else "#1f77b4" for i in range(4)
        ]
        bars_cpc = ax_cpc_probs.bar(
            action_names,
            cpc_prob,
            color=colors_cpc,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )

        if cpc_action is not None:
            bars_cpc[cpc_action].set_alpha(1.0)
            bars_cpc[cpc_action].set_linewidth(3)

        ax_cpc_probs.set_ylim(0, 1)
        ax_cpc_probs.set_ylabel("Probability", fontsize=11)
        ax_cpc_probs.set_title(
            "CPC Action Probs", fontsize=12, fontweight="bold"
        )
        ax_cpc_probs.grid(axis="y", alpha=0.3, linestyle="--")

        for i, (bar, prob) in enumerate(zip(bars_cpc, cpc_prob)):
            label = f"{prob:.2f}"
            if i == cpc_action:
                label += " ✓"
            ax_cpc_probs.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                label,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold" if i == cpc_action else "normal",
            )

        # Optimal environment
        im_optimal.set_array(optimal_frames[frame_idx])
        optimal_state = optimal_states[frame_idx]
        ax_optimal_env.set_title(
            f"Optimal Policy\nPos: ({optimal_state[0]}, {optimal_state[1]})",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )

        # Optimal probabilities
        ax_optimal_probs.clear()
        optimal_prob = optimal_probs[frame_idx]
        optimal_action = optimal_actions[frame_idx]

        colors_optimal = [
            "#ff7f0e" if i == optimal_action else "#1f77b4" for i in range(4)
        ]
        bars_optimal = ax_optimal_probs.bar(
            action_names,
            optimal_prob,
            color=colors_optimal,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )

        if optimal_action is not None:
            bars_optimal[optimal_action].set_alpha(1.0)
            bars_optimal[optimal_action].set_linewidth(3)

        ax_optimal_probs.set_ylim(0, 1)
        ax_optimal_probs.set_ylabel("Probability", fontsize=11)
        ax_optimal_probs.set_title(
            "Optimal Action Probs", fontsize=12, fontweight="bold"
        )
        ax_optimal_probs.grid(axis="y", alpha=0.3, linestyle="--")

        for i, (bar, prob) in enumerate(zip(bars_optimal, optimal_prob)):
            label = f"{prob:.2f}"
            if i == optimal_action:
                label += " ✓"
            ax_optimal_probs.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                label,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold" if i == optimal_action else "normal",
            )

        return [im_cpc, im_optimal]

    anim = FuncAnimation(fig, update, frames=max_len, interval=1000, blit=False)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    writer = PillowWriter(fps=0.5)
    anim.save(save_path, writer=writer)
    plt.close(fig)

    print(f"Side-by-side animation saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare CPC and optimal reach-avoid policies"
    )
    parser.add_argument(
        "--base-policy-path",
        type=str,
        default="out/simple_grid/models/policy_final.npz",
        help="Path to base goal-conditioned policy for CPC",
    )
    parser.add_argument(
        "--optimal-policy-dir",
        type=str,
        default="out/simple_grid/reach_avoid",
        help="Directory containing optimal policies (subdirs: obstacle_0, obstacle_2, obstacle_3)",
    )
    parser.add_argument(
        "--obstacle-idx",
        type=int,
        default=None,
        help="Obstacle index (0, 2, or 3). If None, compare all obstacles",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="out/simple_grid/comparison",
        help="Directory to save comparison results (subdirectories created for each obstacle)",
    )
    parser.add_argument(
        "--L-gain", type=float, default=3.0, help="CPC lambda gain parameter"
    )
    parser.add_argument(
        "--k-steepness",
        type=float,
        default=2.0,
        help="CPC lambda steepness parameter",
    )
    parser.add_argument(
        "--eps-margin",
        type=float,
        default=0.5,
        help="CPC lambda margin parameter",
    )

    args = parser.parse_args()

    # Load base policy
    print(f"Loading base policy from {args.base_policy_path}")
    env = SimpleGridEnv()
    base_policy = TabularGoalConditionedPolicy(
        n_states_x=3, n_states_y=3, n_goals=len(env.goal_positions), n_actions=4
    )
    base_policy.load(args.base_policy_path)

    # Determine which obstacles to compare
    if args.obstacle_idx is not None:
        obstacle_indices = [args.obstacle_idx]
    else:
        # Compare all obstacles except the goal (index 1)
        obstacle_indices = [3, 0, 2]

    # Compare for each obstacle
    for obs_idx in obstacle_indices:
        print("\n" + "=" * 80)
        print(f"COMPARING POLICIES: Goal 1 (top-right), Obstacle {obs_idx}")
        print("=" * 80)

        # Create CPC policy for this obstacle
        cpc_policy = CPCTabularPolicy(
            base_policy=base_policy,
            goal_idx=1,  # Top-right
            obstacle_idx=obs_idx,
            obstacle_pos=env.goal_positions[obs_idx],
            L_gain=args.L_gain,
            k_steepness=args.k_steepness,
            eps_margin=args.eps_margin,
        )

        # Load optimal policy for this obstacle
        optimal_policy_path = os.path.join(
            args.optimal_policy_dir, f"obstacle_{obs_idx}", "policy_final.npz"
        )
        print(f"Loading optimal policy from {optimal_policy_path}")
        optimal_policy = TabularGoalConditionedPolicy(
            n_states_x=3,
            n_states_y=3,
            n_goals=len(env.goal_positions),
            n_actions=4,
        )
        optimal_policy.load(optimal_policy_path)

        # Compare policies
        print("\nComparing policies across all states...")
        comparison_data = compare_policies_statewise(cpc_policy, optimal_policy)

        # Print comparison table
        print_comparison_table(comparison_data)

        # Create save directory for this obstacle
        save_dir = os.path.join(args.save_dir, f"obstacle_{obs_idx}")
        os.makedirs(save_dir, exist_ok=True)

        # Save comparison data
        comparison_path = os.path.join(save_dir, "comparison_data.npz")
        np.savez(
            comparison_path,
            states=comparison_data["states"],
            cpc_probs=comparison_data["cpc_probs"],
            optimal_probs=comparison_data["optimal_probs"],
            cpc_values=comparison_data["cpc_values"],
            optimal_values=comparison_data["optimal_values"],
            kl_divergence=comparison_data["kl_divergence"],
            value_diff=comparison_data["value_diff"],
        )
        print(f"\nComparison data saved to {comparison_path}")

        # Visualize value comparison
        value_viz_path = os.path.join(
            save_dir, f"value_comparison_{obs_idx}.png"
        )
        visualize_value_comparison(comparison_data, value_viz_path)

        # Visualize policy comparison
        policy_viz_path = os.path.join(
            save_dir, f"policy_comparison_{obs_idx}.png"
        )
        visualize_policy_comparison(
            comparison_data, policy_viz_path, cpc_policy
        )

        # Visualize argmax actions
        argmax_viz_path = os.path.join(
            save_dir, f"argmax_actions_comparison_{obs_idx}.png"
        )
        visualize_argmax_actions(comparison_data, argmax_viz_path, cpc_policy)

        # Create side-by-side animation
        animation_path = os.path.join(
            save_dir, f"side_by_side_animation_{obs_idx}.gif"
        )
        print("\nGenerating side-by-side animation...")
        save_side_by_side_animation(
            cpc_policy, optimal_policy, obs_idx, animation_path
        )

        print("\n" + "=" * 60)
        print(f"COMPARISON COMPLETE - Obstacle {obs_idx}")
        print("=" * 60)
        print(f"Results saved to: {save_dir}")
        print("=" * 60)

    print("\n" + "=" * 80)
    print("ALL COMPARISONS COMPLETE")
    print("=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {args.save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
