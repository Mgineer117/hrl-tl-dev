"""
CPC Evaluation for SimpleGrid Environment

This script composes a policy to reach the top-right goal while avoiding the center goal
using Contrastive Policy Composition (CPC) with the tabular goal-conditioned policy.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from hrl_tl.envs.simple_grid import SimpleGridEnv
from hrl_tl.rl.policy.cpc import CPCTabularPolicy
from hrl_tl.rl.policy.tabular_policy import TabularGoalConditionedPolicy


def evaluate_cpc_policy(
    policy: CPCTabularPolicy,
    env: SimpleGridEnv,
    n_eval_episodes: int = 100,
    max_steps: int = 50,
) -> dict:
    """
    Evaluate the CPC composed policy.

    Args:
        policy: CPCTabularPolicy instance
        env: SimpleGrid environment
        n_eval_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode

    Returns:
        Dictionary with evaluation metrics
    """
    successes = 0
    collisions = 0  # Reaching the obstacle
    episode_lengths = []
    episode_rewards = []

    for _ in range(n_eval_episodes):
        obs, info = env.reset()

        # Only evaluate when the goal is top-right (goal_idx=1)
        goal = obs[2]
        if goal != 1:
            continue

        # Set agent to start from bottom-left corner (0, 0) in (x, y) coordinates
        env.agent_pos = np.array([0, 0])
        state = env.agent_pos
        steps = 0
        total_reward = 0

        for _ in range(max_steps):
            action = policy.sample_action(state, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

            state = next_obs[:2]

            # Check if reached obstacle (center)
            if np.array_equal(state, np.array([1, 1])):
                collisions += 1
                break

            if done:
                if terminated and goal == 1:  # Reached top-right goal
                    successes += 1
                break

        episode_lengths.append(steps)
        episode_rewards.append(total_reward)

    results = {
        "success_rate": successes / n_eval_episodes,
        "collision_rate": collisions / n_eval_episodes,
        "avg_episode_length": np.mean(episode_lengths),
        "avg_episode_reward": np.mean(episode_rewards),
    }

    return results


def save_cpc_animation(
    policy: CPCTabularPolicy,
    save_path: str,
    max_steps: int = 50,
    n_attempts: int = 20,
):
    """
    Save animation showing CPC policy behavior with all probability distributions.

    Args:
        policy: CPCTabularPolicy instance
        save_path: Path to save the animation
        max_steps: Maximum steps per episode
        n_attempts: Number of attempts to find episode with goal 1
    """
    env = SimpleGridEnv(render_mode="rgb_array")

    # Try to get an episode with goal 1 (top-right)
    frames = []
    goal_probs_list = []
    obstacle_probs_list = []
    composed_probs_list = []
    lambda_list = []
    actions_taken = []
    states_list = []
    episode_found = False

    for attempt in range(n_attempts):
        obs, info = env.reset()
        goal = obs[2]

        if goal == 1:  # Top-right goal
            episode_found = True

            # Set agent to start from bottom-left corner (2, 0)
            env.agent_pos = np.array([0, 0])
            state = env.agent_pos

            # Capture initial frame
            frames.append(env.render())
            probs = policy.get_action_probs(state)
            goal_probs_list.append(policy.last_goal_probs)
            obstacle_probs_list.append(policy.last_obstacle_probs)
            composed_probs_list.append(policy.last_composed_probs)
            lambda_list.append(policy.last_lambda)
            states_list.append(state.copy())

            for step in range(max_steps):
                action = policy.sample_action(state, deterministic=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Capture frame
                frames.append(env.render())
                state = next_obs[:2]

                probs = policy.get_action_probs(state)
                goal_probs_list.append(policy.last_goal_probs)
                obstacle_probs_list.append(policy.last_obstacle_probs)
                composed_probs_list.append(policy.last_composed_probs)
                lambda_list.append(policy.last_lambda)
                actions_taken.append(action)
                states_list.append(state.copy())

                # Terminate if reached goal or obstacle
                if (
                    done
                    or np.array_equal(state, np.array([1, 1]))
                    or np.array_equal(state, np.array([2, 2]))
                ):
                    # # Add pause frames
                    # for _ in range(5):
                    #     frames.append(env.render())
                    #     goal_probs_list.append(policy.last_goal_probs)
                    #     obstacle_probs_list.append(policy.last_obstacle_probs)
                    #     composed_probs_list.append(policy.last_composed_probs)
                    #     lambda_list.append(policy.last_lambda)
                    #     actions_taken.append(action)
                    #     states_list.append(state.copy())
                    break

            actions_taken.append(None)
            break

    env.close()

    if not episode_found:
        print(
            f"Warning: Could not find episode with goal 1 after {n_attempts} attempts"
        )
        return

    # Create animation with three subplots
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)
    ax_env = fig.add_subplot(gs[0])
    ax_components = fig.add_subplot(gs[1])
    ax_composed = fig.add_subplot(gs[2])

    ax_env.axis("off")
    im = ax_env.imshow(frames[0])

    action_names = ["Up", "Down", "Left", "Right"]

    def update(frame_idx):
        # Update environment
        im.set_array(frames[frame_idx])
        state = states_list[frame_idx]
        action_taken = actions_taken[frame_idx]
        ax_env.set_title(
            f"Step {frame_idx}/{len(frames) - 1}\nPos: ({state[0]}, {state[1]})",
            fontsize=12,
            pad=10,
        )

        # Plot component probabilities (goal and obstacle)
        ax_components.clear()
        goal_probs = goal_probs_list[frame_idx]
        obstacle_probs = obstacle_probs_list[frame_idx]
        lambda_val = lambda_list[frame_idx]

        x = np.arange(len(action_names))
        width = 0.35

        bars1 = ax_components.bar(
            x - width / 2,
            goal_probs,
            width,
            label="Goal (Top-Right)",
            alpha=0.7,
            color="green",
        )
        bars2 = ax_components.bar(
            x + width / 2,
            obstacle_probs,
            width,
            label="Obstacle (Center)",
            alpha=0.7,
            color="red",
        )

        ax_components.set_ylabel("Probability", fontsize=11)
        ax_components.set_title(
            f"Component Policies (λ={lambda_val:.3f})", fontsize=12, pad=10
        )
        ax_components.set_xticks(x)
        ax_components.set_xticklabels(action_names)
        ax_components.set_ylim(0, 1)
        ax_components.legend(fontsize=9)
        ax_components.grid(axis="y", alpha=0.3, linestyle="--")

        # Add values on bars
        for bar in bars1:
            height = bar.get_height()
            ax_components.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        for bar in bars2:
            height = bar.get_height()
            ax_components.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Plot composed probabilities
        ax_composed.clear()
        composed_probs = composed_probs_list[frame_idx]

        colors = ["#1f77b4"] * len(action_names)
        if action_taken is not None:
            colors[action_taken] = "#ff7f0e"

        bars = ax_composed.bar(
            action_names,
            composed_probs,
            color=colors,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )

        if action_taken is not None:
            bars[action_taken].set_alpha(1.0)
            bars[action_taken].set_linewidth(3)

        ax_composed.set_ylabel("Probability", fontsize=11)
        ax_composed.set_title("CPC Composed Policy", fontsize=12, pad=10)
        ax_composed.set_ylim(0, 1)
        ax_composed.grid(axis="y", alpha=0.3, linestyle="--")

        # Add values on bars
        for i, (bar, prob) in enumerate(zip(bars, composed_probs)):
            height = bar.get_height()
            label = f"{prob:.3f}"
            if i == action_taken:
                label = f"{prob:.3f} ✓"
            ax_composed.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold" if i == action_taken else "normal",
            )

        return [im]

    anim = FuncAnimation(
        fig, update, frames=len(frames), interval=1000, blit=False
    )

    # Save as GIF
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    writer = PillowWriter(fps=0.5)
    anim.save(save_path, writer=writer)
    plt.close(fig)

    print(f"CPC animation saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CPC composed policy on SimpleGrid"
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        default="out/simple_grid/models/policy_final.npz",
        help="Path to trained policy",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--L-gain",
        type=float,
        default=3.0,
        help="Lambda weight gain parameter",
    )
    parser.add_argument(
        "--k-steepness",
        type=float,
        default=2.0,
        help="Lambda weight steepness parameter",
    )
    parser.add_argument(
        "--eps-margin",
        type=float,
        default=0.5,
        help="Lambda weight margin parameter",
    )
    parser.add_argument(
        "--save-animation",
        action="store_true",
        default=True,
        help="Save animation of CPC policy",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="out/simple_grid/cpc",
        help="Directory to save results",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Load base policy
    print(f"Loading policy from {args.policy_path}")
    env = SimpleGridEnv()
    base_policy = TabularGoalConditionedPolicy(
        n_states_x=3, n_states_y=3, n_goals=len(env.goal_positions), n_actions=4
    )
    base_policy.load(args.policy_path)

    # Create CPC composed policy
    # Goal: reach top-right (index 1)
    # Obstacle: avoid center (index 0)
    cpc_policy = CPCTabularPolicy(
        base_policy=base_policy,
        goal_idx=1,  # Top-right
        obstacle_idx=0,  # Center (to avoid)
        L_gain=args.L_gain,
        k_steepness=args.k_steepness,
        eps_margin=args.eps_margin,
    )

    print("\n" + "=" * 60)
    print("CPC POLICY COMPOSITION")
    print("=" * 60)
    print(f"Goal: Top-Right (index 1)")
    print(f"Obstacle to Avoid: Center (index 0)")
    print(
        f"Lambda parameters: L_gain={args.L_gain}, k={args.k_steepness}, ε={args.eps_margin}"
    )
    print("=" * 60)

    # Evaluate CPC policy
    env = SimpleGridEnv()
    print(f"\nEvaluating CPC policy on {args.n_eval_episodes} episodes...")
    results = evaluate_cpc_policy(
        cpc_policy, env, n_eval_episodes=args.n_eval_episodes
    )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Success Rate (reach goal): {results['success_rate']:.2%}")
    print(f"Collision Rate (hit obstacle): {results['collision_rate']:.2%}")
    print(f"Avg Episode Length: {results['avg_episode_length']:.2f}")
    print(f"Avg Episode Reward: {results['avg_episode_reward']:.2f}")
    print("=" * 60)

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    results_path = os.path.join(args.save_dir, "evaluation_results.npz")
    np.savez(
        results_path,
        **results,
        L_gain=args.L_gain,
        k_steepness=args.k_steepness,
        eps_margin=args.eps_margin,
    )
    print(f"\nResults saved to {results_path}")

    # Save animation
    if args.save_animation:
        animation_path = os.path.join(args.save_dir, "cpc_policy_animation.gif")
        print(f"\nGenerating CPC policy animation...")
        save_cpc_animation(cpc_policy, animation_path)

    env.close()


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
