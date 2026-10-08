"""
Tabular Goal-Conditioned Policy Gradient for SimpleGrid Environment

This script implements a simple tabular policy gradient algorithm (REINFORCE)
for learning goal-conditioned policies on the SimpleGrid environment.
"""

import argparse
import os

import numpy as np
from tqdm import tqdm

from hrl_tl.envs.simple_grid import SimpleGridEnv
from hrl_tl.eval.simple_grid_utils import (
    evaluate_policy,
    save_animation_for_goal,
)
from hrl_tl.rl.policy.tabular_policy import TabularGoalConditionedPolicy


def train_policy(
    env,
    policy,
    n_episodes=10000,
    gamma=0.99,
    max_steps=50,
    eval_interval=500,
    save_dir="models/simple_grid",
):
    """
    Train the tabular policy using REINFORCE.

    Args:
        env: Gymnasium environment
        policy: TabularGoalConditionedPolicy instance
        n_episodes: Number of training episodes
        gamma: Discount factor
        max_steps: Maximum steps per episode
        eval_interval: Episodes between evaluation
        save_dir: Directory to save models
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    success_rates = []

    print(f"Training for {n_episodes} episodes...")

    for episode in tqdm(range(n_episodes)):
        # Reset environment
        obs, info = env.reset()
        state = obs[:2]  # (x, y)
        goal = obs[2]  # goal index

        # Collect trajectory
        trajectory = []
        episode_reward = 0
        done = False

        for step in range(max_steps):
            # Sample action
            action = policy.sample_action(state, goal)

            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            trajectory.append((state, goal, action, reward))
            episode_reward += reward

            # Update state
            state = next_obs[:2]

            if done:
                break

        # Update policy
        policy.update(trajectory, gamma=gamma)

        # Record metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(len(trajectory))

        # Evaluation
        if (episode + 1) % eval_interval == 0:
            avg_reward = np.mean(episode_rewards[-eval_interval:])
            avg_length = np.mean(episode_lengths[-eval_interval:])
            success_rate = evaluate_policy(env, policy, n_eval_episodes=100)
            success_rates.append(success_rate)

            print(f"\nEpisode {episode + 1}/{n_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Length: {avg_length:.2f}")
            print(f"  Success Rate: {success_rate:.2%}")

            # Save checkpoint
            checkpoint_path = os.path.join(
                save_dir, f"policy_ep{episode + 1}.npz"
            )
            policy.save(checkpoint_path)

    # Save final policy
    final_path = os.path.join(save_dir, "policy_final.npz")
    policy.save(final_path)

    # Save training metrics
    metrics_path = os.path.join(save_dir, "training_metrics.npz")
    np.savez(
        metrics_path,
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        success_rates=success_rates,
    )
    print(f"\nTraining metrics saved to {metrics_path}")

    return policy, episode_rewards, episode_lengths, success_rates


def save_animations_for_all_goals(policy, save_dir, max_steps=50):
    """
    Save animations for all goals.

    Args:
        policy: Trained TabularGoalConditionedPolicy
        save_dir: Directory to save animations
        max_steps: Maximum steps per episode
    """
    print("\nSaving animations for all goals...")

    for goal_idx in range(policy.n_goals):
        save_path = os.path.join(save_dir, f"goal_{goal_idx}_animation.gif")
        save_animation_for_goal(
            SimpleGridEnv, policy, goal_idx, save_path, max_steps=max_steps
        )

    print(f"\nAll animations saved to {save_dir}")


def visualize_policy(env: SimpleGridEnv, policy, n_episodes=5):
    """
    Visualize trained policy.

    Args:
        env: Gymnasium environment with render_mode='human'
        policy: Trained TabularGoalConditionedPolicy
        n_episodes: Number of episodes to visualize
    """
    for ep in range(n_episodes):
        obs, info = env.reset()
        state = obs[:2]
        goal = obs[2]
        done = False
        total_reward = 0

        print(f"\n=== Episode {ep + 1} ===")
        env.render()

        for step in range(50):
            action = policy.sample_action(state, goal)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            print(f"Step {step + 1}: Action={action}, Reward={reward:.1f}")
            env.render()

            state = next_obs[:2]

            if done:
                if terminated:
                    print(f"✓ Goal reached! Total reward: {total_reward:.1f}")
                else:
                    print(
                        f"✗ Episode truncated. Total reward: {total_reward:.1f}"
                    )
                break

        input("Press Enter for next episode...")


def main():
    parser = argparse.ArgumentParser(
        description="Train tabular goal-conditioned policy on SimpleGrid"
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=1000000,
        help="Number of training episodes",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor"
    )
    parser.add_argument(
        "--max-steps", type=int, default=50, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=50000,
        help="Episodes between evaluation",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="out/simple_grid/models",
        help="Model save directory",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize policy after training",
    )
    parser.add_argument(
        "--save-animations",
        default=True,
        action="store_true",
        help="Save animations for each goal",
    )
    parser.add_argument(
        "--load-policy",
        type=str,
        default=None,
        help="Path to load existing policy",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)

    # Create environment
    env = SimpleGridEnv()

    # Create policy
    policy = TabularGoalConditionedPolicy(
        n_states_x=3,
        n_states_y=3,
        n_goals=len(env.goal_positions),
        n_actions=4,
        lr=args.lr,
    )

    # Load existing policy if specified
    if args.load_policy:
        policy.load(args.load_policy)
    else:
        # Train policy
        policy, rewards, lengths, success_rates = train_policy(
            env=env,
            policy=policy,
            n_episodes=args.n_episodes,
            gamma=args.gamma,
            max_steps=args.max_steps,
            eval_interval=args.eval_interval,
            save_dir=args.save_dir,
        )

        # Final evaluation
        print("\n" + "=" * 50)
        print("FINAL EVALUATION")
        print("=" * 50)
        final_success_rate = evaluate_policy(env, policy, n_eval_episodes=1000)
        print(f"Success Rate (1000 episodes): {final_success_rate:.2%}")

    # Save animations if requested
    if args.save_animations:
        animation_dir = os.path.join(args.save_dir, "animations")
        save_animations_for_all_goals(
            policy, animation_dir, max_steps=args.max_steps
        )

    # Visualize if requested
    if args.visualize:
        env_render = SimpleGridEnv(render_mode="human")
        visualize_policy(env_render, policy, n_episodes=5)
        env_render.close()

    env.close()


if __name__ == "__main__":
    main()
