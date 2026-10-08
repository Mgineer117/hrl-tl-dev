"""
Train Optimal Reach-Avoid Policy for SimpleGrid Environment

This script trains a tabular policy to reach the top-right goal (index 1)
while avoiding obstacles at different positions (indices 0, 2, or 3),
for comparison with the CPC composed policy.
"""

import argparse
import os

import numpy as np
from tqdm import tqdm

from hrl_tl.envs.simple_grid import SimpleGridEnv
from hrl_tl.eval.simple_grid_utils import (
    evaluate_reach_avoid_policy,
    save_reach_avoid_animation,
)
from hrl_tl.rl.policy.tabular_policy import TabularGoalConditionedPolicy
from hrl_tl.wrappers.reach_avoid import ReachAvoidWrapper


class SimpleGridTopRightEnv(SimpleGridEnv):
    """SimpleGrid environment with fixed top-right goal."""

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.goal_index = 1
        obs[2] = 1
        return obs, info


def train_reach_avoid_policy(
    env,
    policy,
    n_episodes=10000,
    gamma=0.99,
    max_steps=50,
    eval_interval=500,
    save_dir="models/simple_grid_reach_avoid",
):
    """
    Train the reach-avoid policy using REINFORCE.
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    success_rates = []
    collision_rates = []

    print(f"Training reach-avoid policy for {n_episodes} episodes...")

    for episode in tqdm(range(n_episodes)):
        # Reset environment
        obs, info = env.reset()
        state = obs[:2]  # (x, y)
        goal = obs[2]  # Always 1 (top-right)

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
            success_rate, collision_rate = evaluate_reach_avoid_policy(
                env, policy, n_eval_episodes=100
            )
            success_rates.append(success_rate)
            collision_rates.append(collision_rate)

            print(f"\nEpisode {episode + 1}/{n_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Length: {avg_length:.2f}")
            print(f"  Success Rate: {success_rate:.2%}")
            print(f"  Collision Rate: {collision_rate:.2%}")

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
        collision_rates=collision_rates,
    )
    print(f"\nTraining metrics saved to {metrics_path}")

    return (
        policy,
        episode_rewards,
        episode_lengths,
        success_rates,
        collision_rates,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train optimal reach-avoid policy on SimpleGrid"
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=100000,
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
        default=10000,
        help="Episodes between evaluation",
    )
    parser.add_argument(
        "--obstacle-idx",
        type=int,
        default=None,
        help="Obstacle index (0, 2, or 3). If None, train for all obstacles",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="out/simple_grid/reach_avoid",
        help="Model save directory (subdirectories created for each obstacle)",
    )
    parser.add_argument(
        "--save-animation",
        action="store_true",
        default=True,
        help="Save animation after training",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)

    # Get base environment to access goal positions
    base_env = SimpleGridEnv()
    goal_pos = base_env.goal_positions[1]  # Fixed goal at index 1 (top-right)

    # Determine which obstacles to train for
    if args.obstacle_idx is not None:
        obstacle_indices = [args.obstacle_idx]
    else:
        # Train for all obstacles except the goal (index 1)
        obstacle_indices = [2, 3, 0]

    # Train for each obstacle
    for obs_idx in obstacle_indices:
        obstacle_pos = base_env.goal_positions[obs_idx]

        print("\n" + "=" * 80)
        print(
            f"TRAINING REACH-AVOID POLICY: Goal {1} (top-right), Obstacle {obs_idx}"
        )
        print(f"Goal position: {goal_pos}, Obstacle position: {obstacle_pos}")
        print("=" * 80)

        # Create wrapped environment
        env = ReachAvoidWrapper(
            SimpleGridTopRightEnv(),
            goal_pos=goal_pos,
            obstacle_pos=obstacle_pos,
        )

        # Create policy
        # Note: We still use goal index for compatibility, but it's always 1
        policy = TabularGoalConditionedPolicy(
            n_states_x=3,
            n_states_y=3,
            n_goals=len(base_env.goal_positions),
            n_actions=4,
            lr=args.lr,
        )

        # Create save directory for this obstacle
        save_dir = os.path.join(args.save_dir, f"obstacle_{obs_idx}")

        # Train policy
        policy, rewards, lengths, success_rates, collision_rates = (
            train_reach_avoid_policy(
                env=env,
                policy=policy,
                n_episodes=args.n_episodes,
                gamma=args.gamma,
                max_steps=args.max_steps,
                eval_interval=args.eval_interval,
                save_dir=save_dir,
            )
        )

        # Final evaluation
        print("\n" + "=" * 60)
        print(f"FINAL EVALUATION - Obstacle {obs_idx}")
        print("=" * 60)
        success_rate, collision_rate = evaluate_reach_avoid_policy(
            env, policy, n_eval_episodes=1000
        )
        print(f"Success Rate (1000 episodes): {success_rate:.2%}")
        print(f"Collision Rate (1000 episodes): {collision_rate:.2%}")
        print("=" * 60)

        # Save animation if requested
        if args.save_animation:
            animation_path = os.path.join(save_dir, "reach_avoid_animation.gif")
            # Create wrapped environment with rgb_array render mode for animation
            anim_env = ReachAvoidWrapper(
                SimpleGridTopRightEnv(render_mode="rgb_array"),
                goal_pos=goal_pos,
                obstacle_pos=obstacle_pos,
            )
            save_reach_avoid_animation(
                anim_env, policy, 1, animation_path, max_steps=args.max_steps
            )
            anim_env.close()

        env.close()

    print("\n" + "=" * 80)
    print("ALL TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
