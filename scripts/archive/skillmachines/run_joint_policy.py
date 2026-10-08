import os
from typing import Any, Literal

import gym_multigrid
import gymnasium as gym
import imageio
import numpy as np
import torch
import yaml
from stable_baselines3 import DQN
from stable_baselines3.common.base_class import BaseAlgorithm
from torch import Tensor

if __name__ == "__main__":
    goals: tuple[Literal["goal", "lava", "hole"], ...] = (
        "goal",
        "lava",
        "hole",
    )
    combination: tuple[Literal[1, -1, 0], ...] = (1, -1, -1)
    alg_name: str = "dqn"
    alg_class: type[DQN] = DQN
    gpu_id: int = 0

    agent_pos: tuple[int, int] = (8, 11)

    config_dir: str = "configs/fourroom"
    env_config_file: str = "layout/full.yaml"

    model_save_dir: str = "out/poc/skillmachines/primitives"
    model_name: str = os.path.join(
        model_save_dir, "fourroom_primitives_{goal}", "final_model"
    )
    neg_model_name: str = os.path.join(
        model_save_dir, "fourroom_primitives_neg_{goal}", "final_model"
    )
    model_names: list[str] = [model_name.format(goal=goal) for goal in goals]
    neg_model_names: list[str] = [
        neg_model_name.format(goal=goal) for goal in goals
    ]
    max_model_name: str = model_name.format(goal="max")
    min_model_name: str = model_name.format(goal="min")

    with open(os.path.join(config_dir, env_config_file), "r") as f:
        env_kwargs: dict[str, Any] = yaml.safe_load(f)
    env_kwargs["layout_config"]["spawn_configs"][0]["agent"] = agent_pos
    env_kwargs["random_init_pos"] = False

    env = gym.make("multigrid-rooms-v0", **env_kwargs)

    models: list[DQN] = []

    for i, c in enumerate(combination):
        match c:
            case 1:
                model = alg_class.load(
                    model_names[i], env=env, device=f"cuda:{gpu_id}"
                )
                print(f"Loaded model {i} for {goals[i]}: {model_names[i]}")
            case -1:
                model = alg_class.load(
                    neg_model_names[i], env=env, device=f"cuda:{gpu_id}"
                )
                print(
                    f"Loaded negated model {i} for {goals[i]}: {neg_model_names[i]}"
                )
            case 0:
                print(f"Skipping model {i} for {goals[i]} as it is not used.")
                continue
            case _:
                raise ValueError(
                    f"Invalid combination value {c} for model {i}."
                )

        models.append(model)

    used_combination: tuple[int, ...] = tuple(c for c in combination if c != 0)

    max_model: DQN = alg_class.load(
        max_model_name, env=env, device=f"cuda:{gpu_id}"
    )
    min_model: DQN = alg_class.load(
        min_model_name, env=env, device=f"cuda:{gpu_id}"
    )

    num_actions: int = env.action_space.n  # type: ignore

    # Run an episode with the loaded models
    obs, _ = env.reset()
    terminated: bool = False
    truncated: bool = False
    frames = [env.render()]
    rewards: list[float] = []
    step: int = 0
    while not (terminated or truncated):
        print(f"Step {step}:")
        print("- Combine the Q values...")
        q_values_lists: list[Tensor] = []
        obs_tensor, _ = max_model.policy.obs_to_tensor(obs)
        for i, (model, comb) in enumerate(zip(models, used_combination)):
            with torch.no_grad():
                q_val_tmp: Tensor = model.policy.q_net(obs_tensor)
                # Flatten and normalize the Q-values
                q_values: Tensor = (
                    q_val_tmp.flatten()
                )  # / torch.norm(q_val_tmp.flatten())
            print(
                f"-- Model {i} Q-values: {[f'{val:.3f}' for val in q_values.tolist()]}"
            )
            q_values_lists.append(q_values)

        q_values_lists_tensor: Tensor = torch.stack(q_values_lists, dim=0)
        # Find max/min Q values for each action from `q_values_list_tensor`
        with torch.no_grad():
            max_q_values_tmp: Tensor = max_model.policy.q_net(obs_tensor)
            min_q_values_tmp: Tensor = min_model.policy.q_net(obs_tensor)
        max_q_values: Tensor = max_q_values_tmp.flatten()  # / torch.norm(
        #     max_q_values_tmp.flatten()
        # )
        min_q_values: Tensor = min_q_values_tmp.flatten()  # / torch.norm(
        #     min_q_values_tmp.flatten()
        # )
        print(
            f"-- Max Q-values: {[f'{val:.3f}' for val in max_q_values.tolist()]}"
        )
        print(
            f"-- Min Q-values: {[f'{val:.3f}' for val in min_q_values.tolist()]}"
        )
        used_q_values_lists: list[Tensor] = []
        for q_values, comb in zip(q_values_lists, used_combination):
            match comb:
                case 1:
                    used_q_values_lists.append(q_values)
                case -1:
                    # negated_q_values: Tensor = max_q_values + min_q_values - q_values
                    # used_q_values_lists.append(negated_q_values)
                    used_q_values_lists.append(q_values)
                case _:
                    raise ValueError(
                        f"Combination value {comb} is not allowed in used_combination."
                    )

        # Assume conjugation of the primitives
        used_q_values_lists_tensor: Tensor = torch.stack(
            used_q_values_lists, dim=0
        )
        print(
            f"-- Used Q-values: {[[f'{val:.3f}' for val in row] for row in used_q_values_lists_tensor.tolist()]}"
        )
        conjugated_q_values: Tensor = torch.min(
            used_q_values_lists_tensor, dim=0
        ).values
        print(
            f"-- Conjugated Q-values: {[f'{val:.3f}' for val in conjugated_q_values.tolist()]}"
        )
        action = torch.argmax(conjugated_q_values).item()
        print(f"-- Action chosen: {action}")

        obs, reward, terminated, truncated, info = env.step(
            np.array(action, dtype=np.int64)
        )
        frames.append(env.render())
        rewards.append(reward)
        print(
            f"- Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}, Success: {info['is_success']}"
        )
        step += 1

    env.close()
    print(f"Total reward: {sum(rewards)}")
    # Save the animation
    animation_save_dir: str = os.path.join(model_save_dir, "animations")
    os.makedirs(animation_save_dir, exist_ok=True)
    animation_path: str = os.path.join(
        animation_save_dir,
        "fourroom_primitives_"
        + "_".join(goals)
        + "_"
        + "_".join([f"{c}" for c in combination])
        + ".gif",
    )
    imageio.mimsave(animation_path, frames, fps=10, loop=0)  # type: ignore
