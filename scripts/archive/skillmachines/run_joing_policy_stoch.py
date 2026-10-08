import os
from typing import Any, Literal

import gym_multigrid
import gymnasium as gym
import imageio
import numpy as np
import torch
import yaml
from gym_tl_tools import replace_special_characters
from numpy.typing import NDArray
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from torch import Tensor

from hrl_tl.utils.io import get_class

if __name__ == "__main__":
    primitives: list[str] = [
        "phi_bd",
        "phi_gl",
        "phi_hl",
        "phi_ld",
        "phi_lv",
        "phi_rd",
        "phi_td",
        "c_phi_bd",
        "c_phi_gl",
        "c_phi_hl",
        "c_phi_ld",
        "c_phi_lv",
        "c_phi_rd",
        "c_phi_td",
    ]
    primitive_choices: list[str] = ["phi_ld", "c_phi_gl"]
    state_rep: Literal["tensor", "pos"] = "tensor"

    gpu_id: int = 2
    alg_name: str = "PPO"
    alg_class: type[PPO] | None = get_class(f"stable_baselines3.{alg_name}")
    assert alg_class is not None, (
        f"Algorithm {alg_name} not found in stable_baselines3."
    )

    agent_pos: tuple[int, int] = (8, 11)

    primitive_formula_file: str = (
        "assets/formulae/fourroom/skill_machines_map.yaml"
    )

    config_dir: str = "configs/fourroom"

    match state_rep:
        case "pos":
            env_config_file: str = "env/full.yaml"
        case "tensor":
            env_config_file: str = "env/full_tensor.yaml"

    model_save_dir: str = f"out/fourroom/primitives/{state_rep}"
    model_name: str = os.path.join(
        model_save_dir,
        f"fourroom_tl_prim_ppo_{state_rep}" + "_{spec}",
        "final_model",
    )

    with open(primitive_formula_file, "r") as f:
        primitive_formula_mapping: dict[str, str] = yaml.safe_load(f)

    model_names: list[str] = [
        model_name.format(
            spec=replace_special_characters(primitive_formula_mapping[prim])
        )
        for prim in primitive_choices
    ]

    with open(os.path.join(config_dir, env_config_file), "r") as f:
        data: dict[str, Any] = yaml.safe_load(f)
        env_kwargs: dict[str, Any] = data["env_kwargs"]
        env_id: str = data["id"]
        max_episode_steps: int = data["max_episode_steps"]

    env_kwargs["layout_config"]["spawn_configs"][0]["agent"] = agent_pos

    env = gym.make(env_id, max_episode_steps=max_episode_steps, **env_kwargs)

    models: list[PPO] = [
        alg_class.load(model_name, device=f"cuda:{gpu_id}")
        for model_name in model_names
    ]

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
        dict_obs: dict[str, NDArray]
        if isinstance(obs, dict):
            dict_obs = {**obs, "aut_state": np.array(0)}
        else:
            dict_obs = {"obs": obs, "aut_state": np.array(0)}
        obs_tensor, _ = models[0].policy.obs_to_tensor(dict_obs)
        actions = []
        action_values: list[Tensor] = []

        for model in models:
            features = model.policy.extract_features(obs_tensor)
            if model.policy.share_features_extractor:
                _, latent_vf = model.policy.mlp_extractor(features)
            else:
                pi_features, vf_features = features
                latent_vf = model.policy.mlp_extractor.forward_critic(
                    vf_features
                )
            # Evaluate the values for the given observations
            model_q_values: Tensor = model.policy.value_net(latent_vf)
            action, _ = model.policy.predict(dict_obs)
            action_value = model_q_values.flatten()[action]
            actions.append(action)
            action_values.append(action_value)

            print(
                f"-- Model {model}: Action {action}, Value: {action_value.item()}"
            )

        # Choose the action with the smallest value

        conj_action = actions[
            int(torch.argmin(torch.tensor(action_values)).item())
        ]

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
        f"fourroom_tl_primitives_{state_rep}_"
        + "_".join(primitive_choices)
        + ".gif",
    )
    imageio.mimsave(animation_path, frames, fps=10, loop=0)  # type: ignore
