import os
from typing import Any

import gym_multigrid
import gymnasium as gym
import imageio
import yaml
from gym_tl_tools import (
    Predicate,
    TLObservationReward,
    replace_special_characters,
)
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from hrl_tl.envs.tl_fourroom import var_value_info_generator

if __name__ == "__main__":
    spec_id: int = 1

    tl_specs: list[str] = [
        "Fpsi_gl",
        "Fpsi_ld & G!psi_lv",
        "Fpsi_td & G!psi_hl",
    ]
    tl_spec: str = tl_specs[spec_id]
    predicates: list[Predicate] = [
        Predicate(name="psi_ld", formula="d_ld < 0.5"),
        Predicate(name="psi_bd", formula="d_bd < 0.5"),
        Predicate(name="psi_td", formula="d_td < 0.5"),
        Predicate(name="psi_rd", formula="d_rd < 0.5"),
        Predicate(name="psi_gl", formula="d_gl < 0.5"),
        Predicate(name="psi_lv", formula="d_lv < 0.5"),
        Predicate(name="psi_hl", formula="d_hl < 0.5"),
    ]
    retrain_model: bool = False
    gpu_id: int = 1
    model_name: str = "fourroom_tl_ppo_stay_" + replace_special_characters(
        tl_spec
    )
    model_save_dir: str = "out/fourroom/ltl_ll/ll_policies/"
    animation_save_dir: str = "out/plots/fourroom/ltl_ll/"
    total_timesteps: int = 5_00_000
    max_episode_steps: int = 100
    n_envs: int = 10
    callback_save_frequency: int = int(total_timesteps / 10 / n_envs)

    config_dir: str = "configs/fourroom"
    env_config_file: str = "layout/full.yaml"
    alg_config_file: str = "rl/ppo_tl.yaml"

    with open(os.path.join(config_dir, env_config_file), "r") as f:
        env_kwargs: dict[str, Any] = yaml.safe_load(f)

    with open(os.path.join(config_dir, alg_config_file), "r") as f:
        data = yaml.safe_load(f)
        rl_config: dict[str, Any] = data["rl_config"]
        vec_config: dict[str, Any] = data["vec_config"]
        learn_config: dict[str, Any] = data["learn_config"]

    wrapper_kwargs = {
        "tl_spec": tl_spec,
        "atomic_predicates": predicates,
        "var_value_info_generator": var_value_info_generator,
        "reward_config": {
            "terminal_state_reward": 0,
            "state_trans_reward_scale": 10,
            "dense_reward": True,
            "dense_reward_scale": 0.01,
        },
        "early_termination": True,
    }

    model_save_path: str = os.path.join(model_save_dir, model_name)

    # env = gym.make(
    #     "multigrid-rooms-v0", max_episode_steps=max_episode_steps, spawn_type=3
    # )
    # env = TLObservationReward(env, **wrapper_kwargs)

    env = make_vec_env(
        "multigrid-rooms-v0",
        n_envs=n_envs,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "spawn"},
        wrapper_class=TLObservationReward,
        wrapper_kwargs=wrapper_kwargs,
    )

    env_kwargs["random_init_pos"] = (
        False  # Disable random init pos for demo env
    )
    demo_env = gym.make(
        "multigrid-rooms-v0",
        **env_kwargs,
    )
    demo_env = TLObservationReward(
        demo_env,
        **wrapper_kwargs,
    )
    if not os.path.exists(model_save_path) or retrain_model:
        os.makedirs(model_save_dir, exist_ok=True)
        model = PPO(
            **rl_config,
            env=env,
            tensorboard_log=os.path.join(model_save_path, "tb"),
            device="cuda:{}".format(gpu_id),
        )
        model.learn(total_timesteps=total_timesteps)
        env.close()
        # Save the model
        model.save(os.path.join(model_save_path, "final_model"))
    else:
        print(f"Model {model_name} already exists, loading...")
        model = PPO.load(
            os.path.join(model_save_path, "final_model"), env=demo_env
        )
    # Save the animation

    # Video generation using imageio
    obs, _ = demo_env.reset()
    terminated: bool = False
    truncated: bool = False
    frames = [demo_env.render()]
    rewards: list[float] = []
    while not (terminated or truncated):
        action, _ = model.predict(obs)
        # Ensure action is a numpy int64 scalar
        obs, reward, terminated, truncated, info = demo_env.step(action)
        print(
            f"Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}, Success: {info['is_success']}"
        )
        rewards.append(reward)
        frame = demo_env.render()
        frames.append(frame)
    demo_env.close()
    print(f"Total reward: {sum(rewards)}")
    video_path = os.path.join(animation_save_dir, f"{model_name}.gif")
    print(f"Saving video to {video_path}")
    os.makedirs(animation_save_dir, exist_ok=True)
    imageio.mimsave(video_path, frames, fps=10, dpi=300, loop=10)  # type: ignore
