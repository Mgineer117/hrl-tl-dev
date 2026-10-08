import os
from typing import Any, Literal

import gym_multigrid
import gymnasium as gym
import imageio
import numpy as np
import yaml
from gym_tl_tools import Predicate
from numpy.typing import NDArray
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from hrl_tl.envs.tl_fourroom import PolicyArgsDict, var_value_info_generator
from hrl_tl.utils.io import format_large_number
from hrl_tl.wrappers.low_level_policies.sb3 import (
    SB3LowLevelPolicy,
    SB3PolicyArgsDict,
)
from hrl_tl.wrappers.tl_high_level import TLHighLevelWrapper, TLWrapperArgsDict
from hrl_tl.wrappers.utils.spec_rep import Lv1SpecRep, SpecRepArgsDict

if __name__ == "__main__":
    experiment_id: str = "3.a"
    max_low_level_policy_steps: int = 10
    num_clauses: int = 2
    num_max_predicates: int = 2
    all_formulae_file_path: str = f"assets/formulae/fourroom/all_formulae_{num_clauses}_cla_{num_max_predicates}_max_pred.json"
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
    total_timesteps: int = 1_500_000
    model_name: str = f"hl_policy_rand_{format_large_number(total_timesteps)}"
    invalid_tl_action: Literal["stay", "random"] = "random"

    model_save_dir: str = f"out/fourroom/ltl_ll/{experiment_id}/"
    file_name: str = "final_model"
    retrain_model: bool = False and ("ckpt" not in file_name)

    n_envs: int = 20
    callback_save_frequency: int = int(100_000 / n_envs)

    config_dir: str = "configs/fourroom"
    env_config_file: str = "layout/full_fixed.yaml"
    alg_config_file: str = "rl/ppo_hl.yaml"

    with open(os.path.join(config_dir, env_config_file), "r") as f:
        env_kwargs: dict[str, Any] = yaml.safe_load(f)
    with open(os.path.join(config_dir, alg_config_file), "r") as f:
        data = yaml.safe_load(f)
        rl_config: dict[str, Any] = data["rl_config"]
        vec_config: dict[str, Any] = data["vec_config"]
        vec_config["n_envs"] = n_envs
        learn_config: dict[str, Any] = data["learn_config"]
        learn_config["total_timesteps"] = total_timesteps

    low_level_policy_args: SB3PolicyArgsDict = {
        "algorithm": PPO,
        "algo_config": {
            "policy": "MultiInputPolicy",
            "learning_rate": 0.0003,
            "env": None,
            "n_steps": 1000,
            "batch_size": 1000,
            "n_epochs": 40,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "clip_range_vf": None,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "use_sde": False,
            "sde_sample_freq": -1,
            "rollout_buffer_class": None,
            "rollout_buffer_kwargs": None,
            "target_kl": None,
            "stats_window_size": 100,
            "policy_kwargs": {"net_arch": [128, 128]},
        },
        "model_save_dir": "out/fourroom/ltl_ll/ll_policies",
        "model_prefix": "fourroom_tl_ppo_stay_",
        "model_name": "final_model",
        "training_config": {
            "total_timesteps": 50_000,
            "n_envs": 10,
        },
        "device": "cuda:0",
    }
    tl_wrapper_kwargs: TLWrapperArgsDict = {
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
    spec_rep_class: type[Lv1SpecRep] = Lv1SpecRep
    spec_rep_args: SpecRepArgsDict = {
        "num_clauses": num_clauses,
        "predicate_names": [p.name for p in predicates],
    }
    high_level_wrapper_kwargs: dict[str, Any] = {
        "spec_rep_class": spec_rep_class,
        "spec_rep_args": spec_rep_args,
        "low_level_policy_class": SB3LowLevelPolicy,
        "low_level_policy_args": low_level_policy_args,
        "max_low_level_policy_steps": max_low_level_policy_steps,
        "all_formulae_file_path": all_formulae_file_path,
        "invalid_tl_action": invalid_tl_action,
        "stay_action": np.int64(0),
        "tl_wrapper_args": tl_wrapper_kwargs,
    }

    model_save_path: str = os.path.join(model_save_dir, model_name)
    animation_save_dir: str = os.path.join(model_save_path)

    high_level_env = make_vec_env(
        "multigrid-rooms-v0",
        **vec_config,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv,
        wrapper_class=TLHighLevelWrapper,
        wrapper_kwargs=high_level_wrapper_kwargs,
    )

    env_kwargs["random_init_pos"] = (
        False  # Disable random init pos for demo env
    )
    env = gym.make("multigrid-rooms-v0", **env_kwargs)
    demo_env = TLHighLevelWrapper[
        NDArray[np.int64], np.int64, SB3LowLevelPolicy, PolicyArgsDict
    ](env, verbose=True, **high_level_wrapper_kwargs)

    model_save_filename: str = os.path.join(model_save_path, file_name)
    if not os.path.exists(model_save_filename + ".zip") or retrain_model:
        os.makedirs(model_save_dir, exist_ok=True)
        model = PPO(
            **rl_config,
            env=high_level_env,
            # env=demo_env,
            tensorboard_log=os.path.join(model_save_dir, "tb"),
            device="cuda:{}".format(gpu_id),
        )
        ckpt_callback = CheckpointCallback(
            save_freq=callback_save_frequency,
            save_path=model_save_path,
            name_prefix="ckpt",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        model.learn(
            **learn_config,
            tb_log_name=model_name,
            callback=ckpt_callback,
        )
        high_level_env.close()
        # Save the model
        model.save(os.path.join(model_save_path, "final_model"))
    else:
        print(f"Model {model_name} already exists, loading...")
        model = PPO.load(model_save_filename, env=demo_env)
    # Save the animation

    # Video generation using imageio
    obs, info = demo_env.reset()
    terminated: bool = False
    truncated: bool = False
    frames = [demo_env.render()]
    rewards: list[float] = []
    chosen_specs: list[tuple[int, str]] = []
    step: int = 0
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
        chosen_specs.append((step, info["current_tl_spec"]))
        step += 1
    demo_env.close()
    print(f"Total reward: {sum(rewards)}")
    print("Chosen specs:")
    for step, spec in chosen_specs:
        print(f"- Step {step}: {spec}")
    video_path = os.path.join(animation_save_dir, f"{model_name}.gif")
    os.makedirs(animation_save_dir, exist_ok=True)
    imageio.mimsave(video_path, frames, fps=10, dpi=300, loop=10)  # type: ignore
