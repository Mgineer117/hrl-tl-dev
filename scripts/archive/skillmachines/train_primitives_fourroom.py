import os
from typing import Any, Literal

import gym_multigrid
import gymnasium as gym
import imageio
import yaml
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

Goal = Literal[
    "goal", "lava", "hole", "max", "min", "neg_goal", "neg_lava", "neg_hole"
]
if __name__ == "__main__":
    goal_id: int = 1
    goals: list[Goal] = [
        "goal",
        "lava",
        "hole",
        "max",
        "min",
        "neg_goal",
        "neg_lava",
        "neg_hole",
    ]
    goal: Goal = "neg_hole"
    gpu_id: int = goal_id
    alg_name: str = "dqn"

    config_dir: str = "configs/fourroom"
    layout_config_file: str = f"layout/{goal}_goals.yaml"
    alg_config_file: str = f"rl/{alg_name}.yaml"

    retrain_model: bool = True
    model_save_dir: str = "out/poc/skillmachines/primitives"

    model_name: str = "fourroom_primitives_" + goal

    model_save_path: str = os.path.join(model_save_dir, model_name)
    animation_save_dir: str = os.path.join(model_save_path)

    with open(os.path.join(config_dir, layout_config_file), "r") as f:
        env_kwargs: dict[str, Any] = yaml.safe_load(f)
    with open(os.path.join(config_dir, alg_config_file), "r") as f:
        data = yaml.safe_load(f)
        rl_config: dict[str, Any] = data["rl_config"]
        vec_config: dict[str, Any] = data["vec_config"]
        learn_config: dict[str, Any] = data["learn_config"]

    env = make_vec_env(
        "multigrid-rooms-v0",
        **vec_config,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv,
    )

    demo_env = gym.make("multigrid-rooms-v0", **env_kwargs)

    if not os.path.exists(model_save_path) or retrain_model:
        os.makedirs(model_save_dir, exist_ok=True)
        model = DQN(
            **rl_config,
            env=env,
            tensorboard_log=os.path.join(model_save_path, "tb"),
            device="cuda:{}".format(gpu_id),
        )
        model.learn(**learn_config)
        env.close()
        # Save the model
        model.save(os.path.join(model_save_path, "final_model"))

    else:
        print(f"Model {model_name} already exists, loading...")
        model = DQN.load(
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
    imageio.mimsave(video_path, frames, fps=10, dpi=300, loop=10)  # type: ignore
