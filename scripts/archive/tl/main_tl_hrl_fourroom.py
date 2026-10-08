import os
from typing import Literal

import gym_multigrid
import imageio

from hrl_tl.config.train import SB3TLHRLTrainingConfig
from hrl_tl.pipeline import SB3Pipeline

if __name__ == "__main__":
    training_config_path: str = (
        "configs/fourroom/train/3.a/tl_hrl_pretrained_3.a.q.yaml"
    )
    ckpt_timestep: int | Literal["latest", "final", "best"] = "final"
    evaluate_model: bool = False

    print(f"Using training config: {training_config_path}")
    pipeline: SB3Pipeline[SB3TLHRLTrainingConfig] = SB3Pipeline[
        SB3TLHRLTrainingConfig
    ].from_yaml(training_config_path)
    training_config: SB3TLHRLTrainingConfig = pipeline.config

    if ckpt_timestep == "final":
        model = pipeline.train_on_unsaved_model()
    else:
        model = pipeline.load_model(ckpt_timestep)

    # Evaluate the model
    if evaluate_model:
        eval_result = pipeline.evaluate(
            n_eval_episodes=100, deterministic=False, checkpoint=ckpt_timestep
        )
    else:
        pass

    demo_env = training_config.eval_env()

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
    os.makedirs(
        os.path.dirname(training_config.animation_save_path), exist_ok=True
    )
    imageio.mimsave(
        training_config.animation_save_path,
        frames,  # type: ignore
        fps=10,
        dpi=300,
        loop=10,
    )
