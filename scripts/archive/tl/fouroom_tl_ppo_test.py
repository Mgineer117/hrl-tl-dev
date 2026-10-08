import os
from typing import Any

import gym_multigrid
import gymnasium as gym
import yaml
from gym_multigrid.envs.rooms import RoomsEnvInit
from gym_tl_tools import (
    TLObservationReward,
    TLObservationRewardConfig,
    replace_special_characters,
)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from hrl_tl.config import (
    EnvMakeConfig,
    SB3LowLevelTrainingConfig,
    VecEnvMakeConfig,
)
from hrl_tl.utils.sb3 import make_vec_env, record_replay

if __name__ == "__main__":
    spec_id: int = 0
    tl_specs: list[str] = [
        "F(psi_ld) & G(!psi_lv)",
        "F(psi_td) & G(!psi_hl)",
        "F(psi_gl)",
    ]

    gpu_id: int = 0
    retrain_model: bool = True
    training_config_path: str = "configs/fourroom/train/test_tl_ppo.yaml"

    tl_spec: str = tl_specs[spec_id]
    with open(training_config_path, "r") as f:
        training_config_dict: dict[str, Any] = yaml.safe_load(f)
        training_config_dict.update(
            {
                "gpu_id": gpu_id,
                "retrain_model": retrain_model,
                "model_name_suffix": f"_{replace_special_characters(tl_spec)}",
            }
        )

    training_config = SB3LowLevelTrainingConfig(**training_config_dict)

    env_config = EnvMakeConfig[RoomsEnvInit](**training_config.env_config_dict)
    vec_config_dict: dict[str, Any] = training_config.vec_config_dict
    tl_wrapper_config = TLObservationRewardConfig(
        tl_spec=tl_spec, **training_config.tl_wrapper_config_dict
    )

    demo_env = TLObservationReward(
        gym.make(**env_config.model_dump(context={"flatten": True})),
        **tl_wrapper_config.model_dump(
            context={"excluded": ["atomic_predicates"]}
        ),
    )

    if not os.path.exists(training_config.model_save_path) or retrain_model:
        vec_config = VecEnvMakeConfig[
            RoomsEnvInit,
            dict[str, Any],
            dict[str, Any],
            TLObservationRewardConfig,
        ](
            **vec_config_dict,
            env_id=env_config.id,
            env_kwargs=env_config.env_kwargs,
            monitor_dir=training_config.monitor_save_dir,
            wrapper_kwargs=tl_wrapper_config,
        )
        env = make_vec_env(**vec_config.model_dump())

        model = PPO(
            **training_config.rl_config_dict,
            env=env,
            tensorboard_log=training_config.tb_save_dir,
            device="cuda:{}".format(training_config.gpu_id),
        )
        eval_callback = EvalCallback(
            eval_env=env, **training_config.eval_callback_config.model_dump()
        )
        ckpt_callback = CheckpointCallback(
            **training_config.checkpoint_callback_config.model_dump()
        )
        callbacks = [eval_callback, ckpt_callback]
        model.learn(**training_config.learn_config_dict, callback=callbacks)
        env.close()
        # Save the model
        os.makedirs(training_config.model_save_dir, exist_ok=True)
        model.save(training_config.model_save_path)

    else:
        print(
            f"Model {training_config.model_full_name} already exists, loading..."
        )
        model = PPO.load(training_config.model_save_path, env=demo_env)

    # Save the animation
    record_replay(demo_env, model, training_config.animation_save_path)
