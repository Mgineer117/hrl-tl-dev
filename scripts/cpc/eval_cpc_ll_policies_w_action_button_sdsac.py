import os
from pprint import pprint
from typing import Any, Literal

import contgrid
import imageio
import numpy as np
import yaml
from rl_pipeline.sb3 import (
    SB3Pipeline,
    SB3PipelineConfig,
    SB3PipelineConfigReader,
)
from sb3_soft import SDSAC
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from hrl_tl.config.spec import AvailableSpecs
from hrl_tl.config.wrapper import (
    GCLTLWrapperConfigReader,
    TLSB3PipelineConfigReader,
)
from hrl_tl.envs.tl_zone import ZoneVarValueInfoGenerator
from hrl_tl.utils.render import DiscreteActionPointCloudRenderer
from hrl_tl.wrappers.low_level_policies.cpc import (
    CPCCompositePolicy,
    LambdaConfig,
)

if __name__ == "__main__":
    exp_name: str = "8.f.a"
    scenario_name: str = "8.f.a"
    model_path: str = f"out/zone/cpc/pr/{exp_name}/cpc_40.0M/rep_0/ckpts/ckpt_20000000_steps.zip"
    spec_path: str = (
        "assets/formulae/zone/cpc/all_formulae_1_cla_1_max_pred_sm_hard.yaml"
    )
    gcltl_config_path: str = (
        "configs/zone/cpc/wrapper/cpc_dense_button_0.50.yaml"
    )
    eval_pipeline_config_path: str = (
        f"configs/zone/cpc/pr/{exp_name}_train_primitive_eval_cpc_fix.yaml"
    )
    model_class: type[BaseAlgorithm] = SDSAC  # PPO
    action_space_type: Literal["discrete", "continuous"] = "discrete"
    device: str = "cuda:2"

    lambda_config = LambdaConfig(L_gain=10.0, k_steepness=8, eps_margin=1.5)

    normalize_lambda: bool = False

    used_saved_env_config: bool = True
    env_kwargs_save_path: str = (
        f"env/saved/{scenario_name}_env_kwargs_{{spec}}.yaml"
    )

    var_info_generator = ZoneVarValueInfoGenerator()

    eval_pipeline_config_reader = SB3PipelineConfigReader.from_yaml(
        eval_pipeline_config_path
    )
    gcltl_config_reader = GCLTLWrapperConfigReader.from_yaml(gcltl_config_path)

    avail_specs: AvailableSpecs = AvailableSpecs.from_yaml(spec_path)
    model = model_class.load(model_path, device=device)

    for tl_spec in avail_specs.specifications:
        print(f"Evaluating TL spec: {tl_spec}")
        env_config_save_full_path: str = (
            eval_pipeline_config_reader.config_dir
            + "/"
            + env_kwargs_save_path.format(spec=tl_spec)
        )

        policy: CPCCompositePolicy = CPCCompositePolicy(
            tl_spec=tl_spec,
            predicates=gcltl_config_reader.predicates,
            model=model,
            var_value_info_generator=var_info_generator,
            lambda_config=lambda_config,
            goal_rep="one_hot",
            action_space_type=action_space_type,
            normalize_lambdas=normalize_lambda,
            verbose=True,
        )

        pipeline_config: SB3PipelineConfig = TLSB3PipelineConfigReader(
            tl_spec=tl_spec, pipeline_config=eval_pipeline_config_reader
        ).to_config()

        if used_saved_env_config:
            with open(env_config_save_full_path, "r") as f:
                env_kwargs: dict[str, Any] = yaml.safe_load(f)
            pipeline_config.env_config.env_kwargs["scenario_config"] = (
                env_kwargs
            )

        pipeline = SB3Pipeline(config=pipeline_config, verbose=True)
        # pipeline.record_replay(policy, verbose=True)
        animation_save_path = pipeline.save_config.animation_save_path.replace(
            ".gif", "_action_vec.gif"
        )

        # pipeline.record_replay(policy, verbose=True)

        verbose: bool = True
        demo_env = pipeline.env_loader.env()

        num_dir: int = 8
        num_vel: int = 6
        # Initialize action renderer for multi-discrete actions
        composite_action_renderer = DiscreteActionPointCloudRenderer(
            arrow_width=0.05,
            arrow_color="green",
            num_directions=num_dir,
            num_velocities=num_vel,
        )
        goal_action_renderer = DiscreteActionPointCloudRenderer(
            arrow_color="blue",
            num_directions=num_dir,
            num_velocities=num_vel,
        )
        constraint_action_renderer = DiscreteActionPointCloudRenderer(
            arrow_color="red",
            num_directions=num_dir,
            num_velocities=num_vel,
        )

        obs, _ = demo_env.reset()
        demo_env.unwrapped.save_spawned_config(env_config_save_full_path)  # type: ignore

        terminated: bool = False
        truncated: bool = False
        frames = [demo_env.render()]
        rewards: list[float] = []
        while not (terminated or truncated):
            action, _ = policy.predict(obs, deterministic=False)  # type: ignore

            # Update renderer with latest probabilities from CPC policy
            if (
                policy.action_combinations is not None
                and policy.last_joint_prob is not None
            ):
                # If the action dimension is 1, we need to add velocity to match the expected shape
                if policy.action_combinations.shape[1] == 1:
                    action_combinations = np.hstack(
                        [
                            policy.action_combinations,
                            np.ones(
                                (policy.action_combinations.shape[0], 1),
                            )
                            * 5,
                        ]
                    )
                else:
                    action_combinations = policy.action_combinations
                composite_action_renderer.set_multi_discrete_probabilities(
                    action_combinations,
                    policy.last_joint_prob,
                )
                goal_action_renderer.set_multi_discrete_probabilities(
                    action_combinations,
                    policy.last_goal_prob,
                )
                constraint_action_renderer.set_multi_discrete_probabilities(
                    action_combinations,
                    policy.last_constraint_prob,
                )

            # Ensure action is a numpy int64 scalar
            obs, reward, terminated, truncated, info = demo_env.step(action)
            if verbose:
                print(f"Step {len(rewards) + 1}:")
                print(
                    f" - Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}, Success: {info.get('is_success', 'N/A')}"
                )
                print(" - Obs: ")
                pprint(obs)
                print(" - Info: ")
                pprint(info)
            rewards.append(reward)  # type: ignore
            frame = demo_env.render()

            # Post-render action probabilities onto the frame
            frame_with_actions = constraint_action_renderer.render(
                demo_env.unwrapped.env  # type: ignore
            )
            frame_with_actions = goal_action_renderer.render(
                demo_env.unwrapped.env
            )  # type: ignore
            frame_with_actions = composite_action_renderer.render(
                demo_env.unwrapped.env  # type: ignore
            )
            frames.append(frame_with_actions)

        demo_env.close()
        if verbose:
            print(f" - Total reward: {sum(rewards)}")

        os.makedirs(os.path.dirname(animation_save_path), exist_ok=True)
        # imageio.mimsave(animation_save_path, frames, fps=1, dpi=300, loop=10)  # type: ignore
        # Save gif and mp4
        imageio.mimsave(animation_save_path, frames, fps=2, loop=10)  # type: ignore
        imageio.mimsave(
            animation_save_path.replace(".gif", ".mp4"), frames, fps=2
        )  # type: ignore
        if verbose:
            print(f" - Replay saved to {animation_save_path}")

        # eval_stats = pipeline.evaluate(
        #     n_eval_episodes=200, deterministic=False, checkpoint=policy, env="vec"
        # )
