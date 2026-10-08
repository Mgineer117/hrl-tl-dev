import os
from pprint import pprint
from typing import Any, Literal

import contgrid  # noqa: F401
import imageio
import yaml
from rl_pipeline.sb3 import (
    SB3Pipeline,
    SB3PipelineConfig,
    SB3PipelineConfigReader,
)
from sb3_contrib import TQC
from stable_baselines3.common.base_class import BaseAlgorithm

from hrl_tl.config.spec import AvailableSpecs
from hrl_tl.config.wrapper import (
    GCLTLWrapperConfigReader,
    TLSB3PipelineConfigReader,
)
from hrl_tl.envs.tl_fourroom import ContRoomsObsVarValueInfoGenerator
from hrl_tl.utils.render import GaussianActionDistributionRenderer
from hrl_tl.wrappers.low_level_policies.cpc import (
    CPCCompositePolicy,
    LambdaConfig,
)

if __name__ == "__main__":
    exp_name: str = "3.j.a"
    model_path: str = f"out/fr_cont/cpc/pr/{exp_name}/cpc_40.0M/best_model.zip"
    spec_path: str = "assets/formulae/fourroom/gc_ltl/all_formulae_1_cla_1_max_pred_sm_hard.yaml"
    gcltlt_config_path: str = (
        "configs/fr_cont/gcltl/wrapper/gc_ltl_dense_oh_0.35.yaml"
    )
    eval_pipeline_config_path: str = (
        f"configs/fr_cont/cpc/pr/{exp_name}_train_primitive_eval_cpc.yaml"
    )
    model_class: type[BaseAlgorithm] = TQC
    action_space_type: Literal["discrete", "continuous"] = "continuous"
    device: str = "cuda:2"

    # lambda_config = LambdaConfig(L_gain=5, k_steepness=0.2, eps_margin=1.5)
    # lambda_config = LambdaConfig(L_gain=4, k_steepness=0.2, eps_margin=2)
    # lambda_config = LambdaConfig(L_gain=6, k_steepness=0.25, eps_margin=1.5)
    lambda_config = LambdaConfig(L_gain=10, k_steepness=0.1, eps_margin=2.0)

    used_saved_env_config: bool = True
    env_kwargs_save_path: str = f"env/saved/{exp_name}_env_kwargs_{{spec}}.yaml"

    var_info_generator = ContRoomsObsVarValueInfoGenerator()

    eval_pipeline_config_reader = SB3PipelineConfigReader.from_yaml(
        eval_pipeline_config_path
    )
    gcltlt_config_reader = GCLTLWrapperConfigReader.from_yaml(
        gcltlt_config_path
    )

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
            predicates=gcltlt_config_reader.predicates,
            model=model,
            var_value_info_generator=var_info_generator,
            lambda_config=lambda_config,
            goal_rep="one_hot",
            action_space_type=action_space_type,
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
            ".gif", "_action_vec_cont.gif"
        )

        verbose: bool = True
        demo_env = pipeline.env_loader.env()

        gaussian_renderer = GaussianActionDistributionRenderer()

        obs, _ = demo_env.reset()
        demo_env.unwrapped.save_spawned_config(env_config_save_full_path)  # type: ignore

        terminated: bool = False
        truncated: bool = False
        frames: list[Any] = [demo_env.render()]
        rewards: list[float] = []
        while not (terminated or truncated):
            action, _ = policy.predict(obs, deterministic=False)  # type: ignore

            gaussian_renderer.clear()
            if (
                policy.last_goal_mean is not None
                and policy.last_goal_std is not None
            ):
                gaussian_renderer.set_distribution(
                    mean=policy.last_goal_mean,
                    std=policy.last_goal_std,
                    label="Goal",
                    color="blue",
                )
            if (
                policy.last_constraint_mean is not None
                and policy.last_constraint_std is not None
            ):
                gaussian_renderer.set_distribution(
                    mean=policy.last_constraint_mean,
                    std=policy.last_constraint_std,
                    label="Constraint",
                    color="red",
                )
            if (
                policy.last_joint_mean is not None
                and policy.last_joint_std is not None
            ):
                gaussian_renderer.set_distribution(
                    mean=policy.last_joint_mean,
                    std=policy.last_joint_std,
                    label="Composed",
                    color="green",
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
            demo_env.render()
            frame_with_actions = gaussian_renderer.render(
                demo_env.unwrapped.env
            )  # type: ignore
            if frame_with_actions is None:
                frame_with_actions = demo_env.render()
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
