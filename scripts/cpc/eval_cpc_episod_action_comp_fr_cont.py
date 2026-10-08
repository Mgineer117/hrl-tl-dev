"""Compare composed action distributions of 3.f.f vs 3.k.m on a shared trajectory.

This script:
1. Rolls out an episode using the 3.f.f CPC composed policy (the better-performing one),
   recording the full trajectory of observations, actions, and action distributions.
2. At each recorded state, also queries the 3.k.m CPC composed policy to see what
   its composed action distribution looks like on the same observation.
3. Renders side-by-side frames: left = 3.f.f (actual trajectory), right = 3.k.m
   (counterfactual distribution on the same state). The environment rendering on
   both sides shows the same physical state from 3.f.f's trajectory.
"""

import copy
import os
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
from hrl_tl.envs.tl_fourroom import ContRoomsObsVarValueInfoGenerator
from hrl_tl.utils.render import DiscreteActionVectorRenderer
from hrl_tl.wrappers.low_level_policies.cpc import (
    CPCCompositePolicy,
    LambdaConfig,
)


def _make_policy(
    exp_name: str,
    tl_spec: str,
    model: BaseAlgorithm,
    gcltlt_config_reader: GCLTLWrapperConfigReader,
    var_info_generator: ContRoomsObsVarValueInfoGenerator,
    lambda_config: LambdaConfig,
    action_space_type: Literal["discrete", "continuous"],
    normalize_lambda: bool,
    verbose: bool = False,
) -> CPCCompositePolicy:
    """Create a CPCCompositePolicy for a given experiment."""
    return CPCCompositePolicy(
        tl_spec=tl_spec,
        predicates=gcltlt_config_reader.predicates,
        model=model,
        var_value_info_generator=var_info_generator,
        lambda_config=lambda_config,
        goal_rep="one_hot",
        action_space_type=action_space_type,
        normalize_lambdas=normalize_lambda,
        verbose=verbose,
    )


def _make_renderers(
    num_actions: int, color_prefix: str = ""
) -> dict[str, DiscreteActionVectorRenderer]:
    """Create a set of action vector renderers (composite, goal, constraint)."""
    return {
        "composite": DiscreteActionVectorRenderer(
            arrow_width=0.05,
            arrow_color="green",
            num_directions=num_actions,
        ),
        "goal": DiscreteActionVectorRenderer(
            arrow_color="blue",
            num_directions=num_actions,
        ),
        "constraint": DiscreteActionVectorRenderer(
            arrow_color="red",
            num_directions=num_actions,
        ),
    }


def _update_renderers(
    renderers: dict[str, DiscreteActionVectorRenderer],
    policy: CPCCompositePolicy,
) -> None:
    """Update renderer probabilities from a CPC policy's last cached distributions."""
    if policy.action_combinations is None or policy.last_joint_prob is None:
        return

    # If single action dimension, add dummy velocity column for the renderer
    if policy.action_combinations.shape[1] == 1:
        action_combinations = np.hstack(
            [
                policy.action_combinations,
                np.ones((policy.action_combinations.shape[0], 1)) * 5,
            ]
        )
    else:
        action_combinations = policy.action_combinations

    renderers["composite"].set_multi_discrete_probabilities(
        action_combinations,
        policy.last_joint_prob,
    )
    renderers["goal"].set_multi_discrete_probabilities(
        action_combinations,
        policy.last_goal_prob,
    )
    renderers["constraint"].set_multi_discrete_probabilities(
        action_combinations,
        policy.last_constraint_prob,
    )


def _render_action_overlays(
    renderers: dict[str, DiscreteActionVectorRenderer],
    env_unwrapped: Any,
) -> np.ndarray:
    """Render constraint, goal, and composite arrows onto the environment figure."""
    renderers["constraint"].render(env_unwrapped)
    renderers["goal"].render(env_unwrapped)
    frame = renderers["composite"].render(env_unwrapped)
    return frame


def _hstack_frames(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Horizontally stack two frames, padding the shorter one if needed."""
    h_l, w_l = left.shape[:2]
    h_r, w_r = right.shape[:2]
    h = max(h_l, h_r)
    if h_l < h:
        pad = np.zeros((h - h_l, w_l, 3), dtype=np.uint8)
        left = np.vstack([left, pad])
    if h_r < h:
        pad = np.zeros((h - h_r, w_r, 3), dtype=np.uint8)
        right = np.vstack([right, pad])
    # Add a thin separator column
    sep = np.full((h, 4, 3), 128, dtype=np.uint8)
    return np.hstack([left, sep, right])


def _add_label(
    frame: np.ndarray, label: str, position: str = "top-left"
) -> np.ndarray:
    """Add a text label onto a frame using matplotlib."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    h, w = frame.shape[:2]
    dpi = 100
    fig = Figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(frame)
    ax.axis("off")

    if position == "top-left":
        ax.text(
            10,
            20,
            label,
            fontsize=14,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
        )
    elif position == "top-right":
        ax.text(
            w - 10,
            20,
            label,
            fontsize=14,
            fontweight="bold",
            color="white",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
        )

    canvas.draw()
    buf = canvas.buffer_rgba()
    result = np.asarray(buf)[:, :, :3].copy()
    import matplotlib.pyplot as plt

    plt.close(fig)
    return result


if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #
    # Primary experiment (better performance, drives the trajectory)
    primary_exp_name: str = "3.f.f"
    primary_pr_type: str = "pr"
    primary_model_path: str = (
        f"out/fr_cont/cpc/pr/{primary_exp_name}/cpc_20.0M/rep_0/best_model.zip"
    )
    primary_model_class: type[BaseAlgorithm] = PPO

    # Secondary experiment (compared against, same states)
    secondary_exp_name: str = "3.k.m"
    secondary_pr_type: str = "pr_soft"
    secondary_model_path: str = f"out/fr_cont/cpc/pr/{secondary_exp_name}/cpc_20.0M/rep_20/best_model.zip"
    secondary_model_class: type[BaseAlgorithm] = SDSAC

    # Environment (use same saved env for reproducibility)
    env_exp_name: str = "3.k.g"

    spec_path: str = "assets/formulae/fourroom/gc_ltl/all_formulae_1_cla_1_max_pred_sm_hard.yaml"
    gcltlt_config_path: str = (
        "configs/fr_cont/gcltl/wrapper/gc_ltl_dense_oh_0.35.yaml"
    )

    # Use the primary experiment's eval pipeline config for the environment
    primary_eval_pipeline_config_path: str = f"configs/fr_cont/cpc/{primary_pr_type}/{primary_exp_name}_train_primitive_eval_cpc_fix.yaml"

    action_space_type: Literal["discrete", "continuous"] = "discrete"
    device: str = "cuda:2"

    primary_lambda_config = LambdaConfig(
        L_gain=12, k_steepness=0.3, eps_margin=2
    )
    secondary_lambda_config = LambdaConfig(
        L_gain=40, k_steepness=0.3, eps_margin=2
    )
    normalize_lambda: bool = False
    used_saved_env_config: bool = True
    env_kwargs_save_path: str = (
        f"env/saved/{env_exp_name}_env_kwargs_{{spec}}.yaml"
    )

    verbose: bool = True

    # ------------------------------------------------------------------ #
    # Load shared resources
    # ------------------------------------------------------------------ #
    var_info_generator = ContRoomsObsVarValueInfoGenerator()

    primary_eval_config_reader = SB3PipelineConfigReader.from_yaml(
        primary_eval_pipeline_config_path
    )
    gcltlt_config_reader = GCLTLWrapperConfigReader.from_yaml(
        gcltlt_config_path
    )

    avail_specs: AvailableSpecs = AvailableSpecs.from_yaml(spec_path)

    # Load both models
    print(
        f"Loading primary model ({primary_exp_name}) from {primary_model_path}"
    )
    primary_model = primary_model_class.load(primary_model_path, device=device)

    print(
        f"Loading secondary model ({secondary_exp_name}) from {secondary_model_path}"
    )
    secondary_model = secondary_model_class.load(
        secondary_model_path, device=device
    )

    # ------------------------------------------------------------------ #
    # Per-spec evaluation
    # ------------------------------------------------------------------ #
    for tl_spec in avail_specs.specifications:
        print(f"\n{'=' * 60}")
        print(f"Evaluating TL spec: {tl_spec}")
        print(f"{'=' * 60}")

        env_config_save_full_path: str = (
            primary_eval_config_reader.config_dir
            + "/"
            + env_kwargs_save_path.format(spec=tl_spec)
        )

        # Create both CPC policies for this spec
        primary_policy: CPCCompositePolicy = _make_policy(
            exp_name=primary_exp_name,
            tl_spec=tl_spec,
            model=primary_model,
            gcltlt_config_reader=gcltlt_config_reader,
            var_info_generator=var_info_generator,
            lambda_config=primary_lambda_config,
            action_space_type=action_space_type,
            normalize_lambda=normalize_lambda,
            verbose=verbose,
        )
        secondary_policy: CPCCompositePolicy = _make_policy(
            exp_name=secondary_exp_name,
            tl_spec=tl_spec,
            model=secondary_model,
            gcltlt_config_reader=gcltlt_config_reader,
            var_info_generator=var_info_generator,
            lambda_config=secondary_lambda_config,
            action_space_type=action_space_type,
            normalize_lambda=normalize_lambda,
            verbose=verbose,
        )

        # Build pipeline and environment
        pipeline_config: SB3PipelineConfig = TLSB3PipelineConfigReader(
            tl_spec=tl_spec, pipeline_config=primary_eval_config_reader
        ).to_config()

        if used_saved_env_config:
            with open(env_config_save_full_path, "r") as f:
                env_kwargs: dict[str, Any] = yaml.safe_load(f)
            pipeline_config.env_config.env_kwargs["scenario_config"] = (
                env_kwargs
            )

        pipeline = SB3Pipeline(config=pipeline_config, verbose=True)
        animation_save_path = pipeline.save_config.animation_save_path.replace(
            ".gif",
            f"_action_comp_{primary_exp_name}_vs_{secondary_exp_name}.gif",
        )

        demo_env = pipeline.env_loader.env()
        num_actions = demo_env.action_space.n  # type: ignore

        # Create renderers for both policies
        primary_renderers = _make_renderers(num_actions)
        secondary_renderers = _make_renderers(num_actions)

        obs, _ = demo_env.reset()
        demo_env.unwrapped.save_spawned_config(env_config_save_full_path)  # type: ignore

        # ------------------------------------------------------------------ #
        # Phase 1: Roll out with primary policy, record trajectory
        # ------------------------------------------------------------------ #
        terminated: bool = False
        truncated: bool = False
        rewards: list[float] = []
        combined_frames: list[np.ndarray] = []

        # Render initial frame (no action distributions yet)
        init_frame = demo_env.render()
        combined_frames.append(
            _hstack_frames(
                _add_label(
                    init_frame, f"{primary_exp_name} (trajectory)", "top-left"
                ),
                _add_label(
                    init_frame.copy(),
                    f"{secondary_exp_name} (counterfactual)",
                    "top-left",
                ),
            )
        )

        step_idx = 0
        while not (terminated or truncated):
            # --- Primary policy: predict action (drives the trajectory) ---
            action, _ = primary_policy.predict(obs, deterministic=False)  # type: ignore

            # --- Secondary policy: query distribution on the SAME state ---
            # (we don't use the action, just want the distribution)
            _ = secondary_policy.predict(obs, deterministic=False)  # type: ignore

            # Update renderers from cached distributions
            _update_renderers(primary_renderers, primary_policy)
            _update_renderers(secondary_renderers, secondary_policy)

            # Step the environment with the PRIMARY action
            obs, reward, terminated, truncated, info = demo_env.step(action)
            rewards.append(reward)  # type: ignore

            if verbose:
                print(
                    f"Step {step_idx + 1}: reward={reward:.3f}, "
                    f"terminated={terminated}, truncated={truncated}"
                )

            # Render frame and overlay primary action arrows
            _ = demo_env.render()
            primary_frame = _render_action_overlays(
                primary_renderers,
                demo_env.unwrapped.env,  # type: ignore
            )

            # Re-render the same environment state for secondary overlays.
            # The env render state (fig/ax) was modified by primary rendering,
            # so we re-render to get a clean frame, then overlay secondary arrows.
            secondary_base = demo_env.render()
            secondary_frame = _render_action_overlays(
                secondary_renderers,
                demo_env.unwrapped.env,  # type: ignore
            )

            # Label and combine side by side
            primary_labeled = _add_label(
                primary_frame,
                f"{primary_exp_name} | Step {step_idx}",
                "top-left",
            )
            secondary_labeled = _add_label(
                secondary_frame,
                f"{secondary_exp_name} | Step {step_idx}",
                "top-left",
            )
            combined = _hstack_frames(primary_labeled, secondary_labeled)
            combined_frames.append(combined)

            step_idx += 1

        demo_env.close()

        if verbose:
            print(
                f"\nTotal reward ({primary_exp_name} trajectory): {sum(rewards):.3f}"
            )
            print(f"Episode length: {len(rewards)} steps")

        # ------------------------------------------------------------------ #
        # Save output
        # ------------------------------------------------------------------ #
        os.makedirs(os.path.dirname(animation_save_path), exist_ok=True)
        imageio.mimsave(animation_save_path, combined_frames, fps=2, loop=10)  # type: ignore
        imageio.mimsave(
            animation_save_path.replace(".gif", ".mp4"), combined_frames, fps=2
        )  # type: ignore
        print(f"Saved comparison animation to {animation_save_path}")
        print(
            f"Saved comparison video to {animation_save_path.replace('.gif', '.mp4')}"
        )
