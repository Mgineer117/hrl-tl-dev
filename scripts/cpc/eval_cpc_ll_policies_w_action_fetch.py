"""Evaluation script for CPC composite low-level policies on Fetch Reach-Avoid."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import imageio
import numpy as np
from absl import app, flags
from rl_pipeline.sb3 import (
    SB3Pipeline,
    SB3PipelineConfig,
    SB3PipelineConfigReader,
)
from sb3_soft import SDSAC

import hrl_tl.envs.fetch  # noqa: F401
from hrl_tl.config.spec import AvailableSpecs
from hrl_tl.config.wrapper import (
    GCLTLWrapperConfigReader,
    TLSB3PipelineConfigReader,
)
from hrl_tl.envs.fetch.env import FetchReachAvoidEnv
from hrl_tl.envs.fetch.var_value import FetchVarValueInfoGenerator
from hrl_tl.wrappers.low_level_policies.cpc import (
    CPCCompositePolicy,
    LambdaConfig,
)

_MODEL_PATH = flags.DEFINE_string(
    "model_path",
    default="out/fetch/cpc/pr/fetch_reach_avoid_primitive/cpc_10.0M/rep_0/best_model.zip",
    help="Path to trained primitive policy model.",
)
_SPEC_PATH = flags.DEFINE_string(
    "spec_path",
    default="assets/formulae/fetch/cpc/all_formulae_1_cla_1_max_pred.json",
    help="Path to LTL specifications JSON file.",
)
_GCLTL_CONFIG_PATH = flags.DEFINE_string(
    "gcltl_config_path",
    default="configs/fetch/cpc/wrapper/cpc_dense.yaml",
    help="Path to GCLTL wrapper configuration YAML file.",
)
_PIPELINE_CONFIG_PATH = flags.DEFINE_string(
    "pipeline_config_path",
    default="configs/fetch/cpc/pr/train_primitive_rep.yaml",
    help="Path to pipeline configuration YAML file.",
)
_DEVICE = flags.DEFINE_string(
    "device",
    default="cpu",
    help="Device to load the model on (e.g. 'cpu', 'cuda:0').",
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    default="out/fetch/cpc/eval",
    help="Directory where evaluation animations are saved.",
)
_L_GAIN = flags.DEFINE_float(
    "l_gain",
    default=7.0,
    help="Sigmoid gain for constraint weight.",
)
_K_STEEPNESS = flags.DEFINE_float(
    "k_steepness",
    default=5.0,
    help="Sigmoid steepness for constraint weight.",
)
_EPS_MARGIN = flags.DEFINE_float(
    "eps_margin",
    default=0.5,
    help="Margin for constraint weight sigmoid.",
)


def main(argv: Sequence[str]) -> None:
    """Evaluates CPC composite policies on Fetch Reach-Avoid.

    Args:
        argv: Command-line arguments.

    Raises:
        app.UsageError: If unexpected positional arguments are supplied.
    """
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    os.environ.setdefault("MUJOCO_GL", "egl")
    output_dir = Path(_OUTPUT_DIR.value)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(_MODEL_PATH.value)
    if not model_path.exists():
        print(f"Model path {model_path} does not exist yet.")
        return

    lambda_config = LambdaConfig(
        L_gain=_L_GAIN.value,
        k_steepness=_K_STEEPNESS.value,
        eps_margin=_EPS_MARGIN.value,
    )
    var_info_generator = FetchVarValueInfoGenerator()

    eval_pipeline_config_reader = SB3PipelineConfigReader.from_yaml(
        _PIPELINE_CONFIG_PATH.value
    )
    gcltl_config_reader = GCLTLWrapperConfigReader.from_yaml(
        _GCLTL_CONFIG_PATH.value
    )
    avail_specs = AvailableSpecs.from_yaml(_SPEC_PATH.value)

    model = SDSAC.load(str(model_path), device=_DEVICE.value)

    for tl_spec in avail_specs.specifications:
        print(f"Evaluating TL specification: {tl_spec}")
        policy = CPCCompositePolicy(
            tl_spec=tl_spec,
            predicates=gcltl_config_reader.predicates,
            model=model,
            var_value_info_generator=var_info_generator,
            lambda_config=lambda_config,
            goal_rep="one_hot",
            action_space_type="discrete",
            normalize_lambdas=False,
            verbose=True,
        )

        pipeline_config: SB3PipelineConfig = TLSB3PipelineConfigReader(
            tl_spec=tl_spec, pipeline_config=eval_pipeline_config_reader
        ).to_config()

        pipeline = SB3Pipeline(config=pipeline_config, verbose=True)
        demo_env = pipeline.env_loader.env()

        obs, _ = demo_env.reset()
        base_env: FetchReachAvoidEnv = demo_env.unwrapped  # type: ignore[assignment]
        init_multiview = base_env.render_multiview()

        frames_default: list[np.ndarray] = [init_multiview["default"]]
        frames_topdown: list[np.ndarray] = [init_multiview["top_down"]]
        frames_combined: list[np.ndarray] = [init_multiview["combined"]]
        total_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = policy.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, _ = demo_env.step(action)
            total_reward += float(reward)
            step_views = base_env.render_multiview()
            frames_default.append(step_views["default"])
            frames_topdown.append(step_views["top_down"])
            frames_combined.append(step_views["combined"])

        demo_env.close()
        print(f"Spec {tl_spec} total reward: {total_reward:.3f}")

        safe_spec_name = (
            tl_spec.replace(" ", "_")
            .replace("&", "and")
            .replace("!", "not_")
            .replace("(", "")
            .replace(")", "")
        )
        anim_path_combined = output_dir / f"cpc_eval_{safe_spec_name}.gif"
        anim_path_default = (
            output_dir / f"cpc_eval_{safe_spec_name}_default.gif"
        )
        anim_path_topdown = (
            output_dir / f"cpc_eval_{safe_spec_name}_topdown.gif"
        )

        imageio.mimsave(
            str(anim_path_combined), frames_combined, duration=66, loop=0
        )
        imageio.mimsave(
            str(anim_path_default), frames_default, duration=66, loop=0
        )
        imageio.mimsave(
            str(anim_path_topdown), frames_topdown, duration=66, loop=0
        )
        print(f"Animations saved to {output_dir}")


if __name__ == "__main__":
    app.run(main)
