"""Script to render and save initial state images for the Fetch Reach-Avoid environment."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import yaml
from absl import app, flags
from PIL import Image
from rl_pipeline.gymnasium import GymEnvConfig, GymEnvLoader, MakeEnvConfig

import hrl_tl.envs.fetch  # noqa: F401
from hrl_tl.envs.fetch.env import FetchReachAvoidEnv

os.environ.setdefault("MUJOCO_GL", "egl")
_logger = logging.getLogger(__name__)


_YAML_CONFIG_PATH = flags.DEFINE_string(
    "yaml_config_path",
    default="configs/fetch/env/fetch_reach_avoid_250.yaml",
    help="Path to the environment YAML configuration file.",
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    default="out/plots",
    help="Directory where rendered image artifacts are saved.",
)
_MODEL_NAME = flags.DEFINE_string(
    "model_name",
    default="fetch",
    help="Prefix used when saving rendered image files.",
)


def _load_env_loader(config_path: Path) -> GymEnvLoader:
    """Loads environment configuration and returns a GymEnvLoader.

    Args:
        config_path: Path to the environment YAML configuration.

    Returns:
        Configured GymEnvLoader instance.

    Raises:
        FileNotFoundError: If the configuration file cannot be found.
    """
    try:
        with config_path.open("r", encoding="utf-8") as f:
            make_dict = yaml.safe_load(f)
    except FileNotFoundError:
        _logger.exception("Configuration file not found: %s", config_path)
        raise

    make_config = MakeEnvConfig(**make_dict)
    gym_config = GymEnvConfig(make_env_config=make_config)
    return GymEnvLoader(gym_config)


def _save_image(image_array: np.ndarray, output_path: Path) -> None:
    """Saves a NumPy RGB array as a PNG image.

    Args:
        image_array: Array containing RGB pixel data.
        output_path: Target filesystem path for the output PNG.
    """
    image = Image.fromarray(image_array.astype(np.uint8))
    image.save(output_path)
    print(f"Saved rendering to: {output_path}")


def main(argv: Sequence[str]) -> None:
    """Renders and exports initial state views of Fetch Reach-Avoid environment.

    Args:
        argv: Command-line arguments.

    Raises:
        app.UsageError: If superfluous positional arguments are provided.
        RuntimeError: If the environment fails to render an initial frame.
    """
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    config_path = Path(_YAML_CONFIG_PATH.value)
    output_dir = Path(_OUTPUT_DIR.value)
    model_name = _MODEL_NAME.value
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = _load_env_loader(config_path)
    env = loader.env()
    try:
        env.reset()
        base_env: FetchReachAvoidEnv = env.unwrapped  # type: ignore[assignment]
        if hasattr(base_env, "render_multiview"):
            views = base_env.render_multiview()
            combined_frame = views["combined"]
            default_frame = views["default"]
            topdown_frame: np.ndarray | None = views["top_down"]
        else:
            initial_frame = env.render()
            if initial_frame is None:
                raise RuntimeError(
                    "Failed to render the initial environment frame."
                )
            combined_frame = np.asarray(initial_frame)
            default_frame = combined_frame
            topdown_frame = None

        _save_image(combined_frame, output_dir / f"{model_name}_initial.png")
        _save_image(
            default_frame, output_dir / f"{model_name}_initial_default.png"
        )
        if topdown_frame is not None:
            _save_image(
                topdown_frame, output_dir / f"{model_name}_initial_topdown.png"
            )
    finally:
        env.close()


if __name__ == "__main__":
    app.run(main)
