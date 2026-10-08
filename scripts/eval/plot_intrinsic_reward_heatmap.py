"""Script for plotting intrinsic reward heatmaps from trained ALLO models."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import yaml
from absl import app, flags
from gymnasium import spaces
from gymnasium.spaces import utils as space_utils
from rl_pipeline.sb3 import (
    SB3Pipeline,
    SB3PipelineConfigReader,
    SB3ReplicatePipelineConfigReader,
)

CheckpointType = int | Literal["latest", "final", "best"]

_TRAIN_CONFIG = flags.DEFINE_string(
    "train_config",
    default="configs/zone/train/allo/7.g.c_allo_ext.yaml",
    help="Path to the training pipeline YAML configuration file.",
)
_REPLICATE_INDEX = flags.DEFINE_integer(
    "replicate_index",
    default=0,
    help="Replicate index to load when using replicate configuration.",
    lower_bound=0,
)
_CHECKPOINT = flags.DEFINE_string(
    "checkpoint",
    default="best",
    help="Model checkpoint specifier ('latest', 'final', 'best', or timestep int).",
)
_AGENT_NAME = flags.DEFINE_string(
    "agent_name",
    default=None,
    help="Name of the agent to evaluate. Defaults to the first agent.",
)
_RESOLUTION = flags.DEFINE_string(
    "resolution",
    default="0.25",
    help="Grid sampling resolution (scalar float or comma-separated x,y pair).",
)
_ENV_SEED = flags.DEFINE_integer(
    "env_seed",
    default=None,
    help="Optional seed for environment reset to ensure deterministic layout.",
)
_EIGENVECTOR_INDEX = flags.DEFINE_string(
    "eigenvector_index",
    default="all",
    help="Eigenvector index to plot (integer or 'all').",
)
_PLOT_MODE = flags.DEFINE_enum(
    "plot_mode",
    default="tricontourf",
    enum_values=["scatter", "tricontourf"],
    help="Plotting mode for heatmap.",
)
_CMAP = flags.DEFINE_string(
    "cmap",
    default="coolwarm",
    help="Colormap for the heatmap.",
)
_POINT_SIZE = flags.DEFINE_float(
    "point_size",
    default=25.0,
    help="Scatter point size.",
    lower_bound=0.1,
)
_LEVELS = flags.DEFINE_integer(
    "levels",
    default=30,
    help="Number of contour levels for tricontourf mode.",
    lower_bound=1,
)
_SAVE_PATH = flags.DEFINE_string(
    "save_path",
    default="out/plots/zone/allo/intrinsic_reward_heatmap_7.g.c.png",
    help="Path template for saving output figures.",
)
_SHOW_PLOT = flags.DEFINE_boolean(
    "show_plot",
    default=False,
    help="Whether to display the plot interactively.",
)
_TITLE = flags.DEFINE_string(
    "title",
    default=None,
    help="Optional title for the plot. If None, an automatic title is used.",
)
_DEVICE_OVERRIDE = flags.DEFINE_string(
    "device",
    default=None,
    help="PyTorch device override (e.g. 'cpu', 'cuda:0').",
)


def parse_resolution(resolution: str) -> float | tuple[float, float]:
    """Parses a resolution string into a scalar or (x, y) tuple.

    Args:
      resolution: String containing either a scalar or comma-separated steps.

    Returns:
      A float or a (x_step, y_step) tuple.

    Raises:
      ValueError: If resolution is non-positive or formatted incorrectly.
    """
    value = resolution.strip()
    if "," in value:
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 2:
            raise ValueError(
                "Resolution tuple must have exactly two comma-separated values, e.g. 0.5,1.0."
            )
        x_step = float(parts[0])
        y_step = float(parts[1])
        if x_step <= 0 or y_step <= 0:
            raise ValueError("Resolution values must be positive.")
        return (x_step, y_step)

    scalar = float(value)
    if scalar <= 0:
        raise ValueError("Resolution must be positive.")
    return scalar


def parse_checkpoint(checkpoint: str) -> CheckpointType:
    """Parses checkpoint specifier string into CheckpointType.

    Args:
      checkpoint: Checkpoint string ('latest', 'final', 'best', or timestep).

    Returns:
      Parsed checkpoint identifier.

    Raises:
      ValueError: If checkpoint string cannot be parsed.
    """
    value = checkpoint.strip().lower()
    if value in ("latest", "final", "best"):
        return value

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            "Checkpoint must be one of: latest, final, best, or an integer timestep."
        ) from exc


def try_register_contgrid() -> None:
    """Registers contgrid environments if available."""
    try:
        __import__("contgrid")
    except ModuleNotFoundError:
        pass


def find_state_enumerator_env(env: Any) -> Any:
    """Finds underlying environment providing all_possible_states_at_resolution.

    Args:
      env: Wrapped or unwrapped Gymnasium environment.

    Returns:
      Environment instance exposing all_possible_states_at_resolution.

    Raises:
      AttributeError: If no matching environment can be found in wrapper stack.
    """
    if hasattr(env, "all_possible_states_at_resolution"):
        return env

    visited: set[int] = set()
    stack = [env]

    while stack:
        current = stack.pop()
        current_id = id(current)
        if current_id in visited:
            continue
        visited.add(current_id)

        if hasattr(current, "all_possible_states_at_resolution"):
            return current

        for attr in ("env", "unwrapped"):
            inner = getattr(current, attr, None)
            if inner is not None and id(inner) not in visited:
                stack.append(inner)

    raise AttributeError(
        "Could not find all_possible_states_at_resolution on the evaluation environment or wrapped envs."
    )


def load_model_and_env(
    train_config_path: str | Path,
    checkpoint: CheckpointType,
    replicate_index: int = 0,
) -> tuple[Any, Any]:
    """Loads the trained model and evaluation environment from config.

    Args:
      train_config_path: Path to the training pipeline YAML configuration file.
      checkpoint: Checkpoint specifier (e.g., 'final', 'best', 'latest', or timestep).
      replicate_index: Replicate index to load when using a replicate config.

    Returns:
      A tuple of (loaded_model, evaluation_environment).

    Raises:
      FileNotFoundError: If the train_config_path cannot be opened.
      IndexError: If the replicate_index is out of range.
    """
    try_register_contgrid()
    config_path = Path(train_config_path)
    with config_path.open("r", encoding="utf-8") as f:
        raw_config: Mapping[str, Any] = yaml.safe_load(f)

    if "replicate_config" in raw_config:
        rep_config = (
            SB3ReplicatePipelineConfigReader[SB3PipelineConfigReader]
            .from_yaml(str(config_path))
            .to_config()
        )
        if replicate_index >= len(rep_config.ind_pipeline_configs):
            raise IndexError(
                f"replicate_index {replicate_index} is out of range for "
                f"{len(rep_config.ind_pipeline_configs)} available replicates."
            )
        config: Any = rep_config.ind_pipeline_configs[replicate_index]
    else:
        config = SB3PipelineConfigReader.from_yaml(str(config_path)).to_config()

    pipeline = SB3Pipeline(config=config, verbose=True)
    model = pipeline.load_model(checkpoint)
    env = pipeline.env_loader.env()
    return model, env


def get_model_device(model: Any) -> th.device:
    """Infers the PyTorch device of a loaded model.

    Args:
      model: The model instance.

    Returns:
      The PyTorch device hosting model parameters.
    """
    if hasattr(model, "device"):
        return th.device(model.device)

    if hasattr(model, "feature_net"):
        return next(model.feature_net.parameters()).device

    if hasattr(model, "network"):
        return next(model.network.parameters()).device

    return th.device("cpu")


def build_observation_batch(
    state_map: dict[tuple[float, float], Any],
    observation_space: spaces.Space[Any],
) -> tuple[np.ndarray, th.Tensor]:
    """Flattens sampled state observations into a batched PyTorch tensor.

    Args:
      state_map: Mapping of (x, y) coordinates to observation dicts/arrays.
      observation_space: The Gymnasium observation space for flattening.

    Returns:
      A tuple of (positions_array, observation_tensor).

    Raises:
      ValueError: If state_map is empty.
    """
    if len(state_map) == 0:
        raise ValueError(
            "No states were returned by all_possible_states_at_resolution()."
        )

    positions: list[tuple[float, float]] = []
    flattened_observations: list[np.ndarray] = []

    for pos, obs in state_map.items():
        positions.append(pos)
        if isinstance(obs, dict):
            flat_obs = space_utils.flatten(observation_space, obs)
        else:
            flat_obs = np.asarray(obs)
        flattened_observations.append(
            np.asarray(flat_obs, dtype=np.float32).reshape(-1)
        )

    positions_array = np.asarray(positions, dtype=np.float32)
    obs_array = np.stack(flattened_observations, axis=0).astype(
        np.float32, copy=False
    )
    obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=0.0, neginf=0.0)
    obs_tensor = th.as_tensor(obs_array, dtype=th.float32)
    return positions_array, obs_tensor


def run_allo_inference(
    model: Any, obs_tensor: th.Tensor, device: th.device
) -> th.Tensor:
    """Runs forward inference on ALLO feature extractor.

    Args:
      model: Loaded ALLO model instance.
      obs_tensor: 2D PyTorch observation tensor [N, obs_dim].
      device: Target PyTorch execution device.

    Returns:
      2D feature matrix tensor [N, representation_dim].

    Raises:
      AttributeError: If model lacks feature extraction attributes.
      ValueError: If output feature tensor is not 2-dimensional.
    """
    obs_tensor = obs_tensor.to(device)

    with th.no_grad():
        if hasattr(model, "feature_net"):
            features = model.feature_net(obs_tensor)
        elif hasattr(model, "encode"):
            features = model.encode(obs_tensor)
        elif hasattr(model, "network"):
            output = model.network(obs_tensor)
            features = output[0] if isinstance(output, tuple) else output
        else:
            raise AttributeError(
                "Model does not provide feature_net, encode, or network for inference."
            )

    if not isinstance(features, th.Tensor):
        features = th.as_tensor(features, dtype=th.float32, device=device)

    if features.ndim != 2:
        raise ValueError(
            f"Expected 2D feature matrix [N, d], got shape {tuple(features.shape)}."
        )

    return features


def plot_intrinsic_reward_heatmap(
    positions: np.ndarray,
    values: np.ndarray,
    eigenvector_index: int,
    plot_mode: Literal["scatter", "tricontourf"],
    cmap: str,
    point_size: float,
    levels: int,
    title: str | None,
) -> plt.Figure:
    """Generates a 2D intrinsic reward heatmap figure for a specific eigenvector.

    Args:
      positions: 2D array of (x, y) coordinates of shape [N, 2].
      values: 1D array of eigenvector representation values of shape [N].
      eigenvector_index: Index of the eigenvector being plotted.
      plot_mode: 'scatter' or 'tricontourf'.
      cmap: Colormap name.
      point_size: Point size for scatter plot.
      levels: Contour level count for tricontourf.
      title: Optional custom plot title.

    Returns:
      Rendered Matplotlib Figure.
    """
    x = positions[:, 0]
    y = positions[:, 1]

    fig, ax = plt.subplots(figsize=(9, 7))

    if plot_mode == "tricontourf":
        if len(x) < 3:
            print(
                "[WARN] Fewer than 3 points found. Falling back to scatter plot."
            )
            scatter = ax.scatter(x, y, c=values, cmap=cmap, s=point_size)
        else:
            try:
                contour = ax.tricontourf(x, y, values, levels=levels, cmap=cmap)
                ax.scatter(
                    x,
                    y,
                    c=values,
                    cmap=cmap,
                    s=max(point_size * 0.25, 4.0),
                    alpha=0.45,
                )
                scatter = contour
            except ValueError as err:
                print(
                    f"[WARN] tricontourf failed ({err}). Falling back to scatter plot."
                )
                scatter = ax.scatter(x, y, c=values, cmap=cmap, s=point_size)
    else:
        scatter = ax.scatter(x, y, c=values, cmap=cmap, s=point_size)

    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label(f"Eigenvector {eigenvector_index} value")

    if title is None:
        title = f"Intrinsic Reward Heatmap (Eigenvector {eigenvector_index})"

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig


def main(argv: list[str]) -> None:
    """Main orchestration entry point for extracting and plotting ALLO heatmaps."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    train_config: str = _TRAIN_CONFIG.value
    replicate_index: int = _REPLICATE_INDEX.value
    checkpoint_input: str = _CHECKPOINT.value
    agent_name_input: str | None = _AGENT_NAME.value
    resolution_input: str = _RESOLUTION.value
    env_seed: int | None = _ENV_SEED.value
    eigenvector_input: str = _EIGENVECTOR_INDEX.value
    plot_mode = cast(Literal["scatter", "tricontourf"], _PLOT_MODE.value)
    cmap: str = _CMAP.value
    point_size: float = _POINT_SIZE.value
    levels: int = _LEVELS.value
    save_path: str = _SAVE_PATH.value
    show_plot: bool = _SHOW_PLOT.value
    title: str | None = _TITLE.value
    device_override: str | None = _DEVICE_OVERRIDE.value

    resolution = parse_resolution(resolution_input)
    checkpoint = parse_checkpoint(checkpoint_input)

    model, env = load_model_and_env(
        train_config, checkpoint, replicate_index=replicate_index
    )
    enum_env = find_state_enumerator_env(env)

    if env_seed is not None:
        enum_env.reset(seed=env_seed)
    else:
        enum_env.reset()

    all_states = enum_env.all_possible_states_at_resolution(resolution)
    available_agent_names = list(all_states.keys())
    if len(available_agent_names) == 0:
        raise RuntimeError(
            "No agent entries returned by all_possible_states_at_resolution()."
        )

    agent_name = agent_name_input or available_agent_names[0]
    if agent_name not in all_states:
        raise KeyError(
            f"Agent '{agent_name}' not found. Available agent names: {available_agent_names}."
        )

    state_map = all_states[agent_name]
    positions, obs_tensor = build_observation_batch(
        state_map, enum_env.observation_space
    )

    device = (
        th.device(device_override)
        if device_override is not None
        else get_model_device(model)
    )

    features = run_allo_inference(model, obs_tensor, device=device)
    feature_dim = features.shape[1]

    if eigenvector_input.strip().lower() == "all":
        eig_idx_to_plot: list[int] = list(range(feature_dim))
    else:
        eig_idx_to_plot = [int(eigenvector_input.strip())]

    output_dir = Path(save_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    env_fig: np.ndarray = enum_env.render()
    env_token: str = "env"
    env_plot_path = Path(save_path).with_name(
        f"{Path(save_path).stem}_{env_token}{Path(save_path).suffix}"
    )
    plt.imsave(env_plot_path, env_fig)
    print(f"Saved environment visualization to: {env_plot_path}")

    for idx in eig_idx_to_plot:
        if idx >= feature_dim:
            raise IndexError(
                f"eigenvector index {idx} is out of range for feature dimension {feature_dim}."
            )

        values = features[:, idx].detach().cpu().numpy()

        print(f"Loaded checkpoint: {checkpoint}")
        print(f"Agent name: {agent_name}")
        print(f"Resolution: {resolution}")
        print(f"Number of sampled states: {len(positions)}")
        print(f"Feature matrix shape: {tuple(features.shape)}")
        print(f"Using eigenvector index: {idx}")

        fig = plot_intrinsic_reward_heatmap(
            positions=positions,
            values=values,
            eigenvector_index=idx,
            plot_mode=plot_mode,
            cmap=cmap,
            point_size=point_size,
            levels=levels,
            title=title,
        )

        output_path = Path(save_path)
        eig_token = f"eig{idx}"
        if eig_token not in output_path.stem:
            output_path = output_path.with_name(
                f"{output_path.stem}_{eig_token}{output_path.suffix}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved heatmap to: {output_path}")

        if show_plot:
            plt.show()

        plt.close(fig)

    env.close()


if __name__ == "__main__":
    app.run(main)
