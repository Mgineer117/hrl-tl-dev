"""CLI entry point for plotting robot execution trajectory."""

from __future__ import annotations

import collections.abc
from pathlib import Path

from absl import app, flags, logging

from hrl_tl.robot_demo.rendering import trajectory_data, trajectory_plotter

_INPUT_PATH = flags.DEFINE_string(
    "input_path",
    default="out/demo/trial_1/trajectories.json",
    help="Path to the JSON file containing demo trajectories.",
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    default="",
    help=(
        "Directory where trajectory plots will be saved. Defaults to input"
        " file directory."
    ),
)
_OUTPUT_NAME = flags.DEFINE_string(
    "output_name",
    default="robot_trajectory",
    help="Base filename for generated trajectory plots without extension.",
)
_ARROW_SPACING_M = flags.DEFINE_float(
    "arrow_spacing_m",
    default=0.7,
    help="Target distance in meters between directional arrows along the path.",
    lower_bound=0.1,
)
_LINE_COLOR = flags.DEFINE_string(
    "line_color",
    default="#d55e00",
    help="Hex color string for the robot trajectory line.",
)
_LINE_WIDTH = flags.DEFINE_float(
    "line_width",
    default=2.0,
    help="Line width for the robot trajectory.",
    lower_bound=0.5,
)
_INITIAL_COLOR = flags.DEFINE_string(
    "initial_color",
    default="#0072b2",
    help="Hex color string for the initial position marker.",
)
_DPI = flags.DEFINE_integer(
    "dpi",
    default=300,
    help="Output DPI for rasterized PNG export.",
    lower_bound=72,
)


def main(argv: collections.abc.Sequence[str]) -> None:
    """Main CLI entry point for trajectory plotting."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    input_path_str = _INPUT_PATH.value
    if not input_path_str:
        raise app.UsageError("--input_path cannot be empty.")
    input_path = Path(input_path_str).resolve()

    output_name = _OUTPUT_NAME.value
    if not output_name:
        raise app.UsageError("--output_name cannot be empty.")

    output_dir_str = _OUTPUT_DIR.value
    output_dir = (
        Path(output_dir_str).resolve() if output_dir_str else input_path.parent
    )
    output_base = output_dir / output_name

    logging.info("Loading trajectory from %s", input_path)
    data = trajectory_data.load_trajectory_data(input_path)

    png_path = trajectory_plotter.generate_trajectory_plot(
        data=data,
        output_base=output_base,
        arrow_spacing_m=_ARROW_SPACING_M.value or 0.7,
        line_color=_LINE_COLOR.value or "#d55e00",
        line_width=_LINE_WIDTH.value or 2.0,
        initial_color=_INITIAL_COLOR.value or "#0072b2",
        dpi=_DPI.value or 300,
    )
    logging.info("Saved PNG plot: %s", png_path)


if __name__ == "__main__":
    app.run(main)
