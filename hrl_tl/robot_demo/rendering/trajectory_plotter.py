"""Plotting and visualization routines for robot demonstration trajectories."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

from hrl_tl.robot_demo.rendering.trajectory_data import (
    ArenaWorld,
    DemoTrajectoryData,
    TrajectoryPoint,
)

_ZONE_COLORS: dict[str, str] = {
    "yellow": "#ffd900",
    "red": "#d16100",
    "white": "#ffffff",
    "black": "#050505",
}


def compute_arrow_tangents(
    points: Sequence[TrajectoryPoint],
    spacing_m: float,
) -> list[tuple[float, float, float, float]]:
    """Compute arrow positions and unit tangent vectors along the path.

    Positions are sampled based on cumulative path length. Tangents are
    computed across a small local window to suppress sensor noise.

    Args:
        points: Ordered sequence of trajectory coordinates.
        spacing_m: Approximate distance along path between consecutive arrows.

    Returns:
        List of tuples (px, py, vx, vy) where (px, py) is arrow anchor and
        (vx, vy) is normalized heading vector.
    """
    if len(points) < 2:
        return []

    x = np.array([p.x for p in points])
    y = np.array([p.y for p in points])
    dx = np.diff(x)
    dy = np.diff(y)
    dists = np.hypot(dx, dy)
    cum_dist = np.insert(np.cumsum(dists), 0, 0.0)
    total_dist = float(cum_dist[-1])

    if total_dist <= 0.3:
        return []

    first_arrow_dist = min(0.2, total_dist * 0.1)
    target_dists = [first_arrow_dist]
    current_dist = max(spacing_m, 0.8)
    while current_dist < total_dist - 0.25:
        target_dists.append(current_dist)
        current_dist += spacing_m

    arrows: list[tuple[float, float, float, float]] = []
    window_m = 0.12
    for td in target_dists:
        d_fwd = min(total_dist, td + window_m)
        d_bwd = max(0.0, td - window_m)
        px = float(np.interp(td, cum_dist, x))
        py = float(np.interp(td, cum_dist, y))
        xf = float(np.interp(d_fwd, cum_dist, x))
        xb = float(np.interp(d_bwd, cum_dist, x))
        yf = float(np.interp(d_fwd, cum_dist, y))
        yb = float(np.interp(d_bwd, cum_dist, y))
        vx = xf - xb
        vy = yf - yb
        vlen = math.hypot(vx, vy)
        if vlen > 1e-4:
            arrows.append((px, py, vx / vlen, vy / vlen))

    return arrows


def draw_arena(ax: plt.Axes, arena: ArenaWorld) -> None:
    """Draw arena walls and colored zones on the axes.

    Args:
        ax: Matplotlib axes to render into.
        arena: Arena geometry specification.
    """
    for corners in arena.walls:
        ax.add_patch(
            Polygon(
                corners,
                facecolor="#d8dadd",
                edgecolor="#858a91",
                linewidth=0.45,
            )
        )

    for zone in arena.zones:
        color = _ZONE_COLORS.get(zone.color, zone.color)
        ax.add_patch(
            Circle(
                zone.center,
                zone.radius,
                facecolor=color,
                edgecolor="#30343b",
                linewidth=0.65,
                zorder=2,
            )
        )


def plot_measured_trajectory(
    ax: plt.Axes,
    path: Sequence[TrajectoryPoint],
    arrow_spacing_m: float,
    line_color: str,
    line_width: float,
    initial_color: str,
) -> None:
    """Render the measured trajectory line, directional arrows, and endpoints.

    Args:
        ax: Matplotlib axes to render into.
        path: Ordered list of measured robot trajectory points.
        arrow_spacing_m: Interval distance in meters between arrows.
        line_color: Trajectory line and arrow color.
        line_width: Trajectory stroke width.
        initial_color: Color of the initial state marker.
    """
    if not path:
        return

    ax.plot(
        [p.x for p in path],
        [p.y for p in path],
        color=line_color,
        linewidth=line_width,
        zorder=3,
    )

    start = path[0]
    ax.scatter(
        start.x,
        start.y,
        color=initial_color,
        s=70,
        zorder=6,
        edgecolors="#17191c",
        linewidths=0.6,
    )

    end = path[-1]
    ax.scatter(
        end.x,
        end.y,
        color=line_color,
        marker="x",
        s=70,
        zorder=5,
        linewidths=1.8,
    )

    arrows = compute_arrow_tangents(path, arrow_spacing_m)
    arrow_half_len = 0.04
    for px, py, vx, vy in arrows:
        ax.annotate(
            "",
            xy=(px + vx * arrow_half_len, py + vy * arrow_half_len),
            xytext=(px - vx * arrow_half_len, py - vy * arrow_half_len),
            arrowprops={
                "arrowstyle": "-|>,head_width=0.35,head_length=0.6",
                "facecolor": line_color,
                "edgecolor": line_color,
                "lw": line_width,
            },
            zorder=4,
        )


def generate_trajectory_plot(
    data: DemoTrajectoryData,
    output_base: Path,
    arrow_spacing_m: float = 0.7,
    line_color: str = "#d55e00",
    line_width: float = 2.0,
    initial_color: str = "#0072b2",
    dpi: int = 300,
) -> Path:
    """Generate and export a trajectory comparison plot as PNG.

    Args:
        data: Deserialized demo trajectory log.
        output_base: Base output path without extension.
        arrow_spacing_m: Interval distance in meters between arrows.
        line_color: Trajectory line and arrow color.
        line_width: Trajectory stroke width.
        initial_color: Color of the initial state marker.
        dpi: DPI resolution for raster image export.

    Returns:
        The generated PNG file path.
    """
    fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")

    arena = data.arena_world_m
    draw_arena(ax, arena)
    plot_measured_trajectory(
        ax,
        data.measured_trajectory_world_m,
        arrow_spacing_m=arrow_spacing_m,
        line_color=line_color,
        line_width=line_width,
        initial_color=initial_color,
    )

    if arena.walls:
        all_wall_points = [pt for wall in arena.walls for pt in wall]
        margin = -0.25
        min_x = min(pt[0] for pt in all_wall_points) - margin
        max_x = max(pt[0] for pt in all_wall_points) + margin
        min_y = min(pt[1] for pt in all_wall_points) - margin
        max_y = max(pt[1] for pt in all_wall_points) + margin
    else:
        margin = 0.05
        min_x = arena.bounds.min_x - margin
        max_x = arena.bounds.max_x + margin
        min_y = arena.bounds.min_y - margin
        max_y = arena.bounds.max_y + margin

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect("equal")
    ax.set_axis_off()

    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    return png_path
