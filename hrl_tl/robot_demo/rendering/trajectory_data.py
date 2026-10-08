"""Data structures and geometry for robot demonstration trajectories."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field

_LOGGER: logging.Logger = logging.getLogger(__name__)


class TrajectoryPoint(BaseModel):
    """A single measured or planned trajectory point."""

    x: float
    y: float
    yaw: float = 0.0
    stamp: float = 0.0
    received_at: float = 0.0


def _normalize_bounds(v: Any) -> Any:
    """Normalize bounds from a sequence [min_x, max_x, min_y, max_y] to a dict."""
    if isinstance(v, (Sequence, tuple, list)) and len(v) == 4:
        return {
            "min_x": float(v[0]),
            "max_x": float(v[1]),
            "min_y": float(v[2]),
            "max_y": float(v[3]),
        }
    return v


def _normalize_center(v: Any) -> tuple[float, float]:
    """Normalize a 2D center coordinate from a mapping or sequence to a tuple.

    Args:
        v: Center coordinates represented as a mapping or 2-element sequence.

    Returns:
        A tuple of (x, y) coordinates.

    Raises:
        ValueError: If coordinate format cannot be parsed.
    """
    if isinstance(v, Mapping):
        return (float(v["x"]), float(v["y"]))
    if isinstance(v, (Sequence, tuple, list)) and len(v) == 2:
        return (float(v[0]), float(v[1]))
    raise ValueError(f"Invalid center coordinate format: {v}")


def _normalize_point2d(v: Any) -> Any:
    """Normalize a 2D point from a sequence [x, y] to a dict."""
    if isinstance(v, (Sequence, tuple, list)) and len(v) >= 2:
        return {"x": float(v[0]), "y": float(v[1])}
    return v


def generate_perimeter_walls(
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    tile_size: float = 0.3048,
) -> list[list[tuple[float, float]]]:
    """Generate perimeter wall tile polygons enclosing the arena bounds.

    Wall tiles are generated with tile_size (default 0.3048 m = 1 ft), forming a
    single-tile modular perimeter around the playable boundary.

    Args:
        min_x: Minimum x coordinate of inner arena bounds.
        max_x: Maximum x coordinate of inner arena bounds.
        min_y: Minimum y coordinate of inner arena bounds.
        max_y: Maximum y coordinate of inner arena bounds.
        tile_size: Square tile dimension in meters.

    Returns:
        List of wall tile polygons, where each polygon is four corner tuples.
    """
    walls: list[list[tuple[float, float]]] = []
    nx = round((max_x - min_x) / tile_size)
    ny = round((max_y - min_y) / tile_size)

    if nx <= 0 or ny <= 0:
        return [
            [
                (min_x - tile_size, max_y),
                (max_x + tile_size, max_y),
                (max_x + tile_size, max_y + tile_size),
                (min_x - tile_size, max_y + tile_size),
            ],
            [
                (min_x - tile_size, min_y - tile_size),
                (max_x + tile_size, min_y - tile_size),
                (max_x + tile_size, min_y),
                (min_x - tile_size, min_y),
            ],
            [
                (min_x - tile_size, min_y),
                (min_x, min_y),
                (min_x, max_y),
                (min_x - tile_size, max_y),
            ],
            [
                (max_x, min_y),
                (max_x + tile_size, min_y),
                (max_x + tile_size, max_y),
                (max_x, max_y),
            ],
        ]

    # Top wall row.
    for i in range(nx + 2):
        x0 = min_x - tile_size + i * tile_size
        x1 = x0 + tile_size
        y0 = max_y
        y1 = max_y + tile_size
        walls.append([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

    # Side walls (left and right).
    for j in range(ny):
        y1 = max_y - j * tile_size
        y0 = y1 - tile_size
        walls.append(
            [
                (min_x - tile_size, y0),
                (min_x, y0),
                (min_x, y1),
                (min_x - tile_size, y1),
            ]
        )
        walls.append(
            [
                (max_x, y0),
                (max_x + tile_size, y0),
                (max_x + tile_size, y1),
                (max_x, y1),
            ]
        )

    # Bottom wall row.
    for i in range(nx + 2):
        x0 = min_x - tile_size + i * tile_size
        x1 = x0 + tile_size
        y0 = min_y - tile_size
        y1 = min_y
        walls.append([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

    return walls


class ArenaBounds(BaseModel):
    """Bounding box coordinate limits of the arena."""

    min_x: float
    max_x: float
    min_y: float
    max_y: float
    margin: float = 0.0


BoundsType = Annotated[ArenaBounds, BeforeValidator(_normalize_bounds)]


class ArenaZone(BaseModel):
    """A colored circular goal or obstacle zone."""

    color: str
    center: Annotated[tuple[float, float], BeforeValidator(_normalize_center)]
    radius: float


class ArenaWorld(BaseModel):
    """World specification of arena walls, zones, and bounds."""

    bounds: BoundsType
    zones: list[ArenaZone]
    walls: list[list[tuple[float, float]]] = Field(default_factory=list)

    def model_post_init(self, context: Any, /) -> None:
        """Initialize default perimeter wall tiles if not provided."""
        if not self.walls:
            self.walls = generate_perimeter_walls(
                self.bounds.min_x,
                self.bounds.max_x,
                self.bounds.min_y,
                self.bounds.max_y,
            )


class Point2D(BaseModel):
    """A 2D coordinate point."""

    x: float
    y: float


Point2DType = Annotated[Point2D, BeforeValidator(_normalize_point2d)]


class DemoTrajectoryData(BaseModel):
    """Complete demo trajectory log data loaded from JSON."""

    source: str = ""
    run_id: str = ""
    cut_rule: str = ""
    white_goal_world_m: Point2DType | None = None
    white_goal_radius_m: float = 0.0
    measured_hit_white: bool = False
    open_loop_hit_white: bool = False
    measured_trajectory_world_m: list[TrajectoryPoint]
    open_loop_trajectory_world_m: list[TrajectoryPoint] = Field(
        default_factory=list
    )
    arena_world_m: ArenaWorld


def load_trajectory_data(file_path: Path) -> DemoTrajectoryData:
    """Load and validate demo trajectory data from a JSON file.

    Args:
        file_path: Path to the JSON trajectory log.

    Returns:
        The validated DemoTrajectoryData container.

    Raises:
        FileNotFoundError: If file_path does not exist.
        ValueError: If file content does not conform to DemoTrajectoryData.
    """
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return DemoTrajectoryData.model_validate_json(f.read())
    except FileNotFoundError:
        _LOGGER.exception("Trajectory file not found: %s", file_path)
        raise
    except Exception as e:
        _LOGGER.exception("Failed to parse trajectory JSON: %s", file_path)

        raise ValueError(
            f"Invalid trajectory data format in {file_path}: {e}"
        ) from e
