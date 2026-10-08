"""Unit tests for trajectory rendering data models and geometric routines."""

from __future__ import annotations

from pathlib import Path

from absl.testing import absltest, parameterized

from hrl_tl.robot_demo.rendering.trajectory_data import (
    TrajectoryPoint,
    generate_perimeter_walls,
)
from hrl_tl.robot_demo.rendering.trajectory_plotter import (
    compute_arrow_tangents,
)


class PlotTrajectoriesTest(parameterized.TestCase):
    """Verifies perimeter wall generation and directional arrow calculations."""

    def setUp(self) -> None:
        super().setUp()
        self.repo_root = Path(__file__).resolve().parents[2]

    def test_generate_perimeter_walls(self) -> None:
        """Verifies perimeter walls generate correct tile count and vertices."""
        walls = generate_perimeter_walls(
            min_x=-1.524,
            max_x=1.524,
            min_y=-1.524,
            max_y=1.524,
            tile_size=0.3048,
        )
        self.assertLen(walls, 44)
        self.assertLen(walls[0], 4)

    def test_compute_arrow_tangents(self) -> None:
        """Verifies arrow tangent vectors align with straight upward motion."""
        points = [
            TrajectoryPoint(x=0.0, y=0.0),
            TrajectoryPoint(x=0.0, y=0.5),
            TrajectoryPoint(x=0.0, y=1.0),
            TrajectoryPoint(x=0.0, y=1.5),
            TrajectoryPoint(x=0.0, y=2.0),
        ]
        arrows = compute_arrow_tangents(points, spacing_m=0.5)
        self.assertNotEmpty(arrows)
        for _, _, vx, vy in arrows:
            self.assertAlmostEqual(vx, 0.0, places=2)
            self.assertAlmostEqual(vy, 1.0, places=2)


if __name__ == "__main__":
    absltest.main()
