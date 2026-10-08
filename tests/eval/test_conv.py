from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from absl.testing import absltest, parameterized

from hrl_tl.eval.conv import (
    analyze_convergence,
    calculate_convergence_metrics,
    detect_convergence_point,
    linear_regression,
    moving_average,
)


def _make_converged_returns(
    n_episodes: int = 500, seed: int = 42
) -> np.ndarray:
    np.random.seed(seed)
    base = np.linspace(-50, 100, n_episodes)
    noise = np.random.normal(0, 5, n_episodes)
    returns = base + noise
    conv_start = int(0.7 * n_episodes)
    converged_value = np.mean(returns[conv_start:])
    returns[conv_start:] = converged_value + np.random.normal(
        0, 1.0, n_episodes - conv_start
    )
    return returns.astype(np.float64)


def _make_non_converged_returns(
    n_episodes: int = 500, seed: int = 43
) -> np.ndarray:
    np.random.seed(seed)
    return (
        np.linspace(0, 50, n_episodes) + np.random.normal(0, 20, n_episodes)
    ).astype(np.float64)


def _make_declining_returns(
    n_episodes: int = 500, seed: int = 44
) -> np.ndarray:
    np.random.seed(seed)
    return (
        np.linspace(100, -50, n_episodes) + np.random.normal(0, 5, n_episodes)
    ).astype(np.float64)


class MovingAverageTest(parameterized.TestCase):
    def test_moving_average_basic(self) -> None:
        data = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        result = moving_average(data, window=3)
        expected = np.array([2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_moving_average_window_equals_data(self) -> None:
        data = np.array([1, 2, 3, 4], dtype=np.float64)
        result = moving_average(data, window=4)
        expected = np.array([2.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_moving_average_window_larger_than_data(self) -> None:
        data = np.array([1, 2, 3], dtype=np.float64)
        result = moving_average(data, window=5)
        self.assertEmpty(result)


class LinearRegressionTest(parameterized.TestCase):
    @parameterized.named_parameters(
        (
            "perfect_fit",
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 4.0, 6.0, 8.0, 10.0],
            2.0,
            0.0,
            1.0,
        ),
        (
            "flat_line",
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 5.0, 5.0, 5.0, 5.0],
            0.0,
            5.0,
            0.0,
        ),
        (
            "single_point",
            [1.0],
            [5.0],
            0.0,
            5.0,
            0.0,
        ),
    )
    def test_linear_regression(
        self,
        x_vals: list[float],
        y_vals: list[float],
        expected_slope: float,
        expected_intercept: float,
        expected_r_squared: float,
    ) -> None:
        slope, intercept, r_squared = linear_regression(
            np.array(x_vals, dtype=np.float64),
            np.array(y_vals, dtype=np.float64),
        )
        self.assertAlmostEqual(slope, expected_slope, places=7)
        self.assertAlmostEqual(intercept, expected_intercept, places=7)
        self.assertAlmostEqual(r_squared, expected_r_squared, places=7)


class ConvergenceMetricsTest(absltest.TestCase):
    def test_metrics_converged_data(self) -> None:
        metrics = calculate_convergence_metrics(_make_converged_returns())
        self.assertGreater(metrics.trend_slope, 0.0)
        self.assertLess(metrics.recent_cv, 0.1)

    def test_metrics_declining_data(self) -> None:
        metrics = calculate_convergence_metrics(_make_declining_returns())
        self.assertLess(metrics.trend_slope, 0.0)

    def test_metrics_small_data(self) -> None:
        small_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        metrics = calculate_convergence_metrics(small_data, window=10)
        self.assertAlmostEqual(metrics.overall_mean, 3.0)


class ConvergenceDetectionTest(absltest.TestCase):
    def test_detect_convergence_converged_data(self) -> None:
        conv_point, has_converged = detect_convergence_point(
            _make_converged_returns(), threshold=0.1, window=50
        )
        self.assertTrue(has_converged)
        self.assertGreater(conv_point, 0)

    def test_detect_convergence_non_converged_data(self) -> None:
        conv_point, has_converged = detect_convergence_point(
            _make_non_converged_returns(), threshold=0.05, window=50
        )
        self.assertFalse(has_converged)
        self.assertEqual(conv_point, 500)

    def test_detect_convergence_small_data(self) -> None:
        small_data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        conv_point, has_converged = detect_convergence_point(
            small_data, threshold=0.1, window=50
        )
        self.assertFalse(has_converged)
        self.assertEqual(conv_point, len(small_data))


class AnalyzeConvergenceTest(absltest.TestCase):
    def test_analyze_convergence_1d_returns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "returns.npz"
            np.savez(str(file_path), results=_make_converged_returns())

            results = analyze_convergence(str(file_path), plot=False)
            self.assertTrue(results.has_converged)
            self.assertEqual(results.total_episodes, 500)

    def test_analyze_convergence_2d_returns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "returns_2d.npz"
            returns_2d = np.ones((5, 100), dtype=np.float64) * 42.0
            np.savez(str(file_path), results=returns_2d)

            results = analyze_convergence(str(file_path), plot=False)
            self.assertEqual(results.total_episodes, 100)
            self.assertAlmostEqual(float(np.mean(results.returns)), 42.0)

    def test_analyze_convergence_custom_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "custom.npz"
            np.savez(str(file_path), returns=_make_converged_returns())

            results = analyze_convergence(
                str(file_path),
                convergence_threshold=0.05,
                convergence_window=30,
                plot=False,
            )
            self.assertIsInstance(results.convergence_point, int)


if __name__ == "__main__":
    absltest.main()
