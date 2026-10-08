from __future__ import annotations

import numpy as np
from absl.testing import absltest, parameterized

from hrl_tl.robot_demo.control.smoothing import (
    ActionSmoothingFilter,
    SmoothingMethod,
    create_smoothing_filter,
)


class ActionSmoothingFilterTest(parameterized.TestCase):
    def test_ema_smoothing_filters_alternating_inputs(self) -> None:
        filt = ActionSmoothingFilter(
            method=SmoothingMethod.EMA,
            alpha=0.5,
        )
        first_out = filt.step(np.array([1.0, 0.0]))
        np.testing.assert_allclose(first_out, np.array([1.0, 0.0]))

        # (0.5 * [-1.0, 0.0] + 0.5 * [1.0, 0.0]) = [0.0, 0.0]
        second_out = filt.step(np.array([-1.0, 0.0]))
        np.testing.assert_allclose(second_out, np.array([0.0, 0.0]))

    def test_butterworth_filter_initialization_and_steady_state(self) -> None:
        filt = ActionSmoothingFilter(
            method=SmoothingMethod.BUTTERWORTH,
            cutoff_hz=1.0,
            sample_rate_hz=10.0,
            order=2,
        )
        constant_input = np.array([2.0, -1.0])
        first_out = filt.step(constant_input)
        np.testing.assert_allclose(first_out, constant_input)

        # Constant signal should remain constant under steady state
        out = first_out
        for _ in range(25):
            out = filt.step(constant_input)
        np.testing.assert_allclose(out, constant_input, atol=1e-3)

    def test_filter_reset_reinitializes_state(self) -> None:
        filt = ActionSmoothingFilter(
            method=SmoothingMethod.EMA,
            alpha=0.5,
        )
        filt.step(np.array([1.0, 1.0]))
        filt.step(np.array([0.0, 0.0]))
        filt.reset()

        # After reset, first step acts as initial state without prior memory
        fresh_out = filt.step(np.array([5.0, -5.0]))
        np.testing.assert_allclose(fresh_out, np.array([5.0, -5.0]))

    def test_none_method_passes_through_unchanged(self) -> None:
        filt = ActionSmoothingFilter(method=SmoothingMethod.NONE)
        inp = np.array([3.14, -2.71])
        out = filt.step(inp)
        np.testing.assert_allclose(out, inp)

    @parameterized.named_parameters(
        ("invalid_alpha_zero", 0.0),
        ("invalid_alpha_negative", -0.5),
        ("invalid_alpha_large", 1.2),
    )
    def test_ema_rejects_invalid_alpha(self, alpha: float) -> None:
        with self.assertRaises(ValueError):
            ActionSmoothingFilter(
                method=SmoothingMethod.EMA,
                alpha=alpha,
            )

    def test_butterworth_rejects_frequency_above_nyquist(self) -> None:
        with self.assertRaises(ValueError):
            ActionSmoothingFilter(
                method=SmoothingMethod.BUTTERWORTH,
                cutoff_hz=5.0,
                sample_rate_hz=10.0,
            )

    def test_primitive_zone_integration_with_smoothing(self) -> None:
        from pathlib import Path

        from hrl_tl.robot_demo import pose
        from hrl_tl.robot_demo.world import arena, primitive

        config_path = Path(
            "configs/zone/env/button/button_ang8_vel5_rand_ag_250_st2_0.25.yaml"
        )
        layout = arena.prepare(config_path, seed=383)
        filt = ActionSmoothingFilter(method=SmoothingMethod.EMA, alpha=0.5)
        zone = primitive.PrimitiveZone(layout, smoothing_filter=filt)

        cmd1 = zone.plan(np.array([0, 4]))
        measured = pose.Pose2D(
            x=cmd1.target.x,
            y=cmd1.target.y,
            yaw=0.0,
            stamp=1.0,
            received_at=1.0,
        )
        zone.complete(measured)

        # Action 4 is opposite direction from action 0
        cmd2 = zone.plan(np.array([4, 4]))
        dx = cmd2.native_target.x - cmd1.target.x
        self.assertAlmostEqual(dx, 0.0, places=4)

    def test_create_smoothing_filter_factory(self) -> None:
        self.assertIsNone(create_smoothing_filter("none"))
        self.assertIsNone(create_smoothing_filter(SmoothingMethod.NONE))

        ema_filter = create_smoothing_filter("EMA", alpha=0.5)
        self.assertIsNotNone(ema_filter)
        self.assertEqual(ema_filter.method, SmoothingMethod.EMA)
        self.assertEqual(ema_filter.alpha, 0.5)

        butter_filter = create_smoothing_filter("Butterworth", cutoff_hz=2.0)
        self.assertIsNotNone(butter_filter)
        self.assertEqual(butter_filter.method, SmoothingMethod.BUTTERWORTH)
        self.assertEqual(butter_filter.cutoff_hz, 2.0)


if __name__ == "__main__":
    absltest.main()
