from __future__ import annotations

import numpy as np
from absl.testing import absltest, parameterized

from hrl_tl.envs.fetch.action import FetchMultiDiscreteAction
from hrl_tl.wrappers.low_level_policies.gc_ltl import ActionSummer


class FetchMultiDiscreteActionTest(parameterized.TestCase):
    @parameterized.named_parameters(
        (
            "unbatched_interpolation",
            np.array([4, 2, 2], dtype=np.int64),
            np.array([2, 0, 2], dtype=np.int64),
            0.5,
            np.array([3, 1, 2], dtype=np.int64),
        ),
        (
            "batched_clipping",
            np.array([[4, 4, 4], [0, 0, 0]], dtype=np.int64),
            np.array([[4, 4, 4], [0, 0, 0]], dtype=np.int64),
            0.8,
            np.array([[4, 4, 4], [0, 0, 0]], dtype=np.int64),
        ),
    )
    def test_fetch_multidiscrete_action_sum(
        self,
        a1: np.ndarray,
        a2: np.ndarray,
        weight: float,
        expected: np.ndarray,
    ) -> None:
        action_mode = FetchMultiDiscreteAction(num_bins=5)
        result = action_mode.sum_actions(a1, a2, weight=weight)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.dtype, np.int64)

    def test_action_summer_protocol_conformance(self) -> None:
        self.assertIsInstance(FetchMultiDiscreteAction(5), ActionSummer)


if __name__ == "__main__":
    absltest.main()
