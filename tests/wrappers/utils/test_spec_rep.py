from __future__ import annotations

from typing import Any

import numpy as np
from absl.testing import absltest, parameterized
from gymnasium import spaces
from pydantic import ValidationError

from hrl_tl.wrappers.utils.spec_rep import (
    CPCParamLv1MinimalSpecRep,
    Lv1MinimalSpecRep,
)


class Lv1MinimalSpecRepTest(parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.predicates = ["psi_y", "psi_r", "psi_w"]
        self.spec_rep = Lv1MinimalSpecRep(
            predicate_names=self.predicates, num_clauses=1
        )

    def test_action_space_structure(self) -> None:
        action_space = self.spec_rep.action_space
        self.assertIsInstance(action_space, spaces.MultiDiscrete)
        assert isinstance(action_space, spaces.MultiDiscrete)
        self.assertEqual(list(action_space.nvec), [4, 4])

    @parameterized.named_parameters(
        ("reach_avoid", np.array([3, 2], dtype=np.int64), "Fpsi_w & G!psi_r"),
        ("reach_only", np.array([1, 0], dtype=np.int64), "Fpsi_y"),
        ("empty_action", np.array([0, 0], dtype=np.int64), ""),
    )
    def test_weights2ltl_translation(
        self, action: np.ndarray, expected_spec: str
    ) -> None:
        tl_spec = self.spec_rep.weights2ltl(action)
        self.assertEqual(tl_spec, expected_spec)


class CPCParamLv1MinimalSpecRepTest(parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.predicates = ["psi_y", "psi_r", "psi_w"]

    @parameterized.named_parameters(
        ("omit_L", {"fixed_L_gain": 7.47}, "L_gain", 7.47),
        ("omit_k", {"fixed_k_steepness": 0.286}, "k_steepness", 0.286),
        ("omit_eps", {"fixed_eps_margin": 2.65}, "eps_margin", 2.65),
    )
    def test_parameter_omission(
        self,
        args: dict[str, Any],
        omitted_param: str,
        expected_value: float,
    ) -> None:
        spec_rep = CPCParamLv1MinimalSpecRep(
            predicate_names=self.predicates, num_clauses=1, args=args
        )
        self.assertEqual(len(spec_rep.action_space.nvec), 4)
        policy_args = spec_rep.action2policy_args(np.array([1, 2, 5, 5]))
        self.assertEqual(
            policy_args["lambda_config"][omitted_param], expected_value
        )

    def test_conflicting_configuration_raises_error(self) -> None:
        with self.assertRaises(ValidationError):
            CPCParamLv1MinimalSpecRep(
                predicate_names=self.predicates,
                num_clauses=1,
                args={
                    "L_gain": {"range": (0.0, 10.0), "num_discrete_bins": 11},
                    "fixed_L_gain": 7.47,
                },
            )


if __name__ == "__main__":
    absltest.main()
