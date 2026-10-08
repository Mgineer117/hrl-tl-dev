from __future__ import annotations

from absl.testing import absltest, parameterized

from hrl_tl.wrappers.utils import (
    base_n,
    generate_all_specifications,
    weights2ltl,
)


class BaseNTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("binary_no_pad", 10, 2, None, "1010"),
        ("binary_padded", 10, 2, 6, "001010"),
        ("ternary_no_pad", 18, 3, None, "200"),
        ("ternary_padded", 10, 3, 5, "00101"),
    )
    def test_base_n(
        self, num_10: int, n: int, width: int | None, expected: str
    ) -> None:
        result = base_n(num_10, n, width)
        self.assertEqual(result, expected)


class Weights2LTLTest(parameterized.TestCase):
    @parameterized.named_parameters(
        (
            "f_and_g_clauses",
            [[1, 0], [0, 1]],
            [[2, 1]],
            ["p2", "p1"],
            "F(p1 & p2) & G(!p1 | p2)",
        ),
        (
            "g_clause_only",
            [],
            [[2, 0]],
            ["p1", "p2"],
            "G(!p1)",
        ),
        (
            "multi_clause_complex",
            [[1, 0, 0], [0, 2, 1]],
            [[0, 1, 0], [2, 0, 0], [0, 2, 2]],
            ["p3", "p1", "p2"],
            "F(p1 & (!p2 | p3)) & G(p2 & !p1 & (!p2 | !p3))",
        ),
    )
    def test_weights2ltl(
        self,
        f_weights: list[list[int]],
        g_weights: list[list[int]],
        predicates: list[str],
        expected_tl_spec: str,
    ) -> None:
        tl_spec = weights2ltl(f_weights, g_weights, predicates)
        self.assertEqual(tl_spec, expected_tl_spec)


class GenerateSpecificationsTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("single_predicate", ["p1"], 4),
        ("two_predicates", ["p1", "p2"], 32),
    )
    def test_generate_all_specifications(
        self, predicates: list[str], expected_count: int
    ) -> None:
        all_specs = generate_all_specifications(predicates, num_processes=2)
        self.assertLen(all_specs, expected_count)
        self.assertLen(set(all_specs), expected_count)


if __name__ == "__main__":
    absltest.main()
