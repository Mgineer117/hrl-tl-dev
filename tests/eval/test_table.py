from __future__ import annotations

from absl.testing import absltest, parameterized

from hrl_tl.eval.table import (
    RewardEntry,
    format_cell,
    format_num,
    generate_latex_table,
)


class GenerateBaselineTableTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("large_val_int_only", 100.0, 3, 2, "100"),
        ("two_digit_one_decimal", 65.81, 3, 2, "65.8"),
        ("one_digit_two_decimals", 7.07, 3, 2, "7.07"),
        ("negative_magnitude_under_one", -0.66, 3, 2, "-0.66"),
        ("unlimited_digits", 65.81, None, 2, "65.81"),
        ("nan_value", float("nan"), 3, 2, "--"),
    )
    def test_format_num(
        self,
        val: float,
        max_digits: int | None,
        max_precision: int,
        expected: str,
    ) -> None:
        self.assertEqual(
            format_num(val, max_digits=max_digits, max_precision=max_precision),
            expected,
        )

    @parameterized.named_parameters(
        (
            "best_bracket",
            RewardEntry(mean_reward=86.55, ci_lower=81.09, ci_upper=91.35),
            True,
            "bracket",
            r"$\mathbf{86.5}_{[81.1,\, 91.3]}$",
        ),
        (
            "standard_sub_super",
            RewardEntry(mean_reward=65.81, ci_lower=52.4, ci_upper=77.7),
            False,
            "sub_super",
            r"$65.8_{52.4}^{77.7}$",
        ),
        (
            "missing_entry",
            None,
            False,
            "sub_super",
            "--",
        ),
    )
    def test_format_cell(
        self,
        entry: RewardEntry | None,
        is_best: bool,
        ci_style: str,
        expected: str,
    ) -> None:
        self.assertEqual(
            format_cell(entry, is_best=is_best, ci_style=ci_style),
            expected,
        )

    def test_generate_latex_table_content(self) -> None:
        data_by_env = {
            "EnvA": {
                "Ours": RewardEntry(
                    mean_reward=80.0, ci_lower=75.0, ci_upper=85.0
                ),
                "PPO": RewardEntry(
                    mean_reward=10.0, ci_lower=5.0, ci_upper=15.0
                ),
            },
            "EnvB": {
                "Ours": RewardEntry(
                    mean_reward=90.0, ci_lower=85.0, ci_upper=95.0
                ),
            },
        }
        table_tex = generate_latex_table(
            data_by_env=data_by_env,
            env_names=("EnvA", "EnvB"),
            method_keys=("Ours", "PPO"),
        )
        self.assertIn(
            r"\methodname (Ours) & $\mathbf{80.0}_{75.0}^{85.0}$ &"
            r" $\mathbf{90.0}_{85.0}^{95.0}$ \\",
            table_tex,
        )
        self.assertIn(r"PPO & $10.0_{5.00}^{15.0}$ & -- \\", table_tex)

    def test_generate_latex_table_footnote(self) -> None:
        table_tex = generate_latex_table(
            data_by_env={},
            footnotetext=r"Subscripts denote 95\% bootstrap CI bounds.",
        )
        self.assertIn(
            r"\caption{Final evaluation episodic rewards\protect\footnotemark.}",
            table_tex,
        )
        self.assertIn(
            r"\footnotetext{Subscripts denote 95\% bootstrap CI bounds.}",
            table_tex,
        )


if __name__ == "__main__":
    absltest.main()
