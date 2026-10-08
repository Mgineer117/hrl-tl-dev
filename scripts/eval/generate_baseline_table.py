"""CLI entry point to generate a LaTeX table of baseline evaluation rewards."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from absl import app, flags

from hrl_tl.eval.table import (
    generate_latex_table,
    load_csv_data,
)

_FR_CSV = flags.DEFINE_string(
    "fr_csv",
    default="out/plots/fr_cont/baseline/fr_cont_reward_conv_baselines.csv",
    help="Path to continuous FourRooms baseline summary CSV.",
)
_ZONE_CSV = flags.DEFINE_string(
    "zone_csv",
    default="out/plots/zone/baseline/zone_button_reward_conv_baselines_st2.csv",
    help="Path to Zone baseline summary CSV.",
)
_FETCH_CSV = flags.DEFINE_string(
    "fetch_csv",
    default="out/plots/fetch/baseline/fetch_reward_conv_baselines.csv",
    help="Path to Fetch Reach-Avoid baseline summary CSV.",
)
_OUTPUT_TEX = flags.DEFINE_string(
    "output_tex",
    default="out/tables/baseline_rewards_table.tex",
    help="Path to save the generated LaTeX table snippet.",
)
_CAPTION = flags.DEFINE_string(
    "caption",
    default=r"Final evaluation episodic rewards\protect\footnotemark.",
    help="Caption text for the generated LaTeX table.",
)
_LABEL = flags.DEFINE_string(
    "label",
    default="tab:baseline_rewards",
    help="Label identifier for LaTeX table cross-referencing.",
)
_PRECISION = flags.DEFINE_integer(
    "precision",
    default=2,
    help="Number of decimal digits for mean and CI bounds.",
    lower_bound=0,
)
_MAX_DIGITS = flags.DEFINE_integer(
    "max_digits",
    default=3,
    help="Maximum total digits to display per number (or -1 for unlimited).",
)
_CI_STYLE = flags.DEFINE_enum(
    "ci_style",
    default="bracket",
    enum_values=["sub_super", "bracket"],
    help="Style for CI bounds: 'sub_super' (stacked) or 'bracket'.",
)
_BOLD_BEST = flags.DEFINE_boolean(
    "bold_best",
    default=True,
    help="Whether to bold only the mean reward for the best method in each column.",
)
_FOOTNOTETEXT = flags.DEFINE_string(
    "footnotetext",
    default=r"Subscripts denote 95\% bootstrap CI bounds.",
    help="Optional footnote text to append after \\end{table}.",
)


def main(argv: Sequence[str]) -> None:
    """Loads CSVs, generates the LaTeX table, prints, and saves output.

    Args:
        argv: Command-line arguments.

    Raises:
        app.UsageError: If unexpected positional arguments are provided.
    """
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    fr_path = Path(_FR_CSV.value)
    zone_path = Path(_ZONE_CSV.value)
    fetch_path = Path(_FETCH_CSV.value)

    data_by_env = {
        "FourRooms": load_csv_data(fr_path),
        "Zone": load_csv_data(zone_path),
        "Fetch Reach-Avoid": load_csv_data(fetch_path),
    }

    max_digits = None if _MAX_DIGITS.value < 0 else _MAX_DIGITS.value
    footnotetext = _FOOTNOTETEXT.value if _FOOTNOTETEXT.value else None

    table_tex = generate_latex_table(
        data_by_env=data_by_env,
        caption=_CAPTION.value,
        label=_LABEL.value,
        precision=_PRECISION.value,
        max_digits=max_digits,
        ci_style=_CI_STYLE.value,
        bold_best=_BOLD_BEST.value,
        footnotetext=footnotetext,
    )

    print("\nGenerated LaTeX Table:")
    print("=" * 70)
    print(table_tex)
    print("=" * 70)

    if _OUTPUT_TEX.value:
        out_path = Path(_OUTPUT_TEX.value)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with out_path.open("w", encoding="utf-8") as f:
                f.write(table_tex + "\n")
            print(f"\nTable snippet saved to: {out_path}")
        except OSError as e:
            raise app.UsageError(
                f"Failed to write LaTeX table file: {e}"
            ) from e


if __name__ == "__main__":
    app.run(main)
