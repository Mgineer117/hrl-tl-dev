"""LaTeX table generation utilities for evaluation reward summaries."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

_DEFAULT_METHOD_KEYS: tuple[str, ...] = (
    "Ours",
    "GCLTL-RL",
    "ALLO",
    "HIRO",
    "PPO",
)
_DEFAULT_METHOD_DISPLAY_NAMES: dict[str, str] = {
    "Ours": r"\methodname (Ours)",
    "GCLTL-RL": "GCLTL-RL",
    "ALLO": "ALLO",
    "HIRO": "HIRO",
    "PPO": "PPO",
}
_DEFAULT_ENV_NAMES: tuple[str, ...] = (
    "FourRooms",
    "Zone",
    "Fetch Reach-Avoid",
)


class RewardEntry(BaseModel):
    """Evaluation summary statistics for a single method."""

    mean_reward: float
    ci_lower: float
    ci_upper: float


def load_csv_data(csv_path: Path) -> dict[str, RewardEntry]:
    """Reads a baseline summary CSV and maps method names to reward entries.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        Mapping of method name to RewardEntry. Returns empty dict if file is missing.
    """
    try:
        df = pd.read_csv(csv_path)
    except (FileNotFoundError, OSError):
        return {}

    entry_by_method: dict[str, RewardEntry] = {}
    for _, row in df.iterrows():
        method_name = str(row["method"]).strip()
        mean_val = (
            float(row["mean_success_rate"])
            if "mean_success_rate" in row
            else float(row["mean_reward"])
        )
        entry_by_method[method_name] = RewardEntry(
            mean_reward=mean_val,
            ci_lower=float(row["ci_lower"]),
            ci_upper=float(row["ci_upper"]),
        )
    return entry_by_method


def format_num(
    val: float,
    max_digits: int | None = 3,
    max_precision: int = 2,
) -> str:
    """Formats a float with at most `max_digits` digits and up to `max_precision` decimals.

    Args:
        val: Numerical value to format.
        max_digits: Maximum total digits allowed (e.g. 3 for 65.8 or 7.07). If None
            or non-positive, max_precision is used directly without digit capping.
        max_precision: Maximum decimal precision.

    Returns:
        Formatted string representation.
    """
    if math.isnan(val) or pd.isna(val):
        return "--"

    if max_digits is None or max_digits <= 0:
        res = f"{val:.{max_precision}f}"
        return res[1:] if res.startswith("-") and float(res) == 0.0 else res

    for dec in range(max_precision, -1, -1):
        formatted = f"{val:.{dec}f}"
        if formatted.startswith("-") and float(formatted) == 0.0:
            formatted = formatted[1:]
        digit_count = sum(c.isdigit() for c in formatted)
        if digit_count <= max_digits:
            return formatted

    res = f"{val:.0f}"
    return res[1:] if res.startswith("-") and float(res) == 0.0 else res


def format_cell(
    entry: RewardEntry | None,
    is_best: bool,
    precision: int = 2,
    max_digits: int | None = 3,
    ci_style: str = "sub_super",
) -> str:
    """Formats a single table cell with mean reward and CI bounds.

    Args:
        entry: RewardEntry instance or None if missing.
        is_best: True if this entry represents the best mean in the column.
        precision: Maximum number of decimal places.
        max_digits: Maximum total digits allowed per number.
        ci_style: 'sub_super' for stacked bounds or 'bracket' for horizontal interval.

    Returns:
        Formatted LaTeX cell string.
    """
    if entry is None:
        return "--"

    mean_str = format_num(
        entry.mean_reward, max_digits=max_digits, max_precision=precision
    )
    lower_str = format_num(
        entry.ci_lower, max_digits=max_digits, max_precision=precision
    )
    upper_str = format_num(
        entry.ci_upper, max_digits=max_digits, max_precision=precision
    )

    mean_formatted = f"\\mathbf{{{mean_str}}}" if is_best else mean_str

    if ci_style == "sub_super":
        return f"${mean_formatted}_{{{lower_str}}}^{{{upper_str}}}$"
    return f"${mean_formatted}_{{[{lower_str},\\, {upper_str}]}}$"


def generate_latex_table(
    data_by_env: Mapping[str, Mapping[str, RewardEntry]],
    env_names: Sequence[str] = _DEFAULT_ENV_NAMES,
    method_keys: Sequence[str] = _DEFAULT_METHOD_KEYS,
    method_display_names: Mapping[str, str] = _DEFAULT_METHOD_DISPLAY_NAMES,
    caption: str = r"Final evaluation episodic rewards\protect\footnotemark.",
    label: str = "tab:baseline_rewards",
    precision: int = 2,
    max_digits: int | None = 3,
    ci_style: str = "sub_super",
    bold_best: bool = True,
    footnotetext: str | None = None,
) -> str:
    """Generates a complete LaTeX table code using booktabs.

    Args:
        data_by_env: Mapping from environment name to method data mapping.
        env_names: Sequence of environment column headers.
        method_keys: Sequence of method identifiers.
        method_display_names: Mapping from method keys to LaTeX display names.
        caption: Table caption string.
        label: Table LaTeX label.
        precision: Decimal precision for numerical values.
        max_digits: Maximum total digits allowed per number.
        ci_style: 'sub_super' or 'bracket'.
        bold_best: Whether to format best mean in bold font.
        footnotetext: Optional footnote text to place immediately after table.

    Returns:
        LaTeX table string.
    """
    best_mean_by_env: dict[str, float] = {}
    for env in env_names:
        method_data = data_by_env.get(env, {})
        valid_means = [
            method_data[m].mean_reward for m in method_keys if m in method_data
        ]
        best_mean_by_env[env] = (
            max(valid_means) if valid_means else float("-inf")
        )

    lines: list[str] = [
        r"\begin{table}[tb]",
        r"    \centering",
        r"    \small",
        r"    \setlength{\tabcolsep}{4pt}",
        rf"    \caption{{{caption}}}",
        rf"    \label{{{label}}}",
        r"    \begin{tabular}{l " + "c" * len(env_names) + r"}",
        r"        \toprule",
        r"        Method & " + " & ".join(env_names) + r" \\",
        r"        \midrule",
    ]

    for method in method_keys:
        display_name = method_display_names.get(method, method)
        cells: list[str] = [display_name]
        for env in env_names:
            entry = data_by_env.get(env, {}).get(method)
            is_best = (
                bold_best
                and entry is not None
                and entry.mean_reward == best_mean_by_env[env]
            )
            cells.append(
                format_cell(
                    entry,
                    is_best=is_best,
                    precision=precision,
                    max_digits=max_digits,
                    ci_style=ci_style,
                )
            )
        lines.append(r"        " + " & ".join(cells) + r" \\")

    lines.extend([r"        \bottomrule", r"    \end{tabular}", r"\end{table}"])
    if footnotetext:
        lines.append(rf"\footnotetext{{{footnotetext}}}")
    return "\n".join(lines)
