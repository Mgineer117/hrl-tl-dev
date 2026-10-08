"""Converts TensorBoard scalar events to evaluations.npz format for SB3EvalFileReader."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from absl import app, flags
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)

_MODELS_DIR = flags.DEFINE_string(
    "models_dir",
    default=None,
    help="Path to model directory containing rep_* replicate folders.",
)
_NUM_REPLICATES = flags.DEFINE_integer(
    "num_replicates",
    default=None,
    help="Number of replicates to process (rep_0 to rep_{N-1}). Auto-detects all rep_* if omitted.",
)
_TFEVENTS_FILE = flags.DEFINE_string(
    "tfevents_file",
    default=None,
    help="Path to a single tfevents file to convert (alternative to --models_dir).",
)
_OUTPUT_FILE = flags.DEFINE_string(
    "output_file",
    default=None,
    help="Path to save converted .npz file when --tfevents_file is specified.",
)
_OUTPUT_FILENAME = flags.DEFINE_string(
    "output_filename",
    default="evaluations_rollout.npz",
    help="Filename for output .npz inside each replicate's eval directory.",
)
_TAG = flags.DEFINE_string(
    "tag",
    default="rollout/ep_rew_mean",
    help="TensorBoard scalar tag to extract.",
)
_FALLBACK_TAG = flags.DEFINE_string(
    "fallback_tag",
    default=None,
    help="Secondary scalar tag to search for if primary tag is missing.",
)

_REP_DIR_PATTERN = re.compile(r"^rep_(\d+)$")


def find_best_event_file(
    rep_dir: Path,
    tag: str,
    fallback_tag: str | None = None,
) -> tuple[Path | None, str | None]:
    """Finds the event file in a replicate directory reaching the highest step for a tag.

    When an experiment is restarted, earlier aborted runs may produce small event files.
    This selects the event file that contains the target tag and recorded the largest step.

    Args:
        rep_dir: Directory for a single replicate (e.g. rep_0).
        tag: Scalar tag to search for.
        fallback_tag: Optional secondary tag if primary tag is not found.

    Returns:
        Tuple of (best event file, effective tag) or (None, None).
    """
    event_files = sorted(rep_dir.glob("tb/**/events.out.tfevents*"))
    tags_to_check = [tag]
    if fallback_tag is not None and fallback_tag != tag:
        tags_to_check.append(fallback_tag)

    for check_tag in tags_to_check:
        best_file: Path | None = None
        best_max_step: int = -1
        for event_file in event_files:
            try:
                accumulator = EventAccumulator(str(event_file))
                accumulator.Reload()
                scalar_tags = accumulator.Tags().get("scalars", [])
                if check_tag not in scalar_tags:
                    continue

                events = accumulator.Scalars(check_tag)
                if events and events[-1].step > best_max_step:
                    best_max_step = events[-1].step
                    best_file = event_file
            except (OSError, ValueError) as err:
                print(f"Warning: Failed reading {event_file}: {err}")
                continue

        if best_file is not None:
            return best_file, check_tag

    return None, None


def convert_event_file(
    event_file: Path,
    output_file: Path,
    tag: str,
) -> int:
    """Extracts a scalar tag from an event file and saves it in evaluations.npz format.

    Args:
        event_file: Source TensorBoard event file.
        output_file: Destination .npz file path.
        tag: Scalar tag name to extract.

    Returns:
        Number of scalar points written.

    Raises:
        ValueError: If tag is not found or timesteps are not strictly monotonic.
    """
    print(f"Loading {event_file}...")
    accumulator = EventAccumulator(str(event_file))
    accumulator.Reload()

    available_scalars = accumulator.Tags().get("scalars", [])
    if tag not in available_scalars:
        raise ValueError(
            f"Tag '{tag}' not found in {event_file}. Available: {available_scalars}"
        )

    events = accumulator.Scalars(tag)
    timesteps = np.array([e.step for e in events], dtype=np.int64)
    results = np.array([e.value for e in events], dtype=np.float32)

    # Validate strictly monotonic timesteps
    if len(timesteps) > 1 and not np.all(np.diff(timesteps) > 0):
        raise ValueError(
            f"Timesteps in {event_file} are not strictly monotonically increasing."
        )

    # SB3EvalFileReader expects results shape: (n_timesteps, n_eval_episodes)
    results_2d = results.reshape(-1, 1)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    if "succ" in tag.lower():
        np.savez(
            output_file,
            timesteps=timesteps,
            results=results_2d,
            successes=results_2d,
        )
    else:
        np.savez(output_file, timesteps=timesteps, results=results_2d)

    print(
        f"✓ Saved {len(timesteps)} points (step {timesteps[0]} to {timesteps[-1]}) "
        f"to {output_file}"
    )
    return len(timesteps)


def discover_replicate_dirs(models_dir: Path) -> list[Path]:
    """Discovers all rep_* subdirectories under models_dir sorted by replicate index.

    Args:
        models_dir: Base directory containing replicate folders.

    Returns:
        Sorted list of replicate directory paths.
    """
    rep_dirs: list[tuple[int, Path]] = []
    for item in models_dir.iterdir():
        if not item.is_dir():
            continue
        match = _REP_DIR_PATTERN.match(item.name)
        if match:
            rep_dirs.append((int(match.group(1)), item))

    rep_dirs.sort(key=lambda pair: pair[0])
    return [path for _, path in rep_dirs]


def process_models_directory(
    models_dir: Path,
    tag: str,
    output_filename: str,
    fallback_tag: str | None = None,
    num_replicates: int | None = None,
) -> None:
    """Processes all replicates in a models directory.

    Args:
        models_dir: Directory containing replicate folders.
        tag: Scalar tag to extract.
        output_filename: Output file basename in each replicate's eval directory.
        fallback_tag: Optional secondary scalar tag to search for if tag missing.
        num_replicates: Optional maximum number of replicates to process.
    """
    if num_replicates is not None:
        target_dirs = [models_dir / f"rep_{i}" for i in range(num_replicates)]
    else:
        target_dirs = discover_replicate_dirs(models_dir)

    print(f"Processing {len(target_dirs)} replicates in {models_dir}...")
    successful_count = 0

    for rep_dir in target_dirs:
        if not rep_dir.exists():
            print(f"Warning: Replicate directory not found: {rep_dir}")
            continue

        best_event_file, effective_tag = find_best_event_file(
            rep_dir, tag, fallback_tag
        )
        if best_event_file is None or effective_tag is None:
            print(
                f"Warning: No valid event file found with tag '{tag}' in {rep_dir}"
            )
            continue

        output_file = rep_dir / "eval" / output_filename
        try:
            convert_event_file(best_event_file, output_file, effective_tag)
            successful_count += 1
        except ValueError as err:
            print(f"Error processing {rep_dir.name}: {err}")

    print(
        f"Successfully converted {successful_count}/{len(target_dirs)} replicates."
    )


def main(argv: Sequence[str]) -> None:
    """Main execution function for TensorBoard to NPZ conversion."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    tag = _TAG.value
    fallback_tag = _FALLBACK_TAG.value

    # Mode 1: Single file conversion
    if _TFEVENTS_FILE.value is not None:
        if _OUTPUT_FILE.value is None:
            raise app.UsageError(
                "--output_file must be specified when --tfevents_file is used."
            )
        convert_event_file(
            Path(_TFEVENTS_FILE.value),
            Path(_OUTPUT_FILE.value),
            tag,
        )
        return

    # Mode 2: Multi-replicate directory conversion
    if _MODELS_DIR.value is not None:
        process_models_directory(
            models_dir=Path(_MODELS_DIR.value),
            tag=tag,
            output_filename=_OUTPUT_FILENAME.value,
            fallback_tag=fallback_tag,
            num_replicates=_NUM_REPLICATES.value,
        )
        return

    raise app.UsageError(
        "Either --models_dir or --tfevents_file must be specified."
    )


if __name__ == "__main__":
    app.run(main)
