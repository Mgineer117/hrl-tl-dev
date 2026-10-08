"""CLI utility to consolidate subpolicy intermediate checkpoints into final models."""

from __future__ import annotations

import shutil
import zipfile
from collections.abc import Sequence
from enum import Enum
from pathlib import Path

from absl import app, flags, logging
from pydantic import BaseModel

_SUBPOLICIES_DIR = flags.DEFINE_string(
    "subpolicies_dir",
    default="out/fetch/ablation/subpolicies/12.a.b",
    help="Base directory containing subpolicy runs or replicate directories.",
)
_TIMESTEPS = flags.DEFINE_integer(
    "timesteps",
    default=4_000_000,
    lower_bound=1,
    help="Timestep count of intermediate checkpoint to consolidate.",
)
_TARGET_NAME_PATTERN = flags.DEFINE_string(
    "target_name_pattern",
    default="final_model_{spec}.zip",
    help="Pattern for the final model filename, formatted with {spec}.",
)
_SUBPOLICY_PREFIX = flags.DEFINE_string(
    "subpolicy_prefix",
    default="subpolicy_fetch_",
    help="Directory name prefix identifying subpolicy directories.",
)
_DRY_RUN = flags.DEFINE_boolean(
    "dry_run",
    default=False,
    help="Whether to perform a dry run without modifying files.",
)
_OVERWRITE = flags.DEFINE_boolean(
    "overwrite",
    default=False,
    help="Whether to overwrite existing final models (default preserves existing).",
)


class ActionStatus(str, Enum):
    """Status outcomes for processing a single subpolicy directory."""

    COPIED = "COPIED"
    WOULD_COPY = "WOULD_COPY"
    SKIPPED_EXISTING = "SKIPPED_EXISTING"
    SKIPPED_NO_CHECKPOINT = "SKIPPED_NO_CHECKPOINT"
    FAILED = "FAILED"


class SubpolicyRecord(BaseModel):
    """Consolidation report record for a subpolicy directory."""

    subpolicy_dir: Path
    spec: str
    status: ActionStatus
    source_ckpt: Path | None = None
    target_path: Path | None = None
    message: str = ""


def extract_spec(dir_name: str, prefix: str) -> str | None:
    """Extracts specification suffix from a subpolicy directory name.

    Args:
        dir_name: Name of candidate subpolicy directory.
        prefix: Expected prefix indicating a subpolicy directory.

    Returns:
        Specification string if prefix matches, or None.
    """
    if dir_name.startswith(prefix):
        return dir_name[len(prefix) :]
    return None


def find_subpolicy_directories(
    base_dir: Path, prefix: str
) -> list[tuple[Path, str]]:
    """Discovers subpolicy directories under the base path.

    Recursively scans base_dir for directories matching the subpolicy prefix.

    Args:
        base_dir: Root directory containing runs or replicate folders.
        prefix: Prefix identifying subpolicy directories.

    Returns:
        Sorted list of tuples containing directory Path and extracted spec name.
    """
    found: list[tuple[Path, str]] = []
    if not base_dir.is_dir():
        return found

    for path in base_dir.rglob("*"):
        if not path.is_dir():
            continue
        spec = extract_spec(path.name, prefix)
        if spec is not None:
            found.append((path, spec))

    return sorted(found, key=lambda item: item[0])


def verify_zip_integrity(zip_path: Path) -> bool:
    """Verifies that a file is an intact ZIP archive.

    Args:
        zip_path: Path to the ZIP file.

    Returns:
        True if the ZIP file is valid and readable without corruption.
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            return zf.testzip() is None
    except (zipfile.BadZipFile, OSError):
        logging.exception("Integrity check failed for ZIP: %s", zip_path)
        return False


def process_subpolicy(
    subpolicy_dir: Path,
    spec: str,
    timesteps: int,
    target_name_pattern: str,
    dry_run: bool,
    overwrite: bool,
) -> SubpolicyRecord:
    """Processes an individual subpolicy directory for checkpoint consolidation.

    Args:
        subpolicy_dir: Path to the specific subpolicy directory.
        spec: Specification identifier extracted from folder name.
        timesteps: Target checkpoint timestep count.
        target_name_pattern: Pattern for the target final model filename.
        dry_run: Whether to simulate actions without disk changes.
        overwrite: Whether to overwrite existing final models.

    Returns:
        A SubpolicyRecord detailing the action taken.
    """
    ckpt_filename = f"ckpt_{timesteps}_steps.zip"
    source_ckpt = subpolicy_dir / "ckpts" / ckpt_filename

    if not source_ckpt.is_file():
        return SubpolicyRecord(
            subpolicy_dir=subpolicy_dir,
            spec=spec,
            status=ActionStatus.SKIPPED_NO_CHECKPOINT,
            message=f"Missing checkpoint: {ckpt_filename}",
        )

    target_filename = target_name_pattern.format(spec=spec)
    target_path = subpolicy_dir / target_filename

    if target_path.is_file() and not overwrite:
        return SubpolicyRecord(
            subpolicy_dir=subpolicy_dir,
            spec=spec,
            status=ActionStatus.SKIPPED_EXISTING,
            source_ckpt=source_ckpt,
            target_path=target_path,
            message=f"Existing final model preserved: {target_filename}",
        )

    if dry_run:
        return SubpolicyRecord(
            subpolicy_dir=subpolicy_dir,
            spec=spec,
            status=ActionStatus.WOULD_COPY,
            source_ckpt=source_ckpt,
            target_path=target_path,
            message=f"Would copy {source_ckpt.name} -> {target_filename}",
        )

    try:
        shutil.copy2(source_ckpt, target_path)
        if not verify_zip_integrity(target_path):
            return SubpolicyRecord(
                subpolicy_dir=subpolicy_dir,
                spec=spec,
                status=ActionStatus.FAILED,
                source_ckpt=source_ckpt,
                target_path=target_path,
                message=f"Corrupt archive after copy: {target_filename}",
            )
        return SubpolicyRecord(
            subpolicy_dir=subpolicy_dir,
            spec=spec,
            status=ActionStatus.COPIED,
            source_ckpt=source_ckpt,
            target_path=target_path,
            message=f"Copied {source_ckpt.name} -> {target_filename}",
        )
    except OSError as e:
        logging.exception("Failed to copy %s to %s", source_ckpt, target_path)
        return SubpolicyRecord(
            subpolicy_dir=subpolicy_dir,
            spec=spec,
            status=ActionStatus.FAILED,
            source_ckpt=source_ckpt,
            target_path=target_path,
            message=str(e),
        )


def log_summary(records: Sequence[SubpolicyRecord], dry_run: bool) -> None:
    """Logs a formatted summary of consolidation results.

    Args:
        records: Sequence of SubpolicyRecord entries.
        dry_run: Whether dry-run mode was active.
    """
    mode_str = "[DRY-RUN] " if dry_run else ""
    counts: dict[ActionStatus, int] = {}
    for record in records:
        counts[record.status] = counts.get(record.status, 0) + 1

    logging.info("========== %sConsolidation Summary ==========", mode_str)
    logging.info("Total subpolicies evaluated: %d", len(records))
    for status, count in sorted(counts.items(), key=lambda x: x[0].value):
        logging.info("  %s: %d", status.value, count)

    copied_actions = {ActionStatus.COPIED, ActionStatus.WOULD_COPY}
    active_records = [r for r in records if r.status in copied_actions]
    if active_records:
        logging.info("\nActioned Items:")
        for r in active_records:
            logging.info("  %s: %s", r.subpolicy_dir, r.message)

    skipped_existing = [
        r for r in records if r.status == ActionStatus.SKIPPED_EXISTING
    ]
    if skipped_existing:
        logging.info("\nPreserved Existing Models:")
        for r in skipped_existing:
            logging.info("  %s: %s", r.subpolicy_dir, r.message)


def main(argv: Sequence[str]) -> None:
    """Main execution function for checkpoint consolidation CLI.

    Args:
        argv: Command-line arguments.

    Raises:
        app.UsageError: If unexpected positional arguments are provided.
    """
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    base_dir = Path(_SUBPOLICIES_DIR.value)
    if not base_dir.exists():
        logging.error("Subpolicies directory does not exist: %s", base_dir)
        return

    timesteps: int = _TIMESTEPS.value
    target_pattern: str = _TARGET_NAME_PATTERN.value
    prefix: str = _SUBPOLICY_PREFIX.value
    dry_run: bool = _DRY_RUN.value
    overwrite: bool = _OVERWRITE.value

    logging.info(
        "Scanning %s for subpolicies (timesteps=%d, dry_run=%s, overwrite=%s)",
        base_dir,
        timesteps,
        dry_run,
        overwrite,
    )

    targets = find_subpolicy_directories(base_dir, prefix)
    if not targets:
        logging.warning("No subpolicy directories found under %s", base_dir)
        return

    records: list[SubpolicyRecord] = [
        process_subpolicy(
            subpolicy_dir=sub_dir,
            spec=spec,
            timesteps=timesteps,
            target_name_pattern=target_pattern,
            dry_run=dry_run,
            overwrite=overwrite,
        )
        for sub_dir, spec in targets
    ]

    log_summary(records, dry_run=dry_run)


if __name__ == "__main__":
    app.run(main)
