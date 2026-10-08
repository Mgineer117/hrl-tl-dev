#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

PREFIX = "ckpt_"
SUFFIX = "steps.zip"


def find_ckpt_files(out_dir: Path) -> list[Path]:
    """Collect target checkpoint files from eligible 'ckpts' directories under out_dir."""
    matches: list[Path] = []

    for ckpts_dir in out_dir.rglob("ckpts"):
        if not ckpts_dir.is_dir():
            continue

        # Only process this ckpts directory when best_model.zip exists alongside it.
        if not (ckpts_dir.parent / "best_model.zip").is_file():
            continue

        for candidate in ckpts_dir.iterdir():
            if not candidate.is_file():
                continue
            name = candidate.name
            if name.startswith(PREFIX) and name.endswith(SUFFIX):
                matches.append(candidate)

    return sorted(matches)


def delete_files(files: list[Path], dry_run: bool) -> int:
    deleted = 0

    for file_path in files:
        if dry_run:
            print(f"[DRY-RUN] Would remove: {file_path}")
            continue

        file_path.unlink(missing_ok=True)
        print(f"Removed: {file_path}")
        deleted += 1

    return deleted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove checkpoint files named like 'ckpt_*steps.zip' only from "
            "directories named 'ckpts' under an out directory, and only when "
            "a sibling best_model.zip exists."
        )
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out"),
        help="Path to the output root directory (default: out)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be removed without deleting them",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir

    if not out_dir.exists() or not out_dir.is_dir():
        raise SystemExit(f"Invalid out directory: {out_dir}")

    files = find_ckpt_files(out_dir)
    if not files:
        print("No matching checkpoint files found.")
        return

    print(f"Found {len(files)} matching files.")
    deleted = delete_files(files, dry_run=args.dry_run)

    if args.dry_run:
        print("Dry run completed. No files were deleted.")
    else:
        print(f"Deleted {deleted} files.")


if __name__ == "__main__":
    main()
