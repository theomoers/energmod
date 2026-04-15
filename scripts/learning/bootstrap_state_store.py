#!/usr/bin/env python3
"""Save and restore reusable bootstrap/prerequisite state for stochastic learning runs."""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_SECTOR_ROOT = "Global_200"
DEFAULT_REQUIRED_PATTERNS = (
    "learning/bootstrap_complete_elec_s*.txt",
    "prenetworks/elec_s*export.nc",
)


@dataclass
class MirrorSummary:
    copied: int = 0
    linked: int = 0
    skipped_existing: int = 0
    symlinked: int = 0


def sanitize_token(value: str, fallback: str = "default") -> str:
    raw = str(value or "").strip()
    if not raw:
        return fallback
    token = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw)
    token = token.strip("._-")
    return token or fallback


def resolve_scenario_sector_name(scenario_name: str, sector_root: str = DEFAULT_SECTOR_ROOT) -> str:
    return f"{sector_root.rstrip('/')}/{sanitize_token(scenario_name)}"


def resolve_results_sector_dir(root_dir: Path, sector_name: str) -> Path:
    return (root_dir / "results" / Path(sector_name)).resolve()


def _require_directory(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {resolved}")
    return resolved


def validate_state_source(
    source_dir: Path,
    required_patterns: tuple[str, ...] = DEFAULT_REQUIRED_PATTERNS,
) -> dict[str, int]:
    source_dir = _require_directory(source_dir, "Bootstrap state source")
    counts: dict[str, int] = {}
    missing: list[str] = []
    for pattern in required_patterns:
        matches = [p for p in source_dir.glob(pattern) if p.is_file()]
        if not matches:
            missing.append(pattern)
            counts[pattern] = 0
            continue
        counts[pattern] = len(matches)
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(
            f"Bootstrap state source is missing required artifacts ({missing_list}) under {source_dir}"
        )
    return counts


def _try_hardlink(src: Path, dst: Path) -> bool:
    try:
        os.link(src, dst)
        return True
    except OSError as exc:
        if exc.errno in (errno.EXDEV, errno.EPERM, errno.ENOTSUP, errno.EACCES, errno.EEXIST):
            return False
        raise


def _mirror_file(src: Path, dst: Path, hardlink_first: bool, summary: MirrorSummary) -> None:
    if dst.exists():
        summary.skipped_existing += 1
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_symlink():
        os.symlink(os.readlink(src), dst)
        summary.symlinked += 1
        return
    if hardlink_first and _try_hardlink(src, dst):
        summary.linked += 1
        return
    shutil.copy2(src, dst)
    summary.copied += 1


def mirror_tree(
    source_dir: Path,
    target_dir: Path,
    *,
    hardlink_first: bool,
) -> MirrorSummary:
    source_dir = _require_directory(source_dir, "Mirror source")
    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    summary = MirrorSummary()
    for src in sorted(source_dir.rglob("*")):
        if src.is_dir():
            continue
        rel = src.relative_to(source_dir)
        dst = target_dir / rel
        _mirror_file(src, dst, hardlink_first=hardlink_first, summary=summary)
    return summary


def write_state_manifest(
    target_dir: Path,
    source_dir: Path,
    required_counts: dict[str, int],
) -> Path:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir.resolve()),
        "required_artifact_counts": required_counts,
    }
    manifest_path = target_dir / "bootstrap_state_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def save_bootstrap_state(
    source_sector_dir: Path,
    target_dir: Path,
) -> dict[str, object]:
    source_sector_dir = _require_directory(source_sector_dir, "Bootstrap source sector directory")
    required_counts = validate_state_source(source_sector_dir)
    summary = mirror_tree(source_sector_dir, target_dir, hardlink_first=False)
    manifest_path = write_state_manifest(target_dir.resolve(), source_sector_dir, required_counts)
    return {
        "source_dir": str(source_sector_dir.resolve()),
        "target_dir": str(target_dir.resolve()),
        "required_artifact_counts": required_counts,
        "copied": summary.copied,
        "linked": summary.linked,
        "symlinked": summary.symlinked,
        "skipped_existing": summary.skipped_existing,
        "manifest_path": str(manifest_path),
    }


def restore_bootstrap_state(
    source_dir: Path,
    target_sector_dir: Path,
) -> dict[str, object]:
    source_dir = _require_directory(source_dir, "Bootstrap state source")
    required_counts = validate_state_source(source_dir)
    summary = mirror_tree(source_dir, target_sector_dir, hardlink_first=True)
    return {
        "source_dir": str(source_dir.resolve()),
        "target_dir": str(target_sector_dir.resolve()),
        "required_artifact_counts": required_counts,
        "copied": summary.copied,
        "linked": summary.linked,
        "symlinked": summary.symlinked,
        "skipped_existing": summary.skipped_existing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    save_parser = subparsers.add_parser("save", help="Save bootstrap/prereq state from a results sector")
    save_parser.add_argument("--source-sector-dir", required=True)
    save_parser.add_argument("--target-dir", required=True)

    restore_parser = subparsers.add_parser("restore", help="Restore bootstrap/prereq state into a results sector")
    restore_parser.add_argument("--source-dir", required=True)
    restore_parser.add_argument("--target-sector-dir", required=True)

    args = parser.parse_args()
    if args.command == "save":
        result = save_bootstrap_state(
            Path(os.path.expandvars(args.source_sector_dir)),
            Path(os.path.expandvars(args.target_dir)),
        )
    else:
        result = restore_bootstrap_state(
            Path(os.path.expandvars(args.source_dir)),
            Path(os.path.expandvars(args.target_sector_dir)),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
