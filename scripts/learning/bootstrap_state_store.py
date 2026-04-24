#!/usr/bin/env python3
"""Save and restore reusable bootstrap/prerequisite state for stochastic learning runs."""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
from dataclasses import dataclass
from typing import Callable
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_SECTOR_ROOT = "Global_200"
DEFAULT_REQUIRED_PATTERNS = (
    "learning/bootstrap_complete_elec_s*.txt",
    "prenetworks/elec_s*export.nc",
)
DEFAULT_RESULTS_SHARED_TOPLEVEL_DIRS = {
    "benchmarks",
    "configs",
    "learning",
    "logs",
    "postnetworks",
    "prenetworks",
    "prenetworks-brownfield",
    "prenetworks-learning",
    "tsam_clustering",
}
DEFAULT_RESULTS_SCENARIO_MARKERS = {
    "benchmarks",
    "configs",
    "learning",
    "postnetworks",
    "prenetworks",
    "prenetworks-brownfield",
    "prenetworks-learning",
}
DEFAULT_RESOURCES_SHARED_TOPLEVEL_DIRS = {
    "cops",
    "demand",
    "gas_networks",
    "gdp_shares",
    "heating",
    "pattern_profiles",
    "population_shares",
    "temperatures",
}
DEFAULT_RESOURCES_SCENARIO_MARKERS = {
    "cops",
    "demand",
    "gas_networks",
    "gdp_shares",
    "heating",
    "pattern_profiles",
    "population_shares",
    "temperatures",
}


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


def resolve_resources_sector_dir(root_dir: Path, sector_name: str) -> Path:
    return (root_dir / "resources" / Path(sector_name)).resolve()


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
    if os.path.lexists(dst):
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
    exclude_path: Callable[[Path], bool] | None = None,
) -> MirrorSummary:
    source_dir = _require_directory(source_dir, "Mirror source")
    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    summary = MirrorSummary()
    for src in sorted(source_dir.rglob("*")):
        if exclude_path is not None and exclude_path(src):
            continue
        if src.is_dir():
            continue
        rel = src.relative_to(source_dir)
        dst = target_dir / rel
        _mirror_file(src, dst, hardlink_first=hardlink_first, summary=summary)
    return summary


def _build_nested_scenario_excluder(
    source_root: Path,
    *,
    shared_dir_names: set[str],
    scenario_markers: set[str],
):
    source_root = source_root.resolve()

    def _exclude(path: Path) -> bool:
        try:
            rel = path.resolve().relative_to(source_root)
        except Exception:
            return False
        if len(rel.parts) == 0:
            return False
        top = rel.parts[0]
        if top in shared_dir_names:
            return False
        top_path = source_root / top
        if not top_path.is_dir():
            return False
        if not any((top_path / marker).exists() for marker in scenario_markers):
            return False
        return rel.parts[0] == top

    return _exclude


def _merge_summary(parts: list[MirrorSummary]) -> MirrorSummary:
    merged = MirrorSummary()
    for part in parts:
        merged.copied += part.copied
        merged.linked += part.linked
        merged.skipped_existing += part.skipped_existing
        merged.symlinked += part.symlinked
    return merged


def _infer_sector_name_from_results_dir(sector_dir: Path) -> str:
    sector_dir = sector_dir.resolve()
    try:
        rel = sector_dir.relative_to((ROOT_DIR / "results").resolve())
    except ValueError as exc:
        raise ValueError(
            f"Expected a results sector directory under {ROOT_DIR / 'results'}; got {sector_dir}"
        ) from exc
    return str(rel)


def _companion_resources_dir_for_results_dir(sector_dir: Path) -> Path:
    sector_name = _infer_sector_name_from_results_dir(sector_dir)
    return resolve_resources_sector_dir(ROOT_DIR, sector_name)


def write_state_manifest(
    target_dir: Path,
    source_dir: Path,
    required_counts: dict[str, int],
    resources_source_dir: Path | None = None,
) -> Path:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir.resolve()),
        "required_artifact_counts": required_counts,
    }
    if resources_source_dir is not None:
        payload["resources_source_dir"] = str(resources_source_dir.resolve())
    manifest_path = target_dir / "bootstrap_state_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def save_bootstrap_state(
    source_sector_dir: Path,
    target_dir: Path,
) -> dict[str, object]:
    source_sector_dir = _require_directory(source_sector_dir, "Bootstrap source sector directory")
    required_counts = validate_state_source(source_sector_dir)
    results_excluder = _build_nested_scenario_excluder(
        source_sector_dir,
        shared_dir_names=DEFAULT_RESULTS_SHARED_TOPLEVEL_DIRS,
        scenario_markers=DEFAULT_RESULTS_SCENARIO_MARKERS,
    )
    summaries = [
        mirror_tree(
            source_sector_dir,
            target_dir,
            hardlink_first=False,
            exclude_path=results_excluder,
        )
    ]
    resources_source_dir = _companion_resources_dir_for_results_dir(source_sector_dir)
    resources_target_dir = target_dir.resolve() / "__resources_sector__"
    resources_source_exists = resources_source_dir.exists() and resources_source_dir.is_dir()
    if resources_source_exists:
        resources_excluder = _build_nested_scenario_excluder(
            resources_source_dir,
            shared_dir_names=DEFAULT_RESOURCES_SHARED_TOPLEVEL_DIRS,
            scenario_markers=DEFAULT_RESOURCES_SCENARIO_MARKERS,
        )
        summaries.append(
            mirror_tree(
                resources_source_dir,
                resources_target_dir,
                hardlink_first=False,
                exclude_path=resources_excluder,
            )
        )
    summary = _merge_summary(summaries)
    manifest_path = write_state_manifest(
        target_dir.resolve(),
        source_sector_dir,
        required_counts,
        resources_source_dir=resources_source_dir if resources_source_exists else None,
    )
    return {
        "source_dir": str(source_sector_dir.resolve()),
        "target_dir": str(target_dir.resolve()),
        "required_artifact_counts": required_counts,
        "copied": summary.copied,
        "linked": summary.linked,
        "symlinked": summary.symlinked,
        "skipped_existing": summary.skipped_existing,
        "manifest_path": str(manifest_path),
        "resources_source_dir": str(resources_source_dir.resolve()) if resources_source_exists else None,
    }


def restore_bootstrap_state(
    source_dir: Path,
    target_sector_dir: Path,
    *,
    hardlink_first: bool = True,
    exclude_relative_prefixes: tuple[str, ...] = (),
) -> dict[str, object]:
    source_dir = _require_directory(source_dir, "Bootstrap state source")
    required_counts = validate_state_source(source_dir)
    results_excluder = None
    normalized_excludes = tuple(
        str(prefix).strip("/").split("/") for prefix in exclude_relative_prefixes if str(prefix).strip("/")
    )

    def explicit_excluder(path: Path) -> bool:
        if not normalized_excludes:
            return False
        try:
            rel = path.resolve().relative_to(source_dir.resolve())
        except Exception:
            return False
        return any(rel.parts[: len(prefix_parts)] == tuple(prefix_parts) for prefix_parts in normalized_excludes)

    try:
        _infer_sector_name_from_results_dir(source_dir)
        nested_excluder = _build_nested_scenario_excluder(
            source_dir,
            shared_dir_names=DEFAULT_RESULTS_SHARED_TOPLEVEL_DIRS,
            scenario_markers=DEFAULT_RESULTS_SCENARIO_MARKERS,
        )
        results_excluder = lambda path: explicit_excluder(path) or nested_excluder(path)
    except ValueError:
        results_excluder = explicit_excluder if normalized_excludes else None

    summaries = [
        mirror_tree(
            source_dir,
            target_sector_dir,
            hardlink_first=hardlink_first,
            exclude_path=results_excluder,
        )
    ]
    target_resources_dir = _companion_resources_dir_for_results_dir(target_sector_dir)
    restored_resources = False

    resources_snapshot_dir = source_dir / "__resources_sector__"
    if resources_snapshot_dir.exists():
        summaries.append(mirror_tree(resources_snapshot_dir, target_resources_dir, hardlink_first=True))
        restored_resources = True
    else:
        try:
            companion_resources_source = _companion_resources_dir_for_results_dir(source_dir)
        except ValueError:
            companion_resources_source = None
        if companion_resources_source is not None and companion_resources_source.exists() and companion_resources_source.is_dir():
            resources_excluder = _build_nested_scenario_excluder(
                companion_resources_source,
                shared_dir_names=DEFAULT_RESOURCES_SHARED_TOPLEVEL_DIRS,
                scenario_markers=DEFAULT_RESOURCES_SCENARIO_MARKERS,
            )
            summaries.append(
                mirror_tree(
                    companion_resources_source,
                    target_resources_dir,
                    hardlink_first=hardlink_first,
                    exclude_path=resources_excluder,
                )
            )
            restored_resources = True

    summary = _merge_summary(summaries)
    return {
        "source_dir": str(source_dir.resolve()),
        "target_dir": str(target_sector_dir.resolve()),
        "required_artifact_counts": required_counts,
        "copied": summary.copied,
        "linked": summary.linked,
        "symlinked": summary.symlinked,
        "skipped_existing": summary.skipped_existing,
        "resources_target_dir": str(target_resources_dir.resolve()) if restored_resources else None,
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
