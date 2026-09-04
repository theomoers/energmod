#!/usr/bin/env python3
"""Remove only failed sensitivity-ensemble per-draw working state."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from submit_learning_sensitivity_ensemble_array import (
    ROOT_DIR,
    compact_complete,
    sanitize_token,
)


def load_tasks(submission: Path) -> list[dict]:
    manifest = submission / "task_manifest.json" if submission.is_dir() else submission
    return json.loads(manifest.read_text(encoding="utf-8"))


def job_dir_for_task(task: dict, job_root: Path) -> Path:
    job_tag = (
        f"sensitivity_{sanitize_token(str(task['run_id']))}_"
        f"{sanitize_token(str(task['model']))}_s{int(task['learning_seed']):04d}"
    )
    return job_root / f"energymod_{job_tag}"


def targets_for_task(task: dict) -> tuple[Path, Path, Path]:
    sector = Path(str(task["sector_name"]))
    if not (
        len(sector.parts) == 4
        and sector.parts[0] == "Global_200_v8"
        and sector.parts[1].startswith("sens_")
        and sector.parts[2] == "draws"
    ):
        raise RuntimeError(f"Refusing unexpected sensitivity sector path: {sector}")

    job_root = ROOT_DIR / "cluster_workdirs"
    job_dir = job_dir_for_task(task, job_root)
    if job_dir.parent != job_root or not job_dir.name.startswith("energymod_sensitivity_"):
        raise RuntimeError(f"Refusing unexpected staged job path: {job_dir}")

    return (
        ROOT_DIR / "results" / sector,
        ROOT_DIR / "resources" / sector,
        job_dir,
    )


def remove_target(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    else:
        raise RuntimeError(f"Refusing non-directory cleanup target: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete targets. Without this flag, only report the plan.",
    )
    args = parser.parse_args()

    tasks = load_tasks(args.submission.resolve())
    failed = [task for task in tasks if not compact_complete(task)]
    targets = [path for task in failed for path in targets_for_task(task)]
    existing = [path for path in targets if path.exists() or path.is_symlink()]

    result_root = ROOT_DIR / "results"
    resource_root = ROOT_DIR / "resources"
    job_root = ROOT_DIR / "cluster_workdirs"
    print(f"Incomplete manifest tasks: {len(failed)}")
    print(f"Existing cleanup targets: {len(existing)}")
    print(f"  results: {sum(path.is_relative_to(result_root) for path in existing)}")
    print(f"  resources: {sum(path.is_relative_to(resource_root) for path in existing)}")
    print(f"  cluster_workdirs: {sum(path.is_relative_to(job_root) for path in existing)}")
    print(f"Protected compact-complete tasks: {len(tasks) - len(failed)}")

    if not args.apply:
        print("Dry run only; pass --apply to remove these failed-task directories.")
        return 0

    errors: list[str] = []
    removed = 0
    for path in existing:
        try:
            remove_target(path)
            removed += 1
        except OSError as exc:
            errors.append(f"{path}: {exc}")
    print(f"Removed cleanup targets: {removed}")
    if errors:
        print("Cleanup errors:", *errors, sep="\n", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
