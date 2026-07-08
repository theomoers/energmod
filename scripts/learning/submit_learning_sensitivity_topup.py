#!/usr/bin/env python3
"""Generate or submit a top-up array for incomplete sensitivity ensemble tasks."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

from submit_learning_sensitivity_ensemble_array import (
    DEFAULT_CONDA_ENV,
    DEFAULT_GRID_MEM,
    DEFAULT_GRID_NCPUS,
    DEFAULT_GRID_SUBMIT,
    DEFAULT_JOB_ROOT,
    DEFAULT_RETRY_MIN_SECONDS,
    DEFAULT_SUBMIT_ROOT,
    DEFAULT_TASK_MAX_SECONDS,
    DEFAULT_TASK_RESERVE_SECONDS,
    DEFAULT_TIMEOUT_BIN,
    DEFAULT_TIMEOUT_GRACE_SECONDS,
    build_grid_run_cmd,
    compact_complete,
    resolve_archive_sector_names,
    resolve_task_sector_names,
    sanitize_token,
    write_csv_manifest,
)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
LOG_PATTERN = re.compile(
    r"run_learning_sensitivity_ensemble_array_task\.sh\.(?:p)?[oe](?P<jobid>\d+)\.(?P<taskid>\d+)$"
)


def load_manifest(submission_path: Path) -> tuple[Path, list[dict]]:
    submission_path = submission_path.resolve()
    if submission_path.is_file():
        manifest_path = submission_path
        submit_dir = manifest_path.parent
    else:
        submit_dir = submission_path
        manifest_path = submit_dir / "task_manifest.json"
    tasks = json.loads(manifest_path.read_text(encoding="utf-8"))
    return submit_dir, tasks


def load_submission_metadata(submit_dir: Path) -> dict:
    metadata_path = submit_dir / "submission_metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def source_submission_root(submit_dir: Path) -> Path:
    metadata = load_submission_metadata(submit_dir)
    source = metadata.get("source_submission")
    if source:
        return Path(source).resolve()
    return submit_dir.resolve()


def active_job_tasks() -> set[tuple[int, int]]:
    try:
        out = subprocess.run(["qstat"], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return set()
    active: set[tuple[int, int]] = set()
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("job-ID") or line.startswith("---"):
            continue
        parts = line.split()
        if len(parts) < 9 or not parts[0].isdigit():
            continue
        task_token = parts[-1]
        if task_token.isdigit():
            active.add((int(parts[0]), int(task_token)))
    return active


def latest_log_job_tasks(logs_dir: Path) -> set[tuple[int, int]]:
    tasks: dict[int, int] = {}
    for path in logs_dir.glob("run_learning_sensitivity_ensemble_array_task.sh.*"):
        match = LOG_PATTERN.match(path.name)
        if not match:
            continue
        task_id = int(match.group("taskid"))
        job_id = int(match.group("jobid"))
        tasks[task_id] = max(job_id, tasks.get(task_id, -1))
    return {(job_id, task_id) for task_id, job_id in tasks.items()}


def submission_has_running_tasks(submit_dir: Path) -> bool:
    logged = latest_log_job_tasks(submit_dir / "logs")
    if not logged:
        return False
    active = active_job_tasks()
    return any(item in active for item in logged)


def related_topup_dirs(submit_root: Path, original_submit_dir: Path) -> list[Path]:
    original = original_submit_dir.resolve()
    related = [original]
    if not submit_root.exists():
        return related
    for candidate in sorted(submit_root.iterdir()):
        if not candidate.is_dir() or candidate.resolve() == original:
            continue
        metadata = load_submission_metadata(candidate)
        if not metadata.get("topup"):
            continue
        source = metadata.get("source_submission")
        if source and Path(source).resolve() == original:
            related.append(candidate.resolve())
    return related


def ensure_no_running_tasks(submission_dirs: list[Path], allow_running: bool) -> None:
    if allow_running:
        return
    running = [path for path in submission_dirs if submission_has_running_tasks(path)]
    if running:
        joined = ", ".join(str(path) for path in running)
        raise RuntimeError(
            "Refusing to generate sensitivity top-up while related submissions still have running tasks: "
            f"{joined}. Use --allow-running to override."
        )


def build_topup_tasks(original_tasks: list[dict]) -> tuple[list[dict], dict[str, dict[str, int]]]:
    tasks = []
    target_counts = Counter(str(task["run_id"]) for task in original_tasks)
    completed_counts: Counter[str] = Counter()
    for task in original_tasks:
        if compact_complete(task):
            completed_counts[str(task["run_id"])] += 1
        else:
            topup_task = dict(task)
            topup_task["status"] = "topup"
            tasks.append(topup_task)

    summary = {}
    for run_id, target in sorted(target_counts.items()):
        completed = int(completed_counts.get(run_id, 0))
        summary[run_id] = {
            "target": int(target),
            "completed": completed,
            "missing": max(0, int(target) - completed),
        }
    return tasks, summary


def apply_submission_defaults(args, submit_dir: Path):
    metadata = load_submission_metadata(submit_dir)
    runtime = metadata.get("task_runtime_budget", {}) or {}
    if args.job_root is None:
        args.job_root = metadata.get("job_root", str(DEFAULT_JOB_ROOT))
    if args.conda_env is None:
        args.conda_env = metadata.get("conda_env", DEFAULT_CONDA_ENV)
    if args.grid_mem is None:
        args.grid_mem = metadata.get("grid_mem", DEFAULT_GRID_MEM)
    if args.grid_ncpus is None:
        args.grid_ncpus = metadata.get("grid_ncpus", DEFAULT_GRID_NCPUS)
    if args.grid_submit is None:
        args.grid_submit = metadata.get("grid_submit", DEFAULT_GRID_SUBMIT)
    if args.task_max_seconds is None:
        args.task_max_seconds = int(runtime.get("max_seconds", DEFAULT_TASK_MAX_SECONDS))
    if args.task_reserve_seconds is None:
        args.task_reserve_seconds = int(runtime.get("reserve_seconds", DEFAULT_TASK_RESERVE_SECONDS))
    if args.retry_min_seconds is None:
        args.retry_min_seconds = int(runtime.get("retry_min_seconds", DEFAULT_RETRY_MIN_SECONDS))
    if args.timeout_grace_seconds is None:
        args.timeout_grace_seconds = int(runtime.get("timeout_grace_seconds", DEFAULT_TIMEOUT_GRACE_SECONDS))
    if args.timeout_bin is None:
        args.timeout_bin = runtime.get("timeout_bin", DEFAULT_TIMEOUT_BIN)
    if args.grid_array_concurrency is None:
        args.grid_array_concurrency = None
    args.dry_run = False
    return args


def write_topup_submission(
    original_submit_dir: Path,
    original_metadata: dict,
    tasks: list[dict],
    summary: dict[str, dict[str, int]],
    args,
) -> tuple[Path, Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ensemble_name = str(original_metadata.get("ensemble_name") or (tasks[0]["ensemble_name"] if tasks else "sensitivity_ensemble"))
    submit_root = Path(args.submit_root).resolve()
    submit_dir = submit_root / f"{sanitize_token(ensemble_name)}_topup_{timestamp}"
    submit_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = submit_dir / "task_manifest.json"
    manifest_path.write_text(json.dumps(tasks, indent=2, sort_keys=True), encoding="utf-8")
    write_csv_manifest(submit_dir / "manifest.csv", tasks)

    metadata = {
        "submission_kind": "stochastic_learning_sensitivity_ensemble_topup",
        "source_submission": str(original_submit_dir.resolve()),
        "source_submission_kind": original_metadata.get("submission_kind"),
        "topup": True,
        "task_count": len(tasks),
        "summary": summary,
        "run_ids": sorted({task["run_id"] for task in tasks}),
        "resolved_sector_names": resolve_task_sector_names(tasks),
        "resolved_archive_sector_names": resolve_archive_sector_names(tasks),
        "job_root": str(Path(os.path.expandvars(args.job_root)).resolve()),
        "conda_env": args.conda_env,
        "grid_mem": args.grid_mem,
        "grid_ncpus": args.grid_ncpus,
        "grid_submit": args.grid_submit,
        "grid_array_concurrency": int(args.grid_array_concurrency) if args.grid_array_concurrency else None,
        "overwrite": bool(args.overwrite),
        "task_runtime_budget": {
            "max_seconds": int(args.task_max_seconds),
            "reserve_seconds": int(args.task_reserve_seconds),
            "retry_min_seconds": int(args.retry_min_seconds),
            "timeout_grace_seconds": int(args.timeout_grace_seconds),
            "timeout_bin": args.timeout_bin,
        },
    }
    metadata_path = submit_dir / "submission_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    (submit_dir / "logs").mkdir(parents=True, exist_ok=True)
    return submit_dir, manifest_path, metadata_path


def submit_topup(args) -> int:
    original_submit_dir, original_tasks = load_manifest(Path(args.submission))
    original_metadata = load_submission_metadata(original_submit_dir)
    args = apply_submission_defaults(args, original_submit_dir)
    related = related_topup_dirs(Path(args.submit_root).resolve(), original_submit_dir)
    ensure_no_running_tasks(related, args.allow_running)

    tasks, summary = build_topup_tasks(original_tasks)
    if not tasks:
        print("SENSITIVITY_TOPUP_STATUS no_topup_needed")
        print("SOURCE_SUBMISSION", original_submit_dir)
        print("SUMMARY", json.dumps(summary, sort_keys=True))
        return 0

    submit_dir, manifest_path, metadata_path = write_topup_submission(
        original_submit_dir,
        original_metadata,
        tasks,
        summary,
        args,
    )
    grid_run_cmd = build_grid_run_cmd(args, manifest_path, tasks)

    if args.print_only:
        print("SENSITIVITY_TOPUP_STATUS generated")
        print("SOURCE_SUBMISSION", original_submit_dir)
        print("TASK_MANIFEST", manifest_path)
        print("CSV_MANIFEST", submit_dir / "manifest.csv")
        print("SUBMISSION_METADATA", metadata_path)
        print("ARRAY_SIZE", len(tasks))
        print("RUN_IDS", ",".join(sorted({task["run_id"] for task in tasks})))
        print("LOG_DIR", submit_dir / "logs")
        print("SUMMARY", json.dumps(summary, sort_keys=True))
        print("GRID_RUN_CMD", " ".join(shlex.quote(part) for part in grid_run_cmd))
        return 0

    subprocess.run(grid_run_cmd, check=True, cwd=submit_dir / "logs")
    print("SENSITIVITY_TOPUP_STATUS submitted")
    print("SOURCE_SUBMISSION", original_submit_dir)
    print("TASK_MANIFEST", manifest_path)
    print("CSV_MANIFEST", submit_dir / "manifest.csv")
    print("SUBMISSION_METADATA", metadata_path)
    print("ARRAY_SIZE", len(tasks))
    print("LOG_DIR", submit_dir / "logs")
    print("SUMMARY", json.dumps(summary, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", help="Original sensitivity submission directory or task_manifest.json")
    parser.add_argument("--submit-root", default=str(DEFAULT_SUBMIT_ROOT))
    parser.add_argument("--job-root")
    parser.add_argument("--conda-env")
    parser.add_argument("--grid-mem")
    parser.add_argument("--grid-ncpus")
    parser.add_argument("--grid-submit")
    parser.add_argument(
        "--grid-array-concurrency",
        type=int,
        help=(
            "Maximum concurrent SGE array tasks for this top-up. "
            "Defaults to the sensitivity ensemble default when omitted."
        ),
    )
    parser.add_argument("--task-max-seconds", type=int)
    parser.add_argument("--task-reserve-seconds", type=int)
    parser.add_argument("--retry-min-seconds", type=int)
    parser.add_argument("--timeout-grace-seconds", type=int)
    parser.add_argument("--timeout-bin")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-running", action="store_true")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    return submit_topup(args)


if __name__ == "__main__":
    raise SystemExit(main())
