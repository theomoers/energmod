#!/usr/bin/env python3
"""Summarize one stochastic learning cluster submission from its manifest and logs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from bootstrap_state_store import resolve_scenario_sector_name

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
LEGACY_RESULTS_DIR = ROOT_DIR / "results" / "Global_200"
LOG_PATTERN = re.compile(r"run_learning_(?:cluster|sensitivity_ensemble)_array_task\.sh\.(?P<stream>[oe])(?P<jobid>\d+)\.(?P<taskid>\d+)$")
RULE_BLOCK_RE = re.compile(
    r"rule (?P<rule>[A-Za-z0-9_]+):(?P<body>.*?)(?=\n\[|\nrule |\Z)",
    re.DOTALL,
)
YEAR_RE = re.compile(r"planning_horizons=(\d{4})")


@dataclass
class TaskLogInfo:
    task_id: int
    job_id: int
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    stdout_mtime: float = -1.0
    stderr_mtime: float = -1.0


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


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path.resolve())
    return unique


def results_dirs_for_submission(submit_dir: Path, tasks: list[dict] | None = None) -> list[Path]:
    metadata = load_submission_metadata(submit_dir)
    candidates: list[Path] = []

    for sector_name in metadata.get("resolved_sector_names", []) or []:
        candidates.append((ROOT_DIR / "results" / Path(str(sector_name))).resolve())

    resolved_sector_name = metadata.get("resolved_sector_name")
    if resolved_sector_name:
        candidates.append((ROOT_DIR / "results" / Path(str(resolved_sector_name))).resolve())

    scenario_name = metadata.get("scenario_name")
    if scenario_name:
        candidates.append((ROOT_DIR / "results" / Path(resolve_scenario_sector_name(str(scenario_name)))).resolve())

    if tasks:
        for task in tasks:
            task_scenario = str(task.get("scenario_name", "")).strip()
            if not task_scenario:
                continue
            candidates.append(
                (ROOT_DIR / "results" / Path(resolve_scenario_sector_name(task_scenario))).resolve()
            )

    candidates.append(LEGACY_RESULTS_DIR.resolve())
    return _dedupe_paths(candidates)


def load_planning_horizons() -> list[int]:
    text = (ROOT_DIR / "config.myopic.yaml").read_text(encoding="utf-8", errors="replace")
    marker = "planning_horizons:"
    start = text.find(marker)
    if start == -1:
        return [2020, 2025, 2030, 2035, 2040, 2045, 2050]
    tail = text[start:start + 400]
    years = sorted({int(x) for x in re.findall(r"\b(20\d{2})\b", tail)})
    return years or [2020, 2025, 2030, 2035, 2040, 2045, 2050]


def task_seed(task: dict) -> int:
    value = task.get("seed", task.get("learning_seed"))
    if value is None:
        raise KeyError("task manifest requires seed or learning_seed")
    return int(value)


def token_for_seed(model: str, seed: int) -> str:
    if model == "legacy_curve":
        return "deterministic"
    return f"s{int(seed):04d}"


def _task_results_dirs(task: dict, base_results_dirs: list[Path] | None) -> list[Path]:
    # Prefer the fully resolved per-task sector path. This avoids scanning all
    # sensitivity sectors for every task and supports non-default sector roots.
    # Compact completion markers are archived at archive_sector_name, whereas
    # sector_name points at the per-draw working directory.
    sector_name = str(task.get("archive_sector_name", "")).strip()
    if not sector_name:
        sector_name = str(task.get("sector_name", "")).strip()
    if not sector_name:
        sector_name = str(task.get("resolved_sector_name", "")).strip()
    if sector_name:
        return [(ROOT_DIR / "results" / Path(sector_name)).resolve()]
    scenario = str(task.get("scenario_name", "")).strip()
    if scenario:
        return [(ROOT_DIR / "results" / Path(resolve_scenario_sector_name(scenario))).resolve()]
    if base_results_dirs:
        return _dedupe_paths(list(base_results_dirs))
    return [LEGACY_RESULTS_DIR.resolve()]

def compact_complete(task: dict, results_dirs: list[Path] | None = None) -> bool:
    scenario = str(task["scenario_name"])
    model = str(task["model"])
    token = token_for_seed(model, task_seed(task))
    for results_dir in _task_results_dirs(task, results_dirs):
        compact_dir = results_dir / "learning-compact" / scenario / model / f"seed_{token}"
        if list(compact_dir.glob("raw_cleanup_complete*.txt")):
            return True
        if list(compact_dir.glob("compact_complete*.json")):
            return True
    return False


def fallback_complete(task: dict, horizons: list[int]) -> bool:
    return False


def collect_logs(logs_dir: Path) -> dict[int, TaskLogInfo]:
    info: dict[int, TaskLogInfo] = {}
    for path in logs_dir.glob("run_learning_*_array_task.sh.*"):
        match = LOG_PATTERN.match(path.name)
        if not match:
            continue
        task_id = int(match.group("taskid"))
        job_id = int(match.group("jobid"))
        stream = match.group("stream")
        existing = info.get(task_id)
        if existing is None or job_id > existing.job_id:
            existing = TaskLogInfo(task_id=task_id, job_id=job_id)
            info[task_id] = existing
        if job_id != existing.job_id:
            continue
        mtime = path.stat().st_mtime
        if stream == "o" and mtime >= existing.stdout_mtime:
            existing.stdout_path = path
            existing.stdout_mtime = mtime
        if stream == "e" and mtime >= existing.stderr_mtime:
            existing.stderr_path = path
            existing.stderr_mtime = mtime
    return info


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
        if len(parts) < 9:
            continue
        if not parts[0].isdigit():
            continue
        job_id = int(parts[0])
        task_token = parts[-1]
        if task_token.isdigit():
            active.add((job_id, int(task_token)))
    return active


def submission_has_running_tasks(submit_dir: Path) -> bool:
    logs = collect_logs(submit_dir / "logs")
    active = active_job_tasks()
    return any((info.job_id, task_id) in active for task_id, info in logs.items())


def read_text(path: Path | None, limit: int = 120_000) -> str:
    if path is None or not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > limit:
        return text[-limit:]
    return text


def classify_failure(text: str) -> str | None:
    lowered = text.lower()
    checks = [
        ("too many sessions", "license"),
        ("missinginputexception", "missing-input"),
        ("condition='time_limit'", "time_limit"),
        ("termination condition: time_limit", "time_limit"),
        ("hard wallclock limit", "time_limit"),
        ("timed out", "time_limit"),
        ("timeout: sending signal", "time_limit"),
        ("exit status 124", "time_limit"),
        ("returned non-zero exit status 124", "time_limit"),
        ("out of memory", "memory"),
        ("optimization exhausted available memory", "memory"),
        ("_arraymemoryerror", "memory"),
        ("calledprocesserror", "subprocess"),
        ("keyboardinterrupt", "interrupted"),
    ]
    for needle, label in checks:
        if needle in lowered:
            return label
    return None


def infer_stage(text: str, horizons: list[int]) -> str:
    latest_rule = None
    latest_year = None
    for match in RULE_BLOCK_RE.finditer(text):
        latest_rule = match.group("rule")
        years = YEAR_RE.findall(match.group("body"))
        if years:
            latest_year = int(years[-1])
    if latest_rule is None:
        if "Building DAG of jobs" in text:
            return "building DAG"
        return "starting"

    labels = {
        "add_brownfield": "brownfield",
        "apply_learning_costs": "apply_learning",
        "solve_network_myopic": "solving",
        "export_postsolve_learning_costs": "postsolve_export",
        "export_stochastic_run_bundle": "compact_export",
        "cleanup_stochastic_branch_raw_artifacts": "raw_cleanup",
    }
    label = labels.get(latest_rule, latest_rule)
    if latest_year is None:
        return label
    progress = ""
    if horizons and latest_year in horizons:
        idx = horizons.index(latest_year) + 1
        progress = f" ({idx}/{len(horizons)})"
    return f"{label} {latest_year}{progress}"


def format_table(rows: list[dict]) -> str:
    columns = [
        ("task", "task"),
        ("seed", "seed"),
        ("status", "status"),
        ("stage", "stage"),
        ("job", "job"),
        ("host", "host"),
    ]
    pending_count = sum(1 for row in rows if row.get("status") == "not_started")
    visible_rows = [row for row in rows if row.get("status") != "not_started"]
    if pending_count:
        visible_rows.append(
            {
                "task": f"{pending_count} jobs",
                "seed": "",
                "status": "not_started",
                "stage": "pending",
                "job": "",
                "host": "",
            }
        )
    widths = {}
    for key, header in columns:
        widths[key] = max(len(header), *(len(str(row.get(key, ""))) for row in visible_rows)) if visible_rows else len(header)
    lines = []
    header = "  ".join(header.ljust(widths[key]) for key, header in columns)
    sep = "  ".join("-" * widths[key] for key, _ in columns)
    lines.extend([header, sep])
    for row in visible_rows:
        lines.append("  ".join(str(row.get(key, "")).ljust(widths[key]) for key, _ in columns))
    return "\n".join(lines)


def extract_host(stdout_text: str) -> str:
    for line in stdout_text.splitlines():
        if line.startswith("Host:"):
            return line.split(":", 1)[1].strip()
    return ""


def summarize(submit_dir: Path, tasks: list[dict]) -> str:
    horizons = load_planning_horizons()
    # branch runs skip bootstrap year in progress reporting
    if horizons and horizons[0] == 2020:
        progress_horizons = horizons[1:]
    else:
        progress_horizons = horizons

    logs = collect_logs(submit_dir / "logs")
    active = active_job_tasks()
    submission_results_dirs = results_dirs_for_submission(submit_dir, tasks)
    rows = []
    counts = {"completed": 0, "running": 0, "failed": 0, "not_started": 0}

    for idx, task in enumerate(tasks, start=1):
        info = logs.get(idx)
        stdout_text = read_text(info.stdout_path if info else None)
        stderr_text = read_text(info.stderr_path if info else None)
        combined = stdout_text + "\n" + stderr_text
        job = str(info.job_id) if info else ""
        host = extract_host(stdout_text)
        if compact_complete(task, results_dirs=submission_results_dirs):
            status = "completed"
            stage = "complete"
        elif info is None:
            status = "not_started"
            stage = "pending"
        elif (info.job_id, idx) in active:
            status = "running"
            stage = infer_stage(combined, progress_horizons)
        else:
            reason = classify_failure(combined)
            status = "failed"
            display_status = f"failed:{reason}" if reason else status
            stage = f"failed:{reason}" if reason else infer_stage(combined, progress_horizons)
        counts[status] += 1
        rows.append(
            {
                "task": idx,
                "seed": task_seed(task),
                "status": display_status if status == "failed" else status,
                "stage": stage,
                "job": job,
                "host": host,
            }
        )

    summary_lines = [
        f"Submission: {submit_dir}",
        f"Total tasks: {len(tasks)}",
        f"Completed: {counts['completed']}  Running: {counts['running']}  Failed: {counts['failed']}  Not started: {counts['not_started']}",
        "",
        format_table(rows),
    ]
    return "\n".join(summary_lines)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", help="Submission directory or task_manifest.json path")
    args = parser.parse_args(list(argv) if argv is not None else None)
    submit_dir, tasks = load_manifest(Path(args.submission))
    print(summarize(submit_dir, tasks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
