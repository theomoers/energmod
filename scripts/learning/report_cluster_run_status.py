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

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT_DIR / "results" / "Global_200"
LOG_PATTERN = re.compile(r"run_learning_cluster_array_task\.sh\.(?P<stream>[oe])(?P<jobid>\d+)\.(?P<taskid>\d+)$")
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


def load_planning_horizons() -> list[int]:
    text = (ROOT_DIR / "config.myopic.yaml").read_text(encoding="utf-8", errors="replace")
    marker = "planning_horizons:"
    start = text.find(marker)
    if start == -1:
        return [2020, 2025, 2030, 2035, 2040, 2045, 2050]
    tail = text[start:start + 400]
    years = sorted({int(x) for x in re.findall(r"\b(20\d{2})\b", tail)})
    return years or [2020, 2025, 2030, 2035, 2040, 2045, 2050]


def token_for_seed(model: str, seed: int) -> str:
    if model == "legacy_curve":
        return "deterministic"
    return f"s{int(seed):04d}"


def compact_complete(task: dict) -> bool:
    scenario = str(task["scenario_name"])
    model = str(task["model"])
    token = token_for_seed(model, int(task["seed"]))
    compact_dir = RESULTS_DIR / "learning-compact" / scenario / model / f"seed_{token}"
    if list(compact_dir.glob("raw_cleanup_complete*.txt")):
        return True
    if list(compact_dir.glob("compact_complete*.json")):
        return True
    return False


def fallback_complete(task: dict, horizons: list[int]) -> bool:
    return False


def collect_logs(logs_dir: Path) -> dict[int, TaskLogInfo]:
    info: dict[int, TaskLogInfo] = {}
    for path in logs_dir.glob("run_learning_cluster_array_task.sh.*"):
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


def read_text(path: Path | None, limit: int = 120_000) -> str:
    if path is None or not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > limit:
        return text[-limit:]
    return text


def classify_failure(text: str) -> str | None:
    checks = [
        ("Too many sessions", "license"),
        ("MissingInputException", "missing-input"),
        ("Out of memory", "memory"),
        ("Optimization exhausted available memory", "memory"),
        ("_ArrayMemoryError", "memory"),
        ("CalledProcessError", "subprocess"),
        ("KeyboardInterrupt", "interrupted"),
    ]
    for needle, label in checks:
        if needle in text:
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
    rows = []
    counts = {"completed": 0, "running": 0, "failed": 0, "not_started": 0}

    for idx, task in enumerate(tasks, start=1):
        info = logs.get(idx)
        stdout_text = read_text(info.stdout_path if info else None)
        stderr_text = read_text(info.stderr_path if info else None)
        combined = stdout_text + "\n" + stderr_text
        job = str(info.job_id) if info else ""
        host = extract_host(stdout_text)
        if compact_complete(task):
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
            stage = f"failed:{reason}" if reason else infer_stage(combined, progress_horizons)
        counts[status] += 1
        rows.append(
            {
                "task": idx,
                "seed": task["seed"],
                "status": status,
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
