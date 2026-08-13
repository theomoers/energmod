#!/usr/bin/env python3
"""Generate or submit a stochastic learning top-up array for failed or timed-out draws."""

from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from bootstrap_state_store import resolve_scenario_sector_name
import report_cluster_run_status as status_report
from submit_learning_cluster_array import (
    DEFAULT_CONDA_ENV,
    DEFAULT_JOB_ROOT,
    build_grid_run_cmd,
    default_configfiles,
    load_merged_config,
)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent


def load_submission_metadata(submit_dir: Path) -> dict:
    metadata_path = submit_dir / "submission_metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def scenario_name_from_tasks(tasks: list[dict]) -> str:
    scenarios = sorted({str(task["scenario_name"]) for task in tasks})
    if len(scenarios) != 1:
        raise ValueError(f"Expected exactly one scenario_name in manifest, found: {scenarios}")
    return scenarios[0]


def resolve_configfiles(submit_dir: Path, extra_configfiles: list[str]) -> list[str]:
    if extra_configfiles:
        configfiles = default_configfiles()
        configfiles.extend(extra_configfiles)
        return configfiles

    metadata = load_submission_metadata(submit_dir)
    effective_config = metadata.get("effective_config")
    if effective_config:
        effective_path = Path(effective_config).resolve()
        if effective_path.is_file():
            return [str(effective_path)]
    metadata_configfiles = metadata.get("configfiles", [])
    if metadata_configfiles:
        return [str(Path(cfg).resolve()) for cfg in metadata_configfiles]
    return default_configfiles()


def apply_submission_defaults(args, submit_dir: Path):
    metadata = load_submission_metadata(submit_dir)
    if args.job_root is None:
        args.job_root = metadata.get("job_root", str(DEFAULT_JOB_ROOT))
    if args.conda_env is None:
        args.conda_env = metadata.get("conda_env", DEFAULT_CONDA_ENV)
    if args.run_mode is None:
        args.run_mode = metadata.get("run_mode", "branch")
    if args.grid_mem is None:
        args.grid_mem = metadata.get("grid_mem", "90G")
    if args.grid_ncpus is None:
        args.grid_ncpus = metadata.get("grid_ncpus", "40")
    if args.grid_submit is None:
        args.grid_submit = metadata.get("grid_submit", "batch")
    if getattr(args, "grid_array_concurrency", None) is None:
        args.grid_array_concurrency = metadata.get("grid_array_concurrency", 200)
    return args


def related_submission_dirs(submit_root: Path, scenario_name: str) -> list[Path]:
    related = []
    for candidate in sorted(submit_root.iterdir()):
        manifest_path = candidate / "task_manifest.json"
        if not candidate.is_dir() or not manifest_path.exists():
            continue
        try:
            _, tasks = status_report.load_manifest(candidate)
        except Exception:
            continue
        if tasks and scenario_name_from_tasks(tasks) == scenario_name:
            related.append(candidate)
    return related


def used_seeds_by_model(submission_dirs: list[Path]) -> dict[str, set[int]]:
    used: dict[str, set[int]] = defaultdict(set)
    for submit_dir in submission_dirs:
        _, tasks = status_report.load_manifest(submit_dir)
        for task in tasks:
            used[str(task["model"])].add(int(task["seed"]))
    return used


def completed_seeds_by_model(submission_dirs: list[Path]) -> dict[str, set[int]]:
    completed: dict[str, set[int]] = defaultdict(set)
    for submit_dir in submission_dirs:
        _, tasks = status_report.load_manifest(submit_dir)
        results_dirs = status_report.results_dirs_for_submission(submit_dir, tasks)
        for task in tasks:
            if status_report.compact_complete(task, results_dirs=results_dirs):
                completed[str(task["model"])].add(int(task["seed"]))
    return completed


def ensure_no_running_tasks(submission_dirs: list[Path], allow_running: bool) -> None:
    if allow_running:
        return
    running = [submit_dir for submit_dir in submission_dirs if status_report.submission_has_running_tasks(submit_dir)]
    if running:
        joined = ", ".join(str(path) for path in running)
        raise RuntimeError(
            "Refusing to generate top-up while related submissions still have running tasks: "
            f"{joined}. Use --allow-running to override."
        )


def next_sequential_seeds(used: set[int], count: int) -> list[int]:
    start = (max(used) + 1) if used else 0
    return list(range(start, start + count))


def next_random_seeds(used: set[int], count: int, random_seed: int, seed_upper_bound: int) -> list[int]:
    if len(used) + count > seed_upper_bound:
        raise ValueError("Not enough unused seeds remain under learning.monte_carlo.seed_upper_bound")

    rng = random.Random(random_seed)
    selected = []
    seen = set(used)
    while len(selected) < count:
        candidate = rng.randrange(seed_upper_bound)
        if candidate in seen:
            continue
        seen.add(candidate)
        selected.append(candidate)
    return selected


def build_topup_tasks(
    original_submit_dir: Path,
    submission_dirs: list[Path],
    configfiles: list[str],
) -> tuple[str, list[dict], dict[str, dict[str, int]]]:
    _, original_tasks = status_report.load_manifest(original_submit_dir)
    scenario_name = scenario_name_from_tasks(original_tasks)
    cfg = load_merged_config(configfiles)
    learning_cfg = cfg.get("learning", {}) or {}
    mc_cfg = learning_cfg.get("monte_carlo", {}) or {}
    seed_mode = str(mc_cfg.get("seed_mode", "random"))
    random_seed = int(mc_cfg.get("random_seed", learning_cfg.get("seed", 0) or 0))
    seed_upper_bound = int(mc_cfg.get("seed_upper_bound", 1000000000) or 1000000000)

    target_counts = Counter(str(task["model"]) for task in original_tasks)
    template_by_model = {}
    for task in original_tasks:
        template_by_model.setdefault(str(task["model"]), task)
    used_by_model = used_seeds_by_model(submission_dirs)
    completed_by_model = completed_seeds_by_model(submission_dirs)

    tasks = []
    summary = {}
    for model, target in target_counts.items():
        completed = len(completed_by_model.get(model, set()))
        missing = max(0, target - completed)
        summary[model] = {"target": target, "completed": completed, "missing": missing}
        if missing == 0:
            continue

        used = used_by_model.get(model, set())
        if seed_mode == "sequential":
            replacements = next_sequential_seeds(used, missing)
        elif seed_mode == "random":
            replacements = next_random_seeds(used, missing, random_seed, seed_upper_bound)
        else:
            raise ValueError(f"Unsupported learning.monte_carlo.seed_mode: {seed_mode}")

        for seed in replacements:
            task = dict(template_by_model[model])
            task["seed"] = int(seed)
            tasks.append(task)
            used.add(int(seed))

    return scenario_name, tasks, summary


def write_topup_submission(
    original_submit_dir: Path,
    scenario_name: str,
    tasks: list[dict],
    summary: dict[str, dict[str, int]],
    configfiles: list[str],
    args,
) -> tuple[Path, Path, Path, dict]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submit_root = Path(args.submit_root).resolve()
    submit_dir = submit_root / f"{scenario_name}_topup_{timestamp}"
    submit_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = submit_dir / "task_manifest.json"
    manifest_path.write_text(json.dumps(tasks, indent=2, sort_keys=True), encoding="utf-8")

    resolved_sector_name = str(tasks[0].get("resolved_sector_name") or resolve_scenario_sector_name(scenario_name))
    metadata_payload = {
        "scenario_name": scenario_name,
        "resolved_sector_name": resolved_sector_name,
        "resolved_sector_names": [resolved_sector_name],
        "run_mode": args.run_mode,
        "source_submission": str(original_submit_dir.resolve()),
        "configfiles": [str(Path(cfg).resolve()) for cfg in configfiles],
        "grid_mem": args.grid_mem,
        "grid_ncpus": args.grid_ncpus,
        "grid_submit": args.grid_submit,
        "job_root": str(Path(os.path.expandvars(args.job_root)).resolve()),
        "conda_env": args.conda_env,
        "task_count": len(tasks),
        "topup": True,
        "summary": summary,
    }
    metadata_path = submit_dir / "submission_metadata.json"
    metadata_path.write_text(json.dumps(metadata_payload, indent=2, sort_keys=True), encoding="utf-8")

    logs_dir = submit_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return submit_dir, manifest_path, metadata_path, metadata_payload


def submit_topup(args) -> int:
    original_submit_dir, original_tasks = status_report.load_manifest(Path(args.submission))
    scenario_name = scenario_name_from_tasks(original_tasks)
    submit_root = Path(args.submit_root).resolve()
    args = apply_submission_defaults(args, original_submit_dir)
    configfiles = resolve_configfiles(original_submit_dir, args.configfiles)
    related_dirs = related_submission_dirs(submit_root, scenario_name)
    if original_submit_dir not in related_dirs:
        related_dirs.append(original_submit_dir)
        related_dirs.sort()

    ensure_no_running_tasks(related_dirs, args.allow_running)
    scenario_name, tasks, summary = build_topup_tasks(original_submit_dir, related_dirs, configfiles)
    if not tasks:
        print("TOPUP_STATUS no_topup_needed")
        print("SCENARIO", scenario_name)
        print("SUMMARY", json.dumps(summary, sort_keys=True))
        return 0

    submit_dir, manifest_path, metadata_path, _ = write_topup_submission(
        original_submit_dir,
        scenario_name,
        tasks,
        summary,
        configfiles,
        args,
    )

    worker_script = str((SCRIPT_DIR / "run_learning_cluster_array_task.sh").resolve())
    grid_run_cmd = build_grid_run_cmd(args, manifest_path, len(tasks), worker_script)
    if args.print_only:
        print("TOPUP_STATUS generated")
        print("TASK_MANIFEST", manifest_path)
        print("SUBMISSION_METADATA", metadata_path)
        print("ARRAY_SIZE", len(tasks))
        print("LOG_DIR", submit_dir / "logs")
        print("SUMMARY", json.dumps(summary, sort_keys=True))
        print("GRID_RUN_CMD", " ".join(shlex.quote(part) for part in grid_run_cmd))
        return 0

    subprocess.run(grid_run_cmd, check=True, cwd=submit_dir / "logs")
    print("TOPUP_STATUS submitted")
    print("TASK_MANIFEST", manifest_path)
    print("SUBMISSION_METADATA", metadata_path)
    print("ARRAY_SIZE", len(tasks))
    print("LOG_DIR", submit_dir / "logs")
    print("SUMMARY", json.dumps(summary, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", help="Original submission directory or task_manifest.json path")
    parser.add_argument("--submit-root", default="cluster_submissions")
    parser.add_argument("--job-root")
    parser.add_argument("--conda-env")
    parser.add_argument("--run-mode", choices=["branch", "full"])
    parser.add_argument("--grid-mem")
    parser.add_argument("--grid-ncpus")
    parser.add_argument("--grid-submit")
    parser.add_argument("--grid-array-concurrency", type=int)
    parser.add_argument("--allow-running", action="store_true")
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("configfiles", nargs="*")
    args = parser.parse_args()
    return submit_topup(args)


if __name__ == "__main__":
    raise SystemExit(main())
