#!/usr/bin/env python3
"""Submit or execute a two-run deterministic benchmark array workflow."""

import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
DEFAULT_JOB_ROOT = ROOT_DIR / "cluster_workdirs"
DEFAULT_SUBMIT_ROOT = ROOT_DIR / "cluster_submissions"
DEFAULT_CONDA_ENV = "/shared/share_cki25/envs/sh-pypsa-earth-main"
DEFAULT_GRID_MEM = "90G"
DEFAULT_GRID_NCPUS = "12"
DEFAULT_GRID_SUBMIT = "batch"

COMMON_CONFIG = ROOT_DIR / "configs/benchmarks/learning.benchmark_common.yaml"
RUN1_CONFIG = ROOT_DIR / "configs/benchmarks/learning.benchmark_run1_deterministic_wright.yaml"
RUN2_CONFIG = ROOT_DIR / "configs/benchmarks/learning.benchmark_run2_weo2025_exogenous.yaml"

BENCHMARK_TASKS = [
    {
        "name": "run1_deterministic_wright",
        "scenario_name": "benchmark_det_wright_caps",
        "model": "legacy_curve",
        "run_mode": "full",
        "description": (
            "Deterministic Wright-curve learning with direct lagged C=AQ^-b cost updates "
            "and deterministic deployment hard caps."
        ),
        "configfiles": [
            str(COMMON_CONFIG.resolve()),
            str(RUN1_CONFIG.resolve()),
        ],
    },
    {
        "name": "run2_weo2025_exogenous",
        "scenario_name": "benchmark_weo2025_caps",
        "model": "iea_weo_exogenous_path",
        "run_mode": "full",
        "description": (
            "Exogenous WEO 2025 clean-tech cost path with no endogenous learning and "
            "the same deterministic deployment hard caps."
        ),
        "configfiles": [
            str(COMMON_CONFIG.resolve()),
            str(RUN2_CONFIG.resolve()),
        ],
    },
]


def build_grid_run_cmd(args, manifest_path, worker_script):
    return [
        "grid_run",
        f"--grid_mem={args.grid_mem}",
        f"--grid_ncpus={args.grid_ncpus}",
        f"--grid_submit={args.grid_submit}",
        "--grid_array=1-2",
        worker_script,
        str(manifest_path),
        f"LEARNING_JOB_ROOT={Path(os.path.expandvars(args.job_root)).resolve()}",
        f"LEARNING_CONDA_ENV={args.conda_env}",
    ]


def write_submission_metadata(submit_dir, args, manifest_path):
    metadata = {
        "submission_kind": "deterministic_learning_benchmark_array",
        "task_manifest": str(manifest_path.resolve()),
        "task_count": len(BENCHMARK_TASKS),
        "job_root": str(Path(os.path.expandvars(args.job_root)).resolve()),
        "conda_env": args.conda_env,
        "grid_mem": args.grid_mem,
        "grid_ncpus": args.grid_ncpus,
        "grid_submit": args.grid_submit,
        "tasks": BENCHMARK_TASKS,
    }
    path = submit_dir / "submission_metadata.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return path


def submit_array(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submit_root = Path(args.submit_root).resolve()
    submit_dir = submit_root / f"deterministic_learning_benchmarks_{timestamp}"
    submit_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = submit_dir / "task_manifest.json"
    manifest_path.write_text(json.dumps(BENCHMARK_TASKS, indent=2, sort_keys=True), encoding="utf-8")
    metadata_path = write_submission_metadata(submit_dir, args, manifest_path)

    logs_dir = submit_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    worker_script = str((SCRIPT_DIR / "run_learning_benchmark_array_task.sh").resolve())
    grid_run_cmd = build_grid_run_cmd(args, manifest_path, worker_script)

    if args.print_only:
        print("TASK_MANIFEST", manifest_path)
        print("SUBMISSION_METADATA", metadata_path)
        print("ARRAY_SIZE", len(BENCHMARK_TASKS))
        print("LOG_DIR", logs_dir)
        print("GRID_RUN_CMD", " ".join(shlex.quote(part) for part in grid_run_cmd))
        return

    subprocess.run(grid_run_cmd, check=True, cwd=logs_dir)
    print("TASK_MANIFEST", manifest_path)
    print("SUBMISSION_METADATA", metadata_path)
    print("ARRAY_SIZE", len(BENCHMARK_TASKS))
    print("LOG_DIR", logs_dir)


def run_worker(args):
    sge_task_id = os.environ.get("SGE_TASK_ID")
    if sge_task_id is None:
        raise RuntimeError("SGE_TASK_ID environment variable not found")

    task_id = int(sge_task_id)
    tasks = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    task_index = task_id - 1
    if task_index < 0 or task_index >= len(tasks):
        raise RuntimeError(f"Task index {task_id} outside manifest range 1-{len(tasks)}")

    task = tasks[task_index]
    cmd = [
        "bash",
        str((SCRIPT_DIR / "run_learning_benchmark_job.sh").resolve()),
        str(task["model"]),
        str(Path(os.path.expandvars(args.job_root)).resolve()),
        "--scenario-name",
        str(task["scenario_name"]),
        "--run-mode",
        str(task.get("run_mode", "full")),
    ]
    for configfile in task.get("configfiles", []):
        cmd.extend(["--configfile", str(configfile)])
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--manifest")
    parser.add_argument("--job-root", default=str(DEFAULT_JOB_ROOT))
    parser.add_argument("--submit-root", default=str(DEFAULT_SUBMIT_ROOT))
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV)
    parser.add_argument("--grid-mem", default=DEFAULT_GRID_MEM)
    parser.add_argument("--grid-ncpus", default=DEFAULT_GRID_NCPUS)
    parser.add_argument("--grid-submit", default=DEFAULT_GRID_SUBMIT)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    if args.worker:
        if not args.manifest:
            raise ValueError("--manifest is required in worker mode")
        run_worker(args)
        return

    submit_array(args)


if __name__ == "__main__":
    main()
