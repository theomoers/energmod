#!/usr/bin/env python3
"""Submit or execute the supported SGE array workflow for stochastic learning runs."""

import argparse
import json
import os
import random
import shlex
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
DEFAULT_JOB_ROOT = ROOT_DIR / "cluster_workdirs"
DEFAULT_CONDA_ENV = "/shared/share_cki25/envs/sh-pypsa-earth-main"
SUPPORTED_MODELS = [
    "shared_state_bayesian_regime_wright",
    "way_fixed_rho_benchmark_035",
    "correlated_geometric_random_walk",
]


def deep_update(base, update):
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def load_merged_config(configfiles):
    merged = {}
    for cfg in configfiles:
        with open(cfg, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        deep_update(merged, payload)
    return merged


def build_seed_list(mc_cfg, learning_cfg):
    draws = int(mc_cfg.get("draws", 0) or 0)
    if draws <= 0:
        raise ValueError("learning.monte_carlo.draws must be > 0")

    seed_mode = str(mc_cfg.get("seed_mode", "random"))
    random_seed = int(mc_cfg.get("random_seed", learning_cfg.get("seed", 0) or 0))

    if seed_mode == "sequential":
        return list(range(draws))

    if seed_mode == "random":
        seed_upper_bound = int(mc_cfg.get("seed_upper_bound", 1000000000) or 1000000000)
        if draws > seed_upper_bound:
            raise ValueError(
                "learning.monte_carlo.draws cannot exceed learning.monte_carlo.seed_upper_bound"
            )
        rng = random.Random(random_seed)
        return rng.sample(range(seed_upper_bound), draws)

    raise ValueError(f"Unsupported learning.monte_carlo.seed_mode: {seed_mode}")


def build_tasks(configfiles, scenario_name):
    cfg = load_merged_config(configfiles)
    learning_cfg = cfg.get("learning", {}) or {}
    mc_cfg = learning_cfg.get("monte_carlo", {}) or {}
    if not mc_cfg.get("enable", False):
        raise ValueError("learning.monte_carlo.enable must be true for array submission")

    learning_rates = [
        str(x) for x in ((cfg.get("scenario", {}) or {}).get("learning_rate", []) or [])
    ]
    if any(rate != "base" for rate in learning_rates):
        raise ValueError("Cluster stochastic runs require scenario.learning_rate to contain only 'base'")

    models = [str(x) for x in (mc_cfg.get("stochastic_models", []) or [])]
    if not models:
        raise ValueError("learning.monte_carlo.stochastic_models must be non-empty")
    unsupported = sorted(set(models) - set(SUPPORTED_MODELS))
    if unsupported:
        raise ValueError(f"Unsupported stochastic models: {unsupported}")

    seeds = build_seed_list(mc_cfg, learning_cfg)
    tasks = []
    for model in models:
        for seed in seeds:
            tasks.append(
                {
                    "scenario_name": scenario_name,
                    "model": model,
                    "seed": int(seed),
                }
            )
    return tasks


def build_grid_run_cmd(args, manifest_path, task_count, worker_script):
    cmd = [
        "grid_run",
        f"--grid_mem={args.grid_mem}",
        f"--grid_ncpus={args.grid_ncpus}",
        f"--grid_submit={args.grid_submit}",
        f"--grid_array=1-{task_count}",
        worker_script,
        str(manifest_path),
        f"LEARNING_JOB_ROOT={Path(os.path.expandvars(args.job_root)).resolve()}",
        f"LEARNING_CONDA_ENV={args.conda_env}",
        f"LEARNING_RUN_MODE={args.run_mode}",
    ]
    return cmd


def submit_array(args):
    configfiles = [
        str((ROOT_DIR / "config.myopic.yaml").resolve()),
        str((ROOT_DIR / "config.learning.yaml").resolve()),
    ]
    configfiles.extend(args.configfiles)
    tasks = build_tasks(configfiles, args.scenario_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submit_root = Path(args.submit_root).resolve()
    submit_dir = submit_root / f"{args.scenario_name}_{timestamp}"
    submit_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = submit_dir / "task_manifest.json"
    manifest_path.write_text(json.dumps(tasks, indent=2, sort_keys=True), encoding="utf-8")

    logs_dir = submit_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    worker_script = str((SCRIPT_DIR / "run_learning_cluster_array_task.sh").resolve())
    grid_run_cmd = build_grid_run_cmd(args, manifest_path, len(tasks), worker_script)

    if args.print_only:
        print("TASK_MANIFEST", manifest_path)
        print("ARRAY_SIZE", len(tasks))
        print("LOG_DIR", logs_dir)
        print("GRID_RUN_CMD", " ".join(shlex.quote(part) for part in grid_run_cmd))
        return

    subprocess.run(grid_run_cmd, check=True, cwd=logs_dir)
    print("TASK_MANIFEST", manifest_path)
    print("ARRAY_SIZE", len(tasks))
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
        str((SCRIPT_DIR / "run_learning_cluster_job.sh").resolve()),
        task["model"],
        str(task["seed"]),
        str(Path(os.path.expandvars(args.job_root)).resolve()),
    ]
    env = os.environ.copy()
    env["LEARNING_SCENARIO_NAME"] = str(task["scenario_name"])
    subprocess.run(cmd, check=True, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--manifest")
    parser.add_argument("--job-root", default=str(DEFAULT_JOB_ROOT))
    parser.add_argument("--submit-root", default="cluster_submissions")
    parser.add_argument("--scenario-name", default="learning_mc")
    parser.add_argument("--job-name", default="learnmc")
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV)
    parser.add_argument("--run-mode", choices=["branch", "full"], default="branch")
    parser.add_argument("--grid-mem", default="12G")
    parser.add_argument("--grid-ncpus", default="36")
    parser.add_argument("--grid-submit", default="batch")
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("configfiles", nargs="*")
    args = parser.parse_args()

    if args.worker:
        if not args.manifest:
            raise ValueError("--manifest is required in worker mode")
        run_worker(args)
        return

    submit_array(args)


if __name__ == "__main__":
    main()
