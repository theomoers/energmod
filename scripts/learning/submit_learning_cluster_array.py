#!/usr/bin/env python3
"""Submit or execute the supported SGE array workflow for stochastic learning runs."""

import argparse
import json
import os
import random
import shlex
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
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
            raise ValueError("learning.monte_carlo.draws cannot exceed learning.monte_carlo.seed_upper_bound")
        rng = random.Random(random_seed)
        return rng.sample(range(seed_upper_bound), draws)

    raise ValueError(f"Unsupported learning.monte_carlo.seed_mode: {seed_mode}")


def build_tasks(configfiles, scenario_name):
    cfg = load_merged_config(configfiles)
    learning_cfg = cfg.get("learning", {}) or {}
    mc_cfg = learning_cfg.get("monte_carlo", {}) or {}
    if not mc_cfg.get("enable", False):
        raise ValueError("learning.monte_carlo.enable must be true for array submission")

    learning_rates = [str(x) for x in ((cfg.get("scenario", {}) or {}).get("learning_rate", []) or [])]
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


def submit_array(args):
    configfiles = [str((ROOT_DIR / "config.myopic.yaml").resolve()), str((ROOT_DIR / "config.learning.yaml").resolve())]
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
    env_args = ",".join(
        [
            f"LEARNING_ARRAY_MANIFEST={manifest_path}",
            f"LEARNING_JOB_ROOT={os.path.expandvars(args.job_root)}",
            f"LEARNING_CONDA_ENV={args.conda_env}",
        ]
    )

    qsub_cmd = [
        "qsub",
        "-cwd",
        "-V",
        "-v",
        env_args,
        "-t",
        f"1-{len(tasks)}",
        "-N",
        args.job_name,
        "-o",
        str(logs_dir / "array.out"),
        "-e",
        str(logs_dir / "array.err"),
        worker_script,
    ]

    if args.print_only:
        print("TASK_MANIFEST", manifest_path)
        print("ARRAY_SIZE", len(tasks))
        print("QSUB_CMD", " ".join(qsub_cmd))
        return

    subprocess.run(qsub_cmd, check=True)
    print("TASK_MANIFEST", manifest_path)
    print("ARRAY_SIZE", len(tasks))


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
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--manifest")
    parser.add_argument("--job-root", default="/scratch/$USER")
    parser.add_argument("--submit-root", default="cluster_submissions")
    parser.add_argument("--scenario-name", default="learning_mc")
    parser.add_argument("--job-name", default="learnmc")
    parser.add_argument("--conda-env", default="pypsa-earth")
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
