#!/usr/bin/env python3
"""Submit or execute the stochastic learning sensitivity ensemble as an SGE array."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shlex
import shutil
import subprocess
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import yaml

from bootstrap_state_store import (
    resolve_results_sector_dir,
    resolve_scenario_sector_name,
    restore_bootstrap_state,
    validate_state_source,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
DEFAULT_CONFIG = ROOT_DIR / "configs" / "sensitivity_ensemble.yaml"
DEFAULT_JOB_ROOT = ROOT_DIR / "cluster_workdirs"
DEFAULT_SUBMIT_ROOT = ROOT_DIR / "cluster_submissions"
DEFAULT_CONDA_ENV = "/shared/share_cki25/envs/sh-pypsa-earth-main"
DEFAULT_GRID_MEM = "90G"
DEFAULT_GRID_NCPUS = "12"
DEFAULT_GRID_SUBMIT = "batch"
DEFAULT_GRID_ARRAY_CONCURRENCY = 200
DEFAULT_BASE_CONFIGFILES = [
    "config.myopic.yaml",
    "config.learning.yaml",
]
SUPPORTED_MODELS = {
    "shared_state_bayesian_regime_wright",
    "way_fixed_rho_benchmark",
    "correlated_geometric_random_walk",
}


def deep_update(base: dict, update: dict | None) -> dict:
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def sanitize_token(raw: str, fallback: str = "default") -> str:
    token = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(raw or ""))
    token = token.strip("._-")
    return token or fallback


def stable_uint32(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def load_ensemble_config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "ensemble" not in payload:
        raise ValueError(f"Missing 'ensemble' block in {path}")
    if "baseline" not in payload:
        raise ValueError(f"Missing 'baseline' block in {path}")
    if "draw_budgets" not in payload:
        raise ValueError(f"Missing 'draw_budgets' block in {path}")
    return payload


def parse_csv_list(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    values = [entry.strip() for entry in str(raw).split(",") if entry.strip()]
    return values or None


def resolve_common_random_numbers(args: argparse.Namespace, config: dict) -> bool:
    if args.common_random_numbers is not None:
        return bool(args.common_random_numbers)
    ensemble = config.get("ensemble", {}) or {}
    return bool(ensemble.get("use_common_random_numbers", True))


def resolve_base_seed(args: argparse.Namespace, config: dict) -> int:
    if args.seed is not None:
        return int(args.seed)
    ensemble = config.get("ensemble", {}) or {}
    return int(ensemble.get("base_seed", 0))


def draw_count_for_run(run: dict, args: argparse.Namespace, config: dict) -> int:
    budgets = config.get("draw_budgets", {}) or {}
    if args.test_draws_per_run is not None:
        return int(args.test_draws_per_run)
    if args.n_draws is not None:
        return int(args.n_draws)
    if args.budget and args.budget != "custom":
        if args.budget not in budgets:
            raise ValueError(f"Unknown draw budget {args.budget!r}; available={sorted(budgets)}")
        return int(budgets[args.budget])
    budget_name = str(run.get("budget") or (config.get("ensemble", {}) or {}).get("default_budget", "screening"))
    if budget_name not in budgets:
        raise ValueError(f"Run {run['run_id']} references unknown budget {budget_name!r}")
    return int(budgets[budget_name])


def seed_list_for_run(
    *,
    run_id: str,
    n_draws: int,
    base_seed: int,
    seed_mode: str,
    seed_upper_bound: int,
    common_random_numbers: bool,
) -> list[int]:
    if n_draws <= 0:
        raise ValueError("draw count must be positive")
    if seed_mode == "sequential":
        if common_random_numbers:
            return list(range(n_draws))
        return [stable_uint32(base_seed, run_id, draw_id) for draw_id in range(n_draws)]
    if seed_mode == "random":
        if n_draws > seed_upper_bound:
            raise ValueError("n_draws cannot exceed seed_upper_bound")
        rng_seed = base_seed if common_random_numbers else stable_uint32(base_seed, run_id)
        rng = random.Random(rng_seed)
        return rng.sample(range(seed_upper_bound), n_draws)
    raise ValueError(f"Unsupported ensemble seed_mode: {seed_mode!r}")


def seed_token(model: str, seed: int) -> str:
    if model == "legacy_curve":
        return "deterministic"
    return f"s{int(seed):04d}"


def normalize_run_entry(raw_run: dict, baseline: dict, config: dict) -> dict:
    baseline_run = deepcopy(baseline)
    run = deepcopy(raw_run)
    run_id = str(run.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("Every sensitivity run must have run_id")

    model = str(run.get("model") or run.get("learning_model") or baseline_run.get("model") or baseline_run.get("learning_model") or "").strip()
    if not model:
        raise ValueError(f"Run {run_id} does not resolve a learning model")
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Run {run_id} uses unsupported stochastic model {model!r}")

    scenario_prefix = str((config.get("ensemble", {}) or {}).get("scenario_prefix", "sens")).strip()
    scenario_name = str(run.get("scenario_name") or f"{scenario_prefix}_{run_id}").strip()
    sector_name = str(run.get("sector_name") or resolve_scenario_sector_name(scenario_name)).strip()

    overlay = deepcopy(baseline_run.get("config_overrides", {}) or {})
    deep_update(overlay, run.get("config_overrides", {}) or {})
    deep_update(
        overlay,
        {
            "run": {
                "allow_scenario_failure": False,
                "sector_name": sector_name,
            },
            "scenario": {
                "learning_rate": ["base"],
                "learning_model": [model],
            },
            "permstore": {
                "path": "permstorage",
            },
            "learning": {
                "enabled": True,
                "engine": "stochastic_forecast",
                "selected_model": model,
                "monte_carlo": {
                    "enable": False,
                },
            },
        },
    )

    return {
        "run_id": run_id,
        "group": str(run.get("group") or "sensitivity"),
        "description": str(run.get("description") or ""),
        "budget": run.get("budget") or baseline_run.get("budget"),
        "model": model,
        "scenario_name": scenario_name,
        "sector_name": sector_name,
        "run_mode": str(run.get("run_mode") or baseline_run.get("run_mode") or "branch"),
        "bootstrap_group": str(run.get("bootstrap_group") or baseline_run.get("bootstrap_group") or "AB"),
        "enabled": bool(run.get("enabled", True)),
        "disabled_reason": str(run.get("disabled_reason") or ""),
        "config_overrides": overlay,
    }


def selected_runs(config: dict, args: argparse.Namespace) -> list[dict]:
    baseline = config["baseline"]
    requested = parse_csv_list(args.run_ids)
    include_baseline = bool((config.get("ensemble", {}) or {}).get("submit_baseline", False))
    if requested and str(baseline.get("run_id", "BASE")) in requested:
        include_baseline = True

    runs = []
    if include_baseline:
        runs.append(normalize_run_entry(baseline, baseline, config))
    runs.extend(normalize_run_entry(run, baseline, config) for run in (config.get("runs", []) or []))

    if requested:
        by_id = {run["run_id"]: run for run in runs}
        missing = [run_id for run_id in requested if run_id not in by_id]
        if missing:
            raise ValueError(f"Unknown run IDs requested: {missing}")
        runs = [by_id[run_id] for run_id in requested]

    disabled = [run for run in runs if not run["enabled"]]
    if disabled and requested:
        details = ", ".join(
            f"{run['run_id']} ({run['disabled_reason'] or 'disabled'})" for run in disabled
        )
        raise ValueError(f"Requested disabled sensitivity runs: {details}")
    runs = [run for run in runs if run["enabled"]]

    if args.max_runs is not None:
        runs = runs[: int(args.max_runs)]
    if not runs:
        raise ValueError("No enabled sensitivity runs selected")
    return runs


def build_tasks(config: dict, args: argparse.Namespace) -> list[dict]:
    runs = selected_runs(config, args)
    ensemble = config.get("ensemble", {}) or {}
    seed_mode = str(ensemble.get("seed_mode", "sequential"))
    seed_upper_bound = int(ensemble.get("seed_upper_bound", 1000000000) or 1000000000)
    base_seed = resolve_base_seed(args, config)
    common_random_numbers = resolve_common_random_numbers(args, config)
    tasks: list[dict] = []

    for run in runs:
        n_draws = draw_count_for_run(run, args, config)
        seeds = seed_list_for_run(
            run_id=run["run_id"],
            n_draws=n_draws,
            base_seed=base_seed,
            seed_mode=seed_mode,
            seed_upper_bound=seed_upper_bound,
            common_random_numbers=common_random_numbers,
        )
        for draw_index, seed in enumerate(seeds):
            overlay = deepcopy(run["config_overrides"])
            deep_update(
                overlay,
                {
                    "learning": {
                        "execution_mode": run["run_mode"],
                        "seed": int(seed),
                    }
                },
            )
            tasks.append(
                {
                    "ensemble_name": str(ensemble.get("name", "sensitivity_ensemble")),
                    "run_id": run["run_id"],
                    "group": run["group"],
                    "description": run["description"],
                    "draw_id": int(draw_index),
                    "learning_seed": int(seed),
                    "model": run["model"],
                    "scenario_name": run["scenario_name"],
                    "sector_name": run["sector_name"],
                    "run_mode": run["run_mode"],
                    "bootstrap_group": run["bootstrap_group"],
                    "budget": run["budget"],
                    "n_draws": int(n_draws),
                    "common_random_numbers": common_random_numbers,
                    "config_overrides": overlay,
                    "status": "pending",
                }
            )
    return tasks


def compact_complete(task: dict) -> bool:
    results_dir = ROOT_DIR / "results" / Path(str(task["sector_name"]))
    compact_dir = (
        results_dir
        / "learning-compact"
        / str(task["scenario_name"])
        / str(task["model"])
        / f"seed_{seed_token(str(task['model']), int(task['learning_seed']))}"
    )
    return bool(list(compact_dir.glob("raw_cleanup_complete*.txt")) or list(compact_dir.glob("compact_complete*.json")))


def filter_resume_tasks(tasks: list[dict], resume: bool) -> list[dict]:
    if not resume:
        return tasks
    filtered = []
    for task in tasks:
        if compact_complete(task):
            skipped = deepcopy(task)
            skipped["status"] = "skipped"
            continue
        filtered.append(task)
    return filtered


def resolve_task_sector_names(tasks: list[dict]) -> list[str]:
    return sorted({str(task["sector_name"]) for task in tasks})


def build_grid_run_cmd(args: argparse.Namespace, manifest_path: Path, task_count: int) -> list[str]:
    grid_array = f"1-{task_count}"
    if task_count > DEFAULT_GRID_ARRAY_CONCURRENCY:
        grid_array = f"{grid_array}/{DEFAULT_GRID_ARRAY_CONCURRENCY}"
    worker_script = str((SCRIPT_DIR / "run_learning_sensitivity_ensemble_array_task.sh").resolve())
    return [
        "grid_run",
        f"--grid_mem={args.grid_mem}",
        f"--grid_ncpus={args.grid_ncpus}",
        f"--grid_submit={args.grid_submit}",
        f"--grid_array={grid_array}",
        worker_script,
        str(manifest_path),
        f"SENSITIVITY_JOB_ROOT={Path(os.path.expandvars(args.job_root)).resolve()}",
        f"SENSITIVITY_CONDA_ENV={args.conda_env}",
    ] + (
        ["SENSITIVITY_DRY_RUN=1"] if args.dry_run else []
    ) + (
        ["SENSITIVITY_OVERWRITE=1"] if args.overwrite else []
    )


def tasks_needing_bootstrap_restore(tasks: list[dict]) -> list[dict]:
    """Only branch-mode tasks should receive a copied shared bootstrap.

    Full-mode runs intentionally rebuild their own demand/opts-specific bootstrap.
    This matters for demand sensitivities such as DF/NZ and policy sensitivities
    such as Co2L, where copying an AB bootstrap would be the wrong prerequisite.
    """

    return [task for task in tasks if str(task.get("run_mode", "branch")) == "branch"]


def write_csv_manifest(path: Path, tasks: list[dict]) -> None:
    columns = [
        "ensemble_name",
        "run_id",
        "group",
        "draw_id",
        "learning_seed",
        "model",
        "scenario_name",
        "sector_name",
        "run_mode",
        "bootstrap_group",
        "status",
    ]
    rows = [",".join(columns)]
    for task in tasks:
        rows.append(",".join(str(task.get(column, "")) for column in columns))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_submission_metadata(
    submit_dir: Path,
    args: argparse.Namespace,
    config_path: Path,
    tasks: list[dict],
    selected_task_count_before_resume: int,
) -> Path:
    metadata = {
        "submission_kind": "stochastic_learning_sensitivity_ensemble_array",
        "ensemble_config": str(config_path.resolve()),
        "task_count": len(tasks),
        "selected_task_count_before_resume": selected_task_count_before_resume,
        "run_ids": sorted({task["run_id"] for task in tasks}),
        "resolved_sector_names": resolve_task_sector_names(tasks),
        "bootstrap_groups": sorted({str(task.get("bootstrap_group", "AB")) for task in tasks}),
        "common_random_numbers": bool(tasks[0]["common_random_numbers"]) if tasks else None,
        "job_root": str(Path(os.path.expandvars(args.job_root)).resolve()),
        "conda_env": args.conda_env,
        "grid_mem": args.grid_mem,
        "grid_ncpus": args.grid_ncpus,
        "grid_submit": args.grid_submit,
        "resume": bool(args.resume),
        "overwrite": bool(args.overwrite),
        "dry_run": bool(args.dry_run),
        "assume_bootstrap_state_present": bool(args.assume_bootstrap_state_present),
    }
    path = submit_dir / "submission_metadata.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return path


def parse_bootstrap_source_args(raw_sources: list[str] | None) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for raw in raw_sources or []:
        text = str(raw).strip()
        if not text:
            continue
        if "=" in text:
            group, path = text.split("=", 1)
            group = group.strip()
            if not group:
                raise ValueError(f"Invalid bootstrap source {raw!r}: missing group before '='")
        else:
            group, path = "__default__", text
        sources[group] = Path(os.path.expandvars(path.strip())).resolve()
    return sources


def resolve_bootstrap_sources_for_tasks(
    tasks: list[dict],
    raw_sources: list[str] | None,
) -> dict[str, Path]:
    sources = parse_bootstrap_source_args(raw_sources)
    groups = sorted({str(task.get("bootstrap_group", "AB")) for task in tasks_needing_bootstrap_restore(tasks)})
    if "__default__" in sources:
        if len(groups) > 1:
            raise ValueError(
                "An unqualified --bootstrap-state-source can only be used when one bootstrap_group "
                f"is selected. Selected groups are {groups}. Use GROUP=/path entries instead."
            )
        if groups:
            sources[groups[0]] = sources.pop("__default__")
    else:
        sources.pop("__default__", None)
    return {group: sources[group] for group in groups if group in sources}


def missing_bootstrap_groups(tasks: list[dict], resolved_sources: dict[str, Path]) -> list[str]:
    groups = sorted({str(task.get("bootstrap_group", "AB")) for task in tasks_needing_bootstrap_restore(tasks)})
    return [group for group in groups if group not in resolved_sources]


def restore_bootstrap_state_for_runs(tasks: list[dict], bootstrap_sources: dict[str, Path]) -> None:
    tasks = tasks_needing_bootstrap_restore(tasks)
    if not tasks:
        print("BOOTSTRAP_STATE_RESTORED none_branch_mode_tasks")
        return
    for group, bootstrap_source in sorted(bootstrap_sources.items()):
        group_tasks = [task for task in tasks if str(task.get("bootstrap_group", "AB")) == group]
        if not group_tasks:
            continue
        required_counts = validate_state_source(bootstrap_source)
        print("BOOTSTRAP_STATE_SOURCE", group, str(bootstrap_source.resolve()))
        print("BOOTSTRAP_STATE_VALIDATED", group, json.dumps(required_counts, sort_keys=True))
        for sector_name in resolve_task_sector_names(group_tasks):
            target_dir = resolve_results_sector_dir(ROOT_DIR, sector_name)
            restore_summary = restore_bootstrap_state(bootstrap_source, target_dir)
            print("BOOTSTRAP_STATE_RESTORED", group, sector_name, json.dumps(restore_summary, sort_keys=True))


def submit_array(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    config = load_ensemble_config(config_path)
    all_selected_tasks = build_tasks(config, args)
    tasks = filter_resume_tasks(all_selected_tasks, args.resume)
    if not tasks:
        print("SENSITIVITY_STATUS no_tasks_to_submit")
        print("SELECTED_TASKS_BEFORE_RESUME", len(all_selected_tasks))
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ensemble_name = sanitize_token(str((config.get("ensemble", {}) or {}).get("name", "sensitivity_ensemble")))
    submit_root = Path(args.submit_root).resolve()
    submit_dir = submit_root / f"{ensemble_name}_{timestamp}"
    submit_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = submit_dir / "task_manifest.json"
    manifest_path.write_text(json.dumps(tasks, indent=2, sort_keys=True), encoding="utf-8")
    write_csv_manifest(submit_dir / "manifest.csv", tasks)
    metadata_path = write_submission_metadata(
        submit_dir,
        args,
        config_path,
        tasks,
        selected_task_count_before_resume=len(all_selected_tasks),
    )
    logs_dir = submit_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    grid_run_cmd = build_grid_run_cmd(args, manifest_path, len(tasks))

    bootstrap_sources = resolve_bootstrap_sources_for_tasks(tasks, args.bootstrap_state_source)
    missing_groups = missing_bootstrap_groups(tasks, bootstrap_sources)
    if missing_groups:
        print("BOOTSTRAP_STATE_MISSING_GROUPS", ",".join(missing_groups))
    if missing_groups and not args.print_only and not args.assume_bootstrap_state_present:
        raise ValueError(
            "Missing bootstrap sources for selected bootstrap groups: "
            f"{missing_groups}. Pass --bootstrap-state-source GROUP=/path for each group, "
            "or pass --assume-bootstrap-state-present if these sector prerequisites are already staged."
        )
    if bootstrap_sources:
        bootstrap_tasks = tasks_needing_bootstrap_restore(tasks)
        if args.print_only:
            for group, bootstrap_source in sorted(bootstrap_sources.items()):
                required_counts = validate_state_source(bootstrap_source)
                print("BOOTSTRAP_STATE_SOURCE", group, str(bootstrap_source))
                print("BOOTSTRAP_STATE_VALIDATED", group, json.dumps(required_counts, sort_keys=True))
                group_tasks = [task for task in bootstrap_tasks if str(task.get("bootstrap_group", "AB")) == group]
                for sector_name in resolve_task_sector_names(group_tasks):
                    print("BOOTSTRAP_STATE_TARGET", group, resolve_results_sector_dir(ROOT_DIR, sector_name))
        else:
            if missing_groups:
                raise ValueError(
                    "Missing bootstrap sources for selected bootstrap groups: "
                    f"{missing_groups}. Pass --bootstrap-state-source GROUP=/path for each group."
                )
            restore_bootstrap_state_for_runs(tasks, bootstrap_sources)

    if args.print_only:
        print("TASK_MANIFEST", manifest_path)
        print("CSV_MANIFEST", submit_dir / "manifest.csv")
        print("SUBMISSION_METADATA", metadata_path)
        print("ARRAY_SIZE", len(tasks))
        print("SELECTED_TASKS_BEFORE_RESUME", len(all_selected_tasks))
        print("RUN_IDS", ",".join(sorted({task["run_id"] for task in tasks})))
        print("RESOLVED_SECTOR_NAMES", json.dumps(resolve_task_sector_names(tasks), sort_keys=True))
        print("LOG_DIR", logs_dir)
        print("GRID_RUN_CMD", " ".join(shlex.quote(part) for part in grid_run_cmd))
        return

    subprocess.run(grid_run_cmd, check=True, cwd=logs_dir)
    print("TASK_MANIFEST", manifest_path)
    print("CSV_MANIFEST", submit_dir / "manifest.csv")
    print("SUBMISSION_METADATA", metadata_path)
    print("ARRAY_SIZE", len(tasks))
    print("LOG_DIR", logs_dir)


def _stage_job_dir(task: dict, job_root: Path) -> Path:
    job_tag = (
        f"sensitivity_{sanitize_token(task['run_id'])}_"
        f"{sanitize_token(task['model'])}_s{int(task['learning_seed']):04d}"
    )
    job_root.mkdir(parents=True, exist_ok=True)
    job_dir = job_root.resolve() / f"energymod_{job_tag}"
    job_dir.mkdir(parents=True, exist_ok=True)
    for child in sorted(ROOT_DIR.iterdir(), key=lambda p: p.name):
        if child.name in {".git", ".snakemake", "cluster_workdirs"}:
            continue
        target = job_dir / child.name
        if target.is_symlink():
            continue
        if target.exists():
            raise RuntimeError(f"Existing non-symlink path blocks staged job entry: {target}")
        target.symlink_to(child)
    return job_dir


def _write_overlay(task: dict, output_dir: Path) -> Path:
    fd, raw_path = tempfile.mkstemp(
        prefix=f"energymod_sensitivity_{sanitize_token(task['run_id'])}_",
        suffix=".yaml",
        dir=str(output_dir),
        text=True,
    )
    path = Path(raw_path)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        yaml.safe_dump(task["config_overrides"], handle, sort_keys=False)
    return path


def _compact_bundle_dir(task: dict, job_dir: Path) -> Path:
    return (
        job_dir
        / "results"
        / Path(str(task["sector_name"]))
        / "learning-compact"
        / str(task["scenario_name"])
        / str(task["model"])
        / f"seed_{seed_token(str(task['model']), int(task['learning_seed']))}"
    )


def _write_draw_complete(task: dict, job_dir: Path) -> None:
    bundle_dir = _compact_bundle_dir(task, job_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "ensemble_name": task["ensemble_name"],
        "run_id": task["run_id"],
        "group": task["group"],
        "draw_id": int(task["draw_id"]),
        "learning_seed": int(task["learning_seed"]),
        "model": task["model"],
        "scenario_name": task["scenario_name"],
        "sector_name": task["sector_name"],
        "completed": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_status": "snakemake_complete",
    }
    (bundle_dir / "draw_complete.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def build_snakemake_cmd(
    task: dict,
    overlay_path: Path,
    snakemake_jobs: str,
    dry_run: bool = False,
) -> list[str]:
    run_mode = str(task.get("run_mode") or "branch")
    if run_mode == "full":
        target = "solve_sector_networks_myopic"
    elif run_mode == "branch":
        target = "solve_sector_networks_myopic_stochastic_branch"
    else:
        raise ValueError(f"Unsupported run_mode for task {task['run_id']}: {run_mode!r}")
    cmd = [
        "snakemake",
        f"-j{snakemake_jobs}",
        target,
        "--configfile",
        *DEFAULT_BASE_CONFIGFILES,
        str(overlay_path),
        "--rerun-trigger",
        "mtime",
    ]
    if dry_run:
        cmd.append("-n")
    return cmd


def run_worker(args: argparse.Namespace) -> None:
    sge_task_id = os.environ.get("SGE_TASK_ID")
    if sge_task_id is None:
        raise RuntimeError("SGE_TASK_ID environment variable not found")

    tasks = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    task_id = int(sge_task_id)
    task_index = task_id - 1
    if task_index < 0 or task_index >= len(tasks):
        raise RuntimeError(f"Task index {task_id} outside manifest range 1-{len(tasks)}")

    task = tasks[task_index]
    job_root = Path(os.path.expandvars(args.job_root)).resolve()
    job_dir = _stage_job_dir(task, job_root)
    overlay_path = _write_overlay(task, job_dir)
    if args.overwrite:
        shutil.rmtree(_compact_bundle_dir(task, job_dir), ignore_errors=True)

    snakemake_jobs = os.environ.get("JOBS") or os.environ.get("NSLOTS") or "4"
    cmd = build_snakemake_cmd(task, overlay_path, snakemake_jobs, dry_run=args.dry_run)
    env = os.environ.copy()
    env["LEARNING_SCENARIO_NAME"] = str(task["scenario_name"])
    env["XDG_CACHE_HOME"] = env.get("XDG_CACHE_HOME", "/tmp/energymod_sensitivity_cache")
    env["MPLCONFIGDIR"] = env.get("MPLCONFIGDIR", "/tmp/energymod_sensitivity_matplotlib")
    if "Library/Caches" in env["XDG_CACHE_HOME"]:
        env["XDG_CACHE_HOME"] = "/tmp/energymod_sensitivity_cache"
    if args.dry_run and "Library/Mobile Documents" in str(ROOT_DIR):
        env["HOME"] = "/tmp/energymod_sensitivity_home"
        Path(env["HOME"]).mkdir(parents=True, exist_ok=True)

    print("Running sensitivity ensemble task:")
    print(f"  ensemble={task['ensemble_name']}")
    print(f"  run_id={task['run_id']}")
    print(f"  draw_id={task['draw_id']}")
    print(f"  learning_seed={task['learning_seed']}")
    print(f"  model={task['model']}")
    print(f"  scenario_name={task['scenario_name']}")
    print(f"  sector_name={task['sector_name']}")
    print(f"  run_mode={task['run_mode']}")
    print(f"  job_dir={job_dir}")
    print(f"  overlay={overlay_path}")

    try:
        subprocess.run(cmd, check=True, cwd=job_dir, env=env)
        if not args.dry_run:
            _write_draw_complete(task, job_dir)
    finally:
        overlay_path.unlink(missing_ok=True)

    if not args.dry_run:
        shutil.rmtree(job_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--manifest")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--budget", help="Override all run budgets using any key in draw_budgets, or 'custom' with --n-draws")
    parser.add_argument("--n-draws", type=int)
    parser.add_argument(
        "--test-draws-per-run",
        type=int,
        metavar="N",
        help=(
            "Smoke-test mode: run N draw-matched tasks for every selected sensitivity run. "
            "For example, --test-draws-per-run 2 gives two tasks per run."
        ),
    )
    parser.add_argument("--run-ids")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    crn = parser.add_mutually_exclusive_group()
    crn.add_argument("--common-random-numbers", dest="common_random_numbers", action="store_true")
    crn.add_argument("--no-common-random-numbers", dest="common_random_numbers", action="store_false")
    parser.set_defaults(common_random_numbers=None)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--job-root", default=str(DEFAULT_JOB_ROOT))
    parser.add_argument("--submit-root", default=str(DEFAULT_SUBMIT_ROOT))
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV)
    parser.add_argument("--grid-mem", default=DEFAULT_GRID_MEM)
    parser.add_argument("--grid-ncpus", default=DEFAULT_GRID_NCPUS)
    parser.add_argument("--grid-submit", default=DEFAULT_GRID_SUBMIT)
    parser.add_argument(
        "--bootstrap-state-source",
        action="append",
        help=(
            "Bootstrap/prereq state path. Use GROUP=/path for mixed suites "
            "(for example AB=/path DF=/path NZ=/path CO2CAP=/path)."
        ),
    )
    parser.add_argument(
        "--assume-bootstrap-state-present",
        action="store_true",
        help=(
            "Submit without copying bootstrap state. Use only when every selected branch "
            "sector already has its demand/opts-specific shared prerequisites staged."
        ),
    )
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Submission mode: print the resolved array command; worker mode: pass -n to Snakemake",
    )
    args = parser.parse_args()

    if args.worker:
        if not args.manifest:
            raise ValueError("--manifest is required in worker mode")
        run_worker(args)
        return

    if args.dry_run:
        args.print_only = True
    submit_array(args)


if __name__ == "__main__":
    main()
