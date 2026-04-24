#!/usr/bin/env python3
"""Submit or execute a 2025 phi2/phi3 deployment-wedge calibration array."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

from bootstrap_state_store import (
    resolve_results_sector_dir,
    restore_bootstrap_state,
    validate_state_source,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
DEFAULT_JOB_ROOT = ROOT_DIR / "cluster_workdirs"
DEFAULT_SUBMIT_ROOT = ROOT_DIR / "cluster_submissions"
DEFAULT_RESULTS_ROOT = ROOT_DIR / "results" / "Global_200" / "phi_calibration"
DEFAULT_BOOTSTRAP_STATE_SOURCE = ROOT_DIR / "results" / "Global_200"
DEFAULT_WORKING_SECTOR_NAME = "Global_200/phi_calibration/_bootstrap_state"
DEFAULT_CONDA_ENV = "/shared/share_cki25/envs/sh-pypsa-earth-main"
DEFAULT_GRID_MEM = "90G"
DEFAULT_GRID_NCPUS = "12"
DEFAULT_GRID_SUBMIT = "batch"
DEFAULT_GRID_ARRAY_CONCURRENCY = 100
DEFAULT_MODEL = "shared_state_bayesian_regime_wright"
DEFAULT_SEED = 0
FIXED_B1_MULTIPLIER = 0.8
FIXED_B2_MULTIPLIER = 1.5


def _format_phi(value: float) -> str:
    return f"{value:.2f}"


def _tokenize_float(value: float) -> str:
    return _format_phi(value).replace(".", "p")


def _sanitize_token(raw: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(raw))
    token = token.strip("._-")
    return token or "default"


def _float_grid(start: float, stop: float, count: int) -> list[float]:
    if count <= 1:
        return [round(float(start), 10)]
    step = (float(stop) - float(start)) / float(count - 1)
    return [round(float(start) + step * idx, 10) for idx in range(count)]


def expand_and_resolve(pathlike: str | os.PathLike) -> Path:
    return Path(os.path.expandvars(str(pathlike))).resolve()


def resolve_task_sector_names(tasks: list[dict]) -> list[str]:
    return sorted({str(task["sector_name"]) for task in tasks})


def resolve_working_sector_name() -> str:
    return DEFAULT_WORKING_SECTOR_NAME


def build_tasks(
    *,
    phi2_min: float = 0.0,
    phi2_max: float = 0.45,
    phi2_count: int = 10,
    phi3_min: float = 0.10,
    phi3_max: float = 1.00,
    phi3_count: int = 10,
    model: str = DEFAULT_MODEL,
    seed: int = DEFAULT_SEED,
) -> list[dict]:
    tasks = []
    for phi2 in _float_grid(phi2_min, phi2_max, phi2_count):
        for phi3 in _float_grid(phi3_min, phi3_max, phi3_count):
            if phi3 < phi2:
                continue
            name = f"phi2_{_tokenize_float(phi2)}_phi3_{_tokenize_float(phi3)}"
            tasks.append(
                {
                    "name": name,
                    "sector_name": f"Global_200/phi_calibration/{name}",
                    "model": model,
                    "seed": int(seed),
                    "phi2_pct_capex": float(_format_phi(phi2)),
                    "phi3_pct_capex": float(_format_phi(phi3)),
                    "b1_multiplier": FIXED_B1_MULTIPLIER,
                    "b2_multiplier": FIXED_B2_MULTIPLIER,
                    "planning_horizons": [2020, 2025],
                }
            )
    return tasks


def build_grid_run_cmd(args: argparse.Namespace, manifest_path: Path, task_count: int) -> list[str]:
    grid_array = f"1-{task_count}"
    if task_count > DEFAULT_GRID_ARRAY_CONCURRENCY:
        grid_array = f"{grid_array}/{DEFAULT_GRID_ARRAY_CONCURRENCY}"
    worker_script = str((SCRIPT_DIR / "run_learning_phi_calibration_array_task.sh").resolve())
    return [
        "grid_run",
        f"--grid_mem={args.grid_mem}",
        f"--grid_ncpus={args.grid_ncpus}",
        f"--grid_submit={args.grid_submit}",
        f"--grid_array={grid_array}",
        worker_script,
        str(manifest_path),
        f"PHI_CALIBRATION_JOB_ROOT={Path(os.path.expandvars(args.job_root)).resolve()}",
        f"PHI_CALIBRATION_CONDA_ENV={args.conda_env}",
    ]


def write_submission_metadata(
    submit_dir: Path,
    args: argparse.Namespace,
    manifest_path: Path,
    tasks: list[dict],
) -> Path:
    metadata = {
        "submission_kind": "phi2_phi3_2025_calibration_array",
        "task_manifest": str(manifest_path.resolve()),
        "task_count": len(tasks),
        "results_root": str(Path(args.results_root).resolve()),
        "job_root": str(expand_and_resolve(args.job_root)),
        "bootstrap_state_source": str(expand_and_resolve(args.bootstrap_state_source)),
        "working_sector_name": resolve_working_sector_name(),
        "resolved_sector_names": resolve_task_sector_names(tasks),
        "conda_env": args.conda_env,
        "grid_mem": args.grid_mem,
        "grid_ncpus": args.grid_ncpus,
        "grid_submit": args.grid_submit,
        "grid": {
            "phi2_min": args.phi2_min,
            "phi2_max": args.phi2_max,
            "phi2_count": args.phi2_count,
            "phi3_min": args.phi3_min,
            "phi3_max": args.phi3_max,
            "phi3_count": args.phi3_count,
            "constraint": "phi3_pct_capex >= phi2_pct_capex",
        },
        "fixed_parameters": {
            "b1_multiplier": FIXED_B1_MULTIPLIER,
            "b2_multiplier": FIXED_B2_MULTIPLIER,
            "apply_from_year": 2025,
            "planning_horizons": [2020, 2025],
            "model": args.model,
            "seed": args.seed,
        },
        "tasks": tasks,
    }
    path = submit_dir / "submission_metadata.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return path


def restore_phi_calibration_state(tasks: list[dict], bootstrap_source: Path) -> None:
    """Mirror the regular stochastic-array copied-sector setup once per submission."""
    required_counts = validate_state_source(bootstrap_source)
    working_sector_name = resolve_working_sector_name()
    target_dir = resolve_results_sector_dir(ROOT_DIR, working_sector_name)
    print("BOOTSTRAP_STATE_SOURCE", str(bootstrap_source.resolve()))
    print("BOOTSTRAP_STATE_VALIDATED", json.dumps(required_counts, sort_keys=True))
    restore_summary = restore_bootstrap_state(
        bootstrap_source,
        target_dir,
        exclude_relative_prefixes=("phi_calibration",),
        hardlink_first=False,
    )
    mark_tree_current(target_dir)
    print("BOOTSTRAP_STATE_TARGET", target_dir)
    print("BOOTSTRAP_STATE_RESTORED", json.dumps(restore_summary, sort_keys=True))


def mark_tree_current(path: Path) -> None:
    """Treat a restored sector as the fresh baseline for this calibration run."""
    current = time.time()
    timestamp = (current, current)
    for entry in path.rglob("*"):
        if entry.is_symlink():
            continue
        if entry.is_file():
            os.utime(entry, timestamp)


def submit_array(args: argparse.Namespace) -> None:
    tasks = build_tasks(
        phi2_min=args.phi2_min,
        phi2_max=args.phi2_max,
        phi2_count=args.phi2_count,
        phi3_min=args.phi3_min,
        phi3_max=args.phi3_max,
        phi3_count=args.phi3_count,
        model=args.model,
        seed=args.seed,
    )
    if not tasks:
        raise ValueError("Phi calibration grid produced no valid tasks")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submit_root = Path(args.submit_root).resolve()
    submit_dir = submit_root / f"phi_calibration_{timestamp}"
    submit_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = submit_dir / "task_manifest.json"
    manifest_path.write_text(json.dumps(tasks, indent=2, sort_keys=True), encoding="utf-8")
    metadata_path = write_submission_metadata(submit_dir, args, manifest_path, tasks)

    logs_dir = submit_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    grid_run_cmd = build_grid_run_cmd(args, manifest_path, len(tasks))
    bootstrap_state_source = expand_and_resolve(args.bootstrap_state_source)
    required_counts = validate_state_source(bootstrap_state_source)

    if args.print_only:
        print("BOOTSTRAP_STATE_SOURCE", str(bootstrap_state_source))
        print("BOOTSTRAP_STATE_VALIDATED", json.dumps(required_counts, sort_keys=True))
        print("BOOTSTRAP_STATE_TARGET", resolve_results_sector_dir(ROOT_DIR, resolve_working_sector_name()))
        print("TASK_MANIFEST", manifest_path)
        print("SUBMISSION_METADATA", metadata_path)
        print("ARRAY_SIZE", len(tasks))
        print("RESULTS_ROOT", Path(args.results_root).resolve())
        print("LOG_DIR", logs_dir)
        print("GRID_RUN_CMD", " ".join(shlex.quote(part) for part in grid_run_cmd))
        return

    restore_phi_calibration_state(tasks, bootstrap_state_source)
    subprocess.run(grid_run_cmd, check=True, cwd=logs_dir)
    print("TASK_MANIFEST", manifest_path)
    print("SUBMISSION_METADATA", metadata_path)
    print("ARRAY_SIZE", len(tasks))
    print("RESULTS_ROOT", Path(args.results_root).resolve())
    print("LOG_DIR", logs_dir)


def _stage_job_dir(task: dict, job_root: Path) -> Path:
    job_tag = f"phi_calibration_{_sanitize_token(task['name'])}"
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


def _link_tree(source_dir: Path, target_dir: Path) -> int:
    source_dir = source_dir.resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Shared phi calibration working sector is missing: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Shared phi calibration working sector is not a directory: {source_dir}")

    linked = 0
    target_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(source_dir.rglob("*")):
        rel = src.relative_to(source_dir)
        dst = target_dir / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        if os.path.lexists(dst):
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(src, dst)
        linked += 1
    return linked


def _unlink_output_if_symlink(path: Path) -> None:
    if path.is_symlink():
        path.unlink()


def _stage_task_bootstrap_state(task: dict, job_dir: Path) -> int:
    source = job_dir / "results" / resolve_working_sector_name()
    target = job_dir / "results" / str(task["sector_name"])
    linked = _link_tree(source, target)
    for output_path in phi_calibration_output_paths(task):
        _unlink_output_if_symlink(job_dir / output_path)
    return linked


def _write_overlay(task: dict, output_dir: Path) -> Path:
    fd, raw_path = tempfile.mkstemp(
        prefix=f"energymod_phi_calibration_{_sanitize_token(task['name'])}.",
        suffix=".yaml",
        dir=str(output_dir),
        text=True,
    )
    path = Path(raw_path)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(
            "\n".join(
                [
                    "run:",
                    "  allow_scenario_failure: false",
                    f"  sector_name: \"{task['sector_name']}\"",
                    "",
                    "scenario:",
                    "  planning_horizons:",
                    "    - 2020",
                    "    - 2025",
                    "  learning_rate:",
                    "    - \"base\"",
                    "  learning_model:",
                    f"    - \"{task['model']}\"",
                    "",
                    "permstore:",
                    "  path: permstorage",
                    "",
                    "learning:",
                    "  enabled: true",
                    "  execution_mode: \"bootstrap\"",
                    "  engine: stochastic_forecast",
                    f"  selected_model: \"{task['model']}\"",
                    f"  seed: {int(task['seed'])}",
                    "  monte_carlo:",
                    "    enable: false",
                    "  deployment_constraints:",
                    "    enabled: true",
                    "    apply_from_year: 2025",
                    "    wedge:",
                    f"      b1_multiplier: {float(task['b1_multiplier'])}",
                    f"      b2_multiplier: {float(task['b2_multiplier'])}",
                    f"      phi2_pct_capex: {float(task['phi2_pct_capex'])}",
                    f"      phi3_pct_capex: {float(task['phi3_pct_capex'])}",
                    "",
                ]
            )
        )
    return path


def _scenario_token(year: int) -> str:
    return f"elec_s_200_lcopt_1h_1h_{int(year)}_0.071_AB_0.0export_base"


def _scenario_token_ec(year: int) -> str:
    return f"elec_s_200_ec_lcopt_1h_1h_{int(year)}_0.071_AB_0.0export_base"


def phi_calibration_year(task: dict) -> int:
    return max(int(year) for year in task.get("planning_horizons", [2025]))


def phi_calibration_solve_target(task: dict) -> str:
    year = phi_calibration_year(task)
    token = _scenario_token_ec(year)
    return str(Path("results") / str(task["sector_name"]) / "postnetworks" / f"{token}.nc")


def phi_calibration_target(task: dict) -> str:
    year = phi_calibration_year(task)
    token = _scenario_token(year)
    return str(Path("results") / str(task["sector_name"]) / "learning" / f"cost_log_solved_{token}.csv")


def phi_calibration_output_paths(task: dict) -> list[str]:
    year = phi_calibration_year(task)
    token = _scenario_token(year)
    token_ec = _scenario_token_ec(year)
    sector = Path("results") / str(task["sector_name"])
    return [
        str(sector / "postnetworks" / f"{token_ec}.nc"),
        str(sector / "postnetworks" / "lpfiles" / f"{token_ec}.lp"),
        str(sector / "learning" / f"cost_log_solved_{token}.csv"),
        str(sector / "learning" / f"state_committed_{token}.json"),
        str(sector / "learning" / f"system_costs_{token}.csv"),
        str(sector / "learning" / f"statistics_{token}.csv"),
        str(sector / "learning" / f"deployment_constraints_{token}.csv"),
    ]


def build_worker_snakemake_cmd(
    task: dict,
    job_dir: Path,
    overlay_path: Path,
    snakemake_jobs: str,
    dry_run: bool = False,
    snakemake_executable: str = "snakemake",
) -> list[str]:
    cmd = [
        snakemake_executable,
        f"-j{snakemake_jobs}",
        phi_calibration_target(task),
        "--configfile",
        "config.myopic.yaml",
        "config.learning.yaml",
        str(overlay_path),
        "--rerun-triggers",
        "code",
        "--forcerun",
        phi_calibration_solve_target(task),
        phi_calibration_target(task),
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
    linked = _stage_task_bootstrap_state(task, job_dir)
    validate_state_source(job_dir / "results" / str(task["sector_name"]))
    overlay_path = _write_overlay(task, job_dir)
    snakemake_jobs = os.environ.get("JOBS") or os.environ.get("NSLOTS") or "4"
    snakemake_executable = shutil.which("snakemake") or str(
        Path(os.path.expandvars(args.conda_env)).resolve() / "bin" / "snakemake"
    )
    cmd = build_worker_snakemake_cmd(
        task,
        job_dir,
        overlay_path,
        snakemake_jobs,
        dry_run=args.dry_run,
        snakemake_executable=snakemake_executable,
    )

    env = os.environ.copy()
    env.setdefault("XDG_CACHE_HOME", "/tmp/energymod_phi_calibration_cache")
    env.setdefault("MPLCONFIGDIR", "/tmp/energymod_phi_calibration_matplotlib")

    print("Running phi calibration task:")
    print(f"  task={task['name']}")
    print(f"  sector_name={task['sector_name']}")
    print(f"  phi2_pct_capex={task['phi2_pct_capex']}")
    print(f"  phi3_pct_capex={task['phi3_pct_capex']}")
    print(f"  b1_multiplier={task['b1_multiplier']}")
    print(f"  b2_multiplier={task['b2_multiplier']}")
    print(f"  job_dir={job_dir}")
    print(f"  linked_bootstrap_files={linked}")
    print(f"  overlay={overlay_path}")
    print(f"  target={job_dir / phi_calibration_target(task)}")

    try:
        subprocess.run(cmd, check=True, cwd=job_dir, env=env)
    finally:
        if args.dry_run:
            overlay_path.unlink(missing_ok=True)

    if not args.dry_run:
        overlay_path.unlink(missing_ok=True)
        shutil.rmtree(job_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--manifest")
    parser.add_argument("--job-root", default=str(DEFAULT_JOB_ROOT))
    parser.add_argument(
        "--bootstrap-state-source",
        "--bootstrap-source",
        dest="bootstrap_state_source",
        default=str(DEFAULT_BOOTSTRAP_STATE_SOURCE),
        help="Path to the bootstrap/prerequisite state copied into each phi calibration sector",
    )
    parser.add_argument("--submit-root", default=str(DEFAULT_SUBMIT_ROOT))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV)
    parser.add_argument("--grid-mem", default=DEFAULT_GRID_MEM)
    parser.add_argument("--grid-ncpus", default=DEFAULT_GRID_NCPUS)
    parser.add_argument("--grid-submit", default=DEFAULT_GRID_SUBMIT)
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Worker mode: pass -n to Snakemake")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--phi2-min", type=float, default=0.0)
    parser.add_argument("--phi2-max", type=float, default=0.45)
    parser.add_argument("--phi2-count", type=int, default=10)
    parser.add_argument("--phi3-min", type=float, default=0.10)
    parser.add_argument("--phi3-max", type=float, default=1.00)
    parser.add_argument("--phi3-count", type=int, default=10)
    args = parser.parse_args()

    if args.worker:
        if not args.manifest:
            raise ValueError("--manifest is required in worker mode")
        run_worker(args)
        return

    submit_array(args)


if __name__ == "__main__":
    main()
