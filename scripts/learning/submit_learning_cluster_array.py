#!/usr/bin/env python3
"""Submit or execute the supported SGE array workflow for stochastic learning runs."""

import argparse
import hashlib
import json
import os
import random
import shlex
import shutil
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml
from bootstrap_state_store import (
    DEFAULT_SECTOR_ROOT,
    resolve_results_sector_dir,
    resolve_scenario_sector_name,
    restore_bootstrap_state,
    validate_state_source,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path(
    os.environ.get("LEARNING_REPO_ROOT", SCRIPT_DIR.parent.parent)
).resolve()
DEFAULT_JOB_ROOT = ROOT_DIR / "cluster_workdirs"
DEFAULT_CONDA_ENV = "/shared/share_cki25/envs/sh-pypsa-earth-main"
LAUNCHER_BUNDLE_FILES = (
    "bootstrap_state_store.py",
    "submit_learning_cluster_array.py",
    "run_learning_cluster_array_task.sh",
    "run_learning_cluster_job.sh",
    "run_learning_stochastic_job.sh",
)
WORKFLOW_CONFIGFILES = [
    str((ROOT_DIR / "config.default.yaml").resolve()),
    str((ROOT_DIR / "configs" / "bundle_config.yaml").resolve()),
    str((ROOT_DIR / "configs" / "powerplantmatching_config.yaml").resolve()),
    str((ROOT_DIR / "config.sectorelectrification.yaml").resolve()),
    str((ROOT_DIR / "config.yaml").resolve()),
]
DEFAULT_CONFIGFILES = WORKFLOW_CONFIGFILES + [
    str((ROOT_DIR / "config.myopic.yaml").resolve()),
    str((ROOT_DIR / "config.learning.yaml").resolve()),
]
SUPPORTED_MODELS = [
    "shared_state_bayesian_regime_wright",
    "way_fixed_rho_benchmark",
    "correlated_geometric_random_walk",
]
SUPPORTED_COST_EXPECTATION_KERNEL_MODES = {
    "global_current_window",
    "technology_specific_lagged_window",
}
SUPPORTED_COST_EXPECTATION_LAG_TECHS = ("solar_power", "onwind_power", "battery_energy")
DEFAULT_GRID_ARRAY_CONCURRENCY = 200
DEFAULT_GRID_MEM = "50G"
DEFAULT_GRID_NCPUS = "12"



def stage_launcher_bundle(submit_dir):
    """Snapshot long-lived launchers so repository edits cannot break active jobs."""
    launcher_dir = Path(submit_dir) / "launchers"
    launcher_dir.mkdir(parents=True, exist_ok=True)

    root_assignment = 'ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"'
    staged_root_assignment = f"ROOT_DIR={shlex.quote(str(ROOT_DIR))}"
    staged = {}
    for name in LAUNCHER_BUNDLE_FILES:
        source = SCRIPT_DIR / name
        if not source.is_file():
            raise RuntimeError(f"Launcher bundle source is missing: {source}")
        target = launcher_dir / name

        if source.suffix == ".sh":
            payload = source.read_text(encoding="utf-8")
            if root_assignment not in payload:
                raise RuntimeError(
                    f"Expected repository-root assignment was not found in launcher: {source}"
                )
            payload = payload.replace(root_assignment, staged_root_assignment, 1)
            if name == "run_learning_cluster_array_task.sh":
                live_worker = "python scripts/learning/submit_learning_cluster_array.py"
                snapshot_worker = (
                    f"python {shlex.quote(str((launcher_dir / 'submit_learning_cluster_array.py').resolve()))}"
                )
                if live_worker not in payload:
                    raise RuntimeError(f"Worker invocation was not found in launcher: {source}")
                payload = payload.replace(live_worker, snapshot_worker, 1)
            elif name == "run_learning_cluster_job.sh":
                live_runner = (
                    'RUNNER="${LEARNING_CLUSTER_JOB_RUNNER:-bash '
                    'scripts/learning/run_learning_stochastic_job.sh}"'
                )
                snapshot_runner = (
                    'RUNNER="${LEARNING_CLUSTER_JOB_RUNNER:-bash '
                    f"{shlex.quote(str((launcher_dir / 'run_learning_stochastic_job.sh').resolve()))}"
                    '}"'
                )
                if live_runner not in payload:
                    raise RuntimeError(f"Stochastic runner invocation was not found in launcher: {source}")
                payload = payload.replace(live_runner, snapshot_runner, 1)
            target.write_text(payload, encoding="utf-8")
        else:
            shutil.copyfile(source, target)

        target.chmod(source.stat().st_mode & 0o777)
        staged[name] = hashlib.sha256(target.read_bytes()).hexdigest()

    return launcher_dir.resolve(), staged

def stage_scenario_config(target_dir, configfiles, sector_name, config_overlay=None):
    """Write the effective shared scenario config once before array workers start."""
    effective_configfiles = list(configfiles)
    if config_overlay:
        effective_configfiles.append(str(config_overlay))
    effective_config = load_merged_config(effective_configfiles)
    effective_config.setdefault("run", {})["sector_name"] = str(sector_name)

    config_path = Path(target_dir) / "configs" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(effective_config, stream, sort_keys=False)
    return config_path


def expand_and_resolve(pathlike):
    return Path(os.path.expandvars(str(pathlike))).resolve()


def resolve_task_sector_names(tasks, sector_root=DEFAULT_SECTOR_ROOT):
    return sorted(
        {
            task.get("resolved_sector_name")
            or resolve_scenario_sector_name(str(task["scenario_name"]), sector_root=sector_root)
            for task in tasks
        }
    )


def resolve_sector_root(merged_config, override=None):
    if override:
        return str(override).strip().strip("/")
    run_cfg = (merged_config or {}).get("run", {}) or {}
    sector_name = run_cfg.get("sector_name")
    if sector_name:
        return str(sector_name).strip().strip("/")
    return DEFAULT_SECTOR_ROOT


def attach_sector_names(tasks, sector_root):
    for task in tasks:
        task["sector_root"] = sector_root
        task["resolved_sector_name"] = resolve_scenario_sector_name(
            str(task["scenario_name"]), sector_root=sector_root
        )
    return tasks



def parse_models_arg(raw_models):
    if not raw_models:
        return None
    models = []
    for entry in raw_models:
        if entry is None:
            continue
        for model in str(entry).split(','):
            model = model.strip()
            if model:
                models.append(model)
    return models or None


def parse_seeds_arg(raw_seeds):
    if not raw_seeds:
        return None
    seeds = []
    for entry in raw_seeds:
        for value in str(entry).split(","):
            value = value.strip()
            if value:
                seeds.append(int(value))
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    if any(seed < 0 for seed in seeds):
        raise ValueError("--seeds values must be non-negative integers")
    if len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must not contain duplicates")
    return seeds


def _validate_cost_expectation_weights(weights):
    if not isinstance(weights, (list, tuple)):
        raise ValueError("cost expectation weights must be a list of five non-negative numbers")
    if len(weights) != 5:
        raise ValueError(f"cost expectation weights must have length 5 (got {len(weights)})")
    try:
        values = [float(w) for w in weights]
    except (TypeError, ValueError) as exc:
        raise ValueError("cost expectation weights must contain only numeric values") from exc
    if any(w < 0 for w in values):
        raise ValueError("cost expectation weights must be non-negative")
    if sum(values) <= 0:
        raise ValueError("cost expectation weights must sum to a positive value")
    return values


def _validate_cost_expectation_kernel_mode(mode):
    mode = str(mode).strip()
    if mode not in SUPPORTED_COST_EXPECTATION_KERNEL_MODES:
        raise ValueError(
            f"cost expectation kernel_mode must be one of {sorted(SUPPORTED_COST_EXPECTATION_KERNEL_MODES)}"
        )
    return mode


def _validate_cost_expectation_lag_years_by_tech(payload):
    if not isinstance(payload, dict):
        raise ValueError("lag_years_by_tech must be a mapping of technology -> non-negative integer")
    missing = sorted(set(SUPPORTED_COST_EXPECTATION_LAG_TECHS) - set(payload))
    extra = sorted(set(payload) - set(SUPPORTED_COST_EXPECTATION_LAG_TECHS))
    if missing or extra:
        raise ValueError(
            f"lag_years_by_tech must contain exactly {list(SUPPORTED_COST_EXPECTATION_LAG_TECHS)} "
            f"(missing={missing}, extra={extra})"
        )
    validated = {}
    for tech in SUPPORTED_COST_EXPECTATION_LAG_TECHS:
        value = payload[tech]
        if not isinstance(value, int) or value < 0:
            raise ValueError("lag_years_by_tech values must be non-negative integers")
        validated[tech] = int(value)
    return validated


def load_cost_expectation_scenarios(path):
    if not path:
        return None
    payload = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if isinstance(payload, dict) and 'scenarios' in payload:
        scenarios = payload.get('scenarios') or []
    elif isinstance(payload, list):
        scenarios = payload
    else:
        raise ValueError("cost expectation scenarios must be a list or a mapping with 'scenarios'")

    normalized = []
    for idx, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            raise ValueError(f"scenario #{idx} must be a mapping")
        name = str(scenario.get('name', '')).strip()
        if not name:
            raise ValueError(f"scenario #{idx} is missing a name")
        mode = str(scenario.get('mode', 'block_average_expected')).strip()
        if mode not in {'block_average_expected', 'point_cost'}:
            raise ValueError(
                f"scenario '{name}' has unsupported mode '{mode}' (use block_average_expected or point_cost)"
            )
        kernel_mode = scenario.get('kernel_mode', None)
        lag_years_by_tech = scenario.get('lag_years_by_tech', None)
        weights = scenario.get('annual_weights', None)
        if mode == 'point_cost':
            weights = None
            kernel_mode = None
            lag_years_by_tech = None
        elif weights is None:
            raise ValueError(f"scenario '{name}' requires annual_weights for block_average_expected")
        else:
            weights = _validate_cost_expectation_weights(weights)
            if kernel_mode is not None:
                kernel_mode = _validate_cost_expectation_kernel_mode(kernel_mode)
            if lag_years_by_tech is not None:
                lag_years_by_tech = _validate_cost_expectation_lag_years_by_tech(lag_years_by_tech)
        normalized.append({
            'name': name,
            'mode': mode,
            'annual_weights': weights,
            'kernel_mode': kernel_mode,
            'lag_years_by_tech': lag_years_by_tech,
        })
    return normalized

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
        cfg_path = expand_and_resolve(cfg)
        if not cfg_path.is_file():
            raise ValueError(f"Config file is not a regular file: {cfg_path}")
        raw = cfg_path.read_bytes()
        if not raw.strip():
            raise ValueError(f"Config file is empty: {cfg_path}")
        try:
            payload = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ValueError(f"Config file is invalid YAML: {cfg_path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Config file must contain a YAML mapping: {cfg_path}")
        deep_update(merged, payload)
    return merged


def write_effective_config_snapshot(submit_dir, effective_config):
    """Persist the exact merged configuration consumed by every array worker."""
    path = Path(submit_dir) / "configs" / "effective_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = yaml.safe_dump(effective_config, sort_keys=False).encode("utf-8")
    path.write_bytes(raw)
    return path.resolve(), hashlib.sha256(raw).hexdigest()


def _config_changed_paths(before, after, prefix=()):
    """Return leaf paths whose effective values differ."""
    if isinstance(before, dict) and isinstance(after, dict):
        changed = []
        for key in sorted(set(before) | set(after), key=str):
            path = prefix + (str(key),)
            if key not in before or key not in after:
                changed.append(".".join(path))
            else:
                changed.extend(_config_changed_paths(before[key], after[key], path))
        return changed
    return [".".join(prefix)] if before != after else []


def validate_config_overlay(path, merged_config):
    """Validate that an overlay is a non-empty mapping and changes the config."""
    overlay_path = expand_and_resolve(path)
    if not overlay_path.is_file():
        raise ValueError(f"Config overlay is not a regular file: {overlay_path}")
    raw = overlay_path.read_bytes()
    if not raw.strip():
        raise ValueError(f"Config overlay is empty: {overlay_path}")
    try:
        payload = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"Config overlay is invalid YAML: {overlay_path}") from exc
    if not isinstance(payload, dict) or not payload:
        raise ValueError(
            f"Config overlay must contain a non-empty YAML mapping: {overlay_path}"
        )

    effective = deepcopy(merged_config)
    deep_update(effective, payload)
    changed_paths = _config_changed_paths(merged_config, effective)
    if not changed_paths:
        raise ValueError(
            f"Config overlay makes no effective configuration changes: {overlay_path}"
        )
    return {
        "source_path": str(overlay_path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "changed_paths": changed_paths,
        "raw": raw,
    }


def default_configfiles():
    return list(DEFAULT_CONFIGFILES)


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




def build_tasks(
    configfiles,
    scenario_name,
    override_models=None,
    override_draws=None,
    config_overlay=None,
    override_seeds=None,
    scenario_variants=None,
):
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

    if override_draws is not None:
        mc_cfg = deepcopy(mc_cfg)
        mc_cfg["draws"] = int(override_draws)

    models = override_models or [str(x) for x in (mc_cfg.get("stochastic_models", []) or [])]
    if not models:
        raise ValueError("learning.monte_carlo.stochastic_models must be non-empty")
    unsupported = sorted(set(models) - set(SUPPORTED_MODELS))
    if unsupported:
        raise ValueError(f"Unsupported stochastic models: {unsupported}")

    seeds = override_seeds if override_seeds is not None else build_seed_list(mc_cfg, learning_cfg)
    tasks = []

    if scenario_variants:
        for variant in scenario_variants:
            variant_name = str(variant["name"])
            scenario_tag = f"{scenario_name}-{variant_name}"
            for model in models:
                for seed in seeds:
                    tasks.append(
                        {
                            "scenario_name": scenario_tag,
                            "model": model,
                            "seed": int(seed),
                            "cost_expectation_mode": variant["mode"],
                            "cost_expectation_weights": variant.get("annual_weights"),
                            "cost_expectation_kernel_mode": variant.get("kernel_mode"),
                            "cost_expectation_lag_years_by_tech": variant.get("lag_years_by_tech"),
                            "config_overlay": config_overlay,
                        }
                    )
        return tasks

    for model in models:
        for seed in seeds:
            tasks.append(
                {
                    "scenario_name": scenario_name,
                    "model": model,
                    "seed": int(seed),
                    "config_overlay": config_overlay,
                }
            )
    return tasks


def build_grid_run_cmd(args, manifest_path, task_count, worker_script):
    concurrency = int(getattr(args, "grid_array_concurrency", None) or DEFAULT_GRID_ARRAY_CONCURRENCY)
    if concurrency <= 0:
        raise ValueError("--grid-array-concurrency must be a positive integer")
    cmd = [
        "grid_run",
        f"--grid_mem={args.grid_mem}",
        f"--grid_ncpus={args.grid_ncpus}",
        f"--grid_submit={args.grid_submit}",
    ]
    if task_count == 1:
        # grid_run rejects the otherwise valid SGE range "1-1". Submit a
        # normal batch job and supply the sole array index to the worker.
        task_index_env = "SGE_TASK_ID=1"
    else:
        grid_array = f"1-{task_count}"
        if task_count > concurrency:
            grid_array = f"{grid_array}/{concurrency}"
        cmd.append(f"--grid_array={grid_array}")
        task_index_env = None

    cmd.extend(
        [
            worker_script,
            str(manifest_path),
            *([task_index_env] if task_index_env else []),
            f"LEARNING_JOB_ROOT={Path(os.path.expandvars(args.job_root)).resolve()}",
            f"LEARNING_CONDA_ENV={args.conda_env}",
            f"LEARNING_RUN_MODE={args.run_mode}",
            f"LEARNING_REPO_ROOT={ROOT_DIR}",
        ]
    )
    return cmd


def write_submission_metadata(
    submit_dir,
    args,
    configfiles,
    tasks,
    merged_config=None,
    scenario_variants=None,
    override_draws=None,
    override_models=None,
    override_seeds=None,
    config_overlay=None,
    config_overlay_info=None,
    effective_config_path=None,
    effective_config_sha256=None,
    resolved_sector_name=None,
    resolved_sector_names=None,
    bootstrap_state_source=None,
    sector_root=None,
):
    if merged_config is None:
        merged_config = load_merged_config(configfiles)

    learning_cfg = merged_config.get("learning", {}) or {}
    mc_cfg = learning_cfg.get("monte_carlo", {}) or {}
    metadata = {
        "scenario_name": args.scenario_name,
        "run_mode": args.run_mode,
        "configfiles": [str(Path(cfg).resolve()) for cfg in configfiles],
        "grid_mem": args.grid_mem,
        "grid_ncpus": args.grid_ncpus,
        "grid_submit": args.grid_submit,
        "grid_array_concurrency": int(args.grid_array_concurrency or DEFAULT_GRID_ARRAY_CONCURRENCY),
        "job_root": str(Path(os.path.expandvars(args.job_root)).resolve()),
        "conda_env": args.conda_env,
        "task_count": len(tasks),
        "models": sorted({task["model"] for task in tasks}),
        "monte_carlo": {
            "seed_mode": mc_cfg.get("seed_mode", "random"),
            "random_seed": int(mc_cfg.get("random_seed", learning_cfg.get("seed", 0) or 0)),
            "seed_upper_bound": int(mc_cfg.get("seed_upper_bound", 1000000000) or 1000000000),
            "draws": int(
                len(override_seeds)
                if override_seeds is not None
                else override_draws if override_draws is not None else mc_cfg.get("draws", 0) or 0
            ),
            "explicit_seeds": override_seeds,
        },
        "effective_config": str(effective_config_path) if effective_config_path else None,
        "effective_config_sha256": effective_config_sha256,
        "scenario_variants": scenario_variants or [],
        "models_override": override_models or None,
        "config_overlay": config_overlay,
        "config_overlay_source": (
            config_overlay_info.get("source_path") if config_overlay_info else None
        ),
        "config_overlay_sha256": (
            config_overlay_info.get("sha256") if config_overlay_info else None
        ),
        "config_overlay_changed_paths": config_overlay_info.get("changed_paths", []) if config_overlay_info else [],
        "sector_root": sector_root,
        "resolved_sector_name": resolved_sector_name,
        "resolved_sector_names": resolved_sector_names or ([] if resolved_sector_name is None else [resolved_sector_name]),
        "bootstrap_state_source": bootstrap_state_source,
    }
    metadata_path = submit_dir / "submission_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata_path


def submit_array(args):
    configfiles = default_configfiles()
    merged_config = load_merged_config(configfiles)

    # Positional configs are ordered overrides. Validate each one against the
    # config accumulated so far so empty or semantic no-op ablations cannot be
    # submitted as if they changed the experiment.
    for user_config in args.configfiles:
        info = validate_config_overlay(user_config, merged_config)
        configfiles.append(info["source_path"])
        merged_config = load_merged_config(configfiles)

    scenario_variants = load_cost_expectation_scenarios(args.cost_expectation_scenarios)
    override_models = parse_models_arg(args.models)
    override_seeds = parse_seeds_arg(args.seeds)
    config_overlay_info = (
        validate_config_overlay(args.config_overlay, merged_config)
        if args.config_overlay
        else None
    )
    config_overlay = config_overlay_info["source_path"] if config_overlay_info else None
    if config_overlay:
        configfiles.append(config_overlay)
        merged_config = load_merged_config(configfiles)

    # Resolve task planning from the same merged configuration that workers use.
    sector_root = resolve_sector_root(merged_config, args.sector_root)
    tasks = build_tasks(
        configfiles,
        args.scenario_name,
        override_models=override_models,
        override_draws=args.draws,
        override_seeds=override_seeds,
        scenario_variants=scenario_variants,
    )
    tasks = attach_sector_names(tasks, sector_root)
    resolved_sector_name = resolve_scenario_sector_name(args.scenario_name, sector_root=sector_root)
    resolved_sector_names = resolve_task_sector_names(tasks, sector_root=sector_root)
    bootstrap_state_source = (
        str(expand_and_resolve(args.bootstrap_state_source))
        if args.bootstrap_state_source
        else None
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submit_root = Path(args.submit_root).resolve()
    submit_dir = submit_root / f"{args.scenario_name}_{timestamp}"
    submit_dir.mkdir(parents=True, exist_ok=True)

    if config_overlay_info:
        overlay_snapshot = submit_dir / "configs" / "config_overlay.yaml"
        overlay_snapshot.parent.mkdir(parents=True, exist_ok=True)
        overlay_snapshot.write_bytes(config_overlay_info["raw"])
        config_overlay = str(overlay_snapshot.resolve())

    effective_config_path, effective_config_sha256 = write_effective_config_snapshot(
        submit_dir,
        merged_config,
    )
    launcher_dir, launcher_sha256 = stage_launcher_bundle(submit_dir)
    for task in tasks:
        task["effective_config"] = str(effective_config_path)
        task["effective_config_sha256"] = effective_config_sha256

    manifest_path = submit_dir / "task_manifest.json"
    manifest_path.write_text(json.dumps(tasks, indent=2, sort_keys=True), encoding="utf-8")
    metadata_path = write_submission_metadata(
        submit_dir,
        args,
        configfiles,
        tasks,
        merged_config=merged_config,
        scenario_variants=scenario_variants,
        override_draws=args.draws,
        override_models=override_models,
        override_seeds=override_seeds,
        config_overlay=config_overlay,
        config_overlay_info=config_overlay_info,
        effective_config_path=effective_config_path,
        effective_config_sha256=effective_config_sha256,
        resolved_sector_name=resolved_sector_name,
        resolved_sector_names=resolved_sector_names,
        bootstrap_state_source=bootstrap_state_source,
        sector_root=sector_root,
    )

    logs_dir = submit_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    worker_script = str((launcher_dir / "run_learning_cluster_array_task.sh").resolve())
    grid_run_cmd = build_grid_run_cmd(args, manifest_path, len(tasks), worker_script)

    if bootstrap_state_source:
        required_counts = validate_state_source(Path(bootstrap_state_source))
        if args.print_only:
            print("BOOTSTRAP_STATE_SOURCE", bootstrap_state_source)
            print("BOOTSTRAP_STATE_VALIDATED", json.dumps(required_counts, sort_keys=True))
            for sector_name in resolved_sector_names:
                target_dir = resolve_results_sector_dir(ROOT_DIR, sector_name)
                print("BOOTSTRAP_STATE_TARGET", target_dir)
        else:
            print("BOOTSTRAP_STATE_SOURCE", bootstrap_state_source)
            print("BOOTSTRAP_STATE_VALIDATED", json.dumps(required_counts, sort_keys=True))
            for sector_name in resolved_sector_names:
                target_dir = resolve_results_sector_dir(ROOT_DIR, sector_name)
                restore_summary = restore_bootstrap_state(Path(bootstrap_state_source), target_dir)
                config_path = stage_scenario_config(
                    target_dir,
                    [str(effective_config_path)],
                    sector_name,
                )
                print("BOOTSTRAP_STATE_RESTORED", json.dumps(restore_summary, sort_keys=True))
                print("SCENARIO_CONFIG_STAGED", config_path)

    if args.print_only:
        print("TASK_MANIFEST", manifest_path)
        print("SUBMISSION_METADATA", metadata_path)
        print("EFFECTIVE_CONFIG", effective_config_path)
        print("EFFECTIVE_CONFIG_SHA256", effective_config_sha256)
        print("ARRAY_SIZE", len(tasks))
        print("RESOLVED_SECTOR_NAME", resolved_sector_name)
        print("RESOLVED_SECTOR_NAMES", json.dumps(resolved_sector_names, sort_keys=True))
        print("LAUNCHER_BUNDLE", launcher_dir)
        print("LAUNCHER_SHA256", json.dumps(launcher_sha256, sort_keys=True))
        print("LOG_DIR", logs_dir)
        if config_overlay_info:
            print("CONFIG_OVERLAY_SHA256", config_overlay_info["sha256"])
            print("CONFIG_OVERLAY_CHANGED_PATHS", json.dumps(config_overlay_info["changed_paths"]))
        print("GRID_RUN_CMD", " ".join(shlex.quote(part) for part in grid_run_cmd))
        return

    subprocess.run(grid_run_cmd, check=True, cwd=logs_dir)
    print("TASK_MANIFEST", manifest_path)
    print("SUBMISSION_METADATA", metadata_path)
    print("EFFECTIVE_CONFIG", effective_config_path)
    print("EFFECTIVE_CONFIG_SHA256", effective_config_sha256)
    print("ARRAY_SIZE", len(tasks))
    print("RESOLVED_SECTOR_NAME", resolved_sector_name)
    print("RESOLVED_SECTOR_NAMES", json.dumps(resolved_sector_names, sort_keys=True))
    print("LAUNCHER_BUNDLE", launcher_dir)
    print("LAUNCHER_SHA256", json.dumps(launcher_sha256, sort_keys=True))
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
    env["LEARNING_SECTOR_NAME"] = task.get("resolved_sector_name") or resolve_scenario_sector_name(
        str(task["scenario_name"]), sector_root=str(task.get("sector_root", DEFAULT_SECTOR_ROOT))
    )

    effective_config = task.get("effective_config")
    if effective_config:
        effective_path = expand_and_resolve(effective_config)
        if not effective_path.is_file():
            raise RuntimeError(f"Effective config is missing: {effective_path}")
        raw = effective_path.read_bytes()
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        expected_sha256 = task.get("effective_config_sha256")
        if not expected_sha256:
            raise RuntimeError("Task manifest is missing effective_config_sha256")
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                "Effective config hash mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        env["LEARNING_EFFECTIVE_CONFIG"] = str(effective_path)
        env["LEARNING_EFFECTIVE_CONFIG_SHA256"] = actual_sha256
    elif task.get("config_overlay"):
        # Backward compatibility for manifests created before effective-config
        # snapshots were introduced.
        env["LEARNING_CONFIG_OVERLAY"] = str(task["config_overlay"])
    if "cost_expectation_mode" in task:
        env["LEARNING_COST_EXPECTATION_MODE"] = str(task["cost_expectation_mode"])
    if "cost_expectation_weights" in task and task["cost_expectation_weights"] is not None:
        env["LEARNING_COST_EXPECTATION_WEIGHTS"] = json.dumps(task["cost_expectation_weights"])
    if "cost_expectation_kernel_mode" in task and task["cost_expectation_kernel_mode"] is not None:
        env["LEARNING_COST_EXPECTATION_KERNEL_MODE"] = str(task["cost_expectation_kernel_mode"])
    if (
        "cost_expectation_lag_years_by_tech" in task
        and task["cost_expectation_lag_years_by_tech"] is not None
    ):
        env["LEARNING_COST_EXPECTATION_LAG_YEARS_BY_TECH"] = json.dumps(
            task["cost_expectation_lag_years_by_tech"]
        )
    subprocess.run(cmd, check=True, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--manifest")
    parser.add_argument("--job-root", default=str(DEFAULT_JOB_ROOT))
    parser.add_argument("--submit-root", default="cluster_submissions")
    parser.add_argument("--scenario-name", default="learning_mc")
    parser.add_argument(
        "--sector-root",
        help="Base results sector for scenario branches; defaults to run.sector_name from the merged config",
    )
    parser.add_argument("--job-name", default="learnmc")
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV)
    parser.add_argument("--run-mode", choices=["branch", "full"], default="branch")
    parser.add_argument(
        "--grid-mem",
        default=DEFAULT_GRID_MEM,
        help=(
            "Total memory passed to grid_run (default: %(default)s). "
            "grid_run divides this across --grid-ncpus slots."
        ),
    )
    parser.add_argument("--grid-ncpus", default=DEFAULT_GRID_NCPUS)
    parser.add_argument("--grid-submit", default="batch")
    parser.add_argument(
        "--grid-array-concurrency",
        type=int,
        default=DEFAULT_GRID_ARRAY_CONCURRENCY,
        help="Maximum simultaneous SGE array tasks; emitted as grid_array=1-N/C",
    )
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--models", action="append", help="Override stochastic models (comma-separated or repeatable)")
    parser.add_argument("--draws", type=int, help="Override learning.monte_carlo.draws")
    parser.add_argument(
        "--seeds",
        action="append",
        help="Run these explicit draw seeds (comma-separated or repeatable)",
    )
    parser.add_argument("--cost-expectation-scenarios", help="YAML/JSON file with cost expectation scenario definitions")
    parser.add_argument(
        "--config-overlay",
        help="YAML overlay appended after the normal stochastic-job configuration",
    )
    parser.add_argument(
        "--bootstrap-state-source",
        help="Path to saved bootstrap/prerequisite state to stage before stochastic array submission",
    )
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
