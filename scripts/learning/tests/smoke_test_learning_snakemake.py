#!/usr/bin/env python3
"""Synthetic Snakemake smoke tests for the energymod learning workflow."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from learning_runtime_smoke_utils import (
    ENERGYMOD_ROOT,
    LEGACY_LEARNING_SEED,
    SHORTLIST_MODELS,
    build_mock_network,
    format_learning_seed,
    resolve_mock_cost_file,
    solve_mock_network,
)


DEFAULT_SINGLE_MODEL = "shared_state_bayesian_regime_wright"


def _first(value):
    if isinstance(value, list):
        if not value:
            raise ValueError("Encountered empty scenario/config list while building synthetic smoke config.")
        return value[0]
    return value


def build_synthetic_config(output_root, mode, single_model, mc_models, mc_seeds):
    cfg = yaml.safe_load((ENERGYMOD_ROOT / "config.myopic.yaml").read_text(encoding="utf-8"))
    learning_cfg = yaml.safe_load((ENERGYMOD_ROOT / "config.learning.yaml").read_text(encoding="utf-8"))
    cfg["learning"] = learning_cfg.get("learning", {})

    cfg["results_dir"] = "results/"
    cfg["summary_dir"] = "results/"
    cfg["run"]["name"] = "Earth_200"
    cfg["run"]["sector_name"] = "Global_200_synthetic_smoke"
    cfg["run"]["allow_scenario_failure"] = False

    cfg["scenario"]["simpl"] = [""]
    cfg["scenario"]["ll"] = ["copt"]
    cfg["scenario"]["clusters"] = [200]
    cfg["scenario"]["opts"] = ["1h"]
    cfg["scenario"]["sopts"] = ["1h"]
    cfg["scenario"]["planning_horizons"] = [2020, 2025]
    cfg["scenario"]["demand"] = ["AB"]
    cfg["scenario"]["learning_rate"] = ["base"]
    cfg["scenario"]["learning_model"] = [single_model] if mode == "single" else ["legacy_curve"]

    cfg["solving"]["save_lpfile"] = False
    cfg["solving"]["threads"] = 1
    cfg["solving"]["mem"] = 4000
    cfg["solving"]["solver"]["name"] = "gurobi"
    cfg["solving"]["solver"]["options"] = "gurobi-default"
    cfg["solving"]["solver_options"]["gurobi-default"]["threads"] = 1
    cfg["solving"]["options"]["skip_iterations"] = True
    cfg["solving"]["options"]["track_iterations"] = False
    cfg["solving"]["options"]["noisy_costs"] = False
    cfg["enable"]["retrieve_cost_data"] = False
    cfg["global_specific"]["baseyear_generation"]["baseyear_generation_constraint"] = False
    cfg["global_specific"]["year2025_generation"]["year2025_generation_constraint"] = False
    cfg["global_specific"]["year2025_capacity"]["year2025_capacity_constraint"] = False

    cfg["permstore"]["enable"] = False
    for key in list((cfg["permstore"].get("use", {}) or {}).keys()):
        cfg["permstore"]["use"][key] = False

    cfg["learning"]["monte_carlo"]["enable"] = mode == "mc"
    cfg["learning"]["monte_carlo"]["include_legacy_curve"] = False
    cfg["learning"]["monte_carlo"]["stochastic_models"] = mc_models if mode == "mc" else SHORTLIST_MODELS
    cfg["learning"]["monte_carlo"]["draws"] = len(mc_seeds) if mode == "mc" else 1
    cfg["learning"]["monte_carlo"]["seed_mode"] = "sequential"
    cfg["learning"]["monte_carlo"]["random_seed"] = 0
    cfg["learning"]["monte_carlo"]["seed_upper_bound"] = 1000000
    cfg["learning"]["compact_outputs"]["enable"] = False
    cfg["learning"]["compact_outputs"]["cleanup_heavy_raws"] = True
    cfg["learning"]["compact_outputs"]["price_shadow_resolution"] = "annual_summary"

    config_path = output_root / f"config.synthetic.{mode}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return config_path


def prepare_workspace(workspace_root):
    workspace_root.mkdir(parents=True, exist_ok=True)
    link_map = {
        "Snakefile": ENERGYMOD_ROOT / "Snakefile",
        "config.default.yaml": ENERGYMOD_ROOT / "config.default.yaml",
        "config.sectorelectrification.yaml": ENERGYMOD_ROOT / "config.sectorelectrification.yaml",
        "config.yaml": ENERGYMOD_ROOT / "config.yaml",
        "configs": ENERGYMOD_ROOT / "configs",
        "scripts": ENERGYMOD_ROOT / "scripts",
        "data": ENERGYMOD_ROOT / "data",
    }
    for name, source in link_map.items():
        target = workspace_root / name
        if target.exists() or target.is_symlink():
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)
        target.symlink_to(source, target_is_directory=source.is_dir())


def populate_workspace_resources(workspace_root):
    resources_root = workspace_root / "resources" / "Earth_200"
    resources_root.mkdir(parents=True, exist_ok=True)
    for year in (2020, 2025):
        source = resolve_mock_cost_file(year)
        target = resources_root / f"costs_{year}.csv"
        shutil.copy2(source, target)


def write_workspace_learning_config(workspace_root, seed):
    source = ENERGYMOD_ROOT / "config.learning.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["learning"]["seed"] = int(seed)
    target = workspace_root / "config.learning.yaml"
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return target


def results_root_from_config(cfg, workspace_root):
    results_root = Path(str(cfg["results_dir"]))
    if not results_root.is_absolute():
        results_root = (workspace_root / results_root).resolve()
    return results_root / str(cfg["run"]["sector_name"])


def build_brownfield_filename(cfg, workspace_root, year, learning_model, learning_seed):
    scenario = cfg["scenario"]
    simpl = _first(scenario["simpl"])
    clusters = _first(scenario["clusters"])
    ll = _first(scenario["ll"])
    opts = _first(scenario["opts"])
    sopts = _first(scenario["sopts"])
    demand = _first(scenario["demand"])
    learning_rate = _first(scenario["learning_rate"])
    discountrate = _first(cfg["costs"]["discountrate"])
    h2export = _first(cfg["export"]["h2export"])

    filename = (
        f"elec_s{simpl}_{clusters}_l{ll}_{opts}_{sopts}_{year}_{discountrate}_{demand}_"
        f"{h2export}export_{learning_rate}_model_{learning_model}_seed_{learning_seed}.nc"
    )
    return results_root_from_config(cfg, workspace_root) / "prenetworks-brownfield" / filename


def build_cost_log_filename(cfg, workspace_root, year, learning_model, learning_seed):
    scenario = cfg["scenario"]
    simpl = _first(scenario["simpl"])
    clusters = _first(scenario["clusters"])
    ll = _first(scenario["ll"])
    opts = _first(scenario["opts"])
    sopts = _first(scenario["sopts"])
    demand = _first(scenario["demand"])
    learning_rate = _first(scenario["learning_rate"])
    discountrate = _first(cfg["costs"]["discountrate"])
    h2export = _first(cfg["export"]["h2export"])

    filename = (
        f"cost_log_solved_elec_s{simpl}_{clusters}_l{ll}_{opts}_{sopts}_{year}_{discountrate}_{demand}_"
        f"{h2export}export_{learning_rate}_model_{learning_model}_seed_{learning_seed}.csv"
    )
    return results_root_from_config(cfg, workspace_root) / "learning" / filename


def build_bootstrap_cost_log_filename(cfg, workspace_root, year, learning_model):
    scenario = cfg["scenario"]
    simpl = _first(scenario["simpl"])
    clusters = _first(scenario["clusters"])
    ll = _first(scenario["ll"])
    opts = _first(scenario["opts"])
    sopts = _first(scenario["sopts"])
    demand = _first(scenario["demand"])
    learning_rate = _first(scenario["learning_rate"])
    discountrate = _first(cfg["costs"]["discountrate"])
    h2export = _first(cfg["export"]["h2export"])

    filename = (
        f"cost_log_solved_elec_s{simpl}_{clusters}_l{ll}_{opts}_{sopts}_{year}_{discountrate}_{demand}_"
        f"{h2export}export_{learning_rate}_model_{learning_model}.csv"
    )
    return results_root_from_config(cfg, workspace_root) / "learning" / filename


def build_state_filename(cfg, workspace_root, year, learning_model, learning_seed):
    scenario = cfg["scenario"]
    simpl = _first(scenario["simpl"])
    clusters = _first(scenario["clusters"])
    ll = _first(scenario["ll"])
    opts = _first(scenario["opts"])
    sopts = _first(scenario["sopts"])
    demand = _first(scenario["demand"])
    learning_rate = _first(scenario["learning_rate"])
    discountrate = _first(cfg["costs"]["discountrate"])
    h2export = _first(cfg["export"]["h2export"])

    filename = (
        f"state_committed_elec_s{simpl}_{clusters}_l{ll}_{opts}_{sopts}_{year}_{discountrate}_{demand}_"
        f"{h2export}export_{learning_rate}_model_{learning_model}_seed_{learning_seed}.json"
    )
    return results_root_from_config(cfg, workspace_root) / "learning" / filename


def build_bootstrap_state_filename(cfg, workspace_root, year, learning_model):
    scenario = cfg["scenario"]
    simpl = _first(scenario["simpl"])
    clusters = _first(scenario["clusters"])
    ll = _first(scenario["ll"])
    opts = _first(scenario["opts"])
    sopts = _first(scenario["sopts"])
    demand = _first(scenario["demand"])
    learning_rate = _first(scenario["learning_rate"])
    discountrate = _first(cfg["costs"]["discountrate"])
    h2export = _first(cfg["export"]["h2export"])

    filename = (
        f"state_committed_elec_s{simpl}_{clusters}_l{ll}_{opts}_{sopts}_{year}_{discountrate}_{demand}_"
        f"{h2export}export_{learning_rate}_model_{learning_model}.json"
    )
    return results_root_from_config(cfg, workspace_root) / "learning" / filename


def build_postnetwork_filename(cfg, workspace_root, year, learning_model, learning_seed=None):
    scenario = cfg["scenario"]
    simpl = _first(scenario["simpl"])
    clusters = _first(scenario["clusters"])
    ll = _first(scenario["ll"])
    opts = _first(scenario["opts"])
    sopts = _first(scenario["sopts"])
    demand = _first(scenario["demand"])
    learning_rate = _first(scenario["learning_rate"])
    discountrate = _first(cfg["costs"]["discountrate"])
    h2export = _first(cfg["export"]["h2export"])

    seed_suffix = f"_seed_{learning_seed}" if learning_seed is not None else ""
    filename = (
        f"elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{year}_{discountrate}_{demand}_"
        f"{h2export}export_{learning_rate}_model_{learning_model}{seed_suffix}.nc"
    )
    return results_root_from_config(cfg, workspace_root) / "postnetworks" / filename


def build_prenetwork_learning_filename(cfg, workspace_root, year, learning_model, learning_seed):
    scenario = cfg["scenario"]
    simpl = _first(scenario["simpl"])
    clusters = _first(scenario["clusters"])
    ll = _first(scenario["ll"])
    opts = _first(scenario["opts"])
    sopts = _first(scenario["sopts"])
    demand = _first(scenario["demand"])
    learning_rate = _first(scenario["learning_rate"])
    discountrate = _first(cfg["costs"]["discountrate"])
    h2export = _first(cfg["export"]["h2export"])

    filename = (
        f"elec_s{simpl}_{clusters}_l{ll}_{opts}_{sopts}_{year}_{discountrate}_{demand}_"
        f"{h2export}export_{learning_rate}_model_{learning_model}_seed_{learning_seed}.nc"
    )
    return results_root_from_config(cfg, workspace_root) / "prenetworks-learning" / filename


def build_compact_complete_path(cfg, workspace_root, learning_model, learning_seed):
    scenario = cfg["scenario"]
    simpl = _first(scenario["simpl"])
    clusters = _first(scenario["clusters"])
    ll = _first(scenario["ll"])
    opts = _first(scenario["opts"])
    sopts = _first(scenario["sopts"])
    demand = _first(scenario["demand"])
    learning_rate = _first(scenario["learning_rate"])
    discountrate = _first(cfg["costs"]["discountrate"])
    h2export = _first(cfg["export"]["h2export"])
    scenario_tag = str(cfg["run"]["sector_name"])
    return (
        results_root_from_config(cfg, workspace_root)
        / "learning-compact"
        / scenario_tag
        / learning_model
        / f"seed_{learning_seed}"
        / (
            f"compact_complete_elec_s{simpl}_{clusters}_l{ll}_{opts}_{sopts}_{discountrate}_{demand}_"
            f"{h2export}export_{learning_rate}.json"
        )
    )


def build_compact_cleanup_marker_path(cfg, workspace_root, learning_model, learning_seed):
    scenario = cfg["scenario"]
    simpl = _first(scenario["simpl"])
    clusters = _first(scenario["clusters"])
    ll = _first(scenario["ll"])
    opts = _first(scenario["opts"])
    sopts = _first(scenario["sopts"])
    demand = _first(scenario["demand"])
    learning_rate = _first(scenario["learning_rate"])
    discountrate = _first(cfg["costs"]["discountrate"])
    h2export = _first(cfg["export"]["h2export"])
    scenario_tag = str(cfg["run"]["sector_name"])
    return (
        results_root_from_config(cfg, workspace_root)
        / "learning-compact"
        / scenario_tag
        / learning_model
        / f"seed_{learning_seed}"
        / (
            f"raw_cleanup_complete_elec_s{simpl}_{clusters}_l{ll}_{opts}_{sopts}_{discountrate}_{demand}_"
            f"{h2export}export_{learning_rate}.txt"
        )
    )


def write_mock_learning_outputs(cost_log_path, state_path, model_name, learning_seed, year):
    cost_log_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "technology": "solar_power",
                "c_overnight": 750.0 - year / 100.0,
                "c_overnight_terminal_point": 751.0 - year / 100.0,
                "capital_cost": 120.0 - year / 1000.0,
                "capital_cost_terminal_point": 121.0 - year / 1000.0,
                "log_capex_terminal_point": 0.0,
                "selected_model": model_name,
                "learning_seed": learning_seed,
                "training_window": "origin_cutoff",
                "cost_expectation_mode": "block_average_expected",
                "cost_expectation_weights_json": "[0.2, 0.2, 0.2, 0.2, 0.2]",
            },
            {
                "technology": "onwind_power",
                "c_overnight": 1150.0 - year / 100.0,
                "c_overnight_terminal_point": 1151.0 - year / 100.0,
                "capital_cost": 180.0 - year / 1000.0,
                "capital_cost_terminal_point": 181.0 - year / 1000.0,
                "log_capex_terminal_point": 0.0,
                "selected_model": model_name,
                "learning_seed": learning_seed,
                "training_window": "origin_cutoff",
                "cost_expectation_mode": "block_average_expected",
                "cost_expectation_weights_json": "[0.2, 0.2, 0.2, 0.2, 0.2]",
            },
        ]
    ).to_csv(cost_log_path, index=False)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "planning_horizon": int(year),
                "selected_model": model_name,
                "learning_seed": learning_seed,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def relative_target(path, workspace_root):
    return str(Path(path).resolve().relative_to(Path(workspace_root).resolve()))


def create_synthetic_brownfields(cfg, workspace_root, model_seed_pairs):
    outputs = {}
    for model_name, learning_seed in model_seed_pairs:
        per_model = {}
        for year in (2020, 2025):
            path = build_brownfield_filename(
                cfg,
                workspace_root=workspace_root,
                year=year,
                learning_model=model_name,
                learning_seed=learning_seed,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            build_mock_network(year).export_to_netcdf(path)
            per_model[str(year)] = str(path)
        outputs[f"{model_name}:{learning_seed}"] = per_model
    return outputs


def create_synthetic_bootstrap_outputs(cfg, workspace_root, model_names, root):
    outputs = {}
    for index, model_name in enumerate(model_names):
        input_path = create_temp_network_file(root / f"bootstrap_input_{model_name}_{index}.nc", 2020)
        network_path = build_postnetwork_filename(cfg, workspace_root, 2020, model_name)
        network_path.parent.mkdir(parents=True, exist_ok=True)
        solve_mock_network(input_path, network_path)
        cost_log_path = build_bootstrap_cost_log_filename(cfg, workspace_root, 2020, model_name)
        state_path = build_bootstrap_state_filename(cfg, workspace_root, 2020, model_name)
        write_mock_learning_outputs(cost_log_path, state_path, model_name, LEGACY_LEARNING_SEED, 2020)
        outputs[model_name] = {
            "network": str(network_path),
            "cost_log": str(cost_log_path),
            "state": str(state_path),
        }
    return outputs


def run_snakemake_targets(targets, synthetic_config, workspace_root, allowed_rules=None):
    env = os.environ.copy()
    env.setdefault("HOME", "/tmp")
    env.setdefault("XDG_DATA_HOME", "/tmp")
    env.setdefault("XDG_CACHE_HOME", "/tmp")
    env.setdefault("MPLCONFIGDIR", "/tmp")

    cmd = [
        sys.executable,
        "-m",
        "snakemake",
        "-j1",
        *targets,
        "--configfile",
        str(synthetic_config),
        "config.learning.yaml",
        "--rerun-trigger",
        "mtime",
    ]
    if allowed_rules:
        cmd.extend(["--allowed-rules", *allowed_rules])
    return subprocess.run(
        cmd,
        cwd=workspace_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def validate_runtime_outputs(cfg, workspace_root, model_seed_pairs):
    rows = []
    for model_name, learning_seed in model_seed_pairs:
        cost_log_2025 = build_cost_log_filename(
            cfg,
            workspace_root=workspace_root,
            year=2025,
            learning_model=model_name,
            learning_seed=learning_seed,
        )
        state_2025 = build_state_filename(
            cfg,
            workspace_root=workspace_root,
            year=2025,
            learning_model=model_name,
            learning_seed=learning_seed,
        )
        if not cost_log_2025.exists():
            raise FileNotFoundError(f"Expected solved cost log missing: {cost_log_2025}")
        if not state_2025.exists():
            raise FileNotFoundError(f"Expected committed state missing: {state_2025}")

        df = pd.read_csv(cost_log_2025)
        payload = json.loads(state_2025.read_text(encoding="utf-8"))

        if model_name not in set(df["selected_model"].astype(str)):
            raise RuntimeError(f"{cost_log_2025} missing selected_model={model_name}")
        if learning_seed not in set(df["learning_seed"].astype(str)):
            raise RuntimeError(f"{cost_log_2025} missing learning_seed={learning_seed}")
        if payload.get("selected_model") != model_name:
            raise RuntimeError(f"{state_2025} selected_model mismatch")
        if payload.get("learning_seed") != learning_seed:
            raise RuntimeError(f"{state_2025} learning_seed mismatch")

        rows.append(
            {
                "model_name": model_name,
                "learning_seed": learning_seed,
                "cost_log_2025": str(cost_log_2025),
                "state_committed_2025": str(state_2025),
            }
        )
    return rows


def validate_mc_divergence(cfg, workspace_root, model_name, seeds):
    if len(seeds) < 2:
        raise ValueError("Need at least two seeds to validate Monte Carlo divergence.")
    token_a = format_learning_seed(model_name, seeds[0])
    token_b = format_learning_seed(model_name, seeds[1])
    log_a = build_cost_log_filename(
        cfg,
        workspace_root=workspace_root,
        year=2025,
        learning_model=model_name,
        learning_seed=token_a,
    )
    log_b = build_cost_log_filename(
        cfg,
        workspace_root=workspace_root,
        year=2025,
        learning_model=model_name,
        learning_seed=token_b,
    )
    df_a = pd.read_csv(log_a).sort_values("technology").reset_index(drop=True)
    df_b = pd.read_csv(log_b).sort_values("technology").reset_index(drop=True)

    merged = df_a[["technology", "c_overnight", "capital_cost"]].merge(
        df_b[["technology", "c_overnight", "capital_cost"]],
        on="technology",
        suffixes=("_a", "_b"),
    )
    merged["abs_cost_gap"] = (merged["c_overnight_a"] - merged["c_overnight_b"]).abs()
    merged["abs_capital_gap"] = (merged["capital_cost_a"] - merged["capital_cost_b"]).abs()
    max_gap = float(max(merged["abs_cost_gap"].max(), merged["abs_capital_gap"].max()))
    if max_gap <= 0.0:
        raise RuntimeError(
            f"Two-seed MC smoke produced identical 2025 costs for {model_name}: seeds {token_a} and {token_b}."
        )
    return {
        "model_name": model_name,
        "seed_a": token_a,
        "seed_b": token_b,
        "max_abs_gap": max_gap,
    }


def run_single_smoke(root, model_name, seed):
    root.mkdir(parents=True, exist_ok=True)
    workspace_root = root / "workspace"
    prepare_workspace(workspace_root)
    populate_workspace_resources(workspace_root)
    write_workspace_learning_config(workspace_root, seed)
    config_path = build_synthetic_config(
        workspace_root,
        "single",
        model_name,
        [],
        [],
    )
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    token = format_learning_seed(model_name, seed)
    synthetic_inputs = create_synthetic_brownfields(
        cfg,
        workspace_root,
        [(model_name, token)],
    )
    bootstrap_inputs = create_synthetic_bootstrap_outputs(
        cfg,
        workspace_root,
        [model_name],
        root,
    )
    targets = [
        relative_target(
            build_cost_log_filename(
                cfg,
                workspace_root,
                2025,
                model_name,
                token,
            ),
            workspace_root,
        ),
        relative_target(
            build_state_filename(
                cfg,
                workspace_root,
                2025,
                model_name,
                token,
            ),
            workspace_root,
        ),
    ]
    result = run_snakemake_targets(
        targets,
        config_path,
        workspace_root,
        allowed_rules=["copy_config", "apply_learning_costs", "solve_network_myopic", "export_postsolve_learning_costs"],
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Synthetic single-path Snakemake smoke failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    outputs = validate_runtime_outputs(cfg, workspace_root, [(model_name, token)])
    summary = {
        "mode": "single",
        "targets": targets,
        "config": str(config_path),
        "synthetic_inputs": synthetic_inputs,
        "bootstrap_inputs": bootstrap_inputs,
        "outputs": outputs,
    }
    (root / "summary.single.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def run_mc_smoke(root, model_name, seeds):
    root.mkdir(parents=True, exist_ok=True)
    workspace_root = root / "workspace"
    prepare_workspace(workspace_root)
    populate_workspace_resources(workspace_root)
    write_workspace_learning_config(workspace_root, 0)
    config_path = build_synthetic_config(
        workspace_root,
        "mc",
        DEFAULT_SINGLE_MODEL,
        [model_name],
        seeds,
    )
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    pairs = [(model_name, format_learning_seed(model_name, seed)) for seed in seeds]
    synthetic_inputs = create_synthetic_brownfields(
        cfg,
        workspace_root,
        pairs,
    )
    bootstrap_inputs = create_synthetic_bootstrap_outputs(
        cfg,
        workspace_root,
        [model_name],
        root,
    )
    targets = []
    for _, token in pairs:
        targets.extend(
            [
                relative_target(
                    build_cost_log_filename(
                        cfg,
                        workspace_root,
                        2025,
                        model_name,
                        token,
                    ),
                    workspace_root,
                ),
                relative_target(
                    build_state_filename(
                        cfg,
                        workspace_root,
                        2025,
                        model_name,
                        token,
                    ),
                    workspace_root,
                ),
            ]
        )
    result = run_snakemake_targets(
        targets,
        config_path,
        workspace_root,
        allowed_rules=["copy_config", "apply_learning_costs", "solve_network_myopic", "export_postsolve_learning_costs"],
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Synthetic Monte Carlo Snakemake smoke failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    outputs = validate_runtime_outputs(cfg, workspace_root, pairs)
    divergence = validate_mc_divergence(cfg, workspace_root, model_name, seeds)
    summary = {
        "mode": "mc",
        "targets": targets,
        "config": str(config_path),
        "synthetic_inputs": synthetic_inputs,
        "bootstrap_inputs": bootstrap_inputs,
        "outputs": outputs,
        "divergence": divergence,
    }
    (root / "summary.mc.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def run_compact_cleanup_smoke(root, model_name, seed):
    root.mkdir(parents=True, exist_ok=True)
    workspace_root = root / "workspace"
    prepare_workspace(workspace_root)
    populate_workspace_resources(workspace_root)
    write_workspace_learning_config(workspace_root, seed)
    config_path = build_synthetic_config(
        workspace_root,
        "single",
        model_name,
        [],
        [],
    )
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["scenario"]["planning_horizons"] = [2020, 2025]
    cfg["learning"]["compact_outputs"]["enable"] = True
    cfg["learning"]["compact_outputs"]["cleanup_heavy_raws"] = True
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    token = format_learning_seed(model_name, seed)

    bootstrap_network = build_postnetwork_filename(cfg, workspace_root, 2020, model_name)
    bootstrap_network.parent.mkdir(parents=True, exist_ok=True)
    solve_mock_network(
        input_path=create_temp_network_file(root / "bootstrap_2020_input.nc", 2020),
        output_path=bootstrap_network,
    )
    bootstrap_cost = build_bootstrap_cost_log_filename(cfg, workspace_root, 2020, model_name)
    bootstrap_state = build_bootstrap_state_filename(cfg, workspace_root, 2020, model_name)
    write_mock_learning_outputs(bootstrap_cost, bootstrap_state, model_name, LEGACY_LEARNING_SEED, 2020)

    branch_brownfield = build_brownfield_filename(cfg, workspace_root, 2025, model_name, token)
    branch_learning = build_prenetwork_learning_filename(cfg, workspace_root, 2025, model_name, token)
    branch_postnetwork = build_postnetwork_filename(cfg, workspace_root, 2025, model_name, token)
    branch_brownfield.parent.mkdir(parents=True, exist_ok=True)
    branch_learning.parent.mkdir(parents=True, exist_ok=True)
    branch_postnetwork.parent.mkdir(parents=True, exist_ok=True)
    input_2025 = create_temp_network_file(root / "branch_2025_input.nc", 2025)
    shutil.copy2(input_2025, branch_brownfield)
    shutil.copy2(input_2025, branch_learning)
    solve_mock_network(input_2025, branch_postnetwork)

    branch_cost = build_cost_log_filename(cfg, workspace_root, 2025, model_name, token)
    branch_state = build_state_filename(cfg, workspace_root, 2025, model_name, token)
    write_mock_learning_outputs(branch_cost, branch_state, model_name, token, 2025)

    cleanup_marker = build_compact_cleanup_marker_path(cfg, workspace_root, model_name, token)
    result = run_snakemake_targets(
        [relative_target(cleanup_marker, workspace_root)],
        config_path,
        workspace_root,
        allowed_rules=["export_stochastic_run_bundle", "cleanup_stochastic_branch_raw_artifacts"],
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Synthetic compact-output smoke failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    compact_complete = build_compact_complete_path(cfg, workspace_root, model_name, token)
    if not compact_complete.exists():
        raise FileNotFoundError(f"Expected compact output marker missing: {compact_complete}")
    if not cleanup_marker.exists():
        raise FileNotFoundError(f"Expected cleanup marker missing: {cleanup_marker}")

    bundle_dir = compact_complete.parent
    expected_csvs = [
        "learning_costs.csv",
        "system_summary.csv",
        "generation_country_carrier.csv",
        "ac_energy_balance_country_carrier.csv",
        "capacity_country_carrier.csv",
        "battery_operations_country_year.csv",
        "deployment_country_carrier.csv",
        "deployment_node_carrier.csv",
        "power_emissions_country_carrier.csv",
        "sector_emissions.csv",
        "sector_demands_country.csv",
        "sector_total_demands_country.csv",
        "electricity_demand_country_sector.csv",
        "electricity_price_node_year.csv",
        "electricity_price_country_year.csv",
        "constraint_shadow_system_year.csv",
        "constraint_shadow_asset_year.csv",
        "lcoe_electricity_tech_year.csv",
        "run_manifest.json",
        "compact_complete.json",
    ]
    missing = [name for name in expected_csvs if not (bundle_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Compact bundle missing files: {missing}")

    for raw_path in (branch_brownfield, branch_learning, branch_postnetwork):
        if raw_path.exists():
            raise RuntimeError(f"Expected compact cleanup to remove {raw_path}")
    if not branch_cost.exists() or not branch_state.exists():
        raise RuntimeError("Compact cleanup removed solved learning outputs that should have been retained.")

    summary = {
        "mode": "compact_cleanup",
        "compact_complete": str(compact_complete),
        "cleanup_marker": str(cleanup_marker),
        "bundle_dir": str(bundle_dir),
    }
    (root / "summary.compact_cleanup.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def run_compact_failure_smoke(root, model_name, seed):
    root.mkdir(parents=True, exist_ok=True)
    workspace_root = root / "workspace"
    prepare_workspace(workspace_root)
    populate_workspace_resources(workspace_root)
    write_workspace_learning_config(workspace_root, seed)
    config_path = build_synthetic_config(
        workspace_root,
        "single",
        model_name,
        [],
        [],
    )
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["scenario"]["planning_horizons"] = [2020, 2025]
    cfg["learning"]["compact_outputs"]["enable"] = True
    cfg["learning"]["compact_outputs"]["cleanup_heavy_raws"] = True
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    token = format_learning_seed(model_name, seed)
    bootstrap_network = build_postnetwork_filename(cfg, workspace_root, 2020, model_name)
    bootstrap_network.parent.mkdir(parents=True, exist_ok=True)
    solve_mock_network(
        input_path=create_temp_network_file(root / "bootstrap_fail_2020_input.nc", 2020),
        output_path=bootstrap_network,
    )
    bootstrap_cost = build_bootstrap_cost_log_filename(cfg, workspace_root, 2020, model_name)
    bootstrap_state = build_bootstrap_state_filename(cfg, workspace_root, 2020, model_name)
    write_mock_learning_outputs(bootstrap_cost, bootstrap_state, model_name, LEGACY_LEARNING_SEED, 2020)

    branch_brownfield = build_brownfield_filename(cfg, workspace_root, 2025, model_name, token)
    branch_learning = build_prenetwork_learning_filename(cfg, workspace_root, 2025, model_name, token)
    branch_postnetwork = build_postnetwork_filename(cfg, workspace_root, 2025, model_name, token)
    input_2025 = create_temp_network_file(root / "branch_fail_2025_input.nc", 2025)
    branch_brownfield.parent.mkdir(parents=True, exist_ok=True)
    branch_learning.parent.mkdir(parents=True, exist_ok=True)
    branch_postnetwork.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_2025, branch_brownfield)
    shutil.copy2(input_2025, branch_learning)
    solve_mock_network(input_2025, branch_postnetwork)
    branch_cost = build_cost_log_filename(cfg, workspace_root, 2025, model_name, token)
    branch_state = build_state_filename(cfg, workspace_root, 2025, model_name, token)
    write_mock_learning_outputs(branch_cost, branch_state, model_name, token, 2025)

    compact_complete = build_compact_complete_path(cfg, workspace_root, model_name, token)
    env = os.environ.copy()
    env["LEARNING_COMPACT_FORCE_FAIL"] = "1"
    cmd = [
        sys.executable,
        "-m",
        "snakemake",
        "-j1",
        relative_target(compact_complete, workspace_root),
        "--configfile",
        str(config_path),
        "config.learning.yaml",
        "--rerun-trigger",
        "mtime",
        "--allowed-rules",
        "export_stochastic_run_bundle",
    ]
    result = subprocess.run(
        cmd,
        cwd=workspace_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        raise RuntimeError("Forced compact-output failure smoke unexpectedly succeeded.")
    if not branch_brownfield.exists() or not branch_learning.exists() or not branch_postnetwork.exists():
        raise RuntimeError("Compact-output failure smoke removed raw artifacts on failure.")
    summary = {
        "mode": "compact_failure",
        "stderr_tail": result.stderr[-1000:],
    }
    (root / "summary.compact_failure.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def run_cluster_wrapper_cleanup_smoke(root):
    root.mkdir(parents=True, exist_ok=True)
    job_root = root / "cluster_workdirs"
    job_root.mkdir(parents=True, exist_ok=True)
    success_runner = root / "success_runner.sh"
    failure_runner = root / "failure_runner.sh"
    success_runner.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    failure_runner.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
    success_runner.chmod(0o755)
    failure_runner.chmod(0o755)

    base_env = os.environ.copy()
    base_env["LEARNING_SCENARIO_NAME"] = "wrapper_smoke"

    success_env = base_env.copy()
    success_env["LEARNING_CLUSTER_JOB_RUNNER"] = str(success_runner)
    success = subprocess.run(
        [
            "bash",
            str(ENERGYMOD_ROOT / "scripts" / "learning" / "run_learning_cluster_job.sh"),
            DEFAULT_SINGLE_MODEL,
            "7",
            str(job_root),
        ],
        cwd=ENERGYMOD_ROOT,
        env=success_env,
        text=True,
        capture_output=True,
        check=False,
    )
    success_dir = job_root / f"energymod_wrapper_smoke_{DEFAULT_SINGLE_MODEL}_s0007"
    if success.returncode != 0:
        raise RuntimeError(f"Wrapper cleanup smoke success path failed:\n{success.stderr}")
    if success_dir.exists():
        raise RuntimeError(f"Expected wrapper success path to remove {success_dir}")

    failure_env = base_env.copy()
    failure_env["LEARNING_CLUSTER_JOB_RUNNER"] = str(failure_runner)
    failure = subprocess.run(
        [
            "bash",
            str(ENERGYMOD_ROOT / "scripts" / "learning" / "run_learning_cluster_job.sh"),
            DEFAULT_SINGLE_MODEL,
            "8",
            str(job_root),
        ],
        cwd=ENERGYMOD_ROOT,
        env=failure_env,
        text=True,
        capture_output=True,
        check=False,
    )
    failure_dir = job_root / f"energymod_wrapper_smoke_{DEFAULT_SINGLE_MODEL}_s0008"
    if failure.returncode == 0:
        raise RuntimeError("Wrapper cleanup smoke failure path unexpectedly succeeded.")
    if not failure_dir.exists():
        raise RuntimeError(f"Expected wrapper failure path to keep {failure_dir}")

    summary = {
        "mode": "cluster_wrapper_cleanup",
        "success_dir_removed": str(success_dir),
        "failure_dir_retained": str(failure_dir),
    }
    (root / "summary.cluster_wrapper_cleanup.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def create_temp_network_file(path, planning_year):
    path.parent.mkdir(parents=True, exist_ok=True)
    build_mock_network(planning_year).export_to_netcdf(path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="/tmp/energymod_snakemake_learning_smoke")
    parser.add_argument("--single-model", default=DEFAULT_SINGLE_MODEL)
    parser.add_argument("--single-seed", type=int, default=7)
    parser.add_argument("--mc-model", default=DEFAULT_SINGLE_MODEL)
    parser.add_argument("--mc-seeds", nargs="+", type=int, default=[0, 1])
    args = parser.parse_args()
    if args.single_model not in SHORTLIST_MODELS:
        raise ValueError(f"Unsupported single-model smoke target: {args.single_model}")
    if args.mc_model not in SHORTLIST_MODELS:
        raise ValueError(f"Unsupported MC smoke target: {args.mc_model}")

    output_root = Path(args.output_root).resolve()
    single_summary = run_single_smoke(output_root / "single", args.single_model, args.single_seed)
    mc_summary = run_mc_smoke(output_root / "mc", args.mc_model, args.mc_seeds)
    compact_cleanup_summary = run_compact_cleanup_smoke(
        output_root / "compact_cleanup",
        args.single_model,
        args.single_seed,
    )
    compact_failure_summary = run_compact_failure_smoke(
        output_root / "compact_failure",
        args.single_model,
        args.single_seed + 1,
    )
    cluster_wrapper_cleanup_summary = run_cluster_wrapper_cleanup_smoke(
        output_root / "cluster_wrapper_cleanup"
    )
    summary = {
        "single": single_summary,
        "mc": mc_summary,
        "compact_cleanup": compact_cleanup_summary,
        "compact_failure": compact_failure_summary,
        "cluster_wrapper_cleanup": cluster_wrapper_cleanup_summary,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"LEARNING_SNAKEMAKE_SMOKE_OK {summary_path}")


if __name__ == "__main__":
    main()
