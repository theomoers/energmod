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
        source = ENERGYMOD_ROOT / "resources" / "Earth_200" / f"costs_{year}.csv"
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
    targets = [
        str(
            build_cost_log_filename(
                cfg,
                workspace_root,
                2025,
                model_name,
                token,
            ).relative_to(workspace_root)
        ),
        str(
            build_state_filename(
                cfg,
                workspace_root,
                2025,
                model_name,
                token,
            ).relative_to(workspace_root)
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
    targets = []
    for _, token in pairs:
        targets.extend(
            [
                str(
                    build_cost_log_filename(
                        cfg,
                        workspace_root,
                        2025,
                        model_name,
                        token,
                    ).relative_to(workspace_root)
                ),
                str(
                    build_state_filename(
                        cfg,
                        workspace_root,
                        2025,
                        model_name,
                        token,
                    ).relative_to(workspace_root)
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
        "outputs": outputs,
        "divergence": divergence,
    }
    (root / "summary.mc.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


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
    summary = {
        "single": single_summary,
        "mc": mc_summary,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"LEARNING_SNAKEMAKE_SMOKE_OK {summary_path}")


if __name__ == "__main__":
    main()
