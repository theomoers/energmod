#!/usr/bin/env python3
"""Utilities for lightweight energymod learning-runtime smoke tests."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pypsa
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
LEARNING_ROOT = SCRIPT_DIR.parent
SCRIPTS_ROOT = LEARNING_ROOT.parent
ENERGYMOD_ROOT = SCRIPTS_ROOT.parent

if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import add_brownfield as add_brownfield_module  # noqa: E402
import learning.apply_learning_costs as apply_learning_costs_module  # noqa: E402
import learning.export_postsolve_learning_costs as export_postsolve_learning_costs_module  # noqa: E402


SHORTLIST_MODELS = [
    "shared_state_bayesian_regime_wright",
    "way_fixed_rho_benchmark_035",
    "correlated_geometric_random_walk",
]

ALL_RUNTIME_MODELS = ["legacy_curve", *SHORTLIST_MODELS]
LEGACY_LEARNING_SEED = "deterministic"


class DotDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def make_mock_snakemake(rule, inputs, outputs, wildcards, params):
    log_path = str(Path("/tmp") / "energymod_mock_logs" / f"{rule}.log")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        rule=rule,
        input=DotDict(inputs),
        output=DotDict(outputs),
        wildcards=DotDict(wildcards),
        params=DotDict(params),
        config={"logging": {"level": "INFO"}},
        log=DotDict({0: log_path, "python": log_path}),
    )


def write_learning_config(output_path, selected_model, manifest_path=None, seed=0, sample_mode="single_draw"):
    cfg_path = ENERGYMOD_ROOT / "config.learning.yaml"
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    payload["learning"]["enabled"] = True
    payload["learning"]["engine"] = (
        "legacy_curve" if selected_model == "legacy_curve" else "stochastic_forecast"
    )
    payload["learning"]["selected_model"] = selected_model
    payload["learning"]["artifact_manifest"] = str(
        (manifest_path or ENERGYMOD_ROOT / "data" / "learning-data" / "manifest.json").resolve()
    )
    payload["learning"]["training_window"] = "origin_cutoff"
    payload["learning"]["training_window_origin_year"] = 2020
    payload["learning"]["sample_mode"] = sample_mode
    payload["learning"]["seed"] = int(seed)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def format_learning_seed(model_name, seed):
    if model_name == "legacy_curve":
        return LEGACY_LEARNING_SEED
    return f"s{int(seed):04d}"


def build_mock_network(planning_year):
    n = pypsa.Network()
    snapshots = pd.date_range("2000-01-01 00:00", periods=24, freq="h")
    n.set_snapshots(snapshots)

    n.add("Carrier", "AC")
    n.add("Carrier", "solar")
    n.add("Carrier", "onwind")
    n.add("Carrier", "battery")
    n.add("Bus", "bus0", carrier="AC")
    n.buses.loc["bus0", "location"] = "mock-location"
    n.buses.loc["bus0", "country"] = "GL"

    n.add("Load", "load", bus="bus0", p_set=100.0)
    n.loads_t.p_set = pd.DataFrame({"load": [100.0] * len(snapshots)}, index=snapshots)

    suffix = str(planning_year)
    n.add(
        "Generator",
        f"solar_{suffix}",
        bus="bus0",
        carrier="solar",
        build_year=planning_year,
        lifetime=35,
        p_nom=0.0,
        p_nom_extendable=True,
        p_nom_min=10.0,
        p_nom_max=400.0,
        capital_cost=45000.0,
        marginal_cost=0.0,
    )
    n.add(
        "Generator",
        f"wind_{suffix}",
        bus="bus0",
        carrier="onwind",
        build_year=planning_year,
        lifetime=27,
        p_nom=0.0,
        p_nom_extendable=True,
        p_nom_min=10.0,
        p_nom_max=400.0,
        capital_cost=180000.0,
        marginal_cost=0.0,
    )
    n.generators_t.p_max_pu = pd.DataFrame(
        {
            f"solar_{suffix}": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.1,
                0.3,
                0.6,
                0.8,
                0.9,
                1.0,
                1.0,
                0.9,
                0.8,
                0.6,
                0.4,
                0.2,
                0.1,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            f"wind_{suffix}": [0.35] * len(snapshots),
        },
        index=snapshots,
    )
    n.add(
        "Store",
        f"battery_{suffix}",
        bus="bus0",
        carrier="battery",
        build_year=planning_year,
        lifetime=20,
        e_nom=0.0,
        e_nom_extendable=True,
        e_nom_min=10.0,
        e_nom_max=400.0,
        capital_cost=9000.0,
        marginal_cost=0.0,
        e_cyclic=True,
        e_initial=0.0,
        standing_loss=0.0,
    )
    n.add(
        "Link",
        f"dummy_link_{suffix}",
        bus0="bus0",
        bus1="bus0",
        carrier="AC",
        p_nom=0.0,
        p_nom_extendable=False,
        efficiency=1.0,
        marginal_cost=0.0,
        capital_cost=0.0,
    )
    return n


def solve_mock_network(input_path, output_path, solver_name="glpk"):
    n = pypsa.Network(input_path)
    status, condition = n.optimize(solver_name=solver_name)
    if str(status).lower() != "ok":
        raise RuntimeError(
            f"Mock network solve failed for {input_path.name}: status={status}, condition={condition}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n.export_to_netcdf(output_path)
    return {
        "status": str(status),
        "condition": str(condition),
        "objective": float(n.objective),
    }


def run_apply_learning(
    network_path,
    output_network_path,
    cost_log_path,
    state_proposed_path,
    learning_config,
    costs_file,
    planning_year,
    planning_horizons,
    learning_model,
    learning_rate="base",
    learning_seed=None,
    prev_network_path=None,
    prev_state_path=None,
):
    snakemake = make_mock_snakemake(
        rule="apply_learning_costs",
        inputs={
            "network": str(network_path),
            "learning_config": str(learning_config),
            "costs": str(costs_file),
            "network_p": str(prev_network_path) if prev_network_path else None,
            "prev_state": str(prev_state_path) if prev_state_path else None,
            "anchor_network": None,
        },
        outputs={
            "network": str(output_network_path),
            "cost_log": str(cost_log_path),
            "state_proposed": str(state_proposed_path),
        },
        wildcards={
            "planning_horizons": str(planning_year),
            "learning_rate": learning_rate,
            "learning_model": learning_model,
            "learning_seed": learning_seed or format_learning_seed(learning_model, 0),
        },
        params={"planning_horizons": planning_horizons},
    )
    apply_learning_costs_module.main(snakemake)


def run_export_postsolve(
    solved_network_path,
    base_cost_log_path,
    proposed_state_path,
    learning_config,
    costs_file,
    output_cost_log_path,
    output_state_committed_path,
):
    snakemake = make_mock_snakemake(
        rule="export_postsolve_learning_costs",
        inputs={
            "network": str(solved_network_path),
            "base_cost_log": str(base_cost_log_path),
            "proposed_state": str(proposed_state_path),
            "learning_config": str(learning_config),
            "costs": str(costs_file),
        },
        outputs={
            "cost_log": str(output_cost_log_path),
            "state_committed": str(output_state_committed_path),
        },
        wildcards={},
        params={},
    )
    export_postsolve_learning_costs_module.main(snakemake)


def validate_cost_log(path, expected_model):
    df = pd.read_csv(path)
    required = {"solar_power", "onwind_power", "battery_energy"}
    missing = required - set(df["technology"].tolist())
    if missing:
        raise RuntimeError(f"Cost log {path} missing technologies: {sorted(missing)}")
    for col in (
        "selected_model",
        "engine",
        "seed",
        "learning_seed",
        "runtime_conditioning",
        "manifest_schema_version",
        "manifest_sha256",
        "battery_power_treatment",
    ):
        if col not in df.columns:
            raise RuntimeError(f"Cost log {path} missing metadata column {col}")
    models = set(df["selected_model"].dropna().astype(str).tolist())
    if expected_model not in models:
        raise RuntimeError(f"Cost log {path} missing expected model {expected_model}")


def validate_committed_state(path, expected_model, expected_year):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("selected_model") != expected_model:
        raise RuntimeError(f"Committed state {path} selected_model mismatch")
    if int(payload.get("last_applied_year", -1)) != int(expected_year):
        raise RuntimeError(f"Committed state {path} year mismatch")
    if payload.get("battery_power_treatment") != "deterministic_default_costs":
        raise RuntimeError(f"Committed state {path} lost battery_power treatment")
    for key in ("engine", "seed", "runtime_conditioning", "manifest_schema_version", "manifest_sha256"):
        if key not in payload:
            raise RuntimeError(f"Committed state {path} missing metadata field {key}")
    if "learning_seed" not in payload:
        raise RuntimeError(f"Committed state {path} missing metadata field learning_seed")
    if expected_model == "legacy_curve":
        return
    tech_states = payload.get("technology_states", {}) or {}
    for tech in ("solar_power", "onwind_power", "battery_energy"):
        if tech not in tech_states:
            raise RuntimeError(f"Committed state {path} missing technology state for {tech}")


def validate_lagged_cost_carry_forward(previous_committed_cost_log, next_input_cost_log, label):
    prev_df = pd.read_csv(previous_committed_cost_log).sort_values("technology").reset_index(drop=True)
    next_df = pd.read_csv(next_input_cost_log).sort_values("technology").reset_index(drop=True)
    compare_cols = ["technology", "c_overnight"]
    prev_view = prev_df[compare_cols].copy()
    next_view = next_df[compare_cols].copy()
    merged = prev_view.merge(next_view, on="technology", suffixes=("_prev", "_next"))
    for _, row in merged.iterrows():
        if abs(float(row["c_overnight_prev"]) - float(row["c_overnight_next"])) > 1.0e-9:
            raise RuntimeError(
                f"{label}: c_overnight carry-forward mismatch for {row['technology']}: "
                f"{row['c_overnight_prev']} vs {row['c_overnight_next']}"
            )


def validate_historical_bootstrap_cost_log(cost_log_path, learning_config, costs_file, current_year):
    learning_cfg = apply_learning_costs_module.load_config_learning(str(learning_config))
    apply_learning_costs_module.load_learning_manifest(learning_cfg, str(learning_config))
    params = apply_learning_costs_module.load_learning_params(
        apply_learning_costs_module.get_legacy_params_path(learning_cfg)
    )
    expected = apply_learning_costs_module.calculate_learning_costs(
        params=params,
        learning_cfg=learning_cfg,
        current_year=int(current_year),
        planning_horizons=[2020, 2025, 2030],
        prev_network_path=None,
        costs_file=str(costs_file),
        global_scale_factors=apply_learning_costs_module.get_global_scale_factors(learning_cfg),
        wacc_dict=None,
    )
    actual_df = pd.read_csv(cost_log_path)
    actual = {
        row["technology"]: (
            float(row["cumulative_capacity_GW"]),
            float(row["c_overnight"]),
        )
        for _, row in actual_df.iterrows()
    }
    for tech, expected_values in expected.items():
        if tech not in actual:
            raise RuntimeError(f"Historical bootstrap cost log {cost_log_path} is missing {tech}")
        expected_capacity = float(expected_values["cumulative_capacity_GW"])
        expected_cost = float(expected_values["c_overnight"])
        actual_capacity, actual_cost = actual[tech]
        if abs(expected_capacity - actual_capacity) > 1.0e-9:
            raise RuntimeError(
                f"{cost_log_path}: historical bootstrap capacity mismatch for {tech}: "
                f"{actual_capacity} vs {expected_capacity}"
            )
        if abs(expected_cost - actual_cost) > 1.0e-9:
            raise RuntimeError(
                f"{cost_log_path}: historical bootstrap c_overnight mismatch for {tech}: "
                f"{actual_cost} vs {expected_cost}"
            )


def add_mock_brownfield(current_network_path, previous_solved_network_path, output_path, year):
    n_current = pypsa.Network(current_network_path)
    n_prev = pypsa.Network(previous_solved_network_path)
    add_brownfield_module.snakemake = SimpleNamespace(
        params=SimpleNamespace(
            threshold_capacity=0.0,
            H2_retrofit=False,
            H2_retrofit_capacity_per_CH4=1.0,
        )
    )
    add_brownfield_module.add_brownfield(n_current, n_prev, year)
    n_current.export_to_netcdf(output_path)


def run_model_smoke(model_name, root, seed=0):
    model_root = root / model_name
    model_root.mkdir(parents=True, exist_ok=True)

    learning_config = model_root / "config.learning.yaml"
    write_learning_config(learning_config, model_name, seed=seed)

    costs_2020 = ENERGYMOD_ROOT / "resources" / "Earth_200" / "costs_2020.csv"
    costs_2025 = ENERGYMOD_ROOT / "resources" / "Earth_200" / "costs_2025.csv"
    costs_2030 = ENERGYMOD_ROOT / "resources" / "Earth_200" / "costs_2030.csv"
    for required_path in (costs_2020, costs_2025, costs_2030):
        if not required_path.exists():
            raise FileNotFoundError(f"Required cost file missing: {required_path}")

    base_2020 = model_root / "network_2020_base.nc"
    learned_2020 = model_root / "network_2020_learned.nc"
    solved_2020 = model_root / "network_2020_solved.nc"
    cost_log_2020 = model_root / "cost_log_2020.csv"
    cost_log_solved_2020 = model_root / "cost_log_solved_2020.csv"
    state_proposed_2020 = model_root / "state_proposed_2020.json"
    state_committed_2020 = model_root / "state_committed_2020.json"

    base_2025 = model_root / "network_2025_base.nc"
    brownfield_2025 = model_root / "network_2025_brownfield.nc"
    learned_2025 = model_root / "network_2025_learned.nc"
    solved_2025 = model_root / "network_2025_solved.nc"
    cost_log_2025 = model_root / "cost_log_2025.csv"
    cost_log_solved_2025 = model_root / "cost_log_solved_2025.csv"
    state_proposed_2025 = model_root / "state_proposed_2025.json"
    state_committed_2025 = model_root / "state_committed_2025.json"

    build_mock_network(2020).export_to_netcdf(base_2020)
    run_apply_learning(
        network_path=base_2020,
        output_network_path=learned_2020,
        cost_log_path=cost_log_2020,
        state_proposed_path=state_proposed_2020,
        learning_config=learning_config,
        costs_file=costs_2020,
        planning_year=2020,
        planning_horizons=[2020, 2025],
        learning_model=model_name,
        learning_seed=format_learning_seed(model_name, seed),
    )
    solve_2020 = solve_mock_network(learned_2020, solved_2020)
    run_export_postsolve(
        solved_network_path=solved_2020,
        base_cost_log_path=cost_log_2020,
        proposed_state_path=state_proposed_2020,
        learning_config=learning_config,
        costs_file=costs_2020,
        output_cost_log_path=cost_log_solved_2020,
        output_state_committed_path=state_committed_2020,
    )
    validate_cost_log(cost_log_2020, model_name)
    validate_historical_bootstrap_cost_log(cost_log_2020, learning_config, costs_2020, 2020)
    validate_committed_state(state_committed_2020, model_name, 2020)

    build_mock_network(2025).export_to_netcdf(base_2025)
    add_mock_brownfield(base_2025, solved_2020, brownfield_2025, 2025)
    run_apply_learning(
        network_path=brownfield_2025,
        output_network_path=learned_2025,
        cost_log_path=cost_log_2025,
        state_proposed_path=state_proposed_2025,
        learning_config=learning_config,
        costs_file=costs_2025,
        planning_year=2025,
        planning_horizons=[2020, 2025],
        learning_model=model_name,
        learning_seed=format_learning_seed(model_name, seed),
        prev_network_path=solved_2020,
        prev_state_path=state_committed_2020,
    )
    solve_2025 = solve_mock_network(learned_2025, solved_2025)
    run_export_postsolve(
        solved_network_path=solved_2025,
        base_cost_log_path=cost_log_2025,
        proposed_state_path=state_proposed_2025,
        learning_config=learning_config,
        costs_file=costs_2025,
        output_cost_log_path=cost_log_solved_2025,
        output_state_committed_path=state_committed_2025,
    )
    validate_cost_log(cost_log_2025, model_name)
    validate_historical_bootstrap_cost_log(cost_log_2025, learning_config, costs_2025, 2025)
    validate_committed_state(state_committed_2025, model_name, 2025)

    base_2030 = model_root / "network_2030_base.nc"
    brownfield_2030 = model_root / "network_2030_brownfield.nc"
    learned_2030 = model_root / "network_2030_learned.nc"
    solved_2030 = model_root / "network_2030_solved.nc"
    cost_log_2030 = model_root / "cost_log_2030.csv"
    cost_log_solved_2030 = model_root / "cost_log_solved_2030.csv"
    state_proposed_2030 = model_root / "state_proposed_2030.json"
    state_committed_2030 = model_root / "state_committed_2030.json"

    build_mock_network(2030).export_to_netcdf(base_2030)
    add_mock_brownfield(base_2030, solved_2025, brownfield_2030, 2030)
    run_apply_learning(
        network_path=brownfield_2030,
        output_network_path=learned_2030,
        cost_log_path=cost_log_2030,
        state_proposed_path=state_proposed_2030,
        learning_config=learning_config,
        costs_file=costs_2030,
        planning_year=2030,
        planning_horizons=[2020, 2025, 2030],
        learning_model=model_name,
        learning_seed=format_learning_seed(model_name, seed),
        prev_network_path=solved_2025,
        prev_state_path=state_committed_2025,
    )
    solve_2030 = solve_mock_network(learned_2030, solved_2030)
    run_export_postsolve(
        solved_network_path=solved_2030,
        base_cost_log_path=cost_log_2030,
        proposed_state_path=state_proposed_2030,
        learning_config=learning_config,
        costs_file=costs_2030,
        output_cost_log_path=cost_log_solved_2030,
        output_state_committed_path=state_committed_2030,
    )
    validate_cost_log(cost_log_2030, model_name)
    validate_committed_state(state_committed_2030, model_name, 2030)
    validate_lagged_cost_carry_forward(cost_log_solved_2025, cost_log_2030, f"{model_name} 2030 input")

    learned_2025_network = pypsa.Network(learned_2025)
    for carrier in ("solar", "onwind"):
        if carrier not in set(learned_2025_network.generators.carrier):
            raise RuntimeError(f"Learned 2025 network for {model_name} lost carrier {carrier}")
    if "battery" not in set(learned_2025_network.stores.carrier):
        raise RuntimeError(f"Learned 2025 network for {model_name} lost battery store")

    result = {
        "model_name": model_name,
        "solve_2020": solve_2020,
        "solve_2025": solve_2025,
        "solve_2030": solve_2030,
        "artifacts": {
            "learned_2020": str(learned_2020),
            "solved_2020": str(solved_2020),
            "cost_log_2020": str(cost_log_2020),
            "state_committed_2020": str(state_committed_2020),
            "brownfield_2025": str(brownfield_2025),
            "learned_2025": str(learned_2025),
            "solved_2025": str(solved_2025),
            "cost_log_2025": str(cost_log_2025),
            "state_committed_2025": str(state_committed_2025),
            "learned_2030": str(learned_2030),
            "solved_2030": str(solved_2030),
            "cost_log_2030": str(cost_log_2030),
            "state_committed_2030": str(state_committed_2030),
        },
    }
    (model_root / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result
