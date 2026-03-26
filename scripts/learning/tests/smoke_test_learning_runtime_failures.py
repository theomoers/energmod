#!/usr/bin/env python3
"""Failure-path smoke tests for energymod learning runtime."""

import argparse
import copy
import json
from pathlib import Path

from learning_runtime_smoke_utils import (
    ENERGYMOD_ROOT,
    add_mock_brownfield,
    build_mock_network,
    run_apply_learning,
    run_export_postsolve,
    solve_mock_network,
    write_learning_config,
)


def expect_failure(label, fn, contains):
    try:
        fn()
    except Exception as exc:
        message = str(exc)
        if contains not in message:
            raise RuntimeError(
                f"{label} failed with unexpected message.\nExpected to contain: {contains}\nActual: {message}"
            ) from exc
        return {"label": label, "status": "expected_failure", "message": message}
    raise RuntimeError(f"{label} unexpectedly succeeded")


def make_manifest_copy(root, transform):
    root.mkdir(parents=True, exist_ok=True)
    source = ENERGYMOD_ROOT / "data" / "learning-data" / "manifest.json"
    source_root = source.parent
    manifest = json.loads(source.read_text(encoding="utf-8"))
    manifest = copy.deepcopy(manifest)
    for tech, rel in list((manifest.get("historical_files") or {}).items()):
        if rel and not Path(rel).is_absolute():
            manifest["historical_files"][tech] = str((source_root / rel).resolve())
    legacy = manifest.get("legacy_curve") or {}
    for window_info in (legacy.get("training_windows") or {}).values():
        if window_info.get("params_csv") and not Path(window_info["params_csv"]).is_absolute():
            window_info["params_csv"] = str((source_root / window_info["params_csv"]).resolve())
    training_windows = ((manifest.get("stochastic_forecast") or {}).get("training_windows") or {})
    for window_info in training_windows.values():
        for model_info in (window_info.get("models") or {}).values():
            if model_info.get("artifact_dir") and not Path(model_info["artifact_dir"]).is_absolute():
                model_info["artifact_dir"] = str((source_root / model_info["artifact_dir"]).resolve())
            if model_info.get("initial_state") and not Path(model_info["initial_state"]).is_absolute():
                model_info["initial_state"] = str((source_root / model_info["initial_state"]).resolve())
    transform(manifest, root)
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def copy_model_setup(root, model_name, manifest_path=None):
    root.mkdir(parents=True, exist_ok=True)
    learning_config = root / "config.learning.yaml"
    write_learning_config(learning_config, model_name, manifest_path=manifest_path)
    costs_2020 = ENERGYMOD_ROOT / "resources" / "Earth_200" / "costs_2020.csv"
    costs_2025 = ENERGYMOD_ROOT / "resources" / "Earth_200" / "costs_2025.csv"
    return {
        "learning_config": learning_config,
        "costs_2020": costs_2020,
        "costs_2025": costs_2025,
    }


def prepare_2020_commit(root, model_name, manifest_path=None):
    files = copy_model_setup(root, model_name, manifest_path)
    base_2020 = root / "network_2020_base.nc"
    learned_2020 = root / "network_2020_learned.nc"
    solved_2020 = root / "network_2020_solved.nc"
    cost_log_2020 = root / "cost_log_2020.csv"
    cost_log_solved_2020 = root / "cost_log_solved_2020.csv"
    state_proposed_2020 = root / "state_proposed_2020.json"
    state_committed_2020 = root / "state_committed_2020.json"

    build_mock_network(2020).export_to_netcdf(base_2020)
    run_apply_learning(
        base_2020,
        learned_2020,
        cost_log_2020,
        state_proposed_2020,
        files["learning_config"],
        files["costs_2020"],
        2020,
        [2020, 2025],
        model_name,
    )
    solve_mock_network(learned_2020, solved_2020)
    run_export_postsolve(
        solved_2020,
        cost_log_2020,
        state_proposed_2020,
        files["learning_config"],
        files["costs_2020"],
        cost_log_solved_2020,
        state_committed_2020,
    )
    base_2025 = root / "network_2025_base.nc"
    brownfield_2025 = root / "network_2025_brownfield.nc"
    build_mock_network(2025).export_to_netcdf(base_2025)
    add_mock_brownfield(base_2025, solved_2020, brownfield_2025, 2025)

    return {
        **files,
        "solved_2020": solved_2020,
        "state_committed_2020": state_committed_2020,
        "brownfield_2025": brownfield_2025,
    }


def write_sparse_history_csv(path):
    rows = [
        {"Year": 2018, "Cumulative production, GW": 10.0},
        {"Year": 2019, "Cumulative production, GW": 12.0},
        {"Year": 2020, "Cumulative production, GW": 14.0},
    ]
    content = "Year,Cumulative production, GW\n" + "\n".join(
        f"{row['Year']},{row['Cumulative production, GW']}" for row in rows
    ) + "\n"
    path.write_text(content, encoding="utf-8")


def test_missing_artifact_field(root):
    def transform(manifest, _root):
        del manifest["stochastic_forecast"]["training_windows"]["origin_cutoff"]["models"]["shared_state_bayesian_regime_wright"]["initial_state"]

    manifest_path = make_manifest_copy(root, transform)
    files = copy_model_setup(root, "shared_state_bayesian_regime_wright", manifest_path)
    base_2020 = root / "network_2020_base.nc"
    build_mock_network(2020).export_to_netcdf(base_2020)

    return expect_failure(
        "missing_artifact_field",
        lambda: run_apply_learning(
            network_path=base_2020,
            output_network_path=root / "learned.nc",
            cost_log_path=root / "cost_log.csv",
            state_proposed_path=root / "state.json",
            learning_config=files["learning_config"],
            costs_file=files["costs_2020"],
            planning_year=2020,
            planning_horizons=[2020, 2025],
            learning_model="shared_state_bayesian_regime_wright",
        ),
        "initial_state",
    )


def test_missing_previous_state(root):
    prepared = prepare_2020_commit(root, "way_fixed_rho_benchmark_035")
    return expect_failure(
        "missing_previous_state",
        lambda: run_apply_learning(
            network_path=prepared["brownfield_2025"],
            output_network_path=root / "learned_2025.nc",
            cost_log_path=root / "cost_log_2025.csv",
            state_proposed_path=root / "state_proposed_2025.json",
            learning_config=prepared["learning_config"],
            costs_file=prepared["costs_2025"],
            planning_year=2025,
            planning_horizons=[2020, 2025],
            learning_model="way_fixed_rho_benchmark_035",
            prev_network_path=prepared["solved_2020"],
            prev_state_path=None,
        ),
        "requires previous committed learning state",
    )


def test_non_base_learning_rate(root):
    files = copy_model_setup(root, "correlated_geometric_random_walk")
    base_2020 = root / "network_2020_base.nc"
    build_mock_network(2020).export_to_netcdf(base_2020)
    return expect_failure(
        "non_base_learning_rate",
        lambda: run_apply_learning(
            network_path=base_2020,
            output_network_path=root / "learned.nc",
            cost_log_path=root / "cost_log.csv",
            state_proposed_path=root / "state.json",
            learning_config=files["learning_config"],
            costs_file=files["costs_2020"],
            planning_year=2020,
            planning_horizons=[2020, 2025],
            learning_model="correlated_geometric_random_walk",
            learning_rate="s.m1.1",
        ),
        "Non-base learning_rate scenarios are only supported for legacy_curve",
    )


def test_missing_block_history(root):
    prepared = prepare_2020_commit(root, "way_fixed_rho_benchmark_035")
    bad_state = json.loads(Path(prepared["state_committed_2020"]).read_text(encoding="utf-8"))
    for tech in ("solar_power", "onwind_power", "battery_energy"):
        tech_history = ((bad_state.get("capacity_history", {}) or {}).get(tech, {})) or {}
        tech_history.pop("2020", None)
        bad_state.setdefault("capacity_history", {})[tech] = tech_history
    bad_state_path = root / "bad_state_committed_2020.json"
    bad_state_path.write_text(json.dumps(bad_state, indent=2, sort_keys=True), encoding="utf-8")
    learned_2025 = root / "learned_2025.nc"
    cost_log_2025 = root / "cost_log_2025.csv"
    state_proposed_2025 = root / "state_proposed_2025.json"
    solved_2025 = root / "solved_2025.nc"
    run_apply_learning(
        network_path=prepared["brownfield_2025"],
        output_network_path=learned_2025,
        cost_log_path=cost_log_2025,
        state_proposed_path=state_proposed_2025,
        learning_config=prepared["learning_config"],
        costs_file=prepared["costs_2025"],
        planning_year=2025,
        planning_horizons=[2020, 2025],
        learning_model="way_fixed_rho_benchmark_035",
        prev_network_path=prepared["solved_2020"],
        prev_state_path=bad_state_path,
    )
    solve_mock_network(learned_2025, solved_2025)
    return expect_failure(
        "missing_block_history",
        lambda: run_export_postsolve(
            solved_network_path=solved_2025,
            base_cost_log_path=cost_log_2025,
            proposed_state_path=state_proposed_2025,
            learning_config=prepared["learning_config"],
            costs_file=prepared["costs_2025"],
            output_cost_log_path=root / "cost_log_solved_2025.csv",
            output_state_committed_path=root / "state_committed_2025.json",
        ),
        "Committed cumulative capacity history",
    )


def test_malformed_shared_state(root):
    prepared = prepare_2020_commit(root, "shared_state_bayesian_regime_wright")
    bad_state = json.loads(Path(prepared["state_committed_2020"]).read_text(encoding="utf-8"))
    bad_state["shared_regime_state"] = {"current_regime": 1}
    bad_state_path = root / "bad_state_committed_2020.json"
    bad_state_path.write_text(json.dumps(bad_state, indent=2, sort_keys=True), encoding="utf-8")
    return expect_failure(
        "malformed_shared_state",
        lambda: run_apply_learning(
            network_path=prepared["brownfield_2025"],
            output_network_path=root / "learned_2025.nc",
            cost_log_path=root / "cost_log_2025.csv",
            state_proposed_path=root / "state_proposed_2025.json",
            learning_config=prepared["learning_config"],
            costs_file=prepared["costs_2025"],
            planning_year=2025,
            planning_horizons=[2020, 2025],
            learning_model="shared_state_bayesian_regime_wright",
            prev_network_path=prepared["solved_2020"],
            prev_state_path=bad_state_path,
        ),
        "initial_regime_probs",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="/tmp/energymod_mock_learning_failures")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    results = [
        test_missing_artifact_field(output_root / "missing_artifact_field"),
        test_missing_previous_state(output_root / "missing_previous_state"),
        test_non_base_learning_rate(output_root / "non_base_learning_rate"),
        test_missing_block_history(output_root / "missing_block_history"),
        test_malformed_shared_state(output_root / "malformed_shared_state"),
    ]
    summary_path = output_root / "failure_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"LEARNING_RUNTIME_FAILURE_SMOKE_OK {summary_path}")


if __name__ == "__main__":
    main()
