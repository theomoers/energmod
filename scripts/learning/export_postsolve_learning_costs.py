#!/usr/bin/env python3
"""Export same-year applied learning costs plus post-solve diagnostics.

The compact learning CSVs should reflect the costs that were actually applied
to the solved network for the current planning horizon. This script therefore
uses the base/applied cost log as the primary output and appends explicit
post-solve diagnostic columns that describe the committed next state.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa


SCRIPTS_DIR = Path.cwd() / "scripts"
if (SCRIPTS_DIR / "_helpers.py").exists():
    sys.path.insert(0, str(SCRIPTS_DIR))

from _helpers import mock_snakemake
from learning.apply_learning_costs import (
    build_learning_base_capacity_map,
    convert_to_capital_cost,
    extract_capacity_from_network,
    get_battery_phi_for_block,
    get_battery_energy_bos_multiplier,
    get_learning_engine,
    get_manifest_historical_datafile,
    get_selected_learning_model,
    get_global_scale_factors,
    get_tech_mapping,
    load_config_learning,
    load_learning_manifest,
    resolve_runtime_seed,
    update_stochastic_runtime_state,
    validate_runtime_contract,
)
from learning.deployment_constraints import (
    build_deployment_constraint_block_paths,
    get_deployment_constraint_formulation,
    get_deployment_constraint_cfg,
    get_constraint_basis_unit,
    summarize_realized_deployment_wedge_rows,
    summarize_realized_block_additions,
)
from learning.learning_data_io import load_historical_capacity


logger = logging.getLogger(__name__)


def _cleanup_superseded_learning_inputs(base_cost_log, proposed_state, current_year):
    """Drop branch-only intermediates once solved outputs are written."""
    if int(current_year) <= 2025:
        return
    for candidate in [base_cost_log, proposed_state]:
        path = Path(candidate)
        if path.exists():
            path.unlink()
            logger.info("Removed superseded learning intermediate: %s", path)


def _stat_frame(metric, value_name):
    """Normalize PyPSA statistics outputs to a flat DataFrame."""
    if isinstance(metric, pd.Series):
        frame = metric.rename(value_name).to_frame()
    else:
        frame = metric.copy()
        if value_name not in frame.columns:
            if frame.shape[1] != 1:
                raise ValueError(
                    f"Expected a single-column statistics frame for {value_name}, got {list(frame.columns)}"
                )
            frame = frame.rename(columns={frame.columns[0]: value_name})
    return frame.reset_index()


def export_system_cost_statistics(network_path, output_system_costs, output_statistics, planning_horizon):
    """Export grouped capex/opex system costs and the raw PyPSA statistics table."""
    logger.info("Loading solved network for statistics export: %s", network_path)
    n = pypsa.Network(network_path)

    groupby = n.statistics.groupers.get_country_and_carrier
    capex = _stat_frame(
        n.statistics.capex(groupby=groupby, nice_names=False),
        "annualized_capital_cost_eur",
    )
    opex = _stat_frame(
        n.statistics.opex(groupby=groupby, nice_names=False),
        "operating_cost_eur",
    )

    join_cols = [c for c in ["component", "country", "carrier"] if c in capex.columns or c in opex.columns]
    system_costs = capex.merge(opex, how="outer", on=join_cols).fillna(0.0)
    if "carrier" in system_costs.columns and "technology" not in system_costs.columns:
        system_costs.insert(system_costs.columns.get_loc("carrier") + 1, "technology", system_costs["carrier"])
    system_costs.insert(0, "planning_horizon", int(planning_horizon))
    system_costs["total_system_cost_eur"] = (
        system_costs["annualized_capital_cost_eur"] + system_costs["operating_cost_eur"]
    )
    sort_cols = [c for c in ["planning_horizon", "component", "country", "carrier"] if c in system_costs.columns]
    system_costs = system_costs.sort_values(sort_cols, kind="stable", ignore_index=True)

    Path(output_system_costs).parent.mkdir(parents=True, exist_ok=True)
    system_costs.to_csv(output_system_costs, index=False)
    logger.info("Saved grouped system costs to: %s", output_system_costs)

    Path(output_statistics).parent.mkdir(parents=True, exist_ok=True)
    n.statistics().to_csv(output_statistics)
    logger.info("Saved raw network statistics to: %s", output_statistics)


def _load_historical_cumulative(tech, year, learning_cfg):
    hist_file = get_manifest_historical_datafile(learning_cfg, tech)
    if hist_file is None:
        raise ValueError(f"No historical data file configured for {tech} in manifest")
    hist = load_historical_capacity(hist_file, tech)
    hist["year"] = pd.to_numeric(hist["year"], errors="coerce").astype(int)
    row = hist[hist["year"] == int(year)]
    if row.empty:
        raise ValueError(
            f"Historical cumulative capacity for {tech} is missing at year {year}."
        )
    value = float(row["capacity_GW"].iloc[-1])
    if value <= 0.0:
        raise ValueError(
            f"Historical cumulative capacity for {tech} at year {year} must be > 0, got {value}."
        )
    return value


def _update_committed_capacity_histories(payload, solved_capacity_by_tech, learning_cfg, current_year):
    cumulative_history = payload.get("capacity_history", {}) or {}
    modeled_history = payload.get("modeled_capacity_history", {}) or {}
    tracked_techs = set((payload.get("technology_states", {}) or {}).keys())
    tracked_techs.update(cumulative_history.keys())
    tracked_techs.update(modeled_history.keys())

    for tech, solved_capacity in solved_capacity_by_tech.items():
        if tracked_techs and tech not in tracked_techs:
            logger.info("Skipping untracked technology in committed learning state: %s", tech)
            continue
        tech_cumulative = cumulative_history.get(tech, {}) or {}
        tech_modeled = modeled_history.get(tech, {}) or {}
        current_year_str = str(int(current_year))
        solved_capacity = float(solved_capacity)
        if solved_capacity <= 0.0:
            raise ValueError(
                f"Invalid solved capacity for {tech}: {solved_capacity}. Must be > 0."
            )

        prior_years = sorted(int(y) for y in tech_modeled.keys() if int(y) < int(current_year))
        if not prior_years:
            try:
                cumulative_value = _load_historical_cumulative(tech, current_year, learning_cfg)
            except ValueError:
                if tech == "battery_energy":
                    raise
                cumulative_value = solved_capacity
                logger.info(
                    "No historical cumulative %s value at %s; seeding committed history from solved network: %.3f GW",
                    tech,
                    current_year,
                    solved_capacity,
                )
        else:
            prev_year = prior_years[-1]
            prev_year_str = str(prev_year)
            if prev_year_str not in tech_cumulative:
                raise ValueError(
                    f"Committed cumulative capacity history for {tech} is missing year {prev_year}."
                )
            prev_cumulative = float(tech_cumulative[prev_year_str])
            prev_modeled = float(tech_modeled[prev_year_str])
            additions = max(solved_capacity - prev_modeled, 0.0)
            if tech == "battery_energy":
                phi_block = get_battery_phi_for_block(learning_cfg, prev_year, current_year)
                cumulative_value = prev_cumulative + additions * phi_block
                logger.info(
                    "Battery mapping %s-%s: modeled additions %.3f GWh × phi %.3f = %.3f GWh global Li-ion additions",
                    prev_year,
                    current_year,
                    additions,
                    phi_block,
                    additions * phi_block,
                )
            else:
                cumulative_value = prev_cumulative + additions

        tech_cumulative[current_year_str] = cumulative_value
        tech_modeled[current_year_str] = solved_capacity
        cumulative_history[tech] = tech_cumulative
        modeled_history[tech] = tech_modeled

    payload["capacity_history"] = cumulative_history
    payload["modeled_capacity_history"] = modeled_history
    return payload


def _read_cost_log(path):
    """Read cost log robustly whether technology is a column or CSV index."""
    df = pd.read_csv(path)
    if "technology" not in df.columns:
        if "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "technology"})
        else:
            df = df.reset_index().rename(columns={"index": "technology"})
    return df


def _learning_costs_to_frame(learning_costs):
    rows = []
    for tech, payload in learning_costs.items():
        rows.append({"technology": tech, **payload})
    return pd.DataFrame(rows)


def _learning_costs_from_cost_log_df(cost_log_df):
    if cost_log_df is None or cost_log_df.empty:
        return {}
    required = {"technology", "capital_cost"}
    missing = required - set(cost_log_df.columns)
    if missing:
        raise ValueError(
            "Cannot reconstruct applied learning costs from base cost log; "
            f"missing columns: {sorted(missing)}"
        )
    learning_costs = {}
    for _, row in cost_log_df.iterrows():
        technology = str(row.get("technology", "")).strip()
        if not technology:
            continue
        record = {}
        for column in ("capital_cost", "c_overnight", "unit", "planning_horizon", "selected_model", "seed"):
            if column in cost_log_df.columns:
                record[column] = row.get(column)
        learning_costs[technology] = record
    return learning_costs


def _attach_applied_learning_costs_from_log(network, cost_log_df):
    learning_costs = _learning_costs_from_cost_log_df(cost_log_df)
    if not learning_costs:
        return network
    if not hasattr(network, "meta") or not isinstance(network.meta, dict):
        network.meta = {}
    existing = network.meta.get("learning_costs", {}) or {}
    network.meta["learning_costs"] = {**existing, **learning_costs}
    return network


def _extract_network_capital_cost_summaries(network_path, tech_mapping):
    n = pypsa.Network(network_path)
    carrier_to_tech = dict(tech_mapping)
    records = []

    def _append(component_name, frame):
        if frame.empty or "carrier" not in frame.columns or "capital_cost" not in frame.columns:
            return
        local = frame.reset_index(drop=True).copy()
        local["technology"] = local["carrier"].map(carrier_to_tech)
        local = local.dropna(subset=["technology", "capital_cost"])
        if local.empty:
            return
        local["component"] = component_name
        records.extend(
            local.loc[:, ["technology", "component", "capital_cost"]].to_dict(orient="records")
        )

    _append("Generator", n.generators)
    _append("StorageUnit", n.storage_units)
    _append("Link", n.links)
    _append("Store", n.stores)

    if not records:
        return pd.DataFrame(
            columns=[
                "technology",
                "capital_cost_network_median",
                "capital_cost_network_min",
                "capital_cost_network_max",
            ]
        )

    summary = (
        pd.DataFrame(records)
        .groupby("technology", as_index=False)["capital_cost"]
        .agg(
            capital_cost_network_median="median",
            capital_cost_network_min="min",
            capital_cost_network_max="max",
        )
    )
    return summary


def export_deployment_constraint_diagnostics(
    output_path,
    learning_cfg,
    current_year,
    learning_seed,
    committed_payload,
    solved_capacity_by_tech,
    solved_network,
    base_cost_log_df=None,
):
    rows = []
    cfg = get_deployment_constraint_cfg(learning_cfg)
    if cfg is not None:
        formulation = get_deployment_constraint_formulation(learning_cfg) or "hard_cap"
        cap_payload = build_deployment_constraint_block_paths(
            learning_cfg,
            current_year=current_year,
            learning_seed=learning_seed,
            config_file="config.learning.yaml",
        )
        if isinstance(solved_network, (str, Path)):
            solved_network_obj = pypsa.Network(solved_network)
        else:
            solved_network_obj = solved_network
        solved_network_obj = _attach_applied_learning_costs_from_log(
            solved_network_obj,
            base_cost_log_df,
        )
        if formulation == "three_segment_wedge":
            wedge_rows = summarize_realized_deployment_wedge_rows(
                solved_network_obj,
                current_year=current_year,
                learning_cfg=learning_cfg,
                technologies=list(cfg.get("technologies", [])),
                config_file="config.learning.yaml",
            )
            for _, wedge_row in wedge_rows.iterrows():
                rows.append(
                    {
                        "year": int(current_year),
                        "country": str(wedge_row.get("country", "")),
                        "technology": str(wedge_row["technology"]),
                        "formulation": formulation,
                        "constraint_enabled": True,
                        "mode": str(cfg.get("mode", "")),
                        "uncertainty_enabled": bool(((cfg.get("uncertainty", {}) or {}).get("enabled", False))),
                        "constraint_basis_unit": str(wedge_row.get("constraint_basis_unit", "")),
                        "penalty_basis": str(wedge_row.get("penalty_basis", "")),
                        "cost_granularity": str(wedge_row.get("cost_granularity", "")),
                        "basis_cost_eur_per_unit_min": float(
                            wedge_row.get("basis_cost_eur_per_unit_min", np.nan)
                        ),
                        "basis_cost_eur_per_unit_max": float(
                            wedge_row.get("basis_cost_eur_per_unit_max", np.nan)
                        ),
                        "basis_cost_eur_per_unit_mean": float(
                            wedge_row.get("basis_cost_eur_per_unit_mean", np.nan)
                        ),
                        "basis_cost_eur_per_unit_median": float(
                            wedge_row.get("basis_cost_eur_per_unit_median", np.nan)
                        ),
                        "basis_cost_eur_per_unit_representative": float(
                            wedge_row.get("basis_cost_eur_per_unit_representative", np.nan)
                        ),
                        "basis_cost_source": str(wedge_row.get("basis_cost_source", "")),
                        "learning_capital_cost": float(wedge_row.get("learning_capital_cost", np.nan)),
                        "learning_c_overnight": float(wedge_row.get("learning_c_overnight", np.nan)),
                        "basis_cost_heterogeneous": bool(wedge_row.get("basis_cost_heterogeneous", False)),
                        "asset_count": int(wedge_row.get("asset_count", 0)),
                        "persistent_shock": np.nan,
                        "annual_shock": "",
                        "allowed_block_addition": np.nan,
                        "b1": float(wedge_row.get("b1", np.nan)),
                        "b2": float(wedge_row.get("b2", np.nan)),
                        "width1": float(wedge_row.get("width1", np.nan)),
                        "width2": float(wedge_row.get("width2", np.nan)),
                        "phi2": float(wedge_row.get("phi2", np.nan)),
                        "phi3": float(wedge_row.get("phi3", np.nan)),
                        "realized_block_addition_modeled": np.nan,
                        "realized_block_addition_constrained_basis": float(
                            wedge_row.get("realized_block_addition_constrained_basis", np.nan)
                        ),
                        "realized_seg1": float(wedge_row.get("realized_seg1", np.nan)),
                        "realized_seg2": float(wedge_row.get("realized_seg2", np.nan)),
                        "realized_seg3": float(wedge_row.get("realized_seg3", np.nan)),
                        "realized_wedge_cost_eur": float(wedge_row.get("realized_wedge_cost_eur", np.nan)),
                        "binding_slack": np.nan,
                        "battery_phi_block": np.nan,
                        "history_year": float(wedge_row.get("history_year", np.nan)),
                        "reference_annual_addition": float(wedge_row.get("reference_annual_addition", np.nan)),
                    }
                )
        else:
            realized = summarize_realized_block_additions(
                current_year=current_year,
                solved_capacity_by_tech=solved_capacity_by_tech,
                committed_state_payload=committed_payload,
                learning_cfg=learning_cfg,
                network=solved_network_obj,
            )
            technologies = list(cfg.get("technologies", [])) or sorted(set(cap_payload) | set(realized))
            for technology in technologies:
                cap_info = cap_payload.get(technology, {})
                realized_info = realized.get(technology, {})
                rows.append(
                    {
                        "year": int(current_year),
                        "country": "",
                        "technology": technology,
                        "formulation": formulation,
                        "constraint_enabled": True,
                        "mode": str(cfg.get("mode", "")),
                        "uncertainty_enabled": bool(((cfg.get("uncertainty", {}) or {}).get("enabled", False))),
                        "constraint_basis_unit": str(
                            cap_info.get(
                                "constraint_basis_unit",
                                realized_info.get("constraint_basis_unit", get_constraint_basis_unit(technology)),
                            )
                        ),
                        "penalty_basis": "",
                        "persistent_shock": float(cap_info.get("persistent_shock", 0.0)),
                        "annual_shock": json.dumps(cap_info.get("annual_shocks", {}), sort_keys=True),
                        "allowed_block_addition": float(cap_info.get("allowed_block_addition", np.nan)),
                        "b1": np.nan,
                        "b2": np.nan,
                        "width1": np.nan,
                        "width2": np.nan,
                        "phi2": np.nan,
                        "phi3": np.nan,
                        "realized_block_addition_modeled": float(realized_info.get("modeled_block_addition", np.nan)),
                        "realized_block_addition_constrained_basis": float(realized_info.get("constrained_addition", realized_info.get("constrained_basis_addition", np.nan))),
                        "realized_seg1": np.nan,
                        "realized_seg2": np.nan,
                        "realized_seg3": np.nan,
                        "realized_wedge_cost_eur": np.nan,
                        "binding_slack": float(cap_info.get("allowed_block_addition", np.nan) - realized_info.get("constrained_basis_addition", np.nan)),
                        "battery_phi_block": float(realized_info.get("battery_phi_block", np.nan))
                        if pd.notna(realized_info.get("battery_phi_block", np.nan))
                        else np.nan,
                        "history_year": np.nan,
                        "reference_annual_addition": np.nan,
                    }
                )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    logger.info("Saved deployment constraint diagnostics to: %s", output_path)


def _build_export_learning_cost_log(base_df, postsolve_df, postsolve_state_year, network_capital_summary):
    export_df = base_df.copy()
    if "planning_horizon" not in export_df.columns:
        export_df["planning_horizon"] = pd.to_numeric(export_df.get("year"), errors="coerce")

    rename_map = {
        "c_overnight": "postsolve_c_overnight",
        "c_overnight_terminal_point": "postsolve_c_overnight_terminal_point",
        "capital_cost": "postsolve_capital_cost",
        "capital_cost_terminal_point": "postsolve_capital_cost_terminal_point",
        "log_capex_terminal_point": "postsolve_log_capex_terminal_point",
    }
    keep_cols = ["technology", *rename_map.keys()]
    available_cols = [col for col in keep_cols if col in postsolve_df.columns]
    diagnostics_df = postsolve_df.loc[:, available_cols].rename(columns=rename_map)
    diagnostics_df["postsolve_state_year"] = int(postsolve_state_year)

    export_df = export_df.merge(diagnostics_df, on="technology", how="left")
    export_df = export_df.merge(network_capital_summary, on="technology", how="left")
    return export_df


def _to_float(value, fallback):
    """Convert scalar to float with fallback for NaN/None."""
    if pd.isna(value):
        return float(fallback)
    return float(value)


def _to_int(value, fallback):
    """Convert scalar to int with fallback for NaN/None."""
    if pd.isna(value):
        return int(fallback)
    return int(value)


def _to_bool(value, fallback):
    """Parse CSV bool-like value."""
    if pd.isna(value):
        return bool(fallback)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return bool(fallback)


def build_postsolve_cost_log(
    base_cost_log_df,
    solved_capacity_by_tech,
    learning_cfg,
    costs_file,
):
    """Recalculate learning costs using solved capacity from current horizon."""
    learning_costs = {}
    battery_energy_bos_multiplier = get_battery_energy_bos_multiplier(learning_cfg, costs_file)

    for _, row in base_cost_log_df.iterrows():
        tech = row["technology"]

        if tech not in solved_capacity_by_tech:
            raise ValueError(
                f"Technology {tech} is missing in solved-network extraction. "
                f"Available technologies: {sorted(solved_capacity_by_tech.keys())}"
            )

        cumulative_capacity = float(solved_capacity_by_tech[tech])
        if cumulative_capacity <= 0:
            raise ValueError(
                f"Invalid solved capacity for {tech}: {cumulative_capacity}. Must be > 0."
            )

        unit = row.get("unit")
        if unit not in {"kW", "kWh"}:
            raise ValueError(f"Unexpected unit for {tech}: {unit}")

        model_name = row.get("model_name", "")
        selected_model = row.get("selected_model", model_name)
        runtime_model = str(selected_model or model_name)
        stochastic_runtime = runtime_model in {
            "shared_state_bayesian_regime_wright",
            "correlated_geometric_random_walk",
            "way_fixed_rho_benchmark_035",
        }
        exogenous_runtime = runtime_model == "iea_weo_exogenous_path"

        A = _to_float(row.get("A"), fallback=0.0)
        beta = _to_float(row.get("beta"), fallback=0.0)
        if stochastic_runtime or exogenous_runtime:
            c_overnight = _to_float(row.get("c_overnight"), fallback=0.0)
            capital_cost = _to_float(row.get("capital_cost"), fallback=0.0)
            c_overnight_terminal_point = _to_float(
                row.get("c_overnight_terminal_point"), fallback=c_overnight
            )
            capital_cost_terminal_point = _to_float(
                row.get("capital_cost_terminal_point"), fallback=capital_cost
            )
            log_capex_terminal_point = _to_float(
                row.get("log_capex_terminal_point"),
                fallback=np.log(max(c_overnight_terminal_point, 1.0e-12) * 1000.0),
            )
        else:
            c_overnight = A * (cumulative_capacity ** (-beta))
            if tech == "battery_energy":
                c_overnight *= battery_energy_bos_multiplier

            capital_cost = convert_to_capital_cost(
                c_overnight=c_overnight,
                tech=tech,
                unit=unit,
                learning_cfg=learning_cfg,
                costs_file=costs_file,
            )
            c_overnight_terminal_point = c_overnight
            capital_cost_terminal_point = capital_cost
            log_capex_terminal_point = float(np.log(max(c_overnight, 1.0e-12) * 1000.0))

        A_base = _to_float(row.get("A_base"), fallback=A)
        beta_base = _to_float(row.get("beta_base"), fallback=beta)
        lr_base = _to_float(row.get("lr_base"), fallback=(1.0 - (2.0 ** (-beta_base))))
        lr_scenario = _to_float(
            row.get("lr_scenario"), fallback=(1.0 - (2.0 ** (-beta)))
        )
        planning_horizon = _to_int(row.get("planning_horizon"), fallback=row.get("year"))
        # Keep anchor metadata from original log; lag_year reflects solved-capacity source.
        learning_costs[tech] = {
            "planning_horizon": planning_horizon,
            "lag_year": planning_horizon,
            "anchor_year": _to_int(row.get("anchor_year"), fallback=planning_horizon),
            "anchor_source": row.get("anchor_source", ""),
            "anchor_network": row.get("anchor_network", ""),
            "cumulative_capacity_GW": cumulative_capacity,
            "A_base": A_base,
            "A_scenario": _to_float(row.get("A_scenario"), fallback=A),
            "A": A,
            "beta_base": beta_base,
            "beta": beta,
            "lr_base": lr_base,
            "lr_scenario": lr_scenario,
            "beta_adjusted": _to_bool(
                row.get("beta_adjusted"), fallback=(beta != beta_base)
            ),
            "capital_cost": capital_cost,
            "capital_cost_terminal_point": capital_cost_terminal_point,
            "unit": unit,
            "c_overnight": c_overnight,
            "c_overnight_terminal_point": c_overnight_terminal_point,
            "log_capex_runtime": log_capex_terminal_point,
            "log_capex_terminal_point": log_capex_terminal_point,
            "wacc_dict": row.get("wacc_dict", None),
            "model_name": runtime_model,
            "selected_model": runtime_model,
            "training_window": row.get("training_window", ""),
            "training_window_origin_year": _to_int(row.get("training_window_origin_year"), fallback=2020),
            "engine": row.get("engine", ""),
            "sample_mode": row.get("sample_mode", ""),
            "seed": _to_int(row.get("seed"), fallback=0),
            "learning_seed": row.get("learning_seed", ""),
            "runtime_conditioning": row.get("runtime_conditioning", ""),
            "manifest_path": row.get("manifest_path", ""),
            "manifest_schema_version": row.get("manifest_schema_version", ""),
            "manifest_sha256": row.get("manifest_sha256", ""),
            "battery_power_treatment": row.get("battery_power_treatment", ""),
            "cost_expectation_mode": row.get("cost_expectation_mode", "point_cost"),
            "cost_expectation_kernel_mode": row.get("cost_expectation_kernel_mode", "global_current_window"),
            "cost_expectation_weights_json": row.get("cost_expectation_weights_json", ""),
            "cost_expectation_lag_years_json": row.get("cost_expectation_lag_years_json", ""),
            "kernel_year_start": _to_int(row.get("kernel_year_start"), fallback=planning_horizon),
            "kernel_year_end": _to_int(row.get("kernel_year_end"), fallback=planning_horizon),
            "kernel_years_json": row.get("kernel_years_json", ""),
            "known_kernel_years_json": row.get("known_kernel_years_json", ""),
            "expected_kernel_years_json": row.get("expected_kernel_years_json", ""),
            "applied_kernel_costs_json": row.get("applied_kernel_costs_json", ""),
        }

        logger.info(
            "Post-solve %s: capacity=%.3f, c_overnight=%.3f, capital_cost=%.3f",
            tech,
            cumulative_capacity,
            c_overnight,
            capital_cost,
        )

    return learning_costs


def build_stochastic_postsolve_cost_log(base_cost_log_df, learning_costs, current_year):
    rows = []
    cost_by_tech = {tech: dict(values) for tech, values in learning_costs.items()}
    for _, row in base_cost_log_df.iterrows():
        tech = row["technology"]
        if tech not in cost_by_tech:
            raise ValueError(f"Stochastic post-solve costs are missing technology {tech}")
        payload = cost_by_tech[tech]
        payload.update(
            {
                "planning_horizon": int(current_year),
                "lag_year": int(current_year),
                "anchor_year": int(current_year),
                "anchor_source": "stochastic_postsolve_realized_block",
                "anchor_network": row.get("anchor_network", ""),
            }
        )
        rows.append({"technology": tech, **payload})
    return pd.DataFrame(rows)


def main(snakemake):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    solved_network = snakemake.input.network
    base_cost_log = snakemake.input.base_cost_log
    proposed_state = snakemake.input.proposed_state
    learning_config_path = snakemake.input.learning_config
    costs_file = snakemake.input.costs
    output_cost_log = snakemake.output.cost_log
    output_state = snakemake.output.state_committed
    output_system_costs = getattr(snakemake.output, "system_costs", None)
    output_statistics = getattr(snakemake.output, "statistics", None)
    output_deployment_constraints = getattr(snakemake.output, "deployment_constraints", None)

    logger.info("Loading learning config: %s", learning_config_path)
    learning_cfg = load_config_learning(learning_config_path)
    load_learning_manifest(learning_cfg, learning_config_path)
    global_scale_factors = get_global_scale_factors(learning_cfg)
    tech_mapping = get_tech_mapping(learning_cfg)

    logger.info("Reading base cost log: %s", base_cost_log)
    base_df = _read_cost_log(base_cost_log)
    if base_df.empty:
        raise ValueError(f"Base cost log is empty: {base_cost_log}")

    logger.info("Extracting solved capacities from: %s", solved_network)
    solved_capacity_by_tech = extract_capacity_from_network(
        solved_network,
        tech_mapping=tech_mapping,
        global_scale_factors=global_scale_factors,
    )

    current_year = int(base_df["planning_horizon"].iloc[0])
    if output_system_costs and output_statistics:
        export_system_cost_statistics(
            network_path=solved_network,
            output_system_costs=output_system_costs,
            output_statistics=output_statistics,
            planning_horizon=current_year,
        )
    logger.info("Saving post-solve learning cost log to: %s", output_cost_log)
    logger.info("Committing learning state to: %s", output_state)
    Path(output_state).parent.mkdir(parents=True, exist_ok=True)
    if not proposed_state:
        raise ValueError("Missing proposed_state input for export_postsolve_learning_costs")
    payload = json.loads(Path(proposed_state).read_text(encoding="utf-8"))
    payload["committed_from_network"] = str(solved_network)
    payload = _update_committed_capacity_histories(
        payload,
        solved_capacity_by_tech,
        learning_cfg,
        current_year,
    )
    learning_base_capacity_by_tech = build_learning_base_capacity_map(
        payload,
        solved_capacity_by_tech,
        current_year,
    )
    network_capital_summary = _extract_network_capital_cost_summaries(
        solved_network,
        tech_mapping=tech_mapping,
    )

    postsolve_learning_costs = build_postsolve_cost_log(
        base_cost_log_df=base_df,
        solved_capacity_by_tech=learning_base_capacity_by_tech,
        learning_cfg=learning_cfg,
        costs_file=costs_file,
    )

    learning_model = getattr(snakemake.wildcards, "learning_model", None)
    if learning_model in (None, ""):
        learning_model = str(base_df["selected_model"].iloc[0])

    learning_seed = getattr(snakemake.wildcards, "learning_seed", None)
    if learning_seed in (None, "") and "learning_seed" in base_df.columns:
        seed_value = base_df["learning_seed"].iloc[0]
        if pd.notna(seed_value) and str(seed_value).strip():
            learning_seed = str(seed_value)

    learning_model = get_selected_learning_model(learning_cfg, learning_model)
    learning_engine = get_learning_engine(learning_cfg, learning_model)
    resolve_runtime_seed(learning_cfg, learning_engine, learning_model, learning_seed)
    validate_runtime_contract(learning_cfg, learning_engine, learning_model)

    if output_deployment_constraints:
        export_deployment_constraint_diagnostics(
            output_path=output_deployment_constraints,
            learning_cfg=learning_cfg,
            current_year=current_year,
            learning_seed=learning_seed,
            committed_payload=payload,
            solved_capacity_by_tech=solved_capacity_by_tech,
            solved_network=solved_network,
            base_cost_log_df=base_df,
        )

    if learning_engine == "stochastic_forecast":
        postsolve_learning_costs, payload = update_stochastic_runtime_state(
            learning_cfg=learning_cfg,
            selected_model=learning_model,
            current_year=current_year,
            state=payload,
            costs_file=costs_file,
            solved_capacity_by_tech=learning_base_capacity_by_tech,
        )
        postsolve_df = build_stochastic_postsolve_cost_log(base_df, postsolve_learning_costs, current_year)
        postsolve_state_year = int(payload.get("last_applied_year", current_year))
    else:
        postsolve_df = _learning_costs_to_frame(postsolve_learning_costs)
        postsolve_state_year = int(current_year)
        payload["last_applied_year"] = int(current_year)
        payload["technology_states"] = {
            tech: {
                "capital_cost": float(values["capital_cost"]),
                "c_overnight": float(values["c_overnight"]),
                "cumulative_capacity_GW": float(values["cumulative_capacity_GW"]),
                "unit": values["unit"],
            }
            for tech, values in postsolve_learning_costs.items()
        }

    export_df = _build_export_learning_cost_log(
        base_df=base_df,
        postsolve_df=postsolve_df,
        postsolve_state_year=postsolve_state_year,
        network_capital_summary=network_capital_summary,
    )
    Path(output_cost_log).parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(output_cost_log, index=False)

    Path(output_state).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _cleanup_superseded_learning_inputs(base_cost_log, proposed_state, current_year)
    logger.info("Done")


if __name__ == "__main__":
    if "snakemake" in globals():
        main(snakemake)
    else:
        snakemake = mock_snakemake(
            "export_postsolve_learning_costs",
            simpl="",
            clusters="200",
            ll="copt",
            opts="1h",
            sopts="1h",
            planning_horizons="2050",
            discountrate="0.071",
            demand="AB",
            h2export="0.0",
            learning_rate="base",
            learning_model="legacy_curve",
            configfile="/shared/share_cki25/energymodels/pypsa-earth/config.myopic.yaml",
        )
        main(snakemake)
