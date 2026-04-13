#!/usr/bin/env python3
"""Export learning-implied costs from the solved network of the same horizon.

This script creates a post-solve diagnostic cost log that keeps the same
column schema as `apply_learning_costs.py` (`cost_log_*`) but recomputes
`cumulative_capacity_GW`, `c_overnight`, and `capital_cost` from the solved
network of the current planning horizon.
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
    BOS_multiplier,
    build_learning_base_capacity_map,
    convert_to_capital_cost,
    extract_capacity_from_network,
    get_battery_phi_for_block,
    get_learning_engine,
    get_manifest_historical_datafile,
    get_selected_learning_model,
    get_global_scale_factors,
    get_tech_mapping,
    load_config_learning,
    load_learning_manifest,
    resolve_runtime_seed,
    save_cost_log,
    update_stochastic_runtime_state,
    validate_runtime_contract,
)
from learning.learning_data_io import load_historical_capacity


logger = logging.getLogger(__name__)


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
            cumulative_value = _load_historical_cumulative(tech, current_year, learning_cfg)
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
        stochastic_runtime = str(model_name) in {
            "shared_state_bayesian_regime_wright",
            "correlated_geometric_random_walk",
            "way_fixed_rho_benchmark_035",
        }

        A = _to_float(row.get("A"), fallback=0.0)
        beta = _to_float(row.get("beta"), fallback=0.0)
        if stochastic_runtime:
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
                c_overnight *= BOS_multiplier

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
        selected_model = row.get("selected_model", model_name)

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
            "model_name": model_name,
            "selected_model": selected_model,
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
            "cost_expectation_weights_json": row.get("cost_expectation_weights_json", ""),
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

    learning_costs = build_postsolve_cost_log(
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

    if learning_engine != "legacy_curve":
        learning_costs, payload = update_stochastic_runtime_state(
            learning_cfg=learning_cfg,
            selected_model=learning_model,
            current_year=current_year,
            state=payload,
            costs_file=costs_file,
            solved_capacity_by_tech=learning_base_capacity_by_tech,
        )
        postsolve_df = build_stochastic_postsolve_cost_log(base_df, learning_costs, current_year)
        postsolve_df.to_csv(output_cost_log, index=False)
    else:
        save_cost_log(learning_costs, output_cost_log)
        payload["last_applied_year"] = int(current_year)
        payload["technology_states"] = {
            tech: {
                "capital_cost": float(values["capital_cost"]),
                "c_overnight": float(values["c_overnight"]),
                "cumulative_capacity_GW": float(values["cumulative_capacity_GW"]),
                "unit": values["unit"],
            }
            for tech, values in learning_costs.items()
        }

    Path(output_state).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
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
