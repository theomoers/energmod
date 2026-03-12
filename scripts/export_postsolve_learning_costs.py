#!/usr/bin/env python3
"""Export learning-implied costs from the solved network of the same horizon.

This script creates a post-solve diagnostic cost log that keeps the same
column schema as `apply_learning_costs.py` (`cost_log_*`) but recomputes
`cumulative_capacity_GW`, `c_overnight`, and `capital_cost` from the solved
network of the current planning horizon.
"""

import logging

import pandas as pd

from _helpers import mock_snakemake
from apply_learning_costs import (
    BOS_multiplier,
    convert_to_capital_cost,
    extract_capacity_from_network,
    get_global_scale_factors,
    load_config_learning,
    save_cost_log,
)


logger = logging.getLogger(__name__)


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

        A = _to_float(row.get("A"), fallback=0.0)
        beta = _to_float(row.get("beta"), fallback=0.0)
        unit = row.get("unit")
        if unit not in {"kW", "kWh"}:
            raise ValueError(f"Unexpected unit for {tech}: {unit}")

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
            "unit": unit,
            "c_overnight": c_overnight,
            "wacc_dict": row.get("wacc_dict", None),
        }

        logger.info(
            "Post-solve %s: capacity=%.3f, c_overnight=%.3f, capital_cost=%.3f",
            tech,
            cumulative_capacity,
            c_overnight,
            capital_cost,
        )

    return learning_costs


def main(snakemake):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    solved_network = snakemake.input.network
    base_cost_log = snakemake.input.base_cost_log
    learning_config_path = snakemake.input.learning_config
    costs_file = snakemake.input.costs
    output_cost_log = snakemake.output.cost_log

    logger.info("Loading learning config: %s", learning_config_path)
    learning_cfg = load_config_learning(learning_config_path)
    global_scale_factors = get_global_scale_factors(learning_cfg)
    tech_mapping = learning_cfg.get("tech_mapping", {})

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

    learning_costs = build_postsolve_cost_log(
        base_cost_log_df=base_df,
        solved_capacity_by_tech=solved_capacity_by_tech,
        learning_cfg=learning_cfg,
        costs_file=costs_file,
    )

    logger.info("Saving post-solve learning cost log to: %s", output_cost_log)
    save_cost_log(learning_costs, output_cost_log)
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
            configfile="/shared/share_cki25/energymodels/pypsa-earth/config.myopic.yaml",
        )
        main(snakemake)
