# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: PyPSA-Earth Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Solves a rolling horizon multi-period network and extracts first period results.

This script:
1. Solves the multi-period optimization problem for a rolling window
2. Extracts only the first investment period's optimized capacities
3. Saves a single-period network with committed investment decisions

The saved network can then be used as brownfield input for the next iteration.
"""

import logging
import os
import sys
from functools import partial

import numpy as np
import pandas as pd
import pypsa
import yaml

from _helpers import get_rolling_window_years

# Rolling-horizon endogenous learning
from learning_clean import add_endogenous_learning_rolling_horizon_clean, zero_learning_capital_costs, build_TECH_TO_CARRIERS
from prepare_rolling_horizon import attach_learning_parameters

# Import solve_network components
from solve_network import (
    prepare_network,
    extra_functionality,
)

logger = logging.getLogger(__name__)


def _coord_to_index(coord):
    try:
        return coord.to_index()
    except Exception:
        return pd.Index(coord.values)


def _validate_model_period_coords(model):
    if not hasattr(model, "coords") or "period" not in model.coords:
        return

    base_index = _coord_to_index(model.coords["period"])
    mismatches = []

    for name, var in model.variables.items():
        if "period" in var.coords:
            idx = _coord_to_index(var.coords["period"])
            if not idx.equals(base_index):
                mismatches.append(("variable", name, idx))

    for name, con in model.constraints.items():
        if "period" in con.coords:
            idx = _coord_to_index(con.coords["period"])
            if not idx.equals(base_index):
                mismatches.append(("constraint", name, idx))

    if mismatches:
        sample = "; ".join(
            [f"{kind} {name} (dtype={idx.dtype})" for kind, name, idx in mismatches[:10]]
        )
        raise ValueError(
            "Period coordinate mismatch detected. "
            f"Model period dtype={base_index.dtype}. "
            f"Mismatched items (sample): {sample}"
        )


def cluster_network_to_single_node(n: pypsa.Network, world_tag: str = "WORLD") -> pypsa.Network:
    """
    Aggregate a network spatially to one representative node *per carrier*.

    - Buses: one bus per carrier (e.g. "AC", "H2", "heat")
    - Components: grouped by carrier (+ build_year if present)
      Links are grouped by carrier + bus topology (+ build_year)
    - Time series are aggregated consistently:
      - Loads p_set summed
      - Per-unit series (p_max_pu, p_min_pu, efficiency, ...) averaged
      - Costs averaged
    - Snapshot/investment period weightings preserved.
    """

    n = n.copy()

    if n.buses.empty:
        return n

    # --- 1) Make bus mapping: original bus -> new bus (carrier name) ---
    if "carrier" not in n.buses.columns:
        raise ValueError("n.buses has no 'carrier' column; cannot cluster to single node safely.")

    bus_to_newbus = n.buses["carrier"].astype(str).to_dict()  # e.g. "DE0 0" -> "AC"

    # --- 2) Build new buses table: one bus per carrier ---
    buses_old = n.buses.copy()
    buses_old["new_bus"] = buses_old.index.map(lambda b: bus_to_newbus.get(b, str(b)))

    # aggregate bus attributes
    bus_agg = {}
    for col in buses_old.columns:
        if col in ["x", "y", "lat", "lon"]:
            bus_agg[col] = "mean"
        elif col in ["carrier"]:
            bus_agg[col] = "first"
        elif col in ["country", "location", "region", "sub_network"]:
            bus_agg[col] = "first"
        else:
            bus_agg[col] = "first"

    buses_new = buses_old.groupby("new_bus").agg(bus_agg)
    buses_new.index.name = "Bus"

    # overwrite to global tag (optional but often helpful)
    for col in ["country", "location", "region"]:
        if col in buses_new.columns:
            buses_new[col] = world_tag

    # --- 3) Helper: map all bus columns in a component dataframe ---
    def _map_bus_columns(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        df = df.copy()
        bus_cols = [c for c in df.columns if c.startswith("bus")]
        for c in bus_cols:
            df[c] = df[c].map(lambda b: bus_to_newbus.get(b, b)).astype(str)
        return df

    # --- 4) Choose aggregation rules for static columns ---
    def _static_agg_rules(df: pd.DataFrame) -> dict:
        agg = {}
        for col in df.columns:
            if col.startswith("bus"):
                agg[col] = "first"
            elif col in ["carrier", "build_year"]:
                agg[col] = "first"
            elif df[col].dtype == bool:
                agg[col] = "any"
            elif pd.api.types.is_numeric_dtype(df[col]):
                # capacities should add up
                if any(x in col for x in ["p_nom", "e_nom", "s_nom"]):
                    agg[col] = "sum"
                # potentials add up globally too
                elif col.endswith("_max") or col.endswith("_min"):
                    agg[col] = "sum"
                # costs/efficiencies should not sum
                elif "cost" in col or "efficiency" in col:
                    agg[col] = "mean"
                else:
                    agg[col] = "mean"
            else:
                agg[col] = "first"
        return agg

    # --- 5) Component grouping keys ---
    def _group_keys(component_name: str, df: pd.DataFrame) -> list:
        keys = []
        if "carrier" in df.columns:
            keys.append("carrier")

        # keep link topology distinct (AC->H2 vs AC->heat etc)
        if component_name == "Link":
            for bc in ["bus0", "bus1", "bus2", "bus3", "bus4"]:
                if bc in df.columns:
                    keys.append(bc)

        # preserve investment-period separation
        if "build_year" in df.columns:
            keys.append("build_year")

        # if no keys, don't aggregate
        return keys

    # --- 6) Create clustered network m ---
    m = pypsa.Network()
    m.name = getattr(n, "name", "")
    if hasattr(n, "srid"):
        m.srid = n.srid

    # snapshots + weights (MultiIndex snapshots supported)
    m.set_snapshots(n.snapshots)
    m.snapshot_weightings = n.snapshot_weightings.copy()
    
    # Set investment periods before weightings to avoid validation errors
    if hasattr(n, "investment_periods") and n.investment_periods is not None:
        m.investment_periods = n.investment_periods
    
    if hasattr(n, "investment_period_weightings") and n.investment_period_weightings is not None:
        m.investment_period_weightings = n.investment_period_weightings.copy()

    # carriers unchanged
    if not n.carriers.empty:
        m.import_components_from_dataframe(n.carriers.copy(), "Carrier")

    # buses clustered
    m.import_components_from_dataframe(buses_new, "Bus")

    # global constraints: keep (important for global CO2 caps etc.)
    if hasattr(n, "global_constraints") and not n.global_constraints.empty:
        m.import_components_from_dataframe(n.global_constraints.copy(), "GlobalConstraint")

    # --- 7) Aggregate & import all remaining components (except Lines) ---
    skip_components = {"Line"}  # drop transmission grid completely in a 1-node world
    all_components = sorted(n.all_components - {"Bus", "Carrier"} - skip_components)

    for cname in all_components:
        # Get component using iterate_components
        comp_list = list(n.iterate_components([cname]))
        if not comp_list:
            continue
        comp = comp_list[0]
        df = comp.df
        if df.empty:
            continue

        df_mapped = _map_bus_columns(df)

        keys = _group_keys(cname, df_mapped)
        if not keys:
            # nothing to aggregate -> just import
            m.import_components_from_dataframe(df_mapped, cname)
        else:
            agg = _static_agg_rules(df_mapped)
            df_agg = df_mapped.groupby(keys).agg(agg)

            # build clean string index like "solar-2020" etc.
            if "build_year" in keys:
                if cname == "Link":
                    # include topology in name to avoid accidental merging
                    # name = carrier|bus0->bus1|year
                    idx = []
                    for (carrier, *rest) in df_agg.index:
                        by = rest[-1]
                        buses_part = "->".join(rest[:-1]) if len(rest) > 1 else ""
                        idx.append(f"{carrier}|{buses_part}-{int(by)}")
                    df_agg.index = pd.Index(idx)
                else:
                    df_agg.index = pd.Index([f"{i}-{int(j)}" for i, j in df_agg.index])
            else:
                df_agg.index = pd.Index([str(i) for i in df_agg.index])

            # force mapped buses consistent with new bus names
            for col in df_agg.columns:
                if col.startswith("bus"):
                    df_agg[col] = df_agg[col].astype(str)

            m.import_components_from_dataframe(df_agg, cname)

    # --- 8) Aggregate & import time series data consistently ---
    for cname in all_components:
        # Get component using iterate_components
        comp_list = list(n.iterate_components([cname]))
        if not comp_list:
            continue
        comp = comp_list[0]
        df_old = comp.df
        if df_old.empty:
            continue

        df_mapped = _map_bus_columns(df_old)
        keys = _group_keys(cname, df_mapped)
        if not keys or not comp.pnl:
            continue

        # mapping from old component index -> new aggregated component name
        if "build_year" in keys:
            if cname == "Link":
                def _new_name(row):
                    carrier = row.get("carrier", "X")
                    buses_part = "->".join([str(row.get(b, "")) for b in ["bus0","bus1","bus2","bus3","bus4"] if b in row.index and pd.notna(row.get(b, None))])
                    by = int(row.get("build_year"))
                    return f"{carrier}|{buses_part}-{by}"
            else:
                def _new_name(row):
                    return f"{row.get('carrier','X')}-{int(row.get('build_year'))}"
        else:
            def _new_name(row):
                return str(row.get("carrier", row.name))

        name_map = df_mapped.apply(_new_name, axis=1)

        # add time series to m
        # Get component from m using iterate_components
        mcomp_list = list(m.iterate_components([cname]))
        if not mcomp_list:
            continue
        mcomp = mcomp_list[0]
        for attr, pnl_df in comp.pnl.items():
            if pnl_df is None or pnl_df.empty:
                continue

            pnl_df = pnl_df.reindex(columns=df_old.index).copy()
            pnl_df = pnl_df.rename(columns=name_map.to_dict())

            # aggregation rule by attribute type
            if cname == "Load" and attr == "p_set":
                pnl_agg = pnl_df.T.groupby(level=0).sum().T
            elif attr.endswith("_pu") or "efficiency" in attr or "cost" in attr:
                pnl_agg = pnl_df.T.groupby(level=0).mean().T
            else:
                # safe fallback: sum for additive quantities, else mean
                if attr.startswith("p") and not attr.endswith("_pu"):
                    pnl_agg = pnl_df.T.groupby(level=0).sum().T
                else:
                    pnl_agg = pnl_df.T.groupby(level=0).mean().T

            # keep only columns that exist in the imported clustered component
            pnl_agg = pnl_agg.reindex(columns=mcomp.df.index)

            # use PyPSA import helper
            if not pnl_agg.empty:
                m.import_series_from_dataframe(pnl_agg, cname, attr)

    # --- 9) Drop links that became internal (bus0 == bus1 etc.) ---
    if not m.links.empty and "bus0" in m.links.columns and "bus1" in m.links.columns:
        internal_links = m.links.index[m.links.bus0 == m.links.bus1]
        if len(internal_links):
            m.mremove("Link", internal_links)

    try:
        m.determine_network_topology()
    except Exception:
        pass

    return m


def extra_functionality_rolling_horizon(n, snapshots):
    """
    Wrapper around the standard extra_functionality which additionally injects
    rolling-horizon endogenous learning (investment-period model).
    """
    learning_cfg = getattr(n, "config", {}).get("learning", {}) if hasattr(n, "config") else {}
    
    # Skip endogenous learning for first planning horizon (2020)
    # 2020 costs are exogenously determined from 2018 historical data
    # All generators are non-extendable in 2020, so no learning optimization needed
    # Starting from 2025 window (2025-2030), learning is applied based on optimized 2020 build
    planning_horizons = getattr(n, "planning_horizons", None)
    skip_learning_years = [2020]
    
    if planning_horizons is not None:
        try:
            current_year = int(planning_horizons)
            if current_year in skip_learning_years:
                logger.info("=" * 80)
                logger.info(f"SKIPPING ENDOGENOUS LEARNING for planning horizon {current_year}")
                logger.info(f"Costs are exogenously determined for {current_year}")
                logger.info("=" * 80)
                learning_cfg = {}  # Disable learning for this horizon
        except (ValueError, TypeError):
            pass
    
    if learning_cfg.get("enabled", False):
        logger.info("=" * 80)
        logger.info("ROLLING-HORIZON ENDOGENOUS LEARNING: SETUP PHASE")
        logger.info("=" * 80)
        
        # Ensure learning parameters are attached for this multi-period network.
        # PyPSA's NetCDF export does not persist arbitrary attributes, so we may
        # need to (re)attach here even if prepare_rolling_horizon already did so.
        if not hasattr(n, "learning_params"):
            logger.info("Learning parameters not found on network, attaching now...")
            try:
                # Prefer explicit rolling_horizon_window from meta; fall back to investment_periods.
                window_years = n.meta.get("rolling_horizon_window", None) if hasattr(n, "meta") else None
                if not window_years:
                    window_years = list(getattr(n, "investment_periods", []))
                # Preserve period dtype from the network
                years = list(window_years)
                # Retrieve learning_state_file from network meta (stored by prepare_rolling_horizon)
                learning_state_file = n.meta.get("learning_state_file", None) if hasattr(n, "meta") else None
                attach_learning_parameters(n, years, getattr(n, "config", {}), learning_state_file=learning_state_file)
                logger.info(f"Successfully attached learning_params for rolling window years={years}")
            except Exception as e:
                logger.exception(f"Failed to attach learning parameters in extra_functionality: {e}")
                raise
        else:
            logger.info("Learning parameters already attached to network")
            if hasattr(n, "learning_params"):
                logger.info(f"  Technologies: {n.learning_params.get('technologies', [])}")
                logger.info(f"  Segments: {n.learning_params.get('segments', 'N/A')}")
                logger.info(f"  Lag periods: {n.learning_params.get('lag_periods', 'N/A')}")
        logger.info("")

        #add_endogenous_learning_rolling_horizon(n, snapshots, learning_cfg)

        add_endogenous_learning_rolling_horizon_clean(n, snapshots, learning_cfg)

    # Keep existing constraints/callbacks
    if hasattr(n, "model"):
        _validate_model_period_coords(n.model)
    return extra_functionality(n, snapshots)


def _extract_learning_state_committed(n_solved: pypsa.Network, committed_year: int, config: dict) -> pd.DataFrame:
    """
    Extract committed cumulative capacity from network components.
    This represents the total installed capacity at the end of the committed period.
    
    Uses actual network data (n.generators.p_nom, n.stores.e_nom, etc.) rather than
    learning model variables, so it works regardless of whether learning was enabled.
    """
    learning_cfg = config.get("learning", {}) or {}
    
    # Build technology-to-carrier mapping
    tech_mapping = {
        "solar_power": ["solar"],
        "onwind_power": ["onwind"],
        "offwind_power": ["offwind-ac", "offwind-dc"],
        "battery_power": ["battery inverter"],
        "battery_energy": ["battery"],
        "electrolyser_power": ["H2 Electrolysis"],
        "h2_energy": ["H2 Store"],
    }
    
    # Override with custom mapping if provided
    custom_mapping = learning_cfg.get("tech_mapping", {})
    if custom_mapping:
        # Invert the mapping: carrier -> tech becomes tech -> [carriers]
        inverted = {}
        for carrier, tech in custom_mapping.items():
            inverted.setdefault(tech, []).append(carrier)
        tech_mapping.update(inverted)
    
    rows = []
    
    for tech, carriers in tech_mapping.items():
        total_capacity_GW = 0.0
        
        # Sum capacity from generators (p_nom for power)
        if hasattr(n_solved, "generators") and not n_solved.generators.empty:
            gens = n_solved.generators[n_solved.generators.carrier.isin(carriers)]
            if not gens.empty:
                total_capacity_GW += gens.p_nom.sum() / 1000.0  # MW to GW
        
        # Sum capacity from links (p_nom for power)
        if hasattr(n_solved, "links") and not n_solved.links.empty:
            links = n_solved.links[n_solved.links.carrier.isin(carriers)]
            if not links.empty:
                total_capacity_GW += links.p_nom.sum() / 1000.0  # MW to GW
        
        # Sum capacity from stores (e_nom for energy)
        if hasattr(n_solved, "stores") and not n_solved.stores.empty:
            stores = n_solved.stores[n_solved.stores.carrier.isin(carriers)]
            if not stores.empty:
                total_capacity_GW += stores.e_nom.sum() / 1000.0  # MWh to GWh
        
        # Sum capacity from storage units (p_nom for power)
        if hasattr(n_solved, "storage_units") and not n_solved.storage_units.empty:
            su = n_solved.storage_units[n_solved.storage_units.carrier.isin(carriers)]
            if not su.empty:
                total_capacity_GW += su.p_nom.sum() / 1000.0  # MW to GW
        
        if total_capacity_GW > 0:
            rows.append({
                "technology": tech,
                "year": int(committed_year),
                "E_committed_GW": total_capacity_GW
            })
            logger.info(f"Extracted {tech}: {total_capacity_GW:.2f} GW (cumulative capacity)")
    
    df = pd.DataFrame(rows)
    logger.info(f"Extracted learning state with {len(df)} technologies")
    return df


def _validate_learning_solution(n_solved: pypsa.Network):
    """
    Lightweight post-solve validation/logging for rolling-horizon learning.
    """
    if not hasattr(n_solved, "config"):
        return
    learning_cfg = n_solved.config.get("learning", {}) or {}
    if not learning_cfg.get("enabled", False):
        return
    if not hasattr(n_solved, "model") or "learning_E" not in n_solved.model.variables:
        logger.warning("[learning] enabled but no learning variables found in solved model")
        return

    lag_periods = int(learning_cfg.get("lag_periods", 1))
    E = n_solved.model["learning_E"]
    DK = n_solved.model.variables.get("learning_DK")
    TC = n_solved.model.variables.get("learning_TC")
    inv_cost = n_solved.model.variables.get("learning_inv_cost")

    periods = list(E.coords["period"].values)
    techs = list(E.coords["carrier"].values)

    # Basic range checks (interpolation coverage)
    if hasattr(n_solved, "learning_params"):
        for tech in techs:
            E_points, _ = n_solved.learning_params["interpolation"][tech]
            emin, emax = float(E_points[0]), float(E_points[-1])
            for t in periods:
                val = float(E.sel(carrier=tech, period=int(t)).solution)
                if val < emin - 1e-6 or val > emax + 1e-6:
                    logger.warning(f"[learning] {tech} E[{t}]={val:.4f} outside [{emin:.4f}, {emax:.4f}]")

    # Linking constraint residuals: ΔK[t] == gf*(E[t]-E[t-1]) (gf=1 for solar/onwind)
    if DK is not None:
        max_res = 0.0
        for tech in techs:
            E_prev_map = getattr(n_solved, "learning_params", {}).get("E_previous", {}).get(tech, {})
            gf = float(getattr(n_solved, "learning_params", {}).get("global_factor", 1.0))
            for i, t in enumerate(periods):
                t = int(t)
                dk = float(DK.sel(carrier=tech, period=t).solution)
                e_t = float(E.sel(carrier=tech, period=t).solution)
                if i == 0:
                    e0 = float(E_prev_map.get(t, 0.0))
                    rhs = gf * (e_t - e0)
                else:
                    t_prev = int(periods[i - 1])
                    e_prev = float(E.sel(carrier=tech, period=t_prev).solution)
                    rhs = gf * (e_t - e_prev)
                max_res = max(max_res, abs(dk - rhs))
        logger.info(f"[learning] max |ΔK - gf·ΔE| residual: {max_res:.6e} (GW)")

    # Cost semantics quick check
    if inv_cost is not None and TC is not None:
        max_cost_res = 0.0
        for tech in techs:
            TC_prev_map = getattr(n_solved, "learning_params", {}).get("TC_previous", {}).get(tech, {})
            for i, t in enumerate(periods):
                t = int(t)
                inv = float(inv_cost.sel(carrier=tech, period=t).solution)
                tc_t = float(TC.sel(carrier=tech, period=t).solution)
                if lag_periods == 0:
                    if i == 0:
                        tc0 = float(TC_prev_map.get(t, 0.0))
                        rhs = tc_t - tc0
                    else:
                        t_prev = int(periods[i - 1])
                        tc_prev = float(TC.sel(carrier=tech, period=t_prev).solution)
                        rhs = tc_t - tc_prev
                    max_cost_res = max(max_cost_res, abs(inv - rhs))
        if lag_periods == 0:
            logger.info(f"[learning] max |inv_cost - ΔTC| residual (lag=0): {max_cost_res:.6e} (EUR)")


def add_land_use_constraint_perfect(n):
    """
    Add global constraints for tech capacity limit in multi-period optimization.
    
    This function sets technical capacity limits (p_nom_max) per bus, carrier,
    and investment period. The limits are stored as bus attributes in the format
    'nom_max_{carrier}_{build_year}' which PyPSA uses to enforce constraints.
    
    Based on pypsa-eur implementation, adapted for pypsa-earth.
    
    Parameters
    ----------
    n : pypsa.Network
        Multi-period network with investment periods
    """
    logger.info("Adding land-use constraints for perfect foresight/rolling horizon")
    
    def check_p_min_p_max(p_nom_max, grouper, ext_i):
        """Check if minimum capacities exceed maximum capacities."""
        p_nom_min = n.generators[ext_i].groupby(grouper).sum().p_nom_min
        p_nom_min = p_nom_min.reindex(p_nom_max.index)
        check = (
            p_nom_min.groupby(level=[0, 1]).sum()
            > p_nom_max.groupby(level=[0, 1]).min()
        )
        if check.sum():
            logger.warning(
                f"summed p_min_pu values at node larger than technical potential {check[check].index.tolist()}"
            )
    
    # Group by carrier, bus, and build_year for multi-period constraints
    if "build_year" not in n.generators.columns:
        logger.warning("build_year not found in generators. Skipping land use constraints.")
        return
    
    grouper = [n.generators.carrier, n.generators.bus, n.generators.build_year]
    ext_i = n.generators.p_nom_extendable
    
    # Get technical limit per node and investment period (minimum across generators in group)
    p_nom_max = n.generators[ext_i].groupby(grouper).min().p_nom_max
    
    # Drop carriers without tech limit (inf or nan)
    p_nom_max = p_nom_max[~p_nom_max.isin([np.inf])]
    p_nom_max = p_nom_max[pd.notna(p_nom_max)]
    
    if p_nom_max.empty:
        logger.info("No generators with technical capacity limits found. Skipping constraints.")
        return
    
    # Get carriers that have capacity limits
    carriers = p_nom_max.index.get_level_values(0).unique()
    gen_i = n.generators[(n.generators.carrier.isin(carriers)) & (ext_i)].index
    
    # Set p_nom_min to 0 for extendable generators (constraint handled via bus attributes)
    n.generators.loc[gen_i, "p_nom_min"] = 0
    
    # Check minimum capacities don't exceed maximum
    check_p_min_p_max(p_nom_max, grouper, ext_i)
    
    # Convert to DataFrame and create bus attribute names
    df = p_nom_max.reset_index()
    df["name"] = df.apply(
        lambda row: f"nom_max_{row['carrier']}"
        + (f"_{int(row['build_year'])}" if pd.notna(row["build_year"]) else ""),
        axis=1,
    )
    
    # Set bus attributes for capacity limits
    for name in df.name.unique():
        df_carrier = df[df.name == name]
        bus = df_carrier.bus
        n.buses.loc[bus, name] = df_carrier.p_nom_max.values
    
    logger.info(f"Added land-use constraints for {len(df.name.unique())} carrier-period combinations")


def collect_kwargs(config, solving, planning_horizons=None, log_fn=None, mode="single"):
    """
    Collect keyword arguments for optimization.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    solving : dict
        Solving configuration dictionary
    planning_horizons : str or None
        Planning horizons (not used for multi-period)
    log_fn : str or None
        Path to solver log file
    mode : str
        Solving mode (e.g., "single", "iterative")
    
    Returns
    -------
    dict
        Keyword arguments for n.optimize()
    """
    set_of_options = solving["solver"]["options"]
    solver_options = solving["solver_options"].get(set_of_options, {}) if set_of_options else {}

    # Gurobi solver options for numerical stability and performance   
     
    if solving["solver"]["name"] == "gurobi" and int(planning_horizons) != 2020:
        logger.info(f"Enabling MILP options for {planning_horizons}")
        gurobi_options = {
            "MIPFocus": 1,
            "MIPGap": 0.03,
            "Heuristics": 0.5,
            "Cuts": 1,
            "NodeMethod": 2,
            "NodeFileStart": 0.5,
            "crossover": 0,
            "BarConvTol": 1e-3,
            "FeasibilityTol": 1e-5,
            "AggFill": 0,
            "PreDual": 0,
            "GURO_PAR_BARDENSETHRESH": 200,
            #"NumericFocus": 1,
        }
        
        for key, value in gurobi_options.items():
            if key not in solver_options:
                solver_options[key] = value
                logger.info(f"Added {key}={value} to solver options for Gurobi (endogenous learning, horizon={planning_horizons})")
    
    kwargs = {
        "solver_name": solving["solver"]["name"],
        "solver_options": solver_options,
        "extra_functionality": extra_functionality_rolling_horizon,
        "multi_investment_periods": True,
    }
    
    # Add log file if provided (use log_fn, not solver_logfile - matches solve_network.py)
    if log_fn:
        kwargs["log_fn"] = log_fn
    
    return kwargs


def extract_period_results(n, period_year, commit_assets=True):
    """
    Extract results for a specific investment period.
    
    After solving the multi-period problem, this function:
    1. Identifies assets built in the specified period (build_year == period_year)
    2. Optionally commits their optimized capacities (p_nom_opt -> p_nom)
    3. Creates a single-period network with only the specified period's snapshots
    
    Parameters
    ----------
    n : pypsa.Network
        Solved multi-period network
    period_year : int
        The investment period year to extract
    commit_assets : bool, optional
        Whether to commit optimized capacities (p_nom_opt -> p_nom).
        Set to True for brownfield continuation (default), False for comparison only.
    
    Returns
    -------
    pypsa.Network
        Single-period network with specified period's results
    """
    logger.info(f"Extracting results for period: {period_year} (commit_assets={commit_assets})")
    
    n_period = pypsa.Network()
    
    n_period.name = n.name
    
    # Get period snapshots
    period_snapshots = n.snapshots[n.snapshots.get_level_values(0) == period_year]
    # Convert to single-level index (just the timestamps)
    single_snapshots = period_snapshots.get_level_values(1)
    
    # Extract snapshot weightings for period and convert index to single level
    # Must do this BEFORE setting snapshots to ensure index matches
    weightings = n.snapshot_weightings.loc[period_snapshots].copy()
    weightings.index = single_snapshots
    
    # Set snapshots and weightings together
    n_period.set_snapshots(single_snapshots)
    # Now assign weightings - index must exactly match n_period.snapshots
    n_period.snapshot_weightings = weightings.reindex(single_snapshots)
    
    # Copy carriers
    n_period.import_components_from_dataframe(n.carriers, "Carrier")
    
    # Copy buses
    n_period.import_components_from_dataframe(n.buses, "Bus")
    
    # Process each component type
    for c_name in ["Generator", "Link", "Store", "StorageUnit", "Line", "Load"]:
        c_df = getattr(n, c_name.lower() + "s" if c_name != "StorageUnit" else "storage_units")
        c_pnl = getattr(n, c_name.lower() + "s_t" if c_name != "StorageUnit" else "storage_units_t")
        
        if c_df.empty:
            continue
        
        # Make a copy of the dataframe
        df_copy = c_df.copy()
        
        # Determine the capacity attribute (s for lines, e for stores, p for others)
        if c_name == "Line":
            attr = "s"
        elif c_name == "Store":
            attr = "e"
        else:
            attr = "p"
        
        opt_col = f"{attr}_nom_opt"
        nom_col = f"{attr}_nom"
        extendable_col = f"{attr}_nom_extendable"
        
        if opt_col in df_copy.columns:
            # Identify assets that were optimized
            if extendable_col in df_copy.columns and "build_year" in df_copy.columns:
                # For specified period assets, optionally commit the optimized capacity
                # Other assets (from previous periods) keep their existing capacity
                period_assets = df_copy.index[df_copy.build_year == period_year]
                
                if not period_assets.empty and commit_assets:
                    # Set p_nom/e_nom to the optimized value for period assets
                    df_copy.loc[period_assets, nom_col] = df_copy.loc[period_assets, opt_col]
                    # Make them non-extendable (capacity is now committed)
                    df_copy.loc[period_assets, extendable_col] = False
                    
                    logger.info(
                        f"Committed {len(period_assets)} {c_name} assets "
                        f"with build_year={period_year}"
                    )
                elif not period_assets.empty and not commit_assets:
                    # For comparison networks, keep them extendable with opt values
                    logger.info(
                        f"Extracted {len(period_assets)} {c_name} assets "
                        f"with build_year={period_year} (not committed)"
                    )
            elif extendable_col in df_copy.columns and commit_assets:
                # For components without build_year (like Lines), commit all optimized
                extendable_mask = df_copy[extendable_col]
                if extendable_mask.any():
                    df_copy.loc[extendable_mask, nom_col] = df_copy.loc[extendable_mask, opt_col]
                    logger.info(f"Committed {extendable_mask.sum()} {c_name} capacities")
        
        # Filter out assets from future periods (build_year > period_year)
        if "build_year" in df_copy.columns:
            # Keep assets from specified period and earlier (brownfield)
            df_copy = df_copy[df_copy.build_year <= period_year]
        
        # Set *_opt columns appropriately
        if opt_col in df_copy.columns and nom_col in df_copy.columns:
            if commit_assets:
                # For committed assets, opt should equal nom
                df_copy[opt_col] = df_copy[nom_col]
            # else: keep opt as-is for comparison
        
        # Import static data
        if not df_copy.empty:
            n_period.import_components_from_dataframe(df_copy, c_name)
        
        # Handle time-series data - extract only specified period
        for attr_name in c_pnl.keys():
            pnl_data = c_pnl[attr_name]
            if pnl_data.empty:
                continue
            
            # Filter to period's snapshots and reset index
            period_data = pnl_data.loc[period_snapshots].copy()
            period_data.index = single_snapshots
            
            # Filter to components that exist in the period network
            target_pnl = getattr(n_period, c_name.lower() + "s_t" if c_name != "StorageUnit" else "storage_units_t")
            target_df = getattr(n_period, c_name.lower() + "s" if c_name != "StorageUnit" else "storage_units")
            
            common_cols = period_data.columns.intersection(target_df.index)
            if not common_cols.empty:
                n_period.import_series_from_dataframe(
                    period_data[common_cols], c_name, attr_name
                )
    
    # Copy global constraints (filter to period if applicable)
    if not n.global_constraints.empty:
        gc_df = n.global_constraints.copy()
        
        # Filter constraints to specified period
        if "investment_period" in gc_df.columns:
            gc_df = gc_df[
                (gc_df.investment_period == period_year) | 
                (gc_df.investment_period.isna())
            ]
            # Remove year suffix from constraint names
            gc_df.index = gc_df.index.str.replace(f"-{period_year}", "", regex=False)
        
        if not gc_df.empty:
            n_period.import_components_from_dataframe(gc_df, "GlobalConstraint")
    
    # Extract and preserve temporal clustering period_id for the first period
    # This is needed for proper storage constraints in subsequent solves
    if hasattr(n, "temporal_cluster") and isinstance(n.temporal_cluster, dict):
        if "period_id" in n.temporal_cluster:
            multi_period_id = n.temporal_cluster["period_id"]
            # Filter period_id to specified investment period and convert back to single-index
            extracted_period_id = multi_period_id.loc[period_snapshots]
            # Strip the year prefix from period_id values (e.g., "2020_0" -> 0)
            period_id_values = [int(str(v).split("_")[1]) if "_" in str(v) else v for v in extracted_period_id.values]
            period_id = pd.Series(period_id_values, index=single_snapshots, name="period_id")
            
            # Store in the new network
            if not hasattr(n_period, "temporal_cluster") or not isinstance(n_period.temporal_cluster, dict):
                n_period.temporal_cluster = {}
            n_period.temporal_cluster["period_id"] = period_id
            
            # Also persist to meta for save/load
            if not hasattr(n_period, "meta") or not isinstance(n_period.meta, dict):
                n_period.meta = {}
            n_period.meta["temporal_cluster_period_id"] = period_id.to_json(orient="split")
            
            logger.info(f"Preserved period_id for period {period_year}: {len(period_id)} snapshots, {period_id.nunique()} unique periods")
    
    return n_period


def extract_first_period_results(n, first_year):
    """
    Extract and commit only the first investment period's results.
    
    Wrapper around extract_period_results() with commit_assets=True.
    This is used for brownfield continuation in rolling horizon.
    
    Parameters
    ----------
    n : pypsa.Network
        Solved multi-period network
    first_year : int
        The first investment period year
    
    Returns
    -------
    pypsa.Network
        Single-period network with committed first-period investments
    """
    return extract_period_results(n, first_year, commit_assets=True)


def cleanup_invalid_buses(n):
    """
    Remove components (generators, loads, stores, links) that reference buses
    which are not defined in the network.
    
    This prevents warnings and errors during optimization.
    
    Parameters
    ----------
    n : pypsa.Network
        Network to clean up
    """
    logger.info("Performing preemptive network cleanup...")
    buses_to_keep = set(n.buses.index)
    
    # Clean generators with undefined buses
    generators_with_bad_buses = n.generators[~n.generators.bus.isin(buses_to_keep)]
    if len(generators_with_bad_buses) > 0:
        logger.warning(f"Removing {len(generators_with_bad_buses)} generators with undefined buses before solving")
        logger.info(f"Generators with bad buses:\n{generators_with_bad_buses}")
        n.generators = n.generators.drop(generators_with_bad_buses.index)
        
        # Clean time series data for removed generators
        for attr in ['p_max_pu', 'p_min_pu']:
            if hasattr(n.generators_t, attr):
                attr_data = getattr(n.generators_t, attr)
                cols_to_remove = generators_with_bad_buses.index.intersection(attr_data.columns)
                if len(cols_to_remove) > 0:
                    setattr(n.generators_t, attr, attr_data.drop(columns=cols_to_remove))
    
    # Clean loads with undefined buses
    loads_with_bad_buses = n.loads[~n.loads.bus.isin(buses_to_keep)]
    if len(loads_with_bad_buses) > 0:
        logger.warning(f"Removing {len(loads_with_bad_buses)} loads with undefined buses before solving")
        logger.info(f"Loads with bad buses:\n{loads_with_bad_buses}")
        n.loads = n.loads.drop(loads_with_bad_buses.index)
        
        if hasattr(n.loads_t, 'p_set'):
            cols_to_remove = loads_with_bad_buses.index.intersection(n.loads_t.p_set.columns)
            if len(cols_to_remove) > 0:
                n.loads_t.p_set = n.loads_t.p_set.drop(columns=cols_to_remove)
    
    # Clean stores with undefined buses
    stores_with_bad_buses = n.stores[~n.stores.bus.isin(buses_to_keep)]
    if len(stores_with_bad_buses) > 0:
        logger.warning(f"Removing {len(stores_with_bad_buses)} stores with undefined buses before solving")
        logger.info(f"Stores with bad buses:\n{stores_with_bad_buses}")
        n.stores = n.stores.drop(stores_with_bad_buses.index)
    
    # Clean links with undefined buses (check both bus0 and bus1)
    links_with_bad_bus0 = n.links[~n.links.bus0.isin(buses_to_keep)]
    links_with_bad_bus1 = n.links[~n.links.bus1.isin(buses_to_keep)]
    links_with_bad_buses = links_with_bad_bus0.index.union(links_with_bad_bus1.index)
    
    if len(links_with_bad_buses) > 0:
        logger.warning(f"Removing {len(links_with_bad_buses)} links with undefined buses before solving")
        logger.info(f"Links with bad buses:\n{n.links.loc[links_with_bad_buses]}")
        n.links = n.links.drop(links_with_bad_buses)
        
        # Clean time series data for removed links
        for attr in ["p0", "p1", "p", "r", "p_min_pu", "p_max_pu", "efficiency", "marginal_cost"]:
            if hasattr(n.links_t, attr):
                attr_data = getattr(n.links_t, attr)
                if hasattr(attr_data, "columns"):
                    cols_to_remove = links_with_bad_buses.intersection(attr_data.columns)
                    if len(cols_to_remove) > 0:
                        setattr(n.links_t, attr, attr_data.drop(columns=cols_to_remove))
    
    logger.info("Network cleanup complete")


def solve_rolling_horizon_network(n, config, params, planning_horizons, log_fn=None):
    """
    Solve multi-period network and extract first period results.
    
    Parameters
    ----------
    n : pypsa.Network
        Multi-period network to solve
    config : dict
        Configuration dictionary
    params : dict
        Solving parameters
    planning_horizons : str
        Current planning horizon (first year in rolling window)
    log_fn : str, optional
        Path to solver log file
    
    Returns
    -------
    pypsa.Network
        Solved network with first period results extracted
    """
    solve_opts = params.solving["options"]
    cf_solving = params.solving["options"]

    np.random.seed(solve_opts.get("seed", 123))
    
    cleanup_invalid_buses(n)
    
    if solve_opts.get("cluster_to_single_node", False):
        logger.info("Clustering network to single node for solving...")
        n = cluster_network_to_single_node(n, world_tag="WORLD")
        logger.info(f"[cluster_to_single_node] clustered: {len(n.buses)} buses, "
                    f"{len(n.generators)} generators, {len(n.loads)} loads, {len(n.links)} links")
        
    # Prepare network settings (prepare_network only takes solve_opts)
    n = prepare_network(n, solve_opts)
    
    # Add land use constraint for multi-period (currently a no-op, can be extended)
    if not solve_opts.get("cluster_to_single_node", False):
        add_land_use_constraint_perfect(n)
    else:
        logger.warning("[cluster_to_single_node] skipping land-use constraints (bus-level potentials invalid after aggregation)")

    
    # Handle skip_iterations (PyPSA-Earth config option, not a solver parameter)
    # For rolling horizon multi-period optimization, we always skip iterations
    skip_iterations = cf_solving.get("skip_iterations", True)  # Default to True for multi-period
    if not n.lines.s_nom_extendable.any():
        skip_iterations = True
        logger.info("No expandable lines found. Skipping iterative solving.")
    
    # Collect solving kwargs (skip_iterations is NOT included - it's handled above)
    solve_kwargs = collect_kwargs(
        config,
        params.solving,
        planning_horizons=planning_horizons,  # Multi-period
        log_fn=log_fn,
        mode="single",
    )
    
    # Set config and params on network for extra_functionality
    n.config = config
    n.params = params
    n.planning_horizons = planning_horizons  # Store for learning check
    if not hasattr(n, "opts"):
        n.opts = []  # Can be extended if needed
    
    # Zero out capital costs for learning assets BEFORE optimization
    # This happens before n.optimize() so PyPSA builds objective with capital_cost=0
    learning_cfg = config.get("learning", {}) or {}
    if learning_cfg.get("enabled", False):
        # Hardcoded skip for first horizon (matches extra_functionality_rolling_horizon)
        # 2020 uses exogenous costs from historical data
        skip_learning_years = learning_cfg.get("skip_learning_years", [2020])
        
        current_year = None
        try:
            current_year = int(planning_horizons)
        except (ValueError, TypeError):
            pass
        
        if current_year not in skip_learning_years:            
            # Get technologies from config
            techs = learning_cfg.get("technologies", [])
            
            # Get window years and determine priced periods (lag=1 means only 2nd period is priced)
            window_years = list(n.investment_periods) if hasattr(n, "investment_periods") else []
            lag_periods = int(learning_cfg.get("lag_periods", 1))
            
            if lag_periods >= len(window_years):
                priced_periods = []  # All periods exogenous
            else:
                priced_periods = window_years[lag_periods:]  # e.g., for [2025,2030] with lag=1 -> [2030]
            
            # Build tech-to-carriers mapping
            TECH_TO_CARRIERS = build_TECH_TO_CARRIERS(learning_cfg)
            
            if techs and priced_periods:
                logger.info("="*80)
                logger.info("ZEROING CAPITAL COSTS FOR LEARNING ASSETS (BEFORE OPTIMIZATION)")
                logger.info("="*80)
                logger.info(f"Technologies: {techs}")
                logger.info(f"Priced periods: {priced_periods}")
                zero_learning_capital_costs(n, techs, priced_periods, TECH_TO_CARRIERS)
            else:
                logger.info(f"Learning enabled but no priced periods (window={window_years}, lag={lag_periods}) or no technologies defined!")
                logger.warning(f'WARNING! Capital costs might be applied incorrectly!')
        else:
            logger.info(f"Skipping capital cost zeroing for {current_year} (exogenous costs)")
    
    # Solve using n.optimize() which handles model creation and solving
    # For rolling horizon, we always use direct optimization (skip_iterations=True)
    logger.info("Solving multi-period optimization...")
    
    save_lpfile = params.solving.get("save_lpfile", False)
    if save_lpfile:
        logger.info(f"Saving linopy model to {snakemake.output.lpfile.replace('.lp', '.nc')}")
        try:
            n.model.to_netcdf(snakemake.output.lpfile.replace('.lp', '.nc'))
            logger.info("Linopy model saved successfully")
        except Exception as e:
            logger.warning(f"Could not save linopy model: {e}")

        logger.info(f"Saving LP file to {snakemake.output.lpfile}")
        try:
            n.model.to_file(snakemake.output.lpfile)
            logger.info("LP file saved successfully")
        except Exception as e:
            logger.warning(f"Could not save LP file: {e}")
    elif save_lpfile:
        logger.warning("LP file saving is enabled but lpfile output not defined in Snakefile")
    else:
        logger.info("LP file saving is disabled (set solving.save_lpfile: true to enable)")
    
    status, condition = n.optimize(**solve_kwargs)
    
    if status != "ok":
        logger.warning(
            f"Solving status '{status}' with termination condition '{condition}'"
        )
    
    if "infeasible" in condition:
        labels = n.model.compute_infeasibilities()
        logger.info(f"Labels:\n{labels}")
        n.model.print_infeasibilities()
        raise RuntimeError("Solving status 'infeasible'. Infeasibilities computed.")
    
    if "warning" in condition:
        raise RuntimeError("Solving status 'warning'. Discarding solution.")
    
    logger.info(f"Optimization complete. Objective value: {n.objective}")
    
    # Debug logging for objective components
    try:
        if hasattr(n.model, 'objective') and hasattr(n.model.objective, 'expression'):
            obj_expr = n.model.objective.expression
            if hasattr(obj_expr, 'const'):
                logger.info(f"  Objective constant term: {float(obj_expr.const):,.0f} EUR")
            # Try to get coefficient stats
            if hasattr(obj_expr, 'coeffs'):
                logger.info(f"  Objective has variable coefficients (learning costs)")
    except Exception as e:
        logger.info(f"  Could not extract objective details: {e}")
    
    return n, status, condition


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_rolling_horizon",
            simpl="",
            clusters="110",
            ll="copt",
            opts="1h",
            planning_horizons="2020",
            sopts="1h",
            discountrate="0.071",
            demand="AB",
            h2export="0.0",
            learning_rate="base",
        )

    logging.basicConfig(level=logging.INFO)
    
    learning_cfg = snakemake.config.get("learning", {}) if hasattr(snakemake, "config") else {}

    # Load the multi-period network
    n = pypsa.Network(snakemake.input.network)
    
    # Get the rolling window years from network meta
    window_years = n.meta.get("rolling_horizon_window", [])
    if not window_years:
        # Fallback: use investment periods from network
        window_years = list(n.investment_periods)
    
    first_year = window_years[0]
    
    logger.info(f"Solving rolling horizon network for window: {window_years}")
    logger.info(f"First year (to be saved): {first_year}")
    
    # Solve the multi-period network
    n_solved, status, condition = solve_rolling_horizon_network(
        n,
        config=snakemake.config,
        params=snakemake.params,
        planning_horizons=str(first_year),
        log_fn=getattr(snakemake.log, "solver", None),
    )

    # Post-solve learning diagnostics (best-effort)
    try:
        _validate_learning_solution(n_solved)
    except Exception as e:
        logger.warning(f"[learning] validation skipped due to error: {e}")
    
    # Extract first period results
    n_first = extract_first_period_results(n_solved, first_year)

    # Update meta
    n_first.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n_first.meta["rolling_horizon_solved"] = True
    n_first.meta["rolling_horizon_window"] = window_years
    n_first.meta["objective_value"] = n_solved.objective

    # Persist learning state (committed cumulative capacity) for the committed period
    # This saves the actual built capacity regardless of whether learning was enabled
    if learning_cfg.get("persist_state", True):
        try:
            df_state = _extract_learning_state_committed(n_first, first_year, snakemake.config)
            if not df_state.empty:
                n_first.meta["learning_state_committed"] = df_state.to_dict(orient="records")
                state_path = snakemake.output.learning_state
                df_state.to_csv(state_path, index=False)
                logger.info(f"Saved learning state to {state_path}")
            else:
                logger.warning("df_state empty - no capacity extracted")
        except Exception as e:
            logger.exception(f"Failed to persist learning state: {e}")
            raise
    
    # Export the first period network (committed for brownfield continuation)
    n_first.export_to_netcdf(snakemake.output.network)
    logger.info(f"First period network saved to {snakemake.output.network}")
    
    # Optionally save second period network for validation/comparison
    if learning_cfg.get("save_rolling_second", False) and len(window_years) > 1:
        second_year = window_years[1]
        logger.info(f"Extracting second period for comparison: {second_year}")
        
        # Extract second period WITHOUT committing (for comparison only)
        n_second = extract_period_results(n_solved, second_year, commit_assets=False)
        
        # Update meta
        n_second.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
        n_second.meta["rolling_horizon_solved"] = True
        n_second.meta["rolling_horizon_window"] = window_years
        n_second.meta["rolling_horizon_period"] = "second"
        n_second.meta["rolling_horizon_comparison"] = True
        n_second.meta["objective_value"] = n_solved.objective
        
        # Check if second network output is defined
        second_network_output = snakemake.params.network_second
        if second_network_output:
            # Ensure the directory exists
            import os
            os.makedirs(os.path.dirname(second_network_output), exist_ok=True)
            n_second.export_to_netcdf(second_network_output)
            logger.info(f"Second period network saved to {second_network_output}")
        else:
            # Fallback: save to a default location based on first period path
            import os
            base_path = snakemake.output.network
            second_path = base_path.replace(".nc", "_second.nc")
            # Create a second network directory if it doesn't exist
            second_dir = os.path.join(os.path.dirname(base_path), "second_period")
            os.makedirs(second_dir, exist_ok=True)
            second_path = os.path.join(second_dir, os.path.basename(base_path))
            n_second.export_to_netcdf(second_path)
            logger.info(f"Second period network saved to {second_path} (fallback location)")
    elif learning_cfg.get("save_rolling_second", False):
        logger.warning("save_rolling_second enabled but window has only one period; skipping")
    
    # Save config
    with open(snakemake.output.config, "w") as file:
        yaml.dump(
            n_first.meta,
            file,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )