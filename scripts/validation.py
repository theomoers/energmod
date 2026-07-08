# -*- coding: utf-8 -*-
"""Validation/tuning helper hooks used by core workflow scripts.

This module centralizes validation/tuning extensions so `prepare_sector_network.py`,
`add_existing_baseyear.py`, and `solve_network.py` only need thin imports/aliases.
"""

import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import xarray as xr

from _helpers import BASE_DIR, country_name_2_two_digits, three_2_two_digits_country

logger = logging.getLogger(__name__)


BASEYEAR_BLOCKED_EXTENDABLE_LINK_CARRIERS = ("H2 Fuel Cell",)
STRUCTURAL_BIOMASS_POWER_CARRIERS = (
    "biomass",
    "biomass EOP",
    "urban central solid biomass CHP",
    "urban central solid biomass CHP CC",
)
_STRUCTURAL_BIOMASS_EFFICIENCY_CACHE = {}


def _get_bus_country_for_clustering(n):
    country = (
        n.buses["country"]
        if "country" in n.buses.columns
        else pd.Series("", index=n.buses.index)
    )
    location = (
        n.buses["location"]
        if "location" in n.buses.columns
        else pd.Series("", index=n.buses.index)
    )
    country = country.fillna("").astype(str).str.strip().str.upper()
    location = location.fillna("").astype(str).str.strip().str.upper()
    bus_name = n.buses.index.to_series(index=n.buses.index).astype(str).str.strip().str.upper()

    return (
        country.where(country != "")
        .fillna(location.str.extract(r"^([A-Z]{2})\b", expand=False))
        .fillna(bus_name.str.extract(r"^([A-Z]{2})\b", expand=False))
        .fillna("")
    )


def _profile_col_bus(col, generator, suffix, year):
    marker = f" {generator}{suffix}-{year}"
    if isinstance(col, str) and col.endswith(marker):
        return col[: -len(marker)]
    return None


def _replace_zero_profile_columns_with_nearest(
    n, p_max_pu, generator, suffix, source_year, target_year
):
    """
    Replace all-zero p_max_pu columns by copying the nearest non-zero profile of the
    same generator technology (typically from a neighboring country).
    """
    if p_max_pu.empty:
        return p_max_pu

    zero_cols = p_max_pu.columns[p_max_pu.fillna(0.0).sum(axis=0).abs() <= 1e-12]
    if len(zero_cols) == 0:
        return p_max_pu

    pool_cols = [
        c for c in n.generators_t.p_max_pu.columns
        if isinstance(c, str) and f" {generator}{suffix}-" in c
    ]
    if not pool_cols:
        logger.warning(
            "No fallback p_max_pu pool found for %s zero-profile replacement (%d columns).",
            generator,
            len(zero_cols),
        )
        return p_max_pu

    pool = n.generators_t.p_max_pu[pool_cols].fillna(0.0)
    nonzero_pool_cols = pool.columns[pool.sum(axis=0).abs() > 1e-12]
    if len(nonzero_pool_cols) == 0:
        logger.warning(
            "All candidate p_max_pu profiles are zero for %s; cannot replace %d zero columns.",
            generator,
            len(zero_cols),
        )
        return p_max_pu

    # Build bus coordinate table for candidate source profiles.
    src_records = []
    for col in nonzero_pool_cols:
        bus = _profile_col_bus(col, generator, suffix, source_year)
        if bus is None:
            # Accept any year in the pool as fallback.
            m = re.search(rf"^(.*) {re.escape(generator + suffix)}-\d+$", str(col))
            bus = m.group(1) if m else None
        if bus is None:
            continue
        if bus in n.buses.index and pd.notna(n.buses.at[bus, "x"]) and pd.notna(n.buses.at[bus, "y"]):
            src_records.append((col, bus, float(n.buses.at[bus, "x"]), float(n.buses.at[bus, "y"])))

    if not src_records:
        template = pool[nonzero_pool_cols[0]]
        for col in zero_cols:
            p_max_pu[col] = template.values
        logger.warning(
            "Replaced %d zero %s p_max_pu profiles using generic fallback template %s (no bus coordinates).",
            len(zero_cols),
            generator,
            nonzero_pool_cols[0],
        )
        return p_max_pu

    src_df = pd.DataFrame(src_records, columns=["col", "bus", "x", "y"]).drop_duplicates(subset=["col"])
    replaced = []
    failed = []
    for col in zero_cols:
        target_bus = _profile_col_bus(col, generator, suffix, target_year)
        if target_bus is None:
            m = re.search(rf"^(.*) {re.escape(generator + suffix)}-\d+$", str(col))
            target_bus = m.group(1) if m else None
        if (
            target_bus is None
            or target_bus not in n.buses.index
            or pd.isna(n.buses.at[target_bus, "x"])
            or pd.isna(n.buses.at[target_bus, "y"])
        ):
            failed.append((col, target_bus))
            continue

        tx = float(n.buses.at[target_bus, "x"])
        ty = float(n.buses.at[target_bus, "y"])
        d2 = (src_df["x"] - tx) ** 2 + (src_df["y"] - ty) ** 2
        best = src_df.loc[d2.idxmin()]
        p_max_pu[col] = pool[best["col"]].values
        replaced.append((col, target_bus, best["col"], best["bus"]))

    if replaced:
        sample = ", ".join(
            [f"{tbus}<-{sbus}" for _, tbus, _, sbus in replaced[:5]]
        )
        logger.warning(
            "Replaced %d zero %s p_max_pu profiles using nearest non-zero fallback (sample: %s%s).",
            len(replaced),
            generator,
            sample,
            " ..." if len(replaced) > 5 else "",
        )
    if failed:
        logger.warning(
            "Could not replace %d zero %s p_max_pu profiles due to missing target bus coordinates.",
            len(failed),
            generator,
        )

    return p_max_pu


def apply_renewable_profile_fallbacks(n, investment_year, config):
    """
    Patch missing or all-zero renewable p_max_pu profiles without changing
    valid profiles.

    This is intentionally separate from the baseyear OWID/scaling hooks so
    later planning years keep their original CFs except where profiles are
    missing or pathological.
    """
    global_cfg = config.get("global_specific", {}) if isinstance(config, dict) else {}
    base_cfg = global_cfg.get("baseyear_generation", {})
    if not bool(base_cfg.get("renewable_profile_fallback_enabled", True)):
        return {
            "enabled": False,
            "carriers": {},
        }

    if n.generators.empty:
        return {
            "enabled": True,
            "carriers": {},
        }

    if n.generators_t.p_max_pu.empty:
        n.generators_t.p_max_pu = pd.DataFrame(index=n.snapshots)

    fallback_specs = [
        ("onwind", "onwind", ""),
        ("solar", "solar", ""),
        ("offwind-ac", "offwind", "-ac"),
        ("offwind-dc", "offwind", "-dc"),
    ]

    summary = {}
    for carrier, generator, suffix in fallback_specs:
        idx = n.generators.index[n.generators.carrier.astype(str).eq(carrier)]
        if len(idx) == 0:
            continue

        ts_cols_existing = n.generators_t.p_max_pu.columns.intersection(idx)
        missing_cols = idx.difference(ts_cols_existing)

        patch_cols = pd.Index(ts_cols_existing)
        if len(missing_cols) > 0:
            patch_cols = patch_cols.append(missing_cols)

        if len(patch_cols) == 0:
            continue

        p_max_pu_patch = n.generators_t.p_max_pu.reindex(columns=patch_cols).copy()
        if len(missing_cols) > 0:
            p_max_pu_patch.loc[:, missing_cols] = 0.0

        zero_before = (
            p_max_pu_patch.fillna(0.0).sum(axis=0).abs() <= 1e-12
        )
        pathological_before = int(zero_before.sum())
        if pathological_before == 0:
            continue

        p_max_pu_patch = _replace_zero_profile_columns_with_nearest(
            n=n,
            p_max_pu=p_max_pu_patch,
            generator=generator,
            suffix=suffix,
            source_year=investment_year,
            target_year=investment_year,
        )

        for col in p_max_pu_patch.columns:
            n.generators_t.p_max_pu.loc[:, col] = p_max_pu_patch[col].values

        zero_after = (
            p_max_pu_patch.fillna(0.0).sum(axis=0).abs() <= 1e-12
        )
        pathological_after = int(zero_after.sum())
        patched = pathological_before - pathological_after

        summary[carrier] = {
            "generators": int(len(idx)),
            "missing_profiles": int(len(missing_cols)),
            "pathologies_before": pathological_before,
            "patched": int(patched),
            "residual_after": pathological_after,
        }

    if summary:
        logger.warning(
            "Renewable profile fallback summary for %s: %s",
            investment_year,
            "; ".join(
                (
                    f"{carrier}(gens={stats['generators']}, missing={stats['missing_profiles']}, "
                    f"before={stats['pathologies_before']}, patched={stats['patched']}, "
                    f"residual={stats['residual_after']})"
                )
                for carrier, stats in summary.items()
            ),
        )

    return {
        "enabled": True,
        "carriers": summary,
    }




def _safe_iso3_to_iso2(code):
    if not isinstance(code, str) or len(code) != 3:
        return np.nan
    try:
        iso2 = three_2_two_digits_country(code)
    except Exception:
        return np.nan
    if isinstance(iso2, str) and len(iso2) == 2:
        return iso2
    return np.nan


def _repo_path(path_like):
    path_str = str(path_like)
    return path_str if os.path.isabs(path_str) else os.path.join(BASE_DIR, path_str)


def _structural_biomass_cfg(config):
    global_cfg = config.get("global_specific", {}) if isinstance(config, dict) else {}
    cfg = global_cfg.get("post2020_structural_biomass", {})
    return cfg if isinstance(cfg, dict) else {}


def _modeled_countries_from_config(config):
    countries = config.get("countries", []) if isinstance(config, dict) else []
    return {str(country).strip().upper() for country in countries if str(country).strip()}


def _load_historical_biofuel_electricity_summary(config):
    cfg = _structural_biomass_cfg(config)
    owid_csv = _repo_path(cfg.get("owid_csv", "validation/data/owid-energy-data.csv"))
    if not os.path.exists(owid_csv):
        raise FileNotFoundError(
            f"Structural biomass allocation requires OWID data at {owid_csv}"
        )

    owid = pd.read_csv(
        owid_csv,
        usecols=lambda c: c in {"year", "iso_code", "biofuel_electricity"},
    )
    if "biofuel_electricity" not in owid.columns:
        raise ValueError(
            f"Column 'biofuel_electricity' not found in structural biomass OWID file {owid_csv}"
        )

    owid["country"] = owid["iso_code"].apply(_safe_iso3_to_iso2)
    modeled_countries = _modeled_countries_from_config(config)
    if modeled_countries:
        owid = owid.loc[owid["country"].isin(modeled_countries)].copy()

    owid["year"] = pd.to_numeric(owid["year"], errors="coerce")
    owid["biofuel_electricity"] = pd.to_numeric(
        owid["biofuel_electricity"], errors="coerce"
    )
    owid = owid.dropna(subset=["year"])
    owid["year"] = owid["year"].astype(int)

    summary = (
        owid.groupby("year").agg(
            biofuel_electricity=("biofuel_electricity", lambda s: pd.to_numeric(s, errors="coerce").fillna(0.0).sum()),
            reporting_countries=("biofuel_electricity", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
        )
        .sort_index()
        .astype({"biofuel_electricity": float, "reporting_countries": int})
    )
    return summary


def _load_global_historical_biofuel_electricity_twh(config):
    summary = _load_historical_biofuel_electricity_summary(config)
    return summary["biofuel_electricity"].astype(float)


def _select_structural_biomass_history_year(summary, target_year, config):
    target_year = int(target_year)
    if summary.empty:
        raise ValueError("No historical global biofuel_electricity values available.")

    cfg = _structural_biomass_cfg(config)
    min_ratio = float(cfg.get("minimum_reporting_ratio", 0.8) or 0.8)
    max_countries = int(summary["reporting_countries"].max()) if not summary.empty else 0
    if max_countries <= 0:
        raise ValueError("Historical biofuel_electricity summary has no reporting countries.")

    eligible = summary.loc[
        summary["reporting_countries"] >= max(1, int(np.ceil(max_countries * min_ratio)))
    ].copy()
    if eligible.empty:
        eligible = summary.copy()

    if target_year in eligible.index:
        return int(target_year)

    earlier = eligible.loc[eligible.index <= target_year]
    if bool(cfg.get("freeze_last_historical_value", True)) and not earlier.empty:
        return int(earlier.index.max())

    if target_year in summary.index:
        return int(target_year)

    earlier_any = summary.loc[summary.index <= target_year]
    if bool(cfg.get("freeze_last_historical_value", True)) and not earlier_any.empty:
        return int(earlier_any.index.max())

    raise ValueError(
        f"No historical biofuel_electricity value available for structural biomass year {target_year}."
    )


def _representative_biomass_electric_efficiency(config):
    cfg = _structural_biomass_cfg(config)
    base_cfg = config.get("global_specific", {}).get("baseyear_generation", {})
    baseline_network = cfg.get(
        "baseline_network",
        base_cfg.get("electricity_demand_baseline_network"),
    )
    if not baseline_network:
        raise ValueError(
            "Structural biomass allocation requires a baseline solved network path."
        )

    baseline_path = _repo_path(baseline_network)
    cache_key = os.path.abspath(baseline_path)
    if cache_key in _STRUCTURAL_BIOMASS_EFFICIENCY_CACHE:
        return _STRUCTURAL_BIOMASS_EFFICIENCY_CACHE[cache_key]

    if not os.path.exists(baseline_path):
        raise FileNotFoundError(
            f"Structural biomass allocation baseline network not found at {baseline_path}"
        )

    n = pypsa.Network(baseline_path)
    weights = _snapshot_generator_weights(n)
    total_electricity_mwh = 0.0
    total_biomass_input_mwh = 0.0

    if not n.generators.empty and not n.generators_t.p.empty:
        generators = n.generators.loc[
            n.generators.carrier.astype(str).isin(STRUCTURAL_BIOMASS_POWER_CARRIERS)
        ].copy()
        if not generators.empty:
            generators = generators.loc[
                generators.bus.map(n.buses.carrier).fillna("").eq("AC")
            ].copy()
            if not generators.empty:
                dispatch = (
                    n.generators_t.p.reindex(columns=generators.index)
                    .fillna(0.0)
                    .mul(weights, axis=0)
                    .sum(axis=0)
                )
                efficiency = pd.to_numeric(
                    generators["efficiency"], errors="coerce"
                ).replace(0.0, np.nan)
                total_electricity_mwh += float(dispatch.sum())
                total_biomass_input_mwh += float(
                    dispatch.div(efficiency).replace([np.inf, -np.inf], np.nan).fillna(0.0).sum()
                )

    if not n.links.empty and not n.links_t.p0.empty:
        links = n.links.loc[
            n.links.carrier.astype(str).isin(STRUCTURAL_BIOMASS_POWER_CARRIERS)
        ].copy()
        if not links.empty:
            links = links.loc[
                links.bus1.map(n.buses.carrier).fillna("").eq("AC")
            ].copy()
            if not links.empty:
                p0 = (
                    n.links_t.p0.reindex(columns=links.index)
                    .fillna(0.0)
                    .clip(lower=0.0)
                    .mul(weights, axis=0)
                    .sum(axis=0)
                )
                efficiency = pd.to_numeric(
                    links["efficiency"], errors="coerce"
                ).fillna(0.0).abs()
                total_biomass_input_mwh += float(p0.sum())
                total_electricity_mwh += float(p0.mul(efficiency).sum())

    if total_biomass_input_mwh <= 0.0 or total_electricity_mwh <= 0.0:
        raise ValueError(
            "Could not derive representative biomass-electric efficiency from baseline network."
        )

    efficiency = total_electricity_mwh / total_biomass_input_mwh
    _STRUCTURAL_BIOMASS_EFFICIENCY_CACHE[cache_key] = efficiency
    logger.info(
        "Derived representative biomass-electric efficiency %.4f from %s",
        efficiency,
        baseline_path,
    )
    return efficiency


def derive_post2020_structural_biomass_allocation(
    investment_year,
    config,
    physical_total_twh,
    energy_totals=None,
    industrial_demand=None,
):
    cfg = _structural_biomass_cfg(config)
    if not cfg.get("enable", False):
        return None

    year = int(investment_year)
    start_year = int(cfg.get("start_year", 2025))
    if year < start_year:
        return None

    configured_physical_total_twh = float(physical_total_twh)
    historical_summary = _load_historical_biofuel_electricity_summary(config)
    if historical_summary.empty:
        raise ValueError("No historical global biofuel_electricity values available.")

    history_year = _select_structural_biomass_history_year(historical_summary, year, config)
    bioelectricity_twh = float(historical_summary.loc[history_year, "biofuel_electricity"])
    reporting_countries = int(historical_summary.loc[history_year, "reporting_countries"])

    representative_efficiency = _representative_biomass_electric_efficiency(config)
    power_required_twh = bioelectricity_twh / representative_efficiency

    industry_required_twh = 0.0
    if industrial_demand is not None and "solid biomass" in industrial_demand.columns:
        industry_required_twh = float(
            pd.to_numeric(industrial_demand["solid biomass"], errors="coerce")
            .fillna(0.0)
            .sum()
            / 1e6
        )

    buildings_required_twh = 0.0
    if energy_totals is not None:
        for col in ["services biomass", "residential biomass", "residential heat biomass"]:
            if col in energy_totals.columns:
                buildings_required_twh += float(
                    pd.to_numeric(energy_totals[col], errors="coerce").fillna(0.0).sum()
                )

    total_required_twh = power_required_twh + industry_required_twh + buildings_required_twh
    physical_total_twh = max(configured_physical_total_twh, total_required_twh)
    excess_twh = max(total_required_twh - physical_total_twh, 0.0)
    rel_tolerance = float(cfg.get("physical_balance_relative_tolerance", 0.0) or 0.0)
    abs_tolerance = float(cfg.get("physical_balance_absolute_twh", 0.0) or 0.0)
    tolerance_twh = max(abs_tolerance, physical_total_twh * rel_tolerance)
    feasible = excess_twh <= max(tolerance_twh, 1e-9)
    residual_twh = max(physical_total_twh - total_required_twh, 0.0)

    allocation = pd.Series(
        {
            "year": year,
            "history_year": history_year,
            "physical_total_twh": physical_total_twh,
            "configured_physical_total_twh": configured_physical_total_twh,
            "power_required_twh": power_required_twh,
            "industry_required_twh": industry_required_twh,
            "buildings_required_twh": buildings_required_twh,
            "unallocated_residual_twh": residual_twh,
            "bioelectricity_reference_twh": bioelectricity_twh,
            "representative_efficiency": representative_efficiency,
            "reporting_countries": reporting_countries,
            "physical_feasible": bool(feasible),
        }
    )
    logger.info(
        "Structural biomass allocation for %s: physical=%.3f TWh, power=%.3f TWh, industry=%.3f TWh, buildings=%.3f TWh, residual=%.3f TWh (history_year=%s, bioelectricity=%.3f TWh, eta=%.4f)",
        year,
        physical_total_twh,
        power_required_twh,
        industry_required_twh,
        buildings_required_twh,
        residual_twh,
        history_year,
        bioelectricity_twh,
        representative_efficiency,
    )
    return allocation


def _bus_country_lookup(n):
    country = (
        n.buses["country"].replace("", np.nan)
        if "country" in n.buses.columns
        else pd.Series(np.nan, index=n.buses.index)
    )
    location_iso2 = (
        n.buses["location"].astype(str).str.extract(r"^([A-Z]{2})\b")[0]
        if "location" in n.buses.columns
        else pd.Series(np.nan, index=n.buses.index)
    )
    index_iso2 = (
        n.buses.index.astype(str).to_series(index=n.buses.index).str.extract(r"^([A-Z]{2})\b")[0]
    )
    return country.fillna(location_iso2).fillna(index_iso2).fillna("")


def _snapshot_weight_sum_for_load_energy(n):
    if "generators" in n.snapshot_weightings:
        weights = n.snapshot_weightings["generators"]
    elif "objective" in n.snapshot_weightings:
        weights = n.snapshot_weightings["objective"]
    else:
        weights = pd.Series(1.0, index=n.snapshots)
    weights = pd.to_numeric(weights, errors="coerce").reindex(n.snapshots).fillna(0.0)
    return float(weights.sum())


def _add_missing_direct_electricity_loads(
    n,
    comp_before,
    investment_year,
    min_reference_twh=0.1,
):
    if comp_before.empty:
        return []

    weight_sum = _snapshot_weight_sum_for_load_energy(n)
    if weight_sum <= 0.0:
        logger.warning(
            "Cannot add missing direct electricity loads for %s: non-positive snapshot weight sum.",
            investment_year,
        )
        return []

    bus_country = _bus_country_lookup(n).astype(str).str.strip().str.upper()
    added = []
    missing = comp_before.loc[
        comp_before["direct_model_twh_before"].fillna(0.0).le(1e-9)
        & comp_before["direct_reference_twh"].fillna(0.0).gt(float(min_reference_twh))
    ]

    for country, row in missing.iterrows():
        country = str(country).upper()
        ac_bus = _pick_country_ac_bus(n, country, bus_country)
        if ac_bus is None:
            logger.warning(
                "Skipping missing direct electricity load for %s in %s: no AC bus.",
                country,
                investment_year,
            )
            continue

        direct_reference_twh = float(row["direct_reference_twh"])
        annual_mwh = direct_reference_twh * 1e6
        p_set = annual_mwh / weight_sum
        base_name = f"{ac_bus} validation electricity demand-{int(investment_year)}"
        name = base_name
        suffix = 2
        while name in n.loads.index:
            name = f"{base_name}-{suffix}"
            suffix += 1

        n.add("Load", name, bus=ac_bus, carrier="electricity")
        n.loads_t.p_set[name] = p_set
        added.append((country, name, ac_bus, direct_reference_twh))

    if added:
        logger.info(
            "Added missing direct electricity loads for %s: %s",
            investment_year,
            ", ".join(
                f"{country}={direct_reference_twh:.3f} TWh on {bus}"
                for country, _name, bus, direct_reference_twh in added
            ),
        )
    return added


def _effective_p_nom_mw(df):
    if df.empty:
        return pd.Series(dtype=float)

    p_nom = (
        pd.to_numeric(df["p_nom"], errors="coerce")
        if "p_nom" in df.columns
        else pd.Series(0.0, index=df.index, dtype=float)
    )
    p_nom = p_nom.reindex(df.index).fillna(0.0).astype(float)

    if "p_nom_opt" in df.columns:
        p_nom_opt = pd.to_numeric(df["p_nom_opt"], errors="coerce")
        p_nom_opt = p_nom_opt.reindex(df.index).astype(float)
        # Prenetworks often carry `p_nom_opt=0`; prefer nominal capacity in that case.
        p_nom_opt = p_nom_opt.where(p_nom_opt > 0.0)
        return p_nom_opt.fillna(p_nom).fillna(0.0)

    return p_nom


def _snapshot_generator_weights(n):
    try:
        weights = n.snapshot_weightings["generators"]
    except Exception:
        weights = pd.Series(1.0, index=n.snapshots, dtype=float)
    return pd.to_numeric(weights, errors="coerce").reindex(n.snapshots).fillna(1.0).astype(float)


def _hydro_profile_pathology_assets(n, zero_twh_tol=1e-9):
    """
    Asset-level hydro profile diagnostics for prenetwork structures.

    Flags missing/zero reservoir inflow time series and missing/zero run-of-river
    (`ror`) availability profiles for non-zero capacity assets.
    """
    rows = []
    bus_country = _bus_country_lookup(n)
    weights = _snapshot_generator_weights(n)
    weight_sum = float(weights.sum())

    if not n.storage_units.empty:
        su = n.storage_units.loc[n.storage_units["carrier"].astype(str).eq("hydro")].copy()
        if not su.empty:
            su["capacity_mw"] = _effective_p_nom_mw(su)
            su = su.loc[su["capacity_mw"] > 0.0].copy()
            if not su.empty:
                su["bus"] = su["bus"].fillna("").astype(str)
                su["country"] = su["bus"].map(bus_country).fillna("")
                inflow_cols = (
                    n.storage_units_t.inflow.columns.intersection(su.index)
                    if not n.storage_units_t.inflow.empty
                    else pd.Index([])
                )
                inflow_twh = pd.Series(0.0, index=su.index, dtype=float)
                if len(inflow_cols) > 0:
                    inflow_twh.loc[inflow_cols] = (
                        n.storage_units_t.inflow.reindex(columns=inflow_cols)
                        .fillna(0.0)
                        .mul(weights, axis=0)
                        .sum(axis=0)
                        .astype(float)
                        / 1e6
                    )

                has_col = su.index.isin(inflow_cols)
                has_col = pd.Series(has_col, index=su.index, dtype=bool)
                profile_all_zero = has_col & inflow_twh.le(zero_twh_tol)
                missing_profile = ~has_col
                needs_fallback = missing_profile | profile_all_zero
                issue = pd.Series("", index=su.index, dtype=object)
                issue.loc[missing_profile] = "missing_reservoir_inflow"
                issue.loc[profile_all_zero] = "zero_reservoir_inflow"
                issue.loc[needs_fallback & issue.eq("")] = "reservoir_profile_pathology"

                su_out = su.reset_index().rename(columns={su.index.name or "index": "asset"})
                su_out["component"] = "StorageUnit"
                su_out["carrier"] = "hydro"
                su_out["available_twh"] = inflow_twh.reindex(su_out["asset"]).values
                su_out["has_profile_column"] = has_col.reindex(su_out["asset"]).values
                su_out["profile_all_zero"] = profile_all_zero.reindex(su_out["asset"]).values
                su_out["missing_profile"] = missing_profile.reindex(su_out["asset"]).values
                su_out["needs_fallback"] = needs_fallback.reindex(su_out["asset"]).values
                su_out["issue"] = issue.reindex(su_out["asset"]).values
                rows.extend(
                    su_out[
                        [
                            "component",
                            "carrier",
                            "asset",
                            "bus",
                            "country",
                            "capacity_mw",
                            "available_twh",
                            "has_profile_column",
                            "missing_profile",
                            "profile_all_zero",
                            "needs_fallback",
                            "issue",
                        ]
                    ].to_dict("records")
                )

    if not n.generators.empty:
        ror = n.generators.loc[n.generators["carrier"].astype(str).eq("ror")].copy()
        if not ror.empty:
            ror["capacity_mw"] = _effective_p_nom_mw(ror)
            ror = ror.loc[ror["capacity_mw"] > 0.0].copy()
            if not ror.empty:
                ror["bus"] = ror["bus"].fillna("").astype(str)
                ror["country"] = ror["bus"].map(bus_country).fillna("")

                ts_cols = (
                    n.generators_t.p_max_pu.columns.intersection(ror.index)
                    if not n.generators_t.p_max_pu.empty
                    else pd.Index([])
                )
                ts_cols = pd.Index(ts_cols)
                has_ts = pd.Series(ror.index.isin(ts_cols), index=ror.index, dtype=bool)
                availability_hours = pd.Series(0.0, index=ror.index, dtype=float)
                if len(ts_cols) > 0:
                    availability_hours.loc[ts_cols] = (
                        n.generators_t.p_max_pu.reindex(columns=ts_cols)
                        .fillna(0.0)
                        .mul(weights, axis=0)
                        .sum(axis=0)
                        .astype(float)
                    )
                static_cols = ror.index.difference(ts_cols)
                if len(static_cols) > 0:
                    static_pmax = (
                        pd.to_numeric(ror.loc[static_cols, "p_max_pu"], errors="coerce")
                        .fillna(0.0)
                        .clip(lower=0.0)
                    )
                    availability_hours.loc[static_cols] = static_pmax * weight_sum

                available_twh = availability_hours.mul(ror["capacity_mw"]).div(1e6)
                missing_profile = ~has_ts
                # Zero-profile fallback applies to explicit time-series profiles only.
                profile_all_zero = has_ts & available_twh.le(zero_twh_tol)
                needs_fallback = missing_profile | profile_all_zero
                issue = pd.Series("", index=ror.index, dtype=object)
                issue.loc[missing_profile] = "missing_ror_profile"
                issue.loc[profile_all_zero] = "zero_ror_profile"
                issue.loc[needs_fallback & issue.eq("")] = "ror_profile_pathology"

                ror_out = ror.reset_index().rename(columns={ror.index.name or "index": "asset"})
                ror_out["component"] = "Generator"
                ror_out["carrier"] = "ror"
                ror_out["available_twh"] = available_twh.reindex(ror_out["asset"]).values
                ror_out["has_profile_column"] = has_ts.reindex(ror_out["asset"]).values
                ror_out["missing_profile"] = missing_profile.reindex(ror_out["asset"]).values
                ror_out["profile_all_zero"] = profile_all_zero.reindex(ror_out["asset"]).values
                ror_out["needs_fallback"] = needs_fallback.reindex(ror_out["asset"]).values
                ror_out["issue"] = issue.reindex(ror_out["asset"]).values
                rows.extend(
                    ror_out[
                        [
                            "component",
                            "carrier",
                            "asset",
                            "bus",
                            "country",
                            "capacity_mw",
                            "available_twh",
                            "has_profile_column",
                            "missing_profile",
                            "profile_all_zero",
                            "needs_fallback",
                            "issue",
                        ]
                    ].to_dict("records")
                )

    return pd.DataFrame(
        rows,
        columns=[
            "component",
            "carrier",
            "asset",
            "bus",
            "country",
            "capacity_mw",
            "available_twh",
            "has_profile_column",
            "missing_profile",
            "profile_all_zero",
            "needs_fallback",
            "issue",
        ],
    )


def _hydro_profile_pathology_summary(asset_diag):
    columns = [
        "component",
        "carrier",
        "country",
        "assets",
        "assets_needing_fallback",
        "missing_profile_assets",
        "zero_profile_assets",
        "capacity_mw",
        "capacity_needing_fallback_mw",
        "available_twh",
        "available_twh_affected",
    ]
    if asset_diag.empty:
        return pd.DataFrame(columns=columns)

    diag = asset_diag.copy()
    diag["country"] = diag["country"].fillna("")
    summary = (
        diag.groupby(["component", "carrier", "country"], as_index=False)
        .agg(
            assets=("asset", "count"),
            assets_needing_fallback=("needs_fallback", "sum"),
            missing_profile_assets=("missing_profile", "sum"),
            zero_profile_assets=("profile_all_zero", "sum"),
            capacity_mw=("capacity_mw", "sum"),
            capacity_needing_fallback_mw=(
                "capacity_mw",
                lambda s: float(s[diag.loc[s.index, "needs_fallback"]].sum()),
            ),
            available_twh=("available_twh", "sum"),
            available_twh_affected=(
                "available_twh",
                lambda s: float(s[diag.loc[s.index, "needs_fallback"]].sum()),
            ),
        )
        .sort_values(["component", "carrier", "country"])
        .reset_index(drop=True)
    )
    return summary[columns]


def _hydro_fallback_donor_table(n, asset_diag, component, carrier):
    if asset_diag.empty:
        return pd.DataFrame()
    donors = asset_diag.loc[
        asset_diag["component"].eq(component)
        & asset_diag["carrier"].eq(carrier)
        & asset_diag["has_profile_column"].astype(bool)
        & (~asset_diag["needs_fallback"].astype(bool))
        & (pd.to_numeric(asset_diag["capacity_mw"], errors="coerce").fillna(0.0) > 0.0)
    ].copy()
    if donors.empty:
        return donors
    donors["x"] = pd.to_numeric(donors["bus"].map(n.buses.get("x", pd.Series(dtype=float))), errors="coerce")
    donors["y"] = pd.to_numeric(donors["bus"].map(n.buses.get("y", pd.Series(dtype=float))), errors="coerce")
    donors = donors.sort_values(["country", "asset"]).reset_index(drop=True)
    return donors


def _pick_hydro_fallback_donor(n, donors, target_bus, target_country):
    if donors.empty:
        return None

    target_country = (target_country or "").strip()
    target_x = np.nan
    target_y = np.nan
    if target_bus in n.buses.index:
        target_x = pd.to_numeric(pd.Series([n.buses.at[target_bus, "x"]]), errors="coerce").iloc[0]
        target_y = pd.to_numeric(pd.Series([n.buses.at[target_bus, "y"]]), errors="coerce").iloc[0]

    same_country = donors.loc[donors["country"].fillna("").eq(target_country)].copy()
    donors_xy = donors.loc[donors["x"].notna() & donors["y"].notna()].copy()
    same_country_xy = same_country.loc[same_country["x"].notna() & same_country["y"].notna()].copy()

    if pd.notna(target_x) and pd.notna(target_y) and not same_country_xy.empty:
        d2 = (same_country_xy["x"] - float(target_x)) ** 2 + (same_country_xy["y"] - float(target_y)) ** 2
        picked = same_country_xy.loc[d2.idxmin()].copy()
        picked["distance_xy"] = float(np.sqrt(float(d2.min())))
        picked["match_method"] = "nearest_xy_same_country"
        return picked

    if pd.notna(target_x) and pd.notna(target_y) and not donors_xy.empty:
        d2 = (donors_xy["x"] - float(target_x)) ** 2 + (donors_xy["y"] - float(target_y)) ** 2
        picked = donors_xy.loc[d2.idxmin()].copy()
        picked["distance_xy"] = float(np.sqrt(float(d2.min())))
        picked["match_method"] = "nearest_xy"
        return picked

    if not same_country.empty:
        picked = same_country.iloc[0].copy()
        picked["distance_xy"] = np.nan
        picked["match_method"] = "first_same_country"
        return picked

    picked = donors.iloc[0].copy()
    picked["distance_xy"] = np.nan
    picked["match_method"] = "first_available"
    return picked


def _apply_hydro_reservoir_inflow_fallback(n, asset_diag, max_capacity_scale=None):
    actions = []
    rows = asset_diag.loc[
        asset_diag["component"].eq("StorageUnit") & asset_diag["carrier"].eq("hydro")
    ].copy()
    if rows.empty:
        return pd.DataFrame()

    targets = rows.loc[rows["needs_fallback"].astype(bool)].copy()
    if targets.empty:
        return pd.DataFrame()

    donors = _hydro_fallback_donor_table(n, rows, "StorageUnit", "hydro")
    if donors.empty:
        for _, t in targets.iterrows():
            actions.append(
                {
                    "component": "StorageUnit",
                    "carrier": "hydro",
                    "target_asset": t["asset"],
                    "target_bus": t["bus"],
                    "target_country": t["country"],
                    "target_capacity_mw": float(t["capacity_mw"]),
                    "issue_before": t["issue"],
                    "status": "unpatched",
                    "reason": "no_nonzero_reservoir_inflow_donor",
                }
            )
        return pd.DataFrame(actions)

    for _, t in targets.iterrows():
        donor = _pick_hydro_fallback_donor(n, donors, str(t["bus"]), str(t["country"]))
        if donor is None:
            actions.append(
                {
                    "component": "StorageUnit",
                    "carrier": "hydro",
                    "target_asset": t["asset"],
                    "target_bus": t["bus"],
                    "target_country": t["country"],
                    "target_capacity_mw": float(t["capacity_mw"]),
                    "issue_before": t["issue"],
                    "status": "unpatched",
                    "reason": "no_donor_selected",
                }
            )
            continue

        donor_asset = str(donor["asset"])
        if donor_asset not in n.storage_units_t.inflow.columns:
            actions.append(
                {
                    "component": "StorageUnit",
                    "carrier": "hydro",
                    "target_asset": t["asset"],
                    "target_bus": t["bus"],
                    "target_country": t["country"],
                    "target_capacity_mw": float(t["capacity_mw"]),
                    "issue_before": t["issue"],
                    "status": "unpatched",
                    "reason": "selected_donor_missing_inflow_column",
                }
            )
            continue

        donor_cap = float(pd.to_numeric(pd.Series([donor["capacity_mw"]]), errors="coerce").iloc[0] or 0.0)
        target_cap = float(pd.to_numeric(pd.Series([t["capacity_mw"]]), errors="coerce").iloc[0] or 0.0)
        if donor_cap <= 0.0:
            actions.append(
                {
                    "component": "StorageUnit",
                    "carrier": "hydro",
                    "target_asset": t["asset"],
                    "target_bus": t["bus"],
                    "target_country": t["country"],
                    "target_capacity_mw": target_cap,
                    "issue_before": t["issue"],
                    "donor_asset": donor_asset,
                    "status": "unpatched",
                    "reason": "selected_donor_nonpositive_capacity",
                }
            )
            continue

        scale_factor = target_cap / donor_cap if donor_cap > 0.0 else 1.0
        applied_scale_factor = float(scale_factor)
        clipped = False
        if (
            max_capacity_scale is not None
            and max_capacity_scale > 0.0
            and scale_factor > float(max_capacity_scale)
        ):
            applied_scale_factor = float(max_capacity_scale)
            clipped = True
        donor_series = n.storage_units_t.inflow[donor_asset].fillna(0.0)
        n.storage_units_t.inflow.loc[:, str(t["asset"])] = donor_series.values * applied_scale_factor

        actions.append(
            {
                "component": "StorageUnit",
                "carrier": "hydro",
                "target_asset": t["asset"],
                "target_bus": t["bus"],
                "target_country": t["country"],
                "target_capacity_mw": target_cap,
                "issue_before": t["issue"],
                "donor_asset": donor_asset,
                "donor_bus": donor.get("bus", ""),
                "donor_country": donor.get("country", ""),
                "donor_capacity_mw": donor_cap,
                "donor_available_twh": float(pd.to_numeric(pd.Series([donor.get("available_twh", np.nan)]), errors="coerce").iloc[0]),
                "scale_factor": float(scale_factor),
                "applied_scale_factor": float(applied_scale_factor),
                "scale_factor_cap": float(max_capacity_scale) if max_capacity_scale is not None else np.nan,
                "distance_xy": float(donor.get("distance_xy")) if pd.notna(donor.get("distance_xy")) else np.nan,
                "match_method": donor.get("match_method", ""),
                "status": "patched",
                "reason": "reservoir_capacity_scale_clipped" if clipped else "",
            }
        )

    return pd.DataFrame(actions)


def _apply_ror_profile_fallback(n, asset_diag):
    actions = []
    rows = asset_diag.loc[
        asset_diag["component"].eq("Generator") & asset_diag["carrier"].eq("ror")
    ].copy()
    if rows.empty:
        return pd.DataFrame()

    targets = rows.loc[rows["needs_fallback"].astype(bool)].copy()
    if targets.empty:
        return pd.DataFrame()

    donors = _hydro_fallback_donor_table(n, rows, "Generator", "ror")
    if donors.empty:
        for _, t in targets.iterrows():
            actions.append(
                {
                    "component": "Generator",
                    "carrier": "ror",
                    "target_asset": t["asset"],
                    "target_bus": t["bus"],
                    "target_country": t["country"],
                    "target_capacity_mw": float(t["capacity_mw"]),
                    "issue_before": t["issue"],
                    "status": "unpatched",
                    "reason": "no_nonzero_ror_profile_donor",
                }
            )
        return pd.DataFrame(actions)

    for _, t in targets.iterrows():
        donor = _pick_hydro_fallback_donor(n, donors, str(t["bus"]), str(t["country"]))
        if donor is None:
            actions.append(
                {
                    "component": "Generator",
                    "carrier": "ror",
                    "target_asset": t["asset"],
                    "target_bus": t["bus"],
                    "target_country": t["country"],
                    "target_capacity_mw": float(t["capacity_mw"]),
                    "issue_before": t["issue"],
                    "status": "unpatched",
                    "reason": "no_donor_selected",
                }
            )
            continue

        donor_asset = str(donor["asset"])
        if donor_asset not in n.generators_t.p_max_pu.columns:
            actions.append(
                {
                    "component": "Generator",
                    "carrier": "ror",
                    "target_asset": t["asset"],
                    "target_bus": t["bus"],
                    "target_country": t["country"],
                    "target_capacity_mw": float(t["capacity_mw"]),
                    "issue_before": t["issue"],
                    "status": "unpatched",
                    "reason": "selected_donor_missing_ror_profile",
                }
            )
            continue

        donor_series = n.generators_t.p_max_pu[donor_asset].fillna(0.0).clip(lower=0.0)
        n.generators_t.p_max_pu.loc[:, str(t["asset"])] = donor_series.values

        actions.append(
            {
                "component": "Generator",
                "carrier": "ror",
                "target_asset": t["asset"],
                "target_bus": t["bus"],
                "target_country": t["country"],
                "target_capacity_mw": float(pd.to_numeric(pd.Series([t["capacity_mw"]]), errors="coerce").iloc[0] or 0.0),
                "issue_before": t["issue"],
                "donor_asset": donor_asset,
                "donor_bus": donor.get("bus", ""),
                "donor_country": donor.get("country", ""),
                "donor_capacity_mw": float(pd.to_numeric(pd.Series([donor.get("capacity_mw", np.nan)]), errors="coerce").iloc[0]),
                "donor_available_twh": float(pd.to_numeric(pd.Series([donor.get("available_twh", np.nan)]), errors="coerce").iloc[0]),
                "scale_factor": 1.0,
                "scale_factor_cap": np.nan,
                "distance_xy": float(donor.get("distance_xy")) if pd.notna(donor.get("distance_xy")) else np.nan,
                "match_method": donor.get("match_method", ""),
                "status": "patched",
                "reason": "",
            }
        )

    return pd.DataFrame(actions)


def _hydro_fallback_diagnostics_dir(config, investment_year, output_network_path=None):
    global_cfg = config.get("global_specific", {}) if isinstance(config, dict) else {}
    base_cfg = global_cfg.get("baseyear_generation", {})
    diagnostics_dir_cfg = str(base_cfg.get("hydro_profile_fallback_diagnostics_dir", "") or "").strip()
    if diagnostics_dir_cfg:
        return Path(_repo_path(diagnostics_dir_cfg))

    if output_network_path:
        net_path = Path(str(output_network_path))
        return net_path.parent / "diagnostics" / f"{net_path.stem}.hydro_profile_fallback"

    return Path(BASE_DIR) / "validation" / ".tmp" / "hydro_profile_fallback" / str(investment_year)


def _write_hydro_fallback_diagnostics(
    config,
    investment_year,
    asset_before,
    asset_after,
    actions,
    output_network_path=None,
):
    global_cfg = config.get("global_specific", {}) if isinstance(config, dict) else {}
    base_cfg = global_cfg.get("baseyear_generation", {})
    if not bool(base_cfg.get("hydro_profile_fallback_diagnostics", True)):
        return None

    diag_dir = _hydro_fallback_diagnostics_dir(config, investment_year, output_network_path)
    diag_dir.mkdir(parents=True, exist_ok=True)

    summary_before = _hydro_profile_pathology_summary(asset_before)
    summary_after = _hydro_profile_pathology_summary(asset_after)

    asset_before.to_csv(diag_dir / "hydro_profile_pathology_assets_before.csv", index=False)
    asset_after.to_csv(diag_dir / "hydro_profile_pathology_assets_after.csv", index=False)
    summary_before.to_csv(diag_dir / "hydro_profile_pathology_summary_before.csv", index=False)
    summary_after.to_csv(diag_dir / "hydro_profile_pathology_summary_after.csv", index=False)
    actions.to_csv(diag_dir / "hydro_profile_fallback_actions.csv", index=False)

    return diag_dir


def apply_hydro_profile_fallback_and_diagnostics(
    n,
    investment_year,
    config,
    output_network_path=None,
):
    """
    Patch hydro profile pathologies in prenetwork structures before hydro scaling hooks.

    Fallbacks:
    - `ror`: nearest non-zero `p_max_pu` profile
    - reservoir (`StorageUnit carrier='hydro'`): nearest non-zero inflow profile,
      scaled by `target_capacity / donor_capacity`

    Diagnostics (CSV):
    - asset-level pathology table before/after
    - country summary before/after
    - fallback action log (patched/unpatched)
    """
    global_cfg = config.get("global_specific", {}) if isinstance(config, dict) else {}
    base_cfg = global_cfg.get("baseyear_generation", {})
    if not bool(base_cfg.get("hydro_profile_fallback_enabled", True)):
        return {
            "enabled": False,
            "diagnostics_dir": None,
            "actions": pd.DataFrame(),
        }

    zero_twh_tol = float(base_cfg.get("hydro_profile_fallback_zero_twh_tol", 1e-9))
    reservoir_scale_cap = float(base_cfg.get("hydro_reservoir_fallback_scale_cap", 10.0))
    asset_before = _hydro_profile_pathology_assets(n, zero_twh_tol=zero_twh_tol)

    if asset_before.empty:
        diag_dir = _write_hydro_fallback_diagnostics(
            config,
            investment_year,
            asset_before,
            asset_before.copy(),
            pd.DataFrame(),
            output_network_path=output_network_path,
        )
        logger.info("Hydro profile fallback: no hydro reservoir/ror assets found; diagnostics_dir=%s", diag_dir)
        return {
            "enabled": True,
            "diagnostics_dir": str(diag_dir) if diag_dir is not None else None,
            "actions": pd.DataFrame(),
            "before": asset_before,
            "after": asset_before.copy(),
        }

    reservoir_actions = _apply_hydro_reservoir_inflow_fallback(
        n,
        asset_before,
        max_capacity_scale=reservoir_scale_cap,
    )
    ror_actions = _apply_ror_profile_fallback(n, asset_before)
    actions = pd.concat([reservoir_actions, ror_actions], ignore_index=True, sort=False)
    if actions.empty:
        actions = pd.DataFrame(
            columns=[
                "component",
                "carrier",
                "target_asset",
                "target_bus",
                "target_country",
                "target_capacity_mw",
                "issue_before",
                "donor_asset",
                "donor_bus",
                "donor_country",
                "donor_capacity_mw",
                "donor_available_twh",
                "scale_factor",
                "applied_scale_factor",
                "scale_factor_cap",
                "distance_xy",
                "match_method",
                "status",
                "reason",
            ]
        )

    asset_after = _hydro_profile_pathology_assets(n, zero_twh_tol=zero_twh_tol)
    diag_dir = _write_hydro_fallback_diagnostics(
        config,
        investment_year,
        asset_before,
        asset_after,
        actions,
        output_network_path=output_network_path,
    )

    before_count = int(asset_before["needs_fallback"].sum()) if not asset_before.empty else 0
    after_count = int(asset_after["needs_fallback"].sum()) if not asset_after.empty else 0
    patched_count = int(actions["status"].eq("patched").sum()) if not actions.empty else 0
    unpatched_count = int(actions["status"].eq("unpatched").sum()) if not actions.empty else 0
    cap_clipped_count = (
        int(actions["reason"].eq("reservoir_capacity_scale_clipped").sum())
        if not actions.empty and "reason" in actions.columns
        else 0
    )
    patched_sample = ""
    if patched_count > 0:
        sample_rows = actions.loc[actions["status"].eq("patched")].head(5)
        patched_sample = ", ".join(
            f"{row.target_asset}<-{row.donor_asset}" for row in sample_rows.itertuples()
        )

    if cap_clipped_count > 0:
        cap_rows = actions.loc[actions["reason"].eq("reservoir_capacity_scale_clipped")].head(5)
        cap_sample = ", ".join(
            f"{row.target_asset}<-{row.donor_asset} ({row.scale_factor:.2f}x clipped to {row.applied_scale_factor:.2f}x)"
            for row in cap_rows.itertuples()
        )
        logger.warning(
            "Hydro reservoir inflow fallback clipped %d asset(s) to capacity-scale cap %.2fx (sample: %s%s)",
            cap_clipped_count,
            reservoir_scale_cap,
            cap_sample,
            " ..." if cap_clipped_count > 5 else "",
        )

    logger.warning(
        "Hydro profile fallback diagnostics: pathologies_before=%d, patched=%d, unpatched=%d, residual_after=%d, reservoir_scale_cap=%.2fx, diagnostics_dir=%s%s%s",
        before_count,
        patched_count,
        unpatched_count,
        after_count,
        reservoir_scale_cap,
        diag_dir,
        ", sample=" if patched_sample else "",
        patched_sample if patched_sample else "",
    )

    return {
        "enabled": True,
        "diagnostics_dir": str(diag_dir) if diag_dir is not None else None,
        "actions": actions,
        "before": asset_before,
        "after": asset_after,
    }


def _electric_load_index_and_country(n):
    if n.loads.empty:
        return pd.Index([]), pd.Series(dtype=object)

    load_bus = n.loads["bus"]
    bus_carrier = load_bus.map(n.buses.carrier).fillna("")
    elec_loads = n.loads.index[bus_carrier.isin(["AC", "low voltage"])]
    if len(elec_loads) == 0:
        return pd.Index([]), pd.Series(dtype=object)

    bus_country = _bus_country_lookup(n)
    load_country = load_bus.loc[elec_loads].map(bus_country).fillna("")
    valid = load_country.ne("")
    elec_loads = elec_loads[valid.values]
    load_country = load_country.loc[valid]
    return elec_loads, load_country


def _electric_load_energy_by_load_mwh(n, load_index):
    if len(load_index) == 0:
        return pd.Series(dtype=float)

    weights = n.snapshot_weightings["generators"]
    weight_sum = float(weights.sum())
    weighted = pd.Series(0.0, index=load_index, dtype=float)

    time_cols = n.loads_t.p_set.columns.intersection(load_index)
    if len(time_cols) > 0:
        weighted.loc[time_cols] = (
            n.loads_t.p_set.reindex(columns=time_cols)
            .fillna(0.0)
            .mul(weights, axis=0)
            .sum(axis=0)
            .astype(float)
        )

    static_cols = load_index.difference(time_cols)
    if len(static_cols) > 0:
        static_p = pd.to_numeric(n.loads.loc[static_cols, "p_set"], errors="coerce").fillna(0.0)
        weighted.loc[static_cols] = static_p * weight_sum

    return weighted


def _baseline_electricity_demand_network_path(base_cfg):
    candidates = [
        base_cfg.get("electricity_demand_baseline_network"),
        base_cfg.get("hydro_baseline_network"),
        base_cfg.get("onwind_baseline_network"),
    ]
    for candidate in candidates:
        if candidate:
            return _repo_path(candidate)
    return None


def _country_ac_end_use_link_withdrawal_twh(
    n,
    output_bus_carrier_substrings=("heat",),
):
    if n.links.empty or n.links_t.p0.empty:
        return pd.Series(dtype=float)

    if "bus0" not in n.links.columns:
        return pd.Series(dtype=float)

    input_bus = n.links["bus0"]
    input_carrier = input_bus.map(n.buses.carrier).fillna("").astype(str)
    mask = input_carrier.eq("AC")
    if not mask.any():
        return pd.Series(dtype=float)

    substrings = [
        str(value).strip().lower()
        for value in (output_bus_carrier_substrings or ())
        if str(value).strip()
    ]
    if substrings:
        output_mask = pd.Series(False, index=n.links.index, dtype=bool)
        for port in [1, 2, 3, 4]:
            bus_col = f"bus{port}"
            if bus_col not in n.links.columns:
                continue
            output_carrier = (
                n.links[bus_col]
                .map(n.buses.carrier)
                .fillna("")
                .astype(str)
                .str.lower()
            )
            port_mask = pd.Series(False, index=n.links.index, dtype=bool)
            for substring in substrings:
                port_mask |= output_carrier.str.contains(re.escape(substring), na=False)
            output_mask |= port_mask
        mask &= output_mask

    selected = n.links.index[mask]
    if len(selected) == 0:
        return pd.Series(dtype=float)

    weights = n.snapshot_weightings["generators"]
    link_cols = n.links_t.p0.columns.intersection(selected)
    if len(link_cols) == 0:
        return pd.Series(dtype=float)

    bus_country = _bus_country_lookup(n)
    link_country = input_bus.loc[link_cols].map(bus_country).fillna("")
    valid = link_country.ne("")
    link_cols = link_cols[valid.values]
    link_country = link_country.loc[valid]
    if len(link_cols) == 0:
        return pd.Series(dtype=float)

    energy_mwh = (
        n.links_t.p0.reindex(columns=link_cols)
        .fillna(0.0)
        .clip(lower=0.0)
        .mul(weights, axis=0)
        .sum(axis=0)
        .astype(float)
    )
    return energy_mwh.groupby(link_country).sum() / 1e6



EMBER_ELECTRICITY_BALANCE_COLUMNS = [
    "Other renewables",
    "Bioenergy",
    "Solar",
    "Wind",
    "Hydropower",
    "Nuclear",
    "Oil",
    "Gas",
    "Coal",
]


def _electricity_demand_alignment_config(config):
    global_cfg = config.get("global_specific", {}) if isinstance(config, dict) else {}
    cfg = global_cfg.get("electricity_demand_alignment")
    if isinstance(cfg, dict):
        return cfg if bool(cfg.get("enable", False)) else None

    base_cfg = global_cfg.get("baseyear_generation", {})
    if not isinstance(base_cfg, dict) or not base_cfg.get("baseyear_generation_constraint", False):
        return None

    baseyear = int(base_cfg.get("year", 2020))
    return {
        "enable": True,
        "audit_dir": "validation/results",
        "anchors": {
            str(baseyear): {
                "year": baseyear,
                "csv": base_cfg.get("owid_csv", "validation/data/owid-energy-data.csv"),
                "schema": "owid_energy",
                "target_column": "electricity_demand",
                "electricity_demand_baseline_network": base_cfg.get(
                    "electricity_demand_baseline_network"
                ),
                "electricity_demand_end_use_link_output_bus_carrier_substrings": base_cfg.get(
                    "electricity_demand_end_use_link_output_bus_carrier_substrings",
                    ["heat"],
                ),
            }
        },
    }


def _alignment_anchors(cfg):
    anchors = cfg.get("anchors", {}) if isinstance(cfg, dict) else {}
    if not isinstance(anchors, dict):
        return {}
    out = {}
    for key, value in anchors.items():
        if not isinstance(value, dict):
            continue
        year = int(value.get("year", key))
        anchor = dict(value)
        anchor["year"] = year
        out[year] = anchor
    return out


def _load_single_electricity_demand_reference_twh(anchor_cfg):
    schema = str(anchor_cfg.get("schema", "owid_energy")).strip().lower()
    year = int(anchor_cfg["year"])
    csv_path = _repo_path(anchor_cfg.get("csv", anchor_cfg.get("owid_csv", "")))
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"electricity demand reference not found: {csv_path}")

    if schema == "owid_energy":
        target_col = anchor_cfg.get("target_column", "electricity_demand")
        usecols = {"year", "iso_code", target_col}
        ref = pd.read_csv(csv_path, usecols=lambda c: c in usecols)
        missing = {"year", "iso_code", target_col}.difference(ref.columns)
        if missing:
            raise ValueError(f"missing OWID demand columns in {csv_path}: {sorted(missing)}")
        ref = ref.loc[ref["year"] == year].copy()
        ref["country"] = ref["iso_code"].apply(_safe_iso3_to_iso2)
        ref[target_col] = pd.to_numeric(ref[target_col], errors="coerce")
        ref = ref.loc[ref["country"].notna() & ref[target_col].notna() & ref[target_col].gt(0.0)]
        return ref.groupby("country")[target_col].sum(min_count=1).astype(float)

    if schema == "ember_generation_balance":
        usecols = {"Year", "Code", *EMBER_ELECTRICITY_BALANCE_COLUMNS}
        ref = pd.read_csv(csv_path, usecols=lambda c: c in usecols)
        missing = {"Year", "Code", *EMBER_ELECTRICITY_BALANCE_COLUMNS}.difference(ref.columns)
        if missing:
            raise ValueError(f"missing Ember balance columns in {csv_path}: {sorted(missing)}")
        ref = ref.loc[ref["Year"] == year].copy()
        ref["country"] = ref["Code"].apply(_safe_iso3_to_iso2)
        for col in EMBER_ELECTRICITY_BALANCE_COLUMNS:
            ref[col] = pd.to_numeric(ref[col], errors="coerce").fillna(0.0)
        ref["electricity_demand_reference_twh"] = ref[EMBER_ELECTRICITY_BALANCE_COLUMNS].sum(axis=1)
        ref = ref.loc[
            ref["country"].notna()
            & ref["electricity_demand_reference_twh"].notna()
            & ref["electricity_demand_reference_twh"].gt(0.0)
        ]
        return ref.groupby("country")["electricity_demand_reference_twh"].sum(min_count=1).astype(float)

    raise ValueError(f"unknown electricity demand reference schema: {schema}")


def _normalise_electricity_demand_fallback_references(anchor_cfg):
    fallback_refs = []
    for fallback in anchor_cfg.get("fallback_references", []) or []:
        if isinstance(fallback, dict):
            fallback_ref = dict(anchor_cfg)
            fallback_ref.pop("fallback_references", None)
            fallback_ref.pop("fallback_years", None)
            fallback_ref.update(fallback)
            fallback_refs.append(fallback_ref)

    for fallback_year in anchor_cfg.get("fallback_years", []) or []:
        fallback_ref = dict(anchor_cfg)
        fallback_ref.pop("fallback_references", None)
        fallback_ref.pop("fallback_years", None)
        fallback_ref["year"] = int(fallback_year)
        fallback_refs.append(fallback_ref)

    return fallback_refs


def _load_electricity_demand_reference_twh(anchor_cfg):
    references = []
    primary_cfg = dict(anchor_cfg)
    primary_cfg.pop("fallback_references", None)
    primary_cfg.pop("fallback_years", None)
    references.append(_load_single_electricity_demand_reference_twh(primary_cfg))

    for fallback_cfg in _normalise_electricity_demand_fallback_references(anchor_cfg):
        try:
            references.append(_load_single_electricity_demand_reference_twh(fallback_cfg))
        except FileNotFoundError:
            raise
        except Exception as exc:
            logger.warning(
                "Ignoring electricity demand fallback reference for year %s from %s: %s",
                fallback_cfg.get("year"),
                fallback_cfg.get("csv", fallback_cfg.get("owid_csv", "")),
                exc,
            )

    if not references:
        return pd.Series(dtype=float)

    combined = references[0].copy()
    for fallback in references[1:]:
        missing = fallback.index.difference(combined.index)
        if len(missing):
            combined = pd.concat([combined, fallback.loc[missing]])
    return combined.sort_index().astype(float)


def _alignment_audit_dir(cfg):
    audit_dir = cfg.get("audit_dir", "validation/results") if isinstance(cfg, dict) else "validation/results"
    return Path(_repo_path(audit_dir))


def _alignment_factor_path(cfg, anchor_year):
    return _alignment_audit_dir(cfg) / f"electricity_demand_alignment_factors_{anchor_year}.csv"


def _load_alignment_factors(cfg, anchor_year):
    path = _alignment_factor_path(cfg, anchor_year)
    if not path.exists():
        return pd.Series(dtype=float), path
    df = pd.read_csv(path)
    if "country" not in df.columns or "factor" not in df.columns:
        raise ValueError(f"invalid electricity demand factor file: {path}")
    factors = pd.to_numeric(df["factor"], errors="coerce")
    return pd.Series(factors.values, index=df["country"].astype(str).str.upper()).dropna(), path


def _load_alignment_direct_after_twh(cfg, anchor_year):
    path = _alignment_audit_dir(cfg) / f"electricity_demand_alignment_{int(anchor_year)}.csv"
    if not path.exists():
        return pd.Series(dtype=float), path
    df = pd.read_csv(path)
    if "country" not in df.columns or "direct_model_twh_after" not in df.columns:
        logger.warning("Invalid electricity demand alignment audit for carry-forward loads: %s", path)
        return pd.Series(dtype=float), path
    countries = df["country"].astype(str).str.upper()
    direct_after = pd.to_numeric(df["direct_model_twh_after"], errors="coerce")
    series = pd.Series(direct_after.values, index=countries).replace([np.inf, -np.inf], np.nan).dropna()
    return series.loc[series.gt(0.0)], path


def _load_fallback_alignment_direct_after_twh(cfg, primary_anchor_year):
    direct_after, path = _load_alignment_direct_after_twh(cfg, primary_anchor_year)
    anchors = _alignment_anchors(cfg)
    for year in sorted(anchors):
        if int(year) == int(primary_anchor_year):
            continue
        fallback, _fallback_path = _load_alignment_direct_after_twh(cfg, year)
        if fallback.empty:
            continue
        missing = fallback.index.difference(direct_after.index)
        if len(missing):
            direct_after = pd.concat([direct_after, fallback.loc[missing]])
    return direct_after.sort_index(), path


def _frame_with_country_index(df_or_series):
    out = df_or_series.reset_index()
    out = out.rename(columns={out.columns[0]: "country"})
    return out


def _write_electricity_demand_alignment_audit(cfg, year, audit, factors):
    audit_dir = _alignment_audit_dir(cfg)
    audit_dir.mkdir(parents=True, exist_ok=True)
    _frame_with_country_index(audit.sort_index()).to_csv(
        audit_dir / f"electricity_demand_alignment_{year}.csv",
        index=False,
    )
    _frame_with_country_index(factors.rename("factor")).to_csv(
        _alignment_factor_path(cfg, year),
        index=False,
    )


def _apply_electricity_load_factors(n, elec_loads, load_country, factors_by_country):
    load_factors = load_country.map(factors_by_country).fillna(1.0)
    time_cols = n.loads_t.p_set.columns.intersection(elec_loads)
    if len(time_cols) > 0:
        n.loads_t.p_set.loc[:, time_cols] = n.loads_t.p_set.loc[:, time_cols].mul(
            load_factors.reindex(time_cols).fillna(1.0), axis=1
        )

    static_cols = elec_loads.difference(time_cols)
    if len(static_cols) > 0:
        n.loads.loc[static_cols, "p_set"] = (
            pd.to_numeric(n.loads.loc[static_cols, "p_set"], errors="coerce")
            .fillna(0.0)
            .mul(load_factors.reindex(static_cols).fillna(1.0))
        )


def _estimated_end_use_link_withdrawal_twh(n, anchor_cfg):
    baseline_network = None
    estimated_end_use_link_twh = pd.Series(dtype=float)
    baseline_path = _baseline_electricity_demand_network_path(anchor_cfg)
    substrings = anchor_cfg.get(
        "electricity_demand_end_use_link_output_bus_carrier_substrings",
        ["heat"],
    )

    if baseline_path is not None:
        if os.path.exists(baseline_path):
            try:
                n_baseline = pypsa.Network(baseline_path)
                estimated_end_use_link_twh = _country_ac_end_use_link_withdrawal_twh(
                    n_baseline,
                    output_bus_carrier_substrings=substrings,
                )
                baseline_network = baseline_path
            except Exception as exc:
                logger.warning(
                    "Electricity demand alignment: failed to load baseline network %s (%s); using current network link withdrawals.",
                    baseline_path,
                    exc,
                )
        else:
            logger.warning(
                "Electricity demand alignment: baseline network not found at %s; using current network link withdrawals.",
                baseline_path,
            )

    if estimated_end_use_link_twh.empty:
        estimated_end_use_link_twh = _country_ac_end_use_link_withdrawal_twh(
            n,
            output_bus_carrier_substrings=substrings,
        )

    return estimated_end_use_link_twh, baseline_network


def align_country_electricity_demand_to_owid(n, investment_year, config):
    cfg = _electricity_demand_alignment_config(config)
    if cfg is None:
        return

    investment_year = int(investment_year)
    anchors = _alignment_anchors(cfg)
    anchor_cfg = anchors.get(investment_year)
    carry_forward_anchor_year = cfg.get("carry_forward_anchor_year")
    carry_forward = False
    if anchor_cfg is None and carry_forward_anchor_year is not None:
        carry_forward_anchor_year = int(carry_forward_anchor_year)
        if investment_year > carry_forward_anchor_year and carry_forward_anchor_year in anchors:
            anchor_cfg = anchors[carry_forward_anchor_year]
            carry_forward = True

    if anchor_cfg is None:
        logger.info(
            "Skipping electricity demand alignment for %s; no anchor or carry-forward factor configured.",
            investment_year,
        )
        return

    elec_loads, load_country = _electric_load_index_and_country(n)
    if len(elec_loads) == 0:
        logger.warning(
            "Electricity demand alignment skipped: no electricity loads found on AC/low voltage buses."
        )
        return

    energy_mwh_by_load = _electric_load_energy_by_load_mwh(n, elec_loads)
    model_twh_before = energy_mwh_by_load.groupby(load_country).sum() / 1e6

    if carry_forward:
        factors_by_country, factor_path = _load_alignment_factors(cfg, anchor_cfg["year"])
        if factors_by_country.empty:
            logger.warning(
                "Electricity demand alignment skipped for %s: carry-forward factor file missing or empty at %s.",
                investment_year,
                factor_path,
            )
            return

        anchor_direct_after, anchor_audit_path = _load_fallback_alignment_direct_after_twh(
            cfg,
            anchor_cfg["year"],
        )
        missing_countries = anchor_direct_after.index.difference(model_twh_before.index)
        if len(missing_countries):
            divisor = factors_by_country.reindex(missing_countries).fillna(1.0).replace(0.0, np.nan)
            direct_before_target = anchor_direct_after.reindex(missing_countries).div(divisor).replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            comp_missing = pd.DataFrame(
                {
                    "direct_model_twh_before": 0.0,
                    "direct_reference_twh": direct_before_target,
                }
            )
            added_missing_loads = _add_missing_direct_electricity_loads(
                n,
                comp_missing,
                investment_year,
                min_reference_twh=float(cfg.get("missing_load_min_reference_twh", 0.1)),
            )
            if added_missing_loads:
                elec_loads, load_country = _electric_load_index_and_country(n)
                energy_mwh_by_load = _electric_load_energy_by_load_mwh(n, elec_loads)
                model_twh_before = energy_mwh_by_load.groupby(load_country).sum() / 1e6
                logger.info(
                    "Added carry-forward electricity loads for %s using anchor audit %s.",
                    investment_year,
                    anchor_audit_path,
                )

        _apply_electricity_load_factors(n, elec_loads, load_country, factors_by_country)
        model_twh_after = (
            _electric_load_energy_by_load_mwh(n, elec_loads).groupby(load_country).sum() / 1e6
        )
        audit = pd.DataFrame(
            {
                "direct_model_twh_before": model_twh_before,
                "direct_model_twh_after": model_twh_after,
                "factor": factors_by_country.reindex(model_twh_after.index).fillna(1.0),
                "source_anchor_year": int(anchor_cfg["year"]),
                "carry_forward": True,
            }
        )
        audit_dir = _alignment_audit_dir(cfg)
        audit_dir.mkdir(parents=True, exist_ok=True)
        _frame_with_country_index(audit.sort_index()).to_csv(
            audit_dir / f"electricity_demand_alignment_{investment_year}.csv",
            index=False,
        )
        logger.info(
            "Applied carry-forward electricity demand alignment for %s using %s factors from %s: countries_scaled=%d, global_direct_before=%.1f TWh, global_direct_after=%.1f TWh",
            investment_year,
            anchor_cfg["year"],
            factor_path,
            int((factors_by_country != 1.0).sum()),
            model_twh_before.sum(),
            model_twh_after.sum(),
        )
        return

    try:
        ref_twh = _load_electricity_demand_reference_twh(anchor_cfg)
    except Exception as exc:
        logger.warning("Electricity demand alignment skipped for %s: %s", investment_year, exc)
        return

    if ref_twh.empty:
        logger.warning("Electricity demand alignment skipped: no reference values for %s.", investment_year)
        return

    estimated_end_use_link_twh, baseline_network = _estimated_end_use_link_withdrawal_twh(n, anchor_cfg)
    effective_ref_twh = ref_twh.sub(estimated_end_use_link_twh.reindex(ref_twh.index).fillna(0.0)).clip(lower=0.0)

    country_index = model_twh_before.index.union(ref_twh.index).union(effective_ref_twh.index)
    comp_before = pd.DataFrame(
        {
            "direct_model_twh_before": model_twh_before.reindex(country_index).fillna(0.0),
            "direct_reference_twh": effective_ref_twh.reindex(country_index),
            "total_reference_twh": ref_twh.reindex(country_index),
            "eligible_link_withdrawal_twh": estimated_end_use_link_twh.reindex(country_index).fillna(0.0),
        }
    ).dropna(subset=["direct_reference_twh"])

    min_missing_load_twh = float(cfg.get("missing_load_min_reference_twh", 0.1))
    added_missing_loads = _add_missing_direct_electricity_loads(
        n,
        comp_before,
        investment_year,
        min_reference_twh=min_missing_load_twh,
    )
    if added_missing_loads:
        elec_loads, load_country = _electric_load_index_and_country(n)
        energy_mwh_by_load = _electric_load_energy_by_load_mwh(n, elec_loads)
        model_twh_before = energy_mwh_by_load.groupby(load_country).sum() / 1e6
        country_index = model_twh_before.index.union(ref_twh.index).union(effective_ref_twh.index)
        comp_before = pd.DataFrame(
            {
                "direct_model_twh_before": model_twh_before.reindex(country_index).fillna(0.0),
                "direct_reference_twh": effective_ref_twh.reindex(country_index),
                "total_reference_twh": ref_twh.reindex(country_index),
                "eligible_link_withdrawal_twh": estimated_end_use_link_twh.reindex(country_index).fillna(0.0),
            }
        ).dropna(subset=["direct_reference_twh"])

    eligible = comp_before["direct_model_twh_before"].gt(0.0) & comp_before["direct_reference_twh"].ge(0.0)
    factors_by_country = (
        comp_before.loc[eligible, "direct_reference_twh"]
        / comp_before.loc[eligible, "direct_model_twh_before"]
    )
    if factors_by_country.empty:
        logger.warning(
            "Electricity demand alignment skipped: no countries with positive modeled and reference demand."
        )
        return

    _apply_electricity_load_factors(n, elec_loads, load_country, factors_by_country)

    model_twh_after = (
        _electric_load_energy_by_load_mwh(n, elec_loads).groupby(load_country).sum() / 1e6
    )
    comp_after = comp_before.join(
        model_twh_after.rename("direct_model_twh_after"),
        how="left",
    )
    comp_after["factor"] = factors_by_country.reindex(comp_after.index).fillna(1.0)
    comp_after["source_anchor_year"] = investment_year
    comp_after["carry_forward"] = False

    before_total_twh = comp_after["direct_model_twh_before"] + comp_after["eligible_link_withdrawal_twh"]
    after_total_twh = comp_after["direct_model_twh_after"] + comp_after["eligible_link_withdrawal_twh"]
    comp_after["total_model_twh_before"] = before_total_twh
    comp_after["total_model_twh_after"] = after_total_twh
    denom = comp_after["total_reference_twh"].replace(0.0, np.nan)
    comp_after["pct_error_before"] = 100.0 * (before_total_twh - comp_after["total_reference_twh"]) / denom
    comp_after["pct_error_after"] = 100.0 * (after_total_twh - comp_after["total_reference_twh"]) / denom
    before_abs_err = (before_total_twh - comp_after["total_reference_twh"]).abs().sum()
    after_abs_err = (after_total_twh - comp_after["total_reference_twh"]).abs().sum()
    ref_sum = comp_after["total_reference_twh"].sum()
    before_wape = 100.0 * before_abs_err / ref_sum if ref_sum > 0 else np.nan
    after_wape = 100.0 * after_abs_err / ref_sum if ref_sum > 0 else np.nan

    _write_electricity_demand_alignment_audit(cfg, investment_year, comp_after, factors_by_country)

    logger.info(
        "Aligned country electricity demand for %s: countries_scaled=%d, global_direct_before=%.1f TWh, global_direct_after=%.1f TWh, eligible_end_use_links=%.1f TWh, global_total_before=%.1f TWh, global_total_after=%.1f TWh, global_reference=%.1f TWh, WAPE_before=%.2f%%, WAPE_after=%.2f%%, baseline_network=%s",
        investment_year,
        len(factors_by_country),
        comp_after["direct_model_twh_before"].sum(),
        comp_after["direct_model_twh_after"].sum(),
        comp_after["eligible_link_withdrawal_twh"].sum(),
        before_total_twh.sum(),
        after_total_twh.sum(),
        comp_after["total_reference_twh"].sum(),
        before_wape,
        after_wape,
        baseline_network if baseline_network is not None else "None",
    )

    largest = factors_by_country.sub(1.0).abs().sort_values(ascending=False).head(10)
    if not largest.empty:
        logger.info(
            "Largest electricity demand alignment factors for %s: %s",
            investment_year,
            ", ".join(f"{country}={factors_by_country.at[country]:.3f}" for country in largest.index),
        )


_TRANSMISSION_ISO3_TO_ISO2_CACHE = {}


def _iso3_to_iso2_for_transmission(code):
    if not isinstance(code, str):
        return np.nan
    code = code.strip().upper()
    if code in _TRANSMISSION_ISO3_TO_ISO2_CACHE:
        return _TRANSMISSION_ISO3_TO_ISO2_CACHE[code]
    if code == "KOS":
        iso2 = "XK"
    else:
        iso2 = _safe_iso3_to_iso2(code)
    _TRANSMISSION_ISO3_TO_ISO2_CACHE[code] = iso2
    return iso2


FOSSIL_CAPACITY_TECH_CARRIERS = {
    "coal": ["coal", "lignite"],
    "gas": ["OCGT", "CCGT"],
    "oil": ["oil"],
}
FOSSIL_CAPACITY_CARRIER_TO_TECH = {
    carrier: tech
    for tech, carriers in FOSSIL_CAPACITY_TECH_CARRIERS.items()
    for carrier in carriers
}
FOSSIL_CHP_POWER_CARRIERS = ("urban central gas CHP", "urban central gas CHP CC")
_FOSSIL_GEM_COUNTRY_CACHE = {}


def _parse_gem_year(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    return pd.to_numeric(text[:4], errors="coerce")


def _gem_operating_mask(df, year):
    status = df["Status"].astype(str).str.strip().str.lower()
    start_year = df["Start year"].apply(_parse_gem_year)
    retired_year = df["Retired year"].apply(_parse_gem_year)
    return (
        status.eq("operating")
        & start_year.fillna(-np.inf).le(int(year))
        & (retired_year.isna() | retired_year.gt(int(year)))
    )


def _gem_country_to_iso2(country):
    country = str(country).strip()
    if not country:
        return np.nan
    if country in _FOSSIL_GEM_COUNTRY_CACHE:
        return _FOSSIL_GEM_COUNTRY_CACHE[country]
    special = {
        "Kosovo": "XK",
        "Türkiye": "TR",
        "Turkey": "TR",
        "Russia": "RU",
        "Vietnam": "VN",
        "Iran": "IR",
        "Syria": "SY",
        "Laos": "LA",
        "Bolivia": "BO",
        "Venezuela": "VE",
        "Tanzania": "TZ",
        "Moldova": "MD",
        "South Korea": "KR",
        "North Korea": "KP",
        "Taiwan": "TW",
    }
    if country in special:
        iso2 = special[country]
    else:
        iso2 = country_name_2_two_digits(country)
        if not isinstance(iso2, str) or iso2.upper() in {"NOT FOUND", "NAN"}:
            iso2 = np.nan
        else:
            iso2 = iso2.upper()
    _FOSSIL_GEM_COUNTRY_CACHE[country] = iso2
    return iso2


def _load_gem_coal_capacity_reference(path, year):
    coal = pd.read_csv(path, encoding="utf-8-sig")
    required = {"Country/Area", "Capacity (MW)", "Status", "Start year", "Retired year"}
    missing = required.difference(coal.columns)
    if missing:
        raise ValueError(f"Missing required GEM coal columns in {path}: {sorted(missing)}")
    coal = coal.loc[_gem_operating_mask(coal, year)].copy()
    lookup = {c: _gem_country_to_iso2(c) for c in coal["Country/Area"].dropna().unique()}
    coal["country"] = coal["Country/Area"].map(lookup)
    coal["reference_mw"] = pd.to_numeric(
        coal["Capacity (MW)"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    ).fillna(0.0)
    coal["validation_tech"] = "coal"
    return (
        coal.loc[coal["country"].notna()]
        .groupby(["country", "validation_tech"], as_index=False)["reference_mw"]
        .sum()
    )


def _load_gem_oil_gas_capacity_reference(path, year):
    gogpt = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "Country/Area",
        "Fuel",
        "Fuel classification?",
        "Capacity (MW)",
        "Status",
        "Start year",
        "Retired year",
    }
    missing = required.difference(gogpt.columns)
    if missing:
        raise ValueError(f"Missing required GEM oil/gas columns in {path}: {sorted(missing)}")
    gogpt = gogpt.loc[_gem_operating_mask(gogpt, year)].copy()
    classification = gogpt["Fuel classification?"].astype(str).str.lower()
    fuel = gogpt["Fuel"].astype(str).str.lower()
    gas = classification.str.contains("gas", na=False) | fuel.str.contains(
        "fossil gas|natural gas|lng", na=False, regex=True
    )
    oil = (
        classification.str.contains("oil only|liquid only", na=False)
        | fuel.str.contains("diesel|fuel oil", na=False, regex=True)
    ) & ~gas
    gogpt["validation_tech"] = pd.Series(pd.NA, index=gogpt.index, dtype="object")
    gogpt.loc[gas, "validation_tech"] = "gas"
    gogpt.loc[oil, "validation_tech"] = "oil"
    gogpt = gogpt.loc[gogpt["validation_tech"].notna()].copy()
    lookup = {c: _gem_country_to_iso2(c) for c in gogpt["Country/Area"].dropna().unique()}
    gogpt["country"] = gogpt["Country/Area"].map(lookup)
    gogpt["reference_mw"] = pd.to_numeric(
        gogpt["Capacity (MW)"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    ).fillna(0.0)
    return (
        gogpt.loc[gogpt["country"].notna()]
        .groupby(["country", "validation_tech"], as_index=False)["reference_mw"]
        .sum()
    )


def _load_gem_fossil_capacity_reference(cfg, year):
    coal_csv = _repo_path(cfg.get("coal_csv", "validation/data/Global-Coal-Plant-Tracker-January-2026.csv"))
    oil_gas_csv = _repo_path(cfg.get("oil_gas_csv", "validation/data/Global-Oil-and-Gas-Plant-Tracker-GOGPT-August-2025.csv"))
    ref = pd.concat(
        [
            _load_gem_coal_capacity_reference(coal_csv, year),
            _load_gem_oil_gas_capacity_reference(oil_gas_csv, year),
        ],
        ignore_index=True,
    )
    carriers = set(str(c) for c in cfg.get("carriers", ["coal", "gas", "oil"]))
    ref = ref.loc[ref["validation_tech"].isin(carriers)].copy()
    return ref.groupby(["country", "validation_tech"], as_index=False)["reference_mw"].sum()


def _link_ac_output_port(link_row, n):
    for port in [1, 2, 3, 4]:
        bus_col = f"bus{port}"
        if bus_col not in link_row.index:
            continue
        bus = link_row.get(bus_col)
        if pd.isna(bus) or str(bus) == "" or bus not in n.buses.index:
            continue
        if str(n.buses.at[bus, "carrier"]) == "AC":
            eff_col = "efficiency" if port == 1 else f"efficiency{port}"
            return bus_col, eff_col
    return None, None


def _repair_fossil_link_fuel_buses(n, carrier_to_tech=None):
    carrier_to_tech = carrier_to_tech or FOSSIL_CAPACITY_CARRIER_TO_TECH
    fuel_by_tech = {"gas": "gas", "coal": "coal", "oil": "oil"}
    if n.links.empty:
        return 0

    bus_country = _bus_country_lookup(n).astype(str).str.strip().str.upper()
    repaired = 0
    for asset, row in n.links.iterrows():
        carrier = str(row.get("carrier", ""))
        tech = carrier_to_tech.get(carrier)
        if tech is None:
            continue
        bus0 = row.get("bus0", "")
        if isinstance(bus0, str) and bus0 in n.buses.index:
            continue
        bus_col, _eff_col = _link_ac_output_port(row, n)
        if bus_col is None:
            continue
        ac_bus = row.get(bus_col)
        country = bus_country.get(ac_bus, "")
        if not re.match(r"^[A-Z]{2}$", str(country)):
            continue
        fuel_bus = _ensure_fossil_fuel_bus_and_store(
            n,
            str(ac_bus),
            fuel_by_tech[str(tech)],
            str(country),
        )
        if fuel_bus is None:
            continue
        n.links.at[asset, "bus0"] = fuel_bus
        repaired += 1

    if repaired:
        logger.info("Repaired missing fossil fuel buses for %d inherited fossil links.", repaired)
    return repaired


def _fossil_electric_capacity_assets(n, carrier_to_tech=None):
    carrier_to_tech = carrier_to_tech or FOSSIL_CAPACITY_CARRIER_TO_TECH
    bus_country = _bus_country_lookup(n).astype(str).str.strip().str.upper()
    rows = []

    if not n.links.empty:
        for asset, row in n.links.iterrows():
            carrier = str(row.get("carrier", ""))
            tech = carrier_to_tech.get(carrier)
            if tech is None:
                continue
            bus_col, eff_col = _link_ac_output_port(row, n)
            if bus_col is None:
                continue
            bus = row[bus_col]
            country = bus_country.get(bus, "")
            if not re.match(r"^[A-Z]{2}$", str(country)):
                continue
            p_nom = float(pd.to_numeric(row.get("p_nom", 0.0), errors="coerce") or 0.0)
            eff = float(pd.to_numeric(row.get(eff_col, 1.0), errors="coerce") or 1.0)
            rows.append(
                {
                    "component": "Link",
                    "asset": asset,
                    "country": country,
                    "validation_tech": tech,
                    "carrier": carrier,
                    "bus": bus,
                    "efficiency": abs(eff),
                    "nominal_mw": p_nom,
                    "electric_capacity_mw": p_nom * abs(eff),
                }
            )

    if not n.generators.empty:
        for asset, row in n.generators.iterrows():
            carrier = str(row.get("carrier", ""))
            tech = carrier_to_tech.get(carrier)
            if tech is None:
                continue
            bus = row.get("bus")
            if pd.isna(bus) or bus not in n.buses.index:
                continue
            if str(n.buses.at[bus, "carrier"]) != "AC":
                continue
            country = bus_country.get(bus, "")
            if not re.match(r"^[A-Z]{2}$", str(country)):
                continue
            p_nom = float(pd.to_numeric(row.get("p_nom", 0.0), errors="coerce") or 0.0)
            rows.append(
                {
                    "component": "Generator",
                    "asset": asset,
                    "country": country,
                    "validation_tech": tech,
                    "carrier": carrier,
                    "bus": bus,
                    "efficiency": 1.0,
                    "nominal_mw": p_nom,
                    "electric_capacity_mw": p_nom,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "component",
                "asset",
                "country",
                "validation_tech",
                "carrier",
                "bus",
                "efficiency",
                "nominal_mw",
                "electric_capacity_mw",
            ]
        )
    return pd.DataFrame(rows)


def _numeric_component_column(df, column, default=0.0):
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").reindex(df.index).fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _set_fossil_asset_nominal_capacity(n, asset, component, nominal_mw):
    nominal_mw = max(float(nominal_mw), 0.0)
    if component == "Link" and asset in n.links.index:
        n.links.at[asset, "p_nom"] = nominal_mw
        if "p_nom_min" in n.links.columns:
            n.links.at[asset, "p_nom_min"] = nominal_mw
    elif component == "Generator" and asset in n.generators.index:
        n.generators.at[asset, "p_nom"] = nominal_mw
        if "p_nom_min" in n.generators.columns:
            n.generators.at[asset, "p_nom_min"] = nominal_mw


def _set_fossil_asset_nominal_cap(n, asset, component, nominal_cap_mw, extendable=True):
    finite_cap = pd.notna(nominal_cap_mw) and np.isfinite(float(nominal_cap_mw))
    nominal_cap_mw = max(float(nominal_cap_mw), 0.0) if finite_cap else np.inf
    if component == "Link" and asset in n.links.index:
        if "p_nom_max" not in n.links.columns:
            n.links["p_nom_max"] = np.inf
        if "p_nom_min" not in n.links.columns:
            n.links["p_nom_min"] = 0.0
        min_nom = float(pd.to_numeric(n.links.at[asset, "p_nom_min"], errors="coerce") or 0.0)
        if finite_cap:
            nominal_cap_mw = max(nominal_cap_mw, min_nom)
        n.links.at[asset, "p_nom_max"] = nominal_cap_mw
        if "p_nom_extendable" in n.links.columns:
            n.links.at[asset, "p_nom_extendable"] = bool(
                extendable and (not finite_cap or nominal_cap_mw > min_nom + 1e-9)
            )
    elif component == "Generator" and asset in n.generators.index:
        if "p_nom_max" not in n.generators.columns:
            n.generators["p_nom_max"] = np.inf
        if "p_nom_min" not in n.generators.columns:
            n.generators["p_nom_min"] = 0.0
        min_nom = float(pd.to_numeric(n.generators.at[asset, "p_nom_min"], errors="coerce") or 0.0)
        if finite_cap:
            nominal_cap_mw = max(nominal_cap_mw, min_nom)
        n.generators.at[asset, "p_nom_max"] = nominal_cap_mw
        if "p_nom_extendable" in n.generators.columns:
            n.generators.at[asset, "p_nom_extendable"] = bool(
                extendable and (not finite_cap or nominal_cap_mw > min_nom + 1e-9)
            )


def _cap_fossil_chp_links(n, planning_year, cfg, audit_dir):
    if n.links.empty or not bool(cfg.get("cap_gas_chp_links", True)):
        return pd.DataFrame()

    carriers = set(str(c) for c in cfg.get("chp_carriers", FOSSIL_CHP_POWER_CARRIERS))
    carrier_mask = n.links["carrier"].astype(str).isin(carriers)
    if "build_year" in n.links.columns:
        build_year = pd.to_numeric(n.links["build_year"], errors="coerce").fillna(0)
        year_mask = build_year.le(int(planning_year))
    else:
        year_mask = pd.Series(True, index=n.links.index)
    idx = n.links.index[carrier_mask & year_mask]
    if len(idx) == 0:
        return pd.DataFrame()

    if "p_nom_min" not in n.links.columns:
        n.links["p_nom_min"] = 0.0
    if "p_nom_max" not in n.links.columns:
        n.links["p_nom_max"] = np.inf
    if "p_nom_extendable" not in n.links.columns:
        n.links["p_nom_extendable"] = False

    before = n.links.loc[idx].copy()
    before_p_nom = _numeric_component_column(before, "p_nom", 0.0)
    before_min = _numeric_component_column(before, "p_nom_min", 0.0)
    fixed_nom = pd.concat([before_p_nom, before_min], axis=1).max(axis=1).clip(lower=0.0)

    n.links.loc[idx, "p_nom"] = fixed_nom.values
    n.links.loc[idx, "p_nom_min"] = fixed_nom.values
    n.links.loc[idx, "p_nom_max"] = fixed_nom.values
    n.links.loc[idx, "p_nom_extendable"] = False
    n.links["p_nom_extendable"] = n.links["p_nom_extendable"].fillna(False).astype(bool)

    audit = pd.DataFrame(
        {
            "year": int(planning_year),
            "asset": idx,
            "carrier": before.loc[idx, "carrier"].astype(str).values,
            "p_nom_before_mw": before_p_nom.reindex(idx).values,
            "p_nom_min_before_mw": before_min.reindex(idx).values,
            "p_nom_max_before_mw": _numeric_component_column(before, "p_nom_max", np.inf).reindex(idx).values,
            "p_nom_after_mw": fixed_nom.reindex(idx).values,
            "p_nom_max_after_mw": fixed_nom.reindex(idx).values,
            "policy": "fixed_existing_chp_capacity",
        }
    )
    audit_path = audit_dir / f"gem_fossil_chp_capacity_caps_{int(planning_year)}.csv"
    audit.to_csv(audit_path, index=False)
    logger.info(
        "Fixed %d gas CHP links at existing nominal capacity for %s; audit=%s",
        len(idx),
        planning_year,
        audit_path,
    )
    return audit


def apply_gem_fossil_capacity_caps(n, planning_year, config, context=""):
    """
    Keep historical fossil power capacity close to GEM country/fuel totals.

    GEM references describe electric power capacity. Pure fossil-electric links
    and generators are scaled to the reference where possible, then given only a
    small expansion headroom. Gas CHP is capped separately because it is
    heat-coupled and should not become an unlimited proxy for gas power capacity.
    """
    global_cfg = config.get("global_specific", {}) if isinstance(config, dict) else {}
    parent_cfg = global_cfg.get("fossil_capacity_alignment", {})
    if not isinstance(parent_cfg, dict) or not bool(parent_cfg.get("enable", False)):
        return

    year = int(planning_year)
    base_year = int(parent_cfg.get("year", 2020))
    materialize_cfg = parent_cfg.get("materialize_2025", {}) or {}
    materialize_year = int(materialize_cfg.get("year", 2025))
    if year not in {base_year, materialize_year}:
        return

    merged_cfg = dict(parent_cfg)
    if year == materialize_year and isinstance(materialize_cfg, dict):
        merged_cfg.update(materialize_cfg)

    carriers = set(str(c) for c in merged_cfg.get("carriers", ["coal", "gas", "oil"]))
    min_reference_mw = float(merged_cfg.get("min_reference_mw", 1.0))
    min_model_mw = float(merged_cfg.get("min_model_mw", 1.0))
    cap_headroom_raw = merged_cfg.get("capacity_cap_headroom", None)
    finite_capacity_caps = cap_headroom_raw is not None
    cap_headroom = float(cap_headroom_raw) if finite_capacity_caps else np.inf
    cap_min_headroom_mw = float(merged_cfg.get("capacity_cap_min_headroom_mw", 0.0))
    scale_existing = bool(
        merged_cfg.get("scale_existing_before_capacity_caps", merged_cfg.get("scale_existing", True))
    )
    scale_down_existing = bool(merged_cfg.get("scale_down_existing", True))
    add_missing = bool(
        merged_cfg.get("add_missing_for_capacity_caps", merged_cfg.get("add_missing", year == materialize_year))
    )
    force_add_missing_pairs = {
        (str(country).upper(), str(tech))
        for country, tech in merged_cfg.get("force_add_missing_pairs", [("SG", "gas")])
    }

    audit_dir = Path(_repo_path(merged_cfg.get("audit_dir", "validation/results")))
    audit_dir.mkdir(parents=True, exist_ok=True)

    _repair_fossil_link_fuel_buses(n)
    _cap_fossil_chp_links(n, year, merged_cfg, audit_dir)

    try:
        ref = _load_gem_fossil_capacity_reference(merged_cfg, year)
    except Exception as exc:
        logger.warning("GEM fossil capacity caps skipped for %s: %s", year, exc)
        return
    ref = ref.loc[
        ref["validation_tech"].isin(carriers) & ref["reference_mw"].ge(min_reference_mw)
    ].copy()
    if ref.empty:
        logger.warning("GEM fossil capacity caps skipped for %s: no configured references.", year)
        return

    assets = _fossil_electric_capacity_assets(n)
    if assets.empty:
        logger.warning("GEM fossil capacity caps skipped for %s: no fossil-electric assets.", year)
        return

    current = (
        assets.groupby(["country", "validation_tech"], as_index=False)["electric_capacity_mw"]
        .sum()
        .rename(columns={"electric_capacity_mw": "model_before_mw"})
    )
    audit = ref.merge(current, on=["country", "validation_tech"], how="outer")
    audit["reference_mw"] = audit["reference_mw"].fillna(0.0)
    audit["model_before_mw"] = audit["model_before_mw"].fillna(0.0)
    if finite_capacity_caps:
        audit["cap_target_mw"] = np.maximum(
            audit["reference_mw"] * (1.0 + cap_headroom),
            audit["reference_mw"] + cap_min_headroom_mw,
        )
    else:
        audit["cap_target_mw"] = np.inf
    audit["factor"] = np.nan
    audit["policy"] = "unchanged"
    audit["context"] = context
    added = scaled = capped = fixed_unreferenced = skipped = 0

    for row in audit.itertuples(index=True):
        country = str(row.country).upper()
        tech = str(row.validation_tech)
        target = float(row.reference_mw)
        before = float(row.model_before_mw)
        group_assets = assets.loc[
            assets["country"].eq(country) & assets["validation_tech"].eq(tech)
        ].copy()

        if target <= 0.0:
            for asset in group_assets.itertuples(index=False):
                _set_fossil_asset_nominal_cap(
                    n,
                    asset.asset,
                    asset.component,
                    asset.nominal_mw,
                    extendable=False,
                )
            audit.at[row.Index, "policy"] = "fixed_unreferenced_existing"
            fixed_unreferenced += int(not group_assets.empty)
            continue

        if group_assets.empty:
            allow_add_missing = add_missing or (country, tech) in force_add_missing_pairs
            if allow_add_missing:
                name, status = _add_fixed_fossil_link_capacity(
                    n,
                    country,
                    tech,
                    target,
                    year,
                    preferred_carrier=None,
                    extendable=True,
                )
                audit.at[row.Index, "policy"] = status
                audit.at[row.Index, "added_asset"] = name if name is not None else ""
                if status == "added":
                    added += 1
                else:
                    skipped += 1
                assets = _fossil_electric_capacity_assets(n)
                group_assets = assets.loc[
                    assets["country"].eq(country) & assets["validation_tech"].eq(tech)
                ].copy()
                before = float(group_assets["electric_capacity_mw"].sum()) if not group_assets.empty else 0.0
            else:
                audit.at[row.Index, "policy"] = "skipped_no_model_capacity"
                skipped += 1
                continue

        if before >= min_model_mw and scale_existing:
            factor = target / before if before > 0.0 else 1.0
            if factor < 1.0 and not scale_down_existing:
                audit.at[row.Index, "policy"] = "kept_existing_above_reference"
            else:
                for asset in group_assets.itertuples(index=False):
                    _set_fossil_asset_nominal_capacity(
                        n,
                        asset.asset,
                        asset.component,
                        asset.nominal_mw * factor,
                    )
                audit.at[row.Index, "factor"] = factor
                audit.at[row.Index, "policy"] = "scaled_to_reference"
                scaled += 1

        if not finite_capacity_caps:
            for asset in group_assets.itertuples(index=False):
                _set_fossil_asset_nominal_cap(
                    n,
                    asset.asset,
                    asset.component,
                    np.inf,
                    extendable=True,
                )
            if audit.at[row.Index, "policy"] == "scaled_to_reference":
                audit.at[row.Index, "policy"] = "scaled_to_reference_unbounded"
            elif audit.at[row.Index, "policy"] == "unchanged":
                audit.at[row.Index, "policy"] = "reference_min_unbounded"
            continue

        assets_after_scale = _fossil_electric_capacity_assets(n)
        group_after = assets_after_scale.loc[
            assets_after_scale["country"].eq(country)
            & assets_after_scale["validation_tech"].eq(tech)
        ].copy()
        after_scale = float(group_after["electric_capacity_mw"].sum()) if not group_after.empty else 0.0
        cap_target = max(float(row.cap_target_mw), after_scale)
        if after_scale <= 0.0:
            audit.at[row.Index, "policy"] = "skipped_no_positive_capacity"
            skipped += 1
            continue
        shares = group_after["electric_capacity_mw"].clip(lower=0.0)
        if shares.sum() <= 0.0:
            shares = pd.Series(1.0 / len(group_after), index=group_after.index)
        else:
            shares = shares / shares.sum()
        for idx, asset in group_after.iterrows():
            eff = max(float(asset["efficiency"]), 1e-9)
            electric_cap_mw = cap_target * float(shares.loc[idx])
            nominal_cap_mw = electric_cap_mw / eff
            _set_fossil_asset_nominal_cap(
                n,
                asset["asset"],
                asset["component"],
                nominal_cap_mw,
                extendable=True,
            )
        capped += 1

    assets_after = _fossil_electric_capacity_assets(n)
    after = (
        assets_after.groupby(["country", "validation_tech"], as_index=False)["electric_capacity_mw"]
        .sum()
        .rename(columns={"electric_capacity_mw": "model_after_mw"})
    )
    audit = audit.merge(after, on=["country", "validation_tech"], how="left")
    audit["model_after_mw"] = audit["model_after_mw"].fillna(0.0)
    audit["year"] = year
    audit["pct_error_after"] = np.where(
        audit["reference_mw"].abs().gt(0.0),
        100.0 * (audit["model_after_mw"] - audit["reference_mw"]) / audit["reference_mw"],
        np.nan,
    )
    audit_path = audit_dir / f"gem_fossil_capacity_caps_{year}.csv"
    audit.sort_values(["validation_tech", "country"]).to_csv(audit_path, index=False)
    logger.info(
        "Applied GEM fossil capacity caps for %s: scaled=%d, capped=%d, added=%d, fixed_unreferenced=%d, skipped=%d, model_before=%.1f GW, model_after=%.1f GW, reference=%.1f GW, audit=%s",
        year,
        scaled,
        capped,
        added,
        fixed_unreferenced,
        skipped,
        audit["model_before_mw"].sum() / 1000.0,
        audit["model_after_mw"].sum() / 1000.0,
        audit["reference_mw"].sum() / 1000.0,
        audit_path,
    )


def _find_fossil_fuel_bus(n, ac_bus, fuel_carrier, country):
    candidates = [f"{ac_bus} {fuel_carrier}", f"{country} {fuel_carrier}", fuel_carrier]
    for candidate in candidates:
        if candidate in n.buses.index:
            return candidate
    if "carrier" in n.buses.columns:
        bus_country = _bus_country_lookup(n).astype(str).str.strip().str.upper()
        buses = n.buses.index[
            n.buses["carrier"].astype(str).eq(fuel_carrier)
            & bus_country.eq(str(country).upper())
        ]
        if len(buses) > 0:
            return str(pd.Index(buses).sort_values()[0])
    return None


def _ensure_fossil_fuel_bus_and_store(n, ac_bus, fuel_carrier, country):
    fuel_bus = _find_fossil_fuel_bus(n, ac_bus, fuel_carrier, country)
    if fuel_bus is None:
        fuel_bus = f"{ac_bus} {fuel_carrier}"
        bus_kwargs = {"carrier": fuel_carrier}
        if "country" in n.buses.columns:
            bus_kwargs["country"] = str(country).upper()
        if "location" in n.buses.columns:
            bus_kwargs["location"] = ac_bus
        if "x" in n.buses.columns and ac_bus in n.buses.index:
            bus_kwargs["x"] = n.buses.at[ac_bus, "x"]
        if "y" in n.buses.columns and ac_bus in n.buses.index:
            bus_kwargs["y"] = n.buses.at[ac_bus, "y"]
        if fuel_carrier not in n.carriers.index:
            n.add("Carrier", fuel_carrier)
        n.add("Bus", fuel_bus, **bus_kwargs)

    store_bus = (
        n.stores["bus"].astype(str)
        if not n.stores.empty and "bus" in n.stores.columns
        else pd.Series(dtype=str)
    )
    store_carrier = (
        n.stores["carrier"].astype(str)
        if not n.stores.empty and "carrier" in n.stores.columns
        else pd.Series("", index=n.stores.index, dtype=str)
    )
    has_store = bool((store_bus.eq(fuel_bus) & store_carrier.eq(fuel_carrier)).any())
    if not has_store:
        template = {}
        same_carrier = n.stores.loc[store_carrier.eq(fuel_carrier)].copy() if not n.stores.empty else pd.DataFrame()
        for col, default in {
            "marginal_cost": 0.0,
            "capital_cost": 0.0,
            "standing_loss": 0.0,
            "lifetime": np.inf,
        }.items():
            if col in same_carrier.columns and not same_carrier.empty:
                values = pd.to_numeric(same_carrier[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                template[col] = float(values.median()) if not values.empty else default
            else:
                template[col] = default
        store_name = f"{fuel_bus} validation store"
        suffix = 2
        while store_name in n.stores.index:
            store_name = f"{fuel_bus} validation store-{suffix}"
            suffix += 1
        n.add(
            "Store",
            store_name,
            bus=fuel_bus,
            carrier=fuel_carrier,
            e_nom=0.0,
            e_nom_min=0.0,
            e_nom_max=np.inf,
            e_nom_extendable=True,
            marginal_cost=template["marginal_cost"],
            capital_cost=template["capital_cost"],
            standing_loss=template["standing_loss"],
            lifetime=template["lifetime"],
        )
    return fuel_bus


def _link_template_from_network(n, carrier):
    links = n.links.loc[n.links.carrier.astype(str).eq(str(carrier))].copy()
    numeric_defaults = {
        "marginal_cost": 0.0,
        "capital_cost": 0.0,
        "efficiency": 1.0,
        "efficiency2": 0.0,
        "lifetime": np.inf,
    }
    template = dict(numeric_defaults)
    if not links.empty:
        for col, default in numeric_defaults.items():
            if col not in links.columns:
                continue
            values = pd.to_numeric(links[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            template[col] = float(values.median()) if not values.empty else default
    return template


def _add_fixed_fossil_link_capacity(
    n,
    country,
    validation_tech,
    electric_capacity_mw,
    year,
    preferred_carrier=None,
    extendable=False,
):
    carrier_by_tech = {
        "gas": preferred_carrier or "CCGT",
        "coal": preferred_carrier or "coal",
        "oil": preferred_carrier or "oil",
    }
    fuel_by_tech = {"gas": "gas", "coal": "coal", "oil": "oil"}
    carrier = carrier_by_tech[str(validation_tech)]
    fuel_carrier = fuel_by_tech[str(validation_tech)]
    bus_country = _bus_country_lookup(n).astype(str).str.strip().str.upper()
    ac_bus = _pick_country_ac_bus(n, country, bus_country)
    if ac_bus is None:
        return None, "missing_ac_bus"
    fuel_bus = _ensure_fossil_fuel_bus_and_store(n, ac_bus, fuel_carrier, country)
    if fuel_bus is None:
        return None, "missing_fuel_bus"
    if "co2 atmosphere" not in n.buses.index:
        return None, "missing_co2_bus"
    if carrier not in n.carriers.index:
        n.add("Carrier", carrier)
    template = _link_template_from_network(n, carrier)
    eff = max(float(template.get("efficiency", 1.0)), 1e-9)
    p_nom = float(electric_capacity_mw) / eff
    base_name = f"{ac_bus} {carrier}-{year}-gem-fixed"
    name = base_name
    suffix = 2
    while name in n.links.index:
        name = f"{base_name}-{suffix}"
        suffix += 1
    n.add(
        "Link",
        name,
        bus0=fuel_bus,
        bus1=ac_bus,
        bus2="co2 atmosphere",
        carrier=carrier,
        p_nom=p_nom,
        p_nom_min=p_nom,
        p_nom_max=np.inf if extendable else p_nom,
        p_nom_extendable=bool(extendable),
        marginal_cost=template.get("marginal_cost", 0.0),
        capital_cost=template.get("capital_cost", 0.0),
        efficiency=eff,
        efficiency2=template.get("efficiency2", 0.0),
        build_year=int(year),
        lifetime=template.get("lifetime", np.inf),
    )
    return name, "added"


def materialize_year2025_gem_fossil_capacities(n, planning_year, config):
    global_cfg = config.get("global_specific", {}) if isinstance(config, dict) else {}
    parent_cfg = global_cfg.get("fossil_capacity_alignment", {})
    cfg = parent_cfg.get("materialize_2025", {}) if isinstance(parent_cfg, dict) else {}
    if not isinstance(cfg, dict) or not bool(cfg.get("enable", False)):
        return
    year = int(cfg.get("year", 2025))
    if int(planning_year) != year:
        return

    merged_cfg = dict(parent_cfg)
    merged_cfg.update(cfg)
    carriers = set(str(c) for c in merged_cfg.get("carriers", ["coal", "gas", "oil"]))
    min_reference_mw = float(merged_cfg.get("min_reference_mw", 1.0))
    min_model_mw = float(merged_cfg.get("min_model_mw", 1.0))
    add_missing = bool(merged_cfg.get("add_missing", True))
    scale_existing = bool(merged_cfg.get("scale_existing", True))
    scale_down_existing = bool(merged_cfg.get("scale_down_existing", True))
    make_links_extendable = bool(
        merged_cfg.get("make_links_extendable_after_materialization", False)
    )
    scale_down_existing_carriers = set(
        str(c) for c in merged_cfg.get("scale_down_existing_carriers", [])
    )

    _repair_fossil_link_fuel_buses(n)

    try:
        ref = _load_gem_fossil_capacity_reference(merged_cfg, year)
    except Exception as exc:
        logger.warning("2025 GEM fossil capacity materialization skipped: %s", exc)
        return
    ref = ref.loc[ref["validation_tech"].isin(carriers) & ref["reference_mw"].ge(min_reference_mw)].copy()
    if ref.empty:
        logger.warning("2025 GEM fossil capacity materialization skipped: no GEM references for configured carriers.")
        return

    assets_before = _fossil_electric_capacity_assets(n)
    current = (
        assets_before.groupby(["country", "validation_tech"], as_index=False)["electric_capacity_mw"]
        .sum()
        .rename(columns={"electric_capacity_mw": "model_before_mw"})
    )
    audit = ref.merge(current, on=["country", "validation_tech"], how="left")
    audit["model_before_mw"] = audit["model_before_mw"].fillna(0.0)
    audit["factor"] = np.where(
        audit["model_before_mw"].ge(min_model_mw),
        audit["reference_mw"] / audit["model_before_mw"],
        np.nan,
    )
    audit["policy"] = "unchanged"
    added = scaled = skipped = 0

    for row in audit.itertuples(index=True):
        country = str(row.country).upper()
        tech = str(row.validation_tech)
        target = float(row.reference_mw)
        before = float(row.model_before_mw)
        group_assets = assets_before.loc[
            assets_before["country"].eq(country)
            & assets_before["validation_tech"].eq(tech)
        ]
        if before >= min_model_mw and scale_existing:
            factor = target / before if before > 0.0 else 1.0
            if (
                factor < 1.0
                and not scale_down_existing
                and tech not in scale_down_existing_carriers
            ):
                audit.at[row.Index, "factor"] = factor
                audit.at[row.Index, "policy"] = "kept_existing_above_reference"
                skipped += 1
                continue
            for asset in group_assets.itertuples(index=False):
                if asset.component == "Link" and asset.asset in n.links.index:
                    p_nom_after = None
                    for col in ["p_nom", "p_nom_min", "p_nom_max"]:
                        if col in n.links.columns:
                            val = pd.to_numeric(n.links.at[asset.asset, col], errors="coerce")
                            if pd.notna(val) and np.isfinite(val):
                                scaled_val = float(val) * factor
                                n.links.at[asset.asset, col] = scaled_val
                                if col == "p_nom":
                                    p_nom_after = scaled_val
                    if make_links_extendable:
                        if p_nom_after is None:
                            p_nom_after = pd.to_numeric(
                                n.links.at[asset.asset, "p_nom"], errors="coerce"
                            )
                        if pd.notna(p_nom_after) and np.isfinite(p_nom_after):
                            if "p_nom_min" in n.links.columns:
                                n.links.at[asset.asset, "p_nom_min"] = float(p_nom_after)
                            if "p_nom_max" in n.links.columns:
                                n.links.at[asset.asset, "p_nom_max"] = np.inf
                    if "p_nom_extendable" in n.links.columns:
                        n.links.at[asset.asset, "p_nom_extendable"] = make_links_extendable
                elif asset.component == "Generator" and asset.asset in n.generators.index:
                    for col in ["p_nom", "p_nom_min", "p_nom_max"]:
                        if col in n.generators.columns:
                            val = pd.to_numeric(n.generators.at[asset.asset, col], errors="coerce")
                            if pd.notna(val) and np.isfinite(val):
                                n.generators.at[asset.asset, col] = float(val) * factor
                    if "p_nom_extendable" in n.generators.columns:
                        n.generators.at[asset.asset, "p_nom_extendable"] = False
            audit.at[row.Index, "policy"] = "scaled_existing"
            scaled += 1
        elif add_missing and target > before:
            missing_capacity_mw = max(target - before, 0.0)
            preferred_carrier = None
            if not group_assets.empty:
                preferred_carrier = str(group_assets.sort_values("electric_capacity_mw", ascending=False).iloc[0]["carrier"])
            name, status = _add_fixed_fossil_link_capacity(
                n,
                country,
                tech,
                missing_capacity_mw,
                year,
                preferred_carrier=preferred_carrier,
                extendable=make_links_extendable,
            )
            audit.at[row.Index, "policy"] = status
            audit.at[row.Index, "added_asset"] = name if name is not None else ""
            if status == "added":
                added += 1
            else:
                skipped += 1
        else:
            audit.at[row.Index, "policy"] = "skipped_no_model_capacity"
            skipped += 1

    assets_after = _fossil_electric_capacity_assets(n)
    after = (
        assets_after.groupby(["country", "validation_tech"], as_index=False)["electric_capacity_mw"]
        .sum()
        .rename(columns={"electric_capacity_mw": "model_after_mw"})
    )
    audit = audit.merge(after, on=["country", "validation_tech"], how="left")
    audit["model_after_mw"] = audit["model_after_mw"].fillna(0.0)
    audit["year"] = year
    audit["pct_error_after"] = np.where(
        audit["reference_mw"].abs().gt(0.0),
        100.0 * (audit["model_after_mw"] - audit["reference_mw"]) / audit["reference_mw"],
        np.nan,
    )
    audit_dir = Path(_repo_path(merged_cfg.get("audit_dir", "validation/results")))
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f"gem_fossil_capacity_materialization_{year}.csv"
    audit.sort_values(["validation_tech", "country"]).to_csv(audit_path, index=False)
    logger.info(
        "Materialized 2025 GEM fossil capacities: scaled=%d, added=%d, skipped_or_preserved=%d, model_before=%.1f GW, model_after=%.1f GW, reference=%.1f GW, audit=%s",
        scaled,
        added,
        skipped,
        audit["model_before_mw"].sum() / 1000.0,
        audit["model_after_mw"].sum() / 1000.0,
        audit["reference_mw"].sum() / 1000.0,
        audit_path,
    )


def _transmission_capacity_limits_config(config):
    global_cfg = config.get("global_specific", {}) if isinstance(config, dict) else {}
    cfg = global_cfg.get("transmission_capacity_limits", {})
    return cfg if isinstance(cfg, dict) and bool(cfg.get("enable", False)) else None


def _clean_gtd_numeric(series):
    return pd.to_numeric(series.replace("-", np.nan), errors="coerce")


def _country_pair_key(country0, country1):
    a, b = sorted((str(country0).upper(), str(country1).upper()))
    return f"{a}|{b}"

def _load_gtd_country_pair_capacity(cfg, planning_year):
    existing_csv = _repo_path(
        cfg.get(
            "existing_csv",
            "validation/data/transmission/GTD-v1.1_regional_existing.csv",
        )
    )
    planned_csv = _repo_path(
        cfg.get(
            "planned_csv",
            "validation/data/transmission/GTD-v1.1_regional_planned.csv",
        )
    )
    missing_year_default = int(cfg.get("missing_year_planned_default", 2025))
    planned_policy = str(cfg.get("planned_policy", "by_year")).lower()

    def read_one(path, include_planned=False):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        df = pd.read_csv(path, encoding=cfg.get("encoding", "ISO-8859-1"))
        required = {"from_country", "to_country", "max_flow", "max_counter_flow"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"missing GTD columns in {path}: {sorted(missing)}")
        if include_planned:
            if "year_planned" not in df.columns:
                raise ValueError(f"missing GTD planned column in {path}: year_planned")
            df["year_planned"] = _clean_gtd_numeric(df["year_planned"]).fillna(missing_year_default)
            if planned_policy == "by_year":
                df = df.loc[df["year_planned"] <= int(planning_year)].copy()
            elif planned_policy == "existing_only":
                df = df.iloc[0:0].copy()
            elif planned_policy == "all_by_2025":
                pass
            else:
                raise ValueError(f"unknown planned_policy: {planned_policy}")
            if "pathway" in df.columns and not df.empty:
                df = (
                    df.sort_values(["pathway", "year_planned"], ascending=[True, False])
                    .drop_duplicates(subset="pathway", keep="first")
                )
        df["country0"] = df["from_country"].apply(_iso3_to_iso2_for_transmission)
        df["country1"] = df["to_country"].apply(_iso3_to_iso2_for_transmission)
        df = df.loc[df["country0"].notna() & df["country1"].notna()].copy()
        df = df.loc[df["country0"] != df["country1"]].copy()
        df["max_flow"] = _clean_gtd_numeric(df["max_flow"])
        df["max_counter_flow"] = _clean_gtd_numeric(df["max_counter_flow"])
        df["capacity_mw"] = df[["max_flow", "max_counter_flow"]].min(axis=1, skipna=False)
        df["capacity_mw"] = df["capacity_mw"].fillna(0.0).clip(lower=0.0)
        df["pair"] = [_country_pair_key(a, b) for a, b in zip(df["country0"], df["country1"])]
        return df.groupby("pair")["capacity_mw"].sum()

    existing = read_one(existing_csv, include_planned=False)
    planned = read_one(planned_csv, include_planned=True)
    pairs = existing.index.union(planned.index)
    out = pd.DataFrame(index=pairs)
    out["gtd_existing_mw"] = existing.reindex(pairs).fillna(0.0)
    out["gtd_planned_available_mw"] = planned.reindex(pairs).fillna(0.0)
    out["gtd_total_available_mw"] = out["gtd_existing_mw"] + out["gtd_planned_available_mw"]
    return out


def _transmission_audit_dir(cfg):
    audit_dir = cfg.get("audit_dir", "validation/results") if isinstance(cfg, dict) else "validation/results"
    return Path(_repo_path(audit_dir))


def _transmission_capacity_year_multiplier(cfg, planning_year):
    multipliers = cfg.get("capacity_multiplier_by_year", {}) if isinstance(cfg, dict) else {}
    if not isinstance(multipliers, dict) or not multipliers:
        return 1.0

    parsed = {}
    for year, value in multipliers.items():
        try:
            parsed[int(year)] = float(value)
        except Exception:
            logger.warning(
                "Ignoring invalid transmission capacity multiplier entry %r: %r",
                year,
                value,
            )
    if not parsed:
        return 1.0

    planning_year = int(planning_year)
    if planning_year in parsed:
        return max(0.0, parsed[planning_year])

    earlier = [year for year in parsed if year <= planning_year]
    if earlier:
        return max(0.0, parsed[max(earlier)])

    return max(0.0, parsed[min(parsed)])


def apply_gtd_transmission_capacity_limits(n, planning_year, config, s_max_pu=None):
    cfg = _transmission_capacity_limits_config(config)
    if cfg is None:
        return n

    if n.lines.empty:
        return n

    planning_year = int(planning_year)
    s_max_pu_value = float(s_max_pu if s_max_pu is not None else cfg.get("s_max_pu", 0.7))
    if s_max_pu_value <= 0.0:
        logger.warning("GTD transmission limits skipped: non-positive s_max_pu=%s", s_max_pu_value)
        return n

    try:
        gtd = _load_gtd_country_pair_capacity(cfg, planning_year)
    except Exception as exc:
        logger.warning("GTD transmission limits skipped for %s: %s", planning_year, exc)
        return n

    bus_country = _bus_country_lookup(n)
    lines = n.lines.copy()
    country0 = lines["bus0"].map(bus_country).fillna("")
    country1 = lines["bus1"].map(bus_country).fillna("")
    cross_border = country0.ne("") & country1.ne("") & country0.ne(country1)
    if not cross_border.any():
        logger.info("GTD transmission limits: no inter-country AC lines found for %s.", planning_year)
        return n

    line_index = lines.index[cross_border]
    line_pairs = pd.Series(
        [_country_pair_key(country0.at[line], country1.at[line]) for line in line_index],
        index=line_index,
        dtype=object,
    )
    original_s_nom = pd.to_numeric(n.lines.loc[line_index, "s_nom"], errors="coerce").fillna(0.0).clip(lower=0.0)
    pair_model_nom = original_s_nom.groupby(line_pairs).sum()
    fallback_ratio = float(cfg.get("fallback_ratio", 0.115))
    missing_policy = str(cfg.get("missing_pair_policy", "zero_capacity")).lower()
    explicit_zero_policy = str(cfg.get("explicit_zero_policy", "zero_capacity")).lower()
    year_multiplier = _transmission_capacity_year_multiplier(cfg, planning_year)

    audit_rows = []
    matched = explicit_zero = fallback = 0
    total_effective_before = float((original_s_nom * s_max_pu_value).sum())
    total_effective_after = 0.0

    for pair, pair_nom in pair_model_nom.items():
        pair_lines = line_pairs.index[line_pairs == pair]
        if pair in gtd.index:
            gtd_row = gtd.loc[pair]
            target_effective = float(gtd_row["gtd_total_available_mw"])
            gtd_existing = float(gtd_row["gtd_existing_mw"])
            gtd_planned = float(gtd_row["gtd_planned_available_mw"])
            if target_effective > 0.0:
                policy_source = "gtd"
                matched += 1
            else:
                policy_source = "gtd_explicit_zero"
                explicit_zero += 1
                if explicit_zero_policy != "zero_capacity":
                    target_effective = fallback_ratio * float(pair_nom)
                    policy_source = f"explicit_zero_{explicit_zero_policy}"
        else:
            gtd_existing = 0.0
            gtd_planned = 0.0
            if missing_policy == "pypsa_default_0.7":
                target_effective = float(pair_nom) * s_max_pu_value
                policy_source = "missing_pypsa_default"
            elif missing_policy == "zero_capacity":
                target_effective = 0.0
                policy_source = "missing_zero_capacity"
            else:
                target_effective = fallback_ratio * float(pair_nom)
                policy_source = "missing_derived_low_cap"
            fallback += 1

        target_effective_before_multiplier = max(0.0, target_effective)
        target_effective = target_effective_before_multiplier * year_multiplier
        target_nominal_pair = target_effective / s_max_pu_value if target_effective > 0.0 else 0.0
        weights = original_s_nom.loc[pair_lines]
        if float(weights.sum()) > 0.0:
            line_targets = target_nominal_pair * weights / float(weights.sum())
        else:
            line_targets = pd.Series(target_nominal_pair / max(len(pair_lines), 1), index=pair_lines)

        n.lines.loc[pair_lines, "s_nom_max"] = line_targets.values
        n.lines.loc[pair_lines, "s_nom"] = np.minimum(
            pd.to_numeric(n.lines.loc[pair_lines, "s_nom"], errors="coerce").fillna(0.0).values,
            line_targets.values,
        )
        if "s_nom_min" in n.lines.columns:
            n.lines.loc[pair_lines, "s_nom_min"] = np.minimum(
                pd.to_numeric(n.lines.loc[pair_lines, "s_nom_min"], errors="coerce").fillna(0.0).values,
                line_targets.values,
            )
        else:
            n.lines.loc[pair_lines, "s_nom_min"] = n.lines.loc[pair_lines, "s_nom"]
        if target_nominal_pair <= 0.0:
            n.lines.loc[pair_lines, "s_nom_extendable"] = False
            n.lines.loc[pair_lines, "s_nom"] = 0.0
            n.lines.loc[pair_lines, "s_nom_min"] = 0.0

        total_effective_after += target_effective
        audit_rows.append(
            {
                "country0": pair.split("|")[0],
                "country1": pair.split("|")[1],
                "model_nominal_before_mw": float(pair_nom),
                "model_effective_before_mw": float(pair_nom) * s_max_pu_value,
                "gtd_existing_mw": gtd_existing,
                "gtd_planned_available_mw": gtd_planned,
                "capacity_year_multiplier": year_multiplier,
                "target_effective_cap_before_multiplier_mw": target_effective_before_multiplier,
                "target_effective_cap_mw": target_effective,
                "target_nominal_cap_mw": target_nominal_pair,
                "s_max_pu": s_max_pu_value,
                "policy_source": policy_source,
                "target_to_model_effective_ratio": target_effective / (float(pair_nom) * s_max_pu_value) if pair_nom > 0.0 else np.nan,
            }
        )

    audit = pd.DataFrame(audit_rows)
    audit_dir = _transmission_audit_dir(cfg)
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit.sort_values(["policy_source", "country0", "country1"]).to_csv(
        audit_dir / f"gtd_transmission_capacity_limits_{planning_year}.csv",
        index=False,
    )

    logger.info(
        "Applied GTD inter-country transmission limits for %s: pairs=%d, matched=%d, explicit_zero=%d, fallback=%d, year_multiplier=%.3f, effective_before=%.1f GW, effective_after=%.1f GW, audit_dir=%s",
        planning_year,
        len(pair_model_nom),
        matched,
        explicit_zero,
        fallback,
        year_multiplier,
        total_effective_before / 1000.0,
        total_effective_after / 1000.0,
        audit_dir,
    )
    return n


def align_country_hydro_reservoir_inflow_to_owid(n, investment_year, config):
    """
    Calibrate hydro scaling from a solved baseline network and scale baseyear inputs
    so expected country hydro generation approaches OWID hydro electricity.

    Inputs scaled in the current prenetwork:
    - StorageUnit carrier='hydro': inflow time series
    - Generator carrier='ror': p_max_pu availability

    Expected hydro generation per country is estimated as:
    expected = k_res(country) * reservoir_inflow_twh + k_ror(country) * ror_available_twh
    where k_res and k_ror are derived from the solved baseline network as:
    - k_res = baseline_reservoir_generation_twh / baseline_reservoir_inflow_twh
    - k_ror = baseline_ror_generation_twh / baseline_ror_available_twh
    """
    global_cfg = config.get("global_specific", {})
    base_cfg = global_cfg.get("baseyear_generation", {})
    if not bool(base_cfg.get("hydro_inflow_alignment", False)):
        return

    baseyear = int(base_cfg.get("year", 2020))
    if int(investment_year) != baseyear:
        logger.info(
            "Skipping OWID hydro inflow alignment for %s (configured baseyear is %s).",
            investment_year,
            baseyear,
        )
        return

    owid_csv = _repo_path(base_cfg.get("owid_csv", "validation/data/owid-energy-data.csv"))
    if not os.path.exists(owid_csv):
        logger.warning(
            "OWID hydro inflow alignment skipped: file not found at %s", owid_csv
        )
        return

    usecols = {"year", "iso_code", "hydro_electricity"}
    owid = pd.read_csv(owid_csv, usecols=lambda c: c in usecols)
    if "hydro_electricity" not in owid.columns:
        logger.warning(
            "OWID hydro inflow alignment skipped: column 'hydro_electricity' not found in %s",
            owid_csv,
        )
        return

    owid = owid.loc[owid["year"] == int(baseyear)].copy()
    owid["country"] = owid["iso_code"].apply(_safe_iso3_to_iso2)
    owid = owid.loc[owid["country"].notna()].copy()
    owid["hydro_electricity"] = pd.to_numeric(owid["hydro_electricity"], errors="coerce").fillna(0.0)
    ref_twh = owid.groupby("country")["hydro_electricity"].sum(min_count=1)
    if ref_twh.empty:
        logger.warning(
            "OWID hydro inflow alignment skipped: no reference values for year %s in %s",
            baseyear,
            owid_csv,
        )
        return

    baseline_path_cfg = base_cfg.get(
        "hydro_baseline_network",
        "results/Global_200/postnetworks/baseline_snapshots/"
        "elec_s_200_ec_lcopt_1h_1h_2020_0.071_AB_0.0export_base_baseline_nobal_nohydroscale.nc",
    )
    baseline_network = _repo_path(baseline_path_cfg)
    if not os.path.exists(baseline_network):
        logger.warning(
            "OWID hydro alignment skipped: baseline solved network not found at %s",
            baseline_network,
        )
        return

    def _hydro_assets_by_country(net):
        bus_country_lu = _bus_country_lookup(net)
        weights_local = net.snapshot_weightings["generators"]
        weight_sum = float(weights_local.sum())

        hydro_units = (
            net.storage_units.index[net.storage_units.carrier.astype(str).eq("hydro")]
            if not net.storage_units.empty
            else pd.Index([])
        )
        inflow_cols_local = (
            net.storage_units_t.inflow.columns.intersection(hydro_units)
            if not net.storage_units_t.inflow.empty
            else pd.Index([])
        )
        su_country_local = (
            net.storage_units.loc[inflow_cols_local, "bus"].map(bus_country_lu).fillna("")
            if len(inflow_cols_local) > 0
            else pd.Series(dtype=object)
        )
        if len(su_country_local) > 0:
            su_valid_local = su_country_local.ne("")
            inflow_cols_local = inflow_cols_local[su_valid_local.values]
            su_country_local = su_country_local.loc[su_valid_local]

        ror_units = (
            net.generators.index[net.generators.carrier.astype(str).eq("ror")]
            if not net.generators.empty
            else pd.Index([])
        )
        ror_country_local = (
            net.generators.loc[ror_units, "bus"].map(bus_country_lu).fillna("")
            if len(ror_units) > 0
            else pd.Series(dtype=object)
        )
        if len(ror_country_local) > 0:
            ror_valid_local = ror_country_local.ne("")
            ror_units = ror_units[ror_valid_local.values]
            ror_country_local = ror_country_local.loc[ror_valid_local]

        res_inflow_twh = pd.Series(dtype=float)
        if len(inflow_cols_local) > 0:
            res_inflow_twh = (
                net.storage_units_t.inflow.reindex(columns=inflow_cols_local)
                .fillna(0.0)
                .mul(weights_local, axis=0)
                .sum(axis=0)
                .groupby(su_country_local)
                .sum()
                / 1e6
            )

        ror_available_twh = pd.Series(dtype=float)
        ror_ts_cols_local = pd.Index([])
        ror_static_cols_local = pd.Index([])
        if len(ror_units) > 0:
            if "p_nom_opt" in net.generators.columns:
                p_nom_source = pd.to_numeric(
                    net.generators.loc[ror_units, "p_nom_opt"], errors="coerce"
                )
                p_nom_nominal = pd.to_numeric(net.generators.loc[ror_units, "p_nom"], errors="coerce")
                ror_nom = p_nom_source.fillna(p_nom_nominal).fillna(0.0)
            else:
                ror_nom = pd.to_numeric(net.generators.loc[ror_units, "p_nom"], errors="coerce").fillna(0.0)

            ror_ts_cols_local = net.generators_t.p_max_pu.columns.intersection(ror_units)
            ror_static_cols_local = ror_units.difference(ror_ts_cols_local)
            availability_hours = pd.Series(0.0, index=ror_units, dtype=float)

            if len(ror_ts_cols_local) > 0:
                availability_hours.loc[ror_ts_cols_local] = (
                    net.generators_t.p_max_pu.reindex(columns=ror_ts_cols_local)
                    .fillna(0.0)
                    .mul(weights_local, axis=0)
                    .sum(axis=0)
                    .astype(float)
                )
            if len(ror_static_cols_local) > 0:
                static_pmax = (
                    pd.to_numeric(net.generators.loc[ror_static_cols_local, "p_max_pu"], errors="coerce")
                    .fillna(0.0)
                    .clip(lower=0.0)
                )
                availability_hours.loc[ror_static_cols_local] = static_pmax * weight_sum

            ror_available_twh = (
                availability_hours.mul(ror_nom.reindex(ror_units).fillna(0.0))
                .groupby(ror_country_local)
                .sum()
                / 1e6
            )

        return {
            "res_inflow_twh": res_inflow_twh,
            "ror_available_twh": ror_available_twh,
            "inflow_cols": inflow_cols_local,
            "su_country": su_country_local,
            "ror_idx": ror_units,
            "ror_country": ror_country_local,
            "ror_ts_cols": ror_ts_cols_local,
            "ror_static_cols": ror_static_cols_local,
        }

    def _hydro_generation_by_country(net):
        eb = net.statistics.energy_balance(
            bus_carrier="AC",
            aggregate_time="sum",
            aggregate_groups="sum",
            groupby=net.statistics.groupers.get_country_and_carrier,
            nice_names=False,
        )
        eb = eb.rename("energy_mwh").reset_index()
        eb = eb.loc[(eb.country != "") & (eb.energy_mwh > 0)].copy()

        res_gen_twh = (
            eb.loc[(eb.component == "StorageUnit") & (eb.carrier == "hydro")]
            .groupby("country")["energy_mwh"]
            .sum()
            / 1e6
        )
        ror_gen_twh = (
            eb.loc[(eb.component == "Generator") & (eb.carrier == "ror")]
            .groupby("country")["energy_mwh"]
            .sum()
            / 1e6
        )
        return res_gen_twh, ror_gen_twh

    current = _hydro_assets_by_country(n)
    if len(current["inflow_cols"]) == 0 and len(current["ror_idx"]) == 0:
        logger.warning(
            "OWID hydro alignment skipped: no hydro reservoirs with inflow and no ror generators with country mapping."
        )
        return

    n_baseline = pypsa.Network(baseline_network)
    baseline = _hydro_assets_by_country(n_baseline)
    base_res_gen_twh, base_ror_gen_twh = _hydro_generation_by_country(n_baseline)

    min_model_twh = float(base_cfg.get("hydro_inflow_min_model_twh", 0.1))
    min_reference_twh = float(base_cfg.get("hydro_inflow_min_reference_twh", 1.0))
    scale_min = float(base_cfg.get("hydro_inflow_scale_min", 0.25))
    scale_max = float(base_cfg.get("hydro_inflow_scale_max", 4.0))
    alpha = float(base_cfg.get("hydro_inflow_alignment_alpha", 1.0))

    res_denom = baseline["res_inflow_twh"]
    ror_denom = baseline["ror_available_twh"]
    res_ratio = pd.Series(dtype=float)
    ror_ratio = pd.Series(dtype=float)
    if not res_denom.empty:
        res_ratio = base_res_gen_twh.reindex(res_denom.index).fillna(0.0).div(
            res_denom.where(res_denom >= min_model_twh)
        )
    if not ror_denom.empty:
        ror_ratio = base_ror_gen_twh.reindex(ror_denom.index).fillna(0.0).div(
            ror_denom.where(ror_denom >= min_model_twh)
        )
    res_ratio = res_ratio.replace([np.inf, -np.inf], np.nan).clip(lower=0.0)
    ror_ratio = ror_ratio.replace([np.inf, -np.inf], np.nan).clip(lower=0.0)

    res_ratio_default = (
        float(base_res_gen_twh.sum() / max(res_denom.sum(), 1e-9))
        if not res_denom.empty
        else 0.0
    )
    ror_ratio_default = (
        float(base_ror_gen_twh.sum() / max(ror_denom.sum(), 1e-9))
        if not ror_denom.empty
        else 0.0
    )

    all_countries = current["res_inflow_twh"].index.union(current["ror_available_twh"].index)
    expected_before_twh = (
        current["res_inflow_twh"].reindex(all_countries, fill_value=0.0).mul(
            res_ratio.reindex(all_countries).fillna(res_ratio_default)
        )
        + current["ror_available_twh"].reindex(all_countries, fill_value=0.0).mul(
            ror_ratio.reindex(all_countries).fillna(ror_ratio_default)
        )
    )

    comp_before = pd.DataFrame(
        {
            "model_twh": expected_before_twh,
            "reference_twh": ref_twh.reindex(expected_before_twh.index),
        }
    ).dropna(subset=["reference_twh"])

    eligible = comp_before["reference_twh"].ge(min_reference_twh) & comp_before["model_twh"].ge(min_model_twh)
    raw_ratio = comp_before.loc[eligible, "reference_twh"] / comp_before.loc[eligible, "model_twh"]
    if raw_ratio.empty:
        logger.warning(
            "OWID hydro inflow alignment skipped: no countries met eligibility thresholds (min_reference_twh=%.3f, min_model_twh=%.3f).",
            min_reference_twh,
            min_model_twh,
        )
        return

    target_ratio = raw_ratio.clip(lower=scale_min, upper=scale_max)
    factors_by_country = 1.0 + alpha * (target_ratio - 1.0)
    factors_by_country = factors_by_country.clip(lower=scale_min, upper=scale_max)

    if len(current["inflow_cols"]) > 0:
        inflow_factors = current["su_country"].map(factors_by_country).fillna(1.0)
        n.storage_units_t.inflow.loc[:, current["inflow_cols"]] = n.storage_units_t.inflow.loc[
            :, current["inflow_cols"]
        ].mul(inflow_factors.reindex(current["inflow_cols"]).fillna(1.0), axis=1)

    ror_pmax_pu_cap = float(base_cfg.get("hydro_ror_p_max_pu_cap", 1.0))
    if len(current["ror_idx"]) > 0:
        ror_factors = current["ror_country"].map(factors_by_country).fillna(1.0)
        if len(current["ror_ts_cols"]) > 0:
            scaled_ts = n.generators_t.p_max_pu.loc[
                :, current["ror_ts_cols"]
            ].mul(ror_factors.reindex(current["ror_ts_cols"]).fillna(1.0), axis=1)
            if ror_pmax_pu_cap > 0.0:
                scaled_ts = scaled_ts.clip(lower=0.0, upper=ror_pmax_pu_cap)
            n.generators_t.p_max_pu.loc[:, current["ror_ts_cols"]] = scaled_ts
        if len(current["ror_static_cols"]) > 0:
            scaled_static = (
                pd.to_numeric(n.generators.loc[current["ror_static_cols"], "p_max_pu"], errors="coerce")
                .fillna(0.0)
                .mul(ror_factors.reindex(current["ror_static_cols"]).fillna(1.0))
            )
            if ror_pmax_pu_cap > 0.0:
                scaled_static = scaled_static.clip(lower=0.0, upper=ror_pmax_pu_cap)
            n.generators.loc[current["ror_static_cols"], "p_max_pu"] = scaled_static

    current_after = _hydro_assets_by_country(n)
    all_countries_after = current_after["res_inflow_twh"].index.union(
        current_after["ror_available_twh"].index
    )
    model_after_twh = (
        current_after["res_inflow_twh"].reindex(all_countries_after, fill_value=0.0).mul(
            res_ratio.reindex(all_countries_after).fillna(res_ratio_default)
        )
        + current_after["ror_available_twh"].reindex(all_countries_after, fill_value=0.0).mul(
            ror_ratio.reindex(all_countries_after).fillna(ror_ratio_default)
        )
    )

    comp_after = pd.DataFrame(
        {
            "model_twh": model_after_twh,
            "reference_twh": ref_twh.reindex(model_after_twh.index),
        }
    ).dropna(subset=["reference_twh"])

    before_abs_err = (comp_before["model_twh"] - comp_before["reference_twh"]).abs().sum()
    after_abs_err = (comp_after["model_twh"] - comp_after["reference_twh"]).abs().sum()
    ref_sum = comp_after["reference_twh"].sum()
    before_wape = 100.0 * before_abs_err / ref_sum if ref_sum > 0 else np.nan
    after_wape = 100.0 * after_abs_err / ref_sum if ref_sum > 0 else np.nan

    logger.info(
        "Aligned hydro inputs to OWID with baseline conversion ratios for %s: countries_scaled=%d, baseline_network=%s, global_expected_before=%.1f TWh, global_expected_after=%.1f TWh, global_reference=%.1f TWh, WAPE_before=%.2f%%, WAPE_after=%.2f%%, k_res_default=%.3f, k_ror_default=%.3f, scale_range=[%.3f, %.3f], alpha=%.2f",
        baseyear,
        len(factors_by_country),
        baseline_network,
        comp_before["model_twh"].sum(),
        comp_after["model_twh"].sum(),
        comp_after["reference_twh"].sum(),
        before_wape,
        after_wape,
        res_ratio_default,
        ror_ratio_default,
        scale_min,
        scale_max,
        alpha,
    )


def adjust_hydro(n, investment_year, config):
    if not isinstance(getattr(n, "meta", None), dict):
        n.meta = {}
    meta_key = f"validation_adjust_hydro_applied_{int(investment_year)}"
    if n.meta.get(meta_key):
        logger.info("Skipping hydro adjustment for %s; already applied on this network.", investment_year)
        return

    cfg = (
        (config.get("global_specific", {}) if isinstance(config, dict) else {})
        .get("hydro_generation_adjustment", {})
    )
    if not isinstance(cfg, dict):
        cfg = {}

    # Absolute multipliers relative to raw hydro inputs. The previous hard-coded
    # value was 4347.02 / 3614.71 ~= 1.204, so these defaults already include
    # that existing correction plus the observed residual hydro shortfall.
    default_scale = 1.34
    year_defaults = {
        "2020": 1.34,
        "2025": 1.26,
    }

    year_multipliers = cfg.get("year_multiplier", {}) if isinstance(cfg, dict) else {}
    scale = float(
        year_multipliers.get(
            str(investment_year),
            cfg.get("multiplier", year_defaults.get(str(investment_year), default_scale)),
        )
    )
    hydro = n.storage_units.index[n.storage_units.carrier.astype(str).eq("hydro")] if not n.storage_units.empty else pd.Index([])
    inflow_cols = n.storage_units_t.inflow.columns.intersection(hydro) if not n.storage_units_t.inflow.empty else pd.Index([])
    if len(inflow_cols):
        n.storage_units_t.inflow.loc[:, inflow_cols] = n.storage_units_t.inflow.loc[:, inflow_cols].fillna(0.0) * scale
    ror = n.generators.index[n.generators.carrier.astype(str).eq("ror")] if not n.generators.empty else pd.Index([])
    if len(ror):
        ts_cols = n.generators_t.p_max_pu.columns.intersection(ror) if not n.generators_t.p_max_pu.empty else pd.Index([])
        static_cols = ror.difference(ts_cols)
        if len(ts_cols):
            n.generators_t.p_max_pu.loc[:, ts_cols] = (
                n.generators_t.p_max_pu.loc[:, ts_cols].fillna(0.0) * scale
            ).clip(upper=1.0)
        if len(static_cols):
            n.generators.loc[static_cols, "p_max_pu"] = (
                pd.to_numeric(n.generators.loc[static_cols, "p_max_pu"], errors="coerce")
                .fillna(0.0)
                .mul(scale)
                .clip(upper=1.0)
            )
    n.meta[meta_key] = True
    logger.info(
        "Adjusted hydro for %s with absolute global scale %.6f (reservoir_assets=%d, ror_assets=%d).",
        investment_year,
        scale,
        len(inflow_cols),
        len(ror),
    )


def align_country_onwind_profiles_to_owid(n, investment_year, config):
    """
    Scale onshore-wind availability profiles by country in baseyear using
    baseline conversion ratios and OWID wind electricity reference.

    The target onwind generation is computed as:
    target_onwind = max(OWID_wind_electricity - expected_offwind, 0)
    where expected_offwind is estimated from baseline offwind conversion ratios.
    """
    global_cfg = config.get("global_specific", {})
    base_cfg = global_cfg.get("baseyear_generation", {})
    if not bool(base_cfg.get("onwind_profile_alignment", False)):
        return

    baseyear = int(base_cfg.get("year", 2020))
    if int(investment_year) != baseyear:
        logger.info(
            "Skipping OWID onwind profile alignment for %s (configured baseyear is %s).",
            investment_year,
            baseyear,
        )
        return

    owid_csv = _repo_path(base_cfg.get("owid_csv", "validation/data/owid-energy-data.csv"))
    if not os.path.exists(owid_csv):
        logger.warning(
            "OWID onwind profile alignment skipped: file not found at %s", owid_csv
        )
        return

    usecols = {"year", "iso_code", "wind_electricity"}
    owid = pd.read_csv(owid_csv, usecols=lambda c: c in usecols)
    if "wind_electricity" not in owid.columns:
        logger.warning(
            "OWID onwind profile alignment skipped: column 'wind_electricity' not found in %s",
            owid_csv,
        )
        return

    owid = owid.loc[owid["year"] == int(baseyear)].copy()
    owid["country"] = owid["iso_code"].apply(_safe_iso3_to_iso2)
    owid = owid.loc[owid["country"].notna()].copy()
    owid["wind_electricity"] = pd.to_numeric(owid["wind_electricity"], errors="coerce").fillna(0.0)
    ref_wind_twh = owid.groupby("country")["wind_electricity"].sum(min_count=1)
    if ref_wind_twh.empty:
        logger.warning(
            "OWID onwind profile alignment skipped: no wind reference values for year %s in %s",
            baseyear,
            owid_csv,
        )
        return

    baseline_path_cfg = base_cfg.get(
        "onwind_baseline_network",
        base_cfg.get(
            "hydro_baseline_network",
            "results/Global_200/postnetworks/baseline_snapshots/"
            "elec_s_200_ec_lcopt_1h_1h_2020_0.071_AB_0.0export_base_baseline_nobal_nohydroscale.nc",
        ),
    )
    baseline_network = _repo_path(baseline_path_cfg)
    if not os.path.exists(baseline_network):
        logger.warning(
            "OWID onwind profile alignment skipped: baseline solved network not found at %s",
            baseline_network,
        )
        return

    def _generator_availability_by_country(net, carriers):
        weights_local = net.snapshot_weightings["generators"]
        weight_sum = float(weights_local.sum())
        bus_country_lu = _bus_country_lookup(net)

        idx = (
            net.generators.index[net.generators.carrier.astype(str).isin(carriers)]
            if not net.generators.empty
            else pd.Index([])
        )
        country = (
            net.generators.loc[idx, "bus"].map(bus_country_lu).fillna("")
            if len(idx) > 0
            else pd.Series(dtype=object)
        )
        if len(country) > 0:
            valid = country.ne("")
            idx = idx[valid.values]
            country = country.loc[valid]

        ts_cols = net.generators_t.p_max_pu.columns.intersection(idx)
        static_cols = idx.difference(ts_cols)
        availability_hours = pd.Series(0.0, index=idx, dtype=float)

        if len(ts_cols) > 0:
            availability_hours.loc[ts_cols] = (
                net.generators_t.p_max_pu.reindex(columns=ts_cols)
                .fillna(0.0)
                .mul(weights_local, axis=0)
                .sum(axis=0)
                .astype(float)
            )
        if len(static_cols) > 0:
            static_pmax = (
                pd.to_numeric(net.generators.loc[static_cols, "p_max_pu"], errors="coerce")
                .fillna(0.0)
                .clip(lower=0.0)
            )
            availability_hours.loc[static_cols] = static_pmax * weight_sum

        if "p_nom_opt" in net.generators.columns:
            p_nom_opt = pd.to_numeric(net.generators.loc[idx, "p_nom_opt"], errors="coerce")
            p_nom = pd.to_numeric(net.generators.loc[idx, "p_nom"], errors="coerce")
            p_nom_eff = p_nom_opt.fillna(p_nom).fillna(0.0)
        else:
            p_nom_eff = pd.to_numeric(net.generators.loc[idx, "p_nom"], errors="coerce").fillna(0.0)

        availability_twh = (
            availability_hours.mul(p_nom_eff.reindex(idx).fillna(0.0)).groupby(country).sum() / 1e6
        )
        return {
            "availability_twh": availability_twh,
            "idx": idx,
            "country": country,
            "ts_cols": ts_cols,
            "static_cols": static_cols,
        }

    def _generator_generation_by_country(net, carriers):
        eb = net.statistics.energy_balance(
            bus_carrier="AC",
            aggregate_time="sum",
            aggregate_groups="sum",
            groupby=net.statistics.groupers.get_country_and_carrier,
            nice_names=False,
        )
        eb = eb.rename("energy_mwh").reset_index()
        eb = eb.loc[(eb.country != "") & (eb.energy_mwh > 0)].copy()
        return (
            eb.loc[(eb.component == "Generator") & (eb.carrier.isin(carriers))]
            .groupby("country")["energy_mwh"]
            .sum()
            / 1e6
        )

    onwind_carriers = {"onwind"}
    offwind_carriers = {"offwind-ac", "offwind-dc"}

    current_on = _generator_availability_by_country(n, onwind_carriers)
    if len(current_on["idx"]) == 0:
        logger.warning("OWID onwind profile alignment skipped: no onwind generators with country mapping.")
        return
    current_off = _generator_availability_by_country(n, offwind_carriers)

    n_baseline = pypsa.Network(baseline_network)
    base_on = _generator_availability_by_country(n_baseline, onwind_carriers)
    base_off = _generator_availability_by_country(n_baseline, offwind_carriers)
    base_on_gen = _generator_generation_by_country(n_baseline, onwind_carriers)
    base_off_gen = _generator_generation_by_country(n_baseline, offwind_carriers)

    min_model_twh = float(base_cfg.get("onwind_min_model_twh", 0.1))
    min_reference_twh = float(base_cfg.get("onwind_min_reference_twh", 1.0))
    scale_min = float(base_cfg.get("onwind_scale_min", 0.25))
    scale_max = float(base_cfg.get("onwind_scale_max", 4.0))
    alpha = float(base_cfg.get("onwind_alignment_alpha", 1.0))
    onwind_pmax_pu_cap = float(base_cfg.get("onwind_p_max_pu_cap", 1.0))

    on_ratio = pd.Series(dtype=float)
    off_ratio = pd.Series(dtype=float)
    if not base_on["availability_twh"].empty:
        on_ratio = base_on_gen.reindex(base_on["availability_twh"].index).fillna(0.0).div(
            base_on["availability_twh"].where(base_on["availability_twh"] >= min_model_twh)
        )
    if not base_off["availability_twh"].empty:
        off_ratio = base_off_gen.reindex(base_off["availability_twh"].index).fillna(0.0).div(
            base_off["availability_twh"].where(base_off["availability_twh"] >= min_model_twh)
        )
    on_ratio = on_ratio.replace([np.inf, -np.inf], np.nan).clip(lower=0.0)
    off_ratio = off_ratio.replace([np.inf, -np.inf], np.nan).clip(lower=0.0)

    on_ratio_default = (
        float(base_on_gen.sum() / max(base_on["availability_twh"].sum(), 1e-9))
        if not base_on["availability_twh"].empty
        else 0.0
    )
    off_ratio_default = (
        float(base_off_gen.sum() / max(base_off["availability_twh"].sum(), 1e-9))
        if not base_off["availability_twh"].empty
        else 0.0
    )

    on_countries = current_on["availability_twh"].index
    expected_on_before = current_on["availability_twh"].reindex(on_countries, fill_value=0.0).mul(
        on_ratio.reindex(on_countries).fillna(on_ratio_default)
    )
    expected_off = current_off["availability_twh"].reindex(on_countries, fill_value=0.0).mul(
        off_ratio.reindex(on_countries).fillna(off_ratio_default)
    )
    target_on = (
        ref_wind_twh.reindex(on_countries).fillna(0.0).sub(expected_off, fill_value=0.0).clip(lower=0.0)
    )

    comp_before = pd.DataFrame(
        {
            "model_twh": expected_on_before,
            "reference_twh": target_on,
        }
    )
    eligible = comp_before["reference_twh"].ge(min_reference_twh) & comp_before["model_twh"].ge(min_model_twh)
    raw_ratio = comp_before.loc[eligible, "reference_twh"] / comp_before.loc[eligible, "model_twh"]
    if raw_ratio.empty:
        logger.warning(
            "OWID onwind profile alignment skipped: no countries met eligibility thresholds (min_reference_twh=%.3f, min_model_twh=%.3f).",
            min_reference_twh,
            min_model_twh,
        )
        return

    target_ratio = raw_ratio.clip(lower=scale_min, upper=scale_max)
    factors_by_country = 1.0 + alpha * (target_ratio - 1.0)
    factors_by_country = factors_by_country.clip(lower=scale_min, upper=scale_max)

    on_factors = current_on["country"].map(factors_by_country).fillna(1.0)
    if len(current_on["ts_cols"]) > 0:
        scaled_ts = n.generators_t.p_max_pu.loc[:, current_on["ts_cols"]].mul(
            on_factors.reindex(current_on["ts_cols"]).fillna(1.0),
            axis=1,
        )
        if onwind_pmax_pu_cap > 0.0:
            scaled_ts = scaled_ts.clip(lower=0.0, upper=onwind_pmax_pu_cap)
        n.generators_t.p_max_pu.loc[:, current_on["ts_cols"]] = scaled_ts
    if len(current_on["static_cols"]) > 0:
        scaled_static = (
            pd.to_numeric(n.generators.loc[current_on["static_cols"], "p_max_pu"], errors="coerce")
            .fillna(0.0)
            .mul(on_factors.reindex(current_on["static_cols"]).fillna(1.0))
        )
        if onwind_pmax_pu_cap > 0.0:
            scaled_static = scaled_static.clip(lower=0.0, upper=onwind_pmax_pu_cap)
        n.generators.loc[current_on["static_cols"], "p_max_pu"] = scaled_static

    current_on_after = _generator_availability_by_country(n, onwind_carriers)
    on_countries_after = current_on_after["availability_twh"].index
    expected_on_after = current_on_after["availability_twh"].reindex(on_countries_after, fill_value=0.0).mul(
        on_ratio.reindex(on_countries_after).fillna(on_ratio_default)
    )
    expected_off_after = current_off["availability_twh"].reindex(on_countries_after, fill_value=0.0).mul(
        off_ratio.reindex(on_countries_after).fillna(off_ratio_default)
    )
    target_on_after = (
        ref_wind_twh.reindex(on_countries_after)
        .fillna(0.0)
        .sub(expected_off_after, fill_value=0.0)
        .clip(lower=0.0)
    )
    comp_after = pd.DataFrame(
        {"model_twh": expected_on_after, "reference_twh": target_on_after}
    )

    before_abs_err = (comp_before["model_twh"] - comp_before["reference_twh"]).abs().sum()
    after_abs_err = (comp_after["model_twh"] - comp_after["reference_twh"]).abs().sum()
    ref_sum = comp_after["reference_twh"].sum()
    before_wape = 100.0 * before_abs_err / ref_sum if ref_sum > 0 else np.nan
    after_wape = 100.0 * after_abs_err / ref_sum if ref_sum > 0 else np.nan

    logger.info(
        "Aligned onwind profiles to OWID using baseline conversion ratios for %s: countries_scaled=%d, baseline_network=%s, expected_onwind_before=%.1f TWh, expected_onwind_after=%.1f TWh, onwind_target=%.1f TWh, WAPE_before=%.2f%%, WAPE_after=%.2f%%, on_ratio_default=%.3f, off_ratio_default=%.3f, p_max_pu_cap=%.2f, scale_range=[%.3f, %.3f], alpha=%.2f",
        baseyear,
        len(factors_by_country),
        baseline_network,
        comp_before["model_twh"].sum(),
        comp_after["model_twh"].sum(),
        comp_after["reference_twh"].sum(),
        before_wape,
        after_wape,
        on_ratio_default,
        off_ratio_default,
        onwind_pmax_pu_cap,
        scale_min,
        scale_max,
        alpha,
    )


def apply_country_wind_iteration_scaling(n, investment_year, config):
    """
    Apply per-country iterative p_max_pu scaling factors for onshore and offshore wind.

    This is intended for external calibration loops that update
    `wind_iteration_override_csv` between solves.
    """
    global_cfg = config.get("global_specific", {})
    base_cfg = global_cfg.get("baseyear_generation", {})
    if not bool(base_cfg.get("wind_iteration_scaling_enabled", False)):
        return

    baseyear = int(base_cfg.get("year", 2020))
    if int(investment_year) != baseyear:
        logger.info(
            "Skipping iterative wind scaling for %s (configured baseyear is %s).",
            investment_year,
            baseyear,
        )
        return

    override_csv_cfg = base_cfg.get("wind_iteration_override_csv", "")
    if not override_csv_cfg:
        logger.warning("Iterative wind scaling enabled but no wind_iteration_override_csv configured.")
        return

    override_csv = _repo_path(override_csv_cfg)
    if not os.path.exists(override_csv):
        logger.warning(
            "Iterative wind scaling skipped: override file not found at %s", override_csv
        )
        return

    override = pd.read_csv(override_csv)
    if override.empty:
        logger.warning(
            "Iterative wind scaling skipped: override file is empty at %s", override_csv
        )
        return

    cols = {c.lower().strip(): c for c in override.columns}
    if "country" not in cols:
        logger.warning(
            "Iterative wind scaling skipped: column 'country' missing in %s", override_csv
        )
        return

    rename_cols = {cols["country"]: "country"}
    if "onwind_scale" in cols:
        rename_cols[cols["onwind_scale"]] = "onwind_scale"
    if "offwind_scale" in cols:
        rename_cols[cols["offwind_scale"]] = "offwind_scale"
    override = override.rename(columns=rename_cols)
    if "onwind_scale" not in override.columns:
        override["onwind_scale"] = 1.0
    if "offwind_scale" not in override.columns:
        override["offwind_scale"] = 1.0

    override["country"] = override["country"].astype(str).str.upper().str.strip()
    override = override.loc[override["country"].str.len().eq(2)].copy()
    if override.empty:
        logger.warning(
            "Iterative wind scaling skipped: no valid ISO2 countries in %s", override_csv
        )
        return

    min_scale = float(base_cfg.get("wind_iteration_scale_min", 0.05))
    max_scale = float(base_cfg.get("wind_iteration_scale_max", 20.0))
    for col in ["onwind_scale", "offwind_scale"]:
        override[col] = (
            pd.to_numeric(override[col], errors="coerce")
            .fillna(1.0)
            .clip(lower=min_scale, upper=max_scale)
        )

    scale_by_country = override.groupby("country")[["onwind_scale", "offwind_scale"]].mean()
    bus_country_lu = _bus_country_lookup(n)

    def _apply_for_carriers(carriers, scale_col, pmax_cap):
        idx = (
            n.generators.index[n.generators.carrier.astype(str).isin(carriers)]
            if not n.generators.empty
            else pd.Index([])
        )
        if len(idx) == 0:
            return 0, 0

        country = n.generators.loc[idx, "bus"].map(bus_country_lu).fillna("")
        valid = country.ne("")
        idx = idx[valid.values]
        country = country.loc[valid]
        if len(idx) == 0:
            return 0, 0

        factors = country.map(scale_by_country[scale_col]).fillna(1.0)
        scaled_country_count = int((~np.isclose(factors, 1.0, atol=1e-12)).sum())

        ts_cols = n.generators_t.p_max_pu.columns.intersection(idx)
        static_cols = idx.difference(ts_cols)
        if len(ts_cols) > 0:
            scaled_ts = n.generators_t.p_max_pu.loc[:, ts_cols].mul(
                factors.reindex(ts_cols).fillna(1.0), axis=1
            )
            if pmax_cap > 0.0:
                scaled_ts = scaled_ts.clip(lower=0.0, upper=pmax_cap)
            n.generators_t.p_max_pu.loc[:, ts_cols] = scaled_ts
        if len(static_cols) > 0:
            scaled_static = (
                pd.to_numeric(n.generators.loc[static_cols, "p_max_pu"], errors="coerce")
                .fillna(0.0)
                .mul(factors.reindex(static_cols).fillna(1.0))
            )
            if pmax_cap > 0.0:
                scaled_static = scaled_static.clip(lower=0.0, upper=pmax_cap)
            n.generators.loc[static_cols, "p_max_pu"] = scaled_static

        return len(idx), scaled_country_count

    on_cap = float(base_cfg.get("onwind_p_max_pu_cap", 1.0))
    off_cap = float(base_cfg.get("offwind_p_max_pu_cap", 1.0))
    on_count, on_scaled_countries = _apply_for_carriers({"onwind"}, "onwind_scale", on_cap)
    off_count, off_scaled_countries = _apply_for_carriers(
        {"offwind-ac", "offwind-dc"}, "offwind_scale", off_cap
    )

    logger.info(
        "Applied iterative wind scaling from %s: onwind_generators=%d (scaled_entries=%d), offwind_generators=%d (scaled_entries=%d), country_rows=%d, scale_bounds=[%.3f, %.3f], on_cap=%.2f, off_cap=%.2f",
        override_csv,
        on_count,
        on_scaled_countries,
        off_count,
        off_scaled_countries,
        len(scale_by_country),
        min_scale,
        max_scale,
        on_cap,
        off_cap,
    )


def apply_country_solar_iteration_scaling(n, investment_year, config):
    """
    Apply per-country iterative p_max_pu scaling factors for solar generators.

    This is intended for external calibration loops that update
    `solar_iteration_override_csv` between solves.
    """
    global_cfg = config.get("global_specific", {})
    base_cfg = global_cfg.get("baseyear_generation", {})
    if not bool(base_cfg.get("solar_iteration_scaling_enabled", False)):
        return

    baseyear = int(base_cfg.get("year", 2020))
    if int(investment_year) != baseyear:
        logger.info(
            "Skipping iterative solar scaling for %s (configured baseyear is %s).",
            investment_year,
            baseyear,
        )
        return

    override_csv_cfg = base_cfg.get("solar_iteration_override_csv", "")
    if not override_csv_cfg:
        logger.warning("Iterative solar scaling enabled but no solar_iteration_override_csv configured.")
        return

    override_csv = _repo_path(override_csv_cfg)
    if not os.path.exists(override_csv):
        logger.warning(
            "Iterative solar scaling skipped: override file not found at %s", override_csv
        )
        return

    override = pd.read_csv(override_csv)
    if override.empty:
        logger.warning(
            "Iterative solar scaling skipped: override file is empty at %s", override_csv
        )
        return

    cols = {c.lower().strip(): c for c in override.columns}
    if "country" not in cols:
        logger.warning(
            "Iterative solar scaling skipped: column 'country' missing in %s", override_csv
        )
        return
    rename_cols = {cols["country"]: "country"}
    if "solar_scale" in cols:
        rename_cols[cols["solar_scale"]] = "solar_scale"
    override = override.rename(columns=rename_cols)
    if "solar_scale" not in override.columns:
        override["solar_scale"] = 1.0

    override["country"] = override["country"].astype(str).str.upper().str.strip()
    override = override.loc[override["country"].str.len().eq(2)].copy()
    if override.empty:
        logger.warning(
            "Iterative solar scaling skipped: no valid ISO2 countries in %s", override_csv
        )
        return

    min_scale = float(base_cfg.get("solar_iteration_scale_min", 0.05))
    max_scale = float(base_cfg.get("solar_iteration_scale_max", 20.0))
    override["solar_scale"] = (
        pd.to_numeric(override["solar_scale"], errors="coerce")
        .fillna(1.0)
        .clip(lower=min_scale, upper=max_scale)
    )
    scale_by_country = override.groupby("country")["solar_scale"].mean()

    if n.generators.empty:
        return
    idx = n.generators.index[n.generators.carrier.astype(str).eq("solar")]
    if len(idx) == 0:
        logger.warning("Iterative solar scaling skipped: no solar generators found.")
        return

    bus_country_lu = _bus_country_lookup(n)
    country = n.generators.loc[idx, "bus"].map(bus_country_lu).fillna("")
    valid = country.ne("")
    idx = idx[valid.values]
    country = country.loc[valid]
    if len(idx) == 0:
        logger.warning("Iterative solar scaling skipped: no solar generators with country mapping.")
        return

    factors = country.map(scale_by_country).fillna(1.0)
    scaled_entries = int((~np.isclose(factors, 1.0, atol=1e-12)).sum())
    pmax_cap = float(base_cfg.get("solar_p_max_pu_cap", 1.0))

    ts_cols = n.generators_t.p_max_pu.columns.intersection(idx)
    static_cols = idx.difference(ts_cols)
    if len(ts_cols) > 0:
        scaled_ts = n.generators_t.p_max_pu.loc[:, ts_cols].mul(
            factors.reindex(ts_cols).fillna(1.0), axis=1
        )
        if pmax_cap > 0.0:
            scaled_ts = scaled_ts.clip(lower=0.0, upper=pmax_cap)
        n.generators_t.p_max_pu.loc[:, ts_cols] = scaled_ts
    if len(static_cols) > 0:
        scaled_static = (
            pd.to_numeric(n.generators.loc[static_cols, "p_max_pu"], errors="coerce")
            .fillna(0.0)
            .mul(factors.reindex(static_cols).fillna(1.0))
        )
        if pmax_cap > 0.0:
            scaled_static = scaled_static.clip(lower=0.0, upper=pmax_cap)
        n.generators.loc[static_cols, "p_max_pu"] = scaled_static

    logger.info(
        "Applied iterative solar scaling from %s: solar_generators=%d (scaled_entries=%d), country_rows=%d, scale_bounds=[%.3f, %.3f], p_max_pu_cap=%.2f",
        override_csv,
        len(idx),
        scaled_entries,
        len(scale_by_country),
        min_scale,
        max_scale,
        pmax_cap,
    )


def apply_country_hydro_iteration_scaling(n, investment_year, config):
    """
    Apply per-country iterative hydro scaling jointly to reservoir inflow and ror profiles.

    Override CSV format:
    - country (ISO2)
    - hydro_scale
    """
    global_cfg = config.get("global_specific", {})
    base_cfg = global_cfg.get("baseyear_generation", {})
    if not bool(base_cfg.get("hydro_iteration_scaling_enabled", False)):
        return

    baseyear = int(base_cfg.get("year", 2020))
    if int(investment_year) != baseyear:
        logger.info(
            "Applying iterative hydro scaling to %s using overrides calibrated from baseyear %s.",
            investment_year,
            baseyear,
        )

    override_csv_cfg = base_cfg.get("hydro_iteration_override_csv", "")
    if not override_csv_cfg:
        logger.warning("Iterative hydro scaling enabled but no hydro_iteration_override_csv configured.")
        return

    override_csv = _repo_path(override_csv_cfg)
    if not os.path.exists(override_csv):
        logger.warning(
            "Iterative hydro scaling skipped: override file not found at %s", override_csv
        )
        return

    override = pd.read_csv(override_csv)
    if override.empty:
        logger.warning(
            "Iterative hydro scaling skipped: override file is empty at %s", override_csv
        )
        return

    cols = {c.lower().strip(): c for c in override.columns}
    if "country" not in cols:
        logger.warning(
            "Iterative hydro scaling skipped: column 'country' missing in %s", override_csv
        )
        return
    rename_cols = {cols["country"]: "country"}
    if "hydro_scale" in cols:
        rename_cols[cols["hydro_scale"]] = "hydro_scale"
    override = override.rename(columns=rename_cols)
    if "hydro_scale" not in override.columns:
        override["hydro_scale"] = 1.0

    override["country"] = override["country"].astype(str).str.upper().str.strip()
    override = override.loc[override["country"].str.len().eq(2)].copy()
    if override.empty:
        logger.warning(
            "Iterative hydro scaling skipped: no valid ISO2 countries in %s", override_csv
        )
        return

    min_scale = float(base_cfg.get("hydro_iteration_scale_min", 0.05))
    max_scale = float(base_cfg.get("hydro_iteration_scale_max", 20.0))
    override["hydro_scale"] = (
        pd.to_numeric(override["hydro_scale"], errors="coerce")
        .fillna(1.0)
        .clip(lower=min_scale, upper=max_scale)
    )
    scale_by_country = override.groupby("country")["hydro_scale"].mean()
    bus_country_lu = _bus_country_lookup(n)

    su_idx = (
        n.storage_units.index[n.storage_units.carrier.astype(str).eq("hydro")]
        if not n.storage_units.empty
        else pd.Index([])
    )
    inflow_cols = (
        n.storage_units_t.inflow.columns.intersection(su_idx)
        if not n.storage_units_t.inflow.empty
        else pd.Index([])
    )
    su_country = (
        n.storage_units.loc[inflow_cols, "bus"].map(bus_country_lu).fillna("")
        if len(inflow_cols) > 0
        else pd.Series(dtype=object)
    )
    if len(su_country) > 0:
        valid = su_country.ne("")
        inflow_cols = inflow_cols[valid.values]
        su_country = su_country.loc[valid]
    inflow_scaled = 0
    if len(inflow_cols) > 0:
        inflow_factors = su_country.map(scale_by_country).fillna(1.0)
        inflow_scaled = int((~np.isclose(inflow_factors, 1.0, atol=1e-12)).sum())
        n.storage_units_t.inflow.loc[:, inflow_cols] = n.storage_units_t.inflow.loc[
            :, inflow_cols
        ].mul(inflow_factors.reindex(inflow_cols).fillna(1.0), axis=1)

    ror_idx = (
        n.generators.index[n.generators.carrier.astype(str).eq("ror")]
        if not n.generators.empty
        else pd.Index([])
    )
    ror_country = (
        n.generators.loc[ror_idx, "bus"].map(bus_country_lu).fillna("")
        if len(ror_idx) > 0
        else pd.Series(dtype=object)
    )
    if len(ror_country) > 0:
        valid = ror_country.ne("")
        ror_idx = ror_idx[valid.values]
        ror_country = ror_country.loc[valid]

    ror_scaled = 0
    if len(ror_idx) > 0:
        ror_factors = ror_country.map(scale_by_country).fillna(1.0)
        ror_scaled = int((~np.isclose(ror_factors, 1.0, atol=1e-12)).sum())
        ror_ts_cols = n.generators_t.p_max_pu.columns.intersection(ror_idx)
        ror_static_cols = ror_idx.difference(ror_ts_cols)
        ror_pmax_cap = float(base_cfg.get("hydro_ror_p_max_pu_cap", 1.0))
        if len(ror_ts_cols) > 0:
            scaled_ts = n.generators_t.p_max_pu.loc[:, ror_ts_cols].mul(
                ror_factors.reindex(ror_ts_cols).fillna(1.0), axis=1
            )
            if ror_pmax_cap > 0.0:
                scaled_ts = scaled_ts.clip(lower=0.0, upper=ror_pmax_cap)
            n.generators_t.p_max_pu.loc[:, ror_ts_cols] = scaled_ts
        if len(ror_static_cols) > 0:
            scaled_static = (
                pd.to_numeric(n.generators.loc[ror_static_cols, "p_max_pu"], errors="coerce")
                .fillna(0.0)
                .mul(ror_factors.reindex(ror_static_cols).fillna(1.0))
            )
            if ror_pmax_cap > 0.0:
                scaled_static = scaled_static.clip(lower=0.0, upper=ror_pmax_cap)
            n.generators.loc[ror_static_cols, "p_max_pu"] = scaled_static

    logger.info(
        "Applied iterative hydro scaling from %s: reservoir_inflows=%d (scaled=%d), ror_generators=%d (scaled=%d), country_rows=%d, scale_bounds=[%.3f, %.3f]",
        override_csv,
        len(inflow_cols),
        inflow_scaled,
        len(ror_idx),
        ror_scaled,
        len(scale_by_country),
        min_scale,
        max_scale,
    )


def _load_gogpt_oil_only_capacity_reference(path, year):
    gogpt = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "Country/Area",
        "Fuel classification?",
        "Capacity (MW)",
        "Status",
        "Start year",
        "Retired year",
    }
    missing = required.difference(gogpt.columns)
    if missing:
        raise ValueError(f"Missing required GOGPT oil columns in {path}: {sorted(missing)}")

    status = gogpt["Status"].astype(str).str.strip().str.lower()
    start_year = gogpt["Start year"].apply(_parse_gem_year)
    retired_year = gogpt["Retired year"].apply(_parse_gem_year)
    active = (
        status.eq("operating")
        & (start_year.isna() | start_year.le(int(year)))
        & (retired_year.isna() | retired_year.gt(int(year)))
    )
    oil_only = gogpt["Fuel classification?"].astype(str).str.strip().str.lower().eq("oil only")
    gogpt = gogpt.loc[active & oil_only].copy()
    if gogpt.empty:
        return pd.Series(dtype=float)

    lookup = {c: _gem_country_to_iso2(c) for c in gogpt["Country/Area"].dropna().unique()}
    gogpt["country"] = gogpt["Country/Area"].map(lookup)
    gogpt["reference_mw"] = pd.to_numeric(
        gogpt["Capacity (MW)"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    ).fillna(0.0)
    ref = gogpt.loc[gogpt["country"].notna()].groupby("country")["reference_mw"].sum()
    return ref


def apply_gogpt_oil_capacity_fix(n, investment_year, config):
    year = int(investment_year)
    if year < 2020:
        return
    reference_year = 2020 if year < 2025 else 2025
    fixed_historical_year = year <= 2025
    if n.links.empty:
        return

    oil_csv = _repo_path("validation/data/Global-Oil-and-Gas-Plant-Tracker-GOGPT-August-2025.csv")
    if not os.path.exists(oil_csv):
        logger.warning("GOGPT oil capacity fix skipped: file not found at %s", oil_csv)
        return

    ref = _load_gogpt_oil_only_capacity_reference(oil_csv, reference_year)
    if ref.empty:
        logger.warning("GOGPT oil capacity fix skipped: no oil-only reference rows for %s", reference_year)
        return

    bus_country = _bus_country_lookup(n).astype(str).str.strip().str.upper()
    records = []
    for asset, row in n.links.loc[n.links.carrier.astype(str).eq("oil")].iterrows():
        bus_col, eff_col = _link_ac_output_port(row, n)
        if bus_col is None:
            continue
        bus = row[bus_col]
        country = bus_country.get(bus, "")
        if not re.match(r"^[A-Z]{2}$", str(country)):
            continue
        eff = float(pd.to_numeric(row.get(eff_col, 1.0), errors="coerce") or 0.0)
        if eff <= 0.0:
            continue
        p_nom = float(pd.to_numeric(row.get("p_nom", 0.0), errors="coerce") or 0.0)
        records.append(
            {
                "asset": asset,
                "country": country,
                "efficiency": abs(eff),
                "electric_capacity_mw": p_nom * abs(eff),
            }
        )
    assets = pd.DataFrame(records)
    if assets.empty:
        logger.warning("GOGPT oil capacity fix skipped: no oil links with AC output found.")
        return

    oil_template = n.links.loc[n.links.carrier.astype(str).eq("oil")].iloc[0]
    existing_oil_countries = set(assets["country"])
    added_links = 0
    for country in sorted(set(ref.index).difference(existing_oil_countries)):
        ac_buses = n.buses.index[
            n.buses.carrier.astype(str).eq("AC")
            & n.buses.country.astype(str).str.upper().eq(country)
        ]
        if len(ac_buses) == 0:
            continue
        ac_bus = ac_buses[0]
        oil_bus = f"{ac_bus} oil"
        if oil_bus not in n.buses.index:
            n.add(
                "Bus",
                oil_bus,
                carrier="oil",
                country=country,
                x=0.0,
                y=0.0,
            )
        link_name = f"{ac_bus} oil-{reference_year}"
        if link_name in n.links.index:
            continue
        eff = float(pd.to_numeric(oil_template.get("efficiency", 0.35), errors="coerce") or 0.35)
        bus2 = oil_template.get("bus2", "")
        if not isinstance(bus2, str) or bus2 not in n.buses.index:
            bus2 = "co2 atmosphere" if "co2 atmosphere" in n.buses.index else ""
        attrs = {
            "bus0": oil_bus,
            "bus1": ac_bus,
            "carrier": "oil",
            "p_nom_extendable": False,
            "efficiency": eff,
            "capital_cost": oil_template.get("capital_cost", 0.0),
            "marginal_cost": oil_template.get("marginal_cost", 0.0),
            "lifetime": oil_template.get("lifetime", 25.0),
            "build_year": reference_year,
            "p_min_pu": oil_template.get("p_min_pu", 0.0),
            "p_max_pu": oil_template.get("p_max_pu", 1.0),
            "p_nom": 0.0,
            "p_nom_min": 0.0,
            "p_nom_max": 0.0,
        }
        if bus2:
            attrs["bus2"] = bus2
            attrs["efficiency2"] = oil_template.get("efficiency2", 0.0)
        n.add("Link", link_name, **attrs)
        records.append(
            {
                "asset": link_name,
                "country": country,
                "efficiency": abs(eff),
                "electric_capacity_mw": 0.0,
            }
        )
        added_links += 1

    assets = pd.DataFrame(records)
    matched_countries = sorted(set(assets["country"]).intersection(set(ref.index)))
    if not matched_countries:
        logger.warning("GOGPT oil capacity fix skipped: no model oil countries matched GOGPT.")
        return

    before_mw = float(assets["electric_capacity_mw"].sum())
    target_by_asset = pd.Series(0.0, index=assets["asset"], dtype=float)
    for country, group in assets.groupby("country"):
        country_target = float(ref.get(country, 0.0))
        if country_target <= 0.0:
            continue
        weights = group["electric_capacity_mw"].clip(lower=0.0)
        if float(weights.sum()) <= 0.0:
            weights = pd.Series(1.0, index=group.index)
        shares = weights / float(weights.sum())
        target_by_asset.loc[group["asset"].values] = shares.values * country_target

    link_idx = pd.Index(target_by_asset.index)
    eff = assets.set_index("asset").loc[link_idx, "efficiency"]
    p_nom_target = target_by_asset.div(eff).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    for col in ["p_nom", "p_nom_min", "p_nom_opt"]:
        if col in n.links.columns:
            n.links.loc[link_idx, col] = p_nom_target.reindex(link_idx).to_numpy()
    if "p_nom_max" in n.links.columns:
        if fixed_historical_year:
            n.links.loc[link_idx, "p_nom_max"] = p_nom_target.reindex(link_idx).to_numpy()
        else:
            n.links.loc[link_idx, "p_nom_max"] = np.inf
    if "p_nom_extendable" in n.links.columns:
        n.links.loc[link_idx, "p_nom_extendable"] = not fixed_historical_year

    after_mw = float(target_by_asset.sum())
    zeroed_countries = sorted(set(assets["country"]).difference(set(ref.index)))
    logger.info(
        "Applied GOGPT oil capacity fix for %s from %s: reference_year=%s, fixed_historical_year=%s, oil_links=%d, added_links=%d, matched_countries=%d, electric_capacity_before=%.2f GW, electric_capacity_after=%.2f GW, reference=%.2f GW, zeroed_model_countries=%d%s",
        year,
        oil_csv,
        reference_year,
        fixed_historical_year,
        len(link_idx),
        added_links,
        len(matched_countries),
        before_mw / 1000.0,
        after_mw / 1000.0,
        float(ref.sum()) / 1000.0,
        len(zeroed_countries),
        f" (sample: {', '.join(zeroed_countries[:5])})" if zeroed_countries else "",
    )


def _load_ember_bioenergy_capacity_reference(path, year):
    df = pd.read_csv(path)
    required = {
        "ISO 3 code",
        "Year",
        "Area type",
        "Category",
        "Variable",
        "Unit",
        "Value",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Ember bioenergy capacity reference {path} missing columns: {sorted(missing)}"
        )

    ref = df.loc[
        df["Category"].astype(str).eq("Capacity")
        & df["Variable"].astype(str).eq("Bioenergy")
        & df["Unit"].astype(str).eq("GW")
        & df["Area type"].astype(str).eq("Country or economy")
        & pd.to_numeric(df["Year"], errors="coerce").eq(int(year))
    ].copy()
    if ref.empty:
        return pd.Series(dtype=float)

    ref["country"] = ref["ISO 3 code"].astype(str).str.strip().apply(_safe_iso3_to_iso2)
    ref["capacity_mw"] = pd.to_numeric(ref["Value"], errors="coerce") * 1000.0
    ref = ref.dropna(subset=["country", "capacity_mw"])
    ref = ref.loc[ref["capacity_mw"] >= 0.0]
    return ref.groupby("country")["capacity_mw"].sum()


def apply_ember_bioenergy_capacity_fix(n, investment_year, config):
    year = int(investment_year)
    if year < 2020:
        return
    reference_year = 2020 if year < 2025 else 2025
    fixed_historical_year = year <= 2025
    if n.links.empty:
        return

    ember_csv = _repo_path("validation/data/ember_yearly_full_release_long_format.csv")
    if not os.path.exists(ember_csv):
        logger.warning("Ember bioenergy capacity fix skipped: file not found at %s", ember_csv)
        return

    ref = _load_ember_bioenergy_capacity_reference(ember_csv, reference_year)
    if ref.empty:
        logger.warning(
            "Ember bioenergy capacity fix skipped: no country Bioenergy capacity rows for %s",
            reference_year,
        )
        return

    def _finite_float(value, default=0.0):
        value = pd.to_numeric(value, errors="coerce")
        if pd.isna(value) or not np.isfinite(float(value)):
            return default
        return float(value)

    bus_country = _bus_country_lookup(n).astype(str).str.strip().str.upper()

    def _link_electric_capacity_records(carriers):
        rows = []
        carrier_set = set(carriers)
        for asset, row in n.links.loc[n.links.carrier.astype(str).isin(carrier_set)].iterrows():
            bus_col, eff_col = _link_ac_output_port(row, n)
            if bus_col is None:
                continue
            bus = row[bus_col]
            country = bus_country.get(bus, "")
            if not re.match(r"^[A-Z]{2}$", str(country)):
                continue
            eff = abs(_finite_float(row.get(eff_col, 1.0), 0.0))
            if eff <= 0.0:
                continue
            p_nom = _finite_float(row.get("p_nom", 0.0), 0.0)
            p_nom_opt = _finite_float(row.get("p_nom_opt", np.nan), np.nan)
            effective_p_nom = max([x for x in [p_nom, p_nom_opt, 0.0] if np.isfinite(x)])
            rows.append(
                {
                    "asset": asset,
                    "country": country,
                    "efficiency": eff,
                    "electric_capacity_mw": effective_p_nom * eff,
                }
            )
        return pd.DataFrame(rows)

    dedicated_assets = _link_electric_capacity_records(["biomass"])
    chp_assets = _link_electric_capacity_records(
        ["urban central solid biomass CHP", "urban central solid biomass CHP CC"]
    )
    if dedicated_assets.empty and chp_assets.empty:
        logger.warning("Ember bioenergy capacity fix skipped: no bioenergy links with AC output found.")
        return

    chp_by_country = (
        chp_assets.groupby("country")["electric_capacity_mw"].sum()
        if not chp_assets.empty
        else pd.Series(dtype=float)
    )
    dedicated_target = (ref - chp_by_country.reindex(ref.index).fillna(0.0)).clip(lower=0.0)
    chp_exceeds_ref = ref.index[chp_by_country.reindex(ref.index).fillna(0.0).gt(ref)].tolist()

    biomass_template = None
    if not n.links.loc[n.links.carrier.astype(str).eq("biomass")].empty:
        biomass_template = n.links.loc[n.links.carrier.astype(str).eq("biomass")].iloc[0]

    records = dedicated_assets.to_dict("records") if not dedicated_assets.empty else []
    existing_dedicated_countries = set(dedicated_assets["country"]) if not dedicated_assets.empty else set()
    added_links = 0
    for country in sorted(set(dedicated_target.loc[dedicated_target > 0.0].index).difference(existing_dedicated_countries)):
        if biomass_template is None:
            continue
        bus_carrier = (
            n.buses["carrier"].astype(str)
            if "carrier" in n.buses.columns
            else pd.Series("", index=n.buses.index)
        )
        ac_buses = n.buses.index[
            bus_carrier.eq("AC")
            & bus_country.reindex(n.buses.index).astype(str).str.upper().eq(country)
        ]
        if len(ac_buses) == 0:
            continue
        ac_bus = str(ac_buses[0])
        bus0 = biomass_template.get("bus0", "")
        if not isinstance(bus0, str) or bus0 not in n.buses.index:
            bus0 = "Earth solid biomass power" if "Earth solid biomass power" in n.buses.index else ""
        if not bus0:
            continue
        link_name = f"{ac_bus} biomass-{reference_year}"
        if link_name in n.links.index:
            continue
        eff = abs(_finite_float(biomass_template.get("efficiency", 0.35), 0.35))
        if eff <= 0.0:
            continue
        attrs = {
            "bus0": bus0,
            "bus1": ac_bus,
            "carrier": "biomass",
            "p_nom_extendable": False,
            "efficiency": eff,
            "capital_cost": biomass_template.get("capital_cost", 0.0),
            "marginal_cost": biomass_template.get("marginal_cost", 0.0),
            "lifetime": biomass_template.get("lifetime", 25.0),
            "build_year": reference_year,
            "p_min_pu": biomass_template.get("p_min_pu", 0.0),
            "p_max_pu": biomass_template.get("p_max_pu", 1.0),
            "p_nom": 0.0,
            "p_nom_min": 0.0,
            "p_nom_max": 0.0,
        }
        for port in [2, 3, 4]:
            bus_col = f"bus{port}"
            eff_col = f"efficiency{port}"
            bus_value = biomass_template.get(bus_col, "")
            if isinstance(bus_value, str) and bus_value in n.buses.index:
                attrs[bus_col] = bus_value
                attrs[eff_col] = biomass_template.get(eff_col, 0.0)
        n.add("Link", link_name, **attrs)
        records.append(
            {
                "asset": link_name,
                "country": country,
                "efficiency": eff,
                "electric_capacity_mw": 0.0,
            }
        )
        added_links += 1

    assets = pd.DataFrame(records)
    if assets.empty:
        logger.warning(
            "Ember bioenergy capacity fix skipped: no dedicated biomass links available to adjust."
        )
        return

    adjustable_assets = assets.loc[assets["country"].isin(dedicated_target.index)].copy()
    if adjustable_assets.empty:
        logger.warning("Ember bioenergy capacity fix skipped: no biomass assets in Ember countries.")
        return

    before_dedicated_mw = float(adjustable_assets["electric_capacity_mw"].sum())
    target_by_asset = pd.Series(0.0, index=adjustable_assets["asset"], dtype=float)
    for country, group in adjustable_assets.groupby("country"):
        country_target = float(dedicated_target.get(country, 0.0))
        if country_target <= 0.0:
            continue
        weights = group["electric_capacity_mw"].clip(lower=0.0)
        if float(weights.sum()) <= 0.0:
            weights = pd.Series(1.0, index=group.index)
        shares = weights / float(weights.sum())
        target_by_asset.loc[group["asset"].values] = shares.values * country_target

    link_idx = pd.Index(target_by_asset.index)
    eff = assets.set_index("asset").loc[link_idx, "efficiency"]
    p_nom_target = target_by_asset.div(eff).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    if "p_nom" in n.links.columns:
        n.links.loc[link_idx, "p_nom"] = p_nom_target.reindex(link_idx).to_numpy()
    if "p_nom_opt" in n.links.columns:
        n.links.loc[link_idx, "p_nom_opt"] = p_nom_target.reindex(link_idx).to_numpy()
    if "p_nom_min" in n.links.columns:
        if fixed_historical_year:
            n.links.loc[link_idx, "p_nom_min"] = 0.0
        else:
            n.links.loc[link_idx, "p_nom_min"] = p_nom_target.reindex(link_idx).to_numpy()
    if "p_nom_max" in n.links.columns:
        if fixed_historical_year:
            n.links.loc[link_idx, "p_nom_max"] = p_nom_target.reindex(link_idx).to_numpy()
        else:
            n.links.loc[link_idx, "p_nom_max"] = np.inf
    if "p_nom_extendable" in n.links.columns:
        n.links.loc[link_idx, "p_nom_extendable"] = True

    after_dedicated_mw = float(target_by_asset.sum())
    model_bioenergy_countries = sorted(
        set(ref.index).intersection(set(assets["country"]).union(set(chp_by_country.index)))
    )
    matched_ref_mw = float(ref.reindex(model_bioenergy_countries).fillna(0.0).sum())
    matched_chp_mw = float(chp_by_country.reindex(model_bioenergy_countries).fillna(0.0).sum())
    total_after_mw = after_dedicated_mw + matched_chp_mw
    untouched_dedicated_countries = sorted(set(assets["country"]).difference(set(ref.index)))
    if fixed_historical_year and untouched_dedicated_countries:
        untouched_assets = pd.Index(assets.loc[assets["country"].isin(untouched_dedicated_countries), "asset"])
        for col in ["p_nom", "p_nom_min", "p_nom_max", "p_nom_opt"]:
            if col in n.links.columns:
                n.links.loc[untouched_assets, col] = 0.0
        if "p_nom_extendable" in n.links.columns:
            n.links.loc[untouched_assets, "p_nom_extendable"] = False

    logger.info(
        "Applied Ember bioenergy capacity fix for %s from %s: reference_year=%s, fixed_historical_year=%s, adjusted_dedicated_biomass_links=%d, added_links=%d, matched_countries=%d, dedicated_before=%.2f GW, dedicated_after=%.2f GW, preserved_chp=%.2f GW, total_after=%.2f GW, reference=%.2f GW, chp_exceeds_reference_countries=%d%s, untouched_model_countries_without_ember=%d%s",
        year,
        ember_csv,
        reference_year,
        fixed_historical_year,
        len(link_idx),
        added_links,
        len(model_bioenergy_countries),
        before_dedicated_mw / 1000.0,
        after_dedicated_mw / 1000.0,
        matched_chp_mw / 1000.0,
        total_after_mw / 1000.0,
        matched_ref_mw / 1000.0,
        len(chp_exceeds_ref),
        f" (sample: {', '.join(chp_exceeds_ref[:5])})" if chp_exceeds_ref else "",
        len(untouched_dedicated_countries),
        f" (sample: {', '.join(untouched_dedicated_countries[:5])})" if untouched_dedicated_countries else "",
    )


def apply_pris_nuclear_capacity_and_availability(n, investment_year, config):
    pris_csv = _repo_path("validation/data/pris_nuclear_validation.csv")
    if not os.path.exists(pris_csv):
        logger.warning("PRIS nuclear adjustment skipped: file not found at %s", pris_csv)
        return
    if n.generators.empty:
        return

    idx = n.generators.index[n.generators.carrier.astype(str).eq("nuclear")]
    if len(idx) == 0:
        logger.warning("PRIS nuclear adjustment skipped: no nuclear generators found.")
        return

    pris = pd.read_csv(pris_csv)
    required = {
        "country_iso2",
        "pris_capacity_gw",
        "pris_nuclear_generation_twh",
    }
    missing = required.difference(pris.columns)
    if missing:
        logger.warning(
            "PRIS nuclear adjustment skipped: missing columns %s in %s",
            sorted(missing),
            pris_csv,
        )
        return

    pris = pris.copy()
    pris["country"] = pris["country_iso2"].astype(str).str.upper().str.strip()
    pris["capacity_gw"] = pd.to_numeric(pris["pris_capacity_gw"], errors="coerce")
    pris["generation_twh"] = pd.to_numeric(
        pris["pris_nuclear_generation_twh"], errors="coerce"
    )
    pris = pris.loc[
        pris["country"].str.len().eq(2) & pris["capacity_gw"].gt(0.0)
    ].copy()
    if pris.empty:
        logger.warning("PRIS nuclear adjustment skipped: no valid country rows in %s", pris_csv)
        return

    ref = pris.groupby("country", as_index=True)[["capacity_gw", "generation_twh"]].sum()
    ref["p_max_pu"] = (
        ref["generation_twh"] / (ref["capacity_gw"] * 8.76)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=1.0)

    bus_country_lu = _bus_country_lookup(n)
    country = n.generators.loc[idx, "bus"].map(bus_country_lu).fillna("")
    valid = country.ne("")
    idx = idx[valid.values]
    country = country.loc[valid]
    if len(idx) == 0:
        logger.warning("PRIS nuclear adjustment skipped: no nuclear generators with country mapping.")
        return

    model_cap_mw = pd.to_numeric(n.generators.loc[idx, "p_nom"], errors="coerce").fillna(0.0)
    model_country_cap_mw = model_cap_mw.groupby(country).sum()
    matched_countries = model_country_cap_mw.index.intersection(ref.index)
    if len(matched_countries) == 0:
        logger.warning("PRIS nuclear adjustment skipped: no model nuclear countries matched PRIS.")
        return

    target_cap_mw = ref.loc[matched_countries, "capacity_gw"] * 1000.0
    factors = target_cap_mw.div(model_country_cap_mw.reindex(matched_countries)).replace(
        [np.inf, -np.inf], np.nan
    )
    factors = factors.dropna()
    gen_factor = country.map(factors).fillna(1.0)
    matched_idx = idx[country.isin(factors.index).values]
    if len(matched_idx) > 0:
        n.generators.loc[matched_idx, "p_nom"] = (
            pd.to_numeric(n.generators.loc[matched_idx, "p_nom"], errors="coerce")
            .fillna(0.0)
            .mul(gen_factor.reindex(matched_idx).fillna(1.0))
        )
        if "p_nom_opt" in n.generators.columns:
            n.generators.loc[matched_idx, "p_nom_opt"] = n.generators.loc[matched_idx, "p_nom"]
        if "p_nom_min" in n.generators.columns:
            n.generators.loc[matched_idx, "p_nom_min"] = n.generators.loc[matched_idx, "p_nom"]
        if "p_nom_max" in n.generators.columns:
            current_max = pd.to_numeric(n.generators.loc[matched_idx, "p_nom_max"], errors="coerce")
            n.generators.loc[matched_idx, "p_nom_max"] = current_max.where(
                current_max.ge(n.generators.loc[matched_idx, "p_nom"]),
                n.generators.loc[matched_idx, "p_nom"],
            )
        if "p_nom_extendable" in n.generators.columns:
            n.generators.loc[matched_idx, "p_nom_extendable"] = False

    unmatched_idx = idx[~country.isin(ref.index).values]
    if len(unmatched_idx) > 0:
        for col in ["p_nom", "p_nom_opt", "p_nom_min", "p_nom_max"]:
            if col in n.generators.columns:
                n.generators.loc[unmatched_idx, col] = 0.0
        if "p_nom_extendable" in n.generators.columns:
            n.generators.loc[unmatched_idx, "p_nom_extendable"] = False

    pmax = country.map(ref["p_max_pu"]).dropna()
    pmax_idx = idx.intersection(pmax.index)
    ts_cols = n.generators_t.p_max_pu.columns.intersection(pmax_idx)
    static_cols = pmax_idx.difference(ts_cols)
    if len(ts_cols) > 0:
        n.generators_t.p_max_pu.loc[:, ts_cols] = pmax.reindex(ts_cols).to_numpy()
    if len(static_cols) > 0:
        n.generators.loc[static_cols, "p_max_pu"] = pmax.reindex(static_cols).to_numpy()
    if len(unmatched_idx) > 0:
        unmatched_ts_cols = n.generators_t.p_max_pu.columns.intersection(unmatched_idx)
        unmatched_static_cols = unmatched_idx.difference(unmatched_ts_cols)
        if len(unmatched_ts_cols) > 0:
            n.generators_t.p_max_pu.loc[:, unmatched_ts_cols] = 0.0
        if len(unmatched_static_cols) > 0:
            n.generators.loc[unmatched_static_cols, "p_max_pu"] = 0.0

    missing_model_countries = sorted(set(ref.index).difference(set(model_country_cap_mw.index)))
    zeroed_model_countries = sorted(set(country.loc[unmatched_idx]))
    logger.info(
        "Applied PRIS nuclear capacity/availability for %s from %s: matched_countries=%d, nuclear_generators=%d, capacity_before=%.2f GW, capacity_after=%.2f GW, reference_capacity=%.2f GW, cf_range=[%.3f, %.3f], zeroed_model_countries=%d%s, missing_model_countries=%d%s",
        investment_year,
        pris_csv,
        len(matched_countries),
        len(matched_idx),
        model_country_cap_mw.reindex(matched_countries).sum() / 1000.0,
        pd.to_numeric(n.generators.loc[matched_idx, "p_nom"], errors="coerce").sum() / 1000.0,
        ref.loc[matched_countries, "capacity_gw"].sum(),
        ref.loc[matched_countries, "p_max_pu"].min(),
        ref.loc[matched_countries, "p_max_pu"].max(),
        len(zeroed_model_countries),
        f" (sample: {', '.join(zeroed_model_countries[:5])})" if zeroed_model_countries else "",
        len(missing_model_countries),
        f" (sample: {', '.join(missing_model_countries[:5])})" if missing_model_countries else "",
    )



def apply_final_historical_capacity_validation_fixes(n, investment_year, config, context=""):
    """Re-apply historical capacity validation after baseyear/brownfield stock edits."""
    year = int(investment_year)
    label = f" for {context}" if context else ""
    logger.info("Applying final historical capacity validation fixes for %s%s.", year, label)

    _repair_fossil_link_fuel_buses(n)
    apply_pris_nuclear_capacity_and_availability(n, year, config)
    apply_gogpt_oil_capacity_fix(n, year, config)
    apply_ember_bioenergy_capacity_fix(n, year, config)

    if hasattr(n, "links") and not n.links.empty:
        links = n.links
        if "p_nom" in links.columns:
            nan_p_nom = links["p_nom"].isna()
            if nan_p_nom.any():
                logger.warning(
                    "Final capacity validation cleanup filled %d link p_nom NaNs with 0.",
                    int(nan_p_nom.sum()),
                )
                links.loc[nan_p_nom, "p_nom"] = 0.0
        if "p_nom_min" in links.columns:
            nan_p_nom_min = links["p_nom_min"].isna()
            if nan_p_nom_min.any():
                links.loc[nan_p_nom_min, "p_nom_min"] = 0.0
        if "p_nom_max" in links.columns:
            nan_p_nom_max = links["p_nom_max"].isna()
            if nan_p_nom_max.any():
                links.loc[nan_p_nom_max, "p_nom_max"] = np.inf
        if "p_nom_extendable" in links.columns:
            links["p_nom_extendable"] = links["p_nom_extendable"].fillna(False).astype(bool)

    if hasattr(n, "generators") and not n.generators.empty:
        gens = n.generators
        if "p_nom" in gens.columns:
            nan_p_nom = gens["p_nom"].isna()
            if nan_p_nom.any():
                fallback = (
                    gens["p_nom_min"].fillna(0.0)
                    if "p_nom_min" in gens.columns
                    else pd.Series(0.0, index=gens.index)
                )
                gens.loc[nan_p_nom, "p_nom"] = fallback.loc[nan_p_nom]
        if "p_nom_min" in gens.columns:
            gens["p_nom_min"] = gens["p_nom_min"].fillna(gens["p_nom"].fillna(0.0))
        if "p_nom_max" in gens.columns:
            gens["p_nom_max"] = gens["p_nom_max"].fillna(np.inf)
        if "p_nom_extendable" in gens.columns:
            gens["p_nom_extendable"] = gens["p_nom_extendable"].fillna(False).astype(bool)


def apply_country_nuclear_iteration_scaling(n, investment_year, config):
    """
    Apply per-country iterative p_max_pu scaling for nuclear generators.

    Override CSV format:
    - country (ISO2)
    - nuclear_scale
    """
    global_cfg = config.get("global_specific", {})
    base_cfg = global_cfg.get("baseyear_generation", {})
    if not bool(base_cfg.get("nuclear_iteration_scaling_enabled", False)):
        return

    baseyear = int(base_cfg.get("year", 2020))
    if int(investment_year) != baseyear:
        logger.info(
            "Applying iterative nuclear scaling to %s using overrides calibrated from baseyear %s.",
            investment_year,
            baseyear,
        )

    override_csv_cfg = base_cfg.get("nuclear_iteration_override_csv", "")
    if not override_csv_cfg:
        logger.warning(
            "Iterative nuclear scaling enabled but no nuclear_iteration_override_csv configured."
        )
        return

    override_csv = _repo_path(override_csv_cfg)
    if not os.path.exists(override_csv):
        logger.warning(
            "Iterative nuclear scaling skipped: override file not found at %s", override_csv
        )
        return

    override = pd.read_csv(override_csv)
    if override.empty:
        logger.warning(
            "Iterative nuclear scaling skipped: override file is empty at %s", override_csv
        )
        return

    cols = {c.lower().strip(): c for c in override.columns}
    if "country" not in cols:
        logger.warning(
            "Iterative nuclear scaling skipped: column 'country' missing in %s", override_csv
        )
        return
    rename_cols = {cols["country"]: "country"}
    if "nuclear_scale" in cols:
        rename_cols[cols["nuclear_scale"]] = "nuclear_scale"
    override = override.rename(columns=rename_cols)
    if "nuclear_scale" not in override.columns:
        override["nuclear_scale"] = 1.0

    override["country"] = override["country"].astype(str).str.upper().str.strip()
    override = override.loc[override["country"].str.len().eq(2)].copy()
    if override.empty:
        logger.warning(
            "Iterative nuclear scaling skipped: no valid ISO2 countries in %s", override_csv
        )
        return

    min_scale = float(base_cfg.get("nuclear_iteration_scale_min", 0.05))
    max_scale = float(base_cfg.get("nuclear_iteration_scale_max", 20.0))
    override["nuclear_scale"] = (
        pd.to_numeric(override["nuclear_scale"], errors="coerce")
        .fillna(1.0)
        .clip(lower=min_scale, upper=max_scale)
    )
    scale_by_country = override.groupby("country")["nuclear_scale"].mean()

    if n.generators.empty:
        return
    idx = n.generators.index[n.generators.carrier.astype(str).eq("nuclear")]
    if len(idx) == 0:
        logger.warning("Iterative nuclear scaling skipped: no nuclear generators found.")
        return

    bus_country_lu = _bus_country_lookup(n)
    country = n.generators.loc[idx, "bus"].map(bus_country_lu).fillna("")
    valid = country.ne("")
    idx = idx[valid.values]
    country = country.loc[valid]
    if len(idx) == 0:
        logger.warning(
            "Iterative nuclear scaling skipped: no nuclear generators with country mapping."
        )
        return

    factors = country.map(scale_by_country).fillna(1.0)
    scaled_entries = int((~np.isclose(factors, 1.0, atol=1e-12)).sum())
    pmax_cap = float(base_cfg.get("nuclear_p_max_pu_cap", 1.0))

    ts_cols = n.generators_t.p_max_pu.columns.intersection(idx)
    static_cols = idx.difference(ts_cols)
    if len(ts_cols) > 0:
        scaled_ts = n.generators_t.p_max_pu.loc[:, ts_cols].fillna(1.0).mul(
            factors.reindex(ts_cols).fillna(1.0), axis=1
        )
        if pmax_cap > 0.0:
            scaled_ts = scaled_ts.clip(lower=0.0, upper=pmax_cap)
        n.generators_t.p_max_pu.loc[:, ts_cols] = scaled_ts
    if len(static_cols) > 0:
        scaled_static = (
            pd.to_numeric(n.generators.loc[static_cols, "p_max_pu"], errors="coerce")
            .fillna(1.0)
            .mul(factors.reindex(static_cols).fillna(1.0))
        )
        if pmax_cap > 0.0:
            scaled_static = scaled_static.clip(lower=0.0, upper=pmax_cap)
        n.generators.loc[static_cols, "p_max_pu"] = scaled_static

    logger.info(
        "Applied iterative nuclear scaling from %s: nuclear_generators=%d (scaled_entries=%d), country_rows=%d, scale_bounds=[%.3f, %.3f], p_max_pu_cap=%.2f",
        override_csv,
        len(idx),
        scaled_entries,
        len(scale_by_country),
        min_scale,
        max_scale,
        pmax_cap,
    )





MODEL_CARRIER_TO_OWID_METRIC = {
    "solar": "solar_electricity",
    "onwind": "wind_electricity",
    "offwind-ac": "wind_electricity",
    "offwind-dc": "wind_electricity",
    "ror": "hydro_electricity",
    "hydro": "hydro_electricity",
    "geothermal": "other_renewable_electricity",
    "nuclear": "nuclear_electricity",
    "ocgt": "gas_electricity",
    "ccgt": "gas_electricity",
    "urban central gas chp": "gas_electricity",
    "urban central gas chp cc": "gas_electricity",
    "coal": "coal_electricity",
    "lignite": "coal_electricity",
    "oil": "oil_electricity",
    "biomass": "biofuel_electricity",
    "urban central solid biomass chp": "biofuel_electricity",
    "urban central solid biomass chp cc": "biofuel_electricity",
    "biomass eop": "biofuel_electricity",
}

MODEL_STORAGE_CARRIER_TO_OWID_METRIC = {
    "hydro": "hydro_electricity",
    "phs": "hydro_electricity",
}

OWID_BASE_METRICS = [
    "coal_electricity",
    "gas_electricity",
    "oil_electricity",
    "biofuel_electricity",
    "hydro_electricity",
    "nuclear_electricity",
    "solar_electricity",
    "wind_electricity",
    "other_renewable_electricity",
]

OWID_AGGREGATE_METRICS = [
    "fossil_electricity",
    "renewables_electricity",
    "low_carbon_electricity",
    "electricity_generation",
]

COUNTRY_METRIC_KEY_SEP = "|::|"


def _safe_iso3_to_iso2(code):
    if not isinstance(code, str) or len(code) != 3:
        return np.nan
    try:
        iso2 = three_2_two_digits_country(code)
    except Exception:
        return np.nan
    if not isinstance(iso2, str) or len(iso2) != 2:
        return np.nan
    return iso2.upper()


def _sanitize_constraint_token(value):
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")


def _normalize_string_set(value):
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set, pd.Index, np.ndarray)):
        return {str(v).strip() for v in value if str(v).strip()}
    text = str(value).strip()
    return {text} if text else set()


def _baseyear_blocked_extendable_link_carriers(config):
    global_cfg = config.get("global_specific", {}) if isinstance(config, dict) else {}
    cfg = global_cfg.get("baseyear_generation", {}) if isinstance(global_cfg, dict) else {}
    carriers = cfg.get(
        "blocked_extendable_link_carriers",
        BASEYEAR_BLOCKED_EXTENDABLE_LINK_CARRIERS,
    )
    return _normalize_string_set(carriers)


def _freeze_baseyear_loophole_links(n, baseyear, config=None):
    try:
        current_year = int(float(baseyear))
    except Exception:
        logger.warning(
            "Could not parse baseyear '%s' while freezing loophole-prone links.",
            baseyear,
        )
        return 0

    global_cfg = config.get("global_specific", {}) if isinstance(config, dict) else {}
    cfg = global_cfg.get("baseyear_generation", {}) if isinstance(global_cfg, dict) else {}
    target_year = int(cfg.get("year", 2020))
    if current_year != target_year or not hasattr(n, "links") or n.links.empty:
        return 0

    blocked_carriers = _baseyear_blocked_extendable_link_carriers(config)
    if not blocked_carriers:
        return 0

    links = n.links
    if "carrier" not in links.columns:
        return 0
    if "p_nom_extendable" not in links.columns:
        links["p_nom_extendable"] = np.zeros(len(links), dtype=np.bool_)

    blocked_norm = {carrier.casefold() for carrier in blocked_carriers}
    carrier_mask = links.carrier.fillna("").astype(str).str.casefold().isin(blocked_norm)
    if not carrier_mask.any():
        return 0

    if "build_year" in links.columns:
        build_year = pd.to_numeric(links.build_year, errors="coerce").fillna(0)
        candidate_mask = carrier_mask & build_year.le(float(current_year))
    else:
        logger.warning(
            "Link component has no build_year column; applying loophole freeze to all matching baseyear link carriers."
        )
        candidate_mask = carrier_mask

    extendable_mask = links.p_nom_extendable.fillna(False).astype(bool)
    frozen_assets = links.index[candidate_mask & extendable_mask]
    if len(frozen_assets) == 0:
        return 0

    links.loc[frozen_assets, "p_nom_extendable"] = False
    if "p_nom_min" not in links.columns:
        links["p_nom_min"] = 0.0
    links.loc[frozen_assets, "p_nom_min"] = pd.to_numeric(
        links.loc[frozen_assets, "p_nom"],
        errors="coerce",
    ).fillna(0.0)
    links["p_nom_extendable"] = links["p_nom_extendable"].fillna(False).astype(bool)

    frozen_counts = (
        links.loc[frozen_assets, "carrier"].astype(str).value_counts().to_dict()
    )
    logger.info(
        "In baseyear %s: Set %d blocked link assets to p_nom_extendable=False and p_nom_min=p_nom (carrier_counts=%s).",
        current_year,
        len(frozen_assets),
        frozen_counts,
    )
    return int(len(frozen_assets))


def _geothermal_template_from_network(n):
    gens = n.generators.loc[n.generators.carrier.astype(str).eq("geothermal")].copy()
    if gens.empty:
        return None

    template = {}
    numeric_defaults = {
        "marginal_cost": 0.0,
        "capital_cost": 0.0,
        "efficiency": 1.0,
        "lifetime": np.inf,
        "p_max_pu": 1.0,
    }
    for col, default in numeric_defaults.items():
        if col not in gens.columns:
            template[col] = default
            continue
        values = pd.to_numeric(gens[col], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).dropna()
        template[col] = float(values.median()) if not values.empty else default

    return template


def _generator_template_from_network(n, carrier):
    gens = n.generators.loc[n.generators.carrier.astype(str).eq(str(carrier))].copy()
    template = {}
    numeric_defaults = {
        "marginal_cost": 0.0,
        "capital_cost": 0.0,
        "efficiency": 1.0,
        "lifetime": np.inf,
        "p_max_pu": 1.0,
    }
    if gens.empty:
        return numeric_defaults

    for col, default in numeric_defaults.items():
        if col not in gens.columns:
            template[col] = default
            continue
        values = pd.to_numeric(gens[col], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).dropna()
        template[col] = float(values.median()) if not values.empty else default
    return template


def _pick_country_ac_bus(n, country, bus_country, preferred_gens=None):
    if preferred_gens is not None and not preferred_gens.empty:
        existing_bus = str(preferred_gens.iloc[0]["bus"])
        if existing_bus in n.buses.index:
            return existing_bus

    bus_country = bus_country.astype(str).str.strip().str.upper()
    bus_carrier = (
        n.buses["carrier"].astype(str)
        if "carrier" in n.buses.columns
        else pd.Series("", index=n.buses.index)
    )
    ac_buses = pd.Index(n.buses.index[bus_country.eq(country) & bus_carrier.eq("AC")])
    if len(ac_buses) == 0:
        return None

    if not n.loads.empty and "p_set" in n.loads_t:
        load_bus = n.loads["bus"].reindex(n.loads.index)
        country_loads = n.loads.index[load_bus.isin(ac_buses)]
        if len(country_loads) > 0:
            load_totals = (
                n.loads_t.p_set.reindex(columns=country_loads)
                .fillna(0.0)
                .sum(axis=0)
                .groupby(load_bus.loc[country_loads])
                .sum()
            )
            if not load_totals.empty:
                return str(load_totals.idxmax())

    return str(ac_buses.sort_values()[0])



def _country_ac_bus_load_shares(n, country, bus_country, candidate_buses=None):
    """Return AC-load shares for buses in a country, with equal-share fallback."""
    country = str(country).strip().upper()
    bus_country = bus_country.astype(str).str.strip().str.upper()
    bus_carrier = (
        n.buses["carrier"].astype(str)
        if "carrier" in n.buses.columns
        else pd.Series("", index=n.buses.index)
    )
    if candidate_buses is None:
        buses = pd.Index(n.buses.index[bus_country.eq(country) & bus_carrier.eq("AC")])
    else:
        candidate_buses = pd.Index(candidate_buses).astype(str)
        buses = pd.Index(
            [
                bus
                for bus in candidate_buses
                if bus in n.buses.index
                and bus_country.reindex([bus]).iloc[0] == country
                and bus_carrier.reindex([bus]).iloc[0] == "AC"
            ]
        )
    if len(buses) == 0:
        return pd.Series(dtype=float)

    load_by_bus = pd.Series(0.0, index=buses, dtype=float)
    if not n.loads.empty and hasattr(n, "loads_t") and "p_set" in n.loads_t:
        load_bus = n.loads["bus"].astype(str)
        country_loads = n.loads.index[load_bus.isin(buses)]
        if len(country_loads) > 0:
            load_ts = n.loads_t.p_set.reindex(columns=country_loads).fillna(0.0)
            if "generators" in n.snapshot_weightings:
                weighted = load_ts.mul(n.snapshot_weightings["generators"], axis=0)
                totals = weighted.sum(axis=0)
            else:
                totals = load_ts.sum(axis=0)
            load_by_bus = load_by_bus.add(
                totals.groupby(load_bus.loc[country_loads]).sum(), fill_value=0.0
            ).reindex(buses, fill_value=0.0)

    load_by_bus = load_by_bus.clip(lower=0.0)
    if load_by_bus.sum() > 0.0:
        return load_by_bus / load_by_bus.sum()
    return pd.Series(1.0 / len(buses), index=buses, dtype=float)


def _capacity_share_from_committed_or_load(n, group, country, bus_country):
    """Use committed nodal capacity shares, falling back to AC-load shares."""
    if group is not None and not group.empty:
        committed = pd.concat([group["p_nom"], group["p_nom_min"]], axis=1).max(axis=1)
        committed_by_bus = committed.groupby(group["bus"].astype(str)).sum().clip(lower=0.0)
        committed_by_bus = committed_by_bus.loc[committed_by_bus > 1e-9]
        if committed_by_bus.sum() > 0.0:
            return committed_by_bus / committed_by_bus.sum()

    return _country_ac_bus_load_shares(n, country, bus_country)



def _generator_snapshot_weights(n):
    if hasattr(n, "snapshot_weightings"):
        for column in ("generators", "objective"):
            if column in n.snapshot_weightings:
                weights = pd.to_numeric(
                    n.snapshot_weightings[column], errors="coerce"
                ).fillna(0.0)
                return weights.reindex(n.snapshots).fillna(0.0)
    return pd.Series(1.0, index=n.snapshots, dtype=float)


def _generator_profile_full_load_hours(
    n,
    gen_name=None,
    bus=None,
    carrier=None,
    current_year=None,
):
    if not hasattr(n, "generators_t") or "p_max_pu" not in n.generators_t:
        return np.nan

    if gen_name is not None and gen_name in n.generators_t.p_max_pu.columns:
        profile = pd.to_numeric(
            n.generators_t.p_max_pu[gen_name], errors="coerce"
        ).fillna(0.0)
    elif bus is not None and carrier is not None and current_year is not None:
        probe_name = f"{bus} {carrier}-{current_year}-energy-calibration"
        profile = _fixed_generator_profile(
            n,
            gen_name=probe_name,
            bus=str(bus),
            carrier=str(carrier),
            current_year=int(current_year),
        )
        if profile is None:
            return np.nan
    else:
        return np.nan

    profile = pd.to_numeric(profile, errors="coerce").fillna(0.0).clip(
        lower=0.0,
        upper=1.0,
    )
    if profile.empty or profile.abs().sum() <= 1e-12:
        return np.nan
    if np.isclose(float(profile.min()), 1.0) and np.isclose(float(profile.max()), 1.0):
        return np.nan

    weights = _generator_snapshot_weights(n)
    return float(profile.reindex(n.snapshots).fillna(0.0).mul(weights, axis=0).sum())


def _expected_generation_mwh_from_committed_generators(n, generators):
    if generators is None or generators.empty:
        return 0.0

    total = 0.0
    for gen_name, row in generators.iterrows():
        p_nom = pd.to_numeric(
            pd.Series([row.get("p_nom", 0.0), row.get("p_nom_min", 0.0)]),
            errors="coerce",
        ).fillna(0.0).max()
        if p_nom <= 0.0:
            continue
        flh = _generator_profile_full_load_hours(n, gen_name=gen_name)
        if not np.isfinite(flh):
            continue
        total += float(p_nom) * float(flh)
    return float(total)


def _bounded_share_average_range(lower, upper, values):
    values = pd.Series(values, dtype=float).dropna()
    if values.empty:
        return np.nan, np.nan

    lower = pd.Series(lower, index=values.index, dtype=float).clip(lower=0.0)
    upper = pd.Series(upper, index=values.index, dtype=float).clip(lower=0.0)
    upper = upper.where(upper >= lower, lower)

    def extreme_average(descending):
        shares = lower.copy()
        remaining = max(1.0 - float(shares.sum()), 0.0)
        order = values.sort_values(ascending=not descending).index
        for idx in order:
            room = float(upper.loc[idx] - shares.loc[idx])
            add = min(room, remaining)
            if add > 0.0:
                shares.loc[idx] += add
                remaining -= add
            if remaining <= 1e-12:
                break
        if shares.sum() <= 0.0:
            return np.nan
        shares = shares / shares.sum()
        return float((shares * values).sum())

    return extreme_average(False), extreme_average(True)


def _shift_bounded_shares_to_target_average(
    shares,
    values,
    target_average,
    min_multiplier=0.25,
    max_multiplier=3.0,
):
    """Use a tiny LP to choose bounded shares closest to a target weighted average.

    Primary objective: minimize absolute error in ``sum(shares * values)``.
    Secondary objective: among equally good energy matches, stay close to the
    input shares so calibration does not concentrate capacity unnecessarily.
    """
    shares = pd.Series(shares, dtype=float).clip(lower=0.0)
    values = pd.Series(values, dtype=float).reindex(shares.index)
    valid = values.replace([np.inf, -np.inf], np.nan).notna() & shares.gt(0.0)
    shares = shares.loc[valid]
    values = values.loc[valid]
    if shares.empty or shares.sum() <= 0.0 or not np.isfinite(target_average):
        return shares

    shares = shares / shares.sum()
    lower = shares * max(float(min_multiplier), 0.0)
    upper = shares * max(float(max_multiplier), float(min_multiplier))
    if lower.sum() > 1.0:
        lower = lower / lower.sum()
    if upper.sum() < 1.0:
        upper = upper + (1.0 - upper.sum()) * shares

    target = float(target_average)
    min_average, max_average = _bounded_share_average_range(lower, upper, values)
    if np.isfinite(min_average):
        target = max(target, min_average)
    if np.isfinite(max_average):
        target = min(target, max_average)

    try:
        from scipy.optimize import linprog
    except Exception as exc:
        logger.warning(
            "Falling back to uncalibrated shares because scipy.optimize.linprog is unavailable: %s",
            exc,
        )
        return shares

    n = len(shares)
    # Variables: x[0:n] calibrated shares, t absolute average error,
    # d_plus/d_minus absolute deviation from the input shares.
    t_idx = n
    d_plus_start = n + 1
    d_minus_start = n + 1 + n
    n_vars = n + 1 + 2 * n

    c = np.zeros(n_vars, dtype=float)
    c[t_idx] = 1.0
    # Small tie-breaker: prefer shares close to the load/committed baseline.
    c[d_plus_start:d_plus_start + n] = 1e-6
    c[d_minus_start:d_minus_start + n] = 1e-6

    value_arr = values.to_numpy(dtype=float)
    share_arr = shares.to_numpy(dtype=float)

    a_eq = []
    b_eq = []
    row = np.zeros(n_vars, dtype=float)
    row[:n] = 1.0
    a_eq.append(row)
    b_eq.append(1.0)

    for i in range(n):
        row = np.zeros(n_vars, dtype=float)
        row[i] = 1.0
        row[d_plus_start + i] = -1.0
        row[d_minus_start + i] = 1.0
        a_eq.append(row)
        b_eq.append(share_arr[i])

    a_ub = []
    b_ub = []
    row = np.zeros(n_vars, dtype=float)
    row[:n] = value_arr
    row[t_idx] = -1.0
    a_ub.append(row)
    b_ub.append(target)

    row = np.zeros(n_vars, dtype=float)
    row[:n] = -value_arr
    row[t_idx] = -1.0
    a_ub.append(row)
    b_ub.append(-target)

    bounds = [(float(lower.iloc[i]), float(upper.iloc[i])) for i in range(n)]
    bounds.append((0.0, None))
    bounds.extend([(0.0, None)] * (2 * n))

    result = linprog(
        c,
        A_ub=np.vstack(a_ub),
        b_ub=np.asarray(b_ub, dtype=float),
        A_eq=np.vstack(a_eq),
        b_eq=np.asarray(b_eq, dtype=float),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        logger.warning(
            "Tiny LP share calibration failed (%s); using input shares.",
            result.message,
        )
        return shares

    calibrated = pd.Series(result.x[:n], index=shares.index, dtype=float).clip(lower=0.0)
    if calibrated.sum() > 0.0:
        calibrated = calibrated / calibrated.sum()
    return calibrated


def _energy_calibrated_historical_capacity_shares(
    n,
    country,
    constraint_carrier,
    carrier,
    shares,
    gap_mw,
    gen,
    current_year,
    energy_reference_twh,
    min_reference_twh=1.0,
    min_share_multiplier=0.25,
    max_share_multiplier=3.0,
    bound_shares=None,
):
    metric = MODEL_CARRIER_TO_OWID_METRIC.get(str(carrier).lower())
    if metric is None:
        metric = MODEL_CARRIER_TO_OWID_METRIC.get(str(constraint_carrier).lower())
    if metric is None or energy_reference_twh is None:
        return shares, None

    reference_twh = energy_reference_twh.get((str(country).upper(), metric))
    if reference_twh is None or not np.isfinite(reference_twh):
        return shares, None
    if float(reference_twh) < float(min_reference_twh):
        return shares, None

    metric_carriers = {
        carrier_name
        for carrier_name, mapped_metric in MODEL_CARRIER_TO_OWID_METRIC.items()
        if mapped_metric == metric
    }
    country_gen = gen.loc[
        gen["country"].eq(str(country).upper())
        & gen["carrier"].astype(str).str.lower().isin(metric_carriers)
    ].copy()
    committed_metric_mwh = _expected_generation_mwh_from_committed_generators(
        n,
        country_gen,
    )
    target_added_mwh = max(float(reference_twh) * 1e6 - committed_metric_mwh, 0.0)
    if gap_mw <= 1e-9 or target_added_mwh <= 0.0:
        return shares, None

    flh = pd.Series(
        {
            bus: _generator_profile_full_load_hours(
                n,
                bus=str(bus),
                carrier=str(carrier),
                current_year=int(current_year),
            )
            for bus in shares.index
        },
        dtype=float,
    ).replace([np.inf, -np.inf], np.nan)
    flh = flh.dropna()
    if flh.empty:
        logger.warning(
            "Skipping 2025 energy calibration for %s %s: no valid node-specific %s profiles.",
            country,
            constraint_carrier,
            carrier,
        )
        return shares, None

    shares = shares.reindex(flh.index).fillna(0.0)
    shares = shares.loc[shares > 0.0]
    flh = flh.reindex(shares.index)
    if shares.empty or shares.sum() <= 0.0:
        return shares, None
    shares = shares / shares.sum()

    if bound_shares is None:
        calibration_base = shares.copy()
    else:
        calibration_base = pd.Series(bound_shares, dtype=float).reindex(shares.index).fillna(0.0).clip(lower=0.0)
        if calibration_base.sum() <= 0.0:
            logger.warning(
                "Using committed-share bounds for 2025 energy calibration for %s %s: no positive load-share baseline available.",
                country,
                constraint_carrier,
            )
            calibration_base = shares.copy()
        else:
            calibration_base = calibration_base / calibration_base.sum()

    target_flh = target_added_mwh / float(gap_mw)
    initial_flh = float((shares * flh).sum())
    calibrated = _shift_bounded_shares_to_target_average(
        shares=calibration_base,
        values=flh,
        target_average=target_flh,
        min_multiplier=min_share_multiplier,
        max_multiplier=max_share_multiplier,
    )
    calibrated = calibrated.reindex(shares.index).fillna(0.0)
    if calibrated.sum() > 0.0:
        calibrated = calibrated / calibrated.sum()
    final_flh = float((calibrated * flh).sum())

    info = {
        "metric": metric,
        "reference_twh": float(reference_twh),
        "committed_metric_twh": committed_metric_mwh / 1e6,
        "target_added_twh": target_added_mwh / 1e6,
        "initial_added_twh": float(gap_mw) * initial_flh / 1e6,
        "final_added_twh": float(gap_mw) * final_flh / 1e6,
        "target_flh": float(target_flh),
        "initial_flh": float(initial_flh),
        "final_flh": float(final_flh),
        "min_flh": float(flh.min()),
        "max_flh": float(flh.max()),
    }
    return calibrated, info


def add_year2025_irena_historical_capacity_distribution(n, planning_year, config):
    """
    Materialize 2025 historical renewable capacity additions before optimization.

    The country-level IRENA capacity band should verify total installed capacity,
    not decide nodal siting. For configured carriers, add the gap between the
    committed capacity and the IRENA 2025 reference as fixed 2025 generators.
    The energy-calibrated nodal shares are bounded relative to in-country
    AC-load shares, with committed 2020 shares only used as the initial stock
    signal where available.
    """
    global_cfg = config.get("global_specific", {})
    cfg = global_cfg.get("year2025_capacity", {})
    if not cfg or not cfg.get("year2025_capacity_constraint", False):
        return
    if not bool(cfg.get("materialize_historical_capacity_by_2020_shares", False)):
        return

    try:
        current_year = int(float(planning_year))
    except Exception:
        logger.warning(
            "Could not parse planning year '%s' for 2025 historical capacity materialization",
            planning_year,
        )
        return

    target_year = int(cfg.get("year", 2025))
    if current_year != target_year:
        return

    materialize_constraints = set(
        str(c)
        for c in cfg.get(
            "historical_capacity_distribution_constraints",
            ["solar", "onwind"],
        )
    )
    if not materialize_constraints:
        return

    irena_csv = _repo_path(
        cfg.get("irena_csv", "validation/data/irena_capacity_by_technology.csv")
    )
    if not os.path.exists(irena_csv):
        logger.warning(
            "2025 historical capacity materialization skipped: file not found at %s",
            irena_csv,
        )
        return

    min_reference_mw = float(cfg.get("min_reference_mw", 0.0))
    reference_year = int(cfg.get("reference_year", target_year))
    fallback_to_latest = bool(cfg.get("fallback_to_latest_available", True))
    freeze_endogenous = bool(
        cfg.get("freeze_endogenous_2025_capacity_after_materialization", True)
    )
    constraint_technology_map = cfg.get(
        "irena_technology_by_constraint",
        cfg.get("irena_technology_by_carrier", {}),
    )
    model_carriers_by_constraint = cfg.get(
        "model_carriers_by_constraint",
        {key: [key] for key in (constraint_technology_map or {}).keys()},
    )

    energy_calibration_enabled = bool(
        cfg.get("energy_calibrated_historical_capacity_distribution", False)
    )
    energy_calibration_constraints = set(
        str(c)
        for c in cfg.get(
            "energy_calibration_carriers",
            list(materialize_constraints),
        )
    )
    energy_reference_twh = None
    if energy_calibration_enabled:
        year2025_gen_cfg = global_cfg.get("year2025_generation", {}) or {}
        owid_csv = _repo_path(
            cfg.get(
                "energy_calibration_owid_csv",
                year2025_gen_cfg.get(
                    "owid_csv",
                    "validation/data/owid-elecbalance2025.csv",
                ),
            )
        )
        try:
            energy_ref = _owid_country_metric_reference(
                owid_csv=owid_csv,
                year=int(cfg.get("energy_calibration_reference_year", target_year)),
                metrics=["solar_electricity", "wind_electricity"],
            )
            energy_reference_twh = {
                (str(row.country).upper(), str(row.metric)): float(row.reference_twh)
                for row in energy_ref.itertuples(index=False)
                if pd.notna(row.reference_twh)
            }
            logger.info(
                "Loaded %d OWID 2025 energy references for historical capacity share calibration from %s.",
                len(energy_reference_twh),
                owid_csv,
            )
        except Exception as exc:
            energy_reference_twh = None
            logger.warning(
                "2025 historical capacity energy calibration disabled: could not load OWID references from %s: %s",
                owid_csv,
                exc,
            )

    try:
        ref, used_reference_year = _irena_country_capacity_reference(
            irena_csv=irena_csv,
            year=reference_year,
            carrier_technology_map=constraint_technology_map,
            fallback_to_latest=fallback_to_latest,
        )
    except Exception as exc:
        logger.warning("2025 historical capacity materialization skipped: %s", exc)
        return

    ref = ref.loc[ref["carrier"].astype(str).isin(materialize_constraints)].copy()
    if ref.empty or n.generators.empty:
        return

    bus_country = _get_bus_country_for_clustering(n).astype(str).str.strip().str.upper()
    gen = n.generators.copy()
    gen["country"] = (
        gen["bus"].map(bus_country).fillna("").astype(str).str.strip().str.upper()
    )
    gen["carrier"] = gen["carrier"].astype(str)
    gen["p_nom"] = pd.to_numeric(gen.get("p_nom", 0.0), errors="coerce").fillna(0.0)
    gen["p_nom_min"] = pd.to_numeric(gen.get("p_nom_min", 0.0), errors="coerce").fillna(0.0)
    gen["p_nom_extendable"] = gen.get("p_nom_extendable", False).fillna(False).astype(bool)
    if "build_year" in gen.columns:
        gen["build_year"] = pd.to_numeric(gen["build_year"], errors="coerce")
    else:
        gen["build_year"] = np.nan

    model_carrier_to_constraint = {}
    for constraint_name, carriers in (model_carriers_by_constraint or {}).items():
        if isinstance(carriers, str):
            carriers = [carriers]
        for carrier in carriers or []:
            model_carrier_to_constraint[str(carrier)] = str(constraint_name)
    gen["constraint_carrier"] = gen["carrier"].map(model_carrier_to_constraint)
    gen = gen.loc[
        gen["country"].str.match(r"^[A-Z]{2}$", na=False)
        & gen["constraint_carrier"].notna()
    ].copy()

    added = 0
    added_capacity_mw = 0.0
    frozen = 0
    skipped_no_bus_share = 0
    skipped_no_model_carrier = 0
    skipped_no_gap = 0

    for row in ref.itertuples(index=False):
        country = str(row.country).upper()
        constraint_carrier = str(row.carrier)
        target_mw = float(row.reference_mw)
        if target_mw < min_reference_mw:
            continue

        carriers = model_carriers_by_constraint.get(constraint_carrier, [])
        if isinstance(carriers, str):
            carriers = [carriers]
        carriers = [str(c) for c in carriers]
        if not carriers:
            skipped_no_model_carrier += 1
            continue

        group = gen.loc[
            gen["country"].eq(country)
            & gen["constraint_carrier"].eq(constraint_carrier)
        ].copy()
        committed = 0.0
        if not group.empty:
            committed = float(
                pd.concat([group["p_nom"], group["p_nom_min"]], axis=1).max(axis=1).sum()
            )

        gap_mw = max(target_mw - committed, 0.0)
        if gap_mw <= 1e-6:
            skipped_no_gap += 1
        else:
            shares = _capacity_share_from_committed_or_load(
                n=n,
                group=group,
                country=country,
                bus_country=bus_country,
            )
            shares = shares.loc[shares > 1e-12]
            if shares.empty:
                skipped_no_bus_share += 1
                logger.warning(
                    "Skipping 2025 historical %s materialization for %s: no committed-capacity or AC-load bus share available.",
                    constraint_carrier,
                    country,
                )
            else:
                carrier = carriers[0]
                if (
                    energy_calibration_enabled
                    and energy_reference_twh is not None
                    and constraint_carrier in energy_calibration_constraints
                ):
                    load_bound_shares = _country_ac_bus_load_shares(
                        n=n,
                        country=country,
                        bus_country=bus_country,
                        candidate_buses=shares.index,
                    )
                    shares, calibration_info = _energy_calibrated_historical_capacity_shares(
                        n=n,
                        country=country,
                        constraint_carrier=constraint_carrier,
                        carrier=carrier,
                        shares=shares,
                        gap_mw=gap_mw,
                        gen=gen,
                        current_year=current_year,
                        energy_reference_twh=energy_reference_twh,
                        min_reference_twh=float(
                            cfg.get("energy_calibration_min_reference_twh", 1.0)
                        ),
                        min_share_multiplier=float(
                            cfg.get("energy_calibration_min_share_multiplier", 0.25)
                        ),
                        max_share_multiplier=float(
                            cfg.get("energy_calibration_max_share_multiplier", 3.0)
                        ),
                        bound_shares=load_bound_shares,
                    )
                    shares = shares.loc[shares > 1e-12]
                    if calibration_info is not None:
                        logger.info(
                            "Energy-calibrated 2025 historical %s capacity for %s: OWID %.2f TWh, committed_%s %.2f TWh, target_added %.2f TWh, added %.2f -> %.2f TWh, FLH %.1f -> %.1f h (target %.1f h, node range %.1f-%.1f h).",
                            constraint_carrier,
                            country,
                            calibration_info["reference_twh"],
                            calibration_info["metric"],
                            calibration_info["committed_metric_twh"],
                            calibration_info["target_added_twh"],
                            calibration_info["initial_added_twh"],
                            calibration_info["final_added_twh"],
                            calibration_info["initial_flh"],
                            calibration_info["final_flh"],
                            calibration_info["target_flh"],
                            calibration_info["min_flh"],
                            calibration_info["max_flh"],
                        )
                    if shares.empty:
                        skipped_no_bus_share += 1
                        logger.warning(
                            "Skipping 2025 historical %s materialization for %s after energy calibration: no valid bus shares remain.",
                            constraint_carrier,
                            country,
                        )
                        continue

                if carrier not in n.carriers.index:
                    n.add("Carrier", carrier)
                template = _generator_template_from_network(n, carrier)

                for bus, share in shares.items():
                    cap_mw = float(gap_mw * share)
                    if cap_mw <= 1e-6:
                        continue
                    base_name = f"{bus} {carrier}-{current_year}-irena-historical"
                    gen_name = base_name
                    suffix = 2
                    while gen_name in n.generators.index:
                        gen_name = f"{base_name}-{suffix}"
                        suffix += 1

                    n.add(
                        "Generator",
                        gen_name,
                        bus=str(bus),
                        carrier=carrier,
                        p_nom=cap_mw,
                        p_nom_min=cap_mw,
                        p_nom_max=cap_mw,
                        p_nom_extendable=False,
                        marginal_cost=template["marginal_cost"],
                        capital_cost=template["capital_cost"],
                        efficiency=template["efficiency"],
                        build_year=current_year,
                        lifetime=template["lifetime"],
                    )
                    profile = _fixed_generator_profile(
                        n,
                        gen_name=gen_name,
                        bus=str(bus),
                        carrier=carrier,
                        current_year=current_year,
                    )
                    if profile is not None:
                        n.generators_t.p_max_pu[gen_name] = profile.values
                    added += 1
                    added_capacity_mw += cap_mw

                logger.info(
                    "Materialized 2025 historical %s capacity for %s: added %.2f MW across %d buses (target=%.2f MW, committed_before=%.2f MW, reference_year=%s).",
                    constraint_carrier,
                    country,
                    gap_mw,
                    len(shares),
                    target_mw,
                    committed,
                    used_reference_year,
                )

        if freeze_endogenous:
            current = n.generators.copy()
            current_country = current["bus"].map(bus_country).fillna("").astype(str).str.strip().str.upper()
            current_build_year = pd.to_numeric(current.get("build_year", np.nan), errors="coerce")
            current_extendable = current.get("p_nom_extendable", False).fillna(False).astype(bool)
            freeze_idx = current.index[
                current_country.eq(country)
                & current["carrier"].astype(str).isin(carriers)
                & current_extendable
                & current_build_year.eq(current_year)
            ]
            if len(freeze_idx) > 0:
                mins = pd.to_numeric(
                    n.generators.loc[freeze_idx, "p_nom_min"], errors="coerce"
                ).fillna(0.0)
                n.generators.loc[freeze_idx, "p_nom"] = mins.values
                n.generators.loc[freeze_idx, "p_nom_max"] = mins.values
                n.generators.loc[freeze_idx, "p_nom_extendable"] = False
                frozen += len(freeze_idx)

    logger.info(
        "2025 historical capacity materialization summary: added_generators=%d, added_capacity=%.2f MW, frozen_endogenous_candidates=%d, skipped_no_gap=%d, skipped_no_bus_share=%d, skipped_no_model_carrier=%d",
        added,
        added_capacity_mw,
        frozen,
        skipped_no_gap,
        skipped_no_bus_share,
        skipped_no_model_carrier,
    )

def _carrier_profile_name_parts(carrier):
    carrier = str(carrier)
    if carrier == "offwind-ac":
        return "offwind", "-ac"
    if carrier == "offwind-dc":
        return "offwind", "-dc"
    return carrier, ""


def _fixed_generator_profile(n, gen_name, bus, carrier, current_year):
    if not hasattr(n, "generators_t") or "p_max_pu" not in n.generators_t:
        return None

    generator, suffix = _carrier_profile_name_parts(carrier)
    profile_col = f"{bus} {generator}{suffix}-{current_year}"
    profile = pd.DataFrame(0.0, index=n.snapshots, columns=[profile_col])
    profile = _replace_zero_profile_columns_with_nearest(
        n=n,
        p_max_pu=profile,
        generator=generator,
        suffix=suffix,
        source_year=current_year,
        target_year=current_year,
    )
    series = pd.to_numeric(profile[profile_col], errors="coerce").fillna(0.0)
    if series.abs().sum() > 1e-12:
        series = series.clip(lower=0.0, upper=1.0)
        series.name = gen_name
        return series

    pool_cols = [
        c
        for c in n.generators_t.p_max_pu.columns
        if isinstance(c, str) and f" {generator}{suffix}-" in c
    ]
    if pool_cols:
        pool = n.generators_t.p_max_pu[pool_cols].fillna(0.0)
        nonzero = pool.columns[pool.sum(axis=0).abs() > 1e-12]
        if len(nonzero) > 0:
            raise ValueError(
                f"No node-specific {carrier} p_max_pu profile found for fixed 2025 IRENA generator {gen_name} on {bus}; refusing generic profile {nonzero[0]}."
            )

    raise ValueError(
        f"No node-specific {carrier} p_max_pu profile found for fixed 2025 IRENA generator {gen_name} on {bus}; refusing p_max_pu=1.0 fallback."
    )


def add_year2025_irena_missing_fixed_generators(n, planning_year, config):
    """
    Materialize known 2025 IRENA solar/wind capacity before optimization.

    This handles countries that gain a technology in the 2025 data but have no
    model asset/variable capable of reaching the hard country-capacity lower band.
    The added asset is fixed at the exact missing capacity needed to bring the
    country/constraint group to the IRENA reference.
    """
    global_cfg = config.get("global_specific", {})
    cfg = global_cfg.get("year2025_capacity", {})
    if not cfg or not cfg.get("year2025_capacity_constraint", False):
        return
    if not bool(cfg.get("add_missing_fixed_generators", True)):
        return

    try:
        current_year = int(float(planning_year))
    except Exception:
        logger.warning(
            "Could not parse planning year '%s' for 2025 IRENA fixed-generator materialization",
            planning_year,
        )
        return

    target_year = int(cfg.get("year", 2025))
    if current_year != target_year:
        return

    materialize_constraints = set(
        str(c)
        for c in cfg.get(
            "missing_fixed_generator_constraints",
            ["solar", "onwind", "offshore_wind"],
        )
    )
    if not materialize_constraints:
        return

    irena_csv = _repo_path(
        cfg.get("irena_csv", "validation/data/irena_capacity_by_technology.csv")
    )
    if not os.path.exists(irena_csv):
        logger.warning(
            "2025 IRENA fixed-generator materialization skipped: file not found at %s",
            irena_csv,
        )
        return

    tolerance = float(cfg.get("tolerance", 0.10))
    absolute_tolerance_mw = float(cfg.get("absolute_tolerance_mw", 0.0))
    min_reference_mw = float(cfg.get("min_reference_mw", 0.0))
    reference_year = int(cfg.get("reference_year", target_year))
    fallback_to_latest = bool(cfg.get("fallback_to_latest_available", True))
    constraint_technology_map = cfg.get(
        "irena_technology_by_constraint",
        cfg.get("irena_technology_by_carrier", {}),
    )
    model_carriers_by_constraint = cfg.get(
        "model_carriers_by_constraint",
        {key: [key] for key in (constraint_technology_map or {}).keys()},
    )

    try:
        ref, used_reference_year = _irena_country_capacity_reference(
            irena_csv=irena_csv,
            year=reference_year,
            carrier_technology_map=constraint_technology_map,
            fallback_to_latest=fallback_to_latest,
        )
    except Exception as exc:
        logger.warning("2025 IRENA fixed-generator materialization skipped: %s", exc)
        return

    ref = ref.loc[ref["carrier"].astype(str).isin(materialize_constraints)].copy()
    if ref.empty or n.generators.empty:
        return

    bus_country = _get_bus_country_for_clustering(n).astype(str).str.strip().str.upper()
    gen = n.generators.copy()
    gen["country"] = (
        gen["bus"].map(bus_country).fillna("").astype(str).str.strip().str.upper()
    )
    gen["carrier"] = gen["carrier"].astype(str)
    gen["p_nom"] = pd.to_numeric(gen.get("p_nom", 0.0), errors="coerce").fillna(0.0)
    gen["p_nom_min"] = pd.to_numeric(gen.get("p_nom_min", 0.0), errors="coerce").fillna(0.0)
    if "p_nom_max" in gen.columns:
        gen["p_nom_max"] = pd.to_numeric(gen["p_nom_max"], errors="coerce")
    else:
        gen["p_nom_max"] = np.nan
    gen["p_nom_extendable"] = gen.get("p_nom_extendable", False).fillna(False).astype(bool)

    model_carrier_to_constraint = {}
    for constraint_name, carriers in (model_carriers_by_constraint or {}).items():
        if isinstance(carriers, str):
            carriers = [carriers]
        for c in carriers or []:
            model_carrier_to_constraint[str(c)] = str(constraint_name)
    gen["constraint_carrier"] = gen["carrier"].map(model_carrier_to_constraint)
    gen = gen.loc[
        gen["country"].str.match(r"^[A-Z]{2}$", na=False)
        & gen["constraint_carrier"].notna()
    ].copy()

    added = 0
    skipped_no_bus = 0
    skipped_no_model_carrier = 0
    skipped_already_feasible = 0
    for row in ref.itertuples(index=False):
        country = str(row.country).upper()
        constraint_carrier = str(row.carrier)
        target_mw = float(row.reference_mw)
        if target_mw < min_reference_mw:
            continue

        carriers = model_carriers_by_constraint.get(constraint_carrier, [])
        if isinstance(carriers, str):
            carriers = [carriers]
        carriers = [str(c) for c in carriers]
        if not carriers:
            skipped_no_model_carrier += 1
            continue

        group = gen.loc[
            gen["country"].eq(country)
            & gen["constraint_carrier"].eq(constraint_carrier)
        ].copy()
        committed = 0.0
        max_feasible = 0.0
        if not group.empty:
            committed = float(
                pd.concat([group["p_nom"], group["p_nom_min"]], axis=1).max(axis=1).sum()
            )
            max_vals = pd.to_numeric(group["p_nom_max"], errors="coerce")
            extendable = group["p_nom_extendable"].astype(bool)
            fixed_like = group.loc[~extendable, "p_nom"].sum()
            ext_max = np.inf if max_vals.loc[extendable].isna().any() else max_vals.loc[extendable].sum()
            max_feasible = float(fixed_like + ext_max)

        band_mw = max(absolute_tolerance_mw, tolerance * abs(target_mw))
        lower_total = max(target_mw - band_mw, 0.0)
        if max_feasible + 1e-6 >= lower_total:
            skipped_already_feasible += 1
            continue

        missing_mw = max(target_mw - committed, 0.0)
        if missing_mw <= 1e-6:
            skipped_already_feasible += 1
            continue

        carrier = carriers[0]
        preferred = group.loc[group["carrier"].eq(carrier)] if not group.empty else None
        bus = _pick_country_ac_bus(n, country, bus_country, preferred_gens=preferred)
        if bus is None:
            skipped_no_bus += 1
            logger.warning(
                "Skipping fixed 2025 IRENA %s generator for %s: no valid AC bus found.",
                constraint_carrier,
                country,
            )
            continue

        base_name = f"{bus} {carrier}-{current_year}-irena-fixed"
        gen_name = base_name
        suffix = 2
        while gen_name in n.generators.index:
            gen_name = f"{base_name}-{suffix}"
            suffix += 1

        if carrier not in n.carriers.index:
            n.add("Carrier", carrier)

        template = _generator_template_from_network(n, carrier)
        add_kwargs = {
            "bus": bus,
            "carrier": carrier,
            "p_nom": missing_mw,
            "p_nom_min": missing_mw,
            "p_nom_max": missing_mw,
            "p_nom_extendable": False,
            "marginal_cost": template["marginal_cost"],
            "capital_cost": template["capital_cost"],
            "efficiency": template["efficiency"],
            "build_year": current_year,
            "lifetime": template["lifetime"],
        }
        n.add("Generator", gen_name, **add_kwargs)
        profile = _fixed_generator_profile(
            n,
            gen_name=gen_name,
            bus=bus,
            carrier=carrier,
            current_year=current_year,
        )
        if profile is not None:
            n.generators_t.p_max_pu[gen_name] = profile.values

        added += 1
        logger.info(
            "Added fixed 2025 IRENA %s generator for %s on %s with p_nom=%.2f MW (target=%.2f MW, committed_before=%.2f MW, reference_year=%s).",
            constraint_carrier,
            country,
            bus,
            missing_mw,
            target_mw,
            committed,
            used_reference_year,
        )

    logger.info(
        "2025 IRENA fixed-generator materialization summary: added=%d, skipped_already_feasible=%d, skipped_no_bus=%d, skipped_no_model_carrier=%d",
        added,
        skipped_already_feasible,
        skipped_no_bus,
        skipped_no_model_carrier,
    )


def _pick_country_geothermal_fallback_bus(n, country, bus_country, geo_country_gens):
    if not geo_country_gens.empty:
        existing_bus = str(geo_country_gens.iloc[0]["bus"])
        if existing_bus in n.buses.index:
            return existing_bus

    bus_country = bus_country.astype(str).str.strip().str.upper()
    bus_carrier = (
        n.buses["carrier"].astype(str)
        if "carrier" in n.buses.columns
        else pd.Series("", index=n.buses.index)
    )
    ac_buses = n.buses.index[bus_country.eq(country) & bus_carrier.eq("AC")]
    if len(ac_buses) > 0:
        return str(pd.Index(ac_buses).sort_values()[0])

    any_country_bus = n.buses.index[bus_country.eq(country)]
    if len(any_country_bus) > 0:
        return str(pd.Index(any_country_bus).sort_values()[0])

    return None


def _country_metric_key(country, metric):
    return f"{country}{COUNTRY_METRIC_KEY_SEP}{metric}"


def _decode_country_metric_key(key):
    if COUNTRY_METRIC_KEY_SEP not in str(key):
        return None
    country, metric = str(key).split(COUNTRY_METRIC_KEY_SEP, 1)
    if not country or not metric:
        return None
    return country, metric


def _snapshot_weights_da(n, column="generators"):
    if column not in n.snapshot_weightings.columns:
        column = "generators"
    return xr.DataArray(
        n.snapshot_weightings[column].values,
        coords=[n.snapshots],
        dims=["snapshot"],
    )


def _add_grouped_country_metric_expressions(target, grouped):
    if grouped is None or getattr(grouped, "size", 0) == 0:
        return
    keys = grouped.indexes.get("country_metric", pd.Index([]))
    for key in keys:
        decoded = _decode_country_metric_key(key)
        if decoded is None:
            continue
        expr = grouped.sel(country_metric=key)
        if decoded in target:
            target[decoded] = target[decoded] + expr
        else:
            target[decoded] = expr


def _model_country_metric_energy_expressions(n, gen_bus_carrier="AC", link_bus_carrier="AC"):
    bus_country = _get_bus_country_for_clustering(n).astype(str).str.strip().str.upper()
    out = {}

    # Generator output by (country, metric)
    if not n.generators.empty:
        gen_bus = n.generators.bus.map(n.buses.carrier)
        gen_mask = gen_bus.fillna("").eq(gen_bus_carrier)
        gen = n.generators.loc[gen_mask].copy()
        if not gen.empty:
            gen["country"] = gen.bus.map(bus_country).fillna("").astype(str).str.upper()
            gen["metric"] = (
                gen.carrier.astype(str).str.lower().map(MODEL_CARRIER_TO_OWID_METRIC)
            )
            gen = gen.loc[gen.country.ne("") & gen.metric.notna()].copy()
            if not gen.empty:
                p_g_full = n.model["Generator-p"]
                gen_idx = p_g_full.indexes.get("Generator", pd.Index([]))
                gen_var_mask = gen_idx.isin(gen.index)
                if gen_var_mask.any():
                    p_g = p_g_full.isel(Generator=gen_var_mask)
                    filtered = gen_idx[gen_var_mask]
                    key = (
                        gen.loc[filtered, "country"]
                        + COUNTRY_METRIC_KEY_SEP
                        + gen.loc[filtered, "metric"]
                    )
                    key_da = xr.DataArray(
                        key.values,
                        coords=[filtered],
                        dims=["Generator"],
                        name="country_metric",
                    )
                    grouped = (p_g * _snapshot_weights_da(n, "generators")).sum(
                        "snapshot"
                    ).groupby(key_da).sum("Generator")
                    _add_grouped_country_metric_expressions(out, grouped)

    # Link AC output by (country, metric)
    if not n.links.empty:
        p_l_full = n.model["Link-p"]
        link_idx = p_l_full.indexes.get("Link", pd.Index([]))
        link_df = n.links.copy()
        w_gen = _snapshot_weights_da(n, "generators")
        for port in [1, 2, 3, 4]:
            bus_col = f"bus{port}"
            eff_col = "efficiency" if port == 1 else f"efficiency{port}"
            if bus_col not in link_df.columns or eff_col not in link_df.columns:
                continue
            buses = link_df[bus_col]
            mask = buses.notna() & buses.ne("") & buses.map(n.buses.carrier).eq(link_bus_carrier)
            if not mask.any():
                continue
            sub = link_df.loc[mask, [bus_col, eff_col, "carrier"]].copy()
            sub["country"] = buses.loc[mask].map(bus_country).fillna("").astype(str).str.upper()
            sub["metric"] = (
                sub["carrier"].astype(str).str.lower().map(MODEL_CARRIER_TO_OWID_METRIC)
            )
            sub = sub.loc[sub.country.ne("") & sub.metric.notna()].copy()
            if sub.empty:
                continue

            var_mask = link_idx.isin(sub.index)
            if not var_mask.any():
                continue
            p_l = p_l_full.isel(Link=var_mask)
            filtered = link_idx[var_mask]
            eta = xr.DataArray(
                pd.to_numeric(sub.loc[filtered, eff_col], errors="coerce")
                .fillna(1.0)
                .abs()
                .values,
                coords=[filtered],
                dims=["Link"],
            )
            key = (
                sub.loc[filtered, "country"]
                + COUNTRY_METRIC_KEY_SEP
                + sub.loc[filtered, "metric"]
            )
            key_da = xr.DataArray(
                key.values,
                coords=[filtered],
                dims=["Link"],
                name="country_metric",
            )
            grouped = (p_l * eta * w_gen).sum("snapshot").groupby(key_da).sum("Link")
            _add_grouped_country_metric_expressions(out, grouped)

    # StorageUnit dispatch by (country, metric)
    if not n.storage_units.empty:
        su_bus = n.storage_units.bus.map(n.buses.carrier)
        su_mask = su_bus.fillna("").eq(gen_bus_carrier)
        su = n.storage_units.loc[su_mask].copy()
        if not su.empty:
            su["country"] = su.bus.map(bus_country).fillna("").astype(str).str.upper()
            su["metric"] = (
                su.carrier.astype(str).str.lower().map(MODEL_STORAGE_CARRIER_TO_OWID_METRIC)
            )
            su = su.loc[su.country.ne("") & su.metric.notna()].copy()
            if not su.empty:
                p_su_full = n.model["StorageUnit-p_dispatch"]
                su_idx = p_su_full.indexes.get("StorageUnit", pd.Index([]))
                su_var_mask = su_idx.isin(su.index)
                if su_var_mask.any():
                    p_su = p_su_full.isel(StorageUnit=su_var_mask)
                    filtered = su_idx[su_var_mask]
                    key = (
                        su.loc[filtered, "country"]
                        + COUNTRY_METRIC_KEY_SEP
                        + su.loc[filtered, "metric"]
                    )
                    key_da = xr.DataArray(
                        key.values,
                        coords=[filtered],
                        dims=["StorageUnit"],
                        name="country_metric",
                    )
                    grouped = (p_su * _snapshot_weights_da(n, "stores")).sum(
                        "snapshot"
                    ).groupby(key_da).sum("StorageUnit")
                    _add_grouped_country_metric_expressions(out, grouped)

    # Derived aggregate metrics by country
    by_country = {}
    for (country, metric), expr in out.items():
        by_country.setdefault(country, {})[metric] = expr

    for country, metric_map in by_country.items():
        fossil_terms = [
            metric_map[m]
            for m in ["coal_electricity", "gas_electricity", "oil_electricity"]
            if m in metric_map
        ]
        if fossil_terms:
            fossil_expr = fossil_terms[0]
            for t in fossil_terms[1:]:
                fossil_expr = fossil_expr + t
            out[(country, "fossil_electricity")] = fossil_expr

        renewable_terms = [
            metric_map[m]
            for m in [
                "hydro_electricity",
                "solar_electricity",
                "wind_electricity",
                "biofuel_electricity",
                "other_renewable_electricity",
            ]
            if m in metric_map
        ]
        if renewable_terms:
            renew_expr = renewable_terms[0]
            for t in renewable_terms[1:]:
                renew_expr = renew_expr + t
            out[(country, "renewables_electricity")] = renew_expr

        low_carbon = None
        if renewable_terms:
            low_carbon = renew_expr
        if "nuclear_electricity" in metric_map:
            low_carbon = (
                metric_map["nuclear_electricity"]
                if low_carbon is None
                else low_carbon + metric_map["nuclear_electricity"]
            )

        if low_carbon is not None:
            out[(country, "low_carbon_electricity")] = low_carbon
            if fossil_terms:
                out[(country, "electricity_generation")] = fossil_expr + low_carbon
            else:
                out[(country, "electricity_generation")] = low_carbon
        elif fossil_terms:
            out[(country, "electricity_generation")] = fossil_expr

    return out


def _owid_country_metric_reference(owid_csv, year, metrics):
    usecols = {"year", "iso_code"} | set(metrics)
    if "other_renewable_electricity" in metrics:
        usecols.add("other_renewable_exc_biofuel_electricity")
        usecols.add("other_renewable_electricity")
    ember_col_map = {
        "Code": "iso_code",
        "Year": "year",
        "Coal": "coal_electricity",
        "Gas": "gas_electricity",
        "Oil": "oil_electricity",
        "Bioenergy": "biofuel_electricity",
        "Hydropower": "hydro_electricity",
        "Nuclear": "nuclear_electricity",
        "Solar": "solar_electricity",
        "Wind": "wind_electricity",
        "Other renewables": "other_renewable_electricity",
    }
    owid = pd.read_csv(owid_csv)
    owid = owid.rename(columns={c: ember_col_map.get(c, c) for c in owid.columns})
    if not {"year", "iso_code"}.issubset(owid.columns):
        logger.warning("OWID reference file %s lacks year/iso_code columns", owid_csv)
        return pd.DataFrame(columns=["country", "metric", "reference_twh"])
    keep_cols = [c for c in owid.columns if c in usecols or c in set(ember_col_map.values())]
    owid = owid.loc[:, keep_cols].copy()
    owid = owid.loc[owid["year"] == int(year)].copy()
    if owid.empty:
        return pd.DataFrame(columns=["country", "metric", "reference_twh"])

    owid["country"] = owid["iso_code"].apply(_safe_iso3_to_iso2)
    owid = owid.loc[owid["country"].notna()].copy()
    if owid.empty:
        return pd.DataFrame(columns=["country", "metric", "reference_twh"])

    if (
        "other_renewable_exc_biofuel_electricity" in owid.columns
        and "other_renewable_electricity" in metrics
    ):
        owid["other_renewable_electricity"] = pd.to_numeric(
            owid["other_renewable_exc_biofuel_electricity"], errors="coerce"
        )

    metric_cols = [m for m in metrics if m in owid.columns]
    for col in metric_cols:
        owid[col] = pd.to_numeric(owid[col], errors="coerce")

    aggregate_defs = {
        "fossil_electricity": ["coal_electricity", "gas_electricity", "oil_electricity"],
        "renewables_electricity": [
            "hydro_electricity",
            "solar_electricity",
            "wind_electricity",
            "biofuel_electricity",
            "other_renewable_electricity",
        ],
        "low_carbon_electricity": [
            "hydro_electricity",
            "solar_electricity",
            "wind_electricity",
            "biofuel_electricity",
            "other_renewable_electricity",
            "nuclear_electricity",
        ],
        "electricity_generation": [
            "coal_electricity",
            "gas_electricity",
            "oil_electricity",
            "hydro_electricity",
            "solar_electricity",
            "wind_electricity",
            "biofuel_electricity",
            "other_renewable_electricity",
            "nuclear_electricity",
        ],
    }
    for metric, parts in aggregate_defs.items():
        if metric in metrics and metric not in owid.columns:
            available = [p for p in parts if p in owid.columns]
            if available:
                owid[metric] = owid[available].sum(axis=1, min_count=1)

    metric_cols = [m for m in metrics if m in owid.columns]
    grouped = owid.groupby("country", as_index=False)[metric_cols].sum(min_count=1)
    ref = grouped.melt(
        id_vars="country",
        value_vars=metric_cols,
        var_name="metric",
        value_name="reference_twh",
    )
    ref["reference_twh"] = pd.to_numeric(ref["reference_twh"], errors="coerce").fillna(0.0)
    return ref


def _irena_country_capacity_reference(irena_csv, year, carrier_technology_map, fallback_to_latest=True):
    irena = pd.read_csv(irena_csv)
    irena = irena.rename(
        columns={
            col: str(col).replace("\ufeff", "").strip()
            for col in irena.columns
        }
    )

    tech_to_carrier = {}
    for carrier, techs in (carrier_technology_map or {}).items():
        if isinstance(techs, str):
            techs = [techs]
        for tech in techs or []:
            tech_to_carrier[str(tech)] = str(carrier)
    if not tech_to_carrier:
        return pd.DataFrame(columns=["country", "carrier", "reference_mw"]), int(year)

    requested_year = int(year)

    # Wide legacy format: one column per year (e.g. irena_capacity_by_technology.csv)
    wide_years = sorted(int(c) for c in irena.columns if isinstance(c, str) and c.isdigit())
    if {"Technology", "Country"}.issubset(irena.columns) and wide_years:
        chosen_year = requested_year
        if str(chosen_year) not in irena.columns:
            if not fallback_to_latest:
                raise ValueError(
                    f"Year column '{requested_year}' not found in {irena_csv}. "
                    f"Available years: {wide_years[0]}-{wide_years[-1]}"
                )
            earlier_or_equal = [y for y in wide_years if y <= requested_year]
            chosen_year = max(earlier_or_equal) if earlier_or_equal else wide_years[-1]
            logger.warning(
                "IRENA capacity reference year %s not found in %s; using %s instead.",
                requested_year,
                irena_csv,
                chosen_year,
            )

        year_col = str(chosen_year)
        ref = irena[["Technology", "Country", year_col]].copy()
        ref["Technology"] = ref["Technology"].astype(str).str.strip()
        ref["country"] = ref["Country"].astype(str).str.strip().str.upper()
        ref["carrier"] = ref["Technology"].map(tech_to_carrier)
        ref["reference_mw"] = pd.to_numeric(ref[year_col], errors="coerce").fillna(0.0)
        ref = ref.loc[
            ref["carrier"].notna()
            & ref["country"].str.match(r"^[A-Z]{2}$", na=False)
        ].copy()
        if ref.empty:
            return pd.DataFrame(columns=["country", "carrier", "reference_mw"]), chosen_year
        ref = (
            ref.groupby(["country", "carrier"], as_index=False)["reference_mw"]
            .sum(min_count=1)
            .fillna({"reference_mw": 0.0})
        )
        ref["reference_mw"] = pd.to_numeric(ref["reference_mw"], errors="coerce").fillna(0.0).astype(float)
        return ref, chosen_year

    # New IRENA Renewable Capacity Statistics long format.
    new_long_cols = {
        "Region/area (ISO 3)",
        "Year",
        "Data Type",
        "Product Name",
        "Grid Type",
        "Unit",
        "Value",
    }
    if new_long_cols.issubset(irena.columns):
        irena["Year"] = pd.to_numeric(irena["Year"], errors="coerce")
        available_years = sorted(int(y) for y in irena["Year"].dropna().unique())
        if not available_years:
            raise ValueError(f"No valid `Year` rows found in IRENA capacity file: {irena_csv}")

        chosen_year = requested_year
        if chosen_year not in available_years:
            if not fallback_to_latest:
                raise ValueError(
                    f"Year row '{requested_year}' not found in {irena_csv}. "
                    f"Available years: {available_years[0]}-{available_years[-1]}"
                )
            earlier_or_equal = [y for y in available_years if y <= requested_year]
            chosen_year = max(earlier_or_equal) if earlier_or_equal else available_years[-1]
            logger.warning(
                "IRENA capacity reference year %s not found in %s; using %s instead.",
                requested_year,
                irena_csv,
                chosen_year,
            )

        ref = irena.loc[irena["Year"].eq(chosen_year)].copy()
        ref["Data Type"] = ref["Data Type"].astype(str).str.strip()
        ref["Product Name"] = ref["Product Name"].astype(str).str.strip()
        ref["Grid Type"] = ref["Grid Type"].astype(str).str.strip()
        ref["Unit"] = ref["Unit"].astype(str).str.strip()
        ref = ref.loc[
            ref["Data Type"].eq("Electrical Capacity")
            & ref["Grid Type"].eq("OnGrid")
            & ref["Unit"].eq("Megawatt")
        ].copy()
        ref["carrier"] = ref["Product Name"].map(tech_to_carrier)
        ref = ref.loc[ref["carrier"].notna()].copy()
        iso3_codes = ref["Region/area (ISO 3)"].astype(str).str.strip().unique()
        iso3_map = {code: _safe_iso3_to_iso2(code) for code in iso3_codes}
        ref["country"] = ref["Region/area (ISO 3)"].astype(str).str.strip().map(iso3_map)
        ref["reference_mw"] = pd.to_numeric(
            ref["Value"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
            .replace({"-": np.nan, "": np.nan}),
            errors="coerce",
        ).fillna(0.0)
        ref = ref.loc[
            ref["carrier"].notna() & ref["country"].notna()
        ].copy()
        if ref.empty:
            return pd.DataFrame(columns=["country", "carrier", "reference_mw"]), chosen_year
        ref["country"] = ref["country"].astype(str).str.upper()
        ref = (
            ref.groupby(["country", "carrier"], as_index=False)["reference_mw"]
            .sum(min_count=1)
            .fillna({"reference_mw": 0.0})
        )
        ref["reference_mw"] = pd.to_numeric(ref["reference_mw"], errors="coerce").fillna(0.0).astype(float)
        return ref, chosen_year

    # Long row-wise format: one `Year` column + installed capacity values
    required_long_cols = {"Technology", "Year", "Electricity Installed Capacity (MW)"}
    if required_long_cols.issubset(irena.columns) and (
        "ISO3 code" in irena.columns or "Country" in irena.columns
    ):
        irena["Year"] = pd.to_numeric(irena["Year"], errors="coerce")
        available_years = sorted(int(y) for y in irena["Year"].dropna().unique())
        if not available_years:
            raise ValueError(f"No valid `Year` rows found in IRENA capacity file: {irena_csv}")

        chosen_year = requested_year
        if chosen_year not in available_years:
            if not fallback_to_latest:
                raise ValueError(
                    f"Year row '{requested_year}' not found in {irena_csv}. "
                    f"Available years: {available_years[0]}-{available_years[-1]}"
                )
            earlier_or_equal = [y for y in available_years if y <= requested_year]
            chosen_year = max(earlier_or_equal) if earlier_or_equal else available_years[-1]
            logger.warning(
                "IRENA capacity reference year %s not found in %s; using %s instead.",
                requested_year,
                irena_csv,
                chosen_year,
            )

        ref = irena.loc[irena["Year"].eq(chosen_year)].copy()
        ref["Technology"] = ref["Technology"].astype(str).str.strip()
        ref["carrier"] = ref["Technology"].map(tech_to_carrier)
        ref = ref.loc[ref["carrier"].notna()].copy()
        if "ISO3 code" in ref.columns:
            iso3_codes = ref["ISO3 code"].astype(str).str.strip().unique()
            iso3_map = {code: _safe_iso3_to_iso2(code) for code in iso3_codes}
            ref["country"] = ref["ISO3 code"].astype(str).str.strip().map(iso3_map)
        else:
            ref["country"] = ref["Country"].astype(str).str.strip().str.upper()
            ref["country"] = ref["country"].where(
                ref["country"].str.match(r"^[A-Z]{2}$", na=False),
                np.nan,
            )
        ref["reference_mw"] = pd.to_numeric(
            ref["Electricity Installed Capacity (MW)"], errors="coerce"
        ).fillna(0.0)
        ref = ref.loc[
            ref["carrier"].notna() & ref["country"].notna()
        ].copy()
        if ref.empty:
            return pd.DataFrame(columns=["country", "carrier", "reference_mw"]), chosen_year
        ref["country"] = ref["country"].astype(str).str.upper()
        ref = (
            ref.groupby(["country", "carrier"], as_index=False)["reference_mw"]
            .sum(min_count=1)
            .fillna({"reference_mw": 0.0})
        )
        ref["reference_mw"] = pd.to_numeric(ref["reference_mw"], errors="coerce").fillna(0.0).astype(float)
        return ref, chosen_year

    raise ValueError(
        f"Unsupported IRENA capacity file schema in {irena_csv}. "
        "Expected wide year columns or long format with `Year` and "
        "`Electricity Installed Capacity (MW)`."
    )


def _irena_reference_year(available_years, requested_year, irena_csv, fallback_to_latest=True):
    available_years = sorted(int(y) for y in available_years)
    if not available_years:
        raise ValueError(f"No valid `Year` rows found in IRENA capacity file: {irena_csv}")

    requested_year = int(requested_year)
    if requested_year in available_years:
        return requested_year

    if not fallback_to_latest:
        raise ValueError(
            f"Year row '{requested_year}' not found in {irena_csv}. "
            f"Available years: {available_years[0]}-{available_years[-1]}"
        )

    earlier_or_equal = [y for y in available_years if y <= requested_year]
    chosen_year = max(earlier_or_equal) if earlier_or_equal else available_years[-1]
    logger.warning(
        "IRENA capacity reference year %s not found in %s; using %s instead.",
        requested_year,
        irena_csv,
        chosen_year,
    )
    return chosen_year


def _irena_value_to_mw(value, unit):
    parsed = pd.to_numeric(
        pd.Series(value)
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"-": np.nan, "": np.nan}),
        errors="coerce",
    ).fillna(0.0)
    normalized_unit = pd.Series(unit).astype(str).str.strip().str.lower()
    multipliers = normalized_unit.map(
        {
            "w": 1.0e-6,
            "watt": 1.0e-6,
            "watts": 1.0e-6,
            "kw": 1.0e-3,
            "kilowatt": 1.0e-3,
            "kilowatts": 1.0e-3,
            "mw": 1.0,
            "megawatt": 1.0,
            "megawatts": 1.0,
            "gw": 1.0e3,
            "gigawatt": 1.0e3,
            "gigawatts": 1.0e3,
        }
    )
    unknown_units = sorted(set(normalized_unit.loc[multipliers.isna()].dropna()))
    if unknown_units:
        raise ValueError(f"Unsupported IRENA capacity units: {unknown_units}")
    return (parsed * multipliers.fillna(0.0)).astype(float)


def _global_markers(series):
    values = series.astype(str).str.strip().str.upper()
    return values.isin({"GLO", "GLOBAL", "WORLD", "WRLD", "WLD"})


def _capacity_reference_from_country_sum(irena_csv, year, carrier_technology_map, fallback_to_latest=True):
    ref, chosen_year = _irena_country_capacity_reference(
        irena_csv=irena_csv,
        year=year,
        carrier_technology_map=carrier_technology_map,
        fallback_to_latest=fallback_to_latest,
    )
    if ref.empty:
        return {}, chosen_year
    grouped = ref.groupby("carrier")["reference_mw"].sum(min_count=1)
    return {
        str(carrier): float(reference_mw)
        for carrier, reference_mw in grouped.dropna().items()
    }, chosen_year


def _irena_global_capacity_reference(
    irena_csv,
    year,
    carrier_technology_map,
    fallback_to_latest=True,
    grid_types=None,
):
    irena = pd.read_csv(irena_csv)
    irena = irena.rename(
        columns={
            col: str(col).replace("\ufeff", "").strip()
            for col in irena.columns
        }
    )

    tech_to_carrier = {}
    for carrier, techs in (carrier_technology_map or {}).items():
        if isinstance(techs, str):
            techs = [techs]
        for tech in techs or []:
            tech_to_carrier[str(tech)] = str(carrier)
    if not tech_to_carrier:
        return {}, int(year)

    requested_year = int(year)

    # Wide legacy format: one column per year.
    wide_years = sorted(int(c) for c in irena.columns if isinstance(c, str) and c.isdigit())
    if {"Technology", "Country"}.issubset(irena.columns) and wide_years:
        chosen_year = _irena_reference_year(
            wide_years,
            requested_year,
            irena_csv,
            fallback_to_latest=fallback_to_latest,
        )
        year_col = str(chosen_year)
        ref = irena[["Technology", "Country", year_col]].copy()
        global_rows = _global_markers(ref["Country"])
        if global_rows.any():
            ref = ref.loc[global_rows].copy()
        ref["Technology"] = ref["Technology"].astype(str).str.strip()
        ref["carrier"] = ref["Technology"].map(tech_to_carrier)
        ref["reference_mw"] = pd.to_numeric(ref[year_col], errors="coerce").fillna(0.0)
        ref = ref.loc[ref["carrier"].notna()].copy()
        if ref.empty:
            return {}, chosen_year
        grouped = ref.groupby("carrier")["reference_mw"].sum(min_count=1)
        return {
            str(carrier): float(reference_mw)
            for carrier, reference_mw in grouped.dropna().items()
        }, chosen_year

    # New IRENA Renewable Capacity Statistics long format.
    new_long_cols = {
        "Region/area (ISO 3)",
        "Year",
        "Data Type",
        "Product Name",
        "Grid Type",
        "Unit",
        "Value",
    }
    if new_long_cols.issubset(irena.columns):
        irena["Year"] = pd.to_numeric(irena["Year"], errors="coerce")
        available_years = sorted(int(y) for y in irena["Year"].dropna().unique())
        chosen_year = _irena_reference_year(
            available_years,
            requested_year,
            irena_csv,
            fallback_to_latest=fallback_to_latest,
        )
        ref = irena.loc[irena["Year"].eq(chosen_year)].copy()
        ref["Data Type"] = ref["Data Type"].astype(str).str.strip()
        ref["Product Name"] = ref["Product Name"].astype(str).str.strip()
        ref["Grid Type"] = ref["Grid Type"].astype(str).str.strip()
        ref = ref.loc[
            ref["Data Type"].eq("Electrical Capacity")
            & _global_markers(ref["Region/area (ISO 3)"])
        ].copy()
        if grid_types is not None:
            if isinstance(grid_types, str):
                grid_types = [grid_types]
            allowed_grid_types = {str(grid_type).strip() for grid_type in grid_types}
            ref = ref.loc[ref["Grid Type"].isin(allowed_grid_types)].copy()
        ref["carrier"] = ref["Product Name"].map(tech_to_carrier)
        ref = ref.loc[ref["carrier"].notna()].copy()
        if ref.empty:
            return _capacity_reference_from_country_sum(
                irena_csv,
                chosen_year,
                carrier_technology_map,
                fallback_to_latest=True,
            )
        ref["reference_mw"] = _irena_value_to_mw(ref["Value"], ref["Unit"])
        grouped = ref.groupby("carrier")["reference_mw"].sum(min_count=1)
        return {
            str(carrier): float(reference_mw)
            for carrier, reference_mw in grouped.dropna().items()
        }, chosen_year

    # Long row-wise format without explicit global rows: sum country references.
    required_long_cols = {"Technology", "Year", "Electricity Installed Capacity (MW)"}
    if required_long_cols.issubset(irena.columns):
        return _capacity_reference_from_country_sum(
            irena_csv,
            requested_year,
            carrier_technology_map,
            fallback_to_latest=fallback_to_latest,
        )

    raise ValueError(
        f"Unsupported IRENA capacity file schema in {irena_csv}. "
        "Expected wide year columns or long format with `Year` and "
        "`Electricity Installed Capacity (MW)`."
    )


def _country_ac_load_twh(n):
    if n.loads.empty or "p_set" not in n.loads_t:
        return pd.Series(dtype=float)
    load_bus_carrier = n.loads.bus.map(n.buses.carrier).fillna("")
    ac_loads = n.loads.index[load_bus_carrier.eq("AC")]
    if len(ac_loads) == 0:
        return pd.Series(dtype=float)

    bus_country = _get_bus_country_for_clustering(n).astype(str).str.strip().str.upper()
    load_country = n.loads.loc[ac_loads, "bus"].map(bus_country).fillna("")
    valid = load_country.ne("")
    ac_loads = ac_loads[valid.values]
    load_country = load_country.loc[valid]
    if len(ac_loads) == 0:
        return pd.Series(dtype=float)

    load_ts = n.loads_t.p_set.reindex(columns=ac_loads).fillna(0.0)
    weighted = load_ts.mul(n.snapshot_weightings["generators"], axis=0).sum(axis=0)
    by_country_mwh = weighted.groupby(load_country).sum()
    return by_country_mwh / 1e6


def _generator_output_energy_by_buscarrier(n, bus_carrier="AC"):
    """
    Annual generator output (MWh) grouped by GENERATOR carrier, but only from
    generators whose bus has carrier == bus_carrier (e.g. "AC").
    """
    # pick only generators connected to the requested bus carrier
    ac_gen_i = n.generators.index[
        n.generators.bus.map(n.buses.carrier).fillna("").eq(bus_carrier)
    ]
    if len(ac_gen_i) == 0:
        return xr.DataArray([], dims=["carrier"])

    # Use integer-based indexing for temporal clustering compatibility
    p_g_full = n.model["Generator-p"]
    gen_idx = p_g_full.indexes.get("Generator", pd.Index([]))
    gen_mask = gen_idx.isin(ac_gen_i)
    p_g = p_g_full.isel(Generator=gen_mask)                            # [snapshot, Generator]
    
    # Get the actual generator indices after filtering
    filtered_gen_i = gen_idx[gen_mask]
    
    # snapshot_weightings.generators is a Series (column from DataFrame)
    # Explicitly provide coordinates to avoid timestamp/int comparison warnings
    w = xr.DataArray(
        n.snapshot_weightings.generators.values,
        coords=[n.snapshots],
        dims=["snapshot"]
    )
    g_car = n.generators.loc[filtered_gen_i, "carrier"].rename_axis("Generator").to_xarray()

    # MWh by generator carrier
    return (p_g * w).sum("snapshot").groupby(g_car).sum("Generator")


def _link_output_energy_by_buscarrier(n, bus_carrier="AC"):
    """
    Annual link *output-side* energy (MWh) grouped by LINK carrier, restricted
    to links whose OUTPUT bus (bus1) sits on a bus with carrier == bus_carrier.
    Energy at the output side is p[t,link] * efficiency[link].
    """
    if n.links.empty:
        return xr.DataArray([], dims=["carrier"])

    out_bus_carrier = n.links.bus1.map(n.buses.carrier).fillna("")
    link_i = n.links.index[out_bus_carrier.eq(bus_carrier)]
    if len(link_i) == 0:
        return xr.DataArray([], dims=["carrier"])

    # Use integer-based indexing for temporal clustering compatibility
    p_l_full = n.model["Link-p"]
    link_idx = p_l_full.indexes.get("Link", pd.Index([]))
    link_mask = link_idx.isin(link_i)
    p_l = p_l_full.isel(Link=link_mask)                                # [snapshot, Link]
    
    # Get the actual link indices after filtering
    filtered_link_i = link_idx[link_mask]
    
    eta  = xr.DataArray(n.links.loc[filtered_link_i, "efficiency"].fillna(1.0),
                        coords=[filtered_link_i], dims=["Link"])
    # snapshot_weightings.generators is a Series (column from DataFrame)
    # Explicitly provide coordinates to avoid timestamp/int comparison warnings
    w = xr.DataArray(
        n.snapshot_weightings.generators.values,
        coords=[n.snapshots],
        dims=["snapshot"]
    )
    lcar = n.links.loc[filtered_link_i, "carrier"].rename_axis("Link").to_xarray()

    # MWh by link carrier at AC output
    return (p_l * eta * w).sum("snapshot").groupby(lcar).sum("Link")

def _storageunit_output_energy_by_buscarrier(n, bus_carrier="AC"):
    """
    Annual StorageUnit *output-side* energy (MWh) grouped by STORAGE UNIT carrier,
    restricted to storage units whose bus sits on a bus with carrier == bus_carrier.
    Output is the electric dispatch variable p_dispatch (already AC-side).
    Weighted with snapshot_weightings.stores (consistent with PyPSA stats).
    """
    if n.storage_units.empty:
        return xr.DataArray([], dims=["carrier"])

    ac_su_i = n.storage_units.index[
        n.storage_units.bus.map(n.buses.carrier).fillna("").eq(bus_carrier)
    ]
    if len(ac_su_i) == 0:
        return xr.DataArray([], dims=["carrier"])

    # Use integer-based indexing for temporal clustering compatibility
    p_su_full = n.model["StorageUnit-p_dispatch"]
    su_idx = p_su_full.indexes.get("StorageUnit", pd.Index([]))
    su_mask = su_idx.isin(ac_su_i)
    p_su = p_su_full.isel(StorageUnit=su_mask)                         # [snapshot, StorageUnit]
    
    # Get the actual storage unit indices after filtering
    filtered_su_i = su_idx[su_mask]
    
    # snapshot_weightings.stores is a Series (column from DataFrame)
    # Explicitly provide coordinates to avoid timestamp/int comparison warnings
    w = xr.DataArray(
        n.snapshot_weightings.stores.values,
        coords=[n.snapshots],
        dims=["snapshot"]
    )
    su_car = n.storage_units.loc[filtered_su_i, "carrier"].rename_axis("StorageUnit").to_xarray()

    # MWh by storage-unit carrier at AC output
    return (p_su * w).sum("snapshot").groupby(su_car).sum("StorageUnit")


def _add_owid_country_generation_band(n, planning_year, config, cfg_key, enabled_key, label):
    global_cfg = config.get("global_specific", {})
    cfg = global_cfg.get(cfg_key, {})
    if not cfg or not cfg.get(enabled_key, False):
        return

    target_year = int(cfg.get("year", 2020))
    try:
        current_year = int(float(planning_year))
    except Exception:
        logger.warning("Could not parse planning year '%s' for %s constraints", planning_year, label)
        return
    if current_year != target_year:
        logger.info(
            "Skipping %s generation constraints for %s (configured for %s)",
            label,
            planning_year,
            target_year,
        )
        return

    logger.info("Adding %s OWID country-level generation constraints for %s", label, planning_year)

    tolerance = float(cfg.get("tolerance", 0.10))
    absolute_tolerance_twh = float(cfg.get("absolute_tolerance_twh", 0.5))
    min_reference_twh = float(cfg.get("min_reference_twh", 1.0))
    min_country_electric_load_twh = float(cfg.get("min_country_electric_load_twh", 0.0))
    slack_penalty_eur_per_mwh = float(cfg.get("slack_penalty_eur_per_mwh", 0.0))
    link_bus_carrier = cfg.get("link_bus_carrier", "AC")
    gen_bus_carrier = cfg.get("gen_bus_carrier", "AC")
    owid_csv = _repo_path(cfg.get("owid_csv", "validation/data/owid-energy-data.csv"))
    metrics = cfg.get("metrics", OWID_BASE_METRICS + OWID_AGGREGATE_METRICS)

    model_expr = _model_country_metric_energy_expressions(
        n,
        gen_bus_carrier=gen_bus_carrier,
        link_bus_carrier=link_bus_carrier,
    )
    if not model_expr:
        logger.warning("No model country-metric expressions found for %s generation constraints", label)
        return

    ref = _owid_country_metric_reference(owid_csv, target_year, metrics)
    if ref.empty:
        logger.warning("No OWID references found in %s for year %s", owid_csv, target_year)
        return

    network_countries = {
        c
        for c in _get_bus_country_for_clustering(n).astype(str).str.upper().unique()
        if c and re.match(r"^[A-Z]{2}$", c)
    }
    ref = ref.loc[ref["country"].isin(network_countries)].copy()

    skipped_low_load = 0
    if min_country_electric_load_twh > 0.0:
        country_load_twh = _country_ac_load_twh(n)
        keep_countries = set(
            country_load_twh.loc[
                country_load_twh.ge(min_country_electric_load_twh)
            ].index.astype(str).str.upper()
        )
        skipped_low_load = int((~ref["country"].isin(keep_countries)).sum())
        ref = ref.loc[ref["country"].isin(keep_countries)].copy()

    added = 0
    skipped_low = 0
    skipped_missing = 0
    slack_terms = []
    for row in ref.itertuples(index=False):
        reference_twh = float(row.reference_twh)
        if reference_twh < min_reference_twh:
            skipped_low += 1
            continue

        key = (str(row.country).upper(), str(row.metric))
        lhs = model_expr.get(key)
        if lhs is None:
            skipped_missing += 1
            continue

        band_twh = max(absolute_tolerance_twh, tolerance * abs(reference_twh))
        lower = max(reference_twh - band_twh, 0.0) * 1e6
        upper = (reference_twh + band_twh) * 1e6

        country_token = _sanitize_constraint_token(key[0])
        metric_token = _sanitize_constraint_token(key[1])
        if slack_penalty_eur_per_mwh > 0.0:
            slack_low = n.model.add_variables(
                lower=0.0,
                name=f"{label}_owid_country_metric_min_slack__{country_token}__{metric_token}",
            )
            slack_high = n.model.add_variables(
                lower=0.0,
                name=f"{label}_owid_country_metric_max_slack__{country_token}__{metric_token}",
            )
            n.model.add_constraints(
                lhs + slack_low >= lower,
                name=f"{label}_owid_country_metric_min__{country_token}__{metric_token}",
            )
            n.model.add_constraints(
                lhs - slack_high <= upper,
                name=f"{label}_owid_country_metric_max__{country_token}__{metric_token}",
            )
            slack_terms.extend([slack_low, slack_high])
        else:
            n.model.add_constraints(
                lhs >= lower,
                name=f"{label}_owid_country_metric_min__{country_token}__{metric_token}",
            )
            n.model.add_constraints(
                lhs <= upper,
                name=f"{label}_owid_country_metric_max__{country_token}__{metric_token}",
            )
        added += 1

    if slack_penalty_eur_per_mwh > 0.0 and slack_terms:
        slack_sum = slack_terms[0]
        for term in slack_terms[1:]:
            slack_sum = slack_sum + term
        n.model.objective = n.model.objective + slack_penalty_eur_per_mwh * slack_sum

    logger.info(
        "Added %d country-metric OWID constraints (skipped_low_reference=%d, skipped_missing_metric=%d, skipped_low_country_load=%d, tolerance=%s%%, absolute_tolerance=%.3f TWh, min_country_electric_load=%.3f TWh, slack_penalty=%.2f EUR/MWh)",
        added,
        skipped_low,
        skipped_missing,
        skipped_low_load,
        100.0 * tolerance,
        absolute_tolerance_twh,
        min_country_electric_load_twh,
        slack_penalty_eur_per_mwh,
    )


def add_baseyear_generation_band(n, planning_year, config):
    _add_owid_country_generation_band(
        n,
        planning_year=planning_year,
        config=config,
        cfg_key="baseyear_generation",
        enabled_key="baseyear_generation_constraint",
        label="baseyear",
    )


def add_year2025_generation_band(n, planning_year, config):
    _add_owid_country_generation_band(
        n,
        planning_year=planning_year,
        config=config,
        cfg_key="year2025_generation",
        enabled_key="year2025_generation_constraint",
        label="year2025",
    )


def add_year2025_geothermal_extendable_fallback(n, planning_year, config):
    global_cfg = config.get("global_specific", {})
    cfg = global_cfg.get("year2025_capacity", {})
    if not cfg or not cfg.get("year2025_capacity_constraint", False):
        return
    if not bool(cfg.get("geothermal_extendable_fallback", False)):
        return

    try:
        current_year = int(planning_year)
    except Exception:
        logger.warning(
            "Could not parse planning year '%s' for geothermal extendable fallback",
            planning_year,
        )
        return

    target_year = int(cfg.get("year", 2025))
    if current_year != target_year:
        logger.info(
            "Skipping geothermal extendable fallback for %s (configured for %s)",
            planning_year,
            target_year,
        )
        return

    irena_csv = _repo_path(
        cfg.get("irena_csv", "validation/data/irena_capacity_by_technology.csv")
    )
    if not os.path.exists(irena_csv):
        logger.warning(
            "Geothermal extendable fallback skipped: file not found at %s",
            irena_csv,
        )
        return

    tolerance = float(cfg.get("tolerance", 0.10))
    if tolerance < 0.0:
        raise ValueError(
            f"global_specific.year2025_capacity.tolerance must be >= 0, got {tolerance}"
        )
    absolute_tolerance_mw = float(cfg.get("absolute_tolerance_mw", 0.0))
    min_reference_mw = float(cfg.get("min_reference_mw", 0.0))
    reference_year = int(cfg.get("reference_year", target_year))
    fallback_to_latest = bool(cfg.get("fallback_to_latest_available", True))

    constraint_technology_map = cfg.get(
        "irena_technology_by_constraint",
        cfg.get(
            "irena_technology_by_carrier",
            {
                "solar": ["PV"],
                "onwind": ["Onshore"],
            },
        ),
    )

    try:
        ref, used_reference_year = _irena_country_capacity_reference(
            irena_csv=irena_csv,
            year=reference_year,
            carrier_technology_map=constraint_technology_map,
            fallback_to_latest=fallback_to_latest,
        )
    except Exception as exc:
        logger.warning("Geothermal extendable fallback skipped: %s", exc)
        return

    ref = ref.loc[ref["carrier"].astype(str).eq("geothermal")].copy()
    if ref.empty:
        logger.info(
            "Geothermal extendable fallback skipped: no geothermal IRENA rows found for %s",
            target_year,
        )
        return

    template = _geothermal_template_from_network(n)
    if template is None:
        logger.warning(
            "Geothermal extendable fallback skipped: no geothermal generators available to derive standard parameters."
        )
        return

    gen = n.generators.copy()
    if gen.empty:
        return

    bus_country = _get_bus_country_for_clustering(n).astype(str).str.strip().str.upper()
    gen["country"] = (
        gen["bus"].map(bus_country).fillna("").astype(str).str.strip().str.upper()
    )
    gen["carrier"] = gen["carrier"].astype(str)
    gen["p_nom"] = pd.to_numeric(gen.get("p_nom", 0.0), errors="coerce").fillna(0.0)
    gen["p_nom_min"] = (
        pd.to_numeric(gen.get("p_nom_min", 0.0), errors="coerce").fillna(0.0)
    )
    gen["p_nom_extendable"] = (
        gen.get("p_nom_extendable", False).fillna(False).astype(bool)
    )
    geothermal = gen.loc[
        gen["carrier"].eq("geothermal")
        & gen["country"].str.match(r"^[A-Z]{2}$", na=False)
    ].copy()

    ext_groups = {
        country: pd.Index(d.index)
        for country, d in geothermal.loc[geothermal["p_nom_extendable"]].groupby("country")
    }
    fixed_capacity = (
        geothermal.loc[~geothermal["p_nom_extendable"]]
        .groupby("country")["p_nom"]
        .sum()
        if not geothermal.empty
        else pd.Series(dtype=float)
    )

    added = 0
    skipped_existing_extendable = 0
    skipped_no_bus = 0
    skipped_duplicate = 0

    for row in ref.itertuples(index=False):
        country = str(row.country).upper()
        target_mw = float(row.reference_mw)
        if target_mw < min_reference_mw:
            continue

        existing_capacity_mw = float(fixed_capacity.get(country, 0.0))
        ext_idx = ext_groups.get(country, pd.Index([]))
        if len(ext_idx) > 0:
            skipped_existing_extendable += 1
            continue

        band_mw = max(absolute_tolerance_mw, tolerance * abs(target_mw))
        lower_total = max(target_mw - band_mw, 0.0)
        upper_total = target_mw + band_mw

        if existing_capacity_mw + 1e-6 >= lower_total:
            continue

        additional_headroom_mw = max(upper_total - existing_capacity_mw, 0.0)
        if additional_headroom_mw <= 1e-6:
            continue

        geo_country_gens = geothermal.loc[geothermal["country"].eq(country)]
        bus = _pick_country_geothermal_fallback_bus(
            n,
            country=country,
            bus_country=bus_country,
            geo_country_gens=geo_country_gens,
        )
        if bus is None:
            skipped_no_bus += 1
            logger.warning(
                "Geothermal extendable fallback skipped for %s: no valid bus found.",
                country,
            )
            continue

        base_name = f"{bus} geothermal-{current_year}-irena-fallback"
        gen_name = base_name
        suffix = 2
        while gen_name in n.generators.index:
            existing = n.generators.loc[gen_name]
            if (
                str(existing.get("carrier", "")) == "geothermal"
                and str(existing.get("bus", "")) == bus
                and bool(existing.get("p_nom_extendable", False))
            ):
                skipped_duplicate += 1
                gen_name = None
                break
            gen_name = f"{base_name}-{suffix}"
            suffix += 1
        if gen_name is None:
            continue

        add_kwargs = {
            "bus": bus,
            "carrier": "geothermal",
            "p_nom": 0.0,
            "p_nom_min": 0.0,
            "p_nom_max": additional_headroom_mw,
            "p_nom_extendable": True,
            "marginal_cost": template["marginal_cost"],
            "capital_cost": template["capital_cost"],
            "efficiency": template["efficiency"],
            "build_year": current_year,
            "lifetime": template["lifetime"],
        }
        if "p_max_pu" in n.generators.columns:
            add_kwargs["p_max_pu"] = template["p_max_pu"]

        n.add("Generator", gen_name, **add_kwargs)
        added += 1
        ext_groups[country] = ext_groups.get(country, pd.Index([])).append(
            pd.Index([gen_name])
        )

        logger.info(
            "Added geothermal extendable fallback generator for %s on bus %s with p_nom_max=%.2f MW to satisfy the %s country-capacity band (existing_fixed=%.2f MW, lower=%.2f MW, upper=%.2f MW).",
            country,
            bus,
            additional_headroom_mw,
            used_reference_year,
            existing_capacity_mw,
            lower_total,
            upper_total,
        )

    logger.info(
        "Geothermal extendable fallback summary for %s: added=%d, skipped_existing_extendable=%d, skipped_no_bus=%d, skipped_duplicate=%d",
        planning_year,
        added,
        skipped_existing_extendable,
        skipped_no_bus,
        skipped_duplicate,
    )


def add_year2025_irena_country_capacity_band(n, planning_year, config):
    """
    Add hard country-level renewable capacity constraints around IRENA 2025 references.

    The config block is `global_specific.year2025_capacity`. Constraints apply only when
    `planning_year` matches `cfg["year"]`. Total capacity is generator `p_nom` by
    (country, configured constraint group), including fixed and extendable assets.
    The band is symmetric around the IRENA reference: reference +/- tolerance.
    """
    global_cfg = config.get("global_specific", {})
    cfg = global_cfg.get("year2025_capacity", {})
    if not cfg or not cfg.get("year2025_capacity_constraint", False):
        return

    target_year = int(cfg.get("year", 2025))
    try:
        current_year = int(float(planning_year))
    except Exception:
        logger.warning(
            "Could not parse planning year '%s' for year2025 capacity constraints",
            planning_year,
        )
        return

    if current_year != target_year:
        logger.info(
            "Skipping 2025 country-capacity constraints for %s (configured for %s)",
            planning_year,
            target_year,
        )
        return

    irena_csv = _repo_path(cfg.get("irena_csv", "validation/data/irena_capacity_by_technology.csv"))
    if not os.path.exists(irena_csv):
        logger.warning(
            "IRENA country-capacity constraint skipped: file not found at %s",
            irena_csv,
        )
        return

    tolerance = float(cfg.get("tolerance", 0.10))
    if tolerance < 0.0:
        raise ValueError(
            f"global_specific.year2025_capacity.tolerance must be >= 0, got {tolerance}"
        )
    absolute_tolerance_mw = float(cfg.get("absolute_tolerance_mw", 0.0))
    min_reference_mw = float(cfg.get("min_reference_mw", 0.0))
    fixed_capacity_overshoot_policy = str(
        cfg.get("fixed_capacity_overshoot_policy", "raise")
    ).strip().lower()
    valid_fixed_overshoot_policies = {
        "raise",
        "constrain_additions_only",
    }
    if fixed_capacity_overshoot_policy not in valid_fixed_overshoot_policies:
        raise ValueError(
            "global_specific.year2025_capacity.fixed_capacity_overshoot_policy "
            f"must be one of {sorted(valid_fixed_overshoot_policies)}, got "
            f"{fixed_capacity_overshoot_policy!r}"
        )
    reference_year = int(cfg.get("reference_year", target_year))
    fallback_to_latest = bool(cfg.get("fallback_to_latest_available", True))
    units = str(cfg.get("units", "GW")).lower()
    unit_scale = {"mw": 1.0, "gw": 1e3, "tw": 1e6}.get(units, 1e3)
    constraint_technology_map = cfg.get(
        "irena_technology_by_constraint",
        cfg.get(
            "irena_technology_by_carrier",
            {
                "solar": ["PV"],
                "onwind": ["Onshore"],
            },
        ),
    )
    model_carriers_by_constraint = cfg.get(
        "model_carriers_by_constraint",
        {
            key: [key] for key in (constraint_technology_map or {}).keys()
        },
    )

    try:
        ref, used_reference_year = _irena_country_capacity_reference(
            irena_csv=irena_csv,
            year=reference_year,
            carrier_technology_map=constraint_technology_map,
            fallback_to_latest=fallback_to_latest,
        )
    except Exception as exc:
        logger.warning("IRENA country-capacity constraint skipped: %s", exc)
        return

    if ref.empty:
        logger.warning(
            "IRENA country-capacity constraint skipped: no mapped rows found in %s",
            irena_csv,
        )
        return

    gen = n.generators.copy()
    if gen.empty:
        logger.warning("IRENA country-capacity constraint skipped: network has no generators")
        return

    bus_country = _get_bus_country_for_clustering(n).astype(str).str.strip().str.upper()
    gen["country"] = gen["bus"].map(bus_country).fillna("").astype(str).str.strip().str.upper()
    gen["carrier"] = gen["carrier"].astype(str)
    gen["p_nom"] = pd.to_numeric(gen.get("p_nom", 0.0), errors="coerce").fillna(0.0)
    gen["p_nom_min"] = pd.to_numeric(gen.get("p_nom_min", 0.0), errors="coerce").fillna(0.0)
    if "p_nom_max" in gen.columns:
        gen["p_nom_max"] = pd.to_numeric(gen["p_nom_max"], errors="coerce")
    else:
        gen["p_nom_max"] = np.nan
    gen["p_nom_extendable"] = gen.get("p_nom_extendable", False).fillna(False).astype(bool)
    model_carrier_to_constraint = {}
    for constraint_name, carriers in (model_carriers_by_constraint or {}).items():
        if isinstance(carriers, str):
            carriers = [carriers]
        for c in carriers or []:
            model_carrier_to_constraint[str(c)] = str(constraint_name)
    gen["constraint_carrier"] = gen["carrier"].map(model_carrier_to_constraint)
    gen = gen.loc[gen["country"].str.match(r"^[A-Z]{2}$", na=False)].copy()
    gen = gen.loc[gen["constraint_carrier"].notna()].copy()
    if gen.empty:
        logger.warning("IRENA country-capacity constraint skipped: no generators with country labels")
        return

    network_countries = set(gen["country"].unique())
    ref = ref.loc[ref["country"].isin(network_countries)].copy()
    if ref.empty:
        logger.warning(
            "IRENA country-capacity constraint skipped: no reference rows overlap network countries"
        )
        return

    p_nom_var = n.model["Generator-p_nom"]
    all_ext_gen = gen.loc[gen["p_nom_extendable"]].copy()
    fixed_gen = gen.loc[~gen["p_nom_extendable"]].copy()

    positive_headroom_ext = pd.Series(dtype=bool)
    degenerate_ext_gen = pd.DataFrame(columns=gen.columns)
    ext_gen = all_ext_gen
    if not all_ext_gen.empty:
        positive_headroom_ext = (
            all_ext_gen["p_nom_max"].isna()
            | all_ext_gen["p_nom_max"].gt(all_ext_gen["p_nom_min"] + 1e-9)
        )
        degenerate_ext_gen = all_ext_gen.loc[~positive_headroom_ext].copy()
        ext_gen = all_ext_gen.loc[positive_headroom_ext].copy()
        if not degenerate_ext_gen.empty:
            carrier_counts = (
                degenerate_ext_gen["constraint_carrier"].astype(str).value_counts().to_dict()
            )
            logger.info(
                "Treating %d zero-headroom extendable generators as fixed-equivalent capacity for 2025 IRENA bands (constraint_carrier_counts=%s).",
                len(degenerate_ext_gen),
                carrier_counts,
            )

    fixed_capacity = (
        fixed_gen.groupby(["country", "constraint_carrier"])["p_nom"].sum()
        if not fixed_gen.empty
        else pd.Series(dtype=float)
    )
    if not degenerate_ext_gen.empty:
        fixed_like_capacity = (
            pd.concat(
                [
                    degenerate_ext_gen["p_nom"],
                    degenerate_ext_gen["p_nom_min"],
                ],
                axis=1,
            )
            .max(axis=1)
            .groupby(
                [
                    degenerate_ext_gen["country"],
                    degenerate_ext_gen["constraint_carrier"],
                ]
            )
            .sum()
        )
        if fixed_capacity.empty:
            fixed_capacity = fixed_like_capacity.astype(float)
        else:
            fixed_capacity = fixed_capacity.add(
                fixed_like_capacity,
                fill_value=0.0,
            )
    ext_min_capacity = (
        ext_gen.groupby(["country", "constraint_carrier"])["p_nom_min"].sum()
        if not ext_gen.empty
        else pd.Series(dtype=float)
    )
    ext_max_capacity = pd.Series(dtype=float)
    if not ext_gen.empty:
        ext_max_rows = []
        for key, d in ext_gen.groupby(["country", "constraint_carrier"]):
            max_vals = pd.to_numeric(d["p_nom_max"], errors="coerce")
            total_max = np.inf if max_vals.isna().any() else float(max_vals.sum())
            ext_max_rows.append((key[0], key[1], total_max))
        if ext_max_rows:
            ext_max_capacity = (
                pd.DataFrame(
                    ext_max_rows,
                    columns=["country", "constraint_carrier", "ext_max_capacity_mw"],
                )
                .set_index(["country", "constraint_carrier"])["ext_max_capacity_mw"]
            )

    ext_groups = {}
    if not ext_gen.empty:
        for (country, carrier), d in ext_gen.groupby(["country", "constraint_carrier"]):
            ext_groups[(country, carrier)] = pd.Index(d.index)

    added = 0
    skipped_low_reference = 0
    skipped_no_variable_but_satisfied = 0
    missing_variable_violations = 0
    skipped_infeasible_by_upper_bound = 0
    skipped_infeasible_by_lower_bound = 0
    constrained_fixed_overshoot_additions = 0

    for row in ref.itertuples(index=False):
        country = str(row.country).upper()
        carrier = str(row.carrier)
        target_mw = float(row.reference_mw)

        if target_mw < min_reference_mw:
            skipped_low_reference += 1
            continue

        key = (country, carrier)
        existing_capacity_mw = float(fixed_capacity.get(key, 0.0))
        ext_idx = ext_groups.get(key, pd.Index([]))

        band_mw = max(absolute_tolerance_mw, tolerance * abs(target_mw))
        lower_total = max(target_mw - band_mw, 0.0)
        upper_total = target_mw + band_mw
        if len(ext_idx) == 0:
            if existing_capacity_mw + 1e-6 < lower_total or existing_capacity_mw - 1e-6 > upper_total:
                missing_variable_violations += 1
                if (
                    fixed_capacity_overshoot_policy == "constrain_additions_only"
                    and existing_capacity_mw - 1e-6 > upper_total
                ):
                    constrained_fixed_overshoot_additions += 1
                    logger.warning(
                        "Cannot satisfy hard IRENA country-capacity upper band for %s/%s because fixed capacity is already too high and no extendable generator exists: fixed %.2f %s > upper %.2f %s. No additional-build variable is available, so the row is treated as an exogenous retained-capacity overshoot.",
                        country,
                        carrier,
                        existing_capacity_mw / unit_scale,
                        units.upper(),
                        upper_total / unit_scale,
                        units.upper(),
                    )
                    continue
                raise ValueError(
                    "Cannot enforce hard IRENA country-capacity band for "
                    f"{country}/{carrier}: no extendable generators and fixed "
                    f"capacity {existing_capacity_mw / unit_scale:.2f} {units.upper()} "
                    f"is outside [{lower_total / unit_scale:.2f}, "
                    f"{upper_total / unit_scale:.2f}] {units.upper()}."
                )
            else:
                skipped_no_variable_but_satisfied += 1
            continue

        ext_min_mw = float(ext_min_capacity.get(key, 0.0))
        ext_max_mw = ext_max_capacity.get(key, np.inf)
        try:
            ext_max_mw = float(ext_max_mw)
        except Exception:
            ext_max_mw = np.inf

        min_total_feasible = existing_capacity_mw + ext_min_mw
        max_total_feasible = existing_capacity_mw + ext_max_mw

        if min_total_feasible - 1e-6 > upper_total:
            skipped_infeasible_by_upper_bound += 1
            if fixed_capacity_overshoot_policy == "constrain_additions_only":
                lhs = p_nom_var.loc[ext_idx].sum()
                country_token = _sanitize_constraint_token(country)
                carrier_token = _sanitize_constraint_token(carrier)
                n.model.add_constraints(
                    lhs <= ext_min_mw,
                    name=f"year2025_irena_country_capacity_no_additions__{country_token}__{carrier_token}",
                )
                constrained_fixed_overshoot_additions += 1
                logger.warning(
                    "Cannot satisfy hard IRENA country-capacity upper band for %s/%s because retained fixed capacity is already too high: minimum feasible %.2f %s > upper %.2f %s. Added a hard no-additional-build constraint for extendable capacity instead (fixed=%.2f, extendable p_nom_min=%.2f).",
                    country,
                    carrier,
                    min_total_feasible / unit_scale,
                    units.upper(),
                    upper_total / unit_scale,
                    units.upper(),
                    existing_capacity_mw / unit_scale,
                    ext_min_mw / unit_scale,
                )
                continue
            raise ValueError(
                "Cannot enforce hard IRENA country-capacity band for "
                f"{country}/{carrier}: minimum feasible total "
                f"{min_total_feasible / unit_scale:.2f} {units.upper()} exceeds "
                f"upper bound {upper_total / unit_scale:.2f} {units.upper()} "
                f"(fixed={existing_capacity_mw / unit_scale:.2f}, "
                f"extendable p_nom_min={ext_min_mw / unit_scale:.2f})."
            )

        if max_total_feasible + 1e-6 < lower_total:
            skipped_infeasible_by_lower_bound += 1
            raise ValueError(
                "Cannot enforce hard IRENA country-capacity band for "
                f"{country}/{carrier}: maximum feasible total "
                f"{max_total_feasible / unit_scale:.2f} {units.upper()} is below "
                f"lower bound {lower_total / unit_scale:.2f} {units.upper()} "
                f"(fixed={existing_capacity_mw / unit_scale:.2f}, "
                f"extendable p_nom_max="
                f"{'inf' if not np.isfinite(ext_max_mw) else f'{ext_max_mw / unit_scale:.2f}'})."
            )

        lhs = p_nom_var.loc[ext_idx].sum() + existing_capacity_mw
        country_token = _sanitize_constraint_token(country)
        carrier_token = _sanitize_constraint_token(carrier)

        n.model.add_constraints(
            lhs >= lower_total,
            name=f"year2025_irena_country_capacity_min__{country_token}__{carrier_token}",
        )
        n.model.add_constraints(
            lhs <= upper_total,
            name=f"year2025_irena_country_capacity_max__{country_token}__{carrier_token}",
        )
        added += 1

    logger.info(
        "Added %d hard IRENA country-capacity constraints for planning year %s (reference_year=%s, tolerance=%s%%, absolute_tolerance=%.2f MW, min_reference=%.2f MW, fixed_overshoot_policy=%s, fixed_overshoot_no_additions=%d, skipped_low_reference=%d, skipped_fixed_only_satisfied=%d, fixed_only_violations=%d, infeasible_upper=%d, infeasible_lower=%d)",
        added,
        planning_year,
        used_reference_year,
        100.0 * tolerance,
        absolute_tolerance_mw,
        min_reference_mw,
        fixed_capacity_overshoot_policy,
        constrained_fixed_overshoot_additions,
        skipped_low_reference,
        skipped_no_variable_but_satisfied,
        missing_variable_violations,
        skipped_infeasible_by_upper_bound,
        skipped_infeasible_by_lower_bound,
    )


def add_year2025_global_capacity_band(n, planning_year, config):
    """
    Add 2025 global installed-capacity band constraints for selected technologies.

    This is intended to complement (not replace) the country-level IRENA capacity
    band. Total global installed capacity is constrained around IRENA global
    references, or explicit configured targets when provided, with a symmetric
    tolerance.
    """
    global_cfg = config.get("global_specific", {})
    parent_cfg = global_cfg.get("year2025_capacity", {})
    if not parent_cfg or not parent_cfg.get("year2025_capacity_constraint", False):
        return

    cfg = (parent_cfg.get("global_capacity_band", {}) or {})
    if not cfg or not bool(cfg.get("enable", False)):
        return

    target_year = int(cfg.get("year", parent_cfg.get("year", 2025)))
    try:
        current_year = int(float(planning_year))
    except Exception:
        logger.warning(
            "Could not parse planning year '%s' for 2025 global capacity constraints",
            planning_year,
        )
        return

    if current_year != target_year:
        logger.info(
            "Skipping 2025 global capacity constraints for %s (configured for %s)",
            planning_year,
            target_year,
        )
        return

    tolerance = float(cfg.get("tolerance", 0.05))
    if tolerance < 0.0:
        raise ValueError(
            f"global_specific.year2025_capacity.global_capacity_band.tolerance must be >= 0, got {tolerance}"
        )

    units = str(cfg.get("units", "GW")).lower()
    unit_scale = {"mw": 1.0, "gw": 1e3, "tw": 1e6}.get(units, 1e3)

    manual_targets = cfg.get("targets", None)
    if manual_targets:
        target_mw_by_carrier = {}
        for carrier, target in manual_targets.items():
            if isinstance(target, str) and target.upper().startswith("X"):
                logger.info(
                    "Skipping %s global-capacity target (placeholder target '%s')",
                    carrier,
                    target,
                )
                continue
            target_mw_by_carrier[str(carrier)] = float(target) * unit_scale
        target_source = "configured targets"
        reference_year = target_year
    else:
        irena_csv = _repo_path(parent_cfg.get("irena_csv", "validation/data/irena_capacity_by_technology.csv"))
        reference_year = int(cfg.get("reference_year", parent_cfg.get("reference_year", target_year)))
        fallback_to_latest = bool(
            cfg.get(
                "fallback_to_latest_available",
                parent_cfg.get("fallback_to_latest_available", True),
            )
        )
        carrier_technology_map = cfg.get(
            "irena_technology_by_constraint",
            parent_cfg.get(
                "irena_technology_by_constraint",
                parent_cfg.get(
                    "irena_technology_by_carrier",
                    {
                        "solar": ["PV"],
                        "onwind": ["Onshore"],
                    },
                ),
            ),
        )
        target_carriers = cfg.get("target_carriers", None)
        if target_carriers is not None:
            if isinstance(target_carriers, str):
                target_carriers = [target_carriers]
            target_carriers = {str(carrier) for carrier in target_carriers}
            carrier_technology_map = {
                str(carrier): techs
                for carrier, techs in (carrier_technology_map or {}).items()
                if str(carrier) in target_carriers
            }
        target_mw_by_carrier, reference_year = _irena_global_capacity_reference(
            irena_csv=irena_csv,
            year=reference_year,
            carrier_technology_map=carrier_technology_map,
            fallback_to_latest=fallback_to_latest,
            grid_types=cfg.get("grid_types", None),
        )
        target_source = f"IRENA global reference ({irena_csv}, {reference_year})"

    if not target_mw_by_carrier:
        logger.warning("2025 global capacity constraints enabled but no targets could be resolved.")
        return

    model_carriers_by_constraint = parent_cfg.get(
        "model_carriers_by_constraint",
        {key: [key] for key in target_mw_by_carrier.keys()},
    )

    if n.generators.empty:
        logger.warning("2025 global capacity constraints skipped: network has no generators")
        return

    gen = n.generators.copy()
    gen["carrier"] = gen["carrier"].astype(str)
    gen["p_nom"] = pd.to_numeric(gen.get("p_nom", 0.0), errors="coerce").fillna(0.0)
    gen["p_nom_min"] = pd.to_numeric(gen.get("p_nom_min", 0.0), errors="coerce").fillna(0.0)
    if "p_nom_max" in gen.columns:
        gen["p_nom_max"] = pd.to_numeric(gen["p_nom_max"], errors="coerce")
    else:
        gen["p_nom_max"] = np.nan
    gen["p_nom_extendable"] = gen.get("p_nom_extendable", False).fillna(False).astype(bool)

    model_carrier_to_constraint = {}
    for constraint_name, carriers in (model_carriers_by_constraint or {}).items():
        if isinstance(carriers, str):
            carriers = [carriers]
        for c in carriers or []:
            model_carrier_to_constraint[str(c)] = str(constraint_name)

    gen["constraint_carrier"] = gen["carrier"].map(model_carrier_to_constraint)
    gen = gen.loc[gen["constraint_carrier"].notna()].copy()
    if gen.empty:
        logger.warning("2025 global capacity constraints skipped: no generators matched configured carrier mapping")
        return

    p_nom_var = n.model["Generator-p_nom"]

    all_ext_gen = gen.loc[gen["p_nom_extendable"]].copy()
    fixed_gen = gen.loc[~gen["p_nom_extendable"]].copy()

    degenerate_ext_gen = pd.DataFrame(columns=gen.columns)
    ext_gen = all_ext_gen
    if not all_ext_gen.empty:
        positive_headroom_ext = (
            all_ext_gen["p_nom_max"].isna()
            | all_ext_gen["p_nom_max"].gt(all_ext_gen["p_nom_min"] + 1e-9)
        )
        degenerate_ext_gen = all_ext_gen.loc[~positive_headroom_ext].copy()
        ext_gen = all_ext_gen.loc[positive_headroom_ext].copy()

    fixed_capacity = (
        fixed_gen.groupby("constraint_carrier")["p_nom"].sum()
        if not fixed_gen.empty
        else pd.Series(dtype=float)
    )

    if not degenerate_ext_gen.empty:
        fixed_like_capacity = (
            pd.concat(
                [degenerate_ext_gen["p_nom"], degenerate_ext_gen["p_nom_min"]],
                axis=1,
            )
            .max(axis=1)
            .groupby(degenerate_ext_gen["constraint_carrier"])
            .sum()
        )
        if fixed_capacity.empty:
            fixed_capacity = fixed_like_capacity.astype(float)
        else:
            fixed_capacity = fixed_capacity.add(fixed_like_capacity, fill_value=0.0)

    ext_groups = {}
    if not ext_gen.empty:
        for carrier, d in ext_gen.groupby("constraint_carrier"):
            ext_groups[str(carrier)] = pd.Index(d.index)

    added = 0
    skipped_no_variable_but_satisfied = 0

    for carrier, target_mw in target_mw_by_carrier.items():
        carrier = str(carrier)
        lower_total = target_mw * (1.0 - tolerance)
        upper_total = target_mw * (1.0 + tolerance)

        existing_capacity_mw = float(fixed_capacity.get(carrier, 0.0))
        ext_idx = ext_groups.get(carrier, pd.Index([]))

        if len(ext_idx) == 0:
            total_fixed = existing_capacity_mw
            if total_fixed + 1e-6 < lower_total or total_fixed - 1e-6 > upper_total:
                raise ValueError(
                    "Cannot enforce 2025 global capacity band for "
                    f"{carrier}: no extendable generators and fixed capacity "
                    f"{total_fixed / unit_scale:.2f} {units.upper()} is outside "
                    f"[{lower_total / unit_scale:.2f}, {upper_total / unit_scale:.2f}] {units.upper()}."
                )
            skipped_no_variable_but_satisfied += 1
            continue

        lhs = p_nom_var.loc[ext_idx].sum() + existing_capacity_mw
        carrier_token = _sanitize_constraint_token(carrier)

        n.model.add_constraints(
            lhs >= lower_total,
            name=f"year2025_global_capacity_min__{carrier_token}",
        )
        n.model.add_constraints(
            lhs <= upper_total,
            name=f"year2025_global_capacity_max__{carrier_token}",
        )
        added += 1

        logger.info(
            "Global %s capacity band for %s: %.2f ≤ total installed capacity ≤ %.2f %s "
            "(target: %.2f %s, fixed-equivalent: %.2f %s, source: %s)",
            carrier,
            planning_year,
            lower_total / unit_scale,
            upper_total / unit_scale,
            units.upper(),
            target_mw / unit_scale,
            units.upper(),
            existing_capacity_mw / unit_scale,
            units.upper(),
            target_source,
        )

    logger.info(
        "Added %d global 2025 capacity bands for planning year %s (tolerance=%.2f%%, target_source=%s, skipped_fixed_only_satisfied=%d)",
        added,
        planning_year,
        100.0 * tolerance,
        target_source,
        skipped_no_variable_but_satisfied,
    )


def add_year2025_irena_nodal_distribution_constraints(n, planning_year, config):
    """
    Limit 2025 renewable build concentration by adding linear nodal share constraints.

    The configured nodal_distribution_limit parameters are interpreted as caps on a bus'
    share of country-level *new build* for the configured carriers. The constraint is added
    on the live Generator-p_nom optimization variables instead of rewriting p_nom_max.
    """

    global_cfg = config.get("global_specific", {})
    cfg = global_cfg.get("year2025_capacity", {})
    if not cfg or not cfg.get("year2025_capacity_constraint", False):
        return

    cap_cfg = cfg.get("nodal_distribution_limit", {})
    if not cap_cfg or not bool(cap_cfg.get("enable", False)):
        return

    target_year = int(cfg.get("year", 2025))
    try:
        current_year = int(float(planning_year))
    except Exception:
        logger.warning(
            "Could not parse planning year '%s' for 2025 nodal distribution constraints",
            planning_year,
        )
        return

    if current_year != target_year:
        logger.info(
            "Skipping 2025 nodal distribution constraints for %s (configured for %s)",
            planning_year,
            target_year,
        )
        return

    carriers = [str(c) for c in cap_cfg.get("carriers", ["onwind", "solar"])]
    if not carriers:
        return

    max_baseyear_share_multiplier = float(
        cap_cfg.get("max_baseyear_share_multiplier", 1.5)
    )
    new_bus_share_multiplier = float(cap_cfg.get("new_bus_share_multiplier", 0.5))
    new_bus_flattening_exponent = float(
        cap_cfg.get("new_bus_flattening_exponent", 0.5)
    )
    if max_baseyear_share_multiplier < 0.0 or new_bus_share_multiplier < 0.0:
        logger.warning(
            "2025 nodal distribution constraints skipped: share multipliers must be non-negative."
        )
        return
    if new_bus_flattening_exponent <= 0.0:
        logger.warning(
            "2025 nodal distribution constraints skipped: new_bus_flattening_exponent must be positive."
        )
        return

    if "Generator-p_nom" not in n.model.variables:
        logger.warning(
            "2025 nodal distribution constraints skipped: Generator-p_nom variable is unavailable."
        )
        return

    gen = n.generators.copy()
    if gen.empty:
        return

    bus_country = _get_bus_country_for_clustering(n).astype(str).str.strip().str.upper()
    gen["country"] = (
        gen["bus"].map(bus_country).fillna("").astype(str).str.strip().str.upper()
    )
    gen["carrier"] = gen["carrier"].astype(str)
    gen = gen.loc[
        gen["carrier"].isin(carriers)
        & gen["country"].str.match(r"^[A-Z]{2}$", na=False)
    ].copy()
    if gen.empty:
        logger.warning(
            "2025 nodal distribution constraints skipped: no generators matched configured carriers %s.",
            ",".join(carriers),
        )
        return

    gen["p_nom"] = pd.to_numeric(gen.get("p_nom", 0.0), errors="coerce").fillna(0.0)
    gen["p_nom_min"] = pd.to_numeric(
        gen.get("p_nom_min", 0.0), errors="coerce"
    ).fillna(0.0)
    if "p_nom_max" in gen.columns:
        gen["p_nom_max"] = pd.to_numeric(gen["p_nom_max"], errors="coerce")
    else:
        gen["p_nom_max"] = np.nan
    gen["p_nom_extendable"] = (
        gen.get("p_nom_extendable", False).fillna(False).astype(bool)
    )
    if "build_year" in gen.columns:
        gen["build_year"] = pd.to_numeric(gen["build_year"], errors="coerce")
    else:
        gen["build_year"] = np.nan

    candidate = gen.loc[
        gen["p_nom_extendable"]
        & gen["p_nom_max"].notna()
        & np.isfinite(gen["p_nom_max"])
        & gen["p_nom_max"].gt(0.0)
    ].copy()
    current_year_candidates = candidate.loc[candidate["build_year"].eq(current_year)]
    if not current_year_candidates.empty:
        candidate = current_year_candidates.copy()
    if candidate.empty:
        logger.warning(
            "2025 nodal distribution constraints skipped: no extendable %s generators found for %s.",
            ",".join(carriers),
            planning_year,
        )
        return

    existing = gen.loc[~gen.index.isin(candidate.index)].copy()
    existing["existing_capacity_mw"] = np.maximum(existing["p_nom"], existing["p_nom_min"])
    existing_by_bus = (
        existing.groupby(["country", "carrier", "bus"])["existing_capacity_mw"].sum()
        if not existing.empty
        else pd.Series(dtype=float)
    )
    country_existing = (
        existing.groupby(["country", "carrier"])["existing_capacity_mw"].sum()
        if not existing.empty
        else pd.Series(dtype=float)
    )
    candidate_by_bus = candidate.groupby(["country", "carrier", "bus"])["p_nom_max"].sum()

    p_nom_var = n.model["Generator-p_nom"]
    added = 0
    groups_with_constraints = 0
    skipped_single_bus_groups = 0
    skipped_low_share_sum_groups = 0
    sample_constraints = []

    for (country, carrier), rows in candidate.groupby(["country", "carrier"]):
        buses = pd.Index(rows["bus"].astype(str).unique())
        if len(buses) <= 1:
            skipped_single_bus_groups += 1
            continue

        bus_index = pd.MultiIndex.from_arrays(
            [
                pd.Index([country] * len(buses)),
                pd.Index([carrier] * len(buses)),
                buses,
            ],
            names=["country", "carrier", "bus"],
        )
        bus_existing = existing_by_bus.reindex(bus_index, fill_value=0.0)
        bus_existing.index = buses
        bus_technical = candidate_by_bus.reindex(bus_index, fill_value=0.0)
        bus_technical.index = buses

        flattened_weights = bus_technical.clip(lower=0.0).pow(new_bus_flattening_exponent)
        if flattened_weights.sum() <= 0.0:
            continue
        flattened_share = flattened_weights / flattened_weights.sum()

        existing_total_mw = float(country_existing.get((country, carrier), 0.0))
        if existing_total_mw > 0.0:
            base_share = bus_existing / existing_total_mw
            share_cap = new_bus_share_multiplier * flattened_share
            existing_bus_mask = base_share.gt(0.0)
            share_cap.loc[existing_bus_mask] = (
                max_baseyear_share_multiplier * base_share.loc[existing_bus_mask]
            )
        else:
            share_cap = flattened_share

        share_cap = share_cap.clip(lower=0.0, upper=1.0)
        share_cap_sum = float(share_cap.sum())
        if share_cap_sum < 1.0 - 1e-9:
            skipped_low_share_sum_groups += 1
            logger.warning(
                "Skipping 2025 nodal distribution constraints for %s/%s: bus share caps sum to %.4f across %d candidate buses, which would overconstrain positive build.",
                country,
                carrier,
                share_cap_sum,
                len(buses),
            )
            continue

        country_idx = pd.Index(rows.index)
        country_lhs = p_nom_var.loc[country_idx].sum()
        ext_min_country = float(rows["p_nom_min"].sum())
        country_token = _sanitize_constraint_token(country)
        carrier_token = _sanitize_constraint_token(carrier)

        group_added = 0
        for bus, row_index in rows.groupby("bus").groups.items():
            bus_share_cap = float(share_cap.get(bus, np.nan))
            if not np.isfinite(bus_share_cap) or bus_share_cap >= 1.0 - 1e-9:
                continue

            row_index = pd.Index(row_index)
            bus_lhs = p_nom_var.loc[row_index].sum()
            ext_min_bus = float(rows.loc[row_index, "p_nom_min"].sum())
            lhs = bus_lhs - bus_share_cap * country_lhs
            rhs = ext_min_bus - bus_share_cap * ext_min_country
            bus_token = _sanitize_constraint_token(bus)
            n.model.add_constraints(
                lhs <= rhs,
                name=f"year2025_irena_nodal_distribution_max__{country_token}__{carrier_token}__{bus_token}",
            )
            added += 1
            group_added += 1
            if len(sample_constraints) < 6:
                sample_constraints.append(
                    f"{country}/{carrier}/{bus}:share_cap={bus_share_cap:.4f}"
                )

        if group_added > 0:
            groups_with_constraints += 1

    logger.info(
        "Added %d 2025 nodal distribution constraints for planning year %s: constrained_groups=%d, skipped_single_bus_groups=%d, skipped_low_share_sum_groups=%d, carriers=%s, max_baseyear_share_multiplier=%.3f, new_bus_share_multiplier=%.3f, new_bus_flattening_exponent=%.3f, sample=%s",
        added,
        planning_year,
        groups_with_constraints,
        skipped_single_bus_groups,
        skipped_low_share_sum_groups,
        ",".join(carriers),
        max_baseyear_share_multiplier,
        new_bus_share_multiplier,
        new_bus_flattening_exponent,
        ", ".join(sample_constraints) if sample_constraints else "none",
    )



def apply_country_fuel_price_overrides(fuel_price_dict, investment_year, costs, config):
    """
    Optionally apply country-level fossil fuel price overrides from a CSV.

    Expected columns:
    - country (ISO2)
    - fuel_type (oil/gas/coal)
    - optional year
    - and either `price_eur_mwh` (absolute) or `price_multiplier` (relative)
    """
    global_cfg = config.get("global_specific", {})
    base_cfg = global_cfg.get("baseyear_generation", {})
    if not bool(base_cfg.get("fossil_price_tuning_enabled", False)):
        return fuel_price_dict

    baseyear = int(base_cfg.get("year", 2020))
    apply_years_cfg = base_cfg.get(
        "fossil_price_tuning_apply_years",
        [baseyear, 2025],
    )
    if isinstance(apply_years_cfg, (int, float, str)):
        apply_years_cfg = [apply_years_cfg]
    apply_years = {
        int(year)
        for year in (apply_years_cfg or [])
        if pd.notna(pd.to_numeric(pd.Series([year]), errors="coerce").iloc[0])
    }
    if int(investment_year) not in apply_years:
        logger.info(
            "Skipping fossil price overrides for %s (configured apply years are %s).",
            investment_year,
            sorted(apply_years),
        )
        return fuel_price_dict

    source_years_cfg = base_cfg.get(
        "fossil_price_tuning_source_years",
        [baseyear],
    )
    if isinstance(source_years_cfg, (int, float, str)):
        source_years_cfg = [source_years_cfg]
    source_years = {
        int(year)
        for year in (source_years_cfg or [])
        if pd.notna(pd.to_numeric(pd.Series([year]), errors="coerce").iloc[0])
    }

    override_csv_cfg = base_cfg.get("fossil_price_override_csv", "")
    if not override_csv_cfg:
        logger.warning(
            "Fossil price tuning enabled but no fossil_price_override_csv configured."
        )
        return fuel_price_dict

    override_csv = _repo_path(override_csv_cfg)
    if not os.path.exists(override_csv):
        logger.warning(
            "Fossil price override CSV not found at %s; using original fuel prices.",
            override_csv,
        )
        return fuel_price_dict

    try:
        override = pd.read_csv(override_csv)
    except Exception as exc:
        logger.warning(
            "Could not read fossil price override CSV %s (%s); using original fuel prices.",
            override_csv,
            exc,
        )
        return fuel_price_dict

    if override.empty:
        logger.warning("Fossil price override CSV is empty at %s.", override_csv)
        return fuel_price_dict

    cols = {c.lower().strip(): c for c in override.columns}
    if "country" not in cols or "fuel_type" not in cols:
        logger.warning(
            "Fossil price override CSV %s must contain columns 'country' and 'fuel_type'.",
            override_csv,
        )
        return fuel_price_dict

    has_abs = "price_eur_mwh" in cols
    has_mult = "price_multiplier" in cols
    if not has_abs and not has_mult:
        logger.warning(
            "Fossil price override CSV %s must contain 'price_eur_mwh' and/or 'price_multiplier'.",
            override_csv,
        )
        return fuel_price_dict

    rename_cols = {
        cols["country"]: "country",
        cols["fuel_type"]: "fuel_type",
    }
    if "year" in cols:
        rename_cols[cols["year"]] = "year"
    if has_abs:
        rename_cols[cols["price_eur_mwh"]] = "price_eur_mwh"
    if has_mult:
        rename_cols[cols["price_multiplier"]] = "price_multiplier"
    override = override.rename(columns=rename_cols)

    if "year" in override.columns:
        override["year"] = pd.to_numeric(override["year"], errors="coerce")
        exact_year = override["year"].eq(int(investment_year))
        reusable_source_year = override["year"].isin(source_years)
        no_year = override["year"].isna()
        override = override.loc[
            no_year | exact_year | reusable_source_year
        ].copy()
        override["_year_priority"] = np.select(
            [exact_year.loc[override.index], no_year.loc[override.index]],
            [2, 1],
            default=0,
        )

    override["country"] = override["country"].astype(str).str.upper().str.strip()
    override["fuel_type"] = override["fuel_type"].astype(str).str.lower().str.strip()
    override = override.loc[
        override["country"].str.len().eq(2)
        & override["fuel_type"].isin(["oil", "gas", "coal"])
    ].copy()
    if "_year_priority" in override.columns:
        override = (
            override.sort_values(["country", "fuel_type", "_year_priority"])
            .drop_duplicates(["country", "fuel_type"], keep="last")
            .drop(columns=["_year_priority"])
        )
    if override.empty:
        logger.warning(
            "Fossil price override CSV %s has no valid rows for year %s.",
            override_csv,
            investment_year,
        )
        return fuel_price_dict

    out = {
        "oil": dict(fuel_price_dict.get("oil", {})),
        "gas": dict(fuel_price_dict.get("gas", {})),
        "coal": dict(fuel_price_dict.get("coal", {})),
    }
    counts = {"oil": 0, "gas": 0, "coal": 0}
    applied = 0
    for _, row in override.iterrows():
        fuel = row["fuel_type"]
        country = row["country"]
        default_price = float(costs.at[fuel, "fuel"]) if fuel in costs.index else 0.0
        base_price = float(out.get(fuel, {}).get(country, default_price))
        new_price = np.nan

        if has_abs and "price_eur_mwh" in row.index and pd.notna(row.get("price_eur_mwh")):
            new_price = pd.to_numeric(pd.Series([row["price_eur_mwh"]]), errors="coerce").iloc[0]
        elif has_mult and "price_multiplier" in row.index and pd.notna(row.get("price_multiplier")):
            mult = pd.to_numeric(pd.Series([row["price_multiplier"]]), errors="coerce").iloc[0]
            if pd.notna(mult):
                new_price = base_price * float(mult)

        if pd.isna(new_price):
            continue
        out.setdefault(fuel, {})[country] = float(max(new_price, 0.0))
        counts[fuel] = counts.get(fuel, 0) + 1
        applied += 1

    if applied > 0:
        logger.info(
            "Applied fossil price overrides from %s for %s: rows=%d, oil=%d, gas=%d, coal=%d",
            override_csv,
            investment_year,
            applied,
            counts.get("oil", 0),
            counts.get("gas", 0),
            counts.get("coal", 0),
        )
        return out

    logger.warning(
        "No valid fossil price overrides were applied from %s for year %s.",
        override_csv,
        investment_year,
    )
    return fuel_price_dict
