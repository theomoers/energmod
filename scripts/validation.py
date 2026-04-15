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

from _helpers import BASE_DIR, three_2_two_digits_country

logger = logging.getLogger(__name__)


BASEYEAR_BLOCKED_EXTENDABLE_LINK_CARRIERS = ("H2 Fuel Cell",)


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


def align_country_electricity_demand_to_owid(n, investment_year, config):
    global_cfg = config.get("global_specific", {})
    base_cfg = global_cfg.get("baseyear_generation", {})
    baseyear = int(base_cfg.get("year", 2020))
    if int(investment_year) != baseyear:
        logger.info(
            "Skipping OWID electricity-demand alignment for %s (configured baseyear is %s).",
            investment_year,
            baseyear,
        )
        return

    owid_csv = _repo_path(base_cfg.get("owid_csv", "validation/data/owid-energy-data.csv"))
    if not os.path.exists(owid_csv):
        logger.warning(
            "OWID electricity-demand alignment skipped: file not found at %s", owid_csv
        )
        return

    usecols = {"year", "iso_code", "electricity_demand"}
    owid = pd.read_csv(owid_csv, usecols=lambda c: c in usecols)
    if "electricity_demand" not in owid.columns:
        logger.warning(
            "OWID electricity-demand alignment skipped: column 'electricity_demand' not found in %s",
            owid_csv,
        )
        return

    owid = owid.loc[owid["year"] == int(baseyear)].copy()
    owid["country"] = owid["iso_code"].apply(_safe_iso3_to_iso2)
    owid = owid.loc[owid["country"].notna()].copy()
    owid["electricity_demand"] = pd.to_numeric(owid["electricity_demand"], errors="coerce").fillna(0.0)
    ref_twh = owid.groupby("country")["electricity_demand"].sum(min_count=1)
    if ref_twh.empty:
        logger.warning(
            "OWID electricity-demand alignment skipped: no reference values for year %s in %s",
            baseyear,
            owid_csv,
        )
        return

    elec_loads, load_country = _electric_load_index_and_country(n)
    if len(elec_loads) == 0:
        logger.warning("OWID electricity-demand alignment skipped: no electricity loads found on AC/low voltage buses.")
        return

    energy_mwh_by_load = _electric_load_energy_by_load_mwh(n, elec_loads)
    model_twh_before = energy_mwh_by_load.groupby(load_country).sum() / 1e6

    estimated_end_use_link_twh = pd.Series(dtype=float)
    baseline_network = None
    baseline_path = _baseline_electricity_demand_network_path(base_cfg)
    if baseline_path is not None:
        if os.path.exists(baseline_path):
            try:
                n_baseline = pypsa.Network(baseline_path)
                estimated_end_use_link_twh = _country_ac_end_use_link_withdrawal_twh(
                    n_baseline,
                    output_bus_carrier_substrings=base_cfg.get(
                        "electricity_demand_end_use_link_output_bus_carrier_substrings",
                        ["heat"],
                    ),
                )
                baseline_network = baseline_path
            except Exception as exc:
                logger.warning(
                    "OWID electricity-demand alignment: failed to load baseline network %s (%s); falling back to direct-load-only alignment.",
                    baseline_path,
                    exc,
                )
        else:
            logger.warning(
                "OWID electricity-demand alignment: baseline network not found at %s; falling back to direct-load-only alignment.",
                baseline_path,
            )

    effective_ref_twh = ref_twh.sub(estimated_end_use_link_twh, fill_value=0.0).clip(lower=0.0)

    comp_before = pd.DataFrame(
        {
            "model_twh": model_twh_before,
            "reference_twh": effective_ref_twh.reindex(model_twh_before.index),
            "owid_reference_twh": ref_twh.reindex(model_twh_before.index),
            "estimated_end_use_link_twh": estimated_end_use_link_twh.reindex(model_twh_before.index).fillna(0.0),
        }
    ).dropna(subset=["reference_twh"])
    eligible = comp_before["model_twh"].gt(0.0) & comp_before["reference_twh"].ge(0.0)
    factors_by_country = (
        comp_before.loc[eligible, "reference_twh"] / comp_before.loc[eligible, "model_twh"]
    )
    if factors_by_country.empty:
        logger.warning(
            "OWID electricity-demand alignment skipped: no countries with positive modeled and reference demand."
        )
        return

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

    model_twh_after = (
        _electric_load_energy_by_load_mwh(n, elec_loads).groupby(load_country).sum() / 1e6
    )
    comp_after = pd.DataFrame(
        {
            "model_twh": model_twh_after,
            "reference_twh": effective_ref_twh.reindex(model_twh_after.index),
            "owid_reference_twh": ref_twh.reindex(model_twh_after.index),
            "estimated_end_use_link_twh": estimated_end_use_link_twh.reindex(model_twh_after.index).fillna(0.0),
        }
    ).dropna(subset=["reference_twh"])

    before_total_twh = comp_before["model_twh"] + comp_before["estimated_end_use_link_twh"]
    after_total_twh = comp_after["model_twh"] + comp_after["estimated_end_use_link_twh"]
    before_abs_err = (before_total_twh - comp_before["owid_reference_twh"]).abs().sum()
    after_abs_err = (after_total_twh - comp_after["owid_reference_twh"]).abs().sum()
    ref_sum = comp_after["owid_reference_twh"].sum()
    before_wape = 100.0 * before_abs_err / ref_sum if ref_sum > 0 else np.nan
    after_wape = 100.0 * after_abs_err / ref_sum if ref_sum > 0 else np.nan

    logger.info(
        "Aligned country electricity demand to OWID for %s: countries_scaled=%d, global_direct_before=%.1f TWh, global_direct_after=%.1f TWh, baseline_end_use_links=%.1f TWh, global_total_before=%.1f TWh, global_total_after=%.1f TWh, global_reference=%.1f TWh, WAPE_before=%.2f%%, WAPE_after=%.2f%%, baseline_network=%s",
        baseyear,
        len(factors_by_country),
        comp_before["model_twh"].sum(),
        comp_after["model_twh"].sum(),
        comp_after["estimated_end_use_link_twh"].sum(),
        before_total_twh.sum(),
        after_total_twh.sum(),
        comp_after["owid_reference_twh"].sum(),
        before_wape,
        after_wape,
        baseline_network if baseline_network is not None else "None",
    )


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
    owid = pd.read_csv(owid_csv, usecols=lambda c: c in usecols)
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
        if "ISO3 code" in ref.columns:
            ref["country"] = ref["ISO3 code"].apply(_safe_iso3_to_iso2)
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
        return ref, chosen_year

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


def add_baseyear_generation_band(n, planning_year, config):
    global_cfg = config.get("global_specific", {})
    cfg = global_cfg.get("baseyear_generation", {})
    if not cfg or not cfg.get("baseyear_generation_constraint", False):
        return

    baseyear = int(cfg.get("year", 2020))
    try:
        current_year = int(float(planning_year))
    except Exception:
        logger.warning("Could not parse planning year '%s' for baseyear constraints", planning_year)
        return
    if current_year != baseyear:
        logger.info(
            "Skipping baseyear generation constraints for %s (configured for %s)",
            planning_year,
            baseyear,
        )
        return

    logger.info("Adding baseyear OWID country-level generation constraints for %s", planning_year)

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
        logger.warning("No model country-metric expressions found for baseyear generation constraints")
        return

    ref = _owid_country_metric_reference(owid_csv, baseyear, metrics)
    if ref.empty:
        logger.warning("No OWID references found in %s for year %s", owid_csv, baseyear)
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
                name=f"baseyear_owid_country_metric_min_slack__{country_token}__{metric_token}",
            )
            slack_high = n.model.add_variables(
                lower=0.0,
                name=f"baseyear_owid_country_metric_max_slack__{country_token}__{metric_token}",
            )
            n.model.add_constraints(
                lhs + slack_low >= lower,
                name=f"baseyear_owid_country_metric_min__{country_token}__{metric_token}",
            )
            n.model.add_constraints(
                lhs - slack_high <= upper,
                name=f"baseyear_owid_country_metric_max__{country_token}__{metric_token}",
            )
            slack_terms.extend([slack_low, slack_high])
        else:
            n.model.add_constraints(
                lhs >= lower,
                name=f"baseyear_owid_country_metric_min__{country_token}__{metric_token}",
            )
            n.model.add_constraints(
                lhs <= upper,
                name=f"baseyear_owid_country_metric_max__{country_token}__{metric_token}",
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
    lower_multiplier = cfg.get("lower_multiplier")
    upper_multiplier = cfg.get("upper_multiplier")
    lower_multiplier = (
        float(lower_multiplier) if lower_multiplier is not None else None
    )
    upper_multiplier = (
        float(upper_multiplier) if upper_multiplier is not None else None
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

        if lower_multiplier is not None or upper_multiplier is not None:
            lower_mult = 1.0 if lower_multiplier is None else lower_multiplier
            upper_mult = (
                (1.0 + tolerance) if upper_multiplier is None else upper_multiplier
            )
            lower_total = max(target_mw * lower_mult, 0.0)
            upper_total = target_mw * upper_mult
        else:
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
    Add hard country-level renewable capacity constraints around IRENA references.

    The config block is `global_specific.year2025_capacity`. Constraints apply only when
    `planning_year` matches `cfg["year"]`. Total capacity is generator `p_nom` by
    (country, configured constraint group), including fixed and extendable assets.
    Supports either symmetric `tolerance` bands or explicit one-sided multipliers.
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
    lower_multiplier = cfg.get("lower_multiplier")
    upper_multiplier = cfg.get("upper_multiplier")
    lower_multiplier = float(lower_multiplier) if lower_multiplier is not None else None
    upper_multiplier = float(upper_multiplier) if upper_multiplier is not None else None
    absolute_tolerance_mw = float(cfg.get("absolute_tolerance_mw", 0.0))
    min_reference_mw = float(cfg.get("min_reference_mw", 0.0))
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
    clipped_fixed_upper_overshoot = 0

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

        if lower_multiplier is not None or upper_multiplier is not None:
            lower_mult = 1.0 if lower_multiplier is None else lower_multiplier
            upper_mult = (1.0 + tolerance) if upper_multiplier is None else upper_multiplier
            lower_total = max(target_mw * lower_mult, 0.0)
            upper_total = target_mw * upper_mult
        else:
            band_mw = max(absolute_tolerance_mw, tolerance * abs(target_mw))
            lower_total = max(target_mw - band_mw, 0.0)
            upper_total = target_mw + band_mw
        if len(ext_idx) == 0:
            if existing_capacity_mw + 1e-6 < lower_total or existing_capacity_mw - 1e-6 > upper_total:
                missing_variable_violations += 1
                logger.warning(
                    "Cannot enforce IRENA country-capacity band for %s/%s: no extendable generators and fixed capacity %.2f %s is outside [%.2f, %.2f] %s.",
                    country,
                    carrier,
                    existing_capacity_mw / unit_scale,
                    units.upper(),
                    lower_total / unit_scale,
                    upper_total / unit_scale,
                    units.upper(),
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
            # Pragmatic fallback: if fixed capacity already overshoots the upper band,
            # clip the fixed offset used by this validation constraint.
            clipped_existing_capacity_mw = max(upper_total - ext_min_mw, 0.0)
            if clipped_existing_capacity_mw + 1e-6 < existing_capacity_mw:
                logger.warning(
                    "Clipping effective fixed capacity for IRENA country-capacity band %s/%s from %.2f to %.2f %s to respect upper bound %.2f %s (extendable p_nom_min=%.2f).",
                    country,
                    carrier,
                    existing_capacity_mw / unit_scale,
                    clipped_existing_capacity_mw / unit_scale,
                    units.upper(),
                    upper_total / unit_scale,
                    units.upper(),
                    ext_min_mw / unit_scale,
                )
                existing_capacity_mw = clipped_existing_capacity_mw
                min_total_feasible = existing_capacity_mw + ext_min_mw
                max_total_feasible = existing_capacity_mw + ext_max_mw
                clipped_fixed_upper_overshoot += 1

            if min_total_feasible - 1e-6 > upper_total:
                skipped_infeasible_by_upper_bound += 1
                logger.warning(
                    "Skipping IRENA country-capacity band for %s/%s: minimum feasible total %.2f %s exceeds upper bound %.2f %s even after clipping (fixed=%.2f, extendable p_nom_min=%.2f).",
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

        if max_total_feasible + 1e-6 < lower_total:
            skipped_infeasible_by_lower_bound += 1
            logger.warning(
                "Skipping IRENA country-capacity band for %s/%s: maximum feasible total %.2f %s is below lower bound %.2f %s (fixed=%.2f, extendable p_nom_max=%s).",
                country,
                carrier,
                max_total_feasible / unit_scale,
                units.upper(),
                lower_total / unit_scale,
                units.upper(),
                existing_capacity_mw / unit_scale,
                "inf" if not np.isfinite(ext_max_mw) else f"{ext_max_mw / unit_scale:.2f}",
            )
            continue

        lower_var = lower_total - existing_capacity_mw
        upper_var = upper_total - existing_capacity_mw
        lhs = p_nom_var.loc[ext_idx].sum()
        country_token = _sanitize_constraint_token(country)
        carrier_token = _sanitize_constraint_token(carrier)

        n.model.add_constraints(
            lhs >= lower_var,
            name=f"year2025_irena_country_capacity_min__{country_token}__{carrier_token}",
        )
        n.model.add_constraints(
            lhs <= upper_var,
            name=f"year2025_irena_country_capacity_max__{country_token}__{carrier_token}",
        )
        added += 1

    logger.info(
        "Added %d IRENA country-capacity constraints for planning year %s (reference_year=%s, tolerance=%s%%, lower_multiplier=%s, upper_multiplier=%s, min_reference=%.2f MW, skipped_low_reference=%d, skipped_fixed_only_satisfied=%d, fixed_only_violations=%d, clipped_fixed_upper_overshoot=%d, skipped_infeasible_upper=%d, skipped_infeasible_lower=%d)",
        added,
        planning_year,
        used_reference_year,
        100.0 * tolerance,
        lower_multiplier if lower_multiplier is not None else "default",
        upper_multiplier if upper_multiplier is not None else "default",
        min_reference_mw,
        skipped_low_reference,
        skipped_no_variable_but_satisfied,
        missing_variable_violations,
        clipped_fixed_upper_overshoot,
        skipped_infeasible_by_upper_bound,
        skipped_infeasible_by_lower_bound,
    )


def add_year2025_global_capacity_band(n, planning_year, config):
    """
    Add 2025 global installed-capacity band constraints for selected technologies.

    This is intended to complement (not replace) the country-level IRENA capacity
    band. Total global installed capacity is constrained around configured targets
    (e.g. solar, onwind) with a symmetric tolerance.
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

    targets = cfg.get("targets", {}) or {}
    if not targets:
        logger.warning("2025 global capacity constraints enabled but no targets configured.")
        return

    model_carriers_by_constraint = parent_cfg.get(
        "model_carriers_by_constraint",
        {key: [key] for key in targets.keys()},
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
        fixed_gen.groupby(["constraint_carrier"])["p_nom"].sum()
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
            .groupby([degenerate_ext_gen["constraint_carrier"]])
            .sum()
        )
        if fixed_capacity.empty:
            fixed_capacity = fixed_like_capacity.astype(float)
        else:
            fixed_capacity = fixed_capacity.add(fixed_like_capacity, fill_value=0.0)

    ext_groups = {}
    if not ext_gen.empty:
        for carrier, d in ext_gen.groupby(["constraint_carrier"]):
            ext_groups[carrier] = pd.Index(d.index)

    added = 0
    skipped_no_variable_but_satisfied = 0

    for carrier, target in targets.items():
        if isinstance(target, str) and target.upper().startswith("X"):
            logger.info("Skipping %s global-capacity target (placeholder target '%s')", carrier, target)
            continue

        carrier = str(carrier)
        target_mw = float(target) * unit_scale
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
            "(fixed-equivalent: %.2f %s)",
            carrier,
            planning_year,
            lower_total / unit_scale,
            upper_total / unit_scale,
            units.upper(),
            existing_capacity_mw / unit_scale,
            units.upper(),
        )

    logger.info(
        "Added %d global 2025 capacity bands for planning year %s (tolerance=%.2f%%, skipped_fixed_only_satisfied=%d)",
        added,
        planning_year,
        100.0 * tolerance,
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
    if int(investment_year) != baseyear:
        logger.info(
            "Skipping fossil price overrides for %s (configured baseyear is %s).",
            investment_year,
            baseyear,
        )
        return fuel_price_dict

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
        override = override.loc[
            override["year"].isna() | override["year"].eq(int(investment_year))
        ].copy()

    override["country"] = override["country"].astype(str).str.upper().str.strip()
    override["fuel_type"] = override["fuel_type"].astype(str).str.lower().str.strip()
    override = override.loc[
        override["country"].str.len().eq(2)
        & override["fuel_type"].isin(["oil", "gas", "coal"])
    ].copy()
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
