# -*- coding: utf-8 -*-
"""
Adapted for linopy and pypsa-earth by Filip Matic and Theo Moers
"""

from importlib.util import find_spec
import json
import logging
import time

import numpy as np
import pandas as pd
import xarray as xr
from linopy import Variable

logger = logging.getLogger(__name__)

import tsam.timeseriesaggregation as tsam  # noqa: E402

def _ensure_multiindex_snapshots(df: pd.DataFrame):
    """
    Ensure timeseries_df has MultiIndex [investment_period, snapshot].
    Returns (df_with_multiindex, was_single_index, investment_period_label).
    """
    if isinstance(df.index, pd.MultiIndex):
        return df, False, ""
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError(
            "Expected timeseries index to be a pandas.DatetimeIndex when no "
            "investment_period level is given."
        )
    period_label = str(idx[0].year) if len(idx) else "period_0"
    df2 = df.copy()
    df2.index = pd.MultiIndex.from_arrays(
        [[period_label] * len(idx), idx],
        names=["investment_period", "snapshot"],
    )
    return df2, True, period_label


def _persist_period_id(n, period_id: pd.Series) -> None:
    """Persist period_id into n.temporal_cluster and n.meta for save/load."""
    if not hasattr(n, "temporal_cluster") or not isinstance(n.temporal_cluster, dict):
        n.temporal_cluster = {}
    n.temporal_cluster["period_id"] = period_id

    try:
        if not hasattr(n, "meta") or n.meta is None:
            n.meta = {}
        n.meta["temporal_cluster_period_id"] = period_id.to_json(
            orient="split", date_format="iso"
        )
    except Exception as e:
        logger.warning(f"Could not serialize period_id to n.meta: {e}")


def _restore_period_id(n) -> pd.Series | None:
    """
    Try to restore period_id from n.temporal_cluster or from n.meta.
    
    Returns period_id with index matching n.snapshots exactly (same object reference).
    This ensures xarray operations work correctly without index conflicts.
    """
    # First try temporal_cluster (in-memory, most reliable)
    if hasattr(n, "temporal_cluster") and isinstance(n.temporal_cluster, dict):
        if "period_id" in n.temporal_cluster:
            period_id = n.temporal_cluster["period_id"]
            # Always recreate with snapshots index to ensure exact match
            if hasattr(n, "snapshots"):
                return pd.Series(period_id.values, index=n.snapshots, name="period_id")
            return period_id
    
    # Try to restore from meta (JSON serialized)
    if hasattr(n, "meta") and isinstance(n.meta, dict) and "temporal_cluster_period_id" in n.meta:
        try:
            # Try standard deserialization
            ser = pd.read_json(n.meta["temporal_cluster_period_id"], orient="split", typ="series")
        except (ValueError, TypeError) as e:
            # If that fails, manually extract values from JSON
            try:
                json_data = json.loads(n.meta["temporal_cluster_period_id"])
                if "data" in json_data:
                    ser = pd.Series(json_data["data"], name="period_id")
                else:
                    raise ValueError("Unexpected JSON structure")
            except Exception as e2:
                logger.warning(f"Failed to deserialize period_id from n.meta: {e}, fallback also failed: {e2}")
                return None
        
        # Always recreate with snapshots index to ensure exact match
        if hasattr(n, "snapshots") and len(ser) == len(n.snapshots):
            period_id = pd.Series(ser.values, index=n.snapshots, name="period_id")
            # Store back in temporal_cluster for future use
        if not hasattr(n, "temporal_cluster") or not isinstance(n.temporal_cluster, dict):
            n.temporal_cluster = {}
            n.temporal_cluster["period_id"] = period_id
            return period_id
        else:
            logger.warning(f"period_id length ({len(ser)}) doesn't match snapshots length ({len(n.snapshots) if hasattr(n, 'snapshots') else 'N/A'})")
            return None
    
    return None


def _weight_series_for_stores(n) -> pd.Series:
    """
    Robustly get delta t for storage equations.
    Priority: snapshot_weightings.stores -> 'objective' -> 'weightings' -> 1.0
    """
    sw = n.snapshot_weightings
    if isinstance(sw, pd.Series):
        ser = sw.astype(float)
    elif "stores" in sw.columns:
        ser = sw["stores"].astype(float)
    elif "objective" in sw.columns:
        ser = sw["objective"].astype(float)
    elif "weightings" in sw.columns:
        ser = sw["weightings"].astype(float)
    else:
        ser = pd.Series(1.0, index=n.snapshots, dtype=float)

    # Safety: ensure strictly positive
    if (ser <= 0).any():
        bad = ser[ser <= 0].index.tolist()
        logger.warning("Non-positive snapshot weights at %s; replacing with 1.0.", bad)
        ser = ser.where(ser > 0, 1.0)
    return ser


def aggregate_snapshots(
    n,
    n_periods=12,
    hours=24,
    normed=True,
    solver="highs",
    extremePeriodMethod="None",
    clusterMethod="k_means",
    predefClusterOrder=None,
    predefClusterCenterIndices=None,
    overwrite_time_dfs=False,
):
    """
    Aggregate snapshots to typical periods and set them on the network.
    Saves mapping in `n.cluster` and stores a per-period id in
    `n.temporal_cluster['period_id']` for cyclic SOC wrapping.
    """
    timeseries_df = prepare_timeseries(n, normed)
    timeseries_df, was_single, _ = _ensure_multiindex_snapshots(timeseries_df)

    logger.info(
        "Aggregate snapshots to %s periods with %s hours using method=%s, extreme=%s",
        n_periods, hours, clusterMethod, extremePeriodMethod,
    )

    (
        map_snapshots_to_periods,
        new_snapshots,
        timeseries_clustered,
        cluster_order,
        cluster_centers,
    ) = aggregate_timeseries(
        timeseries_df,
        n_periods,
        hours,
        extremePeriodMethod,
        normed,
        clusterMethod,
        solver,
        predefClusterOrder,
        predefClusterCenterIndices,
    )

    # If we fabricated a MultiIndex, drop back to single level to match original behavior
    if was_single:
        map_snapshots_to_periods.index = map_snapshots_to_periods.index.droplevel(0)
        new_snapshots.index = new_snapshots.index.droplevel(0)
        timeseries_clustered.index = timeseries_clustered.index.droplevel(0)

    # Save mapping original -> typical periods (for backwards compatibility)
    n.cluster = map_snapshots_to_periods
    
    # Store clusterOrder and clusterCenterIndices as network attributes for later saving
    if cluster_order is not None:
        n.cluster_order = cluster_order
    if cluster_centers is not None:
        n.cluster_centers = cluster_centers

    # Re-set snapshots and weightings (preserve column totals exactly)
    old_w = n.snapshot_weightings.copy()

    def _sum_cols(w):
        # Return a Series of totals; Series -> {"__all__": sum}, DataFrame -> per-column sums
        if isinstance(w, pd.Series):
            return pd.Series({"__all__": float(w.sum())})
        elif isinstance(w, pd.DataFrame):
            return w.sum(axis=0).astype(float)
        else:
            return pd.Series({"__all__": float(np.asarray(w).sum())})

    totals_before = _sum_cols(old_w)

    # Align to the new typical-period index, then switch snapshots
    aligned_old = old_w.reindex(new_snapshots.index).fillna(1.0)
    n.set_snapshots(new_snapshots.index)

    # sanity
    assert len(aligned_old) == len(n.snapshots), "weightings misaligned after aggregation"
    snap_idx = pd.Index(n.snapshots)
    assert snap_idx.is_monotonic_increasing and snap_idx.is_unique, "snapshots must be sorted and unique"

    # Apply TSAM period occurrence weights
    n.snapshot_weightings = aligned_old.mul(new_snapshots.weightings.astype(float), axis=0)

    # Renormalize to preserve the original totals (parity)
    totals_after = _sum_cols(n.snapshot_weightings)

    if "__all__" in totals_before.index:
        # Series case: one global factor
        tb = totals_before["__all__"]
        ta = totals_after.get("__all__", 0.0)
        if ta > 0:
            n.snapshot_weightings *= (tb / ta)
    else:
        # DataFrame case: per-column factors; safe for zero/NaN
        factors = (totals_before / totals_after).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        n.snapshot_weightings = n.snapshot_weightings.mul(factors, axis=1)


    # Build a per-period id (0..P-1) aligned with current n.snapshots
    assert len(n.snapshots) % hours == 0, "Non-integer number of periods after aggregation"
    num_periods = len(n.snapshots) // hours
    period_id = pd.Index(
        np.repeat(np.arange(num_periods), hours), name="period_id"
    ).to_series(index=n.snapshots)
    _persist_period_id(n, period_id)

    # Optionally overwrite time-dependent component data with clustered values
    if overwrite_time_dfs:
        overwrite_time_dependent(n, timeseries_clustered)


def prepare_timeseries(n, normed):
    """Collect time-dependent data for TSAM."""
    timeseries_df = pd.DataFrame(index=n.snapshots)
    for component in n.all_components:
        pnl = n.pnl(component)
        for key in pnl.keys():
            if not pnl[key].empty:
                timeseries_df = pd.concat([timeseries_df, pnl[key]], axis=1)

    if normed:
        timeseries_df = timeseries_df / timeseries_df.max().replace(0, 1)

    # Clean up
    to_drop = timeseries_df.columns[timeseries_df.isna().any()]
    if not to_drop.empty:
        logger.warning("Dropping NaNs in time series columns: %s", list(to_drop))
        timeseries_df = timeseries_df.drop(columns=to_drop)

    timeseries_df = timeseries_df.loc[:, ~timeseries_df.columns.duplicated()]

    # Clip tiny negatives
    EPS = 1e-12
    tiny_neg = (timeseries_df < 0) & (timeseries_df.abs() < EPS)
    if tiny_neg.any().any():
        c = int(tiny_neg.sum().sum())
        logger.info("Clipping %d tiny negative values (< %g) to 0.0 before TSAM.", c, EPS)
        timeseries_df = timeseries_df.mask(timeseries_df.abs() < EPS, 0.0).clip(lower=0.0)
    
    if timeseries_df.shape[1] == 0:
        raise ValueError("No valid time-series columns available for TSAM after cleaning.")

    return timeseries_df

def aggregate_timeseries(
    timeseries_df,
    n_periods,
    hours,
    extremePeriodMethod,
    normed,
    clusterMethod,
    solver,
    predefClusterOrder,
    predefClusterCenterIndices,
):
    """Call TSAM to compute typical periods; return mapping and clustered data."""
    def concat_df(df, df_final, year):
        df2 = df.copy()
        mi = pd.MultiIndex.from_product([[year], df.index],
                                        names=["investment_period","snapshot"])
        df2.index = mi
        return pd.concat([df_final, df2])


    map_snapshots_to_periods_all = pd.DataFrame()
    new_snapshots_all = pd.DataFrame()
    clustered_all = pd.DataFrame()
    cluster_order_saved = None  # Will store clusterOrder from first year
    cluster_centers_saved = None  # Will store clusterCenterIndices from first year

    # Handle each investment period (first level of MultiIndex)
    for year in timeseries_df.index.levels[0]:
        _method = clusterMethod
        aggregation = tsam.TimeSeriesAggregation(
            timeseries_df.loc[year],
            noTypicalPeriods=n_periods,
            extremePeriodMethod=extremePeriodMethod,
            rescaleClusterPeriods=False,
            hoursPerPeriod=hours,
            clusterMethod=_method,
            predefClusterOrder=predefClusterOrder,
            predefClusterCenterIndices=predefClusterCenterIndices,
            numericalTolerance=1e-8,
        )

        clustered = aggregation.createTypicalPeriods()
        if normed:
            clustered = clustered.mul(timeseries_df.loc[year].max())

        mapping = aggregation.indexMatching()
        mapping["day_of_year"] = (mapping.index - mapping.index[0]).days + 1
        weights = aggregation.clusterPeriodNoOccur
        
        # Capture clusterOrder and clusterCenterIndices for first year (for saving to pickle)
        if cluster_order_saved is None:
            if hasattr(aggregation, 'clusterOrder'):
                cluster_order_saved = aggregation.clusterOrder
            if hasattr(aggregation, 'clusterCenterIndices'):
                cluster_centers_saved = aggregation.clusterCenterIndices
        # Period column name used by TSAM mapping
        period_col = "PeriodNum" if "PeriodNum" in mapping.columns else (
            "PeriodID" if "PeriodID" in mapping.columns else None
        )
        if period_col is None:
            raise KeyError(f"TSAM mapping missing PeriodNum/PeriodID; have: {list(mapping.columns)}")

        # k-means has no center day indices -> centers=None
        centers = getattr(aggregation, "clusterCenterIndices", None)

        if centers is not None:
            # MEDOIDS PATH (has actual center-day indices)
            # Build exactly `hours` snapshots per cluster, aligned with TSAM medoid days
            tmp = mapping.reset_index()

            # robustly grab the timestamp column, whatever its current name is
            idx_name = mapping.index.name or "index"
            if idx_name not in tmp.columns and "index" in tmp.columns:
                idx_name = "index"
            tmp = tmp.rename(columns={idx_name: "ts"})

            # detect a timestep column if TSAM provides one (names vary)
            timestep_col = next((c for c in ["TimeStep", "Timestep", "timeindex",
                                            "time_index", "time_step", "step"]
                                if c in tmp.columns), None)

            # keep rows that belong to the medoid day per cluster
            medoid_days = {int(c) + 1 for c in np.asarray(centers, dtype=int)}
            tmp = tmp[tmp["day_of_year"].isin(medoid_days)].copy()

            # stable ordering
            sort_cols = [period_col, "day_of_year"] + ([timestep_col] if timestep_col else [])
            tmp = tmp.sort_values(sort_cols)

            # take the first contiguous `hours` snapshots per cluster
            if timestep_col:
                tmp = tmp.sort_values([period_col, timestep_col])
                new = tmp.groupby(period_col, group_keys=False).head(hours)
            else:
                new = tmp.groupby(period_col, group_keys=False).head(hours)

            # restore timestamp index and align clustered index to it
            new = new.set_index("ts")
            clustered = clustered.copy()
            clustered.set_index(new.index, inplace=True)

            # attach occurrence weights from TSAM
            new["weightings"] = new[period_col].map(weights).astype(float)

            # sanity checks
            expected = n_periods * hours
            assert len(new) == expected,  f"MEDOIDS index rows {len(new)} != expected {expected}"
            assert len(clustered) == expected, f"MEDOIDS clustered rows {len(clustered)} != expected {expected}"

        else:
            # K-MEANS PATH (no day indices): pick the earliest day per cluster
            tmp = mapping.reset_index()

            # robustly grab the timestamp column, whatever its current name is
            idx_name = mapping.index.name or "index"
            if idx_name not in tmp.columns and "index" in tmp.columns:
                idx_name = "index"
            tmp = tmp.rename(columns={idx_name: "ts"})

            # detect a timestep column if TSAM provides one (names vary)
            timestep_col = next((c for c in ["TimeStep", "Timestep", "timeindex",
                                            "time_index", "time_step", "step"]
                                if c in tmp.columns), None)

            # ensure sort stability
            sort_cols = [period_col, "day_of_year"] + ([timestep_col] if timestep_col else [])
            tmp = tmp.sort_values(sort_cols)

            # for each cluster, find its earliest represented day
            first_day = (
                tmp.groupby(period_col, as_index=False)["day_of_year"].min()
                .rename(columns={"day_of_year": "min_day"})
            )
            tmp = tmp.merge(first_day, on=period_col, how="left")

            # keep that earliest day per cluster
            tmp = tmp.loc[tmp["day_of_year"] == tmp["min_day"]].copy()

            # limit to exactly `hours` snapshots per cluster (first contiguous block)
            if timestep_col:
                tmp = tmp.sort_values([period_col, timestep_col])
                new = tmp.groupby(period_col, group_keys=False).head(hours)
            else:
                new = tmp.groupby(period_col, group_keys=False).head(hours)

            # restore timestamp index for alignment with `clustered`
            # (make sure it's datetime-like if needed)
            new = new.set_index("ts")
            clustered = clustered.copy()
            clustered.set_index(new.index, inplace=True)

            # build weights from TSAM
            weights = aggregation.clusterPeriodNoOccur
            new["weightings"] = new[period_col].map(weights).astype(float)

            # sanity checks
            expected = n_periods * hours
            assert len(clustered) == expected, f"TSAM clustered rows {len(clustered)} != expected {expected}"
            assert len(new) == expected, f"Representative index rows {len(new)} != expected {expected}"


        map_snapshots_to_periods_all = concat_df(mapping, map_snapshots_to_periods_all, year)
        new_snapshots_all = concat_df(new, new_snapshots_all, year)
        clustered_all = concat_df(clustered, clustered_all, year)

    # Sort new_snapshots_all by index to ensure monotonic increasing timestamps
    new_snapshots_all = new_snapshots_all.sort_index()
    
    return map_snapshots_to_periods_all, new_snapshots_all, clustered_all, cluster_order_saved, cluster_centers_saved


def overwrite_time_dependent(n, df_t):
    """(Optional) overwrite time-dependent data with clustered values."""
    for component in n.all_components:
        pnl = n.pnl(component)
        for key in pnl.keys():
            if not pnl[key].empty:
                pnl[key] = df_t.reindex(columns=pnl[key].columns)

def _as_da(obj):
    """
    Return suitable object for linopy constraint building.
    Linopy Variables are returned as-is (they work directly in constraints).
    xarray DataArrays/Datasets are passed through or converted.
    """
    # Import here to avoid circular imports
    from linopy import Variable
    
    # Linopy Variables work directly in constraints - don't convert
    if isinstance(obj, Variable):
        return obj
    
    if isinstance(obj, xr.DataArray):
        return obj
    if isinstance(obj, xr.Dataset):
        if len(obj.data_vars) == 1:
            return next(iter(obj.data_vars.values()))
        return obj.to_array().squeeze(drop=True)
    if hasattr(obj, "to_xarray"):
        x = obj.to_xarray()
        if isinstance(x, xr.DataArray):
            return x
        if isinstance(x, xr.Dataset):
            if len(x.data_vars) == 1:
                return next(iter(x.data_vars.values()))
            return x.to_array().squeeze(drop=True)
    
    # Last resort: return as-is and let linopy handle it
    return obj


def _get_period_id(n, snapshots, hours=24):
    """
    Get period_id aligned to `snapshots`.
    Expects n.temporal_cluster['period_id'] from your TSAM aggregation,
    otherwise derives as repeating 0..P-1 blocks of `hours`.
    """
    period_id = None
    if hasattr(n, "temporal_cluster") and isinstance(n.temporal_cluster, dict):
        period_id = n.temporal_cluster.get("period_id", None)

    if period_id is None:
        # fallback: assume snapshots already typical periods in blocks of `hours`
        if len(snapshots) % hours != 0:
            raise ValueError("Cannot derive period_id: len(snapshots) not multiple of hours.")
        P = len(snapshots) // hours
        period_id = pd.Series(np.repeat(np.arange(P), hours), index=snapshots, name="period_id")
        logger.warning("period_id missing; derived as repeating blocks. (Assumes snapshots ordered by period.)")
    else:
        # ensure exact alignment
        period_id = pd.Series(np.asarray(period_id), index=n.snapshots, name="period_id").reindex(snapshots)
        if period_id.isna().any():
            raise ValueError("period_id does not align with provided snapshots.")
        period_id = pd.Series(period_id.values, index=snapshots, name="period_id")

    # check uniform period sizes
    counts = period_id.value_counts().sort_index()
    if counts.nunique() != 1:
        raise ValueError(f"Non-uniform typical period length: {counts.to_dict()}")
    inferred_hours = int(counts.iloc[0])
    return period_id, inferred_hours


def _first_last_positions(period_id):
    """
    Return:
      - is_first[pos] boolean array for snapshots
      - idx_last array (length n_periods): position of last snapshot of each period (in snapshot order)
      - idx_last_prev array: last snapshot of previous period (cyclic)
      - per_of_snap_pos array: per snapshot -> period position 0..P-1
      - periods: Index of unique period labels in appearance order
    Uses positional indexing so it works with MultiIndex snapshots.
    """
    per_labels = pd.unique(period_id.values)  # preserves appearance order
    periods = pd.Index(per_labels, name="period")

    # Build table of (pos, period_label)
    df = pd.DataFrame({"pos": np.arange(len(period_id)), "per": period_id.values})
    g = df.groupby("per", sort=False)["pos"]
    first_pos = g.first()
    last_pos = g.last()

    first_positions = set(first_pos.values.tolist())
    is_first = np.array([i in first_positions for i in range(len(period_id))], dtype=bool)

    idx_last = last_pos.reindex(periods).values.astype(int)        # length P
    idx_last_prev = np.roll(idx_last, 1)                           # cyclic previous
    per_of_snap_pos = periods.get_indexer(period_id.values)        # length T

    return is_first, idx_last, idx_last_prev, per_of_snap_pos, periods


def _remap_snapshot_to_period(obj, periods):
    """
    Helper to remap indexed object from snapshot dimension to period dimension.
    Works for both linopy Variables and xarray DataArrays.
    
    Input: obj with 'snapshot' dimension of length P (one per period)
    Output: same obj but with 'period' dimension instead
    """
    return obj.rename({"snapshot": "period"}).assign_coords(period=periods)


def _remove_constraints_for_subset(m, prefix, names_to_remove):
    """
    Remove constraints whose name is in names_to_remove if present.
    We keep this conservative: remove by exact name only.
    """
    for nm in names_to_remove:
        if nm in m.constraints:
            m.remove_constraints(nm)
            logger.info("Removed constraint: %s", nm)


# ---------------------------
# Main entry
# ---------------------------

def add_kotzur_storage_constraints(
    n,
    snapshots=None,
    *,
    hours_per_period=24,
    su_carriers=("phs", "hydro"),
    store_carriers=("battery", "battery storage", "h2", "h2 store tank"),
):
    """
    Enforce Kotzur-style storage constraints on:
      - StorageUnits with carrier in su_carriers
      - Stores with carrier in store_carriers
    Keep default energy_balance for all other Stores (accounting stores).

    Assumes:
      - n.model already built (linopy)
      - snapshots correspond to typical-day reduced horizon (e.g. P*hours)
      - dt for recursion = 1 per snapshot (do NOT use occurrence weights as dt)
    """
    if snapshots is None:
        snapshots = n.snapshots
    if not hasattr(n, "model") or n.model is None:
        raise AttributeError("n.model is missing. Build the linopy model before adding constraints.")

    m = n.model

    # period structure
    period_id, hours = _get_period_id(n, snapshots, hours=hours_per_period)
    is_first, idx_last, idx_last_prev, per_of_snap_pos, periods = _first_last_positions(period_id)

    # coords
    snap_coord = snapshots
    per_coord = periods
    one_snap = xr.DataArray(1.0, dims=["snapshot"], coords={"snapshot": snap_coord})

    # ------------------------------------------------------------
    # StorageUnits: apply to PHS + hydro only, keep default for others
    # ------------------------------------------------------------
    sus = n.storage_units.copy()
    if not sus.empty and "carrier" in sus.columns:
        mask_su = sus["carrier"].astype(str).str.lower().isin([c.lower() for c in su_carriers])
        sus_k = sus.loc[mask_su]
        sus_other = sus.loc[~mask_su]
    else:
        sus_k = sus.iloc[0:0]
        sus_other = sus

    if len(sus_k) > 0:
        # remove global StorageUnit energy balance and restore for non-targets
        # (PyPSA creates a single StorageUnit-energy_balance over all units)
        _remove_constraints_for_subset(m, "StorageUnit-", ["StorageUnit-energy_balance"])

        if len(sus_other) > 0:
            _readd_default_storageunit_energy_balance(n, snapshots, sus_other.index)
            logger.info("Readding storage constraints for Storage Units")

        # now add Kotzur for target units
        _add_kotzur_storageunit_block(n, snapshots, sus_k.index, is_first, idx_last_prev, per_of_snap_pos, periods, hours, one_snap, period_id)

        logger.info("Kotzur constraints added for StorageUnits: %d", len(sus_k))
    else:
        logger.info("No StorageUnits matched carriers %s; skipping StorageUnit Kotzur block.", su_carriers)

    # ------------------------------------------------------------
    # Stores: apply to battery + H2 only, keep default for accounting stores
    # ------------------------------------------------------------
    stores = n.stores.copy()
    if stores.empty or "carrier" not in stores.columns:
        logger.info("No Stores or missing carrier column; skipping Store Kotzur block.")
        return

    mask_st = stores["carrier"].astype(str).str.lower().isin([c.lower() for c in store_carriers])
    stores_k = stores.loc[mask_st]
    stores_other = stores.loc[~mask_st]

    if len(stores_k) == 0:
        logger.info("No Stores matched carriers %s; leaving default Store constraints.", store_carriers)
        return

    # remove global Store energy balance and restore for non-targets
    _remove_constraints_for_subset(m, "Store-", ["Store-energy_balance"])
    if len(stores_other) > 0:
        _readd_default_store_energy_balance(n, snapshots, stores_other.index)
        logger.info("Readding store energy balance")

    # add Kotzur for target stores
    _add_kotzur_store_block(n, snapshots, stores_k.index, is_first, idx_last_prev, per_of_snap_pos, periods, hours, one_snap)

    logger.info("Kotzur constraints added for Stores: %d; default energy_balance restored for accounting stores: %d",
                len(stores_k), len(stores_other))


# ---------------------------
# Default energy balances (restore for non-targets)
# ---------------------------

def _readd_default_storageunit_energy_balance(n, snapshots, su_index):
    """
    Re-add PyPSA-like StorageUnit SOC energy balance for a subset of storage units.
    Uses dt=1 hour per snapshot (typical-day reduced horizon).
    """
    m = n.model
    E = m.variables["StorageUnit-state_of_charge"].loc[snapshots, su_index]
    Pch = m.variables["StorageUnit-p_store"].loc[snapshots, su_index]
    Pdis = m.variables["StorageUnit-p_dispatch"].loc[snapshots, su_index]

    Spill = None
    if "StorageUnit-spill" in m.variables:
        Spill = m.variables["StorageUnit-spill"].loc[snapshots, su_index]

    sus = n.storage_units.loc[su_index]

    # elapsed hours per snapshot
    eh = xr.DataArray(
        n.snapshot_weightings.stores.reindex(snapshots).fillna(1.0).astype(float).values,
        dims=["snapshot"],
        coords={"snapshot": snapshots},
    )
    eh_2d = eh.expand_dims(StorageUnit=su_index).transpose("snapshot", "StorageUnit")

    eta_ch = xr.DataArray(
        sus.efficiency_store.fillna(1.0).values,
        dims=["StorageUnit"],
        coords={"StorageUnit": su_index},
    )
    eta_dis = xr.DataArray(
        sus.efficiency_dispatch.fillna(1.0).values,
        dims=["StorageUnit"],
        coords={"StorageUnit": su_index},
    )
    loss = xr.DataArray(
        sus.standing_loss.fillna(0.0).values,
        dims=["StorageUnit"],
        coords={"StorageUnit": su_index},
    )
    eff_stand = (1.0 - loss).clip(min=1e-12) ** eh_2d

    inflow_df = n.storage_units_t.inflow.reindex(index=snapshots, columns=su_index).fillna(0.0)
    inflow = xr.DataArray(
        inflow_df.values,
        dims=["snapshot", "StorageUnit"],
        coords={"snapshot": snapshots, "StorageUnit": su_index},
    )

    E_da = _as_da(E)
    Pch_da = _as_da(Pch)
    Pdis_da = _as_da(Pdis)
    Spill_da = _as_da(Spill) if Spill is not None else 0.0

    cyclic = sus.cyclic_state_of_charge.fillna(False).astype(bool)
    cyc_idx = cyclic[cyclic].index
    non_idx = cyclic[~cyclic].index

    not_first = xr.DataArray(np.arange(len(snapshots)) > 0, dims=["snapshot"], coords={"snapshot": snapshots})
    first = ~not_first

    # cyclic assets: include previous soc via roll for all snapshots
    if len(cyc_idx) > 0:
        E_c = E_da.loc[:, cyc_idx]
        Pch_c = Pch_da.loc[:, cyc_idx]
        Pdis_c = Pdis_da.loc[:, cyc_idx]
        eh_c = eh_2d.sel(StorageUnit=cyc_idx)
        eta_ch_c = eta_ch.loc[cyc_idx]
        eta_dis_c = eta_dis.loc[cyc_idx]
        eff_stand_c = eff_stand.sel(StorageUnit=cyc_idx)
        inflow_c = inflow.loc[:, cyc_idx]
        spill_c = Spill_da.loc[:, cyc_idx] if Spill is not None else 0.0

        lhs_c = (
            -E_c
            - (eh_c / eta_dis_c) * Pdis_c
            + (eh_c * eta_ch_c) * Pch_c
            + eff_stand_c * E_c.roll(snapshot=1)
            - eh_c * spill_c
        )
        rhs_c = -(eh_c * inflow_c)
        m.add_constraints(lhs_c == rhs_c, name="StorageUnit-energy_balance-other-cyclic")

    # non-cyclic assets: interior snapshots include previous, first snapshot uses initial soc
    if len(non_idx) > 0:
        E_n = E_da.loc[:, non_idx]
        Pch_n = Pch_da.loc[:, non_idx]
        Pdis_n = Pdis_da.loc[:, non_idx]
        eh_n = eh_2d.sel(StorageUnit=non_idx)
        eta_ch_n = eta_ch.loc[non_idx]
        eta_dis_n = eta_dis.loc[non_idx]
        eff_stand_n = eff_stand.sel(StorageUnit=non_idx)
        inflow_n = inflow.loc[:, non_idx]
        spill_n = Spill_da.loc[:, non_idx] if Spill is not None else 0.0

        lhs_n = (
            -E_n
            - (eh_n / eta_dis_n) * Pdis_n
            + (eh_n * eta_ch_n) * Pch_n
            + eff_stand_n * E_n.shift(snapshot=1)
            - eh_n * spill_n
        )
        rhs_n = -(eh_n * inflow_n)
        mask_n = not_first.expand_dims(StorageUnit=non_idx).transpose("snapshot", "StorageUnit")
        m.add_constraints(lhs_n == rhs_n, name="StorageUnit-energy_balance-other-noncyclic", mask=mask_n)

        soc_init_n = xr.DataArray(
            sus.loc[non_idx, "state_of_charge_initial"].fillna(0.0).values,
            dims=["StorageUnit"],
            coords={"StorageUnit": non_idx},
        )
        lhs_n_first = (
            -E_n
            - (eh_n / eta_dis_n) * Pdis_n
            + (eh_n * eta_ch_n) * Pch_n
            - eh_n * spill_n
        )
        rhs_n_first = -(eh_n * inflow_n) - soc_init_n
        mask_f = first.expand_dims(StorageUnit=non_idx).transpose("snapshot", "StorageUnit")
        m.add_constraints(lhs_n_first == rhs_n_first, name="StorageUnit-energy_balance-other-initial", mask=mask_f)


def _readd_default_store_energy_balance(n, snapshots, store_index):
    """
    Re-add PyPSA-like Store energy balance for a subset of stores.
    Uses dt=1 hour per snapshot (typical-day reduced horizon).
    Sign convention: e[t] = decay*e[t-1] - p[t]
    (Store-p positive means discharge to bus, decreases stored energy.)
    """
    m = n.model
    E = m.variables["Store-e"].loc[snapshots, store_index]
    P = m.variables["Store-p"].loc[snapshots, store_index]

    st = n.stores.loc[store_index]

    eh = xr.DataArray(
        n.snapshot_weightings.stores.reindex(snapshots).fillna(1.0).astype(float).values,
        dims=["snapshot"],
        coords={"snapshot": snapshots},
    )
    eh_2d = eh.expand_dims(Store=store_index).transpose("snapshot", "Store")

    loss = xr.DataArray(
        st.standing_loss.fillna(0.0).values,
        dims=["Store"],
        coords={"Store": store_index},
    )
    eff_stand = (1.0 - loss).clip(min=1e-12) ** eh_2d

    E_da = _as_da(E)
    P_da = _as_da(P)

    cyclic = st.e_cyclic.fillna(False).astype(bool)
    cyc_idx = cyclic[cyclic].index
    non_idx = cyclic[~cyclic].index

    not_first = xr.DataArray(np.arange(len(snapshots)) > 0, dims=["snapshot"], coords={"snapshot": snapshots})
    first = ~not_first

    if len(cyc_idx) > 0:
        E_c = E_da.loc[:, cyc_idx]
        P_c = P_da.loc[:, cyc_idx]
        eh_c = eh_2d.sel(Store=cyc_idx)
        eff_c = eff_stand.sel(Store=cyc_idx)
        lhs_c = -E_c - eh_c * P_c + eff_c * E_c.roll(snapshot=1)
        m.add_constraints(lhs_c == 0.0, name="Store-energy_balance-other-cyclic")

    if len(non_idx) > 0:
        E_n = E_da.loc[:, non_idx]
        P_n = P_da.loc[:, non_idx]
        eh_n = eh_2d.sel(Store=non_idx)
        eff_n = eff_stand.sel(Store=non_idx)
        lhs_n = -E_n - eh_n * P_n + eff_n * E_n.shift(snapshot=1)
        mask_n = not_first.expand_dims(Store=non_idx).transpose("snapshot", "Store")
        m.add_constraints(lhs_n == 0.0, name="Store-energy_balance-other-noncyclic", mask=mask_n)

        e_init_n = xr.DataArray(
            st.loc[non_idx, "e_initial"].fillna(0.0).values,
            dims=["Store"],
            coords={"Store": non_idx},
        )
        lhs_n_first = -E_n - eh_n * P_n
        mask_f = first.expand_dims(Store=non_idx).transpose("snapshot", "Store")
        m.add_constraints(lhs_n_first == -e_init_n, name="Store-energy_balance-other-initial", mask=mask_f)


# ---------------------------
# Kotzur blocks
# ---------------------------

def _add_kotzur_storageunit_block(n, snapshots, su_index, is_first, idx_last_prev, per_of_snap_pos, periods, hours, one_snap, period_id):
    """
    Kotzur-like:
      soc_intra recursion inside each typical period (start offset at 0 each period)
      soc_inter recursion across periods (cyclic), using last-hour approximation
      E = soc_intra + soc_inter(period(snapshot))
    dt = 1 per snapshot.
    
    Includes inflow and spill in intra-period energy balance when present.
    Inter-period recursion is linked via last-hour approximation only.
    """
    m = n.model
    sus = n.storage_units.loc[su_index]

    E    = m.variables["StorageUnit-state_of_charge"].loc[snapshots, su_index]
    Pch  = m.variables["StorageUnit-p_store"].loc[snapshots, su_index]
    Pdis = m.variables["StorageUnit-p_dispatch"].loc[snapshots, su_index]
    
    # Get Spill variable if it exists
    Spill = None
    if "StorageUnit-spill" in m.variables:
        Spill = m.variables["StorageUnit-spill"].loc[snapshots, su_index]
    
    # Get Inflow data (time series from n.storage_units_t.inflow)
    Inflow = None
    if hasattr(n, 'storage_units_t') and hasattr(n.storage_units_t, 'inflow'):
        inflow_df = n.storage_units_t.inflow
        if isinstance(inflow_df, pd.DataFrame) and not inflow_df.empty:
            # Get inflow for the subset of storage units we're working with
            available_cols = inflow_df.columns.intersection(su_index)
            if len(available_cols) > 0:
                inflow_subset = inflow_df.loc[snapshots, available_cols]
                Inflow = xr.DataArray(
                    inflow_subset.values,
                    dims=["snapshot", "StorageUnit"],
                    coords={"snapshot": snapshots, "StorageUnit": available_cols}
                )

    eta_ch = xr.DataArray(sus.efficiency_store.fillna(1.0).values, dims=["StorageUnit"], coords={"StorageUnit": su_index})
    eta_dis = xr.DataArray(sus.efficiency_dispatch.fillna(1.0).values, dims=["StorageUnit"], coords={"StorageUnit": su_index})
    loss = xr.DataArray(sus.standing_loss.fillna(0.0).values, dims=["StorageUnit"], coords={"StorageUnit": su_index})

    decay_step = (1.0 - loss).clip(min=1e-12)                      # per hour
    decay_inter = decay_step ** float(hours)                       # over a full day

    # variables
    snap_idx = pd.Index(snapshots, name="snapshot")
    su_idx = pd.Index(su_index, name="StorageUnit")
    per_idx = pd.Index(periods, name="period")

    soc_intra = m.add_variables(0.0, np.inf, coords=[snap_idx, su_idx], name="StorageUnit_soc_intra")
    soc_inter = m.add_variables(0.0, np.inf, coords=[per_idx, su_idx], name="StorageUnit_soc_inter")

    E_da = _as_da(E)
    Pch_da = _as_da(Pch)
    Pdis_da = _as_da(Pdis)
    soc_intra_da = _as_da(soc_intra)
    soc_inter_da = _as_da(soc_inter)
    
    # Handle optional Spill and Inflow
    Spill_da = _as_da(Spill) if Spill is not None else 0.0
    Inflow_da = Inflow if Inflow is not None else 0.0

    # masks
    first_mask = xr.DataArray(is_first, dims=["snapshot"], coords={"snapshot": snapshots})
    interior = (~first_mask).expand_dims(StorageUnit=su_idx).transpose("snapshot", "StorageUnit")
    first2d  = first_mask.expand_dims(StorageUnit=su_idx).transpose("snapshot", "StorageUnit")

    # (1) soc_intra
    # Energy balance: E[t] = decay*E[t-1] + eta_ch*Pch - Pdis/eta_dis + Inflow - Spill
    prev = soc_intra_da.shift(snapshot=1)

    rhs = (decay_step * prev) + (eta_ch * Pch_da) - (Pdis_da / eta_dis) + Inflow_da - Spill_da
    m.add_constraints(soc_intra_da - rhs == 0, name="StorageUnit_soc_intra_recur", mask=interior)

    rhs_first = (eta_ch * Pch_da) - (Pdis_da / eta_dis) + Inflow_da - Spill_da
    m.add_constraints(soc_intra_da - rhs_first == 0, name="StorageUnit_soc_intra_init", mask=first2d)

    # (2) soc_inter (cyclic over typical periods)
    soc_inter_prev = soc_inter_da.roll(period=1)

    # last-hour terms from previous period (cyclic)
    Pch_l = Pch_da.isel(snapshot=idx_last_prev)
    Pdis_l = Pdis_da.isel(snapshot=idx_last_prev)
    intra_l = soc_intra_da.isel(snapshot=idx_last_prev)

    # Remap from snapshot to period dimension
    Pch_l = _remap_snapshot_to_period(Pch_l, periods)
    Pdis_l = _remap_snapshot_to_period(Pdis_l, periods)
    intra_l = _remap_snapshot_to_period(intra_l, periods)

    # Note: Inflow and Spill are already fully accounted for in soc_intra dynamics.
    # The inter-period recursion just links periods via the last-hour approximation.
    # We do NOT add extra inflow/spill terms here.
    rhs_inter = (
        (decay_inter * soc_inter_prev)
        + (decay_step * intra_l)
        + (eta_ch * float(hours) * Pch_l)
        - ((float(hours) / eta_dis) * Pdis_l)
    )
    m.add_constraints(soc_inter_da - rhs_inter == 0, name="StorageUnit_soc_inter_recur")

    # (3) E = soc_intra + soc_inter(period(snapshot))
    soc_inter_on_snap = soc_inter_da.isel(
        period=xr.DataArray(per_of_snap_pos, dims=["snapshot"], coords={"snapshot": snapshots})
    ).assign_coords(snapshot=snapshots)

    m.add_constraints(E_da - (soc_intra_da + soc_inter_on_snap) == 0, name="StorageUnit_soc_sum")

    # (4) SOC upper bound: prefer e_nom if present; else p_nom*max_hours (+ extendable handled by existing model)
    if "e_nom" in sus.columns and sus["e_nom"].notna().any():
        e_nom = xr.DataArray(sus.e_nom.fillna(np.inf).values, dims=["StorageUnit"], coords={"StorageUnit": su_index})
        ub = e_nom.expand_dims(snapshot=snapshots).transpose("snapshot", "StorageUnit")
        m.add_constraints(E_da <= ub, name="StorageUnit_soc_ub_e_nom")
    else:
        extendable = sus.get("p_nom_extendable", pd.Series(False, index=su_index)).astype(bool)
        max_hours = sus.get("max_hours", pd.Series(np.inf, index=su_index)).fillna(np.inf)

        p_nom_fixed = sus.get("p_nom", pd.Series(0.0, index=su_index)).where(~extendable, other=np.nan)
        e_nom_fixed = (p_nom_fixed * max_hours).fillna(np.inf)
        e_nom_fixed_da = xr.DataArray(e_nom_fixed.values, dims=["StorageUnit"], coords={"StorageUnit": su_index})
        ub = e_nom_fixed_da.expand_dims(snapshot=snapshots).transpose("snapshot", "StorageUnit")
        m.add_constraints(E_da <= ub, name="StorageUnit_soc_ub_fixed")


def _add_kotzur_store_block(n, snapshots, store_index, is_first, idx_last_prev, per_of_snap_pos, periods, hours, one_snap):
    """
    Kotzur-like Store constraints (battery/H2):
      soc_intra recursion inside each typical period (start offset at 0 each period)
      soc_inter recursion across periods (cyclic), last-hour approximation
      E = soc_intra + soc_inter(period(snapshot))

    Uses Store-p directly (no split).
    PyPSA sign convention:
      e[t] = decay*e[t-1] - p[t]   (p>0 discharges, reduces e)
    """
    m = n.model
    st = n.stores.loc[store_index]

    E = m.variables["Store-e"].loc[snapshots, store_index]
    P = m.variables["Store-p"].loc[snapshots, store_index]

    loss = xr.DataArray(st.standing_loss.fillna(0.0).values, dims=["Store"], coords={"Store": store_index})
    decay_step = (1.0 - loss).clip(min=1e-12)
    decay_inter = decay_step ** float(hours)

    # variables
    snap_idx = pd.Index(snapshots, name="snapshot")
    store_idx = pd.Index(store_index, name="Store")
    per_idx = pd.Index(periods, name="period")

    soc_intra = m.add_variables(0.0, np.inf, coords=[snap_idx, store_idx], name="Store_soc_intra")
    soc_inter = m.add_variables(0.0, np.inf, coords=[per_idx, store_idx], name="Store_soc_inter")

    E_da = _as_da(E)
    P_da = _as_da(P)
    soc_intra_da = _as_da(soc_intra)
    soc_inter_da = _as_da(soc_inter)

    first_mask = xr.DataArray(is_first, dims=["snapshot"], coords={"snapshot": snapshots})
    interior = (~first_mask).expand_dims(Store=store_idx).transpose("snapshot", "Store")
    first2d  = first_mask.expand_dims(Store=store_idx).transpose("snapshot", "Store")

    # (1) soc_intra
    prev = soc_intra_da.shift(snapshot=1)
    rhs = (decay_step * prev) - P_da
    m.add_constraints(soc_intra_da - rhs == 0, name="Store_soc_intra_recur", mask=interior)

    rhs_first = -P_da
    m.add_constraints(soc_intra_da - rhs_first == 0, name="Store_soc_intra_init", mask=first2d)

    # (2) soc_inter
    soc_inter_prev = soc_inter_da.roll(period=1)

    P_l = P_da.isel(snapshot=idx_last_prev)
    intra_l = soc_intra_da.isel(snapshot=idx_last_prev)

    # Remap from snapshot to period dimension
    P_l = _remap_snapshot_to_period(P_l, periods)
    intra_l = _remap_snapshot_to_period(intra_l, periods)

    rhs_inter = (decay_inter * soc_inter_prev) + (decay_step * intra_l) - (float(hours) * P_l)
    m.add_constraints(soc_inter_da - rhs_inter == 0, name="Store_soc_inter_recur")

    # (3) E sum
    soc_inter_on_snap = soc_inter_da.isel(
        period=xr.DataArray(per_of_snap_pos, dims=["snapshot"], coords={"snapshot": snapshots})
    ).assign_coords(snapshot=snapshots)

    m.add_constraints(E_da - (soc_intra_da + soc_inter_on_snap) == 0, name="Store_soc_sum")

    # (4) SOC upper bounds: fixed e_nom or extendable Store-e_nom if exists
    extendable = st.get("e_nom_extendable", pd.Series(False, index=store_index)).astype(bool)
    non_ext = ~extendable

    if non_ext.any():
        fixed = st.index[non_ext]
        e_nom = st.loc[fixed, "e_nom"].fillna(np.inf) if "e_nom" in st.columns else pd.Series(np.inf, index=fixed)
        ub = xr.DataArray(e_nom.values, dims=["Store"], coords={"Store": fixed}).expand_dims(snapshot=snapshots).transpose("snapshot", "Store")
        m.add_constraints(E_da.loc[:, fixed] <= ub, name="Store_soc_ub_fixed")

    # If Store-e_nom variable exists and is already bounded/costed, keep it; otherwise user must bound it elsewhere
    if extendable.any() and "Store-e_nom" in m.variables:
        idx_var = m.variables["Store-e_nom"].indexes.get("Store-ext", None)
        if idx_var is not None and len(idx_var) > 0:
            e_nom_var = m.variables["Store-e_nom"].loc[idx_var].rename({"Store-ext": "Store"})
            common = store_index.intersection(e_nom_var.indexes["Store"])
            if len(common) > 0:
                m.add_constraints(E_da.loc[:, common] <= _as_da(e_nom_var).loc[common], name="Store_soc_ub_extendable")
