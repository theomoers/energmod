# -*- coding: utf-8 -*-
"""
Linopy-compatible temporal clustering (TSAM) + storage constraints for typical-day models.

Default behaviour = **Pyomo parity**:
- TSAM aggregation across (optional) investment periods
- period_id persisted on the Network (n.temporal_cluster['period_id']) for cyclic SOC wrapping
- Storage equations per Kotzur et al. (2018) **matching the original Pyomo script**:
  (1) soc_intra recursion with **unit time-step** (no dt scaling), **no inflow/spill** terms
  (2) soc_inter recursion across typical periods (cyclic) using **last-hour** terms scaled by `hours`
  (3) total SOC: E = soc_intra + soc_inter(period_of_snapshot)
  (4) SOC upper bounds identical to the original
  (5) Removes PyPSA's default StorageUnit SOC constraints before adding the above

Optional “improved” switches (off by default):
- use_dt_in_intra=False  -> set True to multiply intra recursion by snapshot-weight dt
- include_inflow_in_intra=False -> set True to include inflow (+) and spill (–) in intra recursion

Notes:
- The inter-period recursion follows the original approximation: uses *last hour* of the period.

Original credit to bw0928. Adapted for linopy and pypsa-earth by Filip Matic and Theo Moers
"""

from importlib.util import find_spec
import logging
logger = logging.getLogger(__name__)
import numpy as np
import pandas as pd
import xarray as xr
from linopy import Variable


def _as_xr(obj) -> xr.DataArray:
    """
    Return an xarray.DataArray for linopy Variables / xr objects / numpy.
    Never return an xarray.Dataset.
    """
    # Already a DataArray
    if isinstance(obj, xr.DataArray):
        return obj

    # If it's an xarray.Dataset, collapse to a single DataArray
    if isinstance(obj, xr.Dataset):
        if len(obj.data_vars) == 1:
            # take the single var
            return next(iter(obj.data_vars.values()))
        # otherwise stack and squeeze
        return obj.to_array().squeeze(drop=True)

    # linopy Variable often has .to_xarray()
    if hasattr(obj, "to_xarray"):
        x = obj.to_xarray()
        if isinstance(x, xr.DataArray):
            return x
        if isinstance(x, xr.Dataset):
            if len(x.data_vars) == 1:
                return next(iter(x.data_vars.values()))
            return x.to_array().squeeze(drop=True)

    # Numpy / scalars fallback: only wrap true scalars/ndarrays; otherwise return as-is
    try:
        import numpy as _np
        if _np.isscalar(obj) or isinstance(obj, _np.ndarray):
            return xr.DataArray(obj)
    except Exception:
        pass
    return obj  # don't force-wrap linopy/xarray-like objects


__tcl_version__ = "2025-11-03T"  # bump each edit
logger.info("tcl version: %s", __tcl_version__)


if find_spec("tsam") is None:
    raise ModuleNotFoundError("Optional dependency 'tsam' not found. Install via 'pip install tsam'")
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
    """Try to restore period_id from n.temporal_cluster or from n.meta."""
    if hasattr(n, "temporal_cluster") and isinstance(n.temporal_cluster, dict):
        if "period_id" in n.temporal_cluster:
            return n.temporal_cluster["period_id"]
    ser = None
    if hasattr(n, "meta") and isinstance(n.meta, dict) and "temporal_cluster_period_id" in n.meta:
        try:
            ser = pd.read_json(n.meta["temporal_cluster_period_id"], orient="split", typ="series")
        except Exception as e:
            logger.warning(f"Failed to deserialize period_id from n.meta: {e}")
            ser = None
    if ser is not None:
        if not hasattr(n, "temporal_cluster") or not isinstance(n.temporal_cluster, dict):
            n.temporal_cluster = {}
        n.temporal_cluster["period_id"] = ser
    return ser


def _weight_series_for_stores(n) -> pd.Series:
    """
    Robustly get Δt for storage equations.
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
    ) = aggregate_timeseries(
        timeseries_df,
        n_periods,
        hours,
        extremePeriodMethod,
        normed,
        clusterMethod,
        solver,
        predefClusterOrder,
    )

    # If we fabricated a MultiIndex, drop back to single level to match original behavior
    if was_single:
        map_snapshots_to_periods.index = map_snapshots_to_periods.index.droplevel(0)
        new_snapshots.index = new_snapshots.index.droplevel(0)
        timeseries_clustered.index = timeseries_clustered.index.droplevel(0)

    # Save mapping original -> typical periods (for backwards compatibility)
    n.cluster = map_snapshots_to_periods

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
            numericalTolerance=1e-8,
        )


        clustered = aggregation.createTypicalPeriods()
        if normed:
            clustered = clustered.mul(timeseries_df.loc[year].max())

        mapping = aggregation.indexMatching()
        mapping["day_of_year"] = (mapping.index - mapping.index[0]).days + 1
        weights = aggregation.clusterPeriodNoOccur
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
    
    return map_snapshots_to_periods_all, new_snapshots_all, clustered_all


def overwrite_time_dependent(n, df_t):
    """(Optional) overwrite time-dependent data with clustered values."""
    for component in n.all_components:
        pnl = n.pnl(component)
        for key in pnl.keys():
            if not pnl[key].empty:
                pnl[key] = df_t.reindex(columns=pnl[key].columns)


def temporal_aggregation_storage_constraints(
    n,
    snapshots=None,
    *,
    use_dt_in_intra: bool = False,
    include_inflow_in_intra: bool = False,
):
    """
    Add Linopy storage constraints consistent with typical-period aggregation.
    
    Only applies to:
    - PHS (StorageUnit with carrier='PHS')
    - Battery stores (Store with carrier in ['battery', 'battery storage'])
    - Hydrogen stores (Store with carrier in ['H2', 'H2 Store Tank'])
    
    Hydro reservoirs (StorageUnit with carrier='hydro') and fuel/commodity stores
    (gas, oil, coal, biomass, water tanks, etc.) keep default constraints.

    Defaults match the original Pyomo script exactly:
      - intra recursion uses unit step (no dt scaling)
      - intra recursion does NOT include inflow/spill

    Set use_dt_in_intra=True and/or include_inflow_in_intra=True to enable the
    “improved” variant if desired.
    """
    if snapshots is None:
        snapshots = n.snapshots

    if not hasattr(n, "model") or n.model is None:
        raise AttributeError("Build the Linopy model before adding constraints.")

    # Ensure period_id exists or restore it
    period_id = None
    if hasattr(n, "temporal_cluster") and "period_id" in getattr(n, "temporal_cluster", {}):
        period_id = n.temporal_cluster["period_id"]
    if period_id is None:
        period_id = _restore_period_id(n)

    # Fallback: derive from old Pyomo-style n.cluster structure
    if period_id is None and hasattr(n, "cluster") and isinstance(n.cluster, pd.DataFrame):
        if "TimeStep" in n.cluster.columns:
            try:
                hours = int(n.cluster["TimeStep"].nunique())
                num_periods = len(n.snapshots) // hours
                if num_periods * hours == len(n.snapshots):
                    period_id = pd.Index(
                        np.repeat(np.arange(num_periods), hours), name="period_id"
                    ).to_series(index=n.snapshots)
                    _persist_period_id(n, period_id)
                    logger.info("Derived period_id from n.cluster (Pyomo fallback).")
            except Exception as e:
                logger.warning(f"Could not derive period_id from n.cluster: {e}")

    if period_id is None:
        raise ValueError("Missing period_id mapping; run aggregate_snapshots() from this module.")

    period_id = period_id.reindex(n.snapshots)

    sus = n.storage_units
    if not sus.empty and "carrier" in sus.columns:
        phs_mask = sus["carrier"].str.lower() == "phs"
        sus = sus[phs_mask].copy()
        if not phs_mask.any():
            logger.info("No PHS StorageUnits found; skipping StorageUnit block.")
    
    has_sus = not sus.empty
    if not has_sus:
        logger.info("No PHS StorageUnits; skipping StorageUnit block but will add Store constraints.")

    m = n.model

    if "StorageUnit-soc_intra" in m.variables:
        logger.info("Temporal storage constraints already present; skipping.")
        return

    # unique periods in order of appearance (robust)
    periods = pd.Index(pd.unique(period_id), name="period")

    # constant period length guard
    size_per_period = period_id.value_counts().sort_index()
    if not (size_per_period.nunique() == 1):
        raise ValueError("Typical periods have non-uniform length; expected constant hours.")
    hours = int(size_per_period.iloc[0])

    if len(snapshots) % hours != 0:
        raise ValueError("Snapshot count is not a multiple of hours per typical period.")

    # first/last snapshots of each period (INDEXED BY PERIOD IDS)
    _df = pd.DataFrame({"snapshot": pd.to_datetime(snapshots.values),
                        "period": period_id.values})
    g = _df.groupby("period", sort=False)["snapshot"]
    first_of = g.first()   # index: period ids, values: timestamps
    last_of  = g.last()    # index: period ids, values: timestamps

    # map snapshot -> period (as xr)
    per_of_snap = xr.DataArray(period_id.values, dims=["snapshot"], coords={"snapshot": snapshots})

    snap_idx = pd.Index(pd.to_datetime(snapshots.values))
    per_idx  = pd.Index(periods)

    # last snapshot of each period as positions; then shift to "previous period" positions (cyclic)
    idx_last         = snap_idx.get_indexer(pd.to_datetime(last_of.values))  # length P
    idx_last_prev    = np.roll(idx_last, 1)                                  # shift cyclically

    # map snapshot -> period positions
    idx_per_of_snap  = per_idx.get_indexer(per_of_snap.values)               # length |snapshots|

    # make them xarray indexers with explicit dims
    idx_last_prev_da   = xr.DataArray(idx_last_prev,   dims=["period"])
    idx_per_of_snap_da = xr.DataArray(idx_per_of_snap, dims=["snapshot"])



    if has_sus:

        # Existing PyPSA-Linopy vars
        try:
            E    = m.variables["StorageUnit-state_of_charge"].loc[snapshots, sus.index]
            Pch  = m.variables["StorageUnit-p_store"].loc[snapshots, sus.index]
            Pdis = m.variables["StorageUnit-p_dispatch"].loc[snapshots, sus.index]
        except KeyError as e:
            raise KeyError(
                f"Missing storage model variable: {e}. Ensure storage units exist and model was built."
            )

        # Remove ONLY SOC-related constraints for PHS units; keep the power/energy-balance constraints.
        # Build constraint name patterns for the specific PHS units we're handling
        phs_unit_names = sus.index.tolist()
        try:
            to_drop = []
            for name in list(m.constraints):
                if not name.startswith("StorageUnit-"):
                    continue
                if "energy_balance" in name:
                    continue
                if not ("state_of_charge" in name or "soc" in name.lower()):
                    continue
                # Only drop if it affects our PHS units
                # Check constraint coordinates to see if it includes our PHS units
                try:
                    constr = m.constraints[name]
                    if hasattr(constr, "indexes") and "StorageUnit" in constr.indexes:
                        su_idx = constr.indexes["StorageUnit"]
                        if any(unit in su_idx for unit in phs_unit_names):
                            to_drop.append(name)
                except Exception:
                    raise RuntimeError(f"Could not inspect constraint '{name}' for StorageUnit indexing.")
            
            for name in to_drop:
                m.remove_constraints(name)
            if to_drop:
                logger.info("Removed default storage SOC constraints for PHS: %s", to_drop)
        except Exception as e:
            logger.warning(f"Could not remove default storage SOC constraints: {e}; continuing.")


        # Δt for storage equations (robust)
        dt_ser = _weight_series_for_stores(n).reindex(snapshots).fillna(1.0)
        dt = xr.DataArray(
            dt_ser.astype(float),
            dims=["snapshot"],
            coords={"snapshot": snapshots},
        )
        one = xr.DataArray(1.0, dims=["snapshot"], coords={"snapshot": snapshots})

        # Parameters (fill defaults)
        eta_ch = xr.DataArray(
            sus.efficiency_store.fillna(1.0),
            dims=["StorageUnit"],
            coords={"StorageUnit": sus.index},
        )
        eta_dis = xr.DataArray(
            sus.efficiency_dispatch.fillna(1.0),
            dims=["StorageUnit"],
            coords={"StorageUnit": sus.index},
        )
        standing_loss = xr.DataArray(
            sus.standing_loss.fillna(0.0),
            dims=["StorageUnit"],
            coords={"StorageUnit": sus.index},
        )

        # Standing-loss factor per step:
        # PARITY TWEAK: for parity (default), exponent = 1 per snapshot (unit step).
        # If use_dt_in_intra=True, exponent = dt (improved variant).
        _base = (1.0 - standing_loss).clip(min=1e-12)      # dims: ["StorageUnit"]
        _exp = dt if use_dt_in_intra else one              # dims: ["snapshot"]

        # Broadcast across named dims instead of NumPy indexing
        alpha = (_base.expand_dims(snapshot=snapshots)
                    .transpose("snapshot", "StorageUnit")) ** (
                _exp.expand_dims(StorageUnit=sus.index)
        )


        # Optional inflow/spill (only used if include_inflow_in_intra=True)
        Inflow = 0.0
        if include_inflow_in_intra and hasattr(n, "storage_units_t") and "inflow" in n.storage_units_t:
            inf = n.storage_units_t.inflow.reindex(index=snapshots, columns=sus.index).fillna(0.0)
            Inflow = xr.DataArray(
                inf.values,
                dims=["snapshot", "StorageUnit"],
                coords={"snapshot": snapshots, "StorageUnit": sus.index},
            )
        Spill = 0.0
        if include_inflow_in_intra and "StorageUnit-spill" in m:
            Spill = m.variables["StorageUnit-spill"].loc[snapshots, sus.index]


        # (1) soc_intra variables + recursion (eq. 18)
        soc_intra = m.add_variables(0.0, np.inf, coords=[snapshots, sus.index], name="StorageUnit-soc_intra")

        # Intra recursion for interior snapshots (not first-of-period)
        mask_first = xr.DataArray(np.isin(snapshots, first_of.values),
                                dims=["snapshot"], coords={"snapshot": snapshots})
        interior_mask = (~mask_first)

        # Expand to 2-D so mask dims match constraint dims ('snapshot','StorageUnit')
        mask_first_2d   = mask_first.expand_dims(StorageUnit=sus.index).transpose("snapshot", "StorageUnit")
        interior_mask_2d = interior_mask.expand_dims(StorageUnit=sus.index).transpose("snapshot", "StorageUnit")

        # Previous snapshot via xarray shift (avoids NaT + reindex headaches).
        # First row becomes NaN, but we mask first-of-period rows anyway.
        soc_intra_prev = soc_intra.shift(snapshot=1)

        # Choose unit step (Pyomo parity) or dt-scaled variant
        tstep = dt if use_dt_in_intra else one

        # Prepare broadcast-safe views (xarray-style, no [:, None]/[None, :])
        tstep_2d  = tstep.expand_dims(StorageUnit=sus.index).transpose("snapshot", "StorageUnit")
        eta_ch_2d = eta_ch.expand_dims(snapshot=snapshots).transpose("snapshot", "StorageUnit")
        eta_dis_2d= eta_dis.expand_dims(snapshot=snapshots).transpose("snapshot", "StorageUnit")

        # soc_intra(t) = ...
        rhs_intra = (alpha * soc_intra_prev) + (eta_ch_2d * tstep_2d * Pch) - ((tstep_2d / eta_dis_2d) * Pdis)

        if include_inflow_in_intra and isinstance(Inflow, xr.DataArray):
            rhs_intra = rhs_intra + tstep_2d * Inflow
        if include_inflow_in_intra and isinstance(Spill, xr.DataArray):
            rhs_intra = rhs_intra - tstep_2d * Spill
        m.add_constraints((soc_intra - rhs_intra) == 0,
                        name="StorageUnit-soc_intra-recur",
                        mask=interior_mask_2d)

        m.add_constraints(soc_intra == 0,
                        name="StorageUnit-soc_intra-zero",
                        mask=mask_first_2d)


        # (2) soc_inter variables + recursion (eq. 19)
        soc_inter = m.add_variables(
            0.0, np.inf, coords=[periods, sus.index], name="StorageUnit-soc_inter"
        )

        # efficiencies for inter/intra parts
        eff_inter = (1.0 - standing_loss) ** hours
        eff_intra = (1.0 - standing_loss)

        # Build 2-D broadcast-safe views for ('period','StorageUnit')
        eff_inter_2d = eff_inter.expand_dims(period=periods).transpose("period", "StorageUnit")
        eff_intra_2d = eff_intra.expand_dims(period=periods).transpose("period", "StorageUnit")
        eta_ch_2d_p  = eta_ch.expand_dims(period=periods).transpose("period", "StorageUnit")
        eta_dis_2d_p = eta_dis.expand_dims(period=periods).transpose("period", "StorageUnit")

        # previous period (cyclic)
        soc_inter_prev = soc_inter.roll(period=1)  # period k gets value from k-1, wraps at start

        # soc_inter(k) = ...
        Pch_lhp       = _as_xr(Pch).isel(snapshot=idx_last_prev_da)
        Pdis_lhp      = _as_xr(Pdis).isel(snapshot=idx_last_prev_da)
        soc_intra_lhp = _as_xr(soc_intra).isel(snapshot=idx_last_prev_da)

        rhs_inter = (
            eff_inter_2d * soc_inter_prev
            + eff_intra_2d * soc_intra_lhp
            + eta_ch_2d_p * hours * Pch_lhp
            - (hours / eta_dis_2d_p) * Pdis_lhp
        )
        m.add_constraints((soc_inter - rhs_inter) == 0, name="StorageUnit-soc_inter-recur")

        # (3) total SOC link (eq. 20): E = soc_intra + soc_inter(period_of_snapshot)
        soc_inter_on_snap = _as_xr(soc_inter).isel(period=idx_per_of_snap_da) \
            .rename(period="snapshot").assign_coords(snapshot=snapshots)

        m.add_constraints(E - (soc_intra + soc_inter_on_snap) == 0, name="StorageUnit-SOC-sum")

        try:
            E.set_lower_bounds(0.0)
            Pch.set_lower_bounds(0.0)
            Pdis.set_lower_bounds(0.0)
        except Exception:
            m.add_constraints(E >= 0.0, name="StorageUnit-SOC-lb")
            m.add_constraints(Pch >= 0.0, name="StorageUnit-p_store-lb")
            m.add_constraints(Pdis >= 0.0, name="StorageUnit-p_dispatch-lb")

        e_nom_col_exists = "e_nom" in sus.columns and sus["e_nom"].notna().any()
        if e_nom_col_exists:
            e_nom = xr.DataArray(
                sus.e_nom.fillna(np.inf),
                dims=["StorageUnit"],
                coords={"StorageUnit": sus.index},
            )
            try:
                E.set_upper_bounds(e_nom)
            except Exception:
                e_nom_2d = e_nom.expand_dims(snapshot=snapshots).transpose("snapshot", "StorageUnit")
                m.add_constraints(E <= e_nom_2d, name="StorageUnit-SOC-ub-e_nom")

        else:
            extendable = sus.get("p_nom_extendable", pd.Series(False, index=sus.index)).astype(bool)
            max_hours = sus.get("max_hours", pd.Series(np.inf, index=sus.index)).fillna(np.inf)

            # Fixed portion
            p_nom_fixed = sus.get("p_nom", pd.Series(0.0, index=sus.index)).where(~extendable, other=np.nan)
            e_nom_fixed = (p_nom_fixed * max_hours).fillna(np.inf)
            e_nom_fixed_da = xr.DataArray(e_nom_fixed, dims=["StorageUnit"], coords={"StorageUnit": sus.index})
            e_nom_fixed_2d = e_nom_fixed_da.expand_dims(snapshot=snapshots).transpose("snapshot", "StorageUnit")
            m.add_constraints(E <= e_nom_fixed_2d, name="StorageUnit-SOC-ub-fixed")


            if extendable.any() and "StorageUnit-p_nom" in m:
                idx_var = m.variables["StorageUnit-p_nom"].indexes.get("StorageUnit-ext", pd.Index([], name="StorageUnit-ext"))
                if len(idx_var) > 0:
                    p_nom_var = m.variables["StorageUnit-p_nom"].loc[idx_var].rename({"StorageUnit-ext": "StorageUnit"})

                    common = sus.index.intersection(p_nom_var.indexes["StorageUnit"])
                    if len(common) > 0:
                        ext_mask = xr.DataArray(
                            extendable.astype(float).loc[common],
                            dims=["StorageUnit"],
                            coords={"StorageUnit": common},
                        )
                        mh_da = xr.DataArray(
                            max_hours.astype(float).loc[common],
                            dims=["StorageUnit"],
                            coords={"StorageUnit": common},
                        )
                        # Broadcast without explicit transpose - xarray handles alignment
                        mh_da_bcast = mh_da.loc[common]
                        ext_mask_bcast = ext_mask.loc[common]
                        p_nom_var_bcast = p_nom_var.loc[common]

                        lhs = E.loc[:, common] - (
                            mh_da_bcast * ext_mask_bcast * p_nom_var_bcast
                        )

                        m.add_constraints(lhs <= 0, name="StorageUnit-SOC-ub-extendable")

    logger.info("Added temporal storage constraints for StorageUnits (Kotzur: intra + inter + sum)")
    # Apply identical TSAM-aware SOC to Stores
    _add_store_ts_constraints(n, snapshots, periods, period_id, hours,
        use_dt_in_intra=use_dt_in_intra,
        include_inflow_in_intra=include_inflow_in_intra,
    )


    logger.info("Added temporal storage constraints for StorageUnits and Stores (Kotzur: intra + inter + sum)")

def _add_store_ts_constraints(n, snapshots, periods, period_id, hours, *,
                              use_dt_in_intra=False, include_inflow_in_intra=False):
    m = n.model
    stores = n.stores
    if stores.empty:
        return

    # Filter stores: only battery and hydrogen carriers
    if "carrier" in stores.columns:
        battery_h2_mask = stores["carrier"].isin(['battery', 'battery storage', 'H2 Store Tank'])
        stores = stores[battery_h2_mask].copy()
        if not battery_h2_mask.any():
            logger.info("No battery or hydrogen stores found; skipping Store constraints.")
            return
    else:
        logger.warning("No 'carrier' column in stores; skipping Store constraints.")
        return

    if stores.empty:
        return

    # remove default Store SOC constraints (keep energy_balance)
    to_drop = [name for name in list(m.constraints)
               if name.startswith("Store-")
               and ("state_of_charge" in name or "soc" in name.lower())
               and ("energy_balance" not in name)]
    for name in to_drop:
        m.remove_constraints(name)

    # variables (only for our filtered Battery/H2 stores)
    E    = m.variables["Store-e"].loc[snapshots, stores.index]
    Pnet = m.variables["Store-p"].loc[snapshots, stores.index]
    Ppos = m.add_variables(0.0, np.inf, coords=[snapshots, stores.index], name="Store-p_pos")
    Pneg = m.add_variables(0.0, np.inf, coords=[snapshots, stores.index], name="Store-p_neg")
    m.add_constraints(Pnet - (Ppos - Pneg) == 0, name="Store-p-split")

    # timestep
    dt_ser = _weight_series_for_stores(n).reindex(snapshots).fillna(1.0).astype(float)
    dt     = xr.DataArray(dt_ser, dims=["snapshot"], coords={"snapshot": snapshots})
    one    = xr.DataArray(1.0,   dims=["snapshot"], coords={"snapshot": snapshots})
    tstep  = dt if use_dt_in_intra else one

    # standing loss factor
    standing_loss = xr.DataArray(stores.standing_loss.fillna(0.0),
                                 dims=["Store"], coords={"Store": stores.index})
    base  = (1.0 - standing_loss).clip(min=1e-12)
    alpha = (base.expand_dims(snapshot=snapshots).transpose("snapshot", "Store")
            ) ** (tstep.expand_dims(Store=stores.index))

    # optional inflow/spill
    Inflow = 0.0
    if include_inflow_in_intra and hasattr(n, "stores_t") and "inflow" in n.stores_t:
        inf = n.stores_t.inflow.reindex(index=snapshots, columns=stores.index).fillna(0.0)
        Inflow = xr.DataArray(inf.values, dims=["snapshot","Store"],
                              coords={"snapshot": snapshots, "Store": stores.index})
    Spill = 0.0
    if include_inflow_in_intra and "Store-spill" in m:
        Spill = m.variables["Store-spill"].loc[snapshots, stores.index]

    # per-period first/last
    _df = pd.DataFrame({"snapshot": pd.to_datetime(snapshots.values),
                        "period":   period_id.values})
    first_of = _df.groupby("period", sort=False)["snapshot"].first()
    last_of  = _df.groupby("period", sort=False)["snapshot"].last()

    mask_first = xr.DataArray(np.isin(snapshots, first_of.values),
                              dims=["snapshot"], coords={"snapshot": snapshots})
    interior_mask_2d = (~mask_first).expand_dims(Store=stores.index).transpose("snapshot","Store")
    mask_first_2d    = mask_first.expand_dims(Store=stores.index).transpose("snapshot","Store")

    # (1) soc_intra on snapshots
    soc_intra = m.add_variables(0.0, np.inf, coords=[snapshots, stores.index], name="Store-soc_intra")
    soc_intra_prev = soc_intra.shift(snapshot=1)
    t2d = tstep.expand_dims(Store=stores.index).transpose("snapshot","Store")
    rhs_intra = (alpha * soc_intra_prev) + t2d * (Pneg - Ppos)
    if isinstance(Inflow, xr.DataArray): rhs_intra = rhs_intra + t2d * Inflow
    if isinstance(Spill,  xr.DataArray): rhs_intra = rhs_intra - t2d * Spill
    m.add_constraints(soc_intra - rhs_intra == 0, name="Store-soc_intra-recur", mask=interior_mask_2d)
    m.add_constraints(soc_intra == 0,            name="Store-soc_intra-zero",  mask=mask_first_2d)

    # (2) soc_inter on periods (cyclic, last-hour approx)
    per_idx = pd.Index(pd.unique(period_id), name="period")
    soc_inter = m.add_variables(0.0, np.inf, coords=[per_idx, stores.index], name="Store-soc_inter")

    eff_inter = (1.0 - standing_loss) ** hours
    eff_intra = (1.0 - standing_loss)
    eff_inter_2d = eff_inter.expand_dims(period=per_idx).transpose("period","Store")
    eff_intra_2d = eff_intra.expand_dims(period=per_idx).transpose("period","Store")

    snap_idx = pd.Index(pd.to_datetime(snapshots.values))
    idx_last = snap_idx.get_indexer(pd.to_datetime(last_of.values))
    idx_last_prev = np.roll(idx_last, 1)
    idx_last_prev_da = xr.DataArray(idx_last_prev, dims=["period"])
    
    
    Ppos_lhp_temp      = _as_xr(Ppos).isel(snapshot=idx_last_prev_da)
    Pneg_lhp_temp      = _as_xr(Pneg).isel(snapshot=idx_last_prev_da)
    soc_intra_lhp_temp = _as_xr(soc_intra).isel(snapshot=idx_last_prev_da)
    
    # Fix coordinates: reset 'snapshot' and add proper 'period' coordinate to avoid dimension conflicts
    ppos_data = Ppos_lhp_temp._data.reset_coords('snapshot', drop=True).assign_coords(period=np.arange(len(per_idx)))
    pneg_data = Pneg_lhp_temp._data.reset_coords('snapshot', drop=True).assign_coords(period=np.arange(len(per_idx)))
    soc_intra_data = soc_intra_lhp_temp._data.reset_coords('snapshot', drop=True).assign_coords(period=np.arange(len(per_idx)))
    
    Ppos_lhp = Variable(ppos_data, Ppos_lhp_temp.model, Ppos_lhp_temp.name)
    Pneg_lhp = Variable(pneg_data, Pneg_lhp_temp.model, Pneg_lhp_temp.name)
    soc_intra_lhp = Variable(soc_intra_data, soc_intra_lhp_temp.model, soc_intra_lhp_temp.name)

    soc_inter_prev = soc_inter.roll(period=1)

    # Match StorageUnit pattern: multiply each term separately to avoid bare subtraction
    rhs_inter = (
        eff_inter_2d * soc_inter_prev
        + eff_intra_2d * soc_intra_lhp
        + float(hours) * Pneg_lhp
        - float(hours) * Ppos_lhp
    )
    m.add_constraints(soc_inter - rhs_inter == 0, name="Store-soc_inter-recur")

    # (3) total SOC on snapshots
    idx_per_of_snap_da = xr.DataArray(per_idx.get_indexer(period_id.values), dims=["snapshot"])
    soc_inter_on_snap = _as_xr(soc_inter).isel(period=idx_per_of_snap_da) \
        .rename(period="snapshot").assign_coords(snapshot=snapshots)
    m.add_constraints(E - (soc_intra + soc_inter_on_snap) == 0, name="Store-SOC-sum")

    # bounds
    try:
        E.set_lower_bounds(0.0); Ppos.set_lower_bounds(0.0); Pneg.set_lower_bounds(0.0)
    except Exception:
        m.add_constraints(E   >= 0.0, name="Store-SOC-lb")
        m.add_constraints(Ppos>= 0.0, name="Store-p_pos-lb")
        m.add_constraints(Pneg>= 0.0, name="Store-p_neg-lb")

    # Upper bounds: use e_nom variable for extendable, parameter for fixed
    extendable = stores.get("e_nom_extendable", pd.Series(False, index=stores.index)).astype(bool)
    
    # For extendable stores, use the e_nom optimization variable
    if extendable.any() and "Store-e_nom" in m.variables:
        idx_var = m.variables["Store-e_nom"].indexes.get("Store-ext", pd.Index([], name="Store-ext"))
        if len(idx_var) > 0:
            e_nom_var = m.variables["Store-e_nom"].loc[idx_var].rename({"Store-ext":"Store"})
            common_ext = stores.index.intersection(e_nom_var.indexes["Store"])
            if len(common_ext) > 0:
                # Broadcast without explicit transpose - xarray handles alignment
                m.add_constraints(E.loc[:, common_ext] <= e_nom_var.loc[common_ext],
                                  name="Store-SOC-ub-extendable")
    
    # For non-extendable stores, use the e_nom parameter
    non_extendable = ~extendable
    if non_extendable.any():
        fixed_stores = stores.index[non_extendable]
        if "e_nom" in stores.columns:
            e_nom_fixed = stores.loc[fixed_stores, "e_nom"].fillna(np.inf)
        else:
            # Fallback: calculate from p_nom * max_hours
            p_nom = stores.get("p_nom", pd.Series(0.0, index=stores.index)).loc[fixed_stores]
            max_hours = stores.get("max_hours", pd.Series(np.inf, index=stores.index)).loc[fixed_stores].fillna(np.inf)
            e_nom_fixed = (p_nom * max_hours).fillna(np.inf)
        
        e_nom_fixed_da = xr.DataArray(e_nom_fixed, dims=["Store"], coords={"Store": fixed_stores})
        # Broadcast without explicit transpose - xarray handles alignment
        m.add_constraints(E.loc[:, fixed_stores] <= e_nom_fixed_da,
                          name="Store-SOC-ub-fixed")
    
    logger.info("Added temporal storage constraints for Stores (Kotzur: intra + inter + sum)")

logger.info("Linopy temporal clustering module ready (Pyomo-parity defaults).")