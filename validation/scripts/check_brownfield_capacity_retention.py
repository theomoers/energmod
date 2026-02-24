#!/usr/bin/env python3
"""
Check brownfield capacity retention across a myopic handoff (e.g. solved 2020 -> 2025 brownfield baseline).

The checker compares nominal capacities by asset index, then emits machine-readable diff artifacts with
classification labels for expected vs unexpected losses/gains.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa


LOGGER = logging.getLogger(__name__)


COMPONENT_SPECS = {
    "Generator": {
        "list_name": "generators",
        "extendable_col": "p_nom_extendable",
        "nom_col": "p_nom",
        "nom_opt_col": "p_nom_opt",
        "nom_min_col": "p_nom_min",
        "bus_cols": ["bus"],
        "unit": "MW",
    },
    "Link": {
        "list_name": "links",
        "extendable_col": "p_nom_extendable",
        "nom_col": "p_nom",
        "nom_opt_col": "p_nom_opt",
        "nom_min_col": "p_nom_min",
        "bus_cols": ["bus0", "bus1", "bus2", "bus3", "bus4"],
        "unit": "MW",
    },
    "StorageUnit": {
        "list_name": "storage_units",
        "extendable_col": "p_nom_extendable",
        "nom_col": "p_nom",
        "nom_opt_col": "p_nom_opt",
        "nom_min_col": "p_nom_min",
        "bus_cols": ["bus"],
        "unit": "MW",
    },
    "Store": {
        "list_name": "stores",
        "extendable_col": "e_nom_extendable",
        "nom_col": "e_nom",
        "nom_opt_col": "e_nom_opt",
        "nom_min_col": "e_nom_min",
        "bus_cols": ["bus"],
        "unit": "MWh",
    },
    "Line": {
        "list_name": "lines",
        "extendable_col": "s_nom_extendable",
        "nom_col": "s_nom",
        "nom_opt_col": "s_nom_opt",
        "nom_min_col": "s_nom_min",
        "bus_cols": ["bus0", "bus1"],
        "unit": "MW",
    },
    "Transformer": {
        "list_name": "transformers",
        "extendable_col": "s_nom_extendable",
        "nom_col": "s_nom",
        "nom_opt_col": "s_nom_opt",
        "nom_min_col": "s_nom_min",
        "bus_cols": ["bus0", "bus1"],
        "unit": "MW",
    },
}

DIFF_COLUMNS = [
    "from_year",
    "to_year",
    "component",
    "asset",
    "carrier",
    "country",
    "country_context",
    "capacity_unit",
    "previous_solved_capacity",
    "target_brownfield_retained_capacity",
    "delta_capacity",
    "abs_delta_capacity",
    "classification",
    "merge_status",
    "prev_build_year",
    "prev_lifetime",
    "prev_lifetime_is_inf",
    "prev_extendable",
    "curr_build_year",
    "curr_lifetime",
    "curr_lifetime_is_inf",
    "curr_extendable",
    "curr_nominal",
    "curr_nominal_min",
    "curr_nominal_opt",
    "expected_retirement_before_to_year",
    "notes",
]


def _bus_country_lookup(n: pypsa.Network) -> pd.Series:
    country = (
        n.buses["country"].replace("", np.nan)
        if "country" in n.buses.columns
        else pd.Series("", index=n.buses.index, dtype=object)
    )
    location_series = (
        n.buses["location"]
        if "location" in n.buses.columns
        else pd.Series("", index=n.buses.index, dtype=object)
    )
    location_iso2 = location_series.astype(str).str.extract(r"^([A-Z]{2})\b")[0]
    index_iso2 = n.buses.index.astype(str).to_series(index=n.buses.index).str.extract(r"^([A-Z]{2})\b")[0]
    return country.fillna(location_iso2).fillna(index_iso2).fillna("")


def _country_context(df: pd.DataFrame, bus_cols: list[str], bus_country: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    mapped_cols = []
    for col in bus_cols:
        if col in df.columns:
            mapped_cols.append(df[col].map(bus_country).fillna("").astype(str).rename(col))

    if not mapped_cols:
        out["country"] = ""
        out["country_context"] = ""
        return out

    mapped = pd.concat(mapped_cols, axis=1)
    out["country"] = mapped.replace("", np.nan).bfill(axis=1).iloc[:, 0].fillna("")

    def _ctx(row: pd.Series) -> str:
        vals = sorted({v for v in row.tolist() if isinstance(v, str) and v})
        return "|".join(vals)

    out["country_context"] = mapped.apply(_ctx, axis=1)
    return out


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column].fillna(False).astype(bool)
    return pd.Series(False, index=df.index, dtype=bool)


def _extract_capacity_snapshot(n: pypsa.Network, mode: str) -> pd.DataFrame:
    """
    mode='previous' uses solved capacity (nom_opt fallback nom).
    mode='current'  uses retained brownfield baseline capacity (max(nom, nom_min)).
    """
    if mode not in {"previous", "current"}:
        raise ValueError(f"Unsupported mode: {mode}")

    bus_country = _bus_country_lookup(n)
    parts: list[pd.DataFrame] = []

    for component_name, spec in COMPONENT_SPECS.items():
        if not hasattr(n, spec["list_name"]):
            continue
        df = getattr(n, spec["list_name"])
        if df.empty:
            continue

        nominal = _numeric_series(df, spec["nom_col"])
        nominal_opt = _numeric_series(df, spec["nom_opt_col"])
        nominal_min = _numeric_series(df, spec["nom_min_col"])
        extendable = _bool_series(df, spec["extendable_col"])
        build_year = _numeric_series(df, "build_year")
        lifetime = _numeric_series(df, "lifetime")

        if mode == "previous":
            comparison_capacity = nominal_opt.fillna(nominal).fillna(0.0)
        else:
            comparison_capacity = nominal.fillna(0.0)
            if spec["nom_min_col"] in df.columns:
                comparison_capacity = pd.Series(
                    np.maximum(comparison_capacity.to_numpy(), nominal_min.fillna(0.0).to_numpy()),
                    index=df.index,
                )

        # Keep rows with any capacity signal, or extendable rows (to preserve reset diagnostics context).
        keep_mask = (
            comparison_capacity.fillna(0.0).abs() > 0.0
        ) | (nominal.fillna(0.0).abs() > 0.0) | (nominal_min.fillna(0.0).abs() > 0.0) | (
            nominal_opt.fillna(0.0).abs() > 0.0
        ) | extendable
        if not keep_mask.any():
            continue

        sub = df.loc[keep_mask].copy()
        ctx = _country_context(sub, spec["bus_cols"], bus_country)

        carrier = sub["carrier"].astype(str) if "carrier" in sub.columns else pd.Series("", index=sub.index)
        if component_name == "Line":
            carrier = carrier.replace("", "AC")
        elif component_name == "Transformer":
            carrier = carrier.replace("", "transformer")

        # Exclude non-physical load-shedding / pseudo-load generators from physical capacity retention diagnostics.
        if component_name == "Generator":
            non_physical = carrier.fillna("").eq("Load")
            if non_physical.any():
                sub = sub.loc[~non_physical].copy()
                carrier = carrier.loc[sub.index]
                if sub.empty:
                    continue

        out = pd.DataFrame(index=sub.index)
        out["component"] = component_name
        out["asset"] = sub.index.astype(str)
        out["carrier"] = carrier.fillna("").astype(str)
        out["country"] = ctx["country"].fillna("")
        out["country_context"] = ctx["country_context"].fillna("")
        out["capacity_unit"] = spec["unit"]
        out["comparison_capacity"] = comparison_capacity.loc[sub.index].fillna(0.0)
        out["nominal"] = nominal.loc[sub.index]
        out["nominal_opt"] = nominal_opt.loc[sub.index]
        out["nominal_min"] = nominal_min.loc[sub.index]
        out["extendable"] = extendable.loc[sub.index].astype(bool)
        out["build_year"] = build_year.loc[sub.index]
        out["lifetime"] = lifetime.loc[sub.index]
        out["lifetime_is_inf"] = np.isinf(lifetime.loc[sub.index].fillna(np.nan))
        parts.append(out.reset_index(drop=True))

    if not parts:
        return pd.DataFrame(
            columns=[
                "component",
                "asset",
                "carrier",
                "country",
                "country_context",
                "capacity_unit",
                "comparison_capacity",
                "nominal",
                "nominal_opt",
                "nominal_min",
                "extendable",
                "build_year",
                "lifetime",
                "lifetime_is_inf",
            ]
        )
    return pd.concat(parts, ignore_index=True)


def _is_chp_heat_asset(asset_name: str) -> bool:
    s = str(asset_name)
    return ("urban central" in s) and ("CHP" in s) and ("heat" in s)


def _expected_retirement(build_year, lifetime, to_year: int) -> bool:
    if pd.isna(build_year) or pd.isna(lifetime):
        return False
    if np.isinf(float(lifetime)):
        return False
    return float(build_year) + float(lifetime) < float(to_year)


def _classify_diff(
    row: pd.Series,
    *,
    to_year: int,
    threshold_capacity: float,
    epsilon: float,
    rtol: float,
) -> tuple[str, str]:
    delta = float(row["delta_capacity"])
    prev_capacity = float(row["previous_solved_capacity"])
    curr_capacity = float(row["target_brownfield_retained_capacity"])
    scale = max(abs(prev_capacity), abs(curr_capacity), 1.0)
    tol = max(float(epsilon), float(rtol) * scale)

    if abs(delta) <= tol:
        return "retained_match", ""

    if delta > tol:
        return "unexpected_gain", "Target brownfield baseline exceeds previous solved capacity."

    # delta < 0: loss
    if bool(row["expected_retirement_before_to_year"]):
        return "expected_retirement", "Removed because build_year + lifetime < target year."

    curr_build_year = row.get("curr_build_year")
    prev_build_year = row.get("prev_build_year")
    curr_extendable = bool(row.get("curr_extendable", False))
    if curr_capacity <= epsilon and curr_extendable:
        if pd.notna(curr_build_year) and np.isclose(float(curr_build_year), float(to_year), rtol=0.0, atol=0.0):
            return "expected_reset_new_year", "Current-year asset reset to zero/extendable in brownfield baseline."
    if curr_capacity <= epsilon and pd.notna(prev_build_year):
        if float(prev_build_year) >= float(to_year):
            return "expected_reset_new_year", "Previous solved asset belongs to target/newer year and is not retained as brownfield."

    if bool(row.get("prev_lifetime_is_inf", False)):
        return "expected_filter", "Infinite-lifetime tracking/accounting asset removed before import."

    if threshold_capacity > 0 and prev_capacity <= threshold_capacity + tol:
        return "expected_filter", "Filtered by brownfield minimum-capacity threshold."

    if _is_chp_heat_asset(row.get("asset", "")) and curr_capacity <= epsilon:
        return "expected_filter", "Likely CHP heat threshold filtering in brownfield import."

    return "unexpected_loss", "Capacity loss is not explained by retirement/reset/filter heuristics."


def _build_retention_diff(
    prev_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    *,
    from_year: int,
    to_year: int,
    threshold_capacity: float,
    epsilon: float,
    rtol: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = prev_df.merge(
        curr_df,
        on=["component", "asset"],
        how="outer",
        suffixes=("_prev", "_curr"),
        indicator=True,
    )

    merged["previous_solved_capacity"] = pd.to_numeric(
        merged.get("comparison_capacity_prev"), errors="coerce"
    ).fillna(0.0)
    merged["target_brownfield_retained_capacity"] = pd.to_numeric(
        merged.get("comparison_capacity_curr"), errors="coerce"
    ).fillna(0.0)
    merged["delta_capacity"] = (
        merged["target_brownfield_retained_capacity"] - merged["previous_solved_capacity"]
    )
    merged["abs_delta_capacity"] = merged["delta_capacity"].abs()
    merged["comparison_tolerance"] = np.maximum(
        float(epsilon),
        float(rtol)
        * np.maximum(
            np.maximum(merged["previous_solved_capacity"].abs(), merged["target_brownfield_retained_capacity"].abs()),
            1.0,
        ),
    )

    merged["carrier"] = (
        merged.get("carrier_curr", "").fillna("").astype(str)
        if "carrier_curr" in merged.columns
        else ""
    )
    if "carrier_prev" in merged.columns:
        merged["carrier"] = merged["carrier"].where(
            merged["carrier"].ne(""), merged["carrier_prev"].fillna("").astype(str)
        )

    merged["country"] = (
        merged.get("country_curr", "").fillna("").astype(str)
        if "country_curr" in merged.columns
        else ""
    )
    if "country_prev" in merged.columns:
        merged["country"] = merged["country"].where(
            merged["country"].ne(""), merged["country_prev"].fillna("").astype(str)
        )

    merged["country_context"] = (
        merged.get("country_context_curr", "").fillna("").astype(str)
        if "country_context_curr" in merged.columns
        else ""
    )
    if "country_context_prev" in merged.columns:
        merged["country_context"] = merged["country_context"].where(
            merged["country_context"].ne(""),
            merged["country_context_prev"].fillna("").astype(str),
        )

    merged["capacity_unit"] = (
        merged.get("capacity_unit_curr", "").fillna("").astype(str)
        if "capacity_unit_curr" in merged.columns
        else ""
    )
    if "capacity_unit_prev" in merged.columns:
        merged["capacity_unit"] = merged["capacity_unit"].where(
            merged["capacity_unit"].ne(""),
            merged["capacity_unit_prev"].fillna("").astype(str),
        )

    # Rename frequently-used context columns into flat names.
    rename_pairs = {
        "build_year_prev": "prev_build_year",
        "lifetime_prev": "prev_lifetime",
        "lifetime_is_inf_prev": "prev_lifetime_is_inf",
        "extendable_prev": "prev_extendable",
        "build_year_curr": "curr_build_year",
        "lifetime_curr": "curr_lifetime",
        "lifetime_is_inf_curr": "curr_lifetime_is_inf",
        "extendable_curr": "curr_extendable",
        "nominal_curr": "curr_nominal",
        "nominal_min_curr": "curr_nominal_min",
        "nominal_opt_curr": "curr_nominal_opt",
    }
    for src, dst in rename_pairs.items():
        if src in merged.columns:
            merged[dst] = merged[src]
        else:
            merged[dst] = np.nan if "extendable" not in dst else False

    merged["prev_lifetime_is_inf"] = merged["prev_lifetime_is_inf"].fillna(False).astype(bool)
    merged["curr_lifetime_is_inf"] = merged["curr_lifetime_is_inf"].fillna(False).astype(bool)
    merged["prev_extendable"] = merged["prev_extendable"].fillna(False).astype(bool)
    merged["curr_extendable"] = merged["curr_extendable"].fillna(False).astype(bool)
    merged["expected_retirement_before_to_year"] = merged.apply(
        lambda r: _expected_retirement(r.get("prev_build_year"), r.get("prev_lifetime"), to_year),
        axis=1,
    )

    merged["merge_status"] = merged["_merge"].astype(str).map(
        {"left_only": "missing_in_target", "right_only": "new_in_target", "both": "present_in_both"}
    ).fillna("unknown")

    diff_rows = merged.loc[merged["abs_delta_capacity"] > merged["comparison_tolerance"]].copy()
    if diff_rows.empty:
        diff_rows = pd.DataFrame(columns=DIFF_COLUMNS)
        summary = pd.DataFrame(
            columns=[
                "from_year",
                "to_year",
                "classification",
                "component",
                "country",
                "carrier",
                "prev_build_year",
                "prev_lifetime",
                "capacity_unit",
                "assets",
                "previous_solved_capacity",
                "target_brownfield_retained_capacity",
                "delta_capacity",
                "abs_delta_capacity",
            ]
        )
        status = pd.DataFrame(
            [
                {
                    "from_year": from_year,
                    "to_year": to_year,
                    "threshold_capacity": threshold_capacity,
                    "epsilon": epsilon,
                    "rtol": rtol,
                    "diff_assets": 0,
                    "unexpected_loss_assets": 0,
                    "unexpected_gain_assets": 0,
                    "status": "pass",
                }
            ]
        )
        return diff_rows, summary, status

    labels = diff_rows.apply(
        lambda r: _classify_diff(
            r,
            to_year=to_year,
            threshold_capacity=threshold_capacity,
            epsilon=epsilon,
            rtol=rtol,
        ),
        axis=1,
        result_type="expand",
    )
    diff_rows["classification"] = labels[0]
    diff_rows["notes"] = labels[1]
    diff_rows["from_year"] = int(from_year)
    diff_rows["to_year"] = int(to_year)

    out = diff_rows.loc[:, [c for c in DIFF_COLUMNS if c in diff_rows.columns]].copy()
    for col in DIFF_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out = out.loc[:, DIFF_COLUMNS]
    out = out.sort_values(
        ["classification", "component", "carrier", "country", "abs_delta_capacity"],
        ascending=[True, True, True, True, False],
        ignore_index=True,
    )

    summary = (
        out.groupby(
            [
                "from_year",
                "to_year",
                "classification",
                "component",
                "country",
                "carrier",
                "prev_build_year",
                "prev_lifetime",
                "capacity_unit",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            assets=("asset", "count"),
            previous_solved_capacity=("previous_solved_capacity", "sum"),
            target_brownfield_retained_capacity=("target_brownfield_retained_capacity", "sum"),
            delta_capacity=("delta_capacity", "sum"),
            abs_delta_capacity=("abs_delta_capacity", "sum"),
        )
        .sort_values(["classification", "abs_delta_capacity"], ascending=[True, False], ignore_index=True)
    )

    status = pd.DataFrame(
        [
            {
                "from_year": from_year,
                "to_year": to_year,
                "threshold_capacity": threshold_capacity,
                "epsilon": epsilon,
                "rtol": rtol,
                "diff_assets": int(len(out)),
                "unexpected_loss_assets": int((out["classification"] == "unexpected_loss").sum()),
                "unexpected_gain_assets": int((out["classification"] == "unexpected_gain").sum()),
                "expected_retirement_assets": int((out["classification"] == "expected_retirement").sum()),
                "expected_reset_new_year_assets": int((out["classification"] == "expected_reset_new_year").sum()),
                "expected_filter_assets": int((out["classification"] == "expected_filter").sum()),
                "unexpected_loss_capacity": float(
                    out.loc[out["classification"] == "unexpected_loss", "abs_delta_capacity"].sum()
                ),
                "unexpected_gain_capacity": float(
                    out.loc[out["classification"] == "unexpected_gain", "abs_delta_capacity"].sum()
                ),
                "status": (
                    "fail"
                    if ((out["classification"] == "unexpected_loss").any() or (out["classification"] == "unexpected_gain").any())
                    else "pass"
                ),
            }
        ]
    )
    return out, summary, status


def _round_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    float_cols = out.select_dtypes(include=["float32", "float64"]).columns
    if len(float_cols):
        out.loc[:, float_cols] = out.loc[:, float_cols].round(6)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare solved previous-horizon capacities against target brownfield baseline retention."
    )
    parser.add_argument("--previous-network", type=Path, required=True, help="Solved previous-horizon network (e.g. 2020).")
    parser.add_argument("--brownfield-network", type=Path, required=True, help="Target-year network after brownfield import/pre-solve (e.g. 2025).")
    parser.add_argument("--from-year", type=int, required=True, help="Previous solved horizon year.")
    parser.add_argument("--to-year", type=int, required=True, help="Target brownfield horizon year.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation/results"),
        help="Directory for diff outputs.",
    )
    parser.add_argument(
        "--threshold-capacity",
        type=float,
        default=0.0,
        help="Brownfield threshold capacity used for expected_filter classification (MW/MWh nominal units by component).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="Absolute tolerance for no-diff classification.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for no-diff classification (suppresses tiny numeric drift).",
    )
    parser.add_argument(
        "--fail-on-unexpected",
        action="store_true",
        help="Exit non-zero if unexpected_loss or unexpected_gain entries exist.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading previous solved network: %s", args.previous_network)
    prev = pypsa.Network(args.previous_network)
    LOGGER.info("Loading target brownfield network: %s", args.brownfield_network)
    curr = pypsa.Network(args.brownfield_network)

    prev_df = _extract_capacity_snapshot(prev, mode="previous")
    curr_df = _extract_capacity_snapshot(curr, mode="current")

    diff_df, summary_df, status_df = _build_retention_diff(
        prev_df,
        curr_df,
        from_year=args.from_year,
        to_year=args.to_year,
        threshold_capacity=float(args.threshold_capacity),
        epsilon=float(args.epsilon),
        rtol=float(args.rtol),
    )

    diff_path = args.output_dir / "brownfield_capacity_retention_diff.csv"
    summary_path = args.output_dir / "brownfield_capacity_retention_diff_summary.csv"
    status_path = args.output_dir / "brownfield_capacity_retention_status.csv"
    _round_for_csv(diff_df).to_csv(diff_path, index=False)
    _round_for_csv(summary_df).to_csv(summary_path, index=False)
    _round_for_csv(status_df).to_csv(status_path, index=False)

    status = status_df.iloc[0].to_dict()
    LOGGER.info("Wrote brownfield retention diff artifact: %s", diff_path.resolve())
    LOGGER.info("Wrote brownfield retention summary: %s", summary_path.resolve())
    LOGGER.info("Wrote brownfield retention status: %s", status_path.resolve())
    LOGGER.info(
        "Diff assets=%d, unexpected_loss=%d, unexpected_gain=%d, status=%s",
        int(status.get("diff_assets", 0)),
        int(status.get("unexpected_loss_assets", 0)),
        int(status.get("unexpected_gain_assets", 0)),
        status.get("status", "unknown"),
    )

    if not diff_df.empty:
        unexpected_preview = diff_df.loc[
            diff_df["classification"].isin(["unexpected_loss", "unexpected_gain"])
        ].head(15)
        if not unexpected_preview.empty:
            LOGGER.info(
                "Top unexpected retention diffs:\n%s",
                _round_for_csv(unexpected_preview).to_string(index=False),
            )
        LOGGER.info(
            "Top retention diffs:\n%s",
            _round_for_csv(diff_df.head(15)).to_string(index=False),
        )

    if args.fail_on_unexpected and str(status.get("status", "pass")).lower() != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
