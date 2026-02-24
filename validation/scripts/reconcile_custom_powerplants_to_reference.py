#!/usr/bin/env python3
"""
Reconcile custom powerplant capacities to external reference totals for a base year.

Default reference mapping:
- gas, coal: GEM cleaned reference (country, validation_tech, year, reference_mw)
- oil, nuclear: IRENA 2020 technology totals by country

The script scales plant-level capacities within each (country, technology group) so that
the active-year aggregate matches the selected reference aggregate.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]

IRENA_TECH_TO_VALIDATION = {
    "Oil": "oil",
    "Nuclear": "nuclear",
}

FUELTYPE_TO_VALIDATION = {
    "CCGT": "gas",
    "OCGT": "gas",
    "Natural Gas": "gas",
    "Hard Coal": "coal",
    "Lignite": "coal",
    "Coal": "coal",
    "Oil": "oil",
    "Nuclear": "nuclear",
}

TARGET_TECHS = {"gas", "coal", "oil", "nuclear"}


def _active_mask(df: pd.DataFrame, year: int) -> pd.Series:
    date_in = pd.to_numeric(df["DateIn"], errors="coerce")
    date_out = pd.to_numeric(df["DateOut"], errors="coerce")
    return (date_in.fillna(-np.inf) <= year) & (date_out.isna() | (date_out >= year))


def _load_custom(custom_csv: Path, year: int) -> pd.DataFrame:
    custom = pd.read_csv(custom_csv, index_col=0, encoding="utf-8-sig")
    custom["Country"] = custom["Country"].astype(str).str.upper().str.strip()
    custom["Fueltype"] = custom["Fueltype"].astype(str).str.strip()
    custom["Capacity"] = pd.to_numeric(custom["Capacity"], errors="coerce").fillna(0.0)
    custom["validation_tech"] = custom["Fueltype"].map(FUELTYPE_TO_VALIDATION)
    custom["is_active_year"] = _active_mask(custom, year)
    return custom


def _load_gem_clean(gem_clean_csv: Path, year: int) -> pd.DataFrame:
    gem = pd.read_csv(gem_clean_csv, encoding="utf-8-sig")
    required = {"country", "validation_tech", "reference_mw"}
    missing = required - set(gem.columns)
    if missing:
        raise ValueError(f"Missing required GEM-clean columns: {sorted(missing)}")
    if "year" in gem.columns:
        gem = gem.loc[pd.to_numeric(gem["year"], errors="coerce") == year].copy()
    gem["country"] = gem["country"].astype(str).str.upper().str.strip()
    gem["validation_tech"] = gem["validation_tech"].astype(str).str.strip()
    gem["reference_mw"] = pd.to_numeric(gem["reference_mw"], errors="coerce").fillna(0.0)
    gem = gem.loc[gem["validation_tech"].isin(["gas", "coal"])].copy()
    return gem.groupby(["country", "validation_tech"], as_index=False)["reference_mw"].sum()


def _load_irena_subset(irena_csv: Path, year: int) -> pd.DataFrame:
    irena = pd.read_csv(irena_csv, encoding="utf-8-sig")
    year_col = str(year)
    if year_col not in irena.columns:
        raise ValueError(f"Year column {year_col} not found in {irena_csv}")
    ref = irena[["Country", "Technology", year_col]].copy()
    ref = ref.rename(columns={"Country": "country", year_col: "reference_mw"})
    ref["country"] = ref["country"].astype(str).str.upper().str.strip()
    ref["validation_tech"] = ref["Technology"].map(IRENA_TECH_TO_VALIDATION)
    ref = ref.loc[ref["validation_tech"].notna()].copy()
    ref["reference_mw"] = pd.to_numeric(ref["reference_mw"], errors="coerce").fillna(0.0)
    return ref.groupby(["country", "validation_tech"], as_index=False)["reference_mw"].sum()


def _build_reference(gem_clean_csv: Path, irena_csv: Path, year: int) -> pd.DataFrame:
    gem = _load_gem_clean(gem_clean_csv, year)
    irena = _load_irena_subset(irena_csv, year)
    ref = pd.concat([gem, irena], ignore_index=True)
    ref = ref.groupby(["country", "validation_tech"], as_index=False)["reference_mw"].sum()
    return ref.loc[ref["validation_tech"].isin(TARGET_TECHS)].copy()


def _calculate_scales(custom: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    active = custom.loc[
        custom["is_active_year"] & custom["validation_tech"].isin(TARGET_TECHS)
    ].copy()
    current = (
        active.groupby(["Country", "validation_tech"], as_index=False)["Capacity"]
        .sum()
        .rename(columns={"Country": "country", "Capacity": "current_mw"})
    )
    merged = current.merge(reference, on=["country", "validation_tech"], how="outer")
    merged["current_mw"] = merged["current_mw"].fillna(0.0)
    merged["reference_mw"] = merged["reference_mw"].fillna(0.0)
    merged["scale_factor"] = np.where(
        merged["current_mw"] > 0.0,
        merged["reference_mw"] / merged["current_mw"],
        np.nan,
    )
    merged["scale_factor"] = merged["scale_factor"].replace([np.inf, -np.inf], np.nan)
    merged["scale_factor"] = merged["scale_factor"].fillna(1.0)
    return merged


def _apply_scales(custom: pd.DataFrame, scales: pd.DataFrame) -> pd.DataFrame:
    scaled = custom.copy()
    scale_lookup = scales.set_index(["country", "validation_tech"])["scale_factor"].to_dict()

    key_country = scaled["Country"].astype(str).str.upper().str.strip()
    key_tech = scaled["validation_tech"].astype(str)
    factor = [scale_lookup.get((c, t), 1.0) for c, t in zip(key_country, key_tech)]
    factor = pd.Series(factor, index=scaled.index, dtype=float)

    # Only scale plants active in the target year and in targeted tech groups.
    do_scale = scaled["is_active_year"] & scaled["validation_tech"].isin(TARGET_TECHS)
    scaled.loc[do_scale, "Capacity"] = scaled.loc[do_scale, "Capacity"] * factor.loc[do_scale]
    return scaled


def _summary_table(custom_before: pd.DataFrame, custom_after: pd.DataFrame, year: int) -> pd.DataFrame:
    def agg(df: pd.DataFrame, label: str) -> pd.DataFrame:
        x = df.loc[df["is_active_year"] & df["validation_tech"].isin(TARGET_TECHS)].copy()
        x = (
            x.groupby(["Country", "validation_tech"], as_index=False)["Capacity"]
            .sum()
            .rename(columns={"Country": "country", "Capacity": f"{label}_mw"})
        )
        return x

    before = agg(custom_before, "before")
    after = agg(custom_after, "after")
    out = before.merge(after, on=["country", "validation_tech"], how="outer").fillna(0.0)
    out["delta_mw"] = out["after_mw"] - out["before_mw"]
    out["year"] = year
    return out.sort_values(["validation_tech", "country"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scale custom powerplants to GEM/IRENA totals.")
    parser.add_argument(
        "--custom-csv",
        type=Path,
        default=REPO_ROOT / "data" / "custom_powerplants.csv",
        help="Input custom powerplants CSV",
    )
    parser.add_argument(
        "--gem-clean-csv",
        type=Path,
        default=REPO_ROOT / "validation" / "data" / "gem_capacity_reference_2020.csv",
        help="Cleaned GEM capacity reference CSV (gas, coal)",
    )
    parser.add_argument(
        "--irena-csv",
        type=Path,
        default=REPO_ROOT / "validation" / "data" / "irena_capacity_by_technology.csv",
        help="IRENA capacity CSV (used for oil, nuclear)",
    )
    parser.add_argument("--year", type=int, default=2020, help="Target base year")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "validation" / "ppl" / "custom_powerplants_reconciled_2020.csv",
        help="Output reconciled custom powerplants CSV",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=REPO_ROOT / "validation" / "results" / "custom_powerplants_reconciliation_summary.csv",
        help="Output summary CSV",
    )
    parser.add_argument(
        "--scales-csv",
        type=Path,
        default=REPO_ROOT / "validation" / "results" / "custom_powerplants_reconciliation_scales.csv",
        help="Output scale factors CSV",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    args.scales_csv.parent.mkdir(parents=True, exist_ok=True)

    custom_before = _load_custom(args.custom_csv, args.year)
    ref = _build_reference(args.gem_clean_csv, args.irena_csv, args.year)
    scales = _calculate_scales(custom_before, ref)
    custom_after = _apply_scales(custom_before, scales)
    summary = _summary_table(custom_before, custom_after, args.year)

    out = custom_after.drop(columns=["validation_tech", "is_active_year"])
    out.to_csv(args.output_csv, index=True, encoding="utf-8-sig")
    summary.to_csv(args.summary_csv, index=False)
    scales.to_csv(args.scales_csv, index=False)

    before_totals = (
        custom_before.loc[custom_before["is_active_year"] & custom_before["validation_tech"].isin(TARGET_TECHS)]
        .groupby("validation_tech")["Capacity"]
        .sum()
        .rename("before_mw")
    )
    after_totals = (
        custom_after.loc[custom_after["is_active_year"] & custom_after["validation_tech"].isin(TARGET_TECHS)]
        .groupby("validation_tech")["Capacity"]
        .sum()
        .rename("after_mw")
    )
    ref_totals = ref.groupby("validation_tech")["reference_mw"].sum().rename("reference_mw")
    totals = pd.concat([before_totals, after_totals, ref_totals], axis=1).fillna(0.0)

    LOGGER.info("Wrote reconciled custom powerplants: %s", args.output_csv)
    LOGGER.info("Wrote reconciliation summary: %s", args.summary_csv)
    LOGGER.info("Wrote scale factors: %s", args.scales_csv)
    LOGGER.info("Global totals by technology (MW):\n%s", totals.round(1).to_string())

    cn = summary.loc[summary["country"] == "CN"].copy()
    if not cn.empty:
        LOGGER.info("China reconciliation snapshot (MW):\n%s", cn.round(1).to_string(index=False))


if __name__ == "__main__":
    main()

