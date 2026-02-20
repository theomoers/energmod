#!/usr/bin/env python3
"""
Prepare a cleaned GEM capacity reference table for validation.

Outputs a compact CSV with:
- country (ISO2)
- validation_tech (gas/coal)
- year
- reference_mw
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from _helpers import country_name_2_two_digits


LOGGER = logging.getLogger(__name__)

GEM_COUNTRY_ALIASES = {
    "Czechia": "Czech Republic",
    "Eswatini": "Swaziland",
    "Russia": "Russian Federation",
}


@lru_cache(maxsize=1024)
def _safe_country_name_to_iso2(name: str) -> str | np.nan:
    if not isinstance(name, str):
        return np.nan
    clean = name.strip()
    if not clean:
        return np.nan
    for candidate in [clean, clean.title()]:
        try:
            iso2 = country_name_2_two_digits(candidate)
        except Exception:
            continue
        if isinstance(iso2, str) and len(iso2) == 2:
            return iso2
    return np.nan


@lru_cache(maxsize=1024)
def _safe_country_name_or_iso2_to_iso2(value: str) -> str | np.nan:
    if not isinstance(value, str):
        return np.nan
    clean = value.strip()
    if len(clean) == 2 and clean.isalpha():
        return clean.upper()
    alias = GEM_COUNTRY_ALIASES.get(clean)
    if alias:
        return _safe_country_name_to_iso2(alias)
    return _safe_country_name_to_iso2(clean)


def _active_gem_mask(df: pd.DataFrame, year: int) -> pd.Series:
    status_col = next((c for c in df.columns if "status" in str(c).lower()), None)
    start_col = next((c for c in df.columns if "start year" in str(c).lower()), None)
    retired_col = next((c for c in df.columns if "retired year" in str(c).lower()), None)

    mask = pd.Series(True, index=df.index)
    if status_col is not None:
        status = df[status_col].astype(str).str.lower()
        mask &= status.str.contains(r"\boperating\b|\bmothballed\b", regex=True)
    if start_col is not None:
        start = pd.to_numeric(df[start_col], errors="coerce")
        mask &= start.fillna(-np.inf) <= year
    if retired_col is not None:
        retired = pd.to_numeric(df[retired_col], errors="coerce")
        mask &= retired.isna() | (retired >= year)
    return mask


def _extract_capacity(
    gem_csv: Path,
    *,
    year: int,
    validation_tech: str,
    fuel_pattern: str | None = None,
) -> pd.DataFrame:
    gem = pd.read_csv(gem_csv, encoding="utf-8-sig")
    country_col = next((c for c in gem.columns if "country" in str(c).lower()), None)
    capacity_col = next((c for c in gem.columns if "capacity" in str(c).lower()), None)
    if country_col is None or capacity_col is None:
        raise ValueError(f"Could not find country/capacity columns in {gem_csv}")

    ref = gem.copy()
    ref[capacity_col] = pd.to_numeric(ref[capacity_col], errors="coerce")
    ref = ref.loc[ref[capacity_col].fillna(0.0) > 0.0].copy()
    ref = ref.loc[_active_gem_mask(ref, year)].copy()

    if fuel_pattern is not None:
        fuel_col = next((c for c in ref.columns if "fuel" in str(c).lower()), None)
        if fuel_col is not None:
            fuel = ref[fuel_col].astype(str).str.lower()
            ref = ref.loc[fuel.str.contains(fuel_pattern, regex=True)].copy()

    country_raw = ref[country_col].astype(str).str.strip()
    unique_countries = pd.Index(country_raw.unique())
    country_map = {
        c: _safe_country_name_or_iso2_to_iso2(c)
        for c in unique_countries
        if isinstance(c, str)
    }
    ref["country"] = country_raw.map(country_map)
    ref = ref.loc[ref["country"].notna()].copy()

    out = (
        ref.groupby("country", as_index=False)[capacity_col]
        .sum()
        .rename(columns={capacity_col: "reference_mw"})
    )
    out["validation_tech"] = validation_tech
    out["year"] = year
    out["source_file"] = gem_csv.name
    return out[["country", "validation_tech", "year", "reference_mw", "source_file"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare cleaned GEM capacity reference CSV.")
    parser.add_argument(
        "--gem-gas-csv",
        type=Path,
        default=REPO_ROOT / "validation" / "data" / "Global-Oil-and-Gas-Plant-Tracker-GOGPT-August-2025.csv",
        help="Path to GEM gas/oil tracker CSV",
    )
    parser.add_argument(
        "--gem-coal-csv",
        type=Path,
        default=REPO_ROOT / "validation" / "data" / "Global-Coal-Plant-Tracker-January-2026.csv",
        help="Path to GEM coal tracker CSV (optional; skipped if missing)",
    )
    parser.add_argument("--year", type=int, default=2020, help="Reference year")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "validation" / "data" / "gem_capacity_reference_2020.csv",
        help="Output path for cleaned GEM capacity reference CSV",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    pieces = []
    gas = _extract_capacity(
        args.gem_gas_csv,
        year=args.year,
        validation_tech="gas",
        fuel_pattern=r"fossil gas|natural gas|lng",
    )
    pieces.append(gas)
    LOGGER.info("Loaded GEM gas capacity rows: %d", len(gas))

    if args.gem_coal_csv.exists():
        coal = _extract_capacity(
            args.gem_coal_csv,
            year=args.year,
            validation_tech="coal",
            fuel_pattern=None,
        )
        pieces.append(coal)
        LOGGER.info("Loaded GEM coal capacity rows: %d", len(coal))
    else:
        LOGGER.warning("GEM coal CSV not found at %s; writing gas-only GEM reference", args.gem_coal_csv)

    out = pd.concat(pieces, ignore_index=True)
    out = (
        out.groupby(["country", "validation_tech", "year"], as_index=False)
        .agg(reference_mw=("reference_mw", "sum"), sources=("source_file", lambda s: ";".join(sorted(set(s)))))
        .sort_values(["validation_tech", "country"])
    )

    out.to_csv(args.output_csv, index=False)
    LOGGER.info("Wrote cleaned GEM capacity reference to %s", args.output_csv)
    LOGGER.info("Technologies: %s", ", ".join(sorted(out["validation_tech"].unique())))
    LOGGER.info(
        "Totals (MW):\n%s",
        out.groupby("validation_tech")["reference_mw"].sum().sort_values(ascending=False).to_string(),
    )


if __name__ == "__main__":
    main()
