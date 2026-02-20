#!/usr/bin/env python3
"""
Validate a solved PyPSA network against IRENA/IAEA/GEM capacities and OWID energy balances.

The script compares:
1. Installed/optimized electric capacity (MW) by country and aggregated technology group.
2. Electricity generation balance (TWh) by country and source categories.
3. Non-electric fossil energy use (TWh) by country and fuel.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pypsa

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from _helpers import (
    country_name_2_two_digits,
    three_2_two_digits_country,
    two_2_three_digits_country,
)


LOGGER = logging.getLogger(__name__)


# IRENA technology categories -> validation categories (non-fossil electric techs).
IRENA_TO_VALIDATION_TECH = {
    "PV": "solar",
    "Onshore": "onshore_wind",
    "Offshore": "offshore_wind",
    "Renewable hydropower": "hydro",
    "Pumped Storage": "hydro",
    "Mixed Hydro Plants": "hydro",
    "Geothermal": "geothermal",
    "Nuclear": "nuclear",
    "Biogas": "bioenergy",
    "Liquid biofuels": "bioenergy",
    "Solid biofuels": "bioenergy",
    "Renewable municipal waste": "bioenergy",
}

IRENA_FOSSIL_AGGREGATE_TECHS = {
    "Natural gas",
    "Coal and peat",
    "Oil",
    "Fossil fuels n.e.s.",
    "Other non-renewable energy",
}


# Model carrier -> validation categories for installed capacity
MODEL_CAPACITY_TECH_MAP = {
    "solar": "solar",
    "onwind": "onshore_wind",
    "offwind-ac": "offshore_wind",
    "offwind-dc": "offshore_wind",
    "ror": "hydro",
    "hydro": "hydro",
    "PHS": "hydro",
    "geothermal": "geothermal",
    "nuclear": "nuclear",
    "OCGT": "gas",
    "CCGT": "gas",
    "urban central gas CHP": "gas",
    "urban central gas CHP CC": "gas",
    "coal": "coal",
    "lignite": "coal",
    "oil": "oil",
    "biomass": "bioenergy",
    "urban central solid biomass CHP": "bioenergy",
    "urban central solid biomass CHP CC": "bioenergy",
}

FOSSIL_BUS_CARRIERS = {"gas", "oil", "coal", "lignite"}
FOSSIL_VALIDATION_TECHS = {"gas", "oil", "coal"}
MODEL_FOSSIL_TECHS_FOR_AGGREGATE = {"gas", "coal", "oil"}
VALIDATION_TECH_FOSSIL_AGGREGATED = "fossil_aggregated"


# Fallback hints for unmapped model carriers (used only when a clear match exists).
OTHER_RENEWABLE_HINTS = ("csp", "marine", "wave", "tidal")
OTHER_NON_RENEWABLE_HINTS = ("fossil", "peat", "oil shale", "non-renewable")


# Model carrier -> OWID electricity columns
MODEL_ENERGY_METRIC_MAP = {
    "solar": "solar_electricity",
    "onwind": "wind_electricity",
    "offwind-ac": "wind_electricity",
    "offwind-dc": "wind_electricity",
    "ror": "hydro_electricity",
    "hydro": "hydro_electricity",
    "geothermal": "other_renewable_electricity",
    "nuclear": "nuclear_electricity",
    "OCGT": "gas_electricity",
    "CCGT": "gas_electricity",
    "urban central gas CHP": "gas_electricity",
    "urban central gas CHP CC": "gas_electricity",
    "coal": "coal_electricity",
    "lignite": "coal_electricity",
    "oil": "oil_electricity",
    "biomass": "biofuel_electricity",
    "urban central solid biomass CHP": "biofuel_electricity",
    "urban central solid biomass CHP CC": "biofuel_electricity",
}


OWID_ENERGY_COLUMNS = [
    "coal_electricity",
    "gas_electricity",
    "oil_electricity",
    "biofuel_electricity",
    "hydro_electricity",
    "nuclear_electricity",
    "solar_electricity",
    "wind_electricity",
    "other_renewable_electricity",
    "fossil_electricity",
    "renewables_electricity",
    "low_carbon_electricity",
    "electricity_generation",
]

OWID_NON_ELECTRIC_COLUMNS = [
    "oil_consumption",
    "gas_consumption",
    "coal_consumption",
    "oil_electricity",
    "gas_electricity",
    "coal_electricity",
]

OWID_ELECTRICITY_DEMAND_COLUMN = "electricity_demand"
ELECTRICITY_TECH_METRICS = [
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

FOSSIL_BUS_CARRIER_TO_NON_ELEC_METRIC = {
    "oil": "oil_non_electric_consumption",
    "gas": "gas_non_electric_consumption",
    "coal": "coal_non_electric_consumption",
    "lignite": "coal_non_electric_consumption",
}

IAEA_COUNTRY_ALIASES = {
    "UAE": "United Arab Emirates",
}

def _capacity_column(df: pd.DataFrame) -> str:
    if "p_nom_opt" in df.columns and df["p_nom_opt"].notna().any():
        return "p_nom_opt"
    return "p_nom"


def _bus_country_lookup(n: pypsa.Network) -> pd.Series:
    """
    Build a robust bus->country lookup.

    Some sectoral fuel buses do not populate `buses.country`; for these,
    infer ISO2 from `buses.location` or bus name prefix.
    """
    country = n.buses["country"].replace("", np.nan)
    location_iso2 = (
        n.buses["location"]
        .astype(str)
        .str.extract(r"^([A-Z]{2})\b")[0]
    )
    index_iso2 = n.buses.index.astype(str).to_series(index=n.buses.index).str.extract(r"^([A-Z]{2})\b")[0]
    return country.fillna(location_iso2).fillna(index_iso2).fillna("")


def _safe_iso3_to_iso2(code: str) -> str | np.nan:
    if not isinstance(code, str) or len(code) != 3:
        return np.nan
    try:
        iso2 = three_2_two_digits_country(code)
    except Exception:
        return np.nan
    if not isinstance(iso2, str) or len(iso2) != 2:
        return np.nan
    return iso2


def _link_electric_capacity(
    n: pypsa.Network, exclude_bus0_carriers: set[str] | None = None
) -> pd.DataFrame:
    cap_col = _capacity_column(n.links)
    link_df = n.links.copy()
    bus0_carrier = link_df["bus0"].map(n.buses.carrier)
    bus_country = _bus_country_lookup(n)

    pieces = []
    for port in [1, 2, 3, 4]:
        bus_col = f"bus{port}"
        if bus_col not in link_df.columns:
            continue

        eff_col = "efficiency" if port == 1 else f"efficiency{port}"
        if eff_col not in link_df.columns:
            continue

        buses = link_df[bus_col]
        bus_carrier = buses.map(n.buses.carrier)
        mask = buses.notna() & buses.ne("") & bus_carrier.eq("AC")
        if exclude_bus0_carriers:
            mask &= ~bus0_carrier.isin(exclude_bus0_carriers)
        if not mask.any():
            continue

        sub = link_df.loc[mask, [cap_col, eff_col, "carrier", bus_col]].copy()
        sub["country"] = sub[bus_col].map(bus_country).fillna("")
        sub["component"] = "Link"
        sub["capacity_mw"] = sub[cap_col].astype(float) * sub[eff_col].astype(float).abs()
        sub = sub.loc[sub["capacity_mw"] > 0, ["component", "country", "carrier", "capacity_mw"]]
        pieces.append(sub)

    if not pieces:
        return pd.DataFrame(columns=["component", "country", "carrier", "capacity_mw"])

    return pd.concat(pieces, ignore_index=True)


def _fossil_electric_link_carriers(n: pypsa.Network) -> set[str]:
    """
    Return link carriers that consume fossil fuel (bus0) and produce electricity (any AC output port).
    """
    links = n.links.copy()
    links["bus0_carrier"] = links["bus0"].map(n.buses.carrier)
    has_ac_output = pd.Series(False, index=links.index)

    for port in [1, 2, 3, 4]:
        bus_col = f"bus{port}"
        if bus_col not in links.columns:
            continue
        has_ac_output |= links[bus_col].map(n.buses.carrier).eq("AC")

    mask = links["bus0_carrier"].isin(FOSSIL_BUS_CARRIERS) & has_ac_output
    return set(links.loc[mask, "carrier"].dropna().unique())


def _infer_other_capacity_bucket(carrier: str) -> str | None:
    c = str(carrier).lower()
    if any(k in c for k in OTHER_RENEWABLE_HINTS):
        return "other_renewables"
    if any(k in c for k in OTHER_NON_RENEWABLE_HINTS):
        return "other_non_renewables"
    return None


def _model_capacity_by_country_tech(n: pypsa.Network) -> tuple[pd.DataFrame, pd.DataFrame]:
    gen_col = _capacity_column(n.generators)
    su_col = _capacity_column(n.storage_units)
    bus_country = _bus_country_lookup(n)

    gen = n.generators.copy()
    gen = gen.loc[gen.bus.map(n.buses.carrier).eq("AC")].copy()
    gen["component"] = "Generator"
    gen["country"] = gen.bus.map(bus_country).fillna("")
    gen["capacity_mw"] = gen[gen_col].astype(float)
    gen = gen.loc[:, ["component", "country", "carrier", "capacity_mw"]]
    gen = gen.loc[gen.capacity_mw > 0]

    su = n.storage_units.copy()
    su = su.loc[su.bus.map(n.buses.carrier).eq("AC")].copy()
    su["component"] = "StorageUnit"
    su["country"] = su.bus.map(bus_country).fillna("")
    su["capacity_mw"] = su[su_col].astype(float)
    su = su.loc[:, ["component", "country", "carrier", "capacity_mw"]]
    su = su.loc[su.capacity_mw > 0]

    links = _link_electric_capacity(n)

    model_raw = pd.concat([gen, su, links], ignore_index=True)
    model_raw = model_raw.loc[model_raw.country.ne("")]

    model_raw["validation_tech"] = model_raw["carrier"].map(MODEL_CAPACITY_TECH_MAP)
    # Fossil capacity validation must use conversion links, not fuel-supply generators.
    model_raw = model_raw.loc[
        ~(
            model_raw["component"].ne("Link")
            & model_raw["validation_tech"].isin(FOSSIL_VALIDATION_TECHS)
        )
    ].copy()

    mapped = model_raw.loc[model_raw.validation_tech.notna()].copy()
    mapped = (
        mapped.groupby(["country", "validation_tech"], as_index=False)["capacity_mw"]
        .sum()
    )

    fossil_agg = mapped.loc[
        mapped["validation_tech"].isin(MODEL_FOSSIL_TECHS_FOR_AGGREGATE)
    ].copy()
    if not fossil_agg.empty:
        fossil_agg = (
            fossil_agg.groupby("country", as_index=False)["capacity_mw"].sum()
        )
        fossil_agg["validation_tech"] = VALIDATION_TECH_FOSSIL_AGGREGATED
        mapped = pd.concat([mapped, fossil_agg], ignore_index=True)
        mapped = (
            mapped.groupby(["country", "validation_tech"], as_index=False)["capacity_mw"]
            .sum()
        )

    # Oil is validated only through the aggregated fossil category.
    mapped = mapped.loc[mapped["validation_tech"] != "oil"].copy()

    unmapped = model_raw.loc[model_raw.validation_tech.isna()].copy()
    unmapped = unmapped.loc[~((unmapped["component"] == "Generator") & (unmapped["carrier"] == "Load"))]
    unmapped["inferred_validation_tech"] = unmapped["carrier"].apply(_infer_other_capacity_bucket)

    inferred = unmapped.loc[unmapped["inferred_validation_tech"].notna()].copy()
    if not inferred.empty:
        inferred = (
            inferred.groupby(["country", "inferred_validation_tech"], as_index=False)["capacity_mw"]
            .sum()
            .rename(columns={"inferred_validation_tech": "validation_tech"})
        )
        mapped = pd.concat([mapped, inferred], ignore_index=True)
        mapped = mapped.groupby(["country", "validation_tech"], as_index=False)["capacity_mw"].sum()

    unmapped = unmapped.loc[unmapped["inferred_validation_tech"].isna()].copy()
    unmapped = (
        unmapped.groupby(["component", "carrier"], as_index=False)["capacity_mw"]
        .sum()
        .sort_values("capacity_mw", ascending=False)
    )

    return mapped, unmapped


def _irena_capacity_by_country_tech(irena_csv: Path, year: int) -> pd.DataFrame:
    irena = pd.read_csv(irena_csv)
    year_col = str(year)
    if year_col not in irena.columns:
        raise ValueError(f"Year column '{year_col}' not found in {irena_csv}")

    ref = irena[["Technology", "Country", year_col]].copy()
    ref = ref.rename(columns={"Country": "country", year_col: "reference_mw"})
    ref["country"] = ref["country"].astype(str).str.strip()
    ref["reference_mw"] = pd.to_numeric(ref["reference_mw"], errors="coerce").fillna(0.0)
    ref["validation_tech"] = ref["Technology"].map(IRENA_TO_VALIDATION_TECH)
    ref = ref.loc[ref.validation_tech.notna()].copy()
    ref = ref.groupby(["country", "validation_tech"], as_index=False)["reference_mw"].sum()

    return ref


def _irena_fossil_aggregated_capacity_by_country(irena_csv: Path, year: int) -> pd.DataFrame:
    irena = pd.read_csv(irena_csv)
    year_col = str(year)
    if year_col not in irena.columns:
        raise ValueError(f"Year column '{year_col}' not found in {irena_csv}")

    ref = irena[["Technology", "Country", year_col]].copy()
    ref = ref.rename(columns={"Country": "country", year_col: "reference_mw"})
    ref["country"] = ref["country"].astype(str).str.strip()
    ref["reference_mw"] = pd.to_numeric(ref["reference_mw"], errors="coerce").fillna(0.0)
    ref = ref.loc[ref["Technology"].isin(IRENA_FOSSIL_AGGREGATE_TECHS)].copy()
    ref["validation_tech"] = VALIDATION_TECH_FOSSIL_AGGREGATED
    ref = ref.groupby(["country", "validation_tech"], as_index=False)["reference_mw"].sum()
    return ref


def _irena_single_tech_capacity_by_country(irena_csv: Path, year: int, tech: str) -> pd.DataFrame:
    irena = pd.read_csv(irena_csv)
    year_col = str(year)
    if year_col not in irena.columns:
        raise ValueError(f"Year column '{year_col}' not found in {irena_csv}")
    ref = irena.loc[irena["Technology"] == tech, ["Country", year_col]].copy()
    ref = ref.rename(columns={"Country": "country", year_col: "reference_mw"})
    ref["country"] = ref["country"].astype(str).str.strip()
    ref["reference_mw"] = pd.to_numeric(ref["reference_mw"], errors="coerce").fillna(0.0)
    return ref.groupby("country", as_index=False)["reference_mw"].sum()


@lru_cache(maxsize=1024)
def _safe_country_name_to_iso2(name: str) -> str | np.nan:
    if not isinstance(name, str):
        return np.nan
    clean = name.strip()
    if not clean:
        return np.nan
    alias = IAEA_COUNTRY_ALIASES.get(clean.upper())
    candidates = [clean, clean.title()]
    if alias:
        candidates = [alias] + candidates
    for candidate in candidates:
        try:
            iso2 = country_name_2_two_digits(candidate)
        except Exception:
            continue
        if isinstance(iso2, str) and len(iso2) == 2:
            return iso2
    return np.nan


def _gem_clean_capacity_by_country_tech(gem_clean_csv: Path, year: int) -> pd.DataFrame:
    gem = pd.read_csv(gem_clean_csv, encoding="utf-8-sig")
    required = {"country", "validation_tech", "reference_mw"}
    missing = sorted(required - set(gem.columns))
    if missing:
        raise ValueError(
            f"Missing required columns in {gem_clean_csv}: {', '.join(missing)}"
        )

    if "year" in gem.columns:
        gem = gem.loc[pd.to_numeric(gem["year"], errors="coerce") == year].copy()

    gem["country"] = gem["country"].astype(str).str.strip().str.upper()
    gem["validation_tech"] = gem["validation_tech"].astype(str).str.strip()
    gem["reference_mw"] = pd.to_numeric(gem["reference_mw"], errors="coerce").fillna(0.0)
    gem = gem.loc[gem["validation_tech"].isin(["gas", "coal"])].copy()
    gem = gem.groupby(["country", "validation_tech"], as_index=False)["reference_mw"].sum()
    return gem


def _iaea_nuclear_capacity_by_country(iaea_csv: Path) -> pd.DataFrame:
    iaea = pd.read_csv(iaea_csv, encoding="utf-8-sig")
    country_col = next((c for c in iaea.columns if "country" in str(c).lower()), None)
    capacity_col = next((c for c in iaea.columns if "capacity" in str(c).lower()), None)
    if country_col is None or capacity_col is None:
        raise ValueError(f"Could not find country/capacity columns in {iaea_csv}")

    ref = iaea[[country_col, capacity_col]].copy()
    ref = ref.rename(columns={country_col: "country_name", capacity_col: "reference_mw"})
    ref["country"] = ref["country_name"].apply(_safe_country_name_to_iso2)
    ref["reference_mw"] = pd.to_numeric(ref["reference_mw"], errors="coerce").fillna(0.0)
    ref = ref.loc[ref["country"].notna()].copy()
    ref["validation_tech"] = "nuclear"
    ref = ref.groupby(["country", "validation_tech"], as_index=False)["reference_mw"].sum()
    return ref


def _capacity_reference_by_country_tech(
    irena_csv: Path,
    iaea_csv: Path,
    gem_clean_csv: Path,
    year: int,
) -> pd.DataFrame:
    irena_ref = _irena_capacity_by_country_tech(irena_csv, year)
    irena_ref = irena_ref.loc[
        ~irena_ref["validation_tech"].isin(["nuclear", "gas", "coal", "oil"])
    ].copy()

    iaea_ref = _iaea_nuclear_capacity_by_country(iaea_csv)
    if not gem_clean_csv.exists():
        raise FileNotFoundError(
            f"Cleaned GEM reference not found at {gem_clean_csv}. "
            "Run validation/scripts/prepare_gem_capacity_reference.py first."
        )
    gem_ref = _gem_clean_capacity_by_country_tech(gem_clean_csv, year)
    gem_gas_ref = gem_ref.loc[gem_ref["validation_tech"] == "gas"].copy()
    gem_coal_ref = gem_ref.loc[gem_ref["validation_tech"] == "coal"].copy()
    if gem_coal_ref.empty:
        LOGGER.warning(
            "No coal rows found in cleaned GEM file %s; falling back to IRENA 'Coal and peat'.",
            gem_clean_csv,
        )
        coal = _irena_single_tech_capacity_by_country(irena_csv, year, "Coal and peat")
        coal["validation_tech"] = "coal"
        gem_coal_ref = coal

    irena_fossil_agg = _irena_fossil_aggregated_capacity_by_country(irena_csv, year)

    ref = pd.concat(
        [irena_ref, iaea_ref, gem_gas_ref, gem_coal_ref, irena_fossil_agg], ignore_index=True
    )
    return ref.groupby(["country", "validation_tech"], as_index=False)["reference_mw"].sum()


def _capacity_comparison(model_cap: pd.DataFrame, ref_cap: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cmp_df = model_cap.merge(
        ref_cap,
        on=["country", "validation_tech"],
        how="outer",
    ).fillna({"capacity_mw": 0.0, "reference_mw": 0.0})

    cmp_df["error_mw"] = cmp_df["capacity_mw"] - cmp_df["reference_mw"]
    cmp_df["abs_error_mw"] = cmp_df["error_mw"].abs()
    cmp_df["ape_pct"] = np.where(
        cmp_df["reference_mw"] > 0,
        cmp_df["abs_error_mw"] / cmp_df["reference_mw"] * 100.0,
        np.nan,
    )

    summary = cmp_df.groupby("validation_tech", as_index=False).agg(
        countries=("country", "nunique"),
        model_total_mw=("capacity_mw", "sum"),
        reference_total_mw=("reference_mw", "sum"),
        mae_mw=("abs_error_mw", "mean"),
        rmse_mw=("error_mw", lambda s: float(np.sqrt(np.mean(np.square(s))))),
        mean_ape_pct=("ape_pct", "mean"),
        wape_pct=(
            "abs_error_mw",
            lambda s: (
                float(s.sum())
                / max(
                    cmp_df.loc[s.index, "reference_mw"].sum(),
                    1e-9,
                )
                * 100.0
            ),
        ),
    )

    return cmp_df, summary.sort_values("wape_pct", ascending=False)


def _model_electricity_balance(n: pypsa.Network) -> pd.DataFrame:
    eb = n.statistics.energy_balance(
        bus_carrier="AC",
        aggregate_time="sum",
        aggregate_groups="sum",
        groupby=n.statistics.groupers.get_country_and_carrier,
        nice_names=False,
    )
    eb = eb.rename("energy_mwh").reset_index()
    eb = eb.loc[eb.component.isin(["Generator", "StorageUnit", "Link"])].copy()
    eb = eb.loc[(eb.country != "") & (eb.energy_mwh > 0)].copy()
    eb["metric"] = eb["carrier"].map(MODEL_ENERGY_METRIC_MAP)
    eb = eb.loc[eb.metric.notna()].copy()

    model = eb.groupby(["country", "metric"], as_index=False)["energy_mwh"].sum()
    model["model_twh"] = model["energy_mwh"] / 1e6
    model = model.drop(columns=["energy_mwh"])

    wide = model.pivot(index="country", columns="metric", values="model_twh").fillna(0.0)
    for col in [
        "coal_electricity",
        "gas_electricity",
        "oil_electricity",
        "biofuel_electricity",
        "hydro_electricity",
        "nuclear_electricity",
        "solar_electricity",
        "wind_electricity",
        "other_renewable_electricity",
    ]:
        if col not in wide.columns:
            wide[col] = 0.0

    wide["fossil_electricity"] = (
        wide["coal_electricity"] + wide["gas_electricity"] + wide["oil_electricity"]
    )
    wide["renewables_electricity"] = (
        wide["hydro_electricity"]
        + wide["solar_electricity"]
        + wide["wind_electricity"]
        + wide["biofuel_electricity"]
        + wide["other_renewable_electricity"]
    )
    wide["low_carbon_electricity"] = wide["renewables_electricity"] + wide["nuclear_electricity"]
    wide["electricity_generation"] = wide["fossil_electricity"] + wide["low_carbon_electricity"]

    out = wide.reset_index().melt(
        id_vars="country",
        value_vars=OWID_ENERGY_COLUMNS,
        var_name="metric",
        value_name="model_twh",
    )
    return out


def _owid_electricity_balance(
    owid_csv: Path, year: int, iso3_filter: set[str] | None = None
) -> pd.DataFrame:
    usecols = ["year", "iso_code"] + OWID_ENERGY_COLUMNS + ["other_renewable_exc_biofuel_electricity"]
    owid = pd.read_csv(owid_csv, usecols=lambda c: c in set(usecols))
    owid = owid.loc[owid["year"] == year].copy()
    if iso3_filter is not None:
        owid = owid.loc[owid["iso_code"].isin(iso3_filter)].copy()
    owid["country"] = owid["iso_code"].apply(_safe_iso3_to_iso2)
    owid = owid.loc[owid.country.notna()].copy()

    for col in OWID_ENERGY_COLUMNS:
        if col in owid.columns:
            owid[col] = pd.to_numeric(owid[col], errors="coerce")

    # Prefer OWID's non-biofuel renewable split when available.
    if "other_renewable_exc_biofuel_electricity" in owid.columns:
        owid["other_renewable_electricity"] = owid["other_renewable_exc_biofuel_electricity"]

    keep_cols = [c for c in OWID_ENERGY_COLUMNS if c in owid.columns]
    owid = owid.groupby("country", as_index=False)[keep_cols].sum(min_count=1)
    out = owid.melt(
        id_vars="country",
        value_vars=keep_cols,
        var_name="metric",
        value_name="reference_twh",
    )
    return out


def _model_non_electric_fossil_balance(n: pypsa.Network) -> pd.DataFrame:
    bus_country = _bus_country_lookup(n)
    electric_fossil_links = _fossil_electric_link_carriers(n)

    w = n.statistics.withdrawal(
        aggregate_time="sum",
        aggregate_groups="sum",
        groupby=n.statistics.groupers.get_bus_and_carrier_and_bus_carrier,
        nice_names=False,
    )
    w = w.rename("energy_mwh").reset_index()
    w["country"] = w["bus"].map(bus_country).fillna("")

    w = w.loc[
        w["country"].ne("")
        & w["bus_carrier"].isin(FOSSIL_BUS_CARRIERS)
        & w["energy_mwh"].gt(0.0)
    ].copy()

    is_non_electric = (
        w["component"].eq("Load")
        | (w["component"].eq("Link") & ~w["carrier"].isin(electric_fossil_links))
    )
    non_electric = w.loc[is_non_electric].copy()
    non_electric["metric"] = non_electric["bus_carrier"].map(FOSSIL_BUS_CARRIER_TO_NON_ELEC_METRIC)
    non_electric = non_electric.loc[non_electric["metric"].notna()].copy()
    non_electric = (
        non_electric.groupby(["country", "metric"], as_index=False)["energy_mwh"]
        .sum()
        .rename(columns={"energy_mwh": "model_twh"})
    )
    non_electric["model_twh"] = non_electric["model_twh"] / 1e6
    return non_electric


def _owid_non_electric_fossil_balance(
    owid_csv: Path, year: int, iso3_filter: set[str] | None = None
) -> pd.DataFrame:
    usecols = ["year", "iso_code"] + OWID_NON_ELECTRIC_COLUMNS
    owid = pd.read_csv(owid_csv, usecols=lambda c: c in set(usecols))
    owid = owid.loc[owid["year"] == year].copy()
    if iso3_filter is not None:
        owid = owid.loc[owid["iso_code"].isin(iso3_filter)].copy()
    owid["country"] = owid["iso_code"].apply(_safe_iso3_to_iso2)
    owid = owid.loc[owid.country.notna()].copy()

    for col in OWID_NON_ELECTRIC_COLUMNS:
        if col in owid.columns:
            owid[col] = pd.to_numeric(owid[col], errors="coerce")

    agg_cols = [c for c in OWID_NON_ELECTRIC_COLUMNS if c in owid.columns]
    owid = owid.groupby("country", as_index=False)[agg_cols].sum(min_count=1)
    owid["oil_non_electric_consumption"] = (
        owid["oil_consumption"].fillna(0.0) - owid["oil_electricity"].fillna(0.0)
    ).clip(lower=0.0)
    owid["gas_non_electric_consumption"] = (
        owid["gas_consumption"].fillna(0.0) - owid["gas_electricity"].fillna(0.0)
    ).clip(lower=0.0)
    owid["coal_non_electric_consumption"] = (
        owid["coal_consumption"].fillna(0.0) - owid["coal_electricity"].fillna(0.0)
    ).clip(lower=0.0)

    keep_cols = [
        "oil_non_electric_consumption",
        "gas_non_electric_consumption",
        "coal_non_electric_consumption",
    ]
    out = owid.melt(
        id_vars="country",
        value_vars=keep_cols,
        var_name="metric",
        value_name="reference_twh",
    )
    return out


def _model_electricity_demand(n: pypsa.Network) -> pd.DataFrame:
    if n.loads.empty:
        return pd.DataFrame(columns=["country", "metric", "model_twh"])

    bus_country = _bus_country_lookup(n)
    w = n.statistics.withdrawal(
        aggregate_time="sum",
        aggregate_groups="sum",
        groupby=n.statistics.groupers.get_bus_and_carrier_and_bus_carrier,
        nice_names=False,
    )
    w = w.rename("energy_mwh").reset_index()
    w["country"] = w["bus"].map(bus_country).fillna("")
    load = w.loc[
        w["component"].eq("Load")
        & w["bus_carrier"].eq("AC")
        & w["country"].ne("")
        & w["energy_mwh"].gt(0.0)
    ].copy()
    if load.empty:
        return pd.DataFrame(columns=["country", "metric", "model_twh"])

    by_country_twh = load.groupby("country")["energy_mwh"].sum() / 1e6
    out = by_country_twh.rename_axis("country").rename("model_twh").reset_index()
    out["metric"] = OWID_ELECTRICITY_DEMAND_COLUMN
    return out[["country", "metric", "model_twh"]]


def _owid_electricity_demand(
    owid_csv: Path, year: int, iso3_filter: set[str] | None = None
) -> pd.DataFrame:
    usecols = ["year", "iso_code", OWID_ELECTRICITY_DEMAND_COLUMN]
    owid = pd.read_csv(owid_csv, usecols=lambda c: c in set(usecols))
    if OWID_ELECTRICITY_DEMAND_COLUMN not in owid.columns:
        LOGGER.warning(
            "OWID column '%s' not found in %s; skipping electricity demand validation",
            OWID_ELECTRICITY_DEMAND_COLUMN,
            owid_csv,
        )
        return pd.DataFrame(columns=["country", "metric", "reference_twh"])

    owid = owid.loc[owid["year"] == year].copy()
    if iso3_filter is not None:
        owid = owid.loc[owid["iso_code"].isin(iso3_filter)].copy()

    owid["country"] = owid["iso_code"].apply(_safe_iso3_to_iso2)
    owid = owid.loc[owid.country.notna()].copy()
    owid[OWID_ELECTRICITY_DEMAND_COLUMN] = pd.to_numeric(
        owid[OWID_ELECTRICITY_DEMAND_COLUMN], errors="coerce"
    ).fillna(0.0)

    grouped = (
        owid.groupby("country", as_index=False)[OWID_ELECTRICITY_DEMAND_COLUMN]
        .sum(min_count=1)
        .rename(columns={OWID_ELECTRICITY_DEMAND_COLUMN: "reference_twh"})
    )
    grouped["metric"] = OWID_ELECTRICITY_DEMAND_COLUMN
    return grouped[["country", "metric", "reference_twh"]]


def _energy_comparison(model_energy: pd.DataFrame, ref_energy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cmp_df = model_energy.merge(ref_energy, on=["country", "metric"], how="outer")
    cmp_df["model_twh"] = cmp_df["model_twh"].fillna(0.0)
    cmp_df["reference_twh"] = cmp_df["reference_twh"].fillna(0.0)
    cmp_df["error_twh"] = cmp_df["model_twh"] - cmp_df["reference_twh"]
    cmp_df["abs_error_twh"] = cmp_df["error_twh"].abs()
    cmp_df["ape_pct"] = np.where(
        cmp_df["reference_twh"] > 0,
        cmp_df["abs_error_twh"] / cmp_df["reference_twh"] * 100.0,
        np.nan,
    )

    summary = cmp_df.groupby("metric", as_index=False).agg(
        countries=("country", "nunique"),
        model_total_twh=("model_twh", "sum"),
        reference_total_twh=("reference_twh", "sum"),
        mae_twh=("abs_error_twh", "mean"),
        rmse_twh=("error_twh", lambda s: float(np.sqrt(np.mean(np.square(s))))),
        mean_ape_pct=("ape_pct", "mean"),
        wape_pct=(
            "abs_error_twh",
            lambda s: (
                float(s.sum())
                / max(
                    cmp_df.loc[s.index, "reference_twh"].sum(),
                    1e-9,
                )
                * 100.0
            ),
        ),
    )

    return cmp_df, summary.sort_values("wape_pct", ascending=False)


def _relative_divergence_pct(model: pd.Series, reference: pd.Series) -> pd.Series:
    denom = np.maximum(
        np.maximum(model.abs().astype(float), reference.abs().astype(float)),
        1e-9,
    )
    return (model.astype(float) - reference.astype(float)).abs() / denom * 100.0


def _flag_summary_by_group(
    cmp_df: pd.DataFrame,
    *,
    group_col: str,
    model_col: str,
    reference_col: str,
    abs_error_col: str,
    divergence_col: str,
    flag_col: str,
    ape_col: str,
) -> pd.DataFrame:
    rows = []
    for group_value, group in cmp_df.groupby(group_col):
        reference_total = float(group[reference_col].sum())
        rows.append(
            {
                group_col: group_value,
                "entries": int(len(group)),
                "flagged_entries": int(group[flag_col].sum()),
                "flag_rate_pct": float(group[flag_col].mean() * 100.0) if len(group) else 0.0,
                "model_total": float(group[model_col].sum()),
                "reference_total": reference_total,
                "wape_pct": (
                    float(group[abs_error_col].sum()) / max(reference_total, 1e-9) * 100.0
                ),
                "median_divergence_pct": float(group[divergence_col].median()),
                "p90_divergence_pct": float(group[divergence_col].quantile(0.90)),
                "mean_ape_pct": float(group[ape_col].mean(skipna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values(["flag_rate_pct", "wape_pct"], ascending=False)


def _round_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Round exported CSV values for readability.

    Rules:
    - TWh/MW/GW and error magnitude columns: 1 decimal
    - Percentage/rate columns: 2 decimals
    - Integer count columns are left untouched
    """
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in {"entries", "countries", "flagged_entries"}:
            continue
        lower = col.lower()
        if "pct" in lower or "rate" in lower:
            out[col] = out[col].round(2)
            continue
        if any(
            token in lower
            for token in [
                "_twh",
                "_mw",
                "_gw",
                "model_total",
                "reference_total",
                "mae",
                "rmse",
                "error",
                "divergence",
            ]
        ):
            out[col] = out[col].round(1)
            continue
        out[col] = out[col].round(3)
    return out


def _write_compact_validation_outputs(
    cap_cmp: pd.DataFrame,
    elec_cmp: pd.DataFrame,
    output_dir: Path,
    *,
    capacity_flag_threshold_pct: float,
    electricity_flag_threshold_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cap_focus = cap_cmp.copy()
    cap_focus["divergence_pct"] = _relative_divergence_pct(
        cap_focus["capacity_mw"], cap_focus["reference_mw"]
    )
    cap_focus["flag"] = cap_focus["divergence_pct"] > float(capacity_flag_threshold_pct)

    cap_summary_focus = _flag_summary_by_group(
        cap_focus,
        group_col="validation_tech",
        model_col="capacity_mw",
        reference_col="reference_mw",
        abs_error_col="abs_error_mw",
        divergence_col="divergence_pct",
        flag_col="flag",
        ape_col="ape_pct",
    )
    cap_flags = cap_focus.loc[cap_focus["flag"]].sort_values(
        ["divergence_pct", "abs_error_mw"], ascending=False
    )

    elec_focus = elec_cmp.loc[elec_cmp["metric"].isin(ELECTRICITY_TECH_METRICS)].copy()
    elec_focus["divergence_pct"] = _relative_divergence_pct(
        elec_focus["model_twh"], elec_focus["reference_twh"]
    )
    elec_focus["flag"] = elec_focus["divergence_pct"] > float(electricity_flag_threshold_pct)

    elec_summary_focus = _flag_summary_by_group(
        elec_focus,
        group_col="metric",
        model_col="model_twh",
        reference_col="reference_twh",
        abs_error_col="abs_error_twh",
        divergence_col="divergence_pct",
        flag_col="flag",
        ape_col="ape_pct",
    )
    elec_flags = elec_focus.loc[elec_focus["flag"]].sort_values(
        ["divergence_pct", "abs_error_twh"], ascending=False
    )

    _round_for_csv(cap_summary_focus).to_csv(
        output_dir / "capacity_focus_summary_by_technology.csv", index=False
    )
    _round_for_csv(cap_flags).to_csv(
        output_dir
        / f"capacity_flags_country_technology_gt{int(capacity_flag_threshold_pct)}pct.csv",
        index=False,
    )

    _round_for_csv(elec_summary_focus).to_csv(
        output_dir / "electricity_balance_focus_summary_by_technology.csv", index=False
    )
    _round_for_csv(elec_flags).to_csv(
        output_dir
        / f"electricity_balance_flags_country_technology_gt{int(electricity_flag_threshold_pct)}pct.csv",
        index=False,
    )

    return cap_summary_focus, cap_flags, elec_summary_focus, elec_flags


def _print_summary(
    cap_cmp: pd.DataFrame,
    cap_summary: pd.DataFrame,
    elec_energy_cmp: pd.DataFrame,
    elec_energy_summary: pd.DataFrame,
    non_elec_cmp: pd.DataFrame,
    non_elec_summary: pd.DataFrame,
    demand_cmp: pd.DataFrame,
    demand_summary: pd.DataFrame,
) -> None:
    total_capacity_wape = (
        cap_cmp["abs_error_mw"].sum() / max(cap_cmp["reference_mw"].sum(), 1e-9) * 100.0
    )
    total_elec_wape = (
        elec_energy_cmp["abs_error_twh"].sum()
        / max(elec_energy_cmp["reference_twh"].sum(), 1e-9)
        * 100.0
    )
    total_non_elec_wape = (
        non_elec_cmp["abs_error_twh"].sum()
        / max(non_elec_cmp["reference_twh"].sum(), 1e-9)
        * 100.0
    )
    total_demand_wape = (
        demand_cmp["abs_error_twh"].sum()
        / max(demand_cmp["reference_twh"].sum(), 1e-9)
        * 100.0
    )

    LOGGER.info("Capacity validation summary (sorted by WAPE):")
    for _, row in cap_summary.iterrows():
        LOGGER.info(
            "  %-22s model=%10.1f MW ref=%10.1f MW wape=%7.2f%% mean_ape=%7.2f%%",
            row["validation_tech"],
            row["model_total_mw"],
            row["reference_total_mw"],
            row["wape_pct"],
            row["mean_ape_pct"],
        )
    LOGGER.info("  [overall] weighted error (capacity) ~= %.2f%%", total_capacity_wape)

    LOGGER.info("Electricity balance validation summary (sorted by WAPE):")
    for _, row in elec_energy_summary.iterrows():
        LOGGER.info(
            "  %-25s model=%10.1f TWh ref=%10.1f TWh wape=%7.2f%% mean_ape=%7.2f%%",
            row["metric"],
            row["model_total_twh"],
            row["reference_total_twh"],
            row["wape_pct"],
            row["mean_ape_pct"],
        )
    LOGGER.info("  [overall] weighted error (electricity balance) ~= %.2f%%", total_elec_wape)

    LOGGER.info("Non-electric fossil energy summary (sorted by WAPE):")
    for _, row in non_elec_summary.iterrows():
        LOGGER.info(
            "  %-31s model=%10.1f TWh ref=%10.1f TWh wape=%7.2f%% mean_ape=%7.2f%%",
            row["metric"],
            row["model_total_twh"],
            row["reference_total_twh"],
            row["wape_pct"],
            row["mean_ape_pct"],
        )
    LOGGER.info("  [overall] weighted error (non-electric fossil) ~= %.2f%%", total_non_elec_wape)

    LOGGER.info("Electricity demand validation summary (sorted by WAPE):")
    for _, row in demand_summary.iterrows():
        LOGGER.info(
            "  %-25s model=%10.1f TWh ref=%10.1f TWh wape=%7.2f%% mean_ape=%7.2f%%",
            row["metric"],
            row["model_total_twh"],
            row["reference_total_twh"],
            row["wape_pct"],
            row["mean_ape_pct"],
        )
    LOGGER.info("  [overall] weighted error (electricity demand) ~= %.2f%%", total_demand_wape)


def parse_args() -> argparse.Namespace:
    default_network = (
        "/Users/tlm2160/Library/Mobile Documents/com~apple~CloudDocs/Documents/Research 2/CKI/energmod/"
        "results/Global_200/postnetworks/elec_s_200_ec_lcopt_1h_1h_2020_0.071_AB_0.0export_base.nc"
    )
    parser = argparse.ArgumentParser(
        description="Validate solved network capacities and energy balances by country."
    )
    parser.add_argument("--network", type=Path, default=Path(default_network), help="Path to solved .nc network")
    parser.add_argument(
        "--irena-csv",
        type=Path,
        default=REPO_ROOT / "validation" / "data" / "irena_capacity_by_technology.csv",
        help="Path to IRENA capacity CSV",
    )
    parser.add_argument(
        "--iaea-csv",
        type=Path,
        default=REPO_ROOT / "validation" / "data" / "Nuclear_capacity_IEAE.csv",
        help="Path to IAEA nuclear capacity CSV",
    )
    parser.add_argument(
        "--gem-clean-csv",
        type=Path,
        default=REPO_ROOT / "validation" / "data" / "gem_capacity_reference_2020.csv",
        help="Path to cleaned GEM capacity CSV from prepare_gem_capacity_reference.py",
    )
    parser.add_argument(
        "--owid-csv",
        type=Path,
        default=REPO_ROOT / "validation" / "data" / "owid-energy-data.csv",
        help="Path to OWID energy CSV",
    )
    parser.add_argument("--year", type=int, default=2020, help="Validation year")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "validation" / "results",
        help="Directory for validation outputs",
    )
    parser.add_argument(
        "--capacity-flag-threshold-pct",
        type=float,
        default=20.0,
        help="Flag capacity country-technology entries above this absolute divergence percent",
    )
    parser.add_argument(
        "--electricity-flag-threshold-pct",
        type=float,
        default=10.0,
        help="Flag electricity balance country-technology entries above this absolute divergence percent",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading network: %s", args.network)
    n = pypsa.Network(args.network)

    network_countries = set(n.buses.country.dropna())
    network_countries.discard("")
    network_iso3 = {
        two_2_three_digits_country(c) for c in network_countries if isinstance(c, str) and len(c) == 2
    }
    network_iso3 = {c for c in network_iso3 if isinstance(c, str) and len(c) == 3}

    model_cap, unmapped_cap = _model_capacity_by_country_tech(n)
    model_cap = model_cap.loc[model_cap.country.isin(network_countries)].copy()
    ref_cap = _capacity_reference_by_country_tech(
        args.irena_csv,
        args.iaea_csv,
        args.gem_clean_csv,
        args.year,
    )
    ref_cap = ref_cap.loc[ref_cap.country.isin(network_countries)].copy()

    active_capacity_techs = set(model_cap["validation_tech"].unique())
    dropped_ref_techs = sorted(set(ref_cap["validation_tech"].unique()) - active_capacity_techs)
    ref_cap = ref_cap.loc[ref_cap["validation_tech"].isin(active_capacity_techs)].copy()
    if dropped_ref_techs:
        LOGGER.info(
            "Skipping reference-only capacity categories not represented in model carriers: %s",
            ", ".join(dropped_ref_techs),
        )

    cap_cmp, cap_summary = _capacity_comparison(model_cap, ref_cap)

    model_elec = _model_electricity_balance(n)
    model_elec = model_elec.loc[model_elec.country.isin(network_countries)].copy()
    ref_elec = _owid_electricity_balance(args.owid_csv, args.year, iso3_filter=network_iso3)
    ref_elec = ref_elec.loc[ref_elec.country.isin(network_countries)].copy()
    elec_cmp, elec_summary = _energy_comparison(model_elec, ref_elec)

    model_non_elec = _model_non_electric_fossil_balance(n)
    model_non_elec = model_non_elec.loc[model_non_elec.country.isin(network_countries)].copy()
    ref_non_elec = _owid_non_electric_fossil_balance(
        args.owid_csv, args.year, iso3_filter=network_iso3
    )
    ref_non_elec = ref_non_elec.loc[ref_non_elec.country.isin(network_countries)].copy()
    non_elec_cmp, non_elec_summary = _energy_comparison(model_non_elec, ref_non_elec)

    model_demand = _model_electricity_demand(n)
    model_demand = model_demand.loc[model_demand.country.isin(network_countries)].copy()
    ref_demand = _owid_electricity_demand(args.owid_csv, args.year, iso3_filter=network_iso3)
    ref_demand = ref_demand.loc[ref_demand.country.isin(network_countries)].copy()
    demand_cmp, demand_summary = _energy_comparison(model_demand, ref_demand)

    _round_for_csv(cap_cmp).to_csv(
        args.output_dir / "capacity_comparison_country_technology.csv", index=False
    )
    _round_for_csv(cap_summary).to_csv(args.output_dir / "capacity_summary_by_technology.csv", index=False)
    _round_for_csv(elec_cmp).to_csv(args.output_dir / "energy_comparison_country_metric.csv", index=False)
    _round_for_csv(elec_summary).to_csv(args.output_dir / "energy_summary_by_metric.csv", index=False)
    _round_for_csv(non_elec_cmp).to_csv(
        args.output_dir / "fossil_non_electric_energy_comparison_country_metric.csv", index=False
    )
    _round_for_csv(non_elec_summary).to_csv(
        args.output_dir / "fossil_non_electric_energy_summary_by_metric.csv", index=False
    )
    _round_for_csv(demand_cmp).to_csv(
        args.output_dir / "electricity_demand_comparison_country_metric.csv", index=False
    )
    _round_for_csv(demand_summary).to_csv(
        args.output_dir / "electricity_demand_summary_by_metric.csv", index=False
    )
    _round_for_csv(unmapped_cap).to_csv(
        args.output_dir / "unmapped_electricity_carriers_capacity.csv", index=False
    )

    (
        cap_summary_focus,
        cap_flags,
        elec_summary_focus,
        elec_flags,
    ) = _write_compact_validation_outputs(
        cap_cmp,
        elec_cmp,
        args.output_dir,
        capacity_flag_threshold_pct=args.capacity_flag_threshold_pct,
        electricity_flag_threshold_pct=args.electricity_flag_threshold_pct,
    )

    LOGGER.info("Wrote validation outputs to %s", args.output_dir.resolve())
    LOGGER.info(
        "Capacity flags (>%.1f%% divergence): %d / %d entries",
        args.capacity_flag_threshold_pct,
        len(cap_flags),
        len(cap_cmp),
    )
    LOGGER.info(
        "Electricity-balance flags (>%.1f%% divergence): %d / %d entries (technology metrics only)",
        args.electricity_flag_threshold_pct,
        len(elec_flags),
        len(elec_cmp.loc[elec_cmp.metric.isin(ELECTRICITY_TECH_METRICS)]),
    )
    if not cap_summary_focus.empty:
        LOGGER.info(
            "Top capacity flag rates by technology:\n%s",
            cap_summary_focus.head(8).to_string(index=False),
        )
    if not elec_summary_focus.empty:
        LOGGER.info(
            "Top electricity-balance flag rates by technology:\n%s",
            elec_summary_focus.head(8).to_string(index=False),
        )
    if not unmapped_cap.empty:
        LOGGER.info(
            "Unmapped electricity-output carriers (not compared to IRENA):\n%s",
            unmapped_cap.head(20).to_string(index=False),
        )

    LOGGER.info(
        "Non-electric OWID reference is computed as fuel consumption minus fuel electricity output."
    )
    _print_summary(
        cap_cmp,
        cap_summary,
        elec_cmp,
        elec_summary,
        non_elec_cmp,
        non_elec_summary,
        demand_cmp,
        demand_summary,
    )


if __name__ == "__main__":
    main()
