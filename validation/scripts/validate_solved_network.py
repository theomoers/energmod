#!/usr/bin/env python3
"""
Validate a solved PyPSA network against IRENA capacities and OWID electricity balances.

The script compares:
1. Installed/optimized electric capacity (MW) by country and aggregated technology group.
2. Electricity generation balance (TWh) by country and source categories.
"""

from __future__ import annotations

import argparse
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

from _helpers import three_2_two_digits_country, two_2_three_digits_country


LOGGER = logging.getLogger(__name__)


# IRENA technology categories -> validation categories
IRENA_TO_VALIDATION_TECH = {
    "PV": "solar",
    "Onshore": "onshore_wind",
    "Offshore": "offshore_wind",
    "Renewable hydropower": "hydro",
    "Pumped Storage": "hydro",
    "Mixed Hydro Plants": "hydro",
    "Geothermal": "geothermal",
    "Nuclear": "nuclear",
    "Natural gas": "gas",
    "Coal and peat": "coal",
    "Oil": "oil",
    "Biogas": "bioenergy",
    "Liquid biofuels": "bioenergy",
    "Solid biofuels": "bioenergy",
    "Renewable municipal waste": "bioenergy",
    "CSP": "other_renewables",
    "Marine": "other_renewables",
    "Fossil fuels n.e.s.": "other_non_renewables",
    "Other non-renewable energy": "other_non_renewables",
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


def _capacity_column(df: pd.DataFrame) -> str:
    if "p_nom_opt" in df.columns and df["p_nom_opt"].notna().any():
        return "p_nom_opt"
    return "p_nom"


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


def _link_electric_capacity(n: pypsa.Network) -> pd.DataFrame:
    cap_col = _capacity_column(n.links)
    link_df = n.links.copy()

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
        if not mask.any():
            continue

        sub = link_df.loc[mask, [cap_col, eff_col, "carrier", bus_col]].copy()
        sub["country"] = sub[bus_col].map(n.buses.country).fillna("")
        sub["component"] = "Link"
        sub["capacity_mw"] = sub[cap_col].astype(float) * sub[eff_col].astype(float).abs()
        sub = sub.loc[sub["capacity_mw"] > 0, ["component", "country", "carrier", "capacity_mw"]]
        pieces.append(sub)

    if not pieces:
        return pd.DataFrame(columns=["component", "country", "carrier", "capacity_mw"])

    return pd.concat(pieces, ignore_index=True)


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

    gen = n.generators.copy()
    gen = gen.loc[gen.bus.map(n.buses.carrier).eq("AC")].copy()
    gen["component"] = "Generator"
    gen["country"] = gen.bus.map(n.buses.country).fillna("")
    gen["capacity_mw"] = gen[gen_col].astype(float)
    gen = gen.loc[:, ["component", "country", "carrier", "capacity_mw"]]
    gen = gen.loc[gen.capacity_mw > 0]

    su = n.storage_units.copy()
    su = su.loc[su.bus.map(n.buses.carrier).eq("AC")].copy()
    su["component"] = "StorageUnit"
    su["country"] = su.bus.map(n.buses.country).fillna("")
    su["capacity_mw"] = su[su_col].astype(float)
    su = su.loc[:, ["component", "country", "carrier", "capacity_mw"]]
    su = su.loc[su.capacity_mw > 0]

    links = _link_electric_capacity(n)

    model_raw = pd.concat([gen, su, links], ignore_index=True)
    model_raw = model_raw.loc[model_raw.country.ne("")]

    model_raw["validation_tech"] = model_raw["carrier"].map(MODEL_CAPACITY_TECH_MAP)
    mapped = model_raw.loc[model_raw.validation_tech.notna()].copy()
    mapped = (
        mapped.groupby(["country", "validation_tech"], as_index=False)["capacity_mw"]
        .sum()
    )

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
    usecols = ["year", "iso_code"] + OWID_ENERGY_COLUMNS
    owid = pd.read_csv(owid_csv, usecols=lambda c: c in set(usecols))
    owid = owid.loc[owid["year"] == year].copy()
    if iso3_filter is not None:
        owid = owid.loc[owid["iso_code"].isin(iso3_filter)].copy()
    owid["country"] = owid["iso_code"].apply(_safe_iso3_to_iso2)
    owid = owid.loc[owid.country.notna()].copy()

    for col in OWID_ENERGY_COLUMNS:
        if col in owid.columns:
            owid[col] = pd.to_numeric(owid[col], errors="coerce")

    keep_cols = [c for c in OWID_ENERGY_COLUMNS if c in owid.columns]
    owid = owid.groupby("country", as_index=False)[keep_cols].sum(min_count=1)
    out = owid.melt(
        id_vars="country",
        value_vars=keep_cols,
        var_name="metric",
        value_name="reference_twh",
    )
    return out


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


def _print_summary(
    cap_cmp: pd.DataFrame,
    cap_summary: pd.DataFrame,
    energy_cmp: pd.DataFrame,
    energy_summary: pd.DataFrame,
) -> None:
    total_capacity_wape = (
        cap_cmp["abs_error_mw"].sum() / max(cap_cmp["reference_mw"].sum(), 1e-9) * 100.0
    )
    total_energy_wape = (
        energy_cmp["abs_error_twh"].sum() / max(energy_cmp["reference_twh"].sum(), 1e-9) * 100.0
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
    for _, row in energy_summary.iterrows():
        LOGGER.info(
            "  %-25s model=%10.1f TWh ref=%10.1f TWh wape=%7.2f%% mean_ape=%7.2f%%",
            row["metric"],
            row["model_total_twh"],
            row["reference_total_twh"],
            row["wape_pct"],
            row["mean_ape_pct"],
        )
    LOGGER.info("  [overall] weighted error (electricity balance) ~= %.2f%%", total_energy_wape)


def parse_args() -> argparse.Namespace:
    default_network = (
        "/Users/tlm2160/Library/Mobile Documents/com~apple~CloudDocs/Documents/Research 2/CKI/energmod/"
        "results/Global_200/postnetworks/elec_s_200_ec_lcopt_1h_1h_2020_0.071_AB_0.0export_base.nc"
    )
    parser = argparse.ArgumentParser(
        description="Validate solved network capacities and electricity balance by country."
    )
    parser.add_argument("--network", type=Path, default=Path(default_network), help="Path to solved .nc network")
    parser.add_argument(
        "--irena-csv",
        type=Path,
        default=REPO_ROOT / "validation" / "data" / "irena_capacity_by_technology.csv",
        help="Path to IRENA capacity CSV",
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
    ref_cap = _irena_capacity_by_country_tech(args.irena_csv, args.year)
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

    model_energy = _model_electricity_balance(n)
    model_energy = model_energy.loc[model_energy.country.isin(network_countries)].copy()
    ref_energy = _owid_electricity_balance(args.owid_csv, args.year, iso3_filter=network_iso3)
    ref_energy = ref_energy.loc[ref_energy.country.isin(network_countries)].copy()
    energy_cmp, energy_summary = _energy_comparison(model_energy, ref_energy)

    cap_cmp.to_csv(args.output_dir / "capacity_comparison_country_technology.csv", index=False)
    cap_summary.to_csv(args.output_dir / "capacity_summary_by_technology.csv", index=False)
    energy_cmp.to_csv(args.output_dir / "energy_comparison_country_metric.csv", index=False)
    energy_summary.to_csv(args.output_dir / "energy_summary_by_metric.csv", index=False)
    unmapped_cap.to_csv(args.output_dir / "unmapped_electricity_carriers_capacity.csv", index=False)

    LOGGER.info("Wrote validation outputs to %s", args.output_dir.resolve())
    if not unmapped_cap.empty:
        LOGGER.info(
            "Unmapped electricity-output carriers (not compared to IRENA):\n%s",
            unmapped_cap.head(20).to_string(index=False),
        )

    _print_summary(cap_cmp, cap_summary, energy_cmp, energy_summary)


if __name__ == "__main__":
    main()
