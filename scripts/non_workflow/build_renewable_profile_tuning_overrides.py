#!/usr/bin/env python3
"""Update country-level renewable-profile tuning overrides from a solved network.

The update targets solved country generation, not pre-solve availability. It
can be used repeatedly, compounding damped corrections from the existing
override file.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pycountry
import pypsa


def iso3_to_iso2(code):
    code = str(code).strip().upper()
    special = {"XKX": "XK", "TWN": "TW", "PSE": "PS"}
    if code in special:
        return special[code]
    country = pycountry.countries.get(alpha_3=code)
    return country.alpha_2 if country is not None else None


def snapshot_weights(n):
    weights = n.snapshot_weightings
    if isinstance(weights, pd.DataFrame):
        if "generators" in weights.columns:
            return weights["generators"]
        if "objective" in weights.columns:
            return weights["objective"]
        return weights.iloc[:, 0]
    if isinstance(weights, pd.Series):
        return weights
    return pd.Series(1.0, index=n.snapshots)


def bus_country_lookup(n):
    if "country" in n.buses.columns:
        country = n.buses["country"].fillna("").astype(str).str.strip().str.upper()
    else:
        country = pd.Series("", index=n.buses.index)
    missing = country.eq("")
    if missing.any():
        country.loc[missing] = pd.Index(country.index[missing]).astype(str).str[:2].str.upper()
    return country


def model_generation_by_country(n):
    weights = snapshot_weights(n)
    bus_country = bus_country_lookup(n)
    rows = []
    if not n.generators.empty:
        dispatch = n.generators_t.p.reindex(columns=n.generators.index, fill_value=0.0).clip(lower=0.0)
        energy = dispatch.mul(weights, axis=0).sum(axis=0) / 1e6
        gen = n.generators.copy()
        gen["country"] = gen["bus"].map(bus_country).fillna(gen["bus"].astype(str).str[:2].str.upper())
        gen["energy_twh"] = energy.reindex(gen.index).fillna(0.0)
        carrier = gen["carrier"].astype(str)
        for tech, mask in {
            "solar": carrier.eq("solar"),
            "onwind": carrier.eq("onwind"),
            "offwind": carrier.isin(["offwind-ac", "offwind-dc"]),
            "nuclear": carrier.eq("nuclear"),
            "hydro": carrier.eq("ror"),
        }.items():
            part = gen.loc[mask].groupby("country", as_index=False)["energy_twh"].sum()
            part["technology"] = tech
            rows.append(part.rename(columns={"energy_twh": "model_twh"}))
    if not n.storage_units.empty:
        dispatch = n.storage_units_t.p.reindex(columns=n.storage_units.index, fill_value=0.0).clip(lower=0.0)
        energy = dispatch.mul(weights, axis=0).sum(axis=0) / 1e6
        hydro = n.storage_units.loc[n.storage_units.carrier.astype(str).eq("hydro")].copy()
        hydro["country"] = hydro["bus"].map(bus_country).fillna(hydro["bus"].astype(str).str[:2].str.upper())
        hydro["energy_twh"] = energy.reindex(hydro.index).fillna(0.0)
        part = hydro.groupby("country", as_index=False)["energy_twh"].sum()
        part["technology"] = "hydro"
        rows.append(part.rename(columns={"energy_twh": "model_twh"}))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if out.empty:
        return pd.DataFrame(columns=["country", "technology", "model_twh"])
    return out[["country", "technology", "model_twh"]]


def ember_reference(ember_csv, year):
    ember = pd.read_csv(ember_csv, low_memory=False)
    ref = ember.loc[
        pd.to_numeric(ember["Year"], errors="coerce").le(int(year))
        & ember["Category"].eq("Electricity generation")
        & ember["Unit"].eq("TWh")
        & ember["Area type"].eq("Country or economy")
        & ember["Variable"].isin(["Solar", "Wind", "Nuclear", "Hydro"])
    ].dropna(subset=["ISO 3 code"]).copy()
    ref["Year"] = pd.to_numeric(ref["Year"], errors="coerce")
    ref = (
        ref.sort_values("Year")
        .groupby(["ISO 3 code", "Variable"], as_index=False)
        .tail(1)
    )
    if int(year) >= 2025:
        ref = ref.loc[ref["Year"].ge(2023)].copy()
    ref["country"] = ref["ISO 3 code"].map(iso3_to_iso2)
    ref = ref.loc[ref["country"].notna()].copy()
    ref["technology"] = ref["Variable"].map({"Solar": "solar", "Wind": "wind", "Nuclear": "nuclear", "Hydro": "hydro"})
    ref["reference_twh"] = pd.to_numeric(ref["Value"], errors="coerce")
    ref = ref.dropna(subset=["reference_twh"])
    return ref.groupby(["country", "technology"], as_index=False).agg(
        reference_twh=("reference_twh", "sum"),
        reference_year=("Year", "min"),
    )


def pris_nuclear_reference(pris_csv, year):
    """Return the PRIS country-generation reference used by the LaTeX table."""
    pris = pd.read_csv(pris_csv)
    required = {"generation_year", "country_iso2", "pris_nuclear_generation_twh"}
    missing = required.difference(pris.columns)
    if missing:
        raise ValueError(f"PRIS reference is missing columns: {sorted(missing)}")
    pris["generation_year"] = pd.to_numeric(pris["generation_year"], errors="coerce")
    pris = pris.loc[pris["generation_year"].eq(int(year))].copy()
    if pris.empty:
        return pd.DataFrame(columns=["country", "reference_twh", "reference_year"])
    pris["country"] = pris["country_iso2"].astype(str).str.upper().str.strip()
    pris["reference_twh"] = pd.to_numeric(
        pris["pris_nuclear_generation_twh"], errors="coerce"
    )
    pris = pris.loc[pris["country"].str.len().eq(2)].dropna(subset=["reference_twh"])
    return pris.groupby("country", as_index=False).agg(
        reference_twh=("reference_twh", "sum"),
        reference_year=("generation_year", "min"),
    )


def load_existing(path):
    cols = ["year", "country", "solar_scale", "onwind_scale", "offwind_scale", "nuclear_scale", "hydro_scale"]
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path)
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[cols].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["country"] = df["country"].astype(str).str.upper().str.strip()
    return df.dropna(subset=["year"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", required=True)
    ap.add_argument("--year", required=True, type=int)
    ap.add_argument("--output", default="validation/data/renewable_profile_tuning_overrides.csv")
    ap.add_argument("--existing", default="validation/data/renewable_profile_tuning_overrides.csv")
    ap.add_argument("--ember", default="validation/data/ember_yearly_full_release_long_format.csv")
    ap.add_argument("--pris", default="validation/data/pris_nuclear_validation.csv")
    ap.add_argument(
        "--reset-year",
        action="store_true",
        help="Derive the selected year factors from 1.0 rather than compounding existing overrides.",
    )
    ap.add_argument("--scale-min", default=0.5, type=float)
    ap.add_argument("--scale-max", default=2.0, type=float)
    ap.add_argument("--solar-scale-min", default=None, type=float)
    ap.add_argument("--solar-scale-max", default=None, type=float)
    ap.add_argument("--wind-scale-min", default=None, type=float)
    ap.add_argument("--wind-scale-max", default=None, type=float)
    ap.add_argument("--nuclear-scale-min", default=None, type=float)
    ap.add_argument("--nuclear-scale-max", default=None, type=float)
    ap.add_argument("--hydro-scale-min", default=None, type=float)
    ap.add_argument("--hydro-scale-max", default=None, type=float)
    ap.add_argument("--min-reference-twh", default=0.1, type=float)
    ap.add_argument("--min-model-twh", default=0.1, type=float)
    ap.add_argument(
        "--damping",
        default=1.0,
        type=float,
        help="Exponent on the solved-generation correction ratio; 1.0 is a full update.",
    )
    args = ap.parse_args()
    if not 0.0 <= args.damping <= 1.0:
        ap.error("--damping must lie between 0 and 1")

    n = pypsa.Network(args.network)
    model = model_generation_by_country(n)
    solar_model = model.loc[model["technology"].eq("solar"), ["country", "model_twh"]].rename(columns={"model_twh": "solar_model_twh"})
    wind_model = model.loc[model["technology"].isin(["onwind", "offwind"])].groupby("country", as_index=False)["model_twh"].sum().rename(columns={"model_twh": "wind_model_twh"})
    nuclear_model = model.loc[model["technology"].eq("nuclear"), ["country", "model_twh"]].rename(columns={"model_twh": "nuclear_model_twh"})
    hydro_model = model.loc[model["technology"].eq("hydro")].groupby("country", as_index=False)["model_twh"].sum().rename(columns={"model_twh": "hydro_model_twh"})
    ref = ember_reference(args.ember, args.year)

    def technology_reference(technology):
        cols = ["country", "reference_twh", "reference_year"]
        selected = ref.loc[ref["technology"].eq(technology), cols].copy()
        return selected.rename(columns={
            "reference_twh": f"{technology}_reference_twh",
            "reference_year": f"{technology}_reference_year",
        })

    solar_ref = technology_reference("solar")
    wind_ref = technology_reference("wind")
    hydro_ref = technology_reference("hydro")
    pris_ref = pris_nuclear_reference(args.pris, args.year)
    if int(args.year) >= 2025 and not pris_ref.empty:
        nuclear_ref = pris_ref.rename(columns={
            "reference_twh": "nuclear_reference_twh",
            "reference_year": "nuclear_reference_year",
        })
        nuclear_reference_source = "PRIS"
    else:
        nuclear_ref = technology_reference("nuclear")
        nuclear_reference_source = "Ember"

    countries = sorted(set(solar_model.country) | set(wind_model.country) | set(nuclear_model.country) | set(hydro_model.country) | set(solar_ref.country) | set(wind_ref.country) | set(nuclear_ref.country) | set(hydro_ref.country))
    out = pd.DataFrame({"country": countries})
    out = out.merge(solar_ref, on="country", how="left").merge(solar_model, on="country", how="left")
    out = out.merge(wind_ref, on="country", how="left").merge(wind_model, on="country", how="left")
    out = out.merge(nuclear_ref, on="country", how="left").merge(nuclear_model, on="country", how="left")
    out = out.merge(hydro_ref, on="country", how="left").merge(hydro_model, on="country", how="left")
    out["solar_reference_source"] = "Ember"
    out["wind_reference_source"] = "Ember"
    out["hydro_reference_source"] = "Ember"
    out["nuclear_reference_source"] = nuclear_reference_source

    existing = load_existing(args.existing)
    current = existing.loc[existing["year"].astype("Int64").eq(args.year)].copy()
    if args.reset_year:
        current = pd.DataFrame(columns=current.columns)
    current = current.groupby("country", as_index=False)[["solar_scale", "onwind_scale", "offwind_scale", "nuclear_scale", "hydro_scale"]].mean() if not current.empty else pd.DataFrame(columns=["country", "solar_scale", "onwind_scale", "offwind_scale", "nuclear_scale", "hydro_scale"])
    out = out.merge(current, on="country", how="left")
    for col in ["solar_scale", "onwind_scale", "offwind_scale", "nuclear_scale", "hydro_scale"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(1.0)

    solar_valid = out["solar_reference_twh"].fillna(0.0).ge(args.min_reference_twh) & out["solar_model_twh"].fillna(0.0).ge(args.min_model_twh)
    wind_valid = out["wind_reference_twh"].fillna(0.0).ge(args.min_reference_twh) & out["wind_model_twh"].fillna(0.0).ge(args.min_model_twh)
    nuclear_valid = out["nuclear_reference_twh"].fillna(0.0).ge(args.min_reference_twh) & out["nuclear_model_twh"].fillna(0.0).ge(args.min_model_twh)
    hydro_valid = out["hydro_reference_twh"].fillna(0.0).ge(args.min_reference_twh) & out["hydro_model_twh"].fillna(0.0).ge(args.min_model_twh)
    solar_ratio = out["solar_reference_twh"] / out["solar_model_twh"]
    wind_ratio = out["wind_reference_twh"] / out["wind_model_twh"]
    nuclear_ratio = out["nuclear_reference_twh"] / out["nuclear_model_twh"]
    hydro_ratio = out["hydro_reference_twh"] / out["hydro_model_twh"]

    def damped_update(scale, ratio, valid):
        return np.where(valid, scale * np.power(ratio, args.damping), scale)

    out["solar_scale_new"] = damped_update(out["solar_scale"], solar_ratio, solar_valid)
    out["onwind_scale_new"] = damped_update(out["onwind_scale"], wind_ratio, wind_valid)
    out["offwind_scale_new"] = damped_update(out["offwind_scale"], wind_ratio, wind_valid)
    out["nuclear_scale_new"] = damped_update(out["nuclear_scale"], nuclear_ratio, nuclear_valid)
    out["hydro_scale_new"] = damped_update(out["hydro_scale"], hydro_ratio, hydro_valid)
    solar_min = args.scale_min if args.solar_scale_min is None else args.solar_scale_min
    solar_max = args.scale_max if args.solar_scale_max is None else args.solar_scale_max
    wind_min = args.scale_min if args.wind_scale_min is None else args.wind_scale_min
    wind_max = args.scale_max if args.wind_scale_max is None else args.wind_scale_max
    nuclear_min = args.scale_min if args.nuclear_scale_min is None else args.nuclear_scale_min
    nuclear_max = args.scale_max if args.nuclear_scale_max is None else args.nuclear_scale_max
    hydro_min = args.scale_min if args.hydro_scale_min is None else args.hydro_scale_min
    hydro_max = args.scale_max if args.hydro_scale_max is None else args.hydro_scale_max
    out["solar_scale_new"] = pd.to_numeric(out["solar_scale_new"], errors="coerce").fillna(1.0).clip(solar_min, solar_max)
    for col in ["onwind_scale_new", "offwind_scale_new"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(1.0).clip(wind_min, wind_max)
    out["nuclear_scale_new"] = pd.to_numeric(out["nuclear_scale_new"], errors="coerce").fillna(1.0).clip(nuclear_min, nuclear_max)
    out["hydro_scale_new"] = pd.to_numeric(out["hydro_scale_new"], errors="coerce").fillna(1.0).clip(hydro_min, hydro_max)

    updated = out[["country", "solar_scale_new", "onwind_scale_new", "offwind_scale_new", "nuclear_scale_new", "hydro_scale_new"]].rename(columns={
        "solar_scale_new": "solar_scale",
        "onwind_scale_new": "onwind_scale",
        "offwind_scale_new": "offwind_scale",
        "nuclear_scale_new": "nuclear_scale",
        "hydro_scale_new": "hydro_scale",
    })
    updated.insert(0, "year", args.year)

    keep = existing.loc[~existing["year"].astype("Int64").eq(args.year)].copy()
    final = pd.concat([keep, updated], ignore_index=True)
    final["year"] = final["year"].astype(int)
    final = final.sort_values(["year", "country"])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output, index=False)

    audit = out.copy()
    audit.insert(0, "year", args.year)
    audit["damping"] = args.damping
    audit_path = output.with_name(output.stem + f"_{args.year}_audit.csv")
    audit.to_csv(audit_path, index=False)
    print(f"wrote {output} ({len(final)} rows); audit={audit_path}")
    print("damping", args.damping, "solar tuned countries", int(solar_valid.sum()), "wind tuned countries", int(wind_valid.sum()), "nuclear tuned countries", int(nuclear_valid.sum()), "hydro tuned countries", int(hydro_valid.sum()))


if __name__ == "__main__":
    main()
