#!/usr/bin/env python3
"""Export compact CSV outputs for one stochastic model x seed run."""

import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa


OUTPUT_TABLE_SPECS = {
    "learning_costs.csv": [
        "year",
        "technology",
        "c_overnight",
        "capital_cost",
        "selected_model",
        "learning_seed",
        "training_window",
    ],
    "system_summary.csv": [
        "year",
        "objective_eur",
        "total_electricity_demand_mwh",
        "total_electricity_generation_mwh",
        "renewable_electricity_generation_mwh",
        "renewable_share",
        "curtailment_mwh",
        "storage_power_capacity_mw",
        "storage_energy_capacity_mwh",
        "transmission_ac_capacity_mva",
        "transmission_dc_capacity_mw",
        "transmission_ac_volume_mwkm",
        "transmission_dc_volume_mwkm",
        "total_emissions_mtco2",
    ],
    "generation_country_carrier.csv": [
        "year",
        "component",
        "country",
        "carrier",
        "annual_generation_mwh",
        "annual_generation_twh",
    ],
    "ac_energy_balance_country_carrier.csv": [
        "year",
        "country",
        "component",
        "carrier",
        "energy_balance_mwh",
        "energy_balance_twh",
    ],
    "capacity_country_carrier.csv": [
        "year",
        "component",
        "country",
        "carrier",
        "capacity_unit",
        "capacity_value",
    ],
    "battery_operations_country_year.csv": [
        "year",
        "country",
        "carrier",
        "charge_input_from_ac_mwh",
        "charge_to_store_mwh",
        "discharge_from_store_mwh",
        "discharge_to_ac_mwh",
        "net_ac_supply_mwh",
        "throughput_mwh",
        "charging_losses_mwh",
        "discharging_losses_mwh",
        "total_losses_mwh",
        "store_energy_capacity_mwh",
        "charger_power_capacity_mw",
        "discharger_power_capacity_mw",
        "estimated_cycles",
    ],
    "deployment_country_carrier.csv": [
        "year",
        "previous_year",
        "component",
        "country",
        "carrier",
        "capacity_unit",
        "build_value",
        "retire_value",
        "net_change_value",
    ],
    "deployment_node_carrier.csv": [
        "year",
        "previous_year",
        "component",
        "bus",
        "country",
        "carrier",
        "capacity_unit",
        "build_value",
        "retire_value",
        "net_change_value",
    ],
    "power_emissions_country_carrier.csv": [
        "year",
        "country",
        "carrier",
        "emissions_tco2",
        "emissions_mtco2",
        "emissions_gtco2",
    ],
    "sector_emissions.csv": [
        "year",
        "sector",
        "emissions_mtco2",
    ],
    "sector_demands_country.csv": [
        "year",
        "country",
        "sector",
        "annual_demand_mwh",
        "annual_demand_twh",
    ],
    "electricity_price_node_year.csv": [
        "year",
        "bus",
        "country",
        "mean_eur_per_mwh",
        "load_weighted_mean_eur_per_mwh",
        "min_eur_per_mwh",
        "p05_eur_per_mwh",
        "p25_eur_per_mwh",
        "p50_eur_per_mwh",
        "p75_eur_per_mwh",
        "p95_eur_per_mwh",
        "max_eur_per_mwh",
        "std_eur_per_mwh",
        "negative_hour_share",
        "positive_hour_share",
    ],
    "electricity_price_country_year.csv": [
        "year",
        "country",
        "mean_eur_per_mwh",
        "load_weighted_mean_eur_per_mwh",
        "min_eur_per_mwh",
        "p05_eur_per_mwh",
        "p25_eur_per_mwh",
        "p50_eur_per_mwh",
        "p75_eur_per_mwh",
        "p95_eur_per_mwh",
        "max_eur_per_mwh",
        "std_eur_per_mwh",
        "negative_hour_share",
        "positive_hour_share",
    ],
    "constraint_shadow_system_year.csv": [
        "year",
        "constraint_name",
        "sense",
        "carrier_attribute",
        "constant",
        "mu",
    ],
    "constraint_shadow_asset_year.csv": [
        "year",
        "component",
        "asset",
        "bus",
        "country",
        "carrier",
        "constraint_name",
        "mean_mu",
        "abs_mean_mu",
        "min_mu",
        "max_mu",
        "p95_abs_mu",
        "binding_hours_count",
        "positive_hours_count",
        "negative_hours_count",
    ],
    "lcoe_electricity_tech_year.csv": [
        "year",
        "component",
        "carrier",
        "electricity_output_mwh",
        "annualized_capital_cost_eur",
        "operating_cost_eur",
        "total_cost_eur",
        "lcoe_eur_per_mwh",
    ],
}

ELECTRIC_COMPONENTS = ["Generator", "StorageUnit", "Link", "Store"]
FOSSIL_CARRIER_MAP = {
    "CCGT": "gas",
    "OCGT": "gas",
    "gas": "gas",
    "urban central gas CHP": "gas",
    "urban central gas CHP CC": "gas",
    "coal": "coal",
    "lignite": "lignite",
    "oil": "oil",
}
RENEWABLE_HINTS = {
    "solar",
    "onwind",
    "offwind",
    "offwind-ac",
    "offwind-dc",
    "hydro",
    "ror",
    "geothermal",
    "biomass",
}
SHADOW_COMPONENT_MAP = {
    "Generator": "generators",
    "Link": "links",
    "Store": "stores",
    "StorageUnit": "storage_units",
    "Line": "lines",
    "Transformer": "transformers",
    "Load": "loads",
}


def _empty_frame(name: str) -> pd.DataFrame:
    return pd.DataFrame(columns=OUTPUT_TABLE_SPECS[name])


def _sanitize_json(value):
    if isinstance(value, dict):
        return {str(k): _sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _infer_year(path: Path, years: list[int]) -> int:
    for year in years:
        if f"_{year}_" in path.name:
            return int(year)
    raise ValueError(f"Could not infer planning horizon year from path: {path}")


def _map_paths_by_year(paths: list[str], years: list[int]) -> dict[int, Path]:
    mapping = {}
    for raw_path in paths:
        path = Path(raw_path)
        year = _infer_year(path, years)
        mapping[year] = path
    missing = [year for year in years if year not in mapping]
    if missing:
        raise FileNotFoundError(f"Missing required per-horizon inputs for years: {missing}")
    return mapping


def _snapshot_weights(n: pypsa.Network, attr: str = "objective") -> pd.Series:
    if hasattr(n.snapshot_weightings, attr):
        weights = getattr(n.snapshot_weightings, attr)
    elif hasattr(n.snapshot_weightings, "generators"):
        weights = n.snapshot_weightings.generators
    else:
        weights = pd.Series(1.0, index=n.snapshots)
    return pd.Series(weights, index=n.snapshots, dtype=float).fillna(0.0)


def _bus_country_lookup(n: pypsa.Network) -> pd.Series:
    if "country" in n.buses.columns:
        countries = n.buses["country"].fillna("").astype(str)
        if countries.ne("").any():
            return countries
    return pd.Series("", index=n.buses.index, dtype=object)


def _capacity_column(df: pd.DataFrame, preferred: str, fallback: str) -> str | None:
    if preferred in df.columns:
        return preferred
    if fallback in df.columns:
        return fallback
    return None


def _weighted_quantile(values: pd.Series, weights: pd.Series, quantile: float) -> float:
    series = pd.to_numeric(values, errors="coerce")
    weight_series = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    mask = series.notna() & weight_series.gt(0.0)
    if not mask.any():
        return float("nan")
    values_np = series.loc[mask].to_numpy(dtype=float)
    weights_np = weight_series.loc[mask].to_numpy(dtype=float)
    sorter = np.argsort(values_np)
    values_np = values_np[sorter]
    weights_np = weights_np[sorter]
    cumulative = np.cumsum(weights_np)
    threshold = quantile * cumulative[-1]
    idx = np.searchsorted(cumulative, threshold, side="left")
    idx = min(max(idx, 0), len(values_np) - 1)
    return float(values_np[idx])


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    series = pd.to_numeric(values, errors="coerce")
    weight_series = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    mask = series.notna() & weight_series.gt(0.0)
    if not mask.any():
        return float("nan")
    total_weight = float(weight_series.loc[mask].sum())
    if total_weight <= 0.0:
        return float("nan")
    return float(np.average(series.loc[mask].to_numpy(dtype=float), weights=weight_series.loc[mask].to_numpy(dtype=float)))


def _weighted_std(values: pd.Series, weights: pd.Series) -> float:
    mean = _weighted_mean(values, weights)
    if math.isnan(mean):
        return float("nan")
    series = pd.to_numeric(values, errors="coerce")
    weight_series = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    mask = series.notna() & weight_series.gt(0.0)
    total_weight = float(weight_series.loc[mask].sum())
    if total_weight <= 0.0:
        return float("nan")
    variance = np.average(
        (series.loc[mask].to_numpy(dtype=float) - mean) ** 2,
        weights=weight_series.loc[mask].to_numpy(dtype=float),
    )
    return float(np.sqrt(variance))


def _summarize_series(values: pd.Series, hour_weights: pd.Series, load_weights: pd.Series | None = None) -> dict[str, float]:
    series = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(hour_weights.reindex(series.index), errors="coerce").fillna(0.0)
    positive_weights = weights.clip(lower=0.0)
    total_hours = float(positive_weights.sum())
    load_weights = (
        pd.to_numeric(load_weights.reindex(series.index), errors="coerce").fillna(0.0)
        if load_weights is not None
        else None
    )
    return {
        "mean_eur_per_mwh": _weighted_mean(series, positive_weights),
        "load_weighted_mean_eur_per_mwh": _weighted_mean(series, load_weights)
        if load_weights is not None and float(load_weights.sum()) > 0.0
        else float("nan"),
        "min_eur_per_mwh": float(series.min()) if series.notna().any() else float("nan"),
        "p05_eur_per_mwh": _weighted_quantile(series, positive_weights, 0.05),
        "p25_eur_per_mwh": _weighted_quantile(series, positive_weights, 0.25),
        "p50_eur_per_mwh": _weighted_quantile(series, positive_weights, 0.50),
        "p75_eur_per_mwh": _weighted_quantile(series, positive_weights, 0.75),
        "p95_eur_per_mwh": _weighted_quantile(series, positive_weights, 0.95),
        "max_eur_per_mwh": float(series.max()) if series.notna().any() else float("nan"),
        "std_eur_per_mwh": _weighted_std(series, positive_weights),
        "negative_hour_share": (
            float(positive_weights.loc[series.lt(0.0)].sum()) / total_hours if total_hours > 0.0 else float("nan")
        ),
        "positive_hour_share": (
            float(positive_weights.loc[series.gt(0.0)].sum()) / total_hours if total_hours > 0.0 else float("nan")
        ),
    }


def _primary_bus(df: pd.DataFrame, idx: str) -> str:
    if "bus" in df.columns and pd.notna(df.at[idx, "bus"]):
        return str(df.at[idx, "bus"])
    for column in ("bus0", "bus1", "bus2", "bus3", "bus4"):
        if column in df.columns and pd.notna(df.at[idx, column]) and str(df.at[idx, column]) != "":
            return str(df.at[idx, column])
    return ""


def _primary_country(df: pd.DataFrame, idx: str, bus_country: pd.Series) -> str:
    bus = _primary_bus(df, idx)
    return str(bus_country.get(bus, "")) if bus else ""


def _sector_from_emission_carrier(carrier: str) -> str:
    carrier = str(carrier)
    if carrier in {"coal", "lignite", "oil", "OCGT", "CCGT", "urban central gas CHP", "urban central gas CHP CC"}:
        return "Power"
    if "industry" in carrier or carrier == "process emissions":
        return "Industry"
    if "transport" in carrier:
        return "Transport"
    if "boiler" in carrier or "residential" in carrier or "services" in carrier or "urban central" in carrier:
        return "Heating"
    return "Other"


def _is_renewable_carrier(carrier: str) -> bool:
    carrier_lower = str(carrier).lower()
    if carrier_lower in RENEWABLE_HINTS:
        return True
    return any(token in carrier_lower for token in ("solar", "wind", "hydro", "geo", "biomass", "ror"))


def _load_bus_profiles(n: pypsa.Network) -> pd.DataFrame:
    if n.loads.empty:
        return pd.DataFrame(index=n.snapshots)
    if "p_set" in n.loads_t and not n.loads_t.p_set.empty:
        profiles = n.loads_t.p_set.copy()
    elif "p" in n.loads_t and not n.loads_t.p.empty:
        profiles = n.loads_t.p.copy()
    else:
        return pd.DataFrame(index=n.snapshots)
    load_meta = n.loads.copy()
    ac_mask = load_meta["bus"].map(n.buses.carrier).eq("AC")
    if not ac_mask.any():
        return pd.DataFrame(index=n.snapshots)
    load_meta = load_meta.loc[ac_mask]
    profiles = profiles.reindex(columns=load_meta.index, fill_value=0.0)
    return profiles.groupby(load_meta["bus"], axis=1).sum()


def _collect_generation_records(n: pypsa.Network) -> pd.DataFrame:
    weights = _snapshot_weights(n, "generators")
    bus_country = _bus_country_lookup(n)
    frames = []

    if not n.generators.empty and "p" in n.generators_t and not n.generators_t.p.empty:
        gen = n.generators.loc[n.generators["bus"].map(n.buses.carrier).eq("AC")].copy()
        if not gen.empty:
            series = (
                n.generators_t.p.reindex(columns=gen.index, fill_value=0.0)
                .clip(lower=0.0)
                .mul(weights, axis=0)
                .sum(axis=0)
            )
            frame = pd.DataFrame(
                {
                    "component": "Generator",
                    "asset": series.index.astype(str),
                    "bus": gen["bus"].astype(str).values,
                    "country": gen["bus"].map(bus_country).fillna("").astype(str).values,
                    "carrier": gen["carrier"].astype(str).values,
                    "annual_generation_mwh": series.to_numpy(dtype=float),
                }
            )
            frames.append(frame)

    if not n.storage_units.empty and "p" in n.storage_units_t and not n.storage_units_t.p.empty:
        su = n.storage_units.loc[n.storage_units["bus"].map(n.buses.carrier).eq("AC")].copy()
        if not su.empty:
            series = (
                n.storage_units_t.p.reindex(columns=su.index, fill_value=0.0)
                .clip(lower=0.0)
                .mul(weights, axis=0)
                .sum(axis=0)
            )
            frame = pd.DataFrame(
                {
                    "component": "StorageUnit",
                    "asset": series.index.astype(str),
                    "bus": su["bus"].astype(str).values,
                    "country": su["bus"].map(bus_country).fillna("").astype(str).values,
                    "carrier": su["carrier"].astype(str).values,
                    "annual_generation_mwh": series.to_numpy(dtype=float),
                }
            )
            frames.append(frame)

    if not n.links.empty:
        for port in (1, 2, 3, 4):
            bus_col = f"bus{port}"
            pnl_col = f"p{port}"
            if bus_col not in n.links.columns or pnl_col not in n.links_t or n.links_t[pnl_col].empty:
                continue
            mask = n.links[bus_col].map(n.buses.carrier).eq("AC")
            if not mask.any():
                continue
            link_df = n.links.loc[mask].copy()
            series = (
                -n.links_t[pnl_col].reindex(columns=link_df.index, fill_value=0.0)
            ).clip(lower=0.0).mul(weights, axis=0).sum(axis=0)
            frame = pd.DataFrame(
                {
                    "component": "Link",
                    "asset": [f"{idx}:{bus_col}" for idx in series.index.astype(str)],
                    "bus": link_df[bus_col].astype(str).values,
                    "country": link_df[bus_col].map(bus_country).fillna("").astype(str).values,
                    "carrier": link_df["carrier"].astype(str).values,
                    "annual_generation_mwh": series.to_numpy(dtype=float),
                }
            )
            frames.append(frame)

    if not frames:
        return pd.DataFrame(
            columns=["component", "asset", "bus", "country", "carrier", "annual_generation_mwh"]
        )

    result = pd.concat(frames, ignore_index=True)
    result = result.loc[
        result["country"].ne("") & result["annual_generation_mwh"].gt(0.0)
    ].copy()
    return (
        result.groupby(["component", "asset", "bus", "country", "carrier"], as_index=False)["annual_generation_mwh"]
        .sum()
        .sort_values(["country", "carrier", "component", "asset"], ignore_index=True)
    )


def _ac_energy_balance_country_carrier(n: pypsa.Network) -> pd.DataFrame:
    try:
        balance = n.statistics.energy_balance(
            bus_carrier="AC",
            aggregate_time="sum",
            aggregate_groups="sum",
            groupby=n.statistics.groupers.get_country_and_carrier,
            nice_names=True,
        )
    except Exception:
        return _empty_frame("ac_energy_balance_country_carrier.csv")

    if not isinstance(balance, pd.Series):
        balance = balance.rename("energy_balance_mwh")
    frame = balance.rename("energy_balance_mwh").reset_index()
    required_columns = {"component", "country", "carrier", "energy_balance_mwh"}
    if not required_columns.issubset(frame.columns):
        return _empty_frame("ac_energy_balance_country_carrier.csv")

    frame = frame.loc[
        frame["component"].isin(["Generator", "Link", "StorageUnit", "Load"])
        & frame["country"].fillna("").astype(str).ne("")
        & pd.to_numeric(frame["energy_balance_mwh"], errors="coerce").fillna(0.0).gt(0.0)
    ].copy()
    if frame.empty:
        return _empty_frame("ac_energy_balance_country_carrier.csv")

    frame["country"] = frame["country"].astype(str)
    frame["carrier"] = frame["carrier"].astype(str)
    frame["energy_balance_mwh"] = pd.to_numeric(frame["energy_balance_mwh"], errors="coerce").fillna(0.0)
    frame["energy_balance_twh"] = frame["energy_balance_mwh"] / 1e6
    return (
        frame.groupby(["country", "component", "carrier"], as_index=False)[["energy_balance_mwh", "energy_balance_twh"]]
        .sum()
        .sort_values(["country", "carrier", "component"], ignore_index=True)
    )


def _collect_capacity_records(n: pypsa.Network) -> pd.DataFrame:
    bus_country = _bus_country_lookup(n)
    frames = []

    if not n.generators.empty:
        column = _capacity_column(n.generators, "p_nom_opt", "p_nom")
        if column:
            gen = n.generators.loc[n.generators["bus"].map(n.buses.carrier).eq("AC")].copy()
            if not gen.empty:
                gen["component"] = "Generator"
                gen["asset"] = gen.index.astype(str)
                gen["bus"] = gen["bus"].astype(str)
                gen["country"] = gen["bus"].map(bus_country).fillna("").astype(str)
                gen["capacity_unit"] = "MW"
                gen["capacity_value"] = pd.to_numeric(gen[column], errors="coerce").fillna(0.0)
                frames.append(gen.loc[:, ["component", "asset", "bus", "country", "carrier", "capacity_unit", "capacity_value"]])

    if not n.storage_units.empty:
        column = _capacity_column(n.storage_units, "p_nom_opt", "p_nom")
        if column:
            su = n.storage_units.loc[n.storage_units["bus"].map(n.buses.carrier).eq("AC")].copy()
            if not su.empty:
                su["component"] = "StorageUnit"
                su["asset"] = su.index.astype(str)
                su["bus"] = su["bus"].astype(str)
                su["country"] = su["bus"].map(bus_country).fillna("").astype(str)
                su["capacity_unit"] = "MW"
                su["capacity_value"] = pd.to_numeric(su[column], errors="coerce").fillna(0.0)
                frames.append(su.loc[:, ["component", "asset", "bus", "country", "carrier", "capacity_unit", "capacity_value"]])

    if not n.stores.empty:
        column = _capacity_column(n.stores, "e_nom_opt", "e_nom")
        if column:
            stores = n.stores.loc[n.stores["bus"].map(n.buses.carrier).eq("AC")].copy()
            if not stores.empty:
                stores["component"] = "Store"
                stores["asset"] = stores.index.astype(str)
                stores["bus"] = stores["bus"].astype(str)
                stores["country"] = stores["bus"].map(bus_country).fillna("").astype(str)
                stores["capacity_unit"] = "MWh"
                stores["capacity_value"] = pd.to_numeric(stores[column], errors="coerce").fillna(0.0)
                frames.append(stores.loc[:, ["component", "asset", "bus", "country", "carrier", "capacity_unit", "capacity_value"]])

    if not n.links.empty:
        column = _capacity_column(n.links, "p_nom_opt", "p_nom")
        if column:
            for port in (1, 2, 3, 4):
                bus_col = f"bus{port}"
                eff_col = "efficiency" if port == 1 else f"efficiency{port}"
                if bus_col not in n.links.columns or eff_col not in n.links.columns:
                    continue
                mask = n.links[bus_col].map(n.buses.carrier).eq("AC")
                if not mask.any():
                    continue
                link_df = n.links.loc[mask].copy()
                if link_df.empty:
                    continue
                link_df["component"] = "Link"
                link_df["asset"] = [f"{idx}:{bus_col}" for idx in link_df.index.astype(str)]
                link_df["bus"] = link_df[bus_col].astype(str)
                link_df["country"] = link_df["bus"].map(bus_country).fillna("").astype(str)
                link_df["capacity_unit"] = "MW"
                link_df["capacity_value"] = (
                    pd.to_numeric(link_df[column], errors="coerce").fillna(0.0)
                    * pd.to_numeric(link_df[eff_col], errors="coerce").fillna(0.0).abs()
                )
                frames.append(link_df.loc[:, ["component", "asset", "bus", "country", "carrier", "capacity_unit", "capacity_value"]])

    if not frames:
        return pd.DataFrame(
            columns=["component", "asset", "bus", "country", "carrier", "capacity_unit", "capacity_value"]
        )

    result = pd.concat(frames, ignore_index=True)
    result = result.loc[
        result["country"].ne("")
        & result["carrier"].notna()
        & result["capacity_value"].gt(0.0)
    ].copy()
    return (
        result.groupby(
            ["component", "asset", "bus", "country", "carrier", "capacity_unit"],
            as_index=False,
        )["capacity_value"]
        .sum()
        .sort_values(["country", "carrier", "component", "asset"], ignore_index=True)
    )


def _capacity_country_summary(capacity_df: pd.DataFrame, year: int) -> pd.DataFrame:
    if capacity_df.empty:
        return _empty_frame("capacity_country_carrier.csv")
    result = (
        capacity_df.groupby(["component", "country", "carrier", "capacity_unit"], as_index=False)["capacity_value"]
        .sum()
        .sort_values(["country", "carrier", "component", "capacity_unit"], ignore_index=True)
    )
    result.insert(0, "year", int(year))
    return result


def _battery_operations_country_year(n: pypsa.Network) -> pd.DataFrame:
    if n.links.empty or n.stores.empty:
        return _empty_frame("battery_operations_country_year.csv")

    weights = _snapshot_weights(n, "objective")
    bus_country = _bus_country_lookup(n)
    rows = []

    battery_stores = n.stores.loc[n.stores["carrier"].astype(str).str.lower().eq("battery")].copy() if "carrier" in n.stores.columns else pd.DataFrame()
    if battery_stores.empty:
        return _empty_frame("battery_operations_country_year.csv")

    store_capacity_col = "e_nom_opt" if "e_nom_opt" in battery_stores.columns else ("e_nom" if "e_nom" in battery_stores.columns else None)
    store_summary = (
        battery_stores.assign(
            country=battery_stores["bus"].map(bus_country).fillna("").astype(str),
            store_energy_capacity_mwh=pd.to_numeric(battery_stores[store_capacity_col], errors="coerce").fillna(0.0) if store_capacity_col else 0.0,
        )
        .groupby("country", as_index=False)["store_energy_capacity_mwh"]
        .sum()
    )

    links = n.links.copy()
    if "carrier" not in links.columns:
        return _empty_frame("battery_operations_country_year.csv")

    capacity_col = _capacity_column(links, "p_nom_opt", "p_nom")

    charger_mask = links["carrier"].astype(str).str.lower().eq("battery charger")
    if charger_mask.any() and "p0" in n.links_t and "p1" in n.links_t:
        chargers = links.loc[charger_mask].copy()
        chargers["country"] = chargers["bus0"].map(bus_country).fillna("").astype(str)
        charger_ac_draw = (
            n.links_t.p0.reindex(columns=chargers.index, fill_value=0.0).clip(lower=0.0).mul(weights, axis=0).sum(axis=0)
        )
        charger_to_store = (
            (-n.links_t.p1.reindex(columns=chargers.index, fill_value=0.0)).clip(lower=0.0).mul(weights, axis=0).sum(axis=0)
        )
        charger_power = pd.to_numeric(chargers[capacity_col], errors="coerce").fillna(0.0) if capacity_col else pd.Series(0.0, index=chargers.index)
        charger_rows = pd.DataFrame(
            {
                "country": chargers["country"].values,
                "charge_input_from_ac_mwh": charger_ac_draw.to_numpy(dtype=float),
                "charge_to_store_mwh": charger_to_store.to_numpy(dtype=float),
                "charger_power_capacity_mw": charger_power.to_numpy(dtype=float),
            }
        )
    else:
        charger_rows = pd.DataFrame(columns=["country", "charge_input_from_ac_mwh", "charge_to_store_mwh", "charger_power_capacity_mw"])

    discharger_mask = links["carrier"].astype(str).str.lower().eq("battery discharger")
    if discharger_mask.any() and "p0" in n.links_t and "p1" in n.links_t:
        dischargers = links.loc[discharger_mask].copy()
        dischargers["country"] = dischargers["bus1"].map(bus_country).fillna("").astype(str)
        discharge_from_store = (
            n.links_t.p0.reindex(columns=dischargers.index, fill_value=0.0).clip(lower=0.0).mul(weights, axis=0).sum(axis=0)
        )
        discharge_to_ac = (
            (-n.links_t.p1.reindex(columns=dischargers.index, fill_value=0.0)).clip(lower=0.0).mul(weights, axis=0).sum(axis=0)
        )
        discharger_power = pd.to_numeric(dischargers[capacity_col], errors="coerce").fillna(0.0) if capacity_col else pd.Series(0.0, index=dischargers.index)
        discharger_rows = pd.DataFrame(
            {
                "country": dischargers["country"].values,
                "discharge_from_store_mwh": discharge_from_store.to_numpy(dtype=float),
                "discharge_to_ac_mwh": discharge_to_ac.to_numpy(dtype=float),
                "discharger_power_capacity_mw": discharger_power.to_numpy(dtype=float),
            }
        )
    else:
        discharger_rows = pd.DataFrame(columns=["country", "discharge_from_store_mwh", "discharge_to_ac_mwh", "discharger_power_capacity_mw"])

    charger_country = charger_rows.groupby("country", as_index=False).sum(numeric_only=True) if not charger_rows.empty else pd.DataFrame(columns=["country"])
    discharger_country = discharger_rows.groupby("country", as_index=False).sum(numeric_only=True) if not discharger_rows.empty else pd.DataFrame(columns=["country"])

    result = store_summary.merge(charger_country, on="country", how="outer").merge(discharger_country, on="country", how="outer").fillna(0.0)
    if result.empty:
        return _empty_frame("battery_operations_country_year.csv")

    result["carrier"] = "battery"
    result["net_ac_supply_mwh"] = result["discharge_to_ac_mwh"] - result["charge_input_from_ac_mwh"]
    result["throughput_mwh"] = result["charge_to_store_mwh"] + result["discharge_from_store_mwh"]
    result["charging_losses_mwh"] = (result["charge_input_from_ac_mwh"] - result["charge_to_store_mwh"]).clip(lower=0.0)
    result["discharging_losses_mwh"] = (result["discharge_from_store_mwh"] - result["discharge_to_ac_mwh"]).clip(lower=0.0)
    result["total_losses_mwh"] = result["charging_losses_mwh"] + result["discharging_losses_mwh"]
    result["estimated_cycles"] = np.where(
        result["store_energy_capacity_mwh"] > 0.0,
        result["discharge_to_ac_mwh"] / result["store_energy_capacity_mwh"],
        np.nan,
    )

    result = result.loc[result["country"].astype(str).ne("")].copy()
    result = result.loc[:, OUTPUT_TABLE_SPECS["battery_operations_country_year.csv"][1:]]
    return result.sort_values(["country"], ignore_index=True)


def _deployment_summaries(current_capacity: pd.DataFrame, previous_capacity: pd.DataFrame, year: int, previous_year: int | None):
    if current_capacity.empty and previous_capacity.empty:
        return _empty_frame("deployment_node_carrier.csv"), _empty_frame("deployment_country_carrier.csv")

    merged = current_capacity.merge(
        previous_capacity,
        on=["component", "asset", "bus", "country", "carrier", "capacity_unit"],
        how="outer",
        suffixes=("_current", "_previous"),
    ).fillna({"capacity_value_current": 0.0, "capacity_value_previous": 0.0})
    merged["build_value"] = (merged["capacity_value_current"] - merged["capacity_value_previous"]).clip(lower=0.0)
    merged["retire_value"] = (merged["capacity_value_previous"] - merged["capacity_value_current"]).clip(lower=0.0)
    merged["net_change_value"] = merged["capacity_value_current"] - merged["capacity_value_previous"]

    node = merged.groupby(
        ["component", "bus", "country", "carrier", "capacity_unit"], as_index=False
    )[["build_value", "retire_value", "net_change_value"]].sum()
    node.insert(0, "previous_year", previous_year)
    node.insert(0, "year", int(year))

    country = merged.groupby(
        ["component", "country", "carrier", "capacity_unit"], as_index=False
    )[["build_value", "retire_value", "net_change_value"]].sum()
    country.insert(0, "previous_year", previous_year)
    country.insert(0, "year", int(year))
    return node, country


def _power_emissions_country_carrier(n: pypsa.Network) -> pd.DataFrame:
    weights = _snapshot_weights(n, "objective")
    bus_country = _bus_country_lookup(n)
    if n.links.empty or "p0" not in n.links_t or n.links_t.p0.empty:
        return _empty_frame("power_emissions_country_carrier.csv")

    rows = []
    for idx, row in n.links.iterrows():
        ac_bus = ""
        co2_eff = None
        for port in (1, 2, 3, 4):
            bus_col = f"bus{port}"
            eff_col = "efficiency" if port == 1 else f"efficiency{port}"
            if bus_col not in row.index or eff_col not in row.index:
                continue
            bus_name = row[bus_col]
            if isinstance(bus_name, str) and bus_name in n.buses.index and n.buses.at[bus_name, "carrier"] == "AC" and not ac_bus:
                ac_bus = bus_name
            if bus_name == "co2 atmosphere" and co2_eff is None:
                co2_eff = abs(float(row[eff_col])) if pd.notna(row[eff_col]) else None
        if not ac_bus or co2_eff is None or idx not in n.links_t.p0.columns:
            continue
        emissions_tco2 = float((n.links_t.p0[idx].abs() * weights).sum() * co2_eff)
        if emissions_tco2 <= 0.0:
            continue
        rows.append(
            {
                "country": str(bus_country.get(ac_bus, "")),
                "carrier": FOSSIL_CARRIER_MAP.get(str(row["carrier"]), str(row["carrier"])),
                "emissions_tco2": emissions_tco2,
            }
        )

    if not rows:
        return _empty_frame("power_emissions_country_carrier.csv")

    result = (
        pd.DataFrame(rows)
        .groupby(["country", "carrier"], as_index=False)["emissions_tco2"]
        .sum()
        .sort_values(["country", "carrier"], ignore_index=True)
    )
    result["emissions_mtco2"] = result["emissions_tco2"] / 1e6
    result["emissions_gtco2"] = result["emissions_tco2"] / 1e9
    return result


def _sector_emissions(n: pypsa.Network) -> pd.DataFrame:
    try:
        eb = n.statistics.energy_balance(aggregate_time="sum", aggregate_groups="sum", nice_names=False)
    except Exception:
        return _empty_frame("sector_emissions.csv")
    emissions = eb.rename("emissions_mwh").reset_index()
    emissions = emissions.loc[
        emissions["bus_carrier"].eq("co2") & emissions["emissions_mwh"].gt(0.0)
    ].copy()
    if emissions.empty:
        return _empty_frame("sector_emissions.csv")
    emissions["sector"] = emissions["carrier"].map(_sector_from_emission_carrier)
    result = emissions.groupby("sector", as_index=False)["emissions_mwh"].sum()
    result["emissions_mtco2"] = result["emissions_mwh"] / 1e6
    result = result.loc[:, ["sector", "emissions_mtco2"]]
    total = pd.DataFrame([{"sector": "TOTAL", "emissions_mtco2": float(result["emissions_mtco2"].sum())}])
    return pd.concat([result, total], ignore_index=True)


def _sector_demands_country(n: pypsa.Network) -> pd.DataFrame:
    if n.loads.empty:
        return _empty_frame("sector_demands_country.csv")
    profiles = _load_bus_profiles(n)
    if profiles.empty:
        return _empty_frame("sector_demands_country.csv")
    weights = _snapshot_weights(n, "objective")
    bus_country = _bus_country_lookup(n)
    load_meta = n.loads.copy()
    load_meta["country"] = load_meta["bus"].map(bus_country).fillna("").astype(str)
    if "carrier" not in load_meta.columns:
        load_meta["carrier"] = "electricity"
    data = (
        n.loads_t.p_set.reindex(columns=load_meta.index, fill_value=0.0)
        if "p_set" in n.loads_t and not n.loads_t.p_set.empty
        else n.loads_t.p.reindex(columns=load_meta.index, fill_value=0.0)
    )
    annual = data.mul(weights, axis=0).sum(axis=0)
    result = pd.DataFrame(
        {
            "country": load_meta["country"].values,
            "sector": load_meta["carrier"].astype(str).values,
            "annual_demand_mwh": annual.to_numpy(dtype=float),
        }
    )
    result = (
        result.groupby(["country", "sector"], as_index=False)["annual_demand_mwh"]
        .sum()
        .sort_values(["country", "sector"], ignore_index=True)
    )
    result["annual_demand_twh"] = result["annual_demand_mwh"] / 1e6
    return result


def _price_node_year(n: pypsa.Network) -> pd.DataFrame:
    if n.buses.empty or "marginal_price" not in n.buses_t or n.buses_t.marginal_price.empty:
        return _empty_frame("electricity_price_node_year.csv")
    ac_buses = n.buses.index[n.buses["carrier"].eq("AC")]
    if len(ac_buses) == 0:
        return _empty_frame("electricity_price_node_year.csv")
    prices = n.buses_t.marginal_price.reindex(columns=ac_buses, fill_value=np.nan)
    loads_by_bus = _load_bus_profiles(n).reindex(index=prices.index, columns=ac_buses, fill_value=0.0)
    weights = _snapshot_weights(n, "objective")
    bus_country = _bus_country_lookup(n)
    rows = []
    for bus in ac_buses:
        summary = _summarize_series(prices[bus], weights, loads_by_bus[bus])
        rows.append({"bus": str(bus), "country": str(bus_country.get(bus, "")), **summary})
    return pd.DataFrame(rows)


def _price_country_year(n: pypsa.Network) -> pd.DataFrame:
    if n.buses.empty or "marginal_price" not in n.buses_t or n.buses_t.marginal_price.empty:
        return _empty_frame("electricity_price_country_year.csv")
    ac_buses = n.buses.index[n.buses["carrier"].eq("AC")]
    if len(ac_buses) == 0:
        return _empty_frame("electricity_price_country_year.csv")
    prices = n.buses_t.marginal_price.reindex(columns=ac_buses, fill_value=np.nan)
    loads_by_bus = _load_bus_profiles(n).reindex(index=prices.index, columns=ac_buses, fill_value=0.0)
    weights = _snapshot_weights(n, "objective")
    bus_country = _bus_country_lookup(n)
    rows = []
    for country, buses in bus_country.loc[ac_buses].groupby(bus_country.loc[ac_buses]):
        buses = list(buses.index)
        if not country or not buses:
            continue
        country_loads = loads_by_bus[buses]
        country_prices = prices[buses]
        load_sum = country_loads.sum(axis=1)
        weighted_hourly = (
            (country_prices * country_loads).sum(axis=1).div(load_sum.replace(0.0, np.nan))
        )
        flat_prices = country_prices.stack(dropna=True)
        flat_loads = country_loads.stack(dropna=True).reindex(flat_prices.index).fillna(0.0)
        flat_hour_weights = (
            pd.Series(weights, index=country_prices.index).reindex(flat_prices.index.get_level_values(0)).to_numpy(dtype=float)
        )
        load_weight_series = pd.Series(flat_loads.to_numpy(dtype=float) * flat_hour_weights, index=flat_prices.index)
        summary = _summarize_series(weighted_hourly, weights, weighted_hourly * 0.0 + load_sum)
        summary["load_weighted_mean_eur_per_mwh"] = _weighted_mean(flat_prices, load_weight_series)
        rows.append({"country": str(country), **summary})
    return pd.DataFrame(rows)


def _system_constraint_shadows(n: pypsa.Network) -> pd.DataFrame:
    if n.global_constraints.empty:
        return _empty_frame("constraint_shadow_system_year.csv")
    result = n.global_constraints.copy().reset_index().rename(columns={"index": "constraint_name"})
    for column in ("type", "sense", "carrier_attribute", "constant", "mu"):
        if column not in result.columns:
            result[column] = np.nan
    return result.loc[:, ["constraint_name", "sense", "carrier_attribute", "constant", "mu"]]


def _summarize_shadow_values(values: pd.Series, weights: pd.Series | None = None) -> dict[str, float]:
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        return {
            "mean_mu": float("nan"),
            "abs_mean_mu": float("nan"),
            "min_mu": float("nan"),
            "max_mu": float("nan"),
            "p95_abs_mu": float("nan"),
            "binding_hours_count": 0.0,
            "positive_hours_count": 0.0,
            "negative_hours_count": 0.0,
        }
    if weights is None:
        weights = pd.Series(1.0, index=series.index)
    else:
        weights = pd.to_numeric(weights.reindex(series.index), errors="coerce").fillna(0.0)
    abs_series = series.abs()
    return {
        "mean_mu": _weighted_mean(series, weights),
        "abs_mean_mu": _weighted_mean(abs_series, weights),
        "min_mu": float(series.min()),
        "max_mu": float(series.max()),
        "p95_abs_mu": _weighted_quantile(abs_series, weights, 0.95),
        "binding_hours_count": float(weights.loc[abs_series.gt(1e-9)].sum()),
        "positive_hours_count": float(weights.loc[series.gt(1e-9)].sum()),
        "negative_hours_count": float(weights.loc[series.lt(-1e-9)].sum()),
    }


def _asset_constraint_shadows(n: pypsa.Network) -> pd.DataFrame:
    rows = []
    weights = _snapshot_weights(n, "objective")
    bus_country = _bus_country_lookup(n)
    for component_name, list_name in SHADOW_COMPONENT_MAP.items():
        df = getattr(n, list_name, pd.DataFrame())
        if df is None or df.empty:
            continue
        for column in [col for col in df.columns if str(col).startswith("mu_")]:
            values = pd.to_numeric(df[column], errors="coerce").dropna()
            if values.empty:
                continue
            for asset, value in values.items():
                summary = _summarize_shadow_values(pd.Series([value], index=[0]), pd.Series([1.0], index=[0]))
                rows.append(
                    {
                        "component": component_name,
                        "asset": str(asset),
                        "bus": _primary_bus(df, asset),
                        "country": _primary_country(df, asset, bus_country),
                        "carrier": str(df.at[asset, "carrier"]) if "carrier" in df.columns and pd.notna(df.at[asset, "carrier"]) else "",
                        "constraint_name": str(column),
                        **summary,
                    }
                )
        pnl = getattr(n, f"{list_name}_t", None)
        if pnl is None or not hasattr(pnl, "keys"):
            continue
        for column in [key for key in pnl.keys() if str(key).startswith("mu_")]:
            frame = pnl[column]
            if frame is None or frame.empty:
                continue
            for asset in frame.columns:
                summary = _summarize_shadow_values(frame[asset], weights)
                rows.append(
                    {
                        "component": component_name,
                        "asset": str(asset),
                        "bus": _primary_bus(df, asset) if asset in df.index else "",
                        "country": _primary_country(df, asset, bus_country) if asset in df.index else "",
                        "carrier": str(df.at[asset, "carrier"]) if asset in df.index and "carrier" in df.columns and pd.notna(df.at[asset, "carrier"]) else "",
                        "constraint_name": str(column),
                        **summary,
                    }
                )
    if not rows:
        return _empty_frame("constraint_shadow_asset_year.csv")
    return pd.DataFrame(rows)


def _lcoe_electricity_tech_year(n: pypsa.Network) -> pd.DataFrame:
    weights = _snapshot_weights(n, "objective")
    rows = []

    if not n.generators.empty and "p" in n.generators_t and not n.generators_t.p.empty:
        capacity_col = _capacity_column(n.generators, "p_nom_opt", "p_nom")
        dispatch = n.generators_t.p.clip(lower=0.0).mul(weights, axis=0).sum(axis=0)
        capital = pd.to_numeric(n.generators.get("capital_cost", pd.Series(0.0, index=n.generators.index)), errors="coerce").fillna(0.0)
        capacity = pd.to_numeric(n.generators.get(capacity_col, pd.Series(0.0, index=n.generators.index)), errors="coerce").fillna(0.0)
        marginal = pd.to_numeric(n.generators.get("marginal_cost", pd.Series(0.0, index=n.generators.index)), errors="coerce").fillna(0.0)
        operating = dispatch.mul(marginal, fill_value=0.0)
        for carrier, carrier_assets in n.generators.groupby("carrier").groups.items():
            output = float(dispatch.loc[list(carrier_assets)].sum())
            if output <= 0.0:
                continue
            capex = float((capital.loc[list(carrier_assets)] * capacity.loc[list(carrier_assets)]).sum())
            opex = float(operating.loc[list(carrier_assets)].sum())
            rows.append(
                {
                    "component": "Generator",
                    "carrier": str(carrier),
                    "electricity_output_mwh": output,
                    "annualized_capital_cost_eur": capex,
                    "operating_cost_eur": opex,
                }
            )

    if not n.storage_units.empty and "p" in n.storage_units_t and not n.storage_units_t.p.empty:
        capacity_col = _capacity_column(n.storage_units, "p_nom_opt", "p_nom")
        dispatch = n.storage_units_t.p.clip(lower=0.0).mul(weights, axis=0).sum(axis=0)
        capital = pd.to_numeric(n.storage_units.get("capital_cost", pd.Series(0.0, index=n.storage_units.index)), errors="coerce").fillna(0.0)
        capacity = pd.to_numeric(n.storage_units.get(capacity_col, pd.Series(0.0, index=n.storage_units.index)), errors="coerce").fillna(0.0)
        marginal = pd.to_numeric(n.storage_units.get("marginal_cost", pd.Series(0.0, index=n.storage_units.index)), errors="coerce").fillna(0.0)
        operating = dispatch.mul(marginal, fill_value=0.0)
        for carrier, carrier_assets in n.storage_units.groupby("carrier").groups.items():
            output = float(dispatch.loc[list(carrier_assets)].sum())
            if output <= 0.0:
                continue
            capex = float((capital.loc[list(carrier_assets)] * capacity.loc[list(carrier_assets)]).sum())
            opex = float(operating.loc[list(carrier_assets)].sum())
            rows.append(
                {
                    "component": "StorageUnit",
                    "carrier": str(carrier),
                    "electricity_output_mwh": output,
                    "annualized_capital_cost_eur": capex,
                    "operating_cost_eur": opex,
                }
            )

    if not n.links.empty:
        capacity_col = _capacity_column(n.links, "p_nom_opt", "p_nom")
        capital = pd.to_numeric(n.links.get("capital_cost", pd.Series(0.0, index=n.links.index)), errors="coerce").fillna(0.0)
        capacity = pd.to_numeric(n.links.get(capacity_col, pd.Series(0.0, index=n.links.index)), errors="coerce").fillna(0.0)
        marginal = pd.to_numeric(n.links.get("marginal_cost", pd.Series(0.0, index=n.links.index)), errors="coerce").fillna(0.0)
        operating = n.links_t.p0.abs().mul(weights, axis=0).sum(axis=0).mul(marginal, fill_value=0.0) if "p0" in n.links_t else pd.Series(0.0, index=n.links.index)
        link_rows = []
        for port in (1, 2, 3, 4):
            bus_col = f"bus{port}"
            pnl_col = f"p{port}"
            eff_col = "efficiency" if port == 1 else f"efficiency{port}"
            if bus_col not in n.links.columns or pnl_col not in n.links_t or eff_col not in n.links.columns:
                continue
            mask = n.links[bus_col].map(n.buses.carrier).eq("AC")
            if not mask.any():
                continue
            dispatch = (-n.links_t[pnl_col].reindex(columns=n.links.index[mask], fill_value=0.0)).clip(lower=0.0).mul(weights, axis=0).sum(axis=0)
            eff = pd.to_numeric(n.links.loc[mask, eff_col], errors="coerce").fillna(0.0).abs()
            for carrier, carrier_assets in n.links.loc[mask].groupby("carrier").groups.items():
                asset_index = list(carrier_assets)
                output = float(dispatch.loc[asset_index].sum())
                if output <= 0.0:
                    continue
                capex = float((capital.loc[asset_index] * capacity.loc[asset_index] * eff.loc[asset_index]).sum())
                opex = float(operating.loc[asset_index].sum())
                link_rows.append(
                    {
                        "component": "Link",
                        "carrier": str(carrier),
                        "electricity_output_mwh": output,
                        "annualized_capital_cost_eur": capex,
                        "operating_cost_eur": opex,
                    }
                )
        rows.extend(link_rows)

    if not rows:
        return _empty_frame("lcoe_electricity_tech_year.csv")
    result = pd.DataFrame(rows)
    result = (
        result.groupby(["component", "carrier"], as_index=False)[
            ["electricity_output_mwh", "annualized_capital_cost_eur", "operating_cost_eur"]
        ]
        .sum()
        .sort_values(["component", "carrier"], ignore_index=True)
    )
    result["total_cost_eur"] = result["annualized_capital_cost_eur"] + result["operating_cost_eur"]
    result["lcoe_eur_per_mwh"] = result["total_cost_eur"] / result["electricity_output_mwh"]
    return result.loc[result["electricity_output_mwh"].gt(0.0)].copy()


def _curtailment_total_mwh(n: pypsa.Network) -> float:
    if n.generators.empty or "p_max_pu" not in n.generators_t or "p" not in n.generators_t:
        return 0.0
    weights = _snapshot_weights(n, "generators")
    capacity_col = _capacity_column(n.generators, "p_nom_opt", "p_nom")
    if capacity_col is None:
        return 0.0
    available = (
        n.generators_t.p_max_pu.mul(pd.to_numeric(n.generators[capacity_col], errors="coerce").fillna(0.0), axis=1)
        .mul(weights, axis=0)
        .sum()
    )
    used = n.generators_t.p.clip(lower=0.0).mul(weights, axis=0).sum()
    return float((available - used).clip(lower=0.0).sum())


def _system_summary(n: pypsa.Network, generation_df: pd.DataFrame, power_emissions_df: pd.DataFrame, sector_emissions_df: pd.DataFrame, capacity_df: pd.DataFrame) -> pd.DataFrame:
    load_profiles = _load_bus_profiles(n)
    weights = _snapshot_weights(n, "objective")
    total_electricity_demand_mwh = float(load_profiles.mul(weights, axis=0).sum().sum()) if not load_profiles.empty else 0.0
    total_generation = float(generation_df["annual_generation_mwh"].sum()) if not generation_df.empty else 0.0
    renewable_generation = float(
        generation_df.loc[generation_df["carrier"].map(_is_renewable_carrier), "annual_generation_mwh"].sum()
    ) if not generation_df.empty else 0.0
    storage_power_capacity = float(
        capacity_df.loc[
            capacity_df["component"].isin(["StorageUnit", "Link"]) & capacity_df["capacity_unit"].eq("MW"),
            "capacity_value",
        ].sum()
    ) if not capacity_df.empty else 0.0
    storage_energy_capacity = float(
        capacity_df.loc[
            capacity_df["component"].eq("Store") & capacity_df["capacity_unit"].eq("MWh"),
            "capacity_value",
        ].sum()
    ) if not capacity_df.empty else 0.0
    transmission_ac_capacity = float(pd.to_numeric(n.lines.get("s_nom_opt", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()) if not n.lines.empty else 0.0
    transmission_dc_capacity = float(pd.to_numeric(n.links.loc[n.links.get("carrier", pd.Series(dtype=str)).eq("DC"), "p_nom_opt"], errors="coerce").fillna(0.0).sum()) if not n.links.empty and "carrier" in n.links.columns and "p_nom_opt" in n.links.columns else 0.0
    transmission_ac_volume = float((pd.to_numeric(n.lines.get("length", pd.Series(dtype=float)), errors="coerce").fillna(0.0) * pd.to_numeric(n.lines.get("s_nom_opt", pd.Series(dtype=float)), errors="coerce").fillna(0.0)).sum()) if not n.lines.empty else 0.0
    transmission_dc_volume = float((pd.to_numeric(n.links.loc[n.links.get("carrier", pd.Series(dtype=str)).eq("DC"), "length"], errors="coerce").fillna(0.0) * pd.to_numeric(n.links.loc[n.links.get("carrier", pd.Series(dtype=str)).eq("DC"), "p_nom_opt"], errors="coerce").fillna(0.0)).sum()) if not n.links.empty and "carrier" in n.links.columns and "length" in n.links.columns and "p_nom_opt" in n.links.columns else 0.0
    total_emissions_mtco2 = float(
        sector_emissions_df.loc[sector_emissions_df["sector"].eq("TOTAL"), "emissions_mtco2"].sum()
    ) if not sector_emissions_df.empty else 0.0
    return pd.DataFrame(
        [
            {
                "objective_eur": float(n.objective) if getattr(n, "objective", None) is not None else float("nan"),
                "total_electricity_demand_mwh": total_electricity_demand_mwh,
                "total_electricity_generation_mwh": total_generation,
                "renewable_electricity_generation_mwh": renewable_generation,
                "renewable_share": renewable_generation / total_generation if total_generation > 0.0 else float("nan"),
                "curtailment_mwh": _curtailment_total_mwh(n),
                "storage_power_capacity_mw": storage_power_capacity,
                "storage_energy_capacity_mwh": storage_energy_capacity,
                "transmission_ac_capacity_mva": transmission_ac_capacity,
                "transmission_dc_capacity_mw": transmission_dc_capacity,
                "transmission_ac_volume_mwkm": transmission_ac_volume,
                "transmission_dc_volume_mwkm": transmission_dc_volume,
                "total_emissions_mtco2": total_emissions_mtco2,
            }
        ]
    )


def _prepare_learning_costs(cost_log_paths: dict[int, Path]) -> pd.DataFrame:
    frames = []
    for year, path in sorted(cost_log_paths.items()):
        frame = pd.read_csv(path)
        frame.insert(0, "year", int(year))
        for column in OUTPUT_TABLE_SPECS["learning_costs.csv"]:
            if column not in frame.columns:
                frame[column] = np.nan
        frames.append(frame.loc[:, OUTPUT_TABLE_SPECS["learning_costs.csv"]])
    if not frames:
        return _empty_frame("learning_costs.csv")
    return pd.concat(frames, ignore_index=True)


def _write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _extract_bundle(bundle_dir: Path, years: list[int], network_paths: dict[int, Path], cost_log_paths: dict[int, Path], state_paths: dict[int, Path], metadata: dict):
    tables = {name: _empty_frame(name) for name in OUTPUT_TABLE_SPECS}
    learning_costs = _prepare_learning_costs(cost_log_paths)
    previous_capacity = pd.DataFrame(
        columns=["component", "asset", "bus", "country", "carrier", "capacity_unit", "capacity_value"]
    )
    previous_year = None
    generation_frames = []
    ac_balance_frames = []
    capacity_frames = []
    battery_operation_frames = []
    deployment_country_frames = []
    deployment_node_frames = []
    power_emissions_frames = []
    sector_emissions_frames = []
    sector_demands_frames = []
    node_price_frames = []
    country_price_frames = []
    system_shadow_frames = []
    asset_shadow_frames = []
    system_summary_frames = []
    lcoe_frames = []

    for year in years:
        network = pypsa.Network(str(network_paths[year]))
        generation = _collect_generation_records(network)
        if "year" in generation.columns:
            generation["year"] = int(year)
        else:
            generation.insert(0, "year", int(year))
        generation["annual_generation_twh"] = generation["annual_generation_mwh"] / 1e6
        generation_frames.append(
            generation.loc[:, OUTPUT_TABLE_SPECS["generation_country_carrier.csv"]]
            if not generation.empty
            else _empty_frame("generation_country_carrier.csv")
        )

        ac_balance = _ac_energy_balance_country_carrier(network)
        if "year" in ac_balance.columns:
            ac_balance["year"] = int(year)
        else:
            ac_balance.insert(0, "year", int(year))
        ac_balance_frames.append(
            ac_balance.loc[:, OUTPUT_TABLE_SPECS["ac_energy_balance_country_carrier.csv"]]
            if not ac_balance.empty
            else _empty_frame("ac_energy_balance_country_carrier.csv")
        )

        capacity = _collect_capacity_records(network)
        capacity_country = _capacity_country_summary(capacity, year)
        capacity_frames.append(capacity_country)

        battery_operations = _battery_operations_country_year(network)
        if "year" in battery_operations.columns:
            battery_operations["year"] = int(year)
        else:
            battery_operations.insert(0, "year", int(year))
        battery_operation_frames.append(
            battery_operations.loc[:, OUTPUT_TABLE_SPECS["battery_operations_country_year.csv"]]
            if not battery_operations.empty
            else _empty_frame("battery_operations_country_year.csv")
        )

        deployment_node, deployment_country = _deployment_summaries(capacity, previous_capacity, year, previous_year)
        deployment_node_frames.append(deployment_node)
        deployment_country_frames.append(deployment_country)
        previous_capacity = capacity
        previous_year = year

        power_emissions = _power_emissions_country_carrier(network)
        if "year" in power_emissions.columns:
            power_emissions["year"] = int(year)
        else:
            power_emissions.insert(0, "year", int(year))
        power_emissions_frames.append(
            power_emissions.loc[:, OUTPUT_TABLE_SPECS["power_emissions_country_carrier.csv"]]
            if not power_emissions.empty
            else _empty_frame("power_emissions_country_carrier.csv")
        )

        sector_emissions = _sector_emissions(network)
        if "year" in sector_emissions.columns:
            sector_emissions["year"] = int(year)
        else:
            sector_emissions.insert(0, "year", int(year))
        sector_emissions_frames.append(
            sector_emissions.loc[:, OUTPUT_TABLE_SPECS["sector_emissions.csv"]]
            if not sector_emissions.empty
            else _empty_frame("sector_emissions.csv")
        )

        sector_demands = _sector_demands_country(network)
        if "year" in sector_demands.columns:
            sector_demands["year"] = int(year)
        else:
            sector_demands.insert(0, "year", int(year))
        sector_demands_frames.append(
            sector_demands.loc[:, OUTPUT_TABLE_SPECS["sector_demands_country.csv"]]
            if not sector_demands.empty
            else _empty_frame("sector_demands_country.csv")
        )

        node_prices = _price_node_year(network)
        if "year" in node_prices.columns:
            node_prices["year"] = int(year)
        else:
            node_prices.insert(0, "year", int(year))
        node_price_frames.append(
            node_prices.loc[:, OUTPUT_TABLE_SPECS["electricity_price_node_year.csv"]]
            if not node_prices.empty
            else _empty_frame("electricity_price_node_year.csv")
        )

        country_prices = _price_country_year(network)
        if "year" in country_prices.columns:
            country_prices["year"] = int(year)
        else:
            country_prices.insert(0, "year", int(year))
        country_price_frames.append(
            country_prices.loc[:, OUTPUT_TABLE_SPECS["electricity_price_country_year.csv"]]
            if not country_prices.empty
            else _empty_frame("electricity_price_country_year.csv")
        )

        system_shadows = _system_constraint_shadows(network)
        if "year" in system_shadows.columns:
            system_shadows["year"] = int(year)
        else:
            system_shadows.insert(0, "year", int(year))
        system_shadow_frames.append(
            system_shadows.loc[:, OUTPUT_TABLE_SPECS["constraint_shadow_system_year.csv"]]
            if not system_shadows.empty
            else _empty_frame("constraint_shadow_system_year.csv")
        )

        asset_shadows = _asset_constraint_shadows(network)
        if "year" in asset_shadows.columns:
            asset_shadows["year"] = int(year)
        else:
            asset_shadows.insert(0, "year", int(year))
        asset_shadow_frames.append(
            asset_shadows.loc[:, OUTPUT_TABLE_SPECS["constraint_shadow_asset_year.csv"]]
            if not asset_shadows.empty
            else _empty_frame("constraint_shadow_asset_year.csv")
        )

        lcoe = _lcoe_electricity_tech_year(network)
        if "year" in lcoe.columns:
            lcoe["year"] = int(year)
        else:
            lcoe.insert(0, "year", int(year))
        lcoe_frames.append(
            lcoe.loc[:, OUTPUT_TABLE_SPECS["lcoe_electricity_tech_year.csv"]]
            if not lcoe.empty
            else _empty_frame("lcoe_electricity_tech_year.csv")
        )

        system_summary = _system_summary(
            network,
            generation,
            power_emissions,
            sector_emissions,
            capacity,
        )
        if "year" in system_summary.columns:
            system_summary["year"] = int(year)
        else:
            system_summary.insert(0, "year", int(year))
        system_summary_frames.append(system_summary.loc[:, OUTPUT_TABLE_SPECS["system_summary.csv"]])

    tables["learning_costs.csv"] = learning_costs
    tables["generation_country_carrier.csv"] = pd.concat(generation_frames, ignore_index=True)
    tables["ac_energy_balance_country_carrier.csv"] = pd.concat(ac_balance_frames, ignore_index=True)
    tables["capacity_country_carrier.csv"] = pd.concat(capacity_frames, ignore_index=True)
    tables["battery_operations_country_year.csv"] = pd.concat(battery_operation_frames, ignore_index=True)
    tables["deployment_country_carrier.csv"] = pd.concat(deployment_country_frames, ignore_index=True)
    tables["deployment_node_carrier.csv"] = pd.concat(deployment_node_frames, ignore_index=True)
    tables["power_emissions_country_carrier.csv"] = pd.concat(power_emissions_frames, ignore_index=True)
    tables["sector_emissions.csv"] = pd.concat(sector_emissions_frames, ignore_index=True)
    tables["sector_demands_country.csv"] = pd.concat(sector_demands_frames, ignore_index=True)
    tables["electricity_price_node_year.csv"] = pd.concat(node_price_frames, ignore_index=True)
    tables["electricity_price_country_year.csv"] = pd.concat(country_price_frames, ignore_index=True)
    tables["constraint_shadow_system_year.csv"] = pd.concat(system_shadow_frames, ignore_index=True)
    tables["constraint_shadow_asset_year.csv"] = pd.concat(asset_shadow_frames, ignore_index=True)
    tables["system_summary.csv"] = pd.concat(system_summary_frames, ignore_index=True)
    tables["lcoe_electricity_tech_year.csv"] = pd.concat(lcoe_frames, ignore_index=True)

    run_manifest = {
        "scenario_tag": metadata["scenario_tag"],
        "learning_model": metadata["learning_model"],
        "learning_seed": metadata["learning_seed"],
        "planning_horizons": years,
        "network_sources": {str(year): str(path) for year, path in network_paths.items()},
        "cost_log_sources": {str(year): str(path) for year, path in cost_log_paths.items()},
        "state_sources": {str(year): str(path) for year, path in state_paths.items()},
    }
    (bundle_dir / "run_manifest.json").write_text(
        json.dumps(_sanitize_json(run_manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    checksums = {"run_manifest.json": _sha256(bundle_dir / "run_manifest.json")}
    row_counts = {}
    for filename, df in tables.items():
        output_path = bundle_dir / filename
        _write_csv(df, output_path)
        checksums[filename] = _sha256(output_path)
        row_counts[filename] = int(len(df))

    if os.environ.get("LEARNING_COMPACT_FORCE_FAIL") == "1":
        raise RuntimeError("Forced compact-output failure for test coverage.")

    complete_payload = {
        "scenario_tag": metadata["scenario_tag"],
        "learning_model": metadata["learning_model"],
        "learning_seed": metadata["learning_seed"],
        "planning_horizons": years,
        "row_counts": row_counts,
        "checksums": checksums,
    }
    (bundle_dir / "compact_complete.json").write_text(
        json.dumps(_sanitize_json(complete_payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main(snakemake):  # pragma: no cover - Snakemake entrypoint
    years = [int(year) for year in snakemake.params.planning_horizons]
    network_paths = _map_paths_by_year(list(snakemake.input.networks), years)
    cost_log_paths = _map_paths_by_year(list(snakemake.input.cost_logs), years)
    state_paths = _map_paths_by_year(list(snakemake.input.states), years)

    bundle_dir = Path(str(snakemake.params.bundle_dir)).resolve()
    temp_dir = bundle_dir.parent / f".{bundle_dir.name}.tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "scenario_tag": str(snakemake.params.scenario_tag),
        "learning_model": str(snakemake.wildcards.learning_model),
        "learning_seed": str(snakemake.wildcards.learning_seed),
    }

    try:
        _extract_bundle(temp_dir, years, network_paths, cost_log_paths, state_paths, metadata)
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        temp_dir.rename(bundle_dir)
        bundle_complete = bundle_dir / "compact_complete.json"
        marker_path = Path(str(snakemake.output.complete)).resolve()
        if marker_path != bundle_complete:
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(bundle_complete, marker_path)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise


if __name__ == "__main__":  # pragma: no cover - script mode
    main(snakemake)
