#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / "Snakefile").exists() or (candidate / "notebooks" / "NE_paper" / "notebook_config.py").exists():
            return candidate
    raise RuntimeError("Could not find project root containing Snakefile or notebooks/NE_paper/notebook_config.py")


ROOT = find_project_root()
NE_PAPER_DIR = ROOT / "notebooks" / "NE_paper"
if str(NE_PAPER_DIR) not in sys.path:
    sys.path.insert(0, str(NE_PAPER_DIR))

try:
    from notebook_config import eur2020_series_to_usd2025  # type: ignore  # noqa: E402

    HAS_USD2025_CONVERTER = True
except ModuleNotFoundError:
    HAS_USD2025_CONVERTER = False

    def eur2020_series_to_usd2025(value: Any) -> Any:
        raise RuntimeError("notebooks/NE_paper/notebook_config.py is unavailable; cannot convert EUR2020 to USD2025")


DEFAULT_INPUT_ROOT = ROOT / "results" / "Global_200"
DEFAULT_COMPACT_DIR = "learning-compact"
DEFAULT_MODEL_DIR = "shared_state_bayesian_regime_wright"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "NE_paper" / "tables" / "sensitivity_long"

RENEWABLE_GROUPS = {"Solar", "Onshore Wind", "Offshore Wind", "Wind", "Hydro", "Geothermal", "Biomass"}
FOSSIL_GROUPS = {"Natural Gas", "Coal", "Oil", "Lignite"}
NON_GENERATION_CARRIERS = {"battery discharger", "phs", "h2 fuel cell", "load shedding", "load"}
TECH_ORDER = ["solar_power", "onwind_power", "battery_energy"]
TECH_INVESTMENT_UNITS = {"solar_power": 1e6, "onwind_power": 1e6, "battery_energy": 1e6}

OUTPUT_FIELDS = [
    "run_id",
    "seed",
    "year",
    "metric",
    "value",
    "unit",
    "technology",
    "sector",
    "carrier",
    "component",
    "country",
    "source_table",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract long-form sensitivity metrics from compact run_bundle.nc files.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT, help="Directory containing sens_* folders.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Model subdirectory under each sens_* folder.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for one long CSV per sens_* run.")
    parser.add_argument("--runs", nargs="*", default=None, help="Optional run ids or sens_* folder names to process.")
    parser.add_argument("--smoke-test", action="store_true", help="Process only a few seeds per run.")
    parser.add_argument("--smoke-seeds", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="Delete existing per-run output before processing.")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--strict", action="store_true", help="Fail on unreadable seeds instead of recording failures and continuing.")
    return parser.parse_args()


def safe_slug(text: str) -> str:
    return (
        str(text)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("+", "plus")
    )


def run_id_from_sens_dir(path: Path) -> str:
    return path.name.removeprefix("sens_")


def normalize_run_filter(runs: Iterable[str] | None) -> set[str] | None:
    if not runs:
        return None
    return {run.removeprefix("sens_") for run in runs}


def sens_dirs(input_root: Path, runs: Iterable[str] | None) -> list[Path]:
    run_filter = normalize_run_filter(runs)
    dirs = sorted(path for path in input_root.glob("sens_*") if path.is_dir())
    if run_filter is not None:
        dirs = [path for path in dirs if run_id_from_sens_dir(path) in run_filter]
    return dirs


def seed_dirs(sens_dir: Path, model_dir: str) -> list[Path]:
    roots = [
        sens_dir / DEFAULT_COMPACT_DIR / sens_dir.name / model_dir,
        sens_dir / model_dir,
    ]
    roots.extend(sorted((sens_dir / "draws").glob(f"*/{DEFAULT_COMPACT_DIR}/{sens_dir.name}/{model_dir}")))

    seeds: list[Path] = []
    seen_names: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.iterdir(), key=lambda item: item.name):
            if not path.is_dir() or not path.name.startswith("seed_") or not (path / "run_bundle.nc").exists():
                continue
            if path.name in seen_names:
                continue
            seen_names.add(path.name)
            seeds.append(path)
    return sorted(seeds, key=lambda path: path.name)


def decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.size == 1:
        return decode_attr(value.item())
    return value


def decode_char_array(array: np.ndarray) -> list[str]:
    values = np.char.decode(array, "utf-8")
    return ["".join(row).rstrip("\x00").strip() for row in values]


def read_table(handle: h5py.File, table_name: str, columns: list[str]) -> pd.DataFrame:
    group_name = f"tables/{table_name}"
    if group_name not in handle:
        return pd.DataFrame()
    group = handle[group_name]
    data: dict[str, Any] = {}
    for column in columns:
        if column not in group:
            continue
        array = group[column][()]
        if getattr(array, "dtype", None) is not None and array.dtype.kind == "S" and array.ndim == 2:
            data[column] = decode_char_array(array)
        elif getattr(array, "dtype", None) is not None and array.dtype.kind == "S":
            data[column] = np.char.decode(array, "utf-8")
        else:
            data[column] = array
    return pd.DataFrame(data)


def table_columns(handle: h5py.File, table_name: str) -> list[str]:
    group_name = f"tables/{table_name}"
    if group_name not in handle:
        return []
    columns = []
    for col in handle[group_name].keys():
        if col.startswith("string"):
            continue
        if col.endswith("_row") or "__" in col and col.endswith("_row"):
            continue
        columns.append(col)
    return columns


def read_table_all(handle: h5py.File, table_name: str) -> pd.DataFrame:
    return read_table(handle, table_name, table_columns(handle, table_name))


def as_number(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def carrier_group(label: str) -> str:
    name = str(label).lower()
    if "solar" in name:
        return "Solar"
    if "onwind" in name or "onshore wind" in name:
        return "Onshore Wind"
    if "offwind" in name or "offshore wind" in name or "wind" in name:
        return "Offshore Wind"
    if any(token in name for token in ["ror", "hydro", "reservoir", "run of river"]):
        return "Hydro"
    if "geothermal" in name:
        return "Geothermal"
    if "biomass" in name:
        return "Biomass"
    if any(token in name for token in ["coal", "lignite"]):
        return "Coal"
    if any(token in name for token in ["gas", "ocgt", "ccgt"]):
        return "Natural Gas"
    if "oil" in name:
        return "Oil"
    return str(label)


def metric_row(
    run_id: str,
    seed: str,
    year: Any,
    metric: str,
    value: Any,
    unit: str,
    source_table: str,
    technology: str = "",
    sector: str = "",
    carrier: str = "",
    component: str = "",
    country: str = "",
) -> dict[str, Any]:
    year_out = ""
    if year not in {"", None} and not pd.isna(year):
        year_out = int(year)
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        value_out = ""
    elif isinstance(value, (dict, list)):
        value_out = json.dumps(value, sort_keys=True)
    else:
        value_out = value
    return {
        "run_id": run_id,
        "seed": seed,
        "year": year_out,
        "metric": metric,
        "value": value_out,
        "unit": unit,
        "technology": technology,
        "sector": sector,
        "carrier": carrier,
        "component": component,
        "country": country,
        "source_table": source_table,
    }


def generation_metrics(run_id: str, seed: str, handle: h5py.File) -> list[dict[str, Any]]:
    generation = read_table(handle, "generation_country_carrier", ["year", "component", "country", "carrier", "annual_generation_twh"])
    if generation.empty:
        return []
    generation["annual_generation_twh"] = pd.to_numeric(generation["annual_generation_twh"], errors="coerce").fillna(0.0)
    generation["carrier_lower"] = generation["carrier"].astype(str).str.lower()
    generation["carrier_group"] = generation["carrier"].map(carrier_group)
    generation["exclude_from_total"] = generation["carrier_lower"].isin(NON_GENERATION_CARRIERS)
    generation["is_renewable"] = generation["carrier_group"].isin(RENEWABLE_GROUPS) & ~generation["exclude_from_total"]
    generation["is_fossil"] = generation["carrier_group"].isin(FOSSIL_GROUPS) & ~generation["exclude_from_total"]
    rows = []
    for year, group in generation.groupby("year", sort=True):
        total = group.loc[~group["exclude_from_total"], "annual_generation_twh"].sum()
        renewable = group.loc[group["is_renewable"], "annual_generation_twh"].sum()
        fossil = group.loc[group["is_fossil"], "annual_generation_twh"].sum()
        solar = group.loc[group["carrier_group"].eq("Solar") & ~group["exclude_from_total"], "annual_generation_twh"].sum()
        wind = group.loc[group["carrier_group"].eq("Onshore Wind") & ~group["exclude_from_total"], "annual_generation_twh"].sum()
        all_wind = group.loc[group["carrier_group"].isin(["Onshore Wind", "Offshore Wind", "Wind"]) & ~group["exclude_from_total"], "annual_generation_twh"].sum()
        denom = total if total else math.nan
        rows.extend(
            [
                metric_row(run_id, seed, year, "total_generation", total, "TWh", "generation_country_carrier"),
                metric_row(run_id, seed, year, "renewable_generation", renewable, "TWh", "generation_country_carrier"),
                metric_row(run_id, seed, year, "fossil_generation", fossil, "TWh", "generation_country_carrier"),
                metric_row(run_id, seed, year, "solar_power", solar, "TWh", "generation_country_carrier", carrier="solar"),
                metric_row(run_id, seed, year, "wind_power", wind, "TWh", "generation_country_carrier", carrier="onwind"),
                metric_row(run_id, seed, year, "all_wind_power", all_wind, "TWh", "generation_country_carrier", carrier="onwind+offwind"),
                metric_row(run_id, seed, year, "renewable_share", renewable / denom if denom else math.nan, "fraction", "generation_country_carrier"),
                metric_row(run_id, seed, year, "fossil_share", fossil / denom if denom else math.nan, "fraction", "generation_country_carrier"),
                metric_row(run_id, seed, year, "solar_share", solar / denom if denom else math.nan, "fraction", "generation_country_carrier", carrier="solar"),
                metric_row(run_id, seed, year, "wind_share", wind / denom if denom else math.nan, "fraction", "generation_country_carrier", carrier="onwind"),
                metric_row(run_id, seed, year, "all_wind_share", all_wind / denom if denom else math.nan, "fraction", "generation_country_carrier", carrier="onwind+offwind"),
            ]
        )
    for row in generation.itertuples(index=False):
        rows.append(
            metric_row(
                run_id,
                seed,
                row.year,
                "generation_country_carrier",
                row.annual_generation_twh,
                "TWh",
                "generation_country_carrier",
                carrier=str(row.carrier),
                component=str(row.component),
                country=str(row.country),
            )
        )
    return rows


def capacity_metrics(run_id: str, seed: str, handle: h5py.File) -> list[dict[str, Any]]:
    capacity = read_table(handle, "capacity_country_carrier", ["year", "component", "country", "carrier", "capacity_value", "capacity_unit"])
    rows = []
    if not capacity.empty:
        capacity["capacity_value"] = pd.to_numeric(capacity["capacity_value"], errors="coerce").fillna(0.0)
        capacity["carrier_lower"] = capacity["carrier"].astype(str).str.lower()
        capacity["component"] = capacity["component"].astype(str)
        for year, group in capacity.groupby("year", sort=True):
            solar = group.loc[group["component"].eq("Generator") & group["carrier_lower"].eq("solar"), "capacity_value"].sum() / 1e6
            wind = group.loc[group["component"].eq("Generator") & group["carrier_lower"].eq("onwind"), "capacity_value"].sum() / 1e6
            rows.append(metric_row(run_id, seed, year, "solar_capacity", solar, "TW", "capacity_country_carrier", carrier="solar", component="Generator"))
            rows.append(metric_row(run_id, seed, year, "wind_capacity", wind, "TW", "capacity_country_carrier", carrier="onwind", component="Generator"))
        for row in capacity.itertuples(index=False):
            unit = str(getattr(row, "capacity_unit", ""))
            scale = 1e6 if unit.lower() in {"mw", "mwh"} else 1.0
            rows.append(
                metric_row(
                    run_id,
                    seed,
                    row.year,
                    "capacity_country_carrier",
                    row.capacity_value / scale,
                    "TW/TWh" if scale == 1e6 else unit,
                    "capacity_country_carrier",
                    carrier=str(row.carrier),
                    component=str(row.component),
                    country=str(row.country),
                )
            )

    battery = read_table(handle, "battery_operations_country_year", ["year", "country", "carrier", "store_energy_capacity_mwh", "discharge_to_ac_mwh"])
    if not battery.empty:
        battery = battery.loc[battery["carrier"].astype(str).str.lower().eq("battery")].copy()
        for col in ["store_energy_capacity_mwh", "discharge_to_ac_mwh"]:
            battery[col] = pd.to_numeric(battery[col], errors="coerce").fillna(0.0)
        for year, group in battery.groupby("year", sort=True):
            rows.append(metric_row(run_id, seed, year, "battery_capacity", group["store_energy_capacity_mwh"].sum() / 1e6, "TWh", "battery_operations_country_year", carrier="battery"))
            rows.append(metric_row(run_id, seed, year, "battery_discharge", group["discharge_to_ac_mwh"].sum() / 1e6, "TWh", "battery_operations_country_year", carrier="battery"))
        for row in battery.itertuples(index=False):
            rows.append(metric_row(run_id, seed, row.year, "battery_capacity_country", row.store_energy_capacity_mwh / 1e6, "TWh", "battery_operations_country_year", carrier="battery", country=str(row.country)))
            rows.append(metric_row(run_id, seed, row.year, "battery_discharge_country", row.discharge_to_ac_mwh / 1e6, "TWh", "battery_operations_country_year", carrier="battery", country=str(row.country)))
    return rows


def emissions_metrics(run_id: str, seed: str, handle: h5py.File) -> list[dict[str, Any]]:
    emissions = read_table(handle, "sector_emissions", ["year", "sector", "emissions_mtco2"])
    if emissions.empty:
        return []
    emissions["emissions_mtco2"] = pd.to_numeric(emissions["emissions_mtco2"], errors="coerce")
    rows = []
    for row in emissions.dropna(subset=["emissions_mtco2"]).itertuples(index=False):
        rows.append(metric_row(run_id, seed, row.year, "sector_emissions", row.emissions_mtco2, "MtCO2/yr", "sector_emissions", sector=str(row.sector)))
    power = emissions.loc[emissions["sector"].eq("Power")].groupby("year", as_index=False)["emissions_mtco2"].sum(min_count=1).sort_values("year")
    if not power.empty:
        years = power["year"].to_numpy(dtype=float)
        values = power["emissions_mtco2"].to_numpy(dtype=float)
        for idx, (year, value) in enumerate(zip(years, values, strict=True)):
            if idx == 0:
                cumulative = 0.0
            else:
                cumulative = float(np.trapz(values[: idx + 1], years[: idx + 1]) / 1000.0)
            rows.append(metric_row(run_id, seed, year, "annual_power_emissions", value / 1000.0, "GtCO2/yr", "sector_emissions", sector="Power"))
            rows.append(metric_row(run_id, seed, year, "cumulative_power_emissions", cumulative, "GtCO2", "sector_emissions", sector="Power"))
    return rows


def system_cost_metrics(run_id: str, seed: str, handle: h5py.File) -> list[dict[str, Any]]:
    costs = read_table(handle, "system_costs", ["year", "component", "country", "carrier", "technology", "annualized_capital_cost_eur", "operating_cost_eur", "total_system_cost_eur"])
    if costs.empty:
        return []
    for col in ["annualized_capital_cost_eur", "operating_cost_eur", "total_system_cost_eur"]:
        costs[col] = pd.to_numeric(costs[col], errors="coerce").fillna(0.0)
    analysis_costs = costs.loc[~costs["carrier"].astype(str).eq("Load")].copy()
    annual = analysis_costs.groupby("year", as_index=False)[["annualized_capital_cost_eur", "operating_cost_eur"]].sum(min_count=1)
    annual["system_cost_total_eur"] = annual["annualized_capital_cost_eur"] + annual["operating_cost_eur"]
    rows = []
    for row in annual.itertuples(index=False):
        rows.append(metric_row(run_id, seed, row.year, "system_cost_annualized_capital", row.annualized_capital_cost_eur, "EUR2020/yr", "system_costs_ex_load"))
        rows.append(metric_row(run_id, seed, row.year, "installed_capex_flow", row.annualized_capital_cost_eur, "EUR2020/yr", "system_costs_ex_load"))
        rows.append(metric_row(run_id, seed, row.year, "system_cost_operating", row.operating_cost_eur, "EUR2020/yr", "system_costs_ex_load"))
        rows.append(metric_row(run_id, seed, row.year, "operating_cost_flow", row.operating_cost_eur, "EUR2020/yr", "system_costs_ex_load"))
        rows.append(metric_row(run_id, seed, row.year, "system_cost_total", row.system_cost_total_eur, "EUR2020/yr", "system_costs_ex_load"))
        rows.append(metric_row(run_id, seed, row.year, "installed_system_cost_flow", row.system_cost_total_eur, "EUR2020/yr", "system_costs_ex_load"))
        if HAS_USD2025_CONVERTER:
            rows.append(metric_row(run_id, seed, row.year, "system_cost_total_usd2025", eur2020_series_to_usd2025(row.system_cost_total_eur), "USD2025/yr", "system_costs_ex_load"))
    for row in costs.itertuples(index=False):
        for col, metric in [
            ("annualized_capital_cost_eur", "system_cost_annualized_capital_detailed"),
            ("operating_cost_eur", "system_cost_operating_detailed"),
            ("total_system_cost_eur", "system_cost_total_detailed"),
        ]:
            rows.append(
                metric_row(
                    run_id,
                    seed,
                    row.year,
                    metric,
                    getattr(row, col),
                    "EUR2020/yr",
                    "system_costs",
                    technology=str(getattr(row, "technology", "")),
                    carrier=str(getattr(row, "carrier", "")),
                    component=str(getattr(row, "component", "")),
                    country=str(getattr(row, "country", "")),
                )
            )
    return rows


def transition_spending_metrics(run_id: str, seed: str, handle: h5py.File) -> list[dict[str, Any]]:
    dep = read_table(handle, "deployment_constraints", ["year", "technology", "realized_block_addition_constrained_basis", "realized_block_addition_modeled", "realized_wedge_cost_eur"])
    costs = read_table(handle, "learning_costs", ["year", "technology", "postsolve_c_overnight", "c_overnight"])
    if dep.empty or costs.empty:
        return []
    dep = dep.loc[dep["technology"].isin(TECH_ORDER)].copy()
    costs = costs.loc[costs["technology"].isin(TECH_ORDER)].copy()
    cost_col = next((col for col in ["postsolve_c_overnight", "c_overnight"] if col in costs and pd.to_numeric(costs[col], errors="coerce").notna().any()), None)
    if cost_col is None:
        return []
    lc = costs[["year", "technology", cost_col]].rename(columns={cost_col: "unit_cost_eur"})
    merged = dep.merge(lc, on=["year", "technology"], how="left")
    addition_cols = [col for col in ["realized_block_addition_constrained_basis", "realized_block_addition_modeled"] if col in merged.columns]
    if addition_cols:
        merged["block_addition"] = merged[addition_cols].apply(pd.to_numeric, errors="coerce").bfill(axis=1).iloc[:, 0]
    else:
        merged["block_addition"] = np.nan
    merged["block_addition"] = pd.to_numeric(merged["block_addition"], errors="coerce").clip(lower=0.0)
    merged["unit_cost_eur"] = pd.to_numeric(merged["unit_cost_eur"], errors="coerce")
    merged["realized_wedge_cost_eur"] = pd.to_numeric(merged.get("realized_wedge_cost_eur", 0.0), errors="coerce").fillna(0.0)
    merged["clean_investment_eur"] = merged.apply(
        lambda row: row["block_addition"] * TECH_INVESTMENT_UNITS.get(row["technology"], np.nan) * row["unit_cost_eur"],
        axis=1,
    )

    def block_cost(value_eur: float) -> tuple[float, str, str]:
        if HAS_USD2025_CONVERTER:
            return eur2020_series_to_usd2025(value_eur) / 1e9, "bn USD2025/block", ""
        return value_eur / 1e9, "bn EUR2020/block", "_eur2020"

    rows = []
    for (year, tech), group in merged.groupby(["year", "technology"], sort=True):
        clean = group["clean_investment_eur"].sum(min_count=1)
        wedge = group["realized_wedge_cost_eur"].sum(min_count=1)
        total = clean + wedge
        clean_value, unit, suffix = block_cost(clean)
        wedge_value, _, _ = block_cost(wedge)
        total_value, _, _ = block_cost(total)
        rows.append(metric_row(run_id, seed, year, f"clean_investment{suffix}", clean_value, unit, "deployment_constraints+learning_costs", technology=tech))
        rows.append(metric_row(run_id, seed, year, f"deployment_wedge_cost{suffix}", wedge_value, unit, "deployment_constraints", technology=tech))
        rows.append(metric_row(run_id, seed, year, f"transition_spending{suffix}", total_value, unit, "deployment_constraints+learning_costs", technology=tech))
    annual = merged.groupby("year", as_index=False)[["clean_investment_eur", "realized_wedge_cost_eur"]].sum(min_count=1)
    for row in annual.itertuples(index=False):
        total = row.clean_investment_eur + row.realized_wedge_cost_eur
        clean_value, unit, suffix = block_cost(row.clean_investment_eur)
        wedge_value, _, _ = block_cost(row.realized_wedge_cost_eur)
        total_value, _, _ = block_cost(total)
        rows.append(metric_row(run_id, seed, row.year, f"clean_investment_total{suffix}", clean_value, unit, "deployment_constraints+learning_costs"))
        rows.append(metric_row(run_id, seed, row.year, f"deployment_wedge_cost_total{suffix}", wedge_value, unit, "deployment_constraints"))
        rows.append(metric_row(run_id, seed, row.year, f"transition_spending_total{suffix}", total_value, unit, "deployment_constraints+learning_costs"))
    return rows


def learning_cost_metrics(run_id: str, seed: str, handle: h5py.File) -> list[dict[str, Any]]:
    costs = read_table_all(handle, "learning_costs")
    if costs.empty:
        return []
    rows = []
    tech_metric_prefix = {
        "solar_power": "solar",
        "onwind_power": "wind",
        "battery_energy": "liion",
    }
    cost_columns = {
        "c_overnight": ("overnight_cost", "EUR2020/kW or EUR2020/kWh"),
        "postsolve_c_overnight": ("postsolve_overnight_cost", "EUR2020/kW or EUR2020/kWh"),
        "c_overnight_terminal_point": ("overnight_cost_terminal_point", "EUR2020/kW or EUR2020/kWh"),
        "postsolve_c_overnight_terminal_point": ("postsolve_overnight_cost_terminal_point", "EUR2020/kW or EUR2020/kWh"),
        "battery_store_cost_eur_per_kwh": ("grid_storage_cost", "EUR2020/kWh"),
        "postsolve_battery_store_cost_eur_per_kwh": ("postsolve_grid_storage_cost", "EUR2020/kWh"),
        "reported_4h_bess_capex_eur_per_kwh": ("grid_storage_4h_bess_capex", "EUR2020/kWh"),
        "postsolve_reported_4h_bess_capex_eur_per_kwh": ("postsolve_grid_storage_4h_bess_capex", "EUR2020/kWh"),
        "capital_cost": ("annualized_capital_cost_input", "EUR2020/MW/yr or EUR2020/MWh/yr"),
        "postsolve_capital_cost": ("postsolve_annualized_capital_cost", "EUR2020/MW/yr or EUR2020/MWh/yr"),
    }
    metadata_cols = [
        "selected_model",
        "training_window",
        "cost_expectation_mode",
        "cost_expectation_kernel_mode",
        "cost_expectation_weights_json",
        "cost_expectation_weights_by_tech_json",
        "cost_expectation_lag_years_json",
        "kernel_year_start",
        "kernel_year_end",
        "kernel_years_json",
        "known_kernel_years_json",
        "expected_kernel_years_json",
        "applied_kernel_costs_json",
        "postsolve_state_year",
    ]
    for row in costs.itertuples(index=False):
        tech = str(getattr(row, "technology", ""))
        prefix = tech_metric_prefix.get(tech, safe_slug(tech))
        year = getattr(row, "year")
        for col, (suffix, unit) in cost_columns.items():
            if hasattr(row, col):
                value = as_number(getattr(row, col))
                if math.isfinite(value):
                    rows.append(metric_row(run_id, seed, year, f"{prefix}_{suffix}", value, unit, "learning_costs", technology=tech))
        for col in metadata_cols:
            if hasattr(row, col):
                value = getattr(row, col)
                if pd.notna(value) and str(value) != "":
                    rows.append(metric_row(run_id, seed, year, f"metadata_{col}", value, "", "learning_costs", technology=tech))
    id_cols = {
        "year",
        "technology",
        "cost_basis_conversion_method",
        "selected_model",
        "learning_seed",
        "training_window",
        "cost_expectation_mode",
        "cost_expectation_kernel_mode",
        "cost_expectation_weights_json",
        "cost_expectation_weights_by_tech_json",
        "cost_expectation_lag_years_json",
        "kernel_year_start",
        "kernel_year_end",
        "kernel_years_json",
        "known_kernel_years_json",
        "expected_kernel_years_json",
        "applied_kernel_costs_json",
        "postsolve_state_year",
    }
    for row in costs.itertuples(index=False):
        tech = str(getattr(row, "technology", ""))
        year = getattr(row, "year")
        for col in costs.columns:
            if col in {"year", "technology"}:
                continue
            value = getattr(row, col)
            if pd.isna(value) or str(value) == "":
                continue
            metric = f"learning_raw_{col}"
            unit = ""
            if col not in id_cols:
                unit = "EUR2020 or native"
            rows.append(metric_row(run_id, seed, year, metric, value, unit, "learning_costs", technology=tech))
    return rows


def wedge_metadata_metrics(run_id: str, seed: str, handle: h5py.File) -> list[dict[str, Any]]:
    dep = read_table_all(handle, "deployment_constraints")
    if dep.empty:
        return []
    rows = []
    metadata_cols = [col for col in dep.columns if col not in {"year", "technology"}]
    for row in dep.itertuples(index=False):
        tech = str(getattr(row, "technology", ""))
        year = getattr(row, "year")
        for col in metadata_cols:
            if hasattr(row, col):
                value = getattr(row, col)
                if pd.notna(value) and str(value) != "":
                    prefix = "metadata_wedge" if not isinstance(value, (int, float, np.number)) else "deployment_raw"
                    rows.append(metric_row(run_id, seed, year, f"{prefix}_{col}", value, "", "deployment_constraints", technology=tech))
    return rows


def system_summary_metrics(run_id: str, seed: str, handle: h5py.File) -> list[dict[str, Any]]:
    summary = read_table_all(handle, "system_summary")
    if summary.empty:
        return []
    rows = []
    for row in summary.itertuples(index=False):
        year = getattr(row, "year")
        for col in summary.columns:
            if col == "year":
                continue
            value = getattr(row, col)
            if pd.isna(value) or str(value) == "":
                continue
            rows.append(metric_row(run_id, seed, year, f"system_summary_{col}", value, "", "system_summary"))
    return rows


def manifest_metadata_metrics(run_id: str, seed: str, handle: h5py.File) -> list[dict[str, Any]]:
    raw = handle.attrs.get("run_manifest_json")
    if raw is None:
        return []
    try:
        manifest = json.loads(decode_attr(raw))
    except Exception:
        return []
    rows = []
    for key in ["learning_model", "learning_seed", "scenario_tag"]:
        if key in manifest:
            rows.append(metric_row(run_id, seed, "", f"metadata_manifest_{key}", manifest[key], "", "run_bundle.attrs"))
    if "planning_horizons" in manifest:
        rows.append(metric_row(run_id, seed, "", "metadata_manifest_planning_horizons", manifest["planning_horizons"], "", "run_bundle.attrs"))
    return rows


def extract_seed_rows(run_id: str, seed_dir: Path, retries: int, retry_sleep: float) -> list[dict[str, Any]]:
    bundle = seed_dir / "run_bundle.nc"
    last_exc: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            with h5py.File(bundle, "r") as handle:
                rows: list[dict[str, Any]] = []
                rows.extend(manifest_metadata_metrics(run_id, seed_dir.name, handle))
                rows.extend(system_summary_metrics(run_id, seed_dir.name, handle))
                rows.extend(generation_metrics(run_id, seed_dir.name, handle))
                rows.extend(capacity_metrics(run_id, seed_dir.name, handle))
                rows.extend(emissions_metrics(run_id, seed_dir.name, handle))
                rows.extend(system_cost_metrics(run_id, seed_dir.name, handle))
                rows.extend(transition_spending_metrics(run_id, seed_dir.name, handle))
                rows.extend(learning_cost_metrics(run_id, seed_dir.name, handle))
                rows.extend(wedge_metadata_metrics(run_id, seed_dir.name, handle))
                return rows
        except (OSError, TimeoutError) as exc:
            last_exc = exc
            if attempt >= retries:
                break
            time.sleep(retry_sleep)
    assert last_exc is not None
    raise last_exc


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in OUTPUT_FIELDS})


def completed_seeds(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            seed = row.get("seed")
            if seed:
                out.add(seed)
    return out


def append_failure(path: Path, run_id: str, seed: str, bundle: Path, exc: BaseException) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["run_id", "seed", "bundle", "error_type", "error"])
        if write_header:
            writer.writeheader()
        writer.writerow({"run_id": run_id, "seed": seed, "bundle": bundle, "error_type": type(exc).__name__, "error": str(exc)})


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failure_path = args.output_dir / "failures.csv"
    dirs = sens_dirs(args.input_root, args.runs)
    if not dirs:
        raise SystemExit(f"No sens_* directories found under {args.input_root}")

    for sens_idx, sens_dir in enumerate(dirs, start=1):
        run_id = run_id_from_sens_dir(sens_dir)
        out_path = args.output_dir / f"{sens_dir.name}_long_metrics.csv"
        if args.force:
            out_path.unlink(missing_ok=True)
        dirs_for_run = seed_dirs(sens_dir, args.model_dir)
        if args.smoke_test:
            dirs_for_run = dirs_for_run[: args.smoke_seeds]
        done = completed_seeds(out_path)
        if done:
            print(f"{run_id}: resuming with {len(done)} completed seeds in {out_path}", flush=True)
        print(f"{run_id}: {sens_idx}/{len(dirs)} runs, {len(dirs_for_run)} seed bundles", flush=True)
        for seed_idx, seed_dir in enumerate(dirs_for_run, start=1):
            if seed_dir.name in done:
                continue
            if seed_idx == 1 or seed_idx % 25 == 0 or seed_idx == len(dirs_for_run):
                print(f"{run_id}: seed {seed_idx}/{len(dirs_for_run)} {seed_dir.name}", flush=True)
            try:
                rows = extract_seed_rows(run_id, seed_dir, args.retries, args.retry_sleep)
                append_rows(out_path, rows)
            except Exception as exc:
                append_failure(failure_path, run_id, seed_dir.name, seed_dir / "run_bundle.nc", exc)
                print(f"WARNING: skipped {run_id}/{seed_dir.name}: {type(exc).__name__}: {exc}", flush=True)
                if args.strict:
                    raise
        print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
