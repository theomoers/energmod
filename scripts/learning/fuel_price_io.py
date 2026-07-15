# -*- coding: utf-8 -*-
"""Helpers for deterministic and stochastic fossil-fuel price inputs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping

import pandas as pd

from _helpers import create_logger

logger = create_logger(__name__)

FOSSIL_FUELS = ("oil", "gas", "coal")
DEFAULT_FUEL_PRICE_BUNDLE_ROOT = "data/fuel-price-data"
FUEL_PRICE_BUNDLE_MANIFEST = "manifest.json"
HISTORICAL_AVERAGE_WINDOW_YEARS = 5
GAS_MAPPING_TRANSLATOR = {
    "TTF": "Netherlands_TTF",
    "Henry Hub": "Henry_Hub",
    "Zeebrugge": "Zeebrugge",
    "NBP": "UK_NBP",
}


def extract_country_code_from_bus(bus_name: str) -> str:
    """Extract the ISO2 prefix used in clustered PyPSA bus names."""
    return str(bus_name).split(" ")[0][:2]


def normalize_market_name(fuel_type: str, market_name: str) -> str:
    fuel = str(fuel_type).strip().lower()
    market = str(market_name).strip()
    if fuel == "gas":
        return GAS_MAPPING_TRANSLATOR.get(market, market)
    if fuel == "oil" and market == "Nigerian Forcados":
        return "Forcados"
    return market


def build_country_fuel_price_dict(price_frame: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Convert long-form country fuel prices to nested lookup dict."""
    required = {"fuel_type", "country", "price_eur_mwh"}
    missing = required.difference(price_frame.columns)
    if missing:
        raise ValueError(
            f"Country fuel price frame is missing required columns: {sorted(missing)}"
        )

    fuel_price_dict: Dict[str, Dict[str, float]] = {fuel: {} for fuel in FOSSIL_FUELS}
    for fuel in FOSSIL_FUELS:
        fuel_rows = price_frame.loc[price_frame["fuel_type"].astype(str).str.lower() == fuel]
        if fuel_rows.empty:
            continue
        deduped = (
            fuel_rows.loc[:, ["country", "price_eur_mwh"]]
            .dropna(subset=["country", "price_eur_mwh"])
            .drop_duplicates(subset=["country"], keep="last")
        )
        fuel_price_dict[fuel] = {
            str(country): float(price)
            for country, price in zip(deduped["country"], deduped["price_eur_mwh"])
            if str(country).strip()
        }
    return fuel_price_dict


def _resolve_default_bundle_root(fuelprices_path) -> Path:
    """Resolve the default fuel-price bundle relative to the repo root if possible."""
    path = Path(fuelprices_path)
    candidates = []
    if path.is_absolute():
        candidates.extend(parent / DEFAULT_FUEL_PRICE_BUNDLE_ROOT for parent in path.parents)
    candidates.append(Path.cwd() / DEFAULT_FUEL_PRICE_BUNDLE_ROOT)
    for candidate in candidates:
        if (candidate / FUEL_PRICE_BUNDLE_MANIFEST).exists():
            return candidate.resolve()
    return (Path.cwd() / DEFAULT_FUEL_PRICE_BUNDLE_ROOT).resolve()


def load_historical_average_country_fuel_prices_frame(
    fuelprices_path,
    investment_year,
    window_years: int = HISTORICAL_AVERAGE_WINDOW_YEARS,
    market_overrides=None,
) -> pd.DataFrame:
    """Build country fuel prices from a fixed historical/forecast market window."""
    bundle_root = _resolve_default_bundle_root(fuelprices_path)
    manifest_path = bundle_root / FUEL_PRICE_BUNDLE_MANIFEST
    if not manifest_path.exists():
        logger.info(
            "Fuel-price bundle not found at %s; using static country fuel prices.",
            bundle_root,
        )
        return pd.DataFrame(columns=["fuel_type", "country", "price_eur_mwh"])

    try:
        manifest, resolved_root = load_fuel_price_bundle_manifest(
            str(bundle_root),
            str(Path.cwd() / "config.yaml"),
        )
        country_map_path = resolve_manifest_artifact(
            manifest,
            resolved_root,
            "country_market_map_csv",
        )
        historical_path = resolve_manifest_artifact(
            manifest,
            resolved_root,
            "historical_market_prices_csv",
        )
        ar1_path = None
        if "ar1_parameters_csv" in manifest.get("artifacts", {}):
            ar1_path = resolve_manifest_artifact(
                manifest,
                resolved_root,
                "ar1_parameters_csv",
            )
        country_map = pd.read_csv(country_map_path, keep_default_na=False)
        historical = pd.read_csv(historical_path)
        ar1_parameters = pd.read_csv(ar1_path) if ar1_path is not None else pd.DataFrame()
        static_prices = pd.read_csv(fuelprices_path)
    except Exception as exc:
        logger.warning(
            "Could not load historical fuel-price bundle from %s (%s); using static country fuel prices.",
            bundle_root,
            exc,
        )
        return pd.DataFrame(columns=["fuel_type", "country", "price_eur_mwh"])

    required_map = {"fuel_type", "country", "market"}
    required_hist = {"fuel_type", "market", "year", "price_eur_mwh"}
    required_static = {"fuel_type", "market", "year", "price_eur_mwh"}
    required_ar1 = {
        "fuel_type",
        "market",
        "phi",
        "mu",
        "last_observed_year",
        "last_observed_log_price",
    }
    if (
        required_map.difference(country_map.columns)
        or required_hist.difference(historical.columns)
        or required_static.difference(static_prices.columns)
        or (not ar1_parameters.empty and required_ar1.difference(ar1_parameters.columns))
    ):
        logger.warning(
            "Fuel-price inputs are missing required columns; using static country fuel prices."
        )
        return pd.DataFrame(columns=["fuel_type", "country", "price_eur_mwh"])

    country_map["fuel_type"] = country_map["fuel_type"].astype(str).str.lower()
    country_map["country"] = country_map["country"].astype(str).str.upper().str.strip()
    country_map["market"] = country_map.apply(
        lambda row: normalize_market_name(row["fuel_type"], row["market"]),
        axis=1,
    )

    investment_year = int(investment_year)
    if market_overrides:
        overrides = pd.DataFrame(market_overrides).copy()
        required_override = {"year", "fuel_type", "country", "market"}
        missing_override = required_override.difference(overrides.columns)
        if missing_override:
            raise ValueError(
                "Historical market overrides are missing required columns: "
                f"{sorted(missing_override)}"
            )
        overrides["year"] = pd.to_numeric(overrides["year"], errors="coerce")
        overrides = overrides.loc[overrides["year"].eq(investment_year)].copy()
        overrides["fuel_type"] = overrides["fuel_type"].astype(str).str.lower().str.strip()
        overrides["country"] = overrides["country"].astype(str).str.upper().str.strip()
        if overrides.duplicated(["fuel_type", "country"]).any():
            raise ValueError("Historical market overrides must be unique by year, fuel_type, and country.")
        for row in overrides.itertuples(index=False):
            match = country_map["fuel_type"].eq(row.fuel_type) & country_map["country"].eq(row.country)
            if int(match.sum()) != 1:
                raise ValueError(
                    f"Historical market override has no unique country mapping: {row.fuel_type}/{row.country}."
                )
            country_map.loc[match, "market"] = normalize_market_name(row.fuel_type, row.market)

    historical["fuel_type"] = historical["fuel_type"].astype(str).str.lower()
    historical["market"] = historical.apply(
        lambda row: normalize_market_name(row["fuel_type"], row["market"]),
        axis=1,
    )
    historical["year"] = pd.to_numeric(historical["year"], errors="coerce")
    historical["price_eur_mwh"] = pd.to_numeric(historical["price_eur_mwh"], errors="coerce")

    static_prices["fuel_type"] = static_prices["fuel_type"].astype(str).str.lower()
    static_prices["market"] = static_prices.apply(
        lambda row: normalize_market_name(row["fuel_type"], row["market"]),
        axis=1,
    )
    static_prices["year"] = pd.to_numeric(static_prices["year"], errors="coerce")
    static_prices["price_eur_mwh"] = pd.to_numeric(
        static_prices["price_eur_mwh"],
        errors="coerce",
    )

    if not ar1_parameters.empty:
        ar1_parameters["fuel_type"] = ar1_parameters["fuel_type"].astype(str).str.lower()
        ar1_parameters["market"] = ar1_parameters.apply(
            lambda row: normalize_market_name(row["fuel_type"], row["market"]),
            axis=1,
        )
        for column in [
            "phi",
            "mu",
            "last_observed_year",
            "last_observed_log_price",
        ]:
            ar1_parameters[column] = pd.to_numeric(ar1_parameters[column], errors="coerce")

    first_year = investment_year - int(window_years) + 1
    target_years = set(range(first_year, investment_year + 1))

    historical_window = historical.loc[
        historical["fuel_type"].isin(FOSSIL_FUELS)
        & historical["year"].isin(target_years)
        & historical["price_eur_mwh"].notna(),
        ["fuel_type", "market", "year", "price_eur_mwh"],
    ].copy()
    historical_window["_priority"] = 1
    historical_window["source"] = "historical"

    forecast_window = (
        static_prices.loc[
            static_prices["fuel_type"].isin(FOSSIL_FUELS)
            & static_prices["year"].isin(target_years)
            & static_prices["price_eur_mwh"].notna(),
            ["fuel_type", "market", "year", "price_eur_mwh"],
        ]
        .groupby(["fuel_type", "market", "year"], as_index=False)
        .agg(price_eur_mwh=("price_eur_mwh", "mean"))
    )
    forecast_window["_priority"] = 0
    forecast_window["source"] = "forecast"

    ar1_window = pd.DataFrame(columns=["fuel_type", "market", "year", "price_eur_mwh"])
    if not ar1_parameters.empty:
        rows = []
        for row in ar1_parameters.itertuples(index=False):
            if row.fuel_type not in FOSSIL_FUELS:
                continue
            if pd.isna(row.phi) or pd.isna(row.mu) or pd.isna(row.last_observed_year):
                continue
            if pd.isna(row.last_observed_log_price):
                continue
            for year in sorted(target_years):
                if year <= int(row.last_observed_year):
                    continue
                steps = int(year - int(row.last_observed_year))
                log_price = row.mu + (row.phi ** steps) * (row.last_observed_log_price - row.mu)
                rows.append((row.fuel_type, row.market, year, math.exp(log_price)))
        ar1_window = pd.DataFrame(
            rows,
            columns=["fuel_type", "market", "year", "price_eur_mwh"],
        )
    ar1_window["_priority"] = -1
    ar1_window["source"] = "ar1_expected_forecast"

    window_frames = [
        frame
        for frame in [ar1_window, forecast_window, historical_window]
        if not frame.empty
    ]
    window = pd.concat(window_frames, ignore_index=True, sort=False) if window_frames else pd.DataFrame()
    if window.empty:
        logger.info(
            "No historical or forecast fuel prices available for %s-%s; using static country fuel prices.",
            first_year,
            investment_year,
        )
        return pd.DataFrame(columns=["fuel_type", "country", "price_eur_mwh"])

    window = (
        window.sort_values(["fuel_type", "market", "year", "_priority"])
        .drop_duplicates(["fuel_type", "market", "year"], keep="last")
    )

    market_years = pd.MultiIndex.from_product(
        [
            sorted(country_map.loc[country_map["fuel_type"].isin(FOSSIL_FUELS), "fuel_type"].unique()),
            sorted(country_map.loc[country_map["fuel_type"].isin(FOSSIL_FUELS), "market"].unique()),
            sorted(target_years),
        ],
        names=["fuel_type", "market", "year"],
    ).to_frame(index=False)
    valid_market_years = country_map.loc[
        country_map["fuel_type"].isin(FOSSIL_FUELS),
        ["fuel_type", "market"],
    ].drop_duplicates()
    market_years = market_years.merge(valid_market_years, on=["fuel_type", "market"], how="inner")
    global_forecast = (
        static_prices.loc[
            static_prices["fuel_type"].isin(FOSSIL_FUELS)
            & static_prices["year"].isin(target_years)
            & static_prices["price_eur_mwh"].notna(),
            ["fuel_type", "year", "price_eur_mwh"],
        ]
        .groupby(["fuel_type", "year"], as_index=False)
        .agg(global_forecast_price_eur_mwh=("price_eur_mwh", "mean"))
    )
    window = market_years.merge(
        window.drop(columns=["_priority"], errors="ignore"),
        on=["fuel_type", "market", "year"],
        how="left",
    ).merge(global_forecast, on=["fuel_type", "year"], how="left")
    if "source" not in window.columns:
        window["source"] = pd.Series(dtype="object")
    else:
        window["source"] = window["source"].astype("object")
    ar1_market_forecast = ar1_window.loc[
        :,
        ["fuel_type", "market", "year", "price_eur_mwh"],
    ].rename(columns={"price_eur_mwh": "ar1_forecast_price_eur_mwh"})
    window = window.merge(
        ar1_market_forecast,
        on=["fuel_type", "market", "year"],
        how="left",
    )
    ar1_filled = window["price_eur_mwh"].isna() & window["ar1_forecast_price_eur_mwh"].notna()
    if ar1_filled.any():
        window.loc[ar1_filled, "price_eur_mwh"] = window.loc[
            ar1_filled,
            "ar1_forecast_price_eur_mwh",
        ]
        window.loc[ar1_filled, "source"] = "ar1_expected_forecast"

    filled = window["price_eur_mwh"].isna() & window["global_forecast_price_eur_mwh"].notna()
    if filled.any():
        window.loc[filled, "price_eur_mwh"] = window.loc[filled, "global_forecast_price_eur_mwh"]
        window.loc[filled, "source"] = "forecast_fuel_average"

    window = window.sort_values(["fuel_type", "market", "year"])
    nearest = window.groupby(["fuel_type", "market"])["price_eur_mwh"].transform(
        lambda s: s.infer_objects(copy=False).ffill().bfill()
    )
    nearest_filled = window["price_eur_mwh"].isna() & nearest.notna()
    if nearest_filled.any():
        window.loc[nearest_filled, "price_eur_mwh"] = nearest.loc[nearest_filled]
        window.loc[nearest_filled, "source"] = "nearest_market_fill"
    window = window.loc[window["price_eur_mwh"].notna()].copy()

    market_average = (
        window.groupby(["fuel_type", "market"], as_index=False)
        .agg(
            price_eur_mwh=("price_eur_mwh", "mean"),
            first_window_year=("year", "min"),
            last_window_year=("year", "max"),
            window_year_count=("year", "nunique"),
            forecast_year_count=("source", lambda s: int(s.astype(str).str.startswith("forecast").sum())),
        )
    )
    averaged = country_map.merge(market_average, on=["fuel_type", "market"], how="inner")
    averaged = averaged.loc[
        averaged["fuel_type"].isin(FOSSIL_FUELS)
        & averaged["country"].str.len().eq(2)
    ].copy()
    if averaged.empty:
        logger.info(
            "No country mappings matched fuel prices for %s-%s; using static country fuel prices.",
            first_year,
            investment_year,
        )
        return pd.DataFrame(columns=["fuel_type", "country", "price_eur_mwh"])

    averaged["year"] = investment_year
    averaged["source"] = f"historical_forecast_average_{first_year}_{investment_year}"
    short = int((averaged["window_year_count"] < int(window_years)).sum())
    forecast = int((averaged["forecast_year_count"] > 0).sum())
    logger.info(
        "Using fixed %d-year historical/forecast average fossil prices for %s (%s-%s): %d country/fuel rows, %d with forecast-filled years, %d with fewer observations.",
        window_years,
        investment_year,
        first_year,
        investment_year,
        len(averaged),
        forecast,
        short,
    )
    return averaged.loc[
        :,
        [
            "country",
            "fuel_type",
            "year",
            "price_eur_mwh",
            "market",
            "first_window_year",
            "last_window_year",
            "window_year_count",
            "forecast_year_count",
            "source",
        ],
    ]


def load_country_fuel_prices_frame(fuelprices_path, investment_year, costs) -> pd.DataFrame:
    """Load country fuel prices for a given year as a normalized frame."""
    try:
        fuel_prices_df = pd.read_csv(fuelprices_path)
    except FileNotFoundError:
        logger.warning(
            "Fuel prices file not found at %s. Using global defaults.", fuelprices_path
        )
        fuel_prices_df = pd.DataFrame(columns=["fuel_type", "country", "price_eur_mwh", "year"])

    if "year" not in fuel_prices_df.columns:
        raise ValueError(
            f"Fuel prices file {fuelprices_path} is missing required column 'year'"
        )

    year_data = fuel_prices_df.loc[
        pd.to_numeric(fuel_prices_df["year"], errors="coerce") == int(investment_year)
    ].copy()
    if year_data.empty:
        logger.warning(
            "No static fuel price data for year %s in %s. Historical averages or global defaults will apply.",
            investment_year,
            fuelprices_path,
        )

    if not year_data.empty and "market" in year_data.columns:
        year_data["market"] = year_data.apply(
            lambda row: normalize_market_name(row["fuel_type"], row["market"]),
            axis=1,
        )

    averaged = load_historical_average_country_fuel_prices_frame(
        fuelprices_path,
        investment_year,
    )
    if not averaged.empty:
        if year_data.empty:
            year_data = averaged
        else:
            year_data["_priority"] = 0
            averaged["_priority"] = 1
            year_data = (
                pd.concat([year_data, averaged], ignore_index=True, sort=False)
                .sort_values(["country", "fuel_type", "_priority"])
                .drop_duplicates(["country", "fuel_type"], keep="last")
                .drop(columns=["_priority"])
            )

    for fuel in FOSSIL_FUELS:
        if year_data.empty or fuel not in set(year_data["fuel_type"].astype(str).str.lower()):
            logger.info(
                "Fuel prices for %s/%s missing from historical averages and %s; defaults will apply where needed.",
                investment_year,
                fuel,
                fuelprices_path,
            )

    return year_data


def load_country_fuel_prices_dict(fuelprices_path, investment_year, costs) -> Dict[str, Dict[str, float]]:
    """Load country fuel prices into the legacy nested dict structure."""
    return build_country_fuel_price_dict(
        load_country_fuel_prices_frame(fuelprices_path, investment_year, costs)
    )


def get_fuel_price_by_node(nodes, carrier, costs, fuel_price_dict):
    """Map node names to country fuel prices with default fallback."""
    default_price = float(costs.at[carrier, "fuel"])
    if not fuel_price_dict or carrier not in fuel_price_dict:
        return default_price

    country_prices = fuel_price_dict[carrier]
    prices = []
    missing_countries = set()
    for node in nodes:
        country_code = extract_country_code_from_bus(node)
        if country_code in country_prices:
            prices.append(country_prices[country_code])
        else:
            prices.append(default_price)
            missing_countries.add(country_code)

    if missing_countries:
        logger.warning(
            "Fuel '%s': no country-specific prices found for %s. Using default price %.2f EUR/MWh",
            carrier,
            sorted(missing_countries),
            default_price,
        )

    return pd.Series(prices, index=nodes)


def resolve_bundle_root(bundle_root: str, config_file: str) -> Path:
    root = Path(bundle_root or DEFAULT_FUEL_PRICE_BUNDLE_ROOT)
    if not root.is_absolute():
        root = Path(config_file).resolve().parent / root
    return root.resolve()


def load_fuel_price_bundle_manifest(bundle_root: str, config_file: str) -> tuple[dict, Path]:
    root = resolve_bundle_root(bundle_root, config_file)
    manifest_path = root / FUEL_PRICE_BUNDLE_MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(f"Fuel price bundle manifest not found: {manifest_path}")
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    manifest["_sha256"] = __import__("hashlib").sha256(
        manifest_text.encode("utf-8")
    ).hexdigest()
    return manifest, root


def resolve_manifest_artifact(manifest: Mapping[str, object], bundle_root: Path, key: str) -> Path:
    artifacts = manifest.get("artifacts", {}) or {}
    relative = artifacts.get(key)
    if not relative:
        raise ValueError(f"Fuel price bundle manifest is missing artifact '{key}'")
    return (bundle_root / str(relative)).resolve()


def ensure_unique_mapping_rows(frame: pd.DataFrame, keys: Iterable[str], label: str) -> None:
    duplicates = frame.duplicated(subset=list(keys), keep=False)
    if duplicates.any():
        dup_rows = frame.loc[duplicates, list(keys)].drop_duplicates().to_dict(orient="records")
        raise ValueError(f"Duplicate {label} rows found for keys {dup_rows[:10]}")
