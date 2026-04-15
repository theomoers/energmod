# -*- coding: utf-8 -*-
"""Helpers for deterministic and stochastic fossil-fuel price inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping

import pandas as pd

from _helpers import create_logger

logger = create_logger(__name__)

FOSSIL_FUELS = ("oil", "gas", "coal")
DEFAULT_FUEL_PRICE_BUNDLE_ROOT = "data/fuel-price-data"
FUEL_PRICE_BUNDLE_MANIFEST = "manifest.json"
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


def load_country_fuel_prices_frame(fuelprices_path, investment_year, costs) -> pd.DataFrame:
    """Load country fuel prices for a given year as a normalized frame."""
    try:
        fuel_prices_df = pd.read_csv(fuelprices_path)
    except FileNotFoundError:
        logger.warning(
            "Fuel prices file not found at %s. Using global defaults.", fuelprices_path
        )
        return pd.DataFrame(columns=["fuel_type", "country", "price_eur_mwh"])

    if "year" not in fuel_prices_df.columns:
        raise ValueError(
            f"Fuel prices file {fuelprices_path} is missing required column 'year'"
        )

    year_data = fuel_prices_df.loc[
        pd.to_numeric(fuel_prices_df["year"], errors="coerce") == int(investment_year)
    ].copy()
    if year_data.empty:
        logger.warning(
            "No fuel price data for year %s in %s. Using global defaults.",
            investment_year,
            fuelprices_path,
        )
        return pd.DataFrame(columns=["fuel_type", "country", "price_eur_mwh"])

    if "market" in year_data.columns:
        year_data["market"] = year_data.apply(
            lambda row: normalize_market_name(row["fuel_type"], row["market"]),
            axis=1,
        )

    for fuel in FOSSIL_FUELS:
        if fuel not in set(year_data["fuel_type"].astype(str).str.lower()):
            logger.info(
                "Static fuel prices for %s/%s missing from %s; defaults will apply where needed.",
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
