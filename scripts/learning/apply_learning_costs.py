# -*- coding: utf-8 -*-
"""
Apply learning-driven technology costs to PyPSA-Earth myopic networks.

This script updates overnight CAPEX before each myopic solve and writes the
runtime state needed for the next horizon. The deterministic legacy Wright path
uses the exported regression parameters directly. The stochastic paths use the
bundled runtime artifacts for the shortlisted models.

Key timing rule:
- costs used to solve horizon t are fixed before that solve
- deployment realized in horizon t only affects costs applied in horizon t+5

For deployment-conditioned stochastic models, the t+5 update is based on
realized cumulative deployment between the previous committed horizon and the
current committed horizon. The runtime does not forecast annual deployment
within the block.

Created: 2025-11-06
Updated: 2026-03-26
"""

import logging
import sys
import re
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pypsa
import yaml

SCRIPTS_DIR = Path.cwd() / "scripts"
if (SCRIPTS_DIR / "_helpers.py").exists():
    sys.path.insert(0, str(SCRIPTS_DIR))

from _helpers import configure_logging, create_logger
from learning.fuel_price_io import (
    FOSSIL_FUELS,
    build_country_fuel_price_dict,
    ensure_unique_mapping_rows,
    extract_country_code_from_bus,
    load_fuel_price_bundle_manifest,
    normalize_market_name,
    resolve_manifest_artifact,
)
from learning.learning_data_io import load_historical_capacity, load_historical_cost

# Battery energy-side BOS multiplier reference year.
# We derive the multiplier from data as:
# (battery storage investment cost in costs_2020.csv) /
# (historical Li-ion pack cost in 2020 EUR/kWh),
# then apply that multiplier to learning-based battery_energy costs.
BATTERY_ENERGY_BOS_REFERENCE_YEAR = 2020

# Global scaling factors for technologies where model covers only a fraction of
# global deployment. Battery energy is handled separately via the bundled
# phi_t mapping from modeled grid additions to global Li-ion additions.
DEFAULT_GLOBAL_SCALE_FACTORS = {}

DEFAULT_TECH_MAPPING = {
    "solar": "solar_power",
    "onwind": "onwind_power",
    "battery": "battery_energy",
    "battery inverter": "battery_power",
    "H2 Electrolysis": "electrolyser_power",
    "H2 Store": "h2_energy",
}

DEFAULT_FINANCE = {
    "wacc": {
        "default": 0.07,
        "battery_energy": 0.08,
        "battery_power": 0.08,
    },
    "lifetime": {
        "solar_power": 35,
        "onwind_power": 27,
        "battery_power": 10,
        "battery_energy": 20,
        "electrolyser_power": 15,
        "h2_energy": 30,
    },
}

SUPPORTED_LEARNING_ENGINES = {"legacy_curve", "stochastic_forecast", "exogenous_path"}
SUPPORTED_STOCHASTIC_MODELS = {
    "shared_state_bayesian_regime_wright",
    "way_fixed_rho_benchmark_035",
    "correlated_geometric_random_walk",
}
SUPPORTED_EXOGENOUS_MODELS = {"iea_weo_exogenous_path"}
SUPPORTED_SAMPLE_MODES = {"single_draw", "median"}
SUPPORTED_TRAINING_WINDOWS = {"origin_cutoff", "full_sample"}
SUPPORTED_COST_EXPECTATION_MODES = {"point_cost", "block_average_expected"}
SUPPORTED_FOSSIL_PRICE_EXPECTATION_MODES = {"point_cost", "block_average_expected"}
BATTERY_POWER_TREATMENT = "deterministic_default_costs"
LEGACY_LEARNING_SEED = "deterministic"
DEFAULT_COST_EXPECTATION_MODE = "point_cost"
DEFAULT_FOSSIL_PRICE_EXPECTATION_MODE = "point_cost"
BLOCK_EXPECTATION_DRAWS = 1000
DEFAULT_BLOCK_EXPECTATION_ANNUAL_WEIGHTS = (0.2, 0.2, 0.2, 0.2, 0.2)
DEFAULT_FOSSIL_PRICE_ANNUAL_STEP = 1

# Mapping of solve horizon to the historical lag-year used for bootstrap
# cumulative capacity (experience) before switching to solved lagged blocks.
# 2020 solve uses cumulative_capacity[2015].
# 2025 solve uses cumulative_capacity[2020].
# From 2030 onward, the usual lagged solved rule applies.
COST_HISTORICAL_CAPACITY_YEARS = {
    2020: 2015,
    2025: 2020,
}

# Bootstrap cost windows (inclusive) for deterministic historical pricing.
# 2020 solve uses average historical cost over 2015-2019.
# 2025 solve uses average historical cost over 2020-2024.
COST_HISTORICAL_COST_WINDOWS = {
    2020: (2015, 2019),
    2025: (2020, 2024),
}


def get_global_scale_factors(learning_cfg):
    """
    Build effective global scale factors from defaults + config overrides.

    Config options:
    - learning.global_scale_factors: direct per-technology multipliers
      (global deployment = modeled deployment × factor)
    - learning.global_factor: optional legacy/endogenous-style fraction.
      If provided and positive, derive battery multipliers as 1/global_factor
      unless explicitly overridden in global_scale_factors.
    """
    factors = DEFAULT_GLOBAL_SCALE_FACTORS.copy()

    cfg_scales = learning_cfg.get("global_scale_factors", {}) or {}
    for tech, value in cfg_scales.items():
        factor = float(value)
        if factor <= 0:
            raise ValueError(
                f"Invalid global scale factor for {tech}: {factor}. Must be > 0."
            )
        factors[tech] = factor

    # Optional compatibility bridge with endogenous learning config semantics.
    # There, global_factor is a local/global fraction; here we need global/modeled.
    legacy_gf = learning_cfg.get("global_factor", None)
    if legacy_gf is not None:
        legacy_gf = float(legacy_gf)
        if legacy_gf <= 0:
            raise ValueError(
                f"Invalid learning.global_factor={legacy_gf}. Must be > 0."
            )
        derived_scale = 1.0 / legacy_gf
        for tech in ("battery_energy", "battery_power"):
            if tech not in cfg_scales:
                factors[tech] = derived_scale

    logger.info(f"Using global scale factors: {factors}")
    return factors


def _resolve_reference_costs_file(costs_file, reference_year):
    path = Path(costs_file)
    match = re.search(r"costs_(\d{4})\.csv$", path.name)
    if not match:
        raise ValueError(
            f"Cannot infer reference costs file from path: {costs_file}. "
            "Expected filename pattern costs_<year>.csv."
        )
    candidate = path.with_name(f"costs_{int(reference_year)}.csv")
    if not candidate.exists():
        raise FileNotFoundError(
            f"Reference costs file not found for battery BOS derivation: {candidate}"
        )
    return str(candidate)


def load_battery_storage_investment_cost(costs_file):
    """
    Load battery storage energy investment cost as EUR/kWh from a costs CSV.
    """
    df = pd.read_csv(costs_file)
    rows = df[
        df["technology"].astype(str).str.strip().eq("battery storage")
        & df["parameter"].astype(str).str.strip().eq("investment")
    ]
    if rows.empty:
        raise ValueError(
            f"No battery storage investment row found in costs file: {costs_file}"
        )
    value = float(rows.iloc[0]["value"])
    unit = str(rows.iloc[0]["unit"]).strip().lower()
    if unit == "eur/kwh":
        return value
    if unit == "eur/mwh":
        return value / 1000.0
    raise ValueError(
        f"Unsupported battery storage investment unit '{rows.iloc[0]['unit']}' in {costs_file}. "
        "Expected EUR/kWh or EUR/MWh."
    )


def get_battery_energy_bos_multiplier(learning_cfg, costs_file):
    """
    Return data-derived energy-side multiplier applied to battery_energy costs.

    This multiplier is intended for energy-side adders only and should exclude
    battery_power components (inverter/power electronics), which are modeled
    separately on links.
    """
    cache = learning_cfg.setdefault("_battery_energy_bos_cache", {})
    cache_key = f"{Path(costs_file).resolve()}::{BATTERY_ENERGY_BOS_REFERENCE_YEAR}"
    if cache_key in cache:
        return float(cache[cache_key])

    reference_costs_file = _resolve_reference_costs_file(
        costs_file,
        BATTERY_ENERGY_BOS_REFERENCE_YEAR,
    )
    energy_side_system_cost = load_battery_storage_investment_cost(reference_costs_file)
    liion_pack_cost = float(
        load_cost_from_historical_csv(
            "battery_energy",
            BATTERY_ENERGY_BOS_REFERENCE_YEAR,
            learning_cfg,
        )
    )
    value = energy_side_system_cost / liion_pack_cost
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"Invalid derived battery energy-side multiplier={value!r}; must be finite and > 0."
        )
    logger.info(
        "Derived battery energy-side multiplier %.4f from %s: "
        "battery storage investment %.3f EUR/kWh / Li-ion pack %.3f EUR/kWh (year %s)",
        value,
        reference_costs_file,
        energy_side_system_cost,
        liion_pack_cost,
        BATTERY_ENERGY_BOS_REFERENCE_YEAR,
    )
    cache[cache_key] = float(value)
    return value


def normalize_optional_input(value):
    """Normalize optional Snakemake inputs where missing values may appear as empty lists."""
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value or None


def extract_year_from_network_path(network_path):
    """Extract the planning-horizon year token from a myopic network path."""
    if not network_path:
        return None

    year_match = re.search(r'_(\d{4})(?=_)', str(network_path))
    if not year_match:
        return None
    return int(year_match.group(1))


def get_cost_lag_year(planning_horizons, current_year):
    """Return the deployment year used to price the current horizon."""
    if current_year in COST_HISTORICAL_CAPACITY_YEARS:
        return COST_HISTORICAL_CAPACITY_YEARS[current_year]

    prev_horizon = get_previous_horizon(planning_horizons, current_year)
    if prev_horizon is None:
        raise ValueError(f"No lag year available for horizon {current_year}")
    return prev_horizon

def validate_learning_parameters(tech, A, beta, learning_rate, L_current, L_max=None):
    """
    Validate learning parameters and capacity values for numerical stability.
    
    Args:
        tech: Technology name
        A: Cost coefficient (must be > 0)
        beta: Learning exponent (must be > 0)
        learning_rate: Learning rate (must be 0 < LR < 1)
        L_current: Current cumulative capacity (must be > 0)
        L_max: Optional maximum capacity (must be > L_current if provided)
    
    Raises:
        ValueError: If any validation fails
    """
    # Validate A > 0
    if A <= 0:
        raise ValueError(
            f"{tech}: Invalid cost coefficient A={A}. Must be > 0."
        )
    
    # Validate beta > 0
    if beta <= 0:
        raise ValueError(
            f"{tech}: Invalid learning exponent beta={beta}. Must be > 0 "
            f"(corresponds to positive learning rate)."
        )
    
    # Validate 0 < LR < 1
    if not (0 < learning_rate < 1):
        raise ValueError(
            f"{tech}: Invalid learning rate LR={learning_rate}. Must be in (0, 1). "
            f"Typical values: 0.10-0.30 (10%-30% cost reduction per doubling)."
        )
    
    # Validate L_current > 0
    if L_current <= 0:
        raise ValueError(
            f"{tech}: Invalid current capacity L={L_current}. Must be > 0. "
            f"Check historical data or previous network extraction."
        )
    
    # Validate L_max > L_current if provided
    if L_max is not None and L_max <= L_current:
        raise ValueError(
            f"{tech}: Invalid maximum capacity L_max={L_max}. Must be > L_current={L_current}."
        )
    
    logger.debug(
        f"  {tech}: Numerical validation passed "
        f"(A={A:.2e}, beta={beta:.4f}, LR={learning_rate:.2%}, L={L_current:.1f})"
    )

def load_country_waccs(wacc_file):
    """
    Load country-specific WACCs from CSV file.
    
    Args:
        wacc_file: Path to WACC CSV with columns: country, region, onwind, offwind, solar
    
    Returns:
        Dict of {tech: {country: wacc}} where wacc is decimal (e.g., 0.042 for 4.2%)
    """
    df = pd.read_csv(wacc_file)
    
    wacc_dict = {}
    for tech in ['onwind', 'offwind', 'solar']:
        wacc_dict[tech] = {}
        for _, row in df.iterrows():
            country = row['country']
            wacc_pct = row[tech]
            wacc_dict[tech][country] = wacc_pct / 100.0  # Convert percentage to decimal
    
    return wacc_dict

logger = create_logger(__name__)

# Technology unit definitions for validation
# Power technologies use MW/GW, energy technologies use MWh/GWh
POWER_TECHS = ['solar_power', 'onwind_power', 'battery_power', 'electrolyser_power']
ENERGY_TECHS = ['battery_energy', 'h2_energy']

# Expected units for each technology type
EXPECTED_UNITS = {
    'solar_power': 'kW',
    'onwind_power': 'kW',
    'battery_power': 'kW',
    'battery_energy': 'kWh',
    'electrolyser_power': 'kW',
    'h2_energy': 'kWh',
}

def validate_technology_units(tech, unit):
    """
    Validate that technology unit matches expected type.
    
    Args:
        tech: Technology key (e.g., 'solar_power', 'battery_energy')
        unit: Unit string from learning parameters ('kW' or 'kWh')
    
    Raises:
        ValueError: If unit doesn't match expected type for technology
    """
    expected = EXPECTED_UNITS.get(tech)
    if expected is None:
        logger.warning(f"Unknown technology {tech}, cannot validate units")
        return
    
    if unit != expected:
        raise ValueError(
            f"Unit mismatch for {tech}: got '{unit}', expected '{expected}'. "
            f"Power technologies (generators/links) use 'kW' (MW/GW), "
            f"energy technologies (stores) use 'kWh' (MWh/GWh)."
        )
    
    logger.debug(f"  {tech}: Unit validation passed (unit='{unit}')")

# Mapping from learning tech names to cost file technology names
TECH_TO_COST_NAME = {
    'solar_power': 'solar',
    'onwind_power': 'onwind',
    'battery_power': 'battery inverter',
    'battery_energy': 'battery storage',
    'electrolyser_power': 'electrolysis',
    'h2_energy': 'H2 Store',
}


def parse_learning_rate_wildcard(learning_rate_str):
    """
    Parse learning rate wildcard into beta adjustment configuration.
    
    Format: tech.methodvalue or "base" for no adjustment.
    Examples:
        - "base": No adjustment
        - "s.m1.1": Solar learning rate ×1.1 (10% faster learning)
        - "w.m0.9": Wind learning rate ×0.9 (10% slower learning)
        - "all.m1.5": All technologies learning rate ×1.5 (50% faster learning)
        - "b.m0.8_w.m1.2": Battery LR ×0.8 AND wind LR ×1.2
    
    Tech codes:
        - s: solar_power
        - w: onwind_power (wind)
        - b: battery_energy
        - bi: battery_power (battery inverter)
        - e: electrolyser_power
        - h: h2_energy
        - all: all technologies
    
    Method codes:
        - m: multiplier (multiply learning rate, not β)
        - o: offset (add to learning rate, not β)
    
    Args:
        learning_rate_str: Wildcard string from scenario config
    
    Returns:
        Dict with beta adjustment settings compatible with config.learning.yaml format
    """
    if learning_rate_str == "base" or not learning_rate_str:
        return {
            "enabled": False,
            "method": "multiplier",
            "multiplier": 1.0,
            "offset": 0.0,
            "tech_specific": {},
        }
    
    tech_map = {
        's': 'solar_power',
        'w': 'onwind_power',
        'b': 'battery_energy',
        'bi': 'battery_power',
        'e': 'electrolyser_power',
        'h': 'h2_energy',
    }
    
    # Parse potentially multiple adjustments separated by underscore
    adjustments = learning_rate_str.split('_')
    
    global_method = None
    global_value = None
    tech_specific = {}
    tech_specific_methods = {}  # Track method per technology
    
    for adj in adjustments:
        # Parse format: tech.methodvalue
        # E.g., "s.m1.2" -> tech="s", method="m", value="1.2"
        if '.' not in adj:
            raise ValueError(
                f"Invalid learning rate wildcard '{adj}'. "
                "Use explicit syntax tech.methodvalue, e.g. 's.o0.10' or 'w.m1.10'."
            )
        
        parts = adj.split('.')
        if len(parts) < 2:
            raise ValueError(
                f"Invalid learning rate wildcard '{adj}'. "
                "Use explicit syntax tech.methodvalue, e.g. 's.o0.10' or 'w.m1.10'."
            )
        
        tech_code = parts[0]
        method_value = '.'.join(parts[1:])  # Rejoin in case value has decimal point
        
        # Parse method and value (e.g., "m1.2" or "o0.1")
        if method_value.startswith('m'):
            method = 'multiplier'
            value_str = method_value[1:]
        elif method_value.startswith('o'):
            method = 'offset'
            value_str = method_value[1:]
        else:
            raise ValueError(
                f"Invalid learning rate wildcard '{adj}'. "
                "Use explicit syntax tech.methodvalue, e.g. 's.o0.10' or 'w.m1.10'."
            )
        
        try:
            value = float(value_str)
        except ValueError:
            raise ValueError(f"Invalid numeric value in learning rate wildcard '{adj}': {value_str}")
        
        # Apply to appropriate tech
        if tech_code == 'all':
            global_method = method
            global_value = value
        elif tech_code in tech_map:
            tech_name = tech_map[tech_code]
            tech_specific[tech_name] = value
            tech_specific_methods[tech_name] = method  # Store method per tech
        else:
            raise ValueError(
                f"Unknown tech code '{tech_code}' in learning rate wildcard '{adj}'. "
                "Supported codes: s, w, b, bi, e, h, all."
            )
    
    # Build config dict
    if global_method:
        config = {
            "enabled": True,
            "method": global_method,
            "multiplier": global_value if global_method == "multiplier" else 1.0,
            "offset": global_value if global_method == "offset" else 0.0,
            "tech_specific": tech_specific,
            "tech_specific_methods": tech_specific_methods,
        }
    else:
        # Only tech-specific adjustments, default to multiplier method
        config = {
            "enabled": True,
            "method": "multiplier",
            "multiplier": 1.0,
            "offset": 0.0,
            "tech_specific": tech_specific,
            "tech_specific_methods": tech_specific_methods,
        }
    
    logger.info(f"Parsed learning_rate wildcard '{learning_rate_str}'")
    if global_method:
        logger.info(f"  Global adjustment: {global_method}={global_value}")
    if tech_specific:
        tech_specific_summary = {
            tech: {
                "method": tech_specific_methods.get(tech, config["method"]),
                "value": value,
            }
            for tech, value in tech_specific.items()
        }
        logger.info(f"  Tech-specific adjustments: {tech_specific_summary}")
    if not global_method and not tech_specific:
        raise ValueError(
            f"Learning rate wildcard '{learning_rate_str}' did not define any valid adjustments"
        )
    
    return config


def calculate_annuity(lifetime, discount_rate):
    """
    Calculate annuity factor for capital cost conversion.
    
    Args:
        lifetime: Asset lifetime in years
        discount_rate: Discount rate (WACC)
    
    Returns:
        Annuity factor
    """
    if discount_rate == 0:
        return 1 / lifetime
    return discount_rate / (1 - (1 + discount_rate) ** (-lifetime))


def load_config_learning(config_file):
    """Load learning configuration from config.learning.yaml."""
    logger.info(f"Loading learning configuration from {config_file}")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config["learning"]


def load_learning_manifest(learning_cfg, config_file):
    manifest_path = Path(learning_cfg.get("artifact_manifest", "data/learning-data/manifest.json"))
    if not manifest_path.is_absolute():
        manifest_path = Path(config_file).resolve().parent / manifest_path
    if not manifest_path.exists():
        raise FileNotFoundError(f"Learning artifact manifest not found: {manifest_path}")
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    learning_cfg["_manifest"] = manifest
    learning_cfg["_manifest_root"] = manifest_path.parent
    learning_cfg["_manifest_path"] = str(manifest_path)
    learning_cfg["_manifest_schema_version"] = str(manifest.get("schema_version", "unknown"))
    learning_cfg["_manifest_sha256"] = __import__("hashlib").sha256(
        manifest_text.encode("utf-8")
    ).hexdigest()
    return manifest


def get_fossil_price_cfg(learning_cfg):
    cfg = learning_cfg.get("fossil_price_uncertainty", {}) or {}
    fuels = cfg.get("fuels", list(FOSSIL_FUELS))
    fuels = [str(fuel).strip().lower() for fuel in fuels]
    invalid = sorted(set(fuels) - set(FOSSIL_FUELS))
    if invalid:
        raise ValueError(
            f"Unsupported fossil_price_uncertainty.fuels entries: {invalid}. "
            f"Supported values: {list(FOSSIL_FUELS)}"
        )
    annual_step = int(cfg.get("annual_step", DEFAULT_FOSSIL_PRICE_ANNUAL_STEP))
    if annual_step != 1:
        raise ValueError(
            "Only annual fossil price evolution is currently supported "
            f"(got fossil_price_uncertainty.annual_step={annual_step})."
        )
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "bundle_root": str(cfg.get("bundle_root", "")),
        "fallback_to_static_prices": bool(cfg.get("fallback_to_static_prices", False)),
        "annual_step": annual_step,
        "fuels": fuels,
        "expectation_mode": str(cfg.get("expectation_mode", DEFAULT_FOSSIL_PRICE_EXPECTATION_MODE)),
        "annual_weights": cfg.get("annual_weights", None),
    }


def load_fossil_price_bundle(learning_cfg, config_file):
    cfg = get_fossil_price_cfg(learning_cfg)
    if not cfg["enabled"]:
        learning_cfg["_fossil_price_bundle"] = None
        return None

    manifest, bundle_root = load_fuel_price_bundle_manifest(cfg["bundle_root"], config_file)
    params_path = resolve_manifest_artifact(manifest, bundle_root, "ar1_parameters_csv")
    country_map_path = resolve_manifest_artifact(manifest, bundle_root, "country_market_map_csv")
    historical_path = resolve_manifest_artifact(manifest, bundle_root, "historical_market_prices_csv")

    params_df = pd.read_csv(params_path)
    country_map_df = pd.read_csv(country_map_path)
    historical_df = pd.read_csv(historical_path)
    params_df["fuel_type"] = params_df["fuel_type"].astype(str).str.lower()
    params_df["market"] = params_df.apply(
        lambda row: normalize_market_name(row["fuel_type"], row["market"]),
        axis=1,
    )
    country_map_df["fuel_type"] = country_map_df["fuel_type"].astype(str).str.lower()
    country_map_df["market"] = country_map_df.apply(
        lambda row: normalize_market_name(row["fuel_type"], row["market"]),
        axis=1,
    )
    historical_df["fuel_type"] = historical_df["fuel_type"].astype(str).str.lower()
    historical_df["market"] = historical_df.apply(
        lambda row: normalize_market_name(row["fuel_type"], row["market"]),
        axis=1,
    )
    ensure_unique_mapping_rows(params_df, ["fuel_type", "market"], "fuel AR(1) parameter")
    ensure_unique_mapping_rows(country_map_df, ["fuel_type", "country"], "fuel country-market")
    ensure_unique_mapping_rows(
        historical_df,
        ["fuel_type", "market", "year"],
        "historical market price",
    )
    missing_param_keys = sorted(
        {
            (str(row["fuel_type"]), str(row["market"]))
            for row in country_map_df.to_dict(orient="records")
        }
        - {
            (str(row["fuel_type"]), str(row["market"]))
            for row in params_df.to_dict(orient="records")
        }
    )
    if missing_param_keys:
        raise ValueError(
            "Fuel price bundle country mappings reference markets without AR(1) parameters: "
            f"{missing_param_keys[:10]}"
        )

    params_by_key = {
        (str(row["fuel_type"]), str(row["market"])): {
            key: value for key, value in row.items()
        }
        for row in params_df.to_dict(orient="records")
    }
    historical_by_key = {
        (str(row["fuel_type"]), str(row["market"]), int(row["year"])): float(row["price_eur_mwh"])
        for row in historical_df.to_dict(orient="records")
    }
    country_market_map = {
        (str(row["fuel_type"]), str(row["country"])): str(row["market"])
        for row in country_map_df.to_dict(orient="records")
    }
    bundle = {
        "manifest": manifest,
        "root": bundle_root,
        "params_df": params_df,
        "country_map_df": country_map_df,
        "historical_df": historical_df,
        "params_by_key": params_by_key,
        "historical_by_key": historical_by_key,
        "country_market_map": country_market_map,
        "schema_version": str(manifest.get("schema_version", "unknown")),
        "manifest_path": str((bundle_root / "manifest.json").resolve()),
        "manifest_sha256": str(manifest.get("_sha256", "")),
    }
    learning_cfg["_fossil_price_bundle"] = bundle
    return bundle


def get_learning_engine(learning_cfg, learning_model=None):
    if learning_model:
        model = str(learning_model)
        if model in SUPPORTED_EXOGENOUS_MODELS:
            return "exogenous_path"
        if model != "legacy_curve":
            return "stochastic_forecast"
    return str(learning_cfg.get("engine", "legacy_curve"))


def get_selected_learning_model(learning_cfg, learning_model=None):
    if learning_model:
        return str(learning_model)
    return str(learning_cfg.get("selected_model", "legacy_curve"))


def get_runtime_conditioning_type(learning_engine, selected_model):
    if learning_engine == "legacy_curve":
        return "deployment_conditioned"
    if learning_engine == "exogenous_path":
        return "time_conditioned"
    if selected_model == "correlated_geometric_random_walk":
        return "time_conditioned"
    return "deployment_conditioned"


def get_requested_cost_expectation_mode(learning_cfg):
    cfg = learning_cfg.get("cost_expectations", {}) or {}
    mode = str(cfg.get("mode", DEFAULT_COST_EXPECTATION_MODE))
    if mode not in SUPPORTED_COST_EXPECTATION_MODES:
        raise ValueError(
            f"Unsupported learning.cost_expectations.mode='{mode}'. "
            f"Supported values: {sorted(SUPPORTED_COST_EXPECTATION_MODES)}"
        )
    return mode


def resolve_cost_expectation_mode(learning_cfg, selected_model):
    requested_mode = get_requested_cost_expectation_mode(learning_cfg)
    if str(selected_model) == "legacy_curve" and requested_mode == "block_average_expected":
        logger.warning(
            "learning.cost_expectations.mode=block_average_expected is not supported for legacy_curve; "
            "falling back to point_cost_legacy."
        )
        return "point_cost_legacy"
    return requested_mode


def get_cost_expectation_weights(learning_cfg):
    cfg = learning_cfg.get("cost_expectations", {}) or {}
    raw_weights = cfg.get("annual_weights", None)
    if raw_weights is None:
        weights = np.asarray(DEFAULT_BLOCK_EXPECTATION_ANNUAL_WEIGHTS, dtype=float)
    else:
        if not isinstance(raw_weights, (list, tuple)):
            raise ValueError(
                "learning.cost_expectations.annual_weights must be a list of five non-negative numbers."
            )
        if len(raw_weights) != len(DEFAULT_BLOCK_EXPECTATION_ANNUAL_WEIGHTS):
            raise ValueError(
                "learning.cost_expectations.annual_weights must have length 5 "
                f"(got {len(raw_weights)})."
            )
        try:
            weights = np.asarray([float(weight) for weight in raw_weights], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "learning.cost_expectations.annual_weights must contain only numeric values."
            ) from exc

    if np.any(~np.isfinite(weights)):
        raise ValueError(
            "learning.cost_expectations.annual_weights must contain only finite values."
        )
    if np.any(weights < 0.0):
        raise ValueError(
            "learning.cost_expectations.annual_weights must be non-negative."
        )

    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError(
            "learning.cost_expectations.annual_weights must sum to a positive value."
        )
    return weights / total


def get_cost_expectation_weights_json(learning_cfg):
    weights = get_cost_expectation_weights(learning_cfg)
    return json.dumps([float(weight) for weight in weights.tolist()])


def get_requested_fossil_price_expectation_mode(learning_cfg):
    cfg = get_fossil_price_cfg(learning_cfg)
    mode = str(cfg.get("expectation_mode", DEFAULT_FOSSIL_PRICE_EXPECTATION_MODE))
    if mode not in SUPPORTED_FOSSIL_PRICE_EXPECTATION_MODES:
        raise ValueError(
            f"Unsupported learning.fossil_price_uncertainty.expectation_mode='{mode}'. "
            f"Supported values: {sorted(SUPPORTED_FOSSIL_PRICE_EXPECTATION_MODES)}"
        )
    return mode


def get_fossil_price_expectation_weights(learning_cfg):
    cfg = get_fossil_price_cfg(learning_cfg)
    raw_weights = cfg.get("annual_weights", None)
    if raw_weights is None:
        weights = np.asarray(DEFAULT_BLOCK_EXPECTATION_ANNUAL_WEIGHTS, dtype=float)
    else:
        if not isinstance(raw_weights, (list, tuple)):
            raise ValueError(
                "learning.fossil_price_uncertainty.annual_weights must be a list of five non-negative numbers."
            )
        if len(raw_weights) != len(DEFAULT_BLOCK_EXPECTATION_ANNUAL_WEIGHTS):
            raise ValueError(
                "learning.fossil_price_uncertainty.annual_weights must have length 5 "
                f"(got {len(raw_weights)})."
            )
        try:
            weights = np.asarray([float(weight) for weight in raw_weights], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "learning.fossil_price_uncertainty.annual_weights must contain only numeric values."
            ) from exc

    if np.any(~np.isfinite(weights)):
        raise ValueError(
            "learning.fossil_price_uncertainty.annual_weights must contain only finite values."
        )
    if np.any(weights < 0.0):
        raise ValueError(
            "learning.fossil_price_uncertainty.annual_weights must be non-negative."
        )

    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError(
            "learning.fossil_price_uncertainty.annual_weights must sum to a positive value."
        )
    return weights / total


def get_fossil_price_expectation_weights_json(learning_cfg):
    weights = get_fossil_price_expectation_weights(learning_cfg)
    return json.dumps([float(weight) for weight in weights.tolist()])


def build_runtime_metadata(learning_cfg, learning_engine, selected_model, cost_expectation_mode=None):
    if cost_expectation_mode is None:
        cost_expectation_mode = resolve_cost_expectation_mode(learning_cfg, selected_model)
    fossil_cfg = get_fossil_price_cfg(learning_cfg)
    fossil_bundle = learning_cfg.get("_fossil_price_bundle") or {}
    return {
        "engine": str(learning_engine),
        "selected_model": str(selected_model),
        "model_name": str(selected_model),
        "training_window": str(learning_cfg.get("training_window", "")),
        "training_window_origin_year": int(learning_cfg.get("training_window_origin_year", 2020)),
        "sample_mode": str(learning_cfg.get("sample_mode", "single_draw")),
        "seed": int(learning_cfg.get("seed", 0)),
        "learning_seed": str(learning_cfg.get("_learning_seed_label", "")),
        "runtime_conditioning": get_runtime_conditioning_type(learning_engine, selected_model),
        "manifest_path": str(learning_cfg.get("_manifest_path", "")),
        "manifest_schema_version": str(learning_cfg.get("_manifest_schema_version", "unknown")),
        "manifest_sha256": str(learning_cfg.get("_manifest_sha256", "")),
        "battery_power_treatment": BATTERY_POWER_TREATMENT,
        "cost_expectation_mode": str(cost_expectation_mode),
        "cost_expectation_weights_json": get_cost_expectation_weights_json(learning_cfg),
        "fossil_price_uncertainty_enabled": bool(fossil_cfg["enabled"]),
        "fossil_price_bundle_path": str(fossil_bundle.get("manifest_path", "")),
        "fossil_price_bundle_schema_version": str(fossil_bundle.get("schema_version", "")),
        "fossil_price_bundle_sha256": str(fossil_bundle.get("manifest_sha256", "")),
        "fossil_price_expectation_mode": get_requested_fossil_price_expectation_mode(learning_cfg),
        "fossil_price_expectation_weights_json": get_fossil_price_expectation_weights_json(learning_cfg),
    }


def _implied_lr_from_learning_exponent(beta):
    """Convert a Wright-style learning exponent to a learning rate."""
    if beta is None:
        return None
    return float(1.0 - (2.0 ** (-float(beta))))


def _implied_lr_from_experience_slope(beta_slope):
    """Convert a dlog(cost) / dlog(experience) slope to a learning rate."""
    if beta_slope is None:
        return None
    return float(1.0 - (2.0 ** (float(beta_slope))))


def _json_scalar(value):
    """Normalize runtime diagnostics into JSON/CSV-safe scalar values."""
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, str):
        return value
    return value


def _serialize_stochastic_snapshot(snapshot):
    """Flatten model diagnostics for CSV export while keeping a JSON snapshot."""
    if not snapshot:
        return {}

    sanitized = {key: _json_scalar(value) for key, value in snapshot.items()}
    fields = {
        "stochastic_parameter_snapshot": json.dumps(sanitized, sort_keys=True),
    }
    for key, value in sanitized.items():
        fields[f"stochastic_{key}"] = value
    return fields


def _shared_runtime_snapshot(pars, current_regime, regime_probs, p_ss, p_ff, sample_mode):
    alpha_slow = _single_or_median(pars["alpha_slow"], sample_mode)
    alpha_fast = _single_or_median(pars["alpha_fast"], sample_mode)
    beta_slow_slope = _single_or_median(pars["beta_slow"], sample_mode)
    beta_fast_slope = _single_or_median(pars["beta_fast"], sample_mode)
    sigma = _single_or_median(pars["sigma"], sample_mode)
    p_slow_slow = _single_or_median(p_ss, sample_mode)
    p_fast_fast = _single_or_median(p_ff, sample_mode)
    regime = int(round(_single_or_median(current_regime, sample_mode)))
    active_slope = beta_slow_slope if regime == 0 else beta_fast_slope

    return {
        "family": "shared_state_bayesian_regime_wright",
        "current_regime": regime,
        "regime_prob_slow": float(regime_probs[0]),
        "regime_prob_fast": float(regime_probs[1]),
        "p_slow_slow": p_slow_slow,
        "p_fast_fast": p_fast_fast,
        "alpha_slow": alpha_slow,
        "alpha_fast": alpha_fast,
        "beta_slow_experience": beta_slow_slope,
        "beta_fast_experience": beta_fast_slope,
        "implied_learning_exponent_slow": float(-beta_slow_slope),
        "implied_learning_exponent_fast": float(-beta_fast_slope),
        "implied_learning_exponent_runtime": float(-active_slope),
        "implied_learning_rate_slow": _implied_lr_from_experience_slope(beta_slow_slope),
        "implied_learning_rate_fast": _implied_lr_from_experience_slope(beta_fast_slope),
        "implied_learning_rate_runtime": _implied_lr_from_experience_slope(active_slope),
        "sigma": sigma,
    }


def _way_runtime_snapshot(artifact):
    params = artifact["parameter_summary"]
    learning_exponent = params.get("learning_exponent")
    return {
        "family": "way_fixed_rho_benchmark_035",
        "alpha": float(params["alpha"]),
        "slope_dlog_experience": float(params["slope_dlog_experience"]),
        "learning_exponent": None if learning_exponent is None else float(learning_exponent),
        "implied_learning_rate": _implied_lr_from_learning_exponent(learning_exponent),
        "theta_ma1": float(params["theta_ma1"]),
        "sigma": float(params["sigma"]),
    }


def _cgrw_runtime_snapshot(artifact):
    params = artifact["parameter_summary"]
    return {
        "family": "correlated_geometric_random_walk",
        "alpha": float(params["alpha"]),
        "rho": float(params["rho"]),
        "sigma": float(params["sigma"]),
        "implied_learning_rate": None,
    }


def _generic_stochastic_fields_from_snapshot(snapshot):
    """Map model-specific diagnostics onto generic beta/lr columns when possible."""
    if not snapshot:
        return {
            "A": None,
            "A_base": None,
            "A_scenario": None,
            "beta": None,
            "beta_base": None,
            "lr_base": None,
            "lr_scenario": None,
            "beta_adjusted": False,
        }

    if "implied_learning_exponent_runtime" in snapshot:
        beta = snapshot.get("implied_learning_exponent_runtime")
        lr = snapshot.get("implied_learning_rate_runtime")
    else:
        beta = snapshot.get("learning_exponent")
        lr = snapshot.get("implied_learning_rate")

    return {
        "A": None,
        "A_base": None,
        "A_scenario": None,
        "beta": beta,
        "beta_base": None,
        "lr_base": None,
        "lr_scenario": lr,
        "beta_adjusted": False,
    }


def validate_runtime_contract(learning_cfg, learning_engine, selected_model):
    if learning_engine not in SUPPORTED_LEARNING_ENGINES:
        raise ValueError(
            f"Unsupported learning.engine='{learning_engine}'. "
            f"Supported values: {sorted(SUPPORTED_LEARNING_ENGINES)}"
        )

    get_requested_cost_expectation_mode(learning_cfg)
    get_cost_expectation_weights(learning_cfg)
    get_fossil_price_cfg(learning_cfg)
    get_requested_fossil_price_expectation_mode(learning_cfg)
    get_fossil_price_expectation_weights(learning_cfg)

    sample_mode = str(learning_cfg.get("sample_mode", "single_draw"))
    if sample_mode not in SUPPORTED_SAMPLE_MODES:
        raise ValueError(
            f"Unsupported learning.sample_mode='{sample_mode}'. "
            f"Supported values: {sorted(SUPPORTED_SAMPLE_MODES)}"
        )

    training_window = str(learning_cfg.get("training_window", "origin_cutoff"))
    if training_window not in SUPPORTED_TRAINING_WINDOWS:
        raise ValueError(
            f"Unsupported learning.training_window='{training_window}'. "
            f"Supported values: {sorted(SUPPORTED_TRAINING_WINDOWS)}"
        )

    seed = learning_cfg.get("seed", 0)
    if not isinstance(seed, (int, np.integer)):
        raise ValueError(
            f"learning.seed must be an integer, got {seed!r} ({type(seed).__name__})"
        )

    if learning_engine == "legacy_curve":
        if selected_model != "legacy_curve":
            raise ValueError(
                "learning.engine='legacy_curve' requires "
                "learning.selected_model='legacy_curve'. "
                f"Got selected_model='{selected_model}'."
            )
    elif learning_engine == "exogenous_path":
        if selected_model not in SUPPORTED_EXOGENOUS_MODELS:
            raise ValueError(
                f"Unsupported exogenous learning model '{selected_model}'. "
                f"Supported values: {sorted(SUPPORTED_EXOGENOUS_MODELS)}"
            )
    else:
        if selected_model not in SUPPORTED_STOCHASTIC_MODELS:
            raise ValueError(
                f"Unsupported stochastic learning model '{selected_model}'. "
                f"Supported values: {sorted(SUPPORTED_STOCHASTIC_MODELS)}"
            )


def resolve_runtime_seed(learning_cfg, learning_engine, selected_model, learning_seed=None):
    default_seed = int(learning_cfg.get("seed", 0))

    if learning_engine in {"legacy_curve", "exogenous_path"}:
        token = LEGACY_LEARNING_SEED if learning_seed in (None, "") else str(learning_seed)
        if token != LEGACY_LEARNING_SEED:
            raise ValueError(
                f"{learning_engine} requires learning_seed='deterministic'. "
                f"Got learning_seed={token!r}."
            )
        learning_cfg["seed"] = 0
        learning_cfg["_learning_seed_label"] = LEGACY_LEARNING_SEED
        return

    token = str(default_seed) if learning_seed in (None, "") else str(learning_seed)
    if token == LEGACY_LEARNING_SEED:
        raise ValueError(
            "stochastic_forecast requires a numeric learning_seed token, "
            f"got {token!r} for model {selected_model}."
        )

    if re.fullmatch(r"s\d+", token):
        seed = int(token[1:])
    elif re.fullmatch(r"\d+", token):
        seed = int(token)
    else:
        raise ValueError(
            "Invalid learning_seed token. Expected 'deterministic' for legacy_curve "
            f"or 'sNNNN' / integer token for stochastic_forecast, got {token!r}."
        )

    learning_cfg["seed"] = seed
    learning_cfg["_learning_seed_label"] = f"s{seed:04d}"


def get_tech_mapping(learning_cfg):
    mapping = dict(DEFAULT_TECH_MAPPING)
    mapping.update(learning_cfg.get("tech_mapping", {}) or {})
    return mapping


def get_learning_finance(learning_cfg):
    finance = {
        "wacc": dict(DEFAULT_FINANCE["wacc"]),
        "lifetime": dict(DEFAULT_FINANCE["lifetime"]),
    }
    cfg_finance = learning_cfg.get("finance", {}) or {}
    finance["wacc"].update(cfg_finance.get("wacc", {}) or {})
    finance["lifetime"].update(cfg_finance.get("lifetime", {}) or {})
    return finance


def get_manifest_historical_datafile(learning_cfg, tech_key):
    manifest = learning_cfg.get("_manifest", {}) or {}
    root = Path(learning_cfg.get("_manifest_root", "."))
    relative = ((manifest.get("historical_files", {}) or {}).get(tech_key))
    if not relative:
        return None
    return str((root / relative).resolve())


def get_manifest_exogenous_cost_path_info(learning_cfg, selected_model):
    cfg = learning_cfg.get("exogenous_cost_path", {}) or {}
    manifest = learning_cfg.get("_manifest", {}) or {}
    root = Path(learning_cfg.get("_manifest_root", "."))
    model_info = (((manifest.get("exogenous_cost_paths", {}) or {}).get("models", {}) or {}).get(selected_model))
    if model_info is None and not cfg.get("artifact_csv"):
        raise ValueError(
            f"Selected exogenous model '{selected_model}' not found in learning manifest and no override artifact_csv was provided."
        )

    artifact_relative = str(cfg.get("artifact_csv") or (model_info or {}).get("artifact_csv", "")).strip()
    if not artifact_relative:
        raise ValueError(f"No exogenous artifact_csv configured for {selected_model}")
    artifact_path = Path(artifact_relative)
    if not artifact_path.is_absolute():
        artifact_path = root / artifact_path
    if not artifact_path.exists():
        raise FileNotFoundError(f"Exogenous cost-path artifact not found: {artifact_path}")

    interpolation = str(cfg.get("interpolation") or (model_info or {}).get("interpolation", "log_linear"))
    if interpolation != "log_linear":
        raise ValueError(
            f"Unsupported exogenous interpolation '{interpolation}'. Supported values: ['log_linear']"
        )

    return {
        "artifact_csv": artifact_path,
        "source": str(cfg.get("source") or (model_info or {}).get("source") or selected_model),
        "interpolation": interpolation,
        "start_year": int((model_info or {}).get("start_year", 2030)),
        "end_year": int((model_info or {}).get("end_year", 2050)),
        "weighting_method": str((model_info or {}).get("weighting_method", "")),
    }


def load_exogenous_cost_path(learning_cfg, selected_model):
    info = get_manifest_exogenous_cost_path_info(learning_cfg, selected_model)
    df = pd.read_csv(info["artifact_csv"])
    for col in ("year", "c_overnight"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["technology"].isin(("solar_power", "onwind_power", "battery_energy"))].copy()
    if df.empty:
        raise ValueError(f"Exogenous cost-path artifact {info['artifact_csv']} is empty")
    return df.sort_values(["technology", "year"]).reset_index(drop=True), info


def _lookup_exogenous_point_cost(exogenous_df, technology, year):
    row = exogenous_df[(exogenous_df["technology"] == technology) & (exogenous_df["year"] == int(year))]
    if row.empty:
        raise ValueError(f"Exogenous cost path is missing {technology} at year {year}")
    return float(row.iloc[0]["c_overnight"])


def _lookup_exogenous_future_costs(exogenous_df, technology, start_year, n_years):
    tech_df = exogenous_df[exogenous_df["technology"] == technology].copy().sort_values("year")
    if tech_df.empty:
        raise ValueError(f"Exogenous cost path is missing technology {technology}")

    last_year = int(tech_df["year"].max())
    last_cost = float(tech_df.loc[tech_df["year"] == last_year, "c_overnight"].iloc[0])
    lookup = tech_df.set_index("year")["c_overnight"].to_dict()
    values = []
    for year in range(int(start_year), int(start_year) + int(n_years)):
        values.append(float(lookup.get(year, last_cost if year > last_year else np.nan)))
    arr = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(arr)):
        raise ValueError(
            f"Exogenous cost path for {technology} is missing years in {start_year}-{start_year + n_years - 1}"
        )
    return arr


def calculate_exogenous_learning_costs(
    learning_cfg,
    selected_model,
    current_year,
    costs_file,
    wacc_dict=None,
    runtime_metadata=None,
):
    exogenous_df, info = load_exogenous_cost_path(learning_cfg, selected_model)
    cost_expectation_mode = str((runtime_metadata or {}).get("cost_expectation_mode", DEFAULT_COST_EXPECTATION_MODE))
    weights = get_cost_expectation_weights(learning_cfg)
    horizon_years = len(weights)

    learning_costs = {}
    for tech in ("solar_power", "onwind_power", "battery_energy"):
        point_cost = _lookup_exogenous_point_cost(exogenous_df, tech, current_year)
        if cost_expectation_mode == "block_average_expected":
            future_costs = _lookup_exogenous_future_costs(exogenous_df, tech, int(current_year) + 1, horizon_years)
            applied_c_overnight = float(np.dot(future_costs, weights))
            terminal_c_overnight = float(future_costs[-1])
        else:
            applied_c_overnight = float(point_cost)
            terminal_c_overnight = float(point_cost)

        record = _runtime_cost_record(
            tech=tech,
            cumulative_capacity=0.0,
            runtime_metadata=runtime_metadata or {},
            snapshot={
                "family": "iea_weo_exogenous_path",
                "source": info["source"],
                "interpolation": info["interpolation"],
            },
            learning_cfg=learning_cfg,
            costs_file=costs_file,
            applied_c_overnight=applied_c_overnight,
            terminal_c_overnight=terminal_c_overnight,
            wacc_dict=wacc_dict,
        )
        record["cumulative_capacity_GW"] = 0.0
        learning_costs[tech] = record
    return learning_costs, info


def get_battery_mapping_config(learning_cfg):
    manifest = learning_cfg.get("_manifest", {}) or {}
    root = Path(learning_cfg.get("_manifest_root", "."))
    battery_treatment = manifest.get("battery_treatment", {}) or {}
    mapping = battery_treatment.get("global_liion_mapping", {}) or {}
    if not mapping:
        raise ValueError(
            "Manifest is missing battery_treatment.global_liion_mapping for battery learning."
        )

    relative = mapping.get("series_csv")
    if not relative:
        raise ValueError("Battery global Li-ion mapping is missing series_csv in manifest.")

    config = dict(mapping)
    config["series_csv"] = str((root / relative).resolve())
    config["series_column"] = str(mapping.get("series_column", "rolling_median_5yr_phi"))
    config["block_aggregation"] = str(mapping.get("block_aggregation", "arithmetic_mean"))
    config["fallback_value"] = float(mapping.get("fallback_value"))
    return config


def load_battery_phi_support(learning_cfg):
    cached = learning_cfg.get("_battery_phi_support", None)
    if cached is not None:
        return cached

    mapping = get_battery_mapping_config(learning_cfg)
    series_path = Path(mapping["series_csv"])
    if not series_path.exists():
        raise FileNotFoundError(f"Battery phi mapping series not found: {series_path}")

    df = pd.read_csv(series_path)
    series_col = mapping["series_column"]
    if series_col not in df.columns:
        raise ValueError(
            f"Battery phi mapping series is missing required column '{series_col}': {series_path}"
        )

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df[series_col] = pd.to_numeric(df[series_col], errors="coerce")
    df = df.sort_values("year").reset_index(drop=True)

    support = {"config": mapping, "series": df}
    learning_cfg["_battery_phi_support"] = support
    return support


def get_battery_phi_for_year(learning_cfg, year):
    support = load_battery_phi_support(learning_cfg)
    mapping = support["config"]
    series = support["series"]
    series_col = mapping["series_column"]

    row = series.loc[series["year"] == int(year)]
    if not row.empty:
        value = float(row[series_col].iloc[-1])
        if np.isfinite(value) and value > 0.0:
            return value

    fallback = float(mapping["fallback_value"])
    if not np.isfinite(fallback) or fallback <= 0.0:
        raise ValueError(
            f"Invalid battery phi fallback value {fallback!r}; must be finite and > 0."
        )
    return fallback


def get_battery_phi_for_block(learning_cfg, start_year, end_year):
    years = list(range(int(start_year) + 1, int(end_year) + 1))
    if not years:
        raise ValueError(
            f"Invalid battery phi block [{start_year}, {end_year}]: no annual support years."
        )

    values = [float(get_battery_phi_for_year(learning_cfg, year)) for year in years]
    aggregation = load_battery_phi_support(learning_cfg)["config"]["block_aggregation"]
    if aggregation != "arithmetic_mean":
        raise ValueError(
            f"Unsupported battery phi block aggregation '{aggregation}'."
        )
    phi_block = float(np.mean(values))
    if not np.isfinite(phi_block) or phi_block <= 0.0:
        raise ValueError(
            f"Invalid battery phi block value for [{start_year}, {end_year}]: {phi_block}"
        )
    return phi_block


def get_committed_capacity_history_value(state, tech, year, history_field="capacity_history"):
    history = (state.get(history_field, {}) or {}).get(tech, {}) or {}
    key = str(int(year))
    if key not in history:
        raise ValueError(
            f"Committed {history_field} for {tech} is missing year {year}."
        )
    value = float(history[key])
    if value <= 0.0:
        raise ValueError(
            f"Committed {history_field} for {tech} at {year} must be > 0, got {value}."
        )
    return value


def get_learning_base_capacity_for_year(state, tech, year, fallback_capacity=None):
    if tech == "battery_energy":
        return get_committed_capacity_history_value(state, tech, year, "capacity_history")
    if fallback_capacity is None:
        raise ValueError(
            f"Missing fallback capacity for {tech} at {year} in learning-base lookup."
        )
    value = float(fallback_capacity)
    if value <= 0.0:
        raise ValueError(f"Learning-base capacity for {tech} at {year} must be > 0, got {value}.")
    return value


def build_learning_base_capacity_map(state, solved_capacity_by_tech, current_year):
    learning_base = {}
    for tech, solved_capacity in solved_capacity_by_tech.items():
        if tech == "battery_energy":
            learning_base[tech] = get_learning_base_capacity_for_year(
                state, tech, current_year
            )
        else:
            learning_base[tech] = float(solved_capacity)
    return learning_base


def get_training_window_config(learning_cfg):
    training_window = str(learning_cfg.get("training_window", "origin_cutoff"))
    origin_year = int(learning_cfg.get("training_window_origin_year", 2020))
    return training_window, origin_year


def get_legacy_params_path(learning_cfg):
    manifest = learning_cfg.get("_manifest", {}) or {}
    root = Path(learning_cfg.get("_manifest_root", "."))
    legacy = manifest.get("legacy_curve", {}) or {}
    training_window, origin_year = get_training_window_config(learning_cfg)
    training_windows = legacy.get("training_windows", {}) or {}
    if training_window not in training_windows:
        raise ValueError(
            f"Manifest is missing legacy_curve parameters for training window '{training_window}'."
        )
    mode_info = training_windows[training_window]
    manifest_origin_year = mode_info.get("origin_year", None)
    if training_window == "origin_cutoff" and manifest_origin_year is not None and int(manifest_origin_year) != int(origin_year):
        raise ValueError(
            "learning.training_window_origin_year does not match the bundled legacy_curve origin_cutoff parameters. "
            f"Config requested {origin_year}, manifest contains {manifest_origin_year}."
        )
    relative = mode_info.get("params_csv")
    if not relative:
        raise ValueError(f"Manifest is missing legacy_curve.params_csv for training window '{training_window}'")
    return str((root / relative).resolve())


def load_fom_from_costs(costs_file, tech_key):
    """
    Load Fixed O&M (FOM) percentage from cost CSV file.
    
    Args:
        costs_file: Path to costs CSV (e.g., costs_2025.csv)
        tech_key: Technology key (e.g., 'solar_power')
    
    Returns:
        FOM as decimal (e.g., 0.01578 for 1.578%/year)
    
    Raises:
        FileNotFoundError: If cost file doesn't exist
        ValueError: If FOM data not found for technology
    """
    if not Path(costs_file).exists():
        raise FileNotFoundError(f"Cost file not found: {costs_file}")
    
    # Map to cost file technology name
    cost_tech_name = TECH_TO_COST_NAME.get(tech_key)
    if cost_tech_name is None:
        raise ValueError(f"No cost file mapping for {tech_key} in TECH_TO_COST_NAME")
    
    try:
        df = pd.read_csv(costs_file)
        
        # Filter for this technology and FOM parameter
        fom_data = df[(df['technology'] == cost_tech_name) & (df['parameter'] == 'FOM')]
        
        if fom_data.empty:
            # Default to FOM=0 when not found (matching PyPSA-Earth's fill_values behavior)
            # This applies to technologies like "battery storage" which have no FOM in cost files
            if cost_tech_name == "battery storage":
                logger.debug(f"  No FOM data found for {cost_tech_name}, using default FOM=0. "
                             f"TODO: ADD BATTERY INVERTER LEARNING!")
                return 0.0
            else:
                raise ValueError(
                    f"No FOM data found for technology '{cost_tech_name}' in {costs_file}. "
                    f"Check that the cost file contains FOM parameter for this technology."
                )
        
        fom_value = fom_data.iloc[0]['value']
        unit = fom_data.iloc[0]['unit']
        
        # Check unit and convert to decimal
        if '%/year' in unit or '%' in unit:
            fom_decimal = fom_value / 100.0
        else:
            raise ValueError(
                f"Unexpected FOM unit for {cost_tech_name}: {unit}. Expected '%/year' or '%'."
            )
        
        logger.debug(f"  Loaded FOM for {tech_key}: {fom_value}%/year = {fom_decimal:.6f}")
        return fom_decimal
        
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        raise ValueError(f"Error loading FOM for {tech_key} from {costs_file}: {e}")


def recalibrate_A_for_beta_scenario(A_base, beta_base, beta_scenario, L_anchor, tech):
    """
    Recalibrate A parameter to maintain anchor point when β changes.
    
    Ensures that c(L_anchor) remains constant when β is adjusted for scenarios.
    Uses: c_anchor = A_base × L_anchor^(-β_base) = A_scenario × L_anchor^(-β_scenario)
    Solving for A_scenario: A_scenario = c_anchor × L_anchor^(β_scenario)
                                        = A_base × L_anchor^(β_scenario - β_base)
    
    Args:
        A_base: Original A parameter from calibration
        beta_base: Original β parameter from calibration
        beta_scenario: New β for scenario analysis
        L_anchor: Anchor deployment in GW (e.g., 2020 cumulative capacity)
        tech: Technology name (for logging)
    
    Returns:
        A_scenario: Recalibrated A that maintains anchor point
    """
    if beta_base == beta_scenario:
        return A_base
    
    # Calculate A_scenario = A_base × L_anchor^(β_scenario - β_base)
    A_scenario = A_base * (L_anchor ** (beta_scenario - beta_base))
    
    logger.debug(f"    Recalibrating A for {tech}:")
    logger.debug(f"      β_base={beta_base:.4f} → β_scenario={beta_scenario:.4f}")
    logger.debug(f"      A_base={A_base:.2f} → A_scenario={A_scenario:.2f}")
    logger.debug(f"      L_anchor={L_anchor:.1f} GW")
    
    return A_scenario


def apply_beta_adjustment(params, state_anchor, learning_cfg):
    """
    Apply beta adjustments from configuration and recalibrate A parameters.
    
    Now operates on learning rates (LR) directly instead of beta exponents:
    - Multiplier: LR_new = LR_base × multiplier (e.g., 1.1 = 10% faster learning)
    - Offset: LR_new = LR_base + offset (e.g., 0.05 = +5 percentage points)
    - Then converts back to beta: β = -log2(1 - LR)
    
    Args:
        params: DataFrame with base learning parameters (A_over_unit, beta, unit)
        state_anchor: DataFrame with anchor year deployment state (L_realized_GW)
        learning_cfg: Learning configuration dict
    
    Returns:
        DataFrame with adjusted parameters (A_over_unit, beta, beta_base, unit)
    """
    beta_cfg = learning_cfg.get("beta_adjustment", {})
    
    if not beta_cfg.get("enabled", False):
        logger.info("Beta adjustment DISABLED - using base parameters")
        params_adjusted = params.copy()
        params_adjusted["beta_base"] = params_adjusted["beta"]  # Store base beta
        params_adjusted["A_base"] = params_adjusted["A_over_unit"]
        learning_rates = {}
        for tech in params_adjusted.index:
            beta_value = params_adjusted.loc[tech, "beta"]
            learning_rate = 1 - 2**(-beta_value)
            A_value = params_adjusted.loc[tech, "A_over_unit"]
            learning_rates[tech] = {
                "beta_base": float(beta_value),
                "beta_scenario": float(beta_value),
                "lr_base": float(learning_rate),
                "lr_scenario": float(learning_rate),
                "A_base": float(A_value),
                "A_scenario": float(A_value),
            }
        params_adjusted.attrs["learning_rates"] = learning_rates
        return params_adjusted
    
    method = beta_cfg.get("method", "multiplier")
    multiplier = beta_cfg.get("multiplier", 1.0)
    offset = beta_cfg.get("offset", 0.0)
    tech_specific = beta_cfg.get("tech_specific", {})
    tech_specific_methods = beta_cfg.get("tech_specific_methods", {})  # Get per-tech methods
    anchor_year = beta_cfg.get("anchor_year", 2020)

    logger.info(f"Applying learning-rate adjustment anchored to {anchor_year}")
    if tech_specific_methods:
        logger.info(f"  Tech-specific methods: {tech_specific_methods}")
    elif multiplier != 1.0 or offset != 0.0:
        logger.info(
            f"  Global adjustment: method={method}, multiplier={multiplier}, offset={offset}"
        )
    
    params_adjusted = params.copy()
    params_adjusted["beta_base"] = params_adjusted["beta"]  # Store original beta
    params_adjusted["A_base"] = params_adjusted["A_over_unit"]
    
    # Dictionary to store learning rates for metadata
    learning_rates = {}
    adjusted_techs = []
    missing_anchor = []
    
    for tech in params_adjusted.index:
        A_base = params_adjusted.loc[tech, "A_over_unit"]
        beta_base = params_adjusted.loc[tech, "beta"]
        
        # Get anchor deployment
        if tech not in state_anchor.index:
            missing_anchor.append(tech)
            continue
        
        L_anchor = state_anchor.loc[tech, "L_realized_GW"]
        
        # Calculate base learning rate: LR = 1 - 2^(-β)
        lr_base = 1 - 2**(-beta_base)
        
        # Determine learning rate adjustment for this technology
        if tech in tech_specific:
            # Use tech-specific method if available, otherwise fall back to global method
            tech_method = tech_specific_methods.get(tech, method)
            tech_value = tech_specific[tech]
            
            if tech_method == "multiplier":
                lr_scenario = lr_base * tech_value
                logger.info(f"  {tech}: LR × {tech_value:.2f} = {lr_scenario:.1%} (base: {lr_base:.1%})")
            else:  # offset
                lr_scenario = lr_base + tech_value
                logger.info(f"  {tech}: LR + {tech_value:.4f} = {lr_scenario:.1%} (base: {lr_base:.1%})")
        else:
            if method == "multiplier":
                lr_scenario = lr_base * multiplier
            else:  # offset
                lr_scenario = lr_base + offset
            
            if multiplier != 1.0 or offset != 0.0:
                logger.info(f"  {tech}: LR = {lr_scenario:.1%} (base: {lr_base:.1%})")
        
        # Clamp learning rate to valid range [0, 1)
        if lr_scenario >= 1.0:
            logger.warning(f"  {tech}: LR={lr_scenario:.1%} >= 100%, clamping to 99%")
            lr_scenario = 0.99
        elif lr_scenario < 0:
            logger.warning(f"  {tech}: LR={lr_scenario:.1%} < 0%, clamping to 0%")
            lr_scenario = 0.0
        
        # Convert back to beta: β = -log2(1 - LR)
        # Handle edge case where LR = 0 (no learning)
        if lr_scenario == 0:
            beta_scenario = 0.0
        else:
            beta_scenario = -np.log2(1 - lr_scenario)
        
        # Log the beta values for transparency
        if beta_base != beta_scenario:
            adjusted_techs.append(tech)
            logger.info(f"  {tech}: β = {beta_scenario:.4f} (base: {beta_base:.4f})")
        
        # Recalibrate A to maintain anchor point
        A_scenario = recalibrate_A_for_beta_scenario(
            A_base, beta_base, beta_scenario, L_anchor, tech
        )

        # Store learning rates and A values for metadata
        learning_rates[tech] = {
            "beta_base": float(beta_base),
            "beta_scenario": float(beta_scenario),
            "lr_base": float(lr_base),
            "lr_scenario": float(lr_scenario),
            "A_base": float(A_base),
            "A_scenario": float(A_scenario),
        }
        
        # Update parameters
        params_adjusted.loc[tech, "beta"] = beta_scenario
        params_adjusted.loc[tech, "A_over_unit"] = A_scenario
    
    # Store learning rates in the params_adjusted for passing to main
    params_adjusted.attrs["learning_rates"] = learning_rates
    params_adjusted.attrs["beta_adjustment_summary"] = {
        "anchor_year": anchor_year,
        "adjusted_techs": adjusted_techs,
        "missing_anchor": missing_anchor,
    }

    if missing_anchor:
        logger.warning(f"Missing anchor state for technologies: {missing_anchor}")

    if not adjusted_techs:
        raise ValueError(
            "Learning rate adjustment was enabled but no technologies were adjusted. "
            "Check the anchor-state inputs and learning-rate wildcard."
        )
    
    return params_adjusted


def load_learning_params(params_file):
    """
    Load learning parameters (calibrated once from historical data).
    
    Args:
        params_file: Path to learning_params.csv (single file used for all years)
    
    Returns:
        DataFrame with columns: A_over_unit, beta, unit (indexed by technology)
    """
    logger.info(f"Loading learning parameters from {params_file}")
    params = pd.read_csv(params_file, index_col="technology")
    
    required_cols = ["A_over_unit", "beta", "unit"]
    missing = set(required_cols) - set(params.columns)
    if missing:
        raise ValueError(f"Learning params missing columns: {missing}")
    
    logger.info(f"  Loaded params for {len(params)} technologies")
    return params


def build_anchor_state(
    params,
    learning_cfg,
    global_scale_factors,
    anchor_year,
    anchor_network_path=None,
):
    """
    Build anchor deployment state used to recalibrate A when learning rates change.

    Preference order:
    1. Use an explicit solved anchor-year network when available.
    2. Fall back to historical cumulative deployment for the anchor year.

    Returns:
        DataFrame indexed by technology with column L_realized_GW.
    """
    anchor_network_path = normalize_optional_input(anchor_network_path)
    records = {}

    if anchor_network_path:
        network_year = extract_year_from_network_path(anchor_network_path)
        if network_year is not None and network_year != anchor_year:
            logger.warning(
                f"Anchor network year mismatch: expected {anchor_year}, got {network_year}. "
                f"Path: {anchor_network_path}"
            )

        logger.info(f"Building anchor state for {anchor_year} from solved network {anchor_network_path}")
        tech_mapping = get_tech_mapping(learning_cfg)
        realized_capacity = extract_capacity_from_network(
            anchor_network_path, tech_mapping, global_scale_factors
        )

        for tech in params.index:
            if tech not in realized_capacity:
                logger.warning(f"  {tech} not found in anchor network, skipping")
                continue
            records[tech] = {"L_realized_GW": float(realized_capacity[tech])}
    else:
        logger.info(f"Building anchor state for {anchor_year} from historical deployment data")
        for tech in params.index:
            if tech == "battery_power":
                logger.warning(
                    "  battery_power has no dedicated historical power series; "
                    "provide an anchor network to adjust this technology"
                )
                continue

            hist_file = get_manifest_historical_datafile(learning_cfg, tech)
            if hist_file is None:
                logger.warning(f"  No historical data configured for {tech}, skipping")
                continue

            try:
                capacity = load_capacity_from_historical_csv(tech, anchor_year, learning_cfg)
            except Exception as exc:
                logger.warning(
                    f"  Could not load anchor deployment for {tech} in {anchor_year}: {exc}"
                )
                continue

            records[tech] = {"L_realized_GW": float(capacity)}

    anchor_state = pd.DataFrame.from_dict(records, orient="index")
    anchor_state.index.name = "technology"
    anchor_state.attrs["anchor_year"] = int(anchor_year)
    anchor_state.attrs["anchor_source"] = "network" if anchor_network_path else "historical"
    anchor_state.attrs["anchor_network"] = str(anchor_network_path or "")

    if anchor_state.empty:
        raise ValueError(
            f"Could not construct anchor state for {anchor_year}. "
            "Provide an anchor network or ensure historical learning data is available."
        )

    logger.info(f"  Built anchor state for {len(anchor_state)} technologies")
    return anchor_state


def load_capacity_from_historical_csv(tech, year, learning_cfg):
    """
    Load capacity from historical CSV files for a specific year.

    Args:
        tech: Technology key (e.g., 'solar_power', 'onwind_power')
        year: Year to extract from historical data
        learning_cfg: Learning configuration dict
    
    Returns:
        Capacity in GW for the specified year
    """
    hist_file = get_manifest_historical_datafile(learning_cfg, tech)
    if hist_file is None:
        raise ValueError(
            f"No historical data file configured for {tech}. "
            "Check the learning artifact manifest."
        )
    
    logger.info(f"  Loading {year} capacity for {tech} from historical data")
    hist_data = load_historical_capacity(hist_file, tech)
    
    # Find the year in historical data
    year_data = hist_data[hist_data['year'] == year]
    if year_data.empty:
        raise ValueError(
            f"Year {year} not found in historical data for {tech}. "
            f"Available years: {hist_data['year'].min():.0f}-{hist_data['year'].max():.0f}"
        )
    
    capacity = year_data['capacity_GW'].iloc[0]
    unit = "GWh" if tech in ENERGY_TECHS else "GW"
    logger.info(f"    Found {year} capacity: {capacity:.2f} {unit}")
    return capacity


def load_cost_from_historical_csv(tech, year, learning_cfg):
    """
    Load cost-per-unit from bundled historical support CSVs for a specific year.

    Returns:
        Historical overnight cost in EUR/kW or EUR/kWh.
    """
    hist_file = get_manifest_historical_datafile(learning_cfg, tech)
    if hist_file is None:
        raise ValueError(
            f"No historical data file configured for {tech}. "
            "Check the learning artifact manifest."
        )

    logger.info(f"  Loading {year} cost for {tech} from historical data")
    hist_data = load_historical_cost(hist_file, tech)

    year_data = hist_data[hist_data["year"] == year]
    if year_data.empty:
        raise ValueError(
            f"Year {year} not found in historical cost data for {tech}. "
            f"Available years: {hist_data['year'].min():.0f}-{hist_data['year'].max():.0f}"
        )

    cost_per_unit = float(year_data["cost_per_unit"].iloc[0])
    unit = "EUR/kWh" if tech in ENERGY_TECHS else "EUR/kW"
    logger.info(f"    Found {year} cost: {cost_per_unit:.3f} {unit}")
    return cost_per_unit


def load_average_cost_from_historical_csv(tech, start_year, end_year, learning_cfg):
    """
    Load average historical cost-per-unit over an inclusive year window.

    Returns:
        Average historical overnight cost in EUR/kW or EUR/kWh.
    """
    if int(end_year) < int(start_year):
        raise ValueError(
            f"Invalid historical cost window for {tech}: {start_year}-{end_year}"
        )

    hist_file = get_manifest_historical_datafile(learning_cfg, tech)
    if hist_file is None:
        raise ValueError(
            f"No historical data file configured for {tech}. "
            "Check the learning artifact manifest."
        )

    logger.info(
        "  Loading average %s-%s cost for %s from historical data",
        int(start_year),
        int(end_year),
        tech,
    )
    hist_data = load_historical_cost(hist_file, tech).copy()
    hist_data["year"] = pd.to_numeric(hist_data["year"], errors="coerce")
    hist_data = hist_data.dropna(subset=["year", "cost_per_unit"])
    hist_data["year"] = hist_data["year"].astype(int)

    expected_years = set(range(int(start_year), int(end_year) + 1))
    available_years = set(hist_data["year"].tolist())
    missing_years = sorted(expected_years - available_years)
    if missing_years:
        raise ValueError(
            f"Historical cost data for {tech} is missing required years for window "
            f"{start_year}-{end_year}: {missing_years}"
        )

    window = hist_data[hist_data["year"].isin(expected_years)]
    if window.empty:
        raise ValueError(
            f"No historical cost data found for {tech} in window {start_year}-{end_year}."
        )

    average_cost = float(window["cost_per_unit"].mean())
    unit = "EUR/kWh" if tech in ENERGY_TECHS else "EUR/kW"
    logger.info(
        "    Averaged %s-%s cost: %.3f %s (%s years)",
        int(start_year),
        int(end_year),
        average_cost,
        unit,
        len(window),
    )
    return average_cost


def load_stochastic_model_artifacts(learning_cfg, selected_model):
    manifest = learning_cfg.get("_manifest", {}) or {}
    root = Path(learning_cfg.get("_manifest_root", "."))
    stochastic = (manifest.get("stochastic_forecast", {}) or {})
    training_windows = (stochastic.get("training_windows", {}) or {})
    training_window, origin_year = get_training_window_config(learning_cfg)
    if training_window not in training_windows:
        raise ValueError(
            f"Selected training window '{training_window}' not found in learning manifest. "
            f"Available windows: {sorted(training_windows.keys())}"
        )
    window_info = training_windows[training_window]
    if training_window == "origin_cutoff":
        manifest_origin_year = window_info.get("origin_year", None)
        if manifest_origin_year is not None and int(manifest_origin_year) != int(origin_year):
            raise ValueError(
                "learning.training_window_origin_year does not match the bundled origin_cutoff artifacts. "
                f"Config requested {origin_year}, manifest contains {manifest_origin_year}."
            )
    models = (window_info.get("models", {}) or {})
    if selected_model not in models:
        raise ValueError(f"Selected stochastic model '{selected_model}' not found in manifest")
    model_info = models[selected_model]
    artifact_dir = root / model_info["artifact_dir"]
    artifacts = {}
    for tech in model_info["technologies"]:
        path = artifact_dir / f"{tech}.json"
        artifacts[tech] = json.loads(path.read_text(encoding="utf-8"))
    initial_state_path = root / model_info["initial_state"]
    initial_state = json.loads(initial_state_path.read_text(encoding="utf-8"))
    return artifacts, initial_state


def load_runtime_state(prev_state_path, initial_state):
    prev_state_path = normalize_optional_input(prev_state_path)
    if prev_state_path:
        return json.loads(Path(prev_state_path).read_text(encoding="utf-8"))
    return json.loads(json.dumps(initial_state))


def _reinitialize_stochastic_runtime_state_for_model(state, initial_state, selected_model):
    reinitialized = json.loads(json.dumps(initial_state))
    passthrough_keys = [
        "capacity_history",
        "modeled_capacity_history",
        "last_applied_year",
        "committed_from_network",
        "battery_power_treatment",
        "manifest_path",
        "manifest_schema_version",
        "manifest_sha256",
        "training_window",
        "training_window_origin_year",
        "sample_mode",
        "runtime_conditioning",
        "engine",
    ]
    for key in passthrough_keys:
        if key in state:
            reinitialized[key] = state[key]
    reinitialized["selected_model"] = selected_model
    reinitialized["model_name"] = selected_model
    return reinitialized


def _artifact_train_end_year(artifact, fallback_year):
    sample_window = artifact.get("sample_window", {}) or {}
    train_end_year = sample_window.get("train_end_year", None)
    if train_end_year in (None, ""):
        return int(fallback_year)
    return int(train_end_year)


def _backfill_stochastic_runtime_state_fields(state, initial_state, selected_model, artifacts):
    state_model = str(state.get("selected_model") or state.get("model_name") or "").strip()
    if state_model and state_model != selected_model:
        logger.info(
            "Reinitializing stochastic runtime state for %s from shared/bootstrap state produced by %s",
            selected_model,
            state_model,
        )
        return _reinitialize_stochastic_runtime_state_for_model(state, initial_state, selected_model)

    technology_states = (state.get("technology_states", {}) or {})
    initial_technology_states = (initial_state.get("technology_states", {}) or {})
    merged_states = dict(technology_states)
    for tech in artifacts:
        base_state = json.loads(json.dumps(initial_technology_states.get(tech, {})))
        base_state.update(merged_states.get(tech, {}) or {})
        merged_states[tech] = base_state
    state["technology_states"] = merged_states

    if selected_model == "shared_state_bayesian_regime_wright" and "shared_regime_state" not in state:
        if "shared_regime_state" in initial_state:
            state["shared_regime_state"] = json.loads(json.dumps(initial_state["shared_regime_state"]))

    if selected_model and not state.get("selected_model"):
        state["selected_model"] = selected_model
    if selected_model and not state.get("model_name"):
        state["model_name"] = selected_model
    return state


def load_stochastic_runtime_state(learning_cfg, selected_model, current_year, prev_state_path):
    artifacts, initial_state = load_stochastic_model_artifacts(learning_cfg, selected_model)
    if int(current_year) > min(COST_HISTORICAL_CAPACITY_YEARS) and prev_state_path is None:
        raise ValueError(
            f"Stochastic runtime for {current_year} requires previous committed learning state"
        )
    state = load_runtime_state(prev_state_path, initial_state)
    state = _backfill_stochastic_runtime_state_fields(state, initial_state, selected_model, artifacts)
    validate_stochastic_runtime_state(state, selected_model, artifacts)
    return artifacts, state


def build_capacity_history(tech, learning_cfg, state, base_year):
    hist_file = get_manifest_historical_datafile(learning_cfg, tech)
    if hist_file is None:
        raise ValueError(f"No historical data file configured for {tech} in manifest")
    hist = load_historical_capacity(hist_file, tech)
    hist = hist[hist["capacity_GW"] > 0].copy()
    hist["year"] = pd.to_numeric(hist["year"], errors="coerce").astype(int)
    hist = hist[hist["year"] <= int(base_year)].copy()

    realized_map = ((state.get("capacity_history", {}) or {}).get(tech, {})) or {}
    realized_rows = []
    for year_key, value in realized_map.items():
        realized_rows.append({"year": int(year_key), "capacity_GW": float(value)})
    if realized_rows:
        realized = pd.DataFrame(realized_rows)
        hist = pd.concat([hist[["year", "capacity_GW"]], realized], ignore_index=True)
        hist = hist.sort_values("year").drop_duplicates(subset="year", keep="last")
    hist = hist.sort_values("year").reset_index(drop=True)
    return hist


def compute_realized_block_growth(tech, learning_cfg, state, block_end_year, block_years):
    history = build_capacity_history(tech, learning_cfg, state, block_end_year)
    history = history[history["capacity_GW"] > 0].copy()
    if history.empty:
        raise ValueError(f"No positive capacity history available for {tech}")

    end_year = int(block_end_year)
    start_year = int(block_end_year) - int(block_years)

    end_row = history[history["year"] == end_year]
    start_row = history[history["year"] == start_year]
    if end_row.empty or start_row.empty:
        raise ValueError(
            f"Need realized/historical cumulative capacity at both {start_year} and {end_year} for {tech} "
            "to compute block experience growth."
        )

    q_start = float(start_row["capacity_GW"].iloc[-1])
    q_end = float(end_row["capacity_GW"].iloc[-1])
    if q_start <= 0.0 or q_end <= 0.0:
        raise ValueError(
            f"Invalid cumulative capacity history for {tech}: start={q_start}, end={q_end}. "
            "Both values must be > 0."
        )
    if q_end < q_start:
        raise ValueError(
            f"Cumulative capacity declines for {tech} between {start_year} and {end_year}: "
            f"{q_start} -> {q_end}. Refusing to propagate learning on non-monotone cumulative deployment."
        )

    return float(np.log(q_end) - np.log(q_start))


def validate_stochastic_runtime_state(state, selected_model, artifacts):
    technology_states = (state.get("technology_states", {}) or {})
    missing_techs = sorted(set(artifacts.keys()) - set(technology_states.keys()))
    if missing_techs:
        raise ValueError(
            f"Stochastic runtime state for {selected_model} is missing technology states: {missing_techs}"
        )

    for tech in artifacts:
        state_tech = technology_states[tech] or {}
        if "last_log_capex" not in state_tech:
            raise ValueError(
                f"Stochastic runtime state for {selected_model}/{tech} is missing last_log_capex"
            )
        if selected_model == "correlated_geometric_random_walk" and "last_dlog_capex" not in state_tech:
            raise ValueError(
                f"Stochastic runtime state for {selected_model}/{tech} is missing last_dlog_capex"
            )
        if selected_model == "way_fixed_rho_benchmark_035" and "last_innovation" not in state_tech:
            raise ValueError(
                f"Stochastic runtime state for {selected_model}/{tech} is missing last_innovation"
            )

    if selected_model == "shared_state_bayesian_regime_wright":
        shared_state = state.get("shared_regime_state", {}) or {}
        if "initial_regime_probs" not in shared_state:
            raise ValueError("Shared-state runtime state is missing initial_regime_probs")


def _state_to_runtime_learning_costs(
    artifacts,
    state,
    current_year,
    selected_model,
    learning_cfg,
    costs_file,
    global_scale_factors,
    prev_network_path,
    wacc_dict,
    runtime_metadata,
):
    tech_mapping = get_tech_mapping(learning_cfg)
    runtime_capacity_map = _get_runtime_capacity_map(
        current_year,
        prev_network_path,
        learning_cfg,
        state,
        tech_mapping,
        global_scale_factors,
    )
    return _learning_costs_from_stochastic_state(
        artifacts=artifacts,
        state=state,
        selected_model=selected_model,
        cumulative_capacity_map=runtime_capacity_map,
        learning_cfg=learning_cfg,
        costs_file=costs_file,
        wacc_dict=wacc_dict,
        runtime_metadata=runtime_metadata,
    )


def _learning_costs_from_stochastic_state(
    artifacts,
    state,
    selected_model,
    cumulative_capacity_map,
    learning_cfg,
    costs_file,
    wacc_dict,
    runtime_metadata,
):
    learning_costs = {}
    cost_expectation_mode = str(runtime_metadata.get("cost_expectation_mode", DEFAULT_COST_EXPECTATION_MODE))
    expectation_diagnostics = None
    if cost_expectation_mode == "block_average_expected":
        expectation_diagnostics = _compute_block_average_expected_costs(
            artifacts=artifacts,
            state=state,
            selected_model=selected_model,
            learning_cfg=learning_cfg,
            costs_file=costs_file,
        )
    for tech, artifact in artifacts.items():
        if tech not in cumulative_capacity_map:
            raise ValueError(f"Technology {tech} missing in runtime capacity map")
        state_tech = (state.get("technology_states", {}) or {}).get(tech, {}) or {}
        log_cost = float(state_tech["last_log_capex"])
        point_c_overnight = float(np.exp(log_cost) / 1000.0)
        snapshot = state_tech.get("parameter_snapshot", None)
        if not snapshot:
            if selected_model == "way_fixed_rho_benchmark_035":
                snapshot = _way_runtime_snapshot(artifact)
            elif selected_model == "correlated_geometric_random_walk":
                snapshot = _cgrw_runtime_snapshot(artifact)
        if cost_expectation_mode == "block_average_expected":
            diag = expectation_diagnostics[tech]
            applied_c_overnight = float(diag["c_overnight"])
            terminal_c_overnight = float(diag["c_overnight_terminal_point"])
        else:
            applied_c_overnight = point_c_overnight
            terminal_c_overnight = point_c_overnight
        record = _runtime_cost_record(
            tech=tech,
            cumulative_capacity=cumulative_capacity_map[tech],
            runtime_metadata=runtime_metadata,
            snapshot=snapshot,
            learning_cfg=learning_cfg,
            costs_file=costs_file,
            applied_c_overnight=applied_c_overnight,
            terminal_c_overnight=terminal_c_overnight,
            wacc_dict=wacc_dict,
        )
        if cost_expectation_mode != "block_average_expected":
            record["log_capex_runtime"] = log_cost
            record["log_capex_terminal_point"] = log_cost
        learning_costs[tech] = record
    return learning_costs


def _single_or_median(values, sample_mode):
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    if sample_mode == "median":
        return float(np.median(arr))
    return float(arr[0])


def _runtime_sample_count(sample_mode, n_samples=None):
    if n_samples is not None:
        return int(n_samples)
    return 1 if sample_mode == "single_draw" else 201


def _empty_annual_log_cost_paths(n, elapsed_years):
    return np.empty((int(n), int(elapsed_years)), dtype=float)


def _cost_statistics_from_levels(
    applied_c_overnight,
    terminal_c_overnight,
    learning_cfg,
    costs_file,
    tech,
):
    applied_c_overnight = float(applied_c_overnight)
    terminal_c_overnight = float(terminal_c_overnight)
    return {
        "c_overnight": applied_c_overnight,
        "capital_cost": convert_to_capital_cost(
            applied_c_overnight,
            tech,
            _cost_unit_for_runtime(tech),
            learning_cfg,
            costs_file,
        ),
        "c_overnight_terminal_point": terminal_c_overnight,
        "capital_cost_terminal_point": convert_to_capital_cost(
            terminal_c_overnight,
            tech,
            _cost_unit_for_runtime(tech),
            learning_cfg,
            costs_file,
        ),
        "log_capex_terminal_point": float(np.log(max(terminal_c_overnight, 1.0e-12) * 1000.0)),
    }


def _runtime_cost_record(
    tech,
    cumulative_capacity,
    runtime_metadata,
    snapshot,
    learning_cfg,
    costs_file,
    applied_c_overnight,
    terminal_c_overnight,
    wacc_dict,
):
    cost_stats = _cost_statistics_from_levels(
        applied_c_overnight=applied_c_overnight,
        terminal_c_overnight=terminal_c_overnight,
        learning_cfg=learning_cfg,
        costs_file=costs_file,
        tech=tech,
    )
    return {
        "cumulative_capacity_GW": float(cumulative_capacity),
        **_generic_stochastic_fields_from_snapshot(snapshot),
        "unit": _cost_unit_for_runtime(tech),
        "wacc_dict": wacc_dict,
        **runtime_metadata,
        **cost_stats,
        "log_capex_runtime": float(cost_stats["log_capex_terminal_point"]),
        **_serialize_stochastic_snapshot(snapshot),
    }


def _runtime_cost_expectation_rng(seed, state):
    base_year = int(state.get("last_applied_year", 0))
    return np.random.default_rng(int(seed) + 1000 * base_year + 17)


def _compute_block_average_expected_costs(
    artifacts,
    state,
    selected_model,
    learning_cfg,
    costs_file,
):
    weights = get_cost_expectation_weights(learning_cfg)
    expectation_years = int(len(weights))
    seed = int(learning_cfg.get("seed", 0))
    rng = _runtime_cost_expectation_rng(seed, state)

    if selected_model == "shared_state_bayesian_regime_wright":
        annual_result = simulate_shared_state_runtime(
            artifacts=artifacts,
            state=state,
            block_dlog_experience_by_tech={tech: 0.0 for tech in artifacts},
            elapsed_years=expectation_years,
            rng=rng,
            sample_mode="single_draw",
            n_samples=BLOCK_EXPECTATION_DRAWS,
        )
        annual_paths = annual_result["annual_log_cost_paths"]
    elif selected_model == "correlated_geometric_random_walk":
        annual_paths = {}
        for tech, artifact in artifacts.items():
            annual_paths[tech] = simulate_cgrw_runtime(
                artifact=artifact,
                state=state["technology_states"][tech],
                elapsed_years=expectation_years,
                rng=rng,
                sample_mode="single_draw",
                n_samples=BLOCK_EXPECTATION_DRAWS,
            )["annual_log_cost_paths"]
    elif selected_model == "way_fixed_rho_benchmark_035":
        annual_paths = {}
        for tech, artifact in artifacts.items():
            annual_paths[tech] = simulate_way_runtime(
                artifact=artifact,
                state=state["technology_states"][tech],
                block_dlog_experience=0.0,
                elapsed_years=expectation_years,
                rng=rng,
                sample_mode="single_draw",
                n_samples=BLOCK_EXPECTATION_DRAWS,
            )["annual_log_cost_paths"]
    else:
        raise ValueError(f"Unsupported stochastic model for block-average expectations: {selected_model}")

    diagnostics = {}
    for tech, path in annual_paths.items():
        if path.shape[1] != expectation_years:
            raise ValueError(
                f"Expected {expectation_years} annual path steps for {selected_model}/{tech}, got {path.shape[1]}"
            )
        level_costs = np.exp(path) / 1000.0
        endpoint_level = float(np.exp(state["technology_states"][tech]["last_log_capex"]) / 1000.0)
        expectation_sequence = np.concatenate(
            [
                np.full((level_costs.shape[0], 1), endpoint_level, dtype=float),
                level_costs[:, : max(expectation_years - 1, 0)],
            ],
            axis=1,
        )
        if expectation_sequence.shape[1] != expectation_years:
            raise ValueError(
                f"Expected {expectation_years} expectation-sequence points for {selected_model}/{tech}, "
                f"got {expectation_sequence.shape[1]}"
            )
        weighted_costs = expectation_sequence @ weights
        applied_c_overnight = float(np.mean(weighted_costs))
        terminal_c_overnight = float(np.mean(level_costs[:, -1]))
        diagnostics[tech] = _cost_statistics_from_levels(
            applied_c_overnight=applied_c_overnight,
            terminal_c_overnight=terminal_c_overnight,
            learning_cfg=learning_cfg,
            costs_file=costs_file,
            tech=tech,
        )
    return diagnostics


def simulate_cgrw_runtime(artifact, state, elapsed_years, rng, sample_mode, n_samples=None):
    params = artifact["parameter_summary"]
    sigma = float(params["sigma"])
    alpha = float(params["alpha"])
    rho = float(params["rho"])
    last_log = float(state["last_log_capex"])
    last_dlog = float(state["last_dlog_capex"])
    n = _runtime_sample_count(sample_mode, n_samples=n_samples)
    log_cost = np.full(n, last_log, dtype=float)
    dlog_prev = np.full(n, last_dlog, dtype=float)
    annual_log_cost_paths = _empty_annual_log_cost_paths(n, elapsed_years)
    for _ in range(int(elapsed_years)):
        eps = sigma * rng.standard_normal(n)
        dlog = alpha + rho * dlog_prev + eps
        log_cost = log_cost + dlog
        dlog_prev = dlog
        annual_log_cost_paths[:, _] = log_cost
    snapshot = _cgrw_runtime_snapshot(artifact)
    return {
        "final_log_cost": _single_or_median(log_cost, sample_mode),
        "state": {
            "last_log_capex": _single_or_median(log_cost, sample_mode),
            "last_dlog_capex": _single_or_median(dlog_prev, sample_mode),
            "state_type": state.get("state_type", "geometric_random_walk"),
            "parameter_snapshot": snapshot,
        },
        "annual_log_cost_paths": annual_log_cost_paths,
    }


def simulate_way_runtime(artifact, state, block_dlog_experience, elapsed_years, rng, sample_mode, n_samples=None):
    params = artifact["parameter_summary"]
    alpha = float(params["alpha"])
    beta = float(params["slope_dlog_experience"])
    theta = float(params["theta_ma1"])
    sigma = float(params["sigma"])
    last_log = float(state["last_log_capex"])
    if "last_innovation" not in state:
        raise ValueError("Way runtime state is missing last_innovation")
    last_eps = float(state["last_innovation"])
    n = _runtime_sample_count(sample_mode, n_samples=n_samples)
    log_cost = np.full(n, last_log, dtype=float)
    eps_prev = np.full(n, last_eps, dtype=float)
    snapshot = _way_runtime_snapshot(artifact)
    years = int(elapsed_years)
    if years < 0:
        raise ValueError(f"Elapsed years for Way runtime must be >= 0, got {years}")
    if years == 0:
        return {
            "final_log_cost": _single_or_median(log_cost, sample_mode),
            "state": {
                "last_log_capex": _single_or_median(log_cost, sample_mode),
                "last_innovation": _single_or_median(eps_prev, sample_mode),
                "theta_ma1": theta,
                "state_type": state.get("state_type", "ma1_wright_fixed_rho"),
                "parameter_snapshot": snapshot,
            },
            "annual_log_cost_paths": _empty_annual_log_cost_paths(n, years),
        }
    x_curr = float(block_dlog_experience) / float(years)
    annual_log_cost_paths = _empty_annual_log_cost_paths(n, years)
    for _ in range(years):
        eps = sigma * rng.standard_normal(n)
        dlog = alpha + beta * x_curr + eps + theta * eps_prev
        log_cost = log_cost + dlog
        eps_prev = eps
        annual_log_cost_paths[:, _] = log_cost
    return {
        "final_log_cost": _single_or_median(log_cost, sample_mode),
        "state": {
            "last_log_capex": _single_or_median(log_cost, sample_mode),
            "last_innovation": _single_or_median(eps_prev, sample_mode),
            "theta_ma1": theta,
            "state_type": state.get("state_type", "ma1_wright_fixed_rho"),
            "parameter_snapshot": snapshot,
        },
        "annual_log_cost_paths": annual_log_cost_paths,
    }


def simulate_shared_state_runtime(
    artifacts,
    state,
    block_dlog_experience_by_tech,
    elapsed_years,
    rng,
    sample_mode,
    n_samples=None,
):
    techs = list(artifacts.keys())
    sample_artifact = next(iter(artifacts.values()))
    horizons = int(elapsed_years)
    n = _runtime_sample_count(sample_mode, n_samples=n_samples)
    if horizons < 0:
        raise ValueError(f"Elapsed years for shared-state runtime must be >= 0, got {horizons}")

    p_ss_all = np.asarray(sample_artifact["uncertainty_terms"].get("p_slow_slow_draws", []), dtype=float)
    p_ff_all = np.asarray(sample_artifact["uncertainty_terms"].get("p_fast_fast_draws", []), dtype=float)
    if p_ss_all.size == 0 or p_ff_all.size == 0:
        raise ValueError("Shared-state runtime artifacts are missing transition-probability draws")
    draw_idx = rng.integers(0, len(p_ss_all), size=n)
    p_ss = p_ss_all[draw_idx]
    p_ff = p_ff_all[draw_idx]

    shared_state = state.get("shared_regime_state", {}) or {}
    if "initial_regime_probs" not in shared_state:
        raise ValueError("Shared-state runtime state is missing initial_regime_probs")
    init_probs = list(shared_state["initial_regime_probs"])
    current_regime_value = shared_state.get("current_regime", 0)
    if current_regime_value is None:
        current_regime = (rng.random(n) > float(init_probs[0])).astype(int)
    else:
        current_regime = np.full(n, int(current_regime_value), dtype=int)

    tech_states = {}
    log_costs = {}
    annual_log_cost_paths = {}
    param_cache = {}
    for tech, artifact in artifacts.items():
        state_tech = state["technology_states"][tech]
        log_costs[tech] = np.full(n, float(state_tech["last_log_capex"]), dtype=float)
        annual_log_cost_paths[tech] = _empty_annual_log_cost_paths(n, horizons)
        unc = artifact.get("uncertainty_terms", {}) or {}
        params = artifact["parameter_summary"]
        def _draws(key, mean_key):
            arr = np.asarray(unc.get(key, []), dtype=float)
            if arr.size == 0:
                raise ValueError(f"Shared-state runtime artifacts for {tech} are missing {key}")
            return arr[draw_idx]
        param_cache[tech] = {
            "alpha_slow": _draws("alpha_slow_draws", "alpha_slow_mean"),
            "alpha_fast": _draws("alpha_fast_draws", "alpha_fast_mean"),
            "beta_slow": _draws("beta_slow_draws", "beta_slow_mean"),
            "beta_fast": _draws("beta_fast_draws", "beta_fast_mean"),
            "sigma": _draws("sigma_draws", "sigma_mean"),
        }

    if horizons == 0:
        regime_scalar = _single_or_median(current_regime, sample_mode)
        probs = [float(np.mean(current_regime == 0)), float(np.mean(current_regime == 1))]
        for tech in techs:
            tech_states[tech] = {
                "last_log_capex": _single_or_median(log_costs[tech], sample_mode),
                "parameter_snapshot": _shared_runtime_snapshot(
                    param_cache[tech],
                    current_regime,
                    probs,
                    p_ss,
                    p_ff,
                    sample_mode,
                ),
            }
        return {
            "final_log_costs": {tech: _single_or_median(log_costs[tech], sample_mode) for tech in techs},
            "technology_states": tech_states,
            "shared_regime_state": {
                "current_regime": int(round(regime_scalar)),
                "initial_regime_probs": probs,
                "p_slow_slow": _single_or_median(p_ss, sample_mode),
                "p_fast_fast": _single_or_median(p_ff, sample_mode),
            },
            "annual_log_cost_paths": annual_log_cost_paths,
        }

    x_step_by_tech = {
        tech: float(block_dlog_experience_by_tech[tech]) / float(horizons)
        for tech in techs
    }

    for step in range(horizons):
        stay = rng.random(n)
        stay_prob = np.where(current_regime == 0, p_ss, p_ff)
        current_regime = np.where(stay <= stay_prob, current_regime, 1 - current_regime)
        for tech in techs:
            x_curr = x_step_by_tech[tech]
            pars = param_cache[tech]
            alpha = np.where(current_regime == 0, pars["alpha_slow"], pars["alpha_fast"])
            beta = np.where(current_regime == 0, pars["beta_slow"], pars["beta_fast"])
            dlog = alpha + beta * x_curr + pars["sigma"] * rng.standard_normal(n)
            log_costs[tech] = log_costs[tech] + dlog
            annual_log_cost_paths[tech][:, step] = log_costs[tech]

    regime_scalar = _single_or_median(current_regime, sample_mode)
    probs = [float(np.mean(current_regime == 0)), float(np.mean(current_regime == 1))]
    for tech in techs:
        tech_states[tech] = {
            "last_log_capex": _single_or_median(log_costs[tech], sample_mode),
            "parameter_snapshot": _shared_runtime_snapshot(
                param_cache[tech],
                current_regime,
                probs,
                p_ss,
                p_ff,
                sample_mode,
            ),
        }
    return {
        "final_log_costs": {tech: _single_or_median(log_costs[tech], sample_mode) for tech in techs},
        "technology_states": tech_states,
        "shared_regime_state": {
            "current_regime": int(round(regime_scalar)),
            "initial_regime_probs": probs,
            "p_slow_slow": _single_or_median(p_ss, sample_mode),
            "p_fast_fast": _single_or_median(p_ff, sample_mode),
        },
        "annual_log_cost_paths": annual_log_cost_paths,
    }


def calculate_stochastic_learning_costs(
    learning_cfg,
    selected_model,
    current_year,
    planning_horizons,
    prev_network_path,
    prev_state_path,
    costs_file,
    global_scale_factors,
    wacc_dict=None,
):
    artifacts, state = load_stochastic_runtime_state(
        learning_cfg,
        selected_model,
        current_year,
        prev_state_path,
    )
    cost_expectation_mode = resolve_cost_expectation_mode(learning_cfg, selected_model)
    runtime_metadata = build_runtime_metadata(
        learning_cfg,
        "stochastic_forecast",
        selected_model,
        cost_expectation_mode=cost_expectation_mode,
    )

    if current_year not in COST_HISTORICAL_CAPACITY_YEARS and prev_network_path is None:
        raise ValueError(f"Stochastic runtime for {current_year} requires previous solved network")
    if current_year not in COST_HISTORICAL_CAPACITY_YEARS and prev_state_path is None:
        raise ValueError(
            f"Stochastic runtime for {current_year} requires previous committed learning state"
        )

    learning_costs = _state_to_runtime_learning_costs(
        artifacts=artifacts,
        state=state,
        current_year=current_year,
        selected_model=selected_model,
        learning_cfg=learning_cfg,
        costs_file=costs_file,
        global_scale_factors=global_scale_factors,
        prev_network_path=prev_network_path,
        wacc_dict=wacc_dict,
        runtime_metadata=runtime_metadata,
    )

    proposed_state = json.loads(json.dumps(state))
    proposed_state.update(runtime_metadata)
    return learning_costs, proposed_state


def _stochastic_alignment_years_by_tech(artifacts, learning_cfg, current_year, base_year):
    training_window = str(learning_cfg.get("training_window", "origin_cutoff"))
    if training_window != "full_sample":
        return {tech: 0 for tech in artifacts}

    target_year = min(int(current_year), int(base_year))
    if target_year < int(current_year):
        return {tech: 0 for tech in artifacts}

    alignment_years = {}
    for tech, artifact in artifacts.items():
        train_end_year = _artifact_train_end_year(artifact, base_year)
        alignment_years[tech] = max(0, target_year - int(train_end_year))
    return alignment_years


def _shared_result_from_current_state(state, artifacts):
    return {
        "final_log_costs": {
            tech: float(state["technology_states"][tech]["last_log_capex"]) for tech in artifacts
        },
        "technology_states": {
            tech: {
                "last_log_capex": float(state["technology_states"][tech]["last_log_capex"]),
                "parameter_snapshot": (state["technology_states"][tech].get("parameter_snapshot", None)),
            }
            for tech in artifacts
        },
        "shared_regime_state": state.get("shared_regime_state", {}),
    }


def update_stochastic_runtime_state(
    learning_cfg,
    selected_model,
    current_year,
    state,
    costs_file,
    solved_capacity_by_tech,
    wacc_dict=None,
):
    artifacts, initial_state = load_stochastic_model_artifacts(learning_cfg, selected_model)
    state = _backfill_stochastic_runtime_state_fields(state, initial_state, selected_model, artifacts)
    validate_stochastic_runtime_state(state, selected_model, artifacts)
    sample_mode = str(learning_cfg.get("sample_mode", "single_draw"))
    seed = int(learning_cfg.get("seed", 0))
    rng = np.random.default_rng(seed + 1000 * int(current_year))
    cost_expectation_mode = resolve_cost_expectation_mode(learning_cfg, selected_model)
    runtime_metadata = build_runtime_metadata(
        learning_cfg,
        "stochastic_forecast",
        selected_model,
        cost_expectation_mode=cost_expectation_mode,
    )
    if state.get("seed") not in (None, ""):
        runtime_metadata["seed"] = int(state["seed"])
    if state.get("learning_seed") not in (None, ""):
        runtime_metadata["learning_seed"] = str(state["learning_seed"])
    passthrough_fields = {}
    for key in ("committed_from_network", "enabled", "fossil_price_states"):
        if key in state:
            passthrough_fields[key] = state[key]

    base_year = int(state.get("last_applied_year", current_year))
    elapsed_years = max(0, int(current_year) - base_year)
    if elapsed_years < 0:
        raise ValueError(
            f"Invalid stochastic runtime state transition: current_year={current_year}, last_applied_year={base_year}"
        )
    alignment_years_by_tech = _stochastic_alignment_years_by_tech(
        artifacts,
        learning_cfg,
        current_year,
        base_year,
    )
    positive_alignment_years = sorted({years for years in alignment_years_by_tech.values() if years > 0})
    if len(positive_alignment_years) > 1:
        raise ValueError(
            "Stochastic runtime alignment only supports a single positive catch-up horizon per update. "
            f"Got {alignment_years_by_tech}."
        )
    alignment_years = positive_alignment_years[0] if positive_alignment_years else 0
    next_state_year = int(current_year) if int(current_year) > base_year else int(base_year)

    if selected_model == "shared_state_bayesian_regime_wright":
        if int(current_year) <= base_year:
            if alignment_years > 0:
                block_dlog_by_tech = {
                    tech: (
                        compute_realized_block_growth(tech, learning_cfg, state, current_year, years)
                        if years > 0 else 0.0
                    )
                    for tech, years in alignment_years_by_tech.items()
                }
                shared_result = simulate_shared_state_runtime(
                    artifacts,
                    state,
                    block_dlog_by_tech,
                    alignment_years,
                    rng,
                    sample_mode,
                )
                for tech, years in alignment_years_by_tech.items():
                    if years == 0:
                        shared_result["technology_states"][tech]["last_log_capex"] = float(
                            state["technology_states"][tech]["last_log_capex"]
                        )
            else:
                shared_result = _shared_result_from_current_state(state, artifacts)
        else:
            block_dlog_by_tech = {
                tech: compute_realized_block_growth(tech, learning_cfg, state, current_year, elapsed_years)
                for tech in artifacts
            }
            shared_result = simulate_shared_state_runtime(
                artifacts,
                state,
                block_dlog_by_tech,
                elapsed_years,
                rng,
                sample_mode,
            )

        next_state = {
            **passthrough_fields,
            **runtime_metadata,
            "last_applied_year": next_state_year,
            "technology_states": shared_result["technology_states"],
            "shared_regime_state": shared_result["shared_regime_state"],
            "capacity_history": state.get("capacity_history", {}),
            "modeled_capacity_history": state.get("modeled_capacity_history", {}),
        }
        learning_costs = _learning_costs_from_stochastic_state(
            artifacts=artifacts,
            state=next_state,
            selected_model=selected_model,
            cumulative_capacity_map=solved_capacity_by_tech,
            learning_cfg=learning_cfg,
            costs_file=costs_file,
            wacc_dict=wacc_dict,
            runtime_metadata=runtime_metadata,
        )
        return learning_costs, next_state

    next_state = {
        **passthrough_fields,
        **runtime_metadata,
        "last_applied_year": next_state_year,
        "technology_states": {},
        "capacity_history": state.get("capacity_history", {}),
        "modeled_capacity_history": state.get("modeled_capacity_history", {}),
    }
    for tech, artifact in artifacts.items():
        state_tech = state["technology_states"][tech]
        if int(current_year) <= base_year:
            tech_elapsed_years = int(alignment_years_by_tech.get(tech, 0))
        else:
            tech_elapsed_years = elapsed_years
        if selected_model == "correlated_geometric_random_walk":
            result = simulate_cgrw_runtime(artifact, state_tech, tech_elapsed_years, rng, sample_mode)
        else:
            if tech_elapsed_years > 0:
                block_dlog = compute_realized_block_growth(
                    tech,
                    learning_cfg,
                    state,
                    current_year,
                    tech_elapsed_years,
                )
            else:
                block_dlog = 0.0
            result = simulate_way_runtime(artifact, state_tech, block_dlog, tech_elapsed_years, rng, sample_mode)
        next_state["technology_states"][tech] = result["state"]
    learning_costs = _learning_costs_from_stochastic_state(
        artifacts=artifacts,
        state=next_state,
        selected_model=selected_model,
        cumulative_capacity_map=solved_capacity_by_tech,
        learning_cfg=learning_cfg,
        costs_file=costs_file,
        wacc_dict=wacc_dict,
        runtime_metadata=runtime_metadata,
    )
    return learning_costs, next_state


def _cost_unit_for_runtime(tech):
    return "kWh" if tech in ENERGY_TECHS else "kW"


def _runtime_capacity_for_log(tech, learning_cfg, state, current_year, prev_network_path, tech_mapping, global_scale_factors):
    if current_year in COST_HISTORICAL_CAPACITY_YEARS:
        hist_year = COST_HISTORICAL_CAPACITY_YEARS[current_year]
        return float(load_capacity_from_historical_csv(tech, hist_year, learning_cfg))
    if prev_network_path is None:
        raise ValueError(f"Missing previous solved network for runtime capacity extraction of {tech} in {current_year}")
    realized = extract_capacity_from_network(prev_network_path, tech_mapping, global_scale_factors)
    if tech not in realized:
        raise ValueError(f"Technology {tech} missing in previous solved network")
    lag_year = int(state.get("last_applied_year", current_year))
    return float(get_learning_base_capacity_for_year(state, tech, lag_year, fallback_capacity=realized[tech]))


def _get_runtime_capacity_map(current_year, prev_network_path, learning_cfg, state, tech_mapping, global_scale_factors):
    if current_year in COST_HISTORICAL_CAPACITY_YEARS:
        historical_year = COST_HISTORICAL_CAPACITY_YEARS[current_year]
        return {
            tech: float(load_capacity_from_historical_csv(tech, historical_year, learning_cfg))
            for tech in ("solar_power", "onwind_power", "battery_energy")
        }
    if prev_network_path is None:
        raise ValueError(
            f"Missing previous solved network for runtime capacity extraction in {current_year}"
        )
    realized = extract_capacity_from_network(prev_network_path, tech_mapping, global_scale_factors)
    lag_year = int(state.get("last_applied_year", current_year))
    runtime_map = {}
    for tech in ("solar_power", "onwind_power", "battery_energy"):
        if tech not in realized and tech != "battery_energy":
            raise ValueError(f"Technology {tech} missing in previous solved network")
        runtime_map[tech] = float(
            get_learning_base_capacity_for_year(
                state,
                tech,
                lag_year,
                fallback_capacity=realized.get(tech),
            )
        )
    return runtime_map


def extract_capacity_from_network(network_path, tech_mapping, global_scale_factors):
    """
    Extract realized installed capacity from solved network.
    
    Sums p_nom_opt (for generators, links) or e_nom_opt (for stores) by carrier.
    Converts carrier names to technology keys using tech_mapping.
    Applies optional global scaling factors for non-battery technologies.
    Battery energy is handled separately through the bundled phi_t mapping.
    
    Args:
        network_path: Path to solved network
        tech_mapping: Dict mapping carrier names to technology keys (e.g., "solar" -> "solar_power")
    
    Returns:
        Dict of {tech: capacity_GW}
    """
    logger.info(f"  Extracting capacity from solved network: {network_path}")
    n = pypsa.Network(network_path)
    
    # Mapping: carrier to tech (tech_mapping is already carrier -> tech from config)
    carrier_to_tech = tech_mapping
    
    realized = {}
    
    # Extract from generators (p_nom_opt)
    if not n.generators.empty:
        gen_capacity = n.generators.groupby("carrier")["p_nom_opt"].sum() / 1e3  # MW -> GW
        for carrier, capacity in gen_capacity.items():
            if carrier in carrier_to_tech:
                tech = carrier_to_tech[carrier]
                realized[tech] = realized.get(tech, 0.0) + capacity
    
    # Extract from storage_units (p_nom_opt)
    if not n.storage_units.empty:
        su_capacity = n.storage_units.groupby("carrier")["p_nom_opt"].sum() / 1e3
        for carrier, capacity in su_capacity.items():
            if carrier in carrier_to_tech:
                tech = carrier_to_tech[carrier]
                realized[tech] = realized.get(tech, 0.0) + capacity
    
    # Extract from links (p_nom_opt)
    if not n.links.empty:
        link_capacity = n.links.groupby("carrier")["p_nom_opt"].sum() / 1e3
        for carrier, capacity in link_capacity.items():
            if carrier in carrier_to_tech:
                tech = carrier_to_tech[carrier]
                realized[tech] = realized.get(tech, 0.0) + capacity

    # Fallback for sector batteries where carriers are charger/discharger (not "battery inverter")
    # Use a single directional link capacity (max of charger/discharger) to avoid double counting.
    if not n.links.empty and "battery_power" not in realized:
        links_carriers = set(n.links.carrier.unique())
        if {"battery charger", "battery discharger"} & links_carriers:
            charger_cap = (
                n.links.loc[n.links.carrier == "battery charger", "p_nom_opt"].sum() / 1e3
            )
            discharger_cap = (
                n.links.loc[n.links.carrier == "battery discharger", "p_nom_opt"].sum() / 1e3
            )
            inferred_battery_power = max(charger_cap, discharger_cap)
            realized["battery_power"] = inferred_battery_power
            logger.info(
                "    Inferred battery_power from battery charger/discharger links: "
                f"max({charger_cap:.2f}, {discharger_cap:.2f}) = {inferred_battery_power:.2f} GW"
            )
    
    # Extract from stores (e_nom_opt) - energy capacity
    if not n.stores.empty:
        store_capacity = n.stores.groupby("carrier")["e_nom_opt"].sum() / 1e3  # MWh to GWh
        for carrier, capacity in store_capacity.items():
            if carrier in carrier_to_tech:
                tech = carrier_to_tech[carrier]
                # Note: For energy storage, capacity is in GWh not GW
                realized[tech] = realized.get(tech, 0.0) + capacity
    
    # Apply optional global scaling factors for non-battery technologies that
    # represent only a fraction of global deployment. battery_energy is kept in
    # raw modeled units here and is mapped to global Li-ion experience later via
    # the committed-state phi_t recursion.
    for tech in list(realized.keys()):
        if tech in global_scale_factors and tech not in {"battery_energy", "battery_power"}:
            modeled_cap = realized[tech]
            scale_factor = global_scale_factors[tech]
            global_cap = modeled_cap * scale_factor
            realized[tech] = global_cap
            unit = "GWh" if tech in ['battery_energy', 'h2_energy'] else "GW"
            logger.info(f"    {tech}: {modeled_cap:.2f} {unit} (modeled) × {scale_factor:.2f} = {global_cap:.2f} {unit} (global)")
        else:
            unit = "GWh" if tech in ['battery_energy', 'h2_energy'] else "GW"
            logger.info(f"    {tech}: {realized[tech]:.2f} {unit}")
    
    return realized


def calculate_learning_costs(
    params,
    learning_cfg,
    current_year,
    planning_horizons,
    prev_network_path,
    prior_state_payload,
    costs_file,
    global_scale_factors,
    wacc_dict=None,
):
    """
    Calculate learning-based costs for all technologies at current horizon.
    
    WORKFLOW (Exogenous Learning):
    - Uses lag_periods parameter from config to determine capacity lag
    - lag_periods=0: NOT handled here (use learning.py for immediate endogenous learning)
    - lag_periods=1: Uses previous period's capacity (exogenous costs calculated here)
      * For 2020: Uses 2015 historical cumulative capacity
      * For 2025: Uses 2020 historical cumulative capacity
      * For 2030: Uses 2025 solved capacity / committed cumulative experience
      * And so on...
    
    Args:
        params: DataFrame with learning parameters (A, β) from learning_params.csv
        learning_cfg: Learning configuration dict (must include lag_periods)
        current_year: Current planning horizon year
        planning_horizons: List of all planning horizons
        prev_network_path: Path to previous solved network (for extracting capacity)
        prior_state_payload: Prior committed learning state from the lagged horizon
        costs_file: Path to cost CSV file for this year
        global_scale_factors: Dict of per-technology global scaling multipliers
        wacc_dict: Optional dict of regional WACCs {tech: {country: wacc}}
    
    Returns:
        Dict of {tech: {capital_cost, cumulative_capacity_GW, c_overnight, unit, ...}}
    
    Raises:
        ValueError: If lag_periods=0 (should use learning.py), invalid parameters, or capacity not found
    """
    # Get lag_periods configuration (number of periods to lag)
    # lag_periods=1 means use previous period's capacity for costs
    # lag_periods=0 would be endogenous (not handled here)
    lag_periods = learning_cfg.get("lag_periods", 1)  # Default to 1 period lag
    
    # Validate lag_periods > 0 (lag_periods=0 should use endogenous learning.py)
    if lag_periods == 0:
        raise ValueError(
            "lag_periods=0 is not supported in apply_learning_costs.py. "
            "For immediate learning, use endogenous optimization with learning.py (SOS2/MILP)."
        )
    
    logger.info(f"Using lagged learning with lag_periods={lag_periods} (exogenous costs)")
    
    # Determine lag year based on previous planning horizon
    lag_year = get_cost_lag_year(planning_horizons, current_year)
    
    logger.info(f"Calculating costs for {current_year} based on capacity from {lag_year} (lag_periods={lag_periods})")
    
    # Find available historical horizons
    horizons = sorted([h for h in planning_horizons if h <= lag_year])

    # For 2030+, extract lagged realized capacities once to avoid repeated imports/log spam.
    realized_capacity = None
    if current_year not in COST_HISTORICAL_CAPACITY_YEARS:
        if lag_year not in horizons:
            raise ValueError(
                f"Cannot find capacity for lag year {lag_year}. "
                f"Available horizons <= {lag_year}: {horizons}. "
                f"For lag_periods={lag_periods} in year {current_year}, need solved network from {lag_year}."
            )

        if prev_network_path is None:
            raise ValueError(
                f"No previous network path provided for {current_year}. "
                f"Cannot extract capacity from {lag_year} for learning curve calculation. "
                f"This is required for exogenous learning with lag_periods={lag_periods}."
            )

        # Verify the previous network is from the expected lag year.
        # Extract a 4-digit year token robustly from myopic filenames, e.g.:
        # ..._2020_0.071_AB_0.0export_base.nc
        network_year = extract_year_from_network_path(prev_network_path)
        if network_year is not None:
            if network_year != lag_year:
                logger.warning(
                    f"Network year mismatch! Expected {lag_year} for {current_year} costs "
                    f"(lag_periods={lag_periods}), but network path indicates year {network_year}. "
                    f"Path: {prev_network_path}"
                )
            else:
                logger.info(f"✓ Verified: Using {network_year} network for {current_year} costs (exogenous lag)")
        else:
            logger.warning(
                "Could not extract year from network path to verify lag period. "
                f"Path: {prev_network_path}"
            )

        logger.info(f"Extracting solved capacities from {lag_year} network once for all technologies")
        tech_mapping = get_tech_mapping(learning_cfg)
        realized_capacity = extract_capacity_from_network(
            prev_network_path, tech_mapping, global_scale_factors
        )
    
    learning_costs = {}
    battery_energy_bos_multiplier = get_battery_energy_bos_multiplier(learning_cfg, costs_file)
    
    for tech in params.index:
        logger.info(f"  Processing {tech}...")
        
        # Get parameters
        A = params.loc[tech, "A_over_unit"]
        beta = params.loc[tech, "beta"]
        unit = params.loc[tech, "unit"]
        
        # Validate technology units
        try:
            validate_technology_units(tech, unit)
        except ValueError as e:
            logger.error(f"    Unit validation failed: {e}")
            raise
        
        # Calculate learning rate from beta for validation
        # LR = 1 - 2^(-beta)
        learning_rate = 1.0 - (2.0 ** (-beta))
        
        # Load cumulative capacity: historical bootstrap horizons or solved lagged horizon.
        if current_year in COST_HISTORICAL_CAPACITY_YEARS:
            # For historical bootstrap horizons: use historical cumulative capacity data only.
            historical_year = COST_HISTORICAL_CAPACITY_YEARS[current_year]
            logger.info(
                f"    Using {historical_year} historical cumulative capacity "
                f"(deterministic historical bootstrap for {current_year})"
            )
            
            try:
                cumulative_capacity = load_capacity_from_historical_csv(tech, historical_year, learning_cfg)
                capacity_unit = "GWh" if tech in ['battery_energy', 'h2_energy'] else "GW"
                logger.info(f"    Historical capacity {historical_year}: {cumulative_capacity:.1f} {capacity_unit}")
                
            except Exception as e:
                logger.error(f"    Failed to load historical capacity for {tech}: {e}")
                raise ValueError(
                    f"Failed to load {historical_year} capacity for {tech}: {e}"
                ) from e
        
        else:
            # For 2025+: Use pre-extracted capacities from previous horizon network
            if tech not in realized_capacity:
                raise ValueError(
                    f"Technology {tech} not found in network from {lag_year}. "
                    f"Network path: {prev_network_path}. "
                    f"Available technologies: {list(realized_capacity.keys())}"
                )
            
            # Use the lagged learning base from the previous committed horizon.
            # For battery_energy this is cumulative global Li-ion experience,
            # not the raw modeled battery stock in the solved network.
            cumulative_capacity = get_learning_base_capacity_for_year(
                prior_state_payload,
                tech,
                lag_year,
                fallback_capacity=realized_capacity[tech],
            )
            capacity_unit = "GWh" if tech in ENERGY_TECHS else "GW"
            logger.info(f"    Solved capacity from {lag_year}: {cumulative_capacity:.1f} {capacity_unit}")
        
        # Validate numerical parameters
        try:
            validate_learning_parameters(tech, A, beta, learning_rate, cumulative_capacity)
        except ValueError as e:
            logger.error(f"    Numerical validation failed: {e}")
            raise
        
        if current_year in COST_HISTORICAL_CAPACITY_YEARS:
            historical_year = COST_HISTORICAL_CAPACITY_YEARS[current_year]
            cost_window = COST_HISTORICAL_COST_WINDOWS.get(
                current_year,
                (historical_year, historical_year),
            )
            try:
                c_overnight = load_average_cost_from_historical_csv(
                    tech,
                    cost_window[0],
                    cost_window[1],
                    learning_cfg,
                )
            except Exception as e:
                logger.error(f"    Failed to load historical cost for {tech}: {e}")
                raise ValueError(
                    f"Failed to load average historical cost {cost_window[0]}-{cost_window[1]} "
                    f"for {tech}: {e}"
                ) from e
            if tech == 'battery_energy':
                c_overnight_cell = c_overnight
                c_overnight = c_overnight_cell * battery_energy_bos_multiplier
                logger.info(
                    "    Battery historical Li-ion cost: %.3f EUR/%s, energy-side cost (×%.4f): %.3f EUR/%s",
                    c_overnight_cell,
                    unit,
                    battery_energy_bos_multiplier,
                    c_overnight,
                    unit,
                )
            logger.info(
                f"    Using historical average cost window for bootstrap: "
                f"{current_year} <- {cost_window[0]}-{cost_window[1]}, "
                f"c_overnight={c_overnight:.3f} EUR/{unit}"
            )
        else:
            # Calculate overnight cost using learning curve: c = A * L^(-β)
            c_overnight_cell = A * (cumulative_capacity ** (-beta))

            # For batteries, apply data-derived energy-side multiplier
            # (learning at Li-ion pack level, converted to model battery-energy cost object).
            if tech == 'battery_energy':
                c_overnight = c_overnight_cell * battery_energy_bos_multiplier
                logger.info(
                    f"    Battery cell cost: {c_overnight_cell:.2f} EUR/{unit}, "
                    f"energy-side cost (×{battery_energy_bos_multiplier:.4f}): {c_overnight:.2f} EUR/{unit}"
                )
            else:
                c_overnight = c_overnight_cell
        
        # Convert to capital_cost (EUR/MW-yr or EUR/MWh-yr)
        # NOTE: For renewable technologies, we calculate a global average capital cost here
        # The actual regional WACCs will be applied in update_network_costs() on a per-bus basis
        capital_cost = convert_to_capital_cost(
            c_overnight, tech, unit, learning_cfg, costs_file
        )
        
        # Store both adjusted and base parameters for transparency
        A_base = params.loc[tech, "A_base"] if "A_base" in params.columns else A
        beta_base = params.loc[tech, "beta_base"] if "beta_base" in params.columns else beta
        lr_base = 1.0 - (2.0 ** (-beta_base))
        
        learning_costs[tech] = {
            "capital_cost": capital_cost,
            "capital_cost_terminal_point": capital_cost,
            "cumulative_capacity_GW": cumulative_capacity,
            "A": A,
            "A_base": A_base,
            "A_scenario": A,
            "beta": beta,
            "beta_base": beta_base,
            "lr_base": lr_base,
            "lr_scenario": learning_rate,
            "beta_adjusted": beta != beta_base,
            "unit": unit,
            "c_overnight": c_overnight,
            "c_overnight_terminal_point": c_overnight,
            "log_capex_runtime": float(np.log(max(c_overnight, 1.0e-12) * 1000.0)),
            "log_capex_terminal_point": float(np.log(max(c_overnight, 1.0e-12) * 1000.0)),
            "wacc_dict": wacc_dict,  # Store for per-bus calculation
        }
        
        # Determine capital cost unit for logging
        cap_cost_unit = "EUR/MW-yr" if unit == "kW" else "EUR/MWh-yr"
        
        logger.info(
            f"  {tech}: capacity={cumulative_capacity:.1f}{capacity_unit}, "
            f"c_overnight={c_overnight:.2f}EUR/{unit}, "
            f"capital_cost={capital_cost:.2f}{cap_cost_unit}"
        )
    
    return learning_costs


def convert_to_capital_cost(c_overnight, tech, unit, learning_cfg, costs_file, wacc_override=None):
    """
    Convert overnight CAPEX to PyPSA capital_cost.
    
    capital_cost = (annuity + FOM) * investment
    where FOM is loaded from the cost CSV file as a decimal (e.g., 0.01578 for 1.578%/year)
    
    Args:
        c_overnight: Overnight CAPEX in EUR/kW or EUR/kWh
        tech: Technology key
        unit: 'kW' or 'kWh'
        learning_cfg: Learning configuration
        costs_file: Path to cost CSV file for this year
        wacc_override: Optional WACC to use instead of config value (for regional WACCs)
    
    Returns:
        capital_cost in EUR/MW-yr or EUR/MWh-yr
    """
    finance = get_learning_finance(learning_cfg)
    
    # Get financial parameters - use override if provided (for regional WACCs)
    if wacc_override is not None:
        wacc = wacc_override
    else:
        wacc = finance["wacc"].get(tech, finance["wacc"]["default"])
    lifetime = finance["lifetime"][tech]
    
    # Load FOM from cost file
    fom = load_fom_from_costs(costs_file, tech)
    
    # Calculate annuity factor
    annuity = calculate_annuity(lifetime, wacc)
    
    if unit == "kW":
        # Power technology: EUR/kW → EUR/MW-yr
        investment_mw = c_overnight * 1e3  # kW → MW conversion
        
        # Capital cost formula: (annuity + FOM) * investment
        # FOM is already in decimal form (e.g., 0.01578 for 1.578%/year)
        capital_cost = (annuity + fom) * investment_mw
        
    elif unit == "kWh":
        # Energy technology: EUR/kWh → EUR/MWh-yr
        investment_mwh = c_overnight * 1e3  # kWh → MWh conversion
        
        # Capital cost formula: (annuity + FOM) * investment
        capital_cost = (annuity + fom) * investment_mwh
        
    else:
        raise ValueError(f"Invalid unit '{unit}' for technology {tech}")
    
    return capital_cost


def _stable_uint64_from_key(*parts):
    key = "::".join(str(part) for part in parts).encode("utf-8")
    digest = __import__("hashlib").sha256(key).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _fossil_standard_normal(seed, fuel_type, market, year):
    rng = np.random.default_rng(_stable_uint64_from_key(seed, fuel_type, market, int(year)))
    return float(rng.standard_normal())


def _resolve_fossil_state(
    params_row,
    existing_state,
    target_year,
    seed,
    historical_by_key,
):
    fuel_type = str(params_row["fuel_type"])
    market = str(params_row["market"])
    target_year = int(target_year)

    if existing_state:
        start_year = int(existing_state.get("last_state_year", existing_state.get("source_state_year", target_year)))
        log_price = float(existing_state["realized_log_price"])
        price = float(existing_state["realized_price_eur_mwh"])
        source_state_year = int(existing_state.get("source_state_year", start_year))
        if start_year > target_year:
            existing_state = None
        elif start_year == target_year:
            return {
                "realized_log_price": log_price,
                "realized_price_eur_mwh": price,
                "last_state_year": start_year,
                "source_state_year": source_state_year,
            }
    if not existing_state:
        last_observed_year = int(params_row["last_observed_year"])
        if target_year <= last_observed_year:
            historical_key = (fuel_type, market, target_year)
            if historical_key not in historical_by_key:
                raise ValueError(
                    f"Historical market price missing for {fuel_type}/{market}/{target_year} "
                    "in fossil price bundle."
                )
            price = float(historical_by_key[historical_key])
            return {
                "realized_log_price": float(np.log(price)),
                "realized_price_eur_mwh": price,
                "last_state_year": target_year,
                "source_state_year": target_year,
            }
        start_year = last_observed_year
        log_price = float(params_row["last_observed_log_price"])
        price = float(params_row["last_observed_price_eur_mwh"])
        source_state_year = last_observed_year

    for year in range(int(start_year) + 1, target_year + 1):
        epsilon = float(params_row["sigma_epsilon"]) * _fossil_standard_normal(
            seed,
            fuel_type,
            market,
            year,
        )
        log_price = float(params_row["kappa"]) + float(params_row["phi"]) * float(log_price) + epsilon
        price = float(np.exp(log_price))
        start_year = year

    return {
        "realized_log_price": float(log_price),
        "realized_price_eur_mwh": float(price),
        "last_state_year": int(start_year),
        "source_state_year": int(source_state_year),
    }


def _build_fossil_price_payload(learning_cfg, runtime_metadata, current_year, state_payload):
    cfg = get_fossil_price_cfg(learning_cfg)
    expectation_mode = str(
        runtime_metadata.get(
            "fossil_price_expectation_mode",
            get_requested_fossil_price_expectation_mode(learning_cfg),
        )
    )
    if not cfg["enabled"]:
        return {
            "enabled": False,
            "log_df": pd.DataFrame(
                columns=[
                    "learning_seed",
                    "planning_horizon",
                    "fuel_type",
                    "market",
                    "country",
                    "applied_price_eur_mwh",
                    "price_terminal_point_eur_mwh",
                    "fossil_price_expectation_mode",
                    "fossil_price_expectation_weights_json",
                    "last_state_year",
                    "source_state_year",
                ]
            ),
            "price_dict": {},
            "states": state_payload.get("fossil_price_states", {}) or {},
            "summary": {"fossil_price_uncertainty_enabled": False},
        }

    bundle = learning_cfg.get("_fossil_price_bundle")
    if not bundle:
        raise ValueError("Fossil price uncertainty is enabled but no bundle is loaded.")

    seed = int(learning_cfg.get("seed", 0))
    input_states = (state_payload.get("fossil_price_states", {}) or {})
    next_states = {fuel: dict(input_states.get(fuel, {}) or {}) for fuel in cfg["fuels"]}
    expectation_weights = get_fossil_price_expectation_weights(learning_cfg)
    expectation_weights_json = get_fossil_price_expectation_weights_json(learning_cfg)
    for fuel_type in cfg["fuels"]:
        fuel_params = bundle["params_df"].loc[
            bundle["params_df"]["fuel_type"].astype(str).str.lower() == fuel_type
        ]
        for params_row in fuel_params.to_dict(orient="records"):
            market = str(params_row["market"])
            existing_state = (input_states.get(fuel_type, {}) or {}).get(market)
            resolved = _resolve_fossil_state(
                params_row=params_row,
                existing_state=existing_state,
                target_year=current_year,
                seed=seed,
                historical_by_key=bundle["historical_by_key"],
            )
            next_states.setdefault(fuel_type, {})[market] = resolved

    country_rows = []
    for fuel_type in cfg["fuels"]:
        mapping_rows = bundle["country_map_df"].loc[
            bundle["country_map_df"]["fuel_type"].astype(str).str.lower() == fuel_type
        ]
        for row in mapping_rows.to_dict(orient="records"):
            country = str(row["country"])
            market = str(row["market"])
            state = (next_states.get(fuel_type, {}) or {}).get(market)
            if state is None:
                if cfg["fallback_to_static_prices"]:
                    continue
                raise ValueError(
                    f"Missing realized fossil state for {fuel_type}/{market} while pricing country {country}."
                )
            terminal_price = float(state["realized_price_eur_mwh"])
            applied_price = terminal_price
            if expectation_mode == "block_average_expected":
                expectation_years = int(len(expectation_weights))
                first_weight_year = int(current_year) - expectation_years
                expectation_sequence = []
                start_state = (input_states.get(fuel_type, {}) or {}).get(market)
                params_row = bundle["params_by_key"][(fuel_type, market)]
                for expectation_year in range(first_weight_year, int(current_year)):
                    expectation_state = _resolve_fossil_state(
                        params_row=params_row,
                        existing_state=start_state,
                        target_year=expectation_year,
                        seed=seed,
                        historical_by_key=bundle["historical_by_key"],
                    )
                    expectation_sequence.append(float(expectation_state["realized_price_eur_mwh"]))
                if len(expectation_sequence) != expectation_years:
                    raise ValueError(
                        f"Expected {expectation_years} fossil expectation points for {fuel_type}/{market}, "
                        f"got {len(expectation_sequence)}"
                    )
                applied_price = float(np.dot(np.asarray(expectation_sequence, dtype=float), expectation_weights))
            country_rows.append(
                {
                    "learning_seed": runtime_metadata.get("learning_seed", ""),
                    "planning_horizon": int(current_year),
                    "fuel_type": fuel_type,
                    "market": market,
                    "country": country,
                    "applied_price_eur_mwh": applied_price,
                    "price_terminal_point_eur_mwh": terminal_price,
                    "fossil_price_expectation_mode": expectation_mode,
                    "fossil_price_expectation_weights_json": expectation_weights_json,
                    "last_state_year": int(state["last_state_year"]),
                    "source_state_year": int(state["source_state_year"]),
                }
            )

    log_df = pd.DataFrame(
        country_rows,
        columns=[
            "learning_seed",
            "planning_horizon",
            "fuel_type",
            "market",
            "country",
            "applied_price_eur_mwh",
            "price_terminal_point_eur_mwh",
            "fossil_price_expectation_mode",
            "fossil_price_expectation_weights_json",
            "last_state_year",
            "source_state_year",
        ],
    )
    if not log_df.empty:
        log_df = log_df.sort_values(["fuel_type", "country"]).reset_index(drop=True)
    price_dict = build_country_fuel_price_dict(
        log_df.rename(columns={"applied_price_eur_mwh": "price_eur_mwh"})
    ) if not log_df.empty else {}
    summary = {
        "fossil_price_uncertainty_enabled": True,
        "fossil_price_seed": int(seed),
        "fossil_price_learning_seed": runtime_metadata.get("learning_seed", ""),
        "fossil_price_bundle_path": runtime_metadata.get("fossil_price_bundle_path", ""),
        "fossil_price_bundle_schema_version": runtime_metadata.get("fossil_price_bundle_schema_version", ""),
        "fossil_price_bundle_sha256": runtime_metadata.get("fossil_price_bundle_sha256", ""),
        "fossil_price_expectation_mode": expectation_mode,
        "fossil_price_expectation_weights_json": expectation_weights_json,
    }
    if not log_df.empty:
        summary["fossil_price_summary_json"] = json.dumps(
            {
                fuel: {
                    "applied_min": float(group["applied_price_eur_mwh"].min()),
                    "applied_median": float(group["applied_price_eur_mwh"].median()),
                    "applied_max": float(group["applied_price_eur_mwh"].max()),
                    "terminal_min": float(group["price_terminal_point_eur_mwh"].min()),
                    "terminal_median": float(group["price_terminal_point_eur_mwh"].median()),
                    "terminal_max": float(group["price_terminal_point_eur_mwh"].max()),
                    "markets": {
                        str(market): {
                            "applied_price_eur_mwh": float(applied),
                            "price_terminal_point_eur_mwh": float(terminal),
                        }
                        for market, applied, terminal in (
                            log_df.loc[log_df["fuel_type"] == fuel]
                            .drop_duplicates(subset=["market"], keep="last")
                            .loc[:, ["market", "applied_price_eur_mwh", "price_terminal_point_eur_mwh"]]
                            .itertuples(index=False, name=None)
                        )
                    },
                }
                for fuel, group in log_df.groupby("fuel_type")
            },
            sort_keys=True,
        )
        for fuel, group in log_df.groupby("fuel_type"):
            summary[f"fossil_{fuel}_price_min_eur_mwh"] = float(group["applied_price_eur_mwh"].min())
            summary[f"fossil_{fuel}_price_median_eur_mwh"] = float(group["applied_price_eur_mwh"].median())
            summary[f"fossil_{fuel}_price_max_eur_mwh"] = float(group["applied_price_eur_mwh"].max())
            summary[f"fossil_{fuel}_price_terminal_point_median_eur_mwh"] = float(
                group["price_terminal_point_eur_mwh"].median()
            )
    else:
        summary["fossil_price_summary_json"] = json.dumps({}, sort_keys=True)
    return {
        "enabled": True,
        "log_df": log_df,
        "price_dict": price_dict,
        "states": next_states,
        "summary": summary,
        "fallback_to_static_prices": cfg["fallback_to_static_prices"],
    }


def save_fossil_price_log(fossil_payload, output_file):
    logger.info("Saving fossil price log to %s", output_file)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    log_df = pd.DataFrame()
    if fossil_payload:
        candidate = fossil_payload.get("log_df")
        if candidate is not None:
            log_df = candidate
    log_df.to_csv(output_file, index=False)


def _apply_fossil_price_updates_to_network(n, fossil_payload):
    if not fossil_payload or not fossil_payload.get("enabled"):
        return []

    price_dict = fossil_payload.get("price_dict", {}) or {}
    fallback = bool(fossil_payload.get("fallback_to_static_prices", False))
    updates_log = []
    if n.generators.empty:
        return updates_log

    for idx, row in n.generators.iterrows():
        carrier = str(row["carrier"])
        if carrier not in FOSSIL_FUELS:
            continue
        country = extract_country_code_from_bus(row["bus"])
        carrier_prices = price_dict.get(carrier, {}) or {}
        if country not in carrier_prices:
            if fallback:
                logger.warning(
                    "Fossil price mapping missing for %s/%s; retaining existing generator marginal_cost.",
                    carrier,
                    country,
                )
                continue
            raise ValueError(
                f"Missing stochastic fossil price for generator {idx} carrier={carrier} country={country}."
            )
        old_cost = float(row["marginal_cost"])
        new_cost = float(carrier_prices[country])
        n.generators.at[idx, "marginal_cost"] = new_cost
        updates_log.append(
            {
                "component": "generators",
                "index": str(idx),
                "carrier": carrier,
                "country": country,
                "old_marginal_cost": old_cost,
                "new_marginal_cost": new_cost,
            }
        )
    return updates_log


def update_network_costs(
    network_path,
    learning_costs,
    tech_mapping,
    output_path,
    learning_cfg,
    costs_file,
    learning_rates=None,
    fossil_payload=None,
):
    """
    Update network component costs in-memory and save.
    
    Only updates technologies specified in learning_costs.
    Saves updated costs to n.meta for transparency.
    
    IMPORTANT: Checks unit dimension to prevent applying energy costs (EUR/MWh-yr) 
    to power components (generators/links) or vice versa.
    
    For renewable technologies (solar, onwind, offwind), applies regional WACCs on a per-bus basis
    by recalculating capital costs using the overnight cost from learning curves
    and the region-specific WACC.
    
    Args:
        network_path: Path to input brownfield network
        learning_costs: Dict of learning-based costs per technology
        tech_mapping: Dict mapping carriers to technologies
        output_path: Path to save updated network
        learning_cfg: Learning configuration dict
        costs_file: Path to cost CSV file
        learning_rates: Optional dict of learning rates to store in metadata
        fossil_payload: Optional stochastic fossil fuel price payload
    
    Returns:
        Tuple of (network, updates_log)
    """
    logger.info(f"Loading network from {network_path}")
    n = pypsa.Network(network_path)
    
    # Reverse mapping: carrier -> tech
    carrier_to_tech = {carrier: tech for carrier, tech in tech_mapping.items()}

    # Sector model battery links are often represented as charger/discharger carriers
    # rather than "battery inverter". In prepare_sector_network, inverter CAPEX is
    # assigned to charger only, while discharger has no capital_cost.
    # Keep the same convention here to avoid double counting inverter CAPEX.
    if "battery_power" in learning_costs and not n.links.empty:
        link_carriers = set(n.links.carrier.unique())
        if "battery charger" in link_carriers:
            carrier_to_tech.setdefault("battery charger", "battery_power")
            logger.info(
                "Applying battery_power learning costs to link carrier: ['battery charger']"
            )
    
    # Map learning tech names to carrier names for regional WACC lookup
    LEARNING_TO_WACC_TECH = {
        'solar_power': 'solar',
        'onwind_power': 'onwind',
        'offwind_power': 'offwind',  # Will match both offwind-ac and offwind-dc
    }
    
    updates_log = []
    updates_count = 0
    
    # Update generators (POWER components - require kW-based technologies)
    if not n.generators.empty:
        for carrier in n.generators.carrier.unique():
            if carrier not in carrier_to_tech:
                logger.info(f"Using default costs for {carrier} - no tech mapping found")
                continue
            
            tech = carrier_to_tech[carrier]
            if tech not in learning_costs:
                logger.warning(f" Carrier '{carrier}' in tech-mapping but not in learning_costs - skipping")
                continue
            
            # Check dimensional compatibility
            pred_unit = learning_costs[tech]["unit"]
            if pred_unit != "kW":
                logger.warning(
                    f"  Skipping generators[{carrier}]: technology '{tech}' has unit='{pred_unit}' "
                    f"but generators are POWER components requiring 'kW'. Check tech_mapping."
                )
                raise ValueError("Dimensional mismatch in technology unit") # For now
                continue
            
            mask = n.generators.carrier == carrier
            if not mask.any():
                logger.warning(f"No generators found for carrier '{carrier}' - skipping")
                continue
            
            # Check if this is a renewable technology requiring regional WACC treatment
            wacc_dict = learning_costs[tech].get("wacc_dict")
            is_renewable = tech in LEARNING_TO_WACC_TECH and wacc_dict is not None
            
            if is_renewable:
                # Apply regional WACCs on a per-bus basis
                wacc_tech = LEARNING_TO_WACC_TECH[tech]
                c_overnight = learning_costs[tech]["c_overnight"]
                unit = learning_costs[tech]["unit"]
                
                # Get FOM and lifetime from config
                finance = get_learning_finance(learning_cfg)
                lifetime = finance["lifetime"][tech]
                fom = load_fom_from_costs(costs_file, tech)
                
                # Calculate capital cost for each generator based on its bus location
                # Get old cost BEFORE updating
                old_cost = n.generators.loc[mask, "capital_cost"].iloc[0] if mask.any() else np.nan
                
                new_costs = []
                for idx in n.generators.index[mask]:
                    bus = n.generators.loc[idx, "bus"]
                    # Extract country code from bus name (e.g., "NG 0" -> "NG")
                    country = bus.split()[0] if ' ' in bus else bus
                    
                    # Get regional WACC
                    regional_wacc = wacc_dict[wacc_tech].get(country)
                    if regional_wacc is None:
                        logger.warning(f"    No regional WACC for {country}, using global default")
                        regional_wacc = finance["wacc"].get(tech, finance["wacc"]["default"])
                    
                    # Calculate capital cost with regional WACC
                    annuity = calculate_annuity(lifetime, regional_wacc)
                    investment_mw = c_overnight * 1e3  # kW → MW
                    capital_cost = (annuity + fom) * investment_mw
                    new_costs.append(capital_cost)
                
                # Apply all costs
                n.generators.loc[mask, "capital_cost"] = new_costs
                new_cost_avg = np.mean(new_costs)
                
                updates_log.append({
                    "component": "generators",
                    "carrier": carrier,
                    "tech": tech,
                    "old_cost": old_cost,
                    "new_cost": new_cost_avg,
                    "regional_wacc": True,
                })
                logger.info(f"  Updated generators[{carrier}] with REGIONAL WACCs: avg={new_cost_avg:.2f} EUR/MW-yr (range: {min(new_costs):.2f}-{max(new_costs):.2f})")
            else:
                # Use global capital cost for non-renewable technologies
                new_cost = learning_costs[tech]["capital_cost"]
                old_cost = n.generators.loc[mask, "capital_cost"].iloc[0] if mask.any() else np.nan
                
                n.generators.loc[mask, "capital_cost"] = new_cost
                updates_log.append({
                    "component": "generators",
                    "carrier": carrier,
                    "tech": tech,
                    "old_cost": old_cost,
                    "new_cost": new_cost,
                })
                logger.info(f"  Updated generators[{carrier}]: {old_cost:.2f} → {new_cost:.2f} EUR/MW-yr")
            
            updates_count += 1
    
    # Update storage_units (POWER components - require kW-based technologies)
    if not n.storage_units.empty:
        for carrier in n.storage_units.carrier.unique():
            if carrier not in carrier_to_tech:
                continue
            
            tech = carrier_to_tech[carrier]
            if tech not in learning_costs:
                continue
            
            # Check dimensional compatibility
            pred_unit = learning_costs[tech]["unit"]
            if pred_unit != "kW":
                logger.warning(
                    f"  Skipping storage_units[{carrier}]: technology '{tech}' has unit='{pred_unit}' "
                    f"but storage_units are POWER components requiring 'kW'. Check tech_mapping."
                )
                raise ValueError("Dimensional mismatch in technology unit") # For now
                continue
            
            mask = n.storage_units.carrier == carrier
            if not mask.any():
                logger.warning(f"No storage_units found for carrier '{carrier}' - skipping")
                continue
            
            new_cost = learning_costs[tech]["capital_cost"]
            old_cost = n.storage_units.loc[mask, "capital_cost"].iloc[0] if mask.any() else np.nan
            
            n.storage_units.loc[mask, "capital_cost"] = new_cost
            updates_log.append({
                "component": "storage_units",
                "carrier": carrier,
                "tech": tech,
                "old_cost": old_cost,
                "new_cost": new_cost,
            })
            logger.info(f"  Updated storage_units[{carrier}]: {old_cost:.2f} → {new_cost:.2f} EUR/MW-yr")
            updates_count += 1
    
    # Update links (POWER components - require kW-based technologies)
    if not n.links.empty:
        for carrier in n.links.carrier.unique():
            if carrier not in carrier_to_tech:
                continue
            
            tech = carrier_to_tech[carrier]
            if tech not in learning_costs:
                continue
            
            # Check dimensional compatibility
            pred_unit = learning_costs[tech]["unit"]
            if pred_unit != "kW":
                logger.warning(
                    f"  Skipping links[{carrier}]: technology '{tech}' has unit='{pred_unit}' "
                    f"but links are POWER components requiring 'kW'. Check tech_mapping."
                )
                raise ValueError("Dimensional mismatch in technology unit") # For now
                continue
            
            mask = n.links.carrier == carrier
            if not mask.any():
                logger.warning(f"No links found for carrier '{carrier}' - skipping")
                continue
            
            new_cost = learning_costs[tech]["capital_cost"]
            old_cost = n.links.loc[mask, "capital_cost"].iloc[0] if mask.any() else np.nan
            
            n.links.loc[mask, "capital_cost"] = new_cost
            updates_log.append({
                "component": "links",
                "carrier": carrier,
                "tech": tech,
                "old_cost": old_cost,
                "new_cost": new_cost,
            })
            logger.info(f"  Updated links[{carrier}]: {old_cost:.2f} → {new_cost:.2f} EUR/MW-yr")
            updates_count += 1
    
    # Update stores (ENERGY components - require kWh-based technologies)
    if not n.stores.empty:
        for carrier in n.stores.carrier.unique():
            if carrier not in carrier_to_tech:
                continue
            
            tech = carrier_to_tech[carrier]
            if tech not in learning_costs:
                continue
            
            # Check dimensional compatibility
            pred_unit = learning_costs[tech]["unit"]
            if pred_unit != "kWh":
                logger.warning(
                    f"  Skipping stores[{carrier}]: technology '{tech}' has unit='{pred_unit}' "
                    f"but stores are ENERGY components requiring 'kWh'. Check tech_mapping."
                )
                continue
            
            mask = n.stores.carrier == carrier
            if not mask.any():
                continue
            
            new_cost = learning_costs[tech]["capital_cost"]
            old_cost = n.stores.loc[mask, "capital_cost"].iloc[0] if mask.any() else np.nan
            
            n.stores.loc[mask, "capital_cost"] = new_cost
            updates_log.append({
                "component": "stores",
                "carrier": carrier,
                "tech": tech,
                "old_cost": old_cost,
                "new_cost": new_cost,
            })
            logger.info(f"  Updated stores[{carrier}]: {old_cost:.2f} → {new_cost:.2f} EUR/MWh-yr")
            updates_count += 1
    
    fossil_updates_log = _apply_fossil_price_updates_to_network(n, fossil_payload)

    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        """Convert numpy types to native Python types."""
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    n.meta["learning_costs_applied"] = True
    n.meta["learning_costs"] = convert_to_native(learning_costs)
    n.meta["learning_updates_log"] = convert_to_native(updates_log)
    n.meta["fossil_prices_applied"] = bool(fossil_payload and fossil_payload.get("enabled"))
    if fossil_payload:
        n.meta["fossil_price_states"] = convert_to_native(fossil_payload.get("states", {}))
        n.meta["fossil_price_updates_log"] = convert_to_native(fossil_updates_log)
        n.meta["fossil_price_summary"] = convert_to_native(fossil_payload.get("summary", {}))
    
    # Store learning rates in metadata if provided
    if learning_rates is not None:
        n.meta["learning_rates"] = convert_to_native(learning_rates)
        logger.info("Stored learning rates in network metadata")
    
    logger.info(f"Saving updated network to {output_path}")
    logger.info(f"  Updated {updates_count} carrier types across components")
    if fossil_updates_log:
        logger.info("  Updated %s fossil fuel supply generator marginal costs", len(fossil_updates_log))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    n.export_to_netcdf(output_path)
    
    return n, updates_log


def save_cost_log(learning_costs, output_file):
    """Save applied costs and capacity data to CSV for diagnostics and transparency."""
    logger.info(f"Saving cost log to {output_file}")
    
    # Ensure directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame.from_dict(learning_costs, orient="index")
    df.index.name = "technology"
    preferred_columns = [
        "planning_horizon",
        "lag_year",
        "anchor_year",
        "anchor_source",
        "anchor_network",
        "engine",
        "selected_model",
        "model_name",
        "sample_mode",
        "seed",
        "runtime_conditioning",
        "manifest_schema_version",
        "manifest_sha256",
        "manifest_path",
        "battery_power_treatment",
        "cost_expectation_mode",
        "cost_expectation_weights_json",
        "fossil_price_uncertainty_enabled",
        "fossil_price_bundle_path",
        "fossil_price_bundle_schema_version",
        "fossil_price_bundle_sha256",
        "fossil_price_expectation_mode",
        "fossil_price_expectation_weights_json",
        "fossil_price_seed",
        "fossil_price_learning_seed",
        "fossil_oil_price_min_eur_mwh",
        "fossil_oil_price_median_eur_mwh",
        "fossil_oil_price_max_eur_mwh",
        "fossil_oil_price_terminal_point_median_eur_mwh",
        "fossil_gas_price_min_eur_mwh",
        "fossil_gas_price_median_eur_mwh",
        "fossil_gas_price_max_eur_mwh",
        "fossil_gas_price_terminal_point_median_eur_mwh",
        "fossil_coal_price_min_eur_mwh",
        "fossil_coal_price_median_eur_mwh",
        "fossil_coal_price_max_eur_mwh",
        "fossil_coal_price_terminal_point_median_eur_mwh",
        "fossil_price_summary_json",
        "cumulative_capacity_GW",
        "A_base",
        "A_scenario",
        "A",
        "beta_base",
        "beta",
        "lr_base",
        "lr_scenario",
        "beta_adjusted",
        "capital_cost",
        "capital_cost_terminal_point",
        "unit",
        "c_overnight",
        "c_overnight_terminal_point",
        "log_capex_runtime",
        "log_capex_terminal_point",
        "wacc_dict",
        "stochastic_family",
        "stochastic_current_regime",
        "stochastic_implied_learning_exponent_runtime",
        "stochastic_implied_learning_rate_runtime",
        "stochastic_learning_exponent",
        "stochastic_implied_learning_rate",
        "stochastic_parameter_snapshot",
    ]
    ordered_columns = [col for col in preferred_columns if col in df.columns]
    remaining_columns = [col for col in df.columns if col not in ordered_columns]
    df = df[ordered_columns + remaining_columns]
    df.to_csv(output_file)


def save_learning_state(state_payload, output_file):
    """Save proposed learning state for the next myopic horizon."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(state_payload, indent=2, sort_keys=True), encoding="utf-8")


def prepare_fossil_payload_for_horizon(learning_cfg, runtime_metadata, year, state_payload, learning_costs):
    fossil_payload = _build_fossil_price_payload(
        learning_cfg=learning_cfg,
        runtime_metadata=runtime_metadata,
        current_year=year,
        state_payload=state_payload,
    )
    summary = fossil_payload.get("summary", {}) or {}
    for tech_costs in learning_costs.values():
        tech_costs.update(summary)
    return fossil_payload


def get_timestep(planning_horizons, current_year):
    """
    Determine timestep (years between horizons) dynamically.
    
    Args:
        planning_horizons: List of years [2020, 2030, 2040, 2050]
        current_year: Current horizon year
    
    Returns:
        Number of years since previous horizon
    """
    horizons = sorted(planning_horizons)
    if current_year not in horizons:
        raise ValueError(f"Current year {current_year} not in planning_horizons")
    
    idx = horizons.index(current_year)
    if idx == 0:
        # First horizon - no learning applied
        return 0
    
    return current_year - horizons[idx - 1]


def get_previous_horizon(planning_horizons, current_year):
    horizons = sorted(planning_horizons)
    idx = horizons.index(current_year)
    if idx == 0:
        return None
    return horizons[idx - 1]


def use_base_learning_for_horizon(planning_horizons, current_year):
    """
    Return True when learning-rate scenarios should be suppressed for this horizon.

    Policy: always use base learning parameters for the first planning horizon
    only (e.g., 2020 in a 5-year myopic setup).
    """
    horizons = sorted(planning_horizons)
    if not horizons:
        return True
    return current_year == horizons[0]


def main(snakemake):

    configure_logging(snakemake)
    
    year = int(snakemake.wildcards.planning_horizons)
    logger.info("=" * 70)
    logger.info(f"Applying learning-based costs for horizon {year}")
    logger.info("=" * 70)
    
    learning_cfg = load_config_learning(snakemake.input.learning_config)
    load_learning_manifest(learning_cfg, snakemake.input.learning_config)
    load_fossil_price_bundle(learning_cfg, snakemake.input.learning_config)
    learning_model = getattr(snakemake.wildcards, "learning_model", None)
    learning_seed = getattr(snakemake.wildcards, "learning_seed", None)
    learning_engine = get_learning_engine(learning_cfg, learning_model)
    selected_model = get_selected_learning_model(learning_cfg, learning_model)
    resolve_runtime_seed(learning_cfg, learning_engine, selected_model, learning_seed)
    validate_runtime_contract(learning_cfg, learning_engine, selected_model)
    runtime_metadata = build_runtime_metadata(learning_cfg, learning_engine, selected_model)
    logger.info("Learning runtime: engine=%s, model=%s", learning_engine, selected_model)
    logger.info(
        "Runtime contract: sample_mode=%s, seed=%s, learning_seed=%s, battery_power=%s, conditioning=%s, cost_expectations=%s",
        runtime_metadata["sample_mode"],
        runtime_metadata["seed"],
        runtime_metadata["learning_seed"],
        BATTERY_POWER_TREATMENT,
        runtime_metadata["runtime_conditioning"],
        runtime_metadata["cost_expectation_mode"],
    )
    logger.info(
        "Learning manifest: schema_version=%s, sha256=%s",
        runtime_metadata["manifest_schema_version"],
        runtime_metadata["manifest_sha256"],
    )
    
    # Parse learning_rate wildcard and override beta_adjustment config
    learning_rate_str = None
    if hasattr(snakemake.wildcards, 'learning_rate'):
        learning_rate_str = snakemake.wildcards.learning_rate
        logger.info(f"Learning rate wildcard detected: '{learning_rate_str}'")
        if learning_engine == "legacy_curve":
            # Parse wildcard into beta adjustment config
            wildcard_beta_cfg = parse_learning_rate_wildcard(learning_rate_str)
            learning_cfg["beta_adjustment"] = wildcard_beta_cfg
            logger.info("Beta adjustment config overridden by learning_rate wildcard")
        elif learning_rate_str != "base":
            raise ValueError(
                "Non-base learning_rate scenarios are only supported for legacy_curve. "
                f"Got learning_model={selected_model}, learning_rate={learning_rate_str}."
            )
    
    # Check if learning is enabled
    if not learning_cfg.get("enabled", False):
        logger.info("Learning is DISABLED - passing through network unchanged")
        # Just copy network without modifications
        n = pypsa.Network(snakemake.input.network)
        n.meta["learning_costs_applied"] = False
        n.meta["fossil_prices_applied"] = False
        n.export_to_netcdf(snakemake.output.network)
        
        # Create empty log file
        Path(snakemake.output.cost_log).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(snakemake.output.cost_log)
        if hasattr(snakemake.output, "fossil_price_log"):
            Path(snakemake.output.fossil_price_log).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame().to_csv(snakemake.output.fossil_price_log, index=False)
        save_learning_state(
            {
                **runtime_metadata,
                "enabled": False,
                "last_applied_year": int(year),
            },
            snakemake.output.state_proposed,
        )
        return
    
    logger.info("Learning is ENABLED")
    global_scale_factors = get_global_scale_factors(learning_cfg)
    
    # Determine timestep
    planning_horizons = snakemake.params.planning_horizons
    timestep = get_timestep(planning_horizons, year)
    
    # Apply learning costs for all horizons.
    # 2020 and 2025 are deterministic historical bootstrap horizons;
    # 2030+ uses the strictly lagged solved block.
    logger.info(f"Timestep: {timestep} years since previous horizon (or first horizon if {year} == 2020)")
    
    # Load learning data
    prev_network_path = normalize_optional_input(snakemake.input.get('network_p', None))
    prev_state_path = normalize_optional_input(snakemake.input.get('prev_state', None))
    anchor_network_path = normalize_optional_input(
        snakemake.input.get('anchor_network', None)
    )
    costs_file = snakemake.input.costs
    logger.info(f"Using costs file: {costs_file}")
    prior_state_payload = {}
    if prev_state_path:
        prior_state_payload = json.loads(Path(prev_state_path).read_text(encoding="utf-8"))

    # Load regional WACCs if available
    wacc_dict = None
    if hasattr(snakemake.input, 'waccs') and snakemake.input.waccs:
        logger.info(f"Loading regional WACCs from {snakemake.input.waccs}")
        wacc_dict = load_country_waccs(snakemake.input.waccs)
        logger.info(f"  Loaded WACCs for {len(wacc_dict)} renewable technologies")
    else:
        logger.info("No regional WACC file provided - using global WACCs from config")

    if learning_engine == "stochastic_forecast":
        if year in COST_HISTORICAL_CAPACITY_YEARS:
            logger.info(
                "Using deterministic historical bootstrap for stochastic horizon %s based on historical year %s",
                year,
                COST_HISTORICAL_CAPACITY_YEARS[year],
            )
            params = load_learning_params(get_legacy_params_path(learning_cfg))
            learning_costs = calculate_learning_costs(
                params,
                learning_cfg,
                year,
                planning_horizons,
                prev_network_path=None,
                prior_state_payload=prior_state_payload,
                costs_file=costs_file,
                global_scale_factors=global_scale_factors,
                wacc_dict=wacc_dict,
            )
            _, state = load_stochastic_runtime_state(
                learning_cfg,
                selected_model,
                year,
                prev_state_path,
            )
            proposed_state = json.loads(json.dumps(state))
            proposed_state.update(runtime_metadata)
            cost_log_context = {
                **runtime_metadata,
                "planning_horizon": int(year),
                "lag_year": int(get_cost_lag_year(planning_horizons, year)),
                "anchor_year": int(get_cost_lag_year(planning_horizons, year)),
                "anchor_source": "historical_deterministic_bootstrap",
                "anchor_network": "",
            }
            for tech_costs in learning_costs.values():
                tech_costs.update(cost_log_context)
            fossil_payload = prepare_fossil_payload_for_horizon(
                learning_cfg,
                runtime_metadata,
                year,
                proposed_state,
                learning_costs,
            )
            proposed_state["fossil_price_states"] = fossil_payload.get("states", {})

            logger.info(
                "battery_power remains on deterministic default costs from costs_%s.csv in stochastic runtime",
                year,
            )
            logger.info(
                "Stochastic runtime uses deterministic historical bootstrap logic for %s",
                year,
            )

            tech_mapping = get_tech_mapping(learning_cfg)
            n, updates_log = update_network_costs(
                snakemake.input.network,
                learning_costs,
                tech_mapping,
                snakemake.output.network,
                learning_cfg,
                costs_file,
                learning_rates=None,
                fossil_payload=fossil_payload,
            )
            save_cost_log(learning_costs, snakemake.output.cost_log)
            if hasattr(snakemake.output, "fossil_price_log"):
                save_fossil_price_log(fossil_payload, snakemake.output.fossil_price_log)
            save_learning_state(proposed_state, snakemake.output.state_proposed)

            logger.info("=" * 70)
            logger.info(f"Stochastic learning cost application completed for {year}")
            logger.info("=" * 70)
            return

        if prev_network_path:
            logger.info(f"Previous network available: {prev_network_path}")
        else:
            logger.info("No previous network (first horizon: 2020)")
        learning_costs, proposed_state = calculate_stochastic_learning_costs(
            learning_cfg,
            selected_model,
            year,
            planning_horizons,
            prev_network_path,
            prev_state_path,
            costs_file,
            global_scale_factors,
            wacc_dict,
        )
        cost_log_context = {
            **runtime_metadata,
            "planning_horizon": int(year),
            "lag_year": int(get_cost_lag_year(planning_horizons, year)),
            "anchor_year": int(proposed_state.get("last_applied_year", year)),
            "anchor_source": "stochastic_runtime_state",
            "anchor_network": str(prev_network_path or ""),
        }
        for tech_costs in learning_costs.values():
            tech_costs.update(cost_log_context)
        fossil_payload = prepare_fossil_payload_for_horizon(
            learning_cfg,
            runtime_metadata,
            year,
            proposed_state,
            learning_costs,
        )
        proposed_state["fossil_price_states"] = fossil_payload.get("states", {})

        logger.info(
            "battery_power remains on deterministic default costs from costs_%s.csv in stochastic runtime",
            year,
        )
        logger.info(
            "Stochastic runtime uses %s carry-forward logic",
            runtime_metadata["runtime_conditioning"],
        )

        tech_mapping = get_tech_mapping(learning_cfg)
        n, updates_log = update_network_costs(
            snakemake.input.network,
            learning_costs,
            tech_mapping,
            snakemake.output.network,
            learning_cfg,
            costs_file,
            learning_rates=None,
            fossil_payload=fossil_payload,
        )
        save_cost_log(learning_costs, snakemake.output.cost_log)
        if hasattr(snakemake.output, "fossil_price_log"):
            save_fossil_price_log(fossil_payload, snakemake.output.fossil_price_log)
        save_learning_state(proposed_state, snakemake.output.state_proposed)

        logger.info("=" * 70)
        logger.info(f"Stochastic learning cost application completed for {year}")
        logger.info("=" * 70)
        return

    if learning_engine == "exogenous_path":
        if year in COST_HISTORICAL_CAPACITY_YEARS:
            logger.info(
                "Using deterministic historical bootstrap for exogenous horizon %s based on historical year %s",
                year,
                COST_HISTORICAL_CAPACITY_YEARS[year],
            )
            params = load_learning_params(get_legacy_params_path(learning_cfg))
            learning_costs = calculate_learning_costs(
                params,
                learning_cfg,
                year,
                planning_horizons,
                prev_network_path=None,
                prior_state_payload=prior_state_payload,
                costs_file=costs_file,
                global_scale_factors=global_scale_factors,
                wacc_dict=wacc_dict,
            )
            proposed_state = {
                **runtime_metadata,
                "last_applied_year": int(year),
                "technology_states": {},
                "capacity_history": prior_state_payload.get("capacity_history", {}),
                "modeled_capacity_history": prior_state_payload.get("modeled_capacity_history", {}),
                "fossil_price_states": prior_state_payload.get("fossil_price_states", {}),
            }
            cost_log_context = {
                **runtime_metadata,
                "planning_horizon": int(year),
                "lag_year": int(get_cost_lag_year(planning_horizons, year)),
                "anchor_year": int(get_cost_lag_year(planning_horizons, year)),
                "anchor_source": "historical_deterministic_bootstrap",
                "anchor_network": "",
            }
        else:
            learning_costs, exogenous_info = calculate_exogenous_learning_costs(
                learning_cfg=learning_cfg,
                selected_model=selected_model,
                current_year=year,
                costs_file=costs_file,
                wacc_dict=wacc_dict,
                runtime_metadata=runtime_metadata,
            )
            proposed_state = {
                **runtime_metadata,
                "last_applied_year": int(year),
                "technology_states": {},
                "capacity_history": prior_state_payload.get("capacity_history", {}),
                "modeled_capacity_history": prior_state_payload.get("modeled_capacity_history", {}),
                "exogenous_cost_path_source": exogenous_info["source"],
                "exogenous_cost_path_interpolation": exogenous_info["interpolation"],
                "fossil_price_states": prior_state_payload.get("fossil_price_states", {}),
            }
            cost_log_context = {
                **runtime_metadata,
                "planning_horizon": int(year),
                "lag_year": int(year),
                "anchor_year": int(year),
                "anchor_source": f"exogenous_cost_path:{exogenous_info['source']}",
                "anchor_network": "",
            }

        for tech_costs in learning_costs.values():
            tech_costs.update(cost_log_context)
        fossil_payload = prepare_fossil_payload_for_horizon(
            learning_cfg,
            runtime_metadata,
            year,
            proposed_state,
            learning_costs,
        )
        proposed_state["fossil_price_states"] = fossil_payload.get("states", {})

        logger.info(
            "battery_power remains on deterministic default costs from costs_%s.csv in exogenous runtime",
            year,
        )
        logger.info("Exogenous runtime ignores realized deployment for 2030+ cost formation")

        tech_mapping = get_tech_mapping(learning_cfg)
        n, updates_log = update_network_costs(
            snakemake.input.network,
            learning_costs,
            tech_mapping,
            snakemake.output.network,
            learning_cfg,
            costs_file,
            learning_rates=None,
            fossil_payload=fossil_payload,
        )
        save_cost_log(learning_costs, snakemake.output.cost_log)
        if hasattr(snakemake.output, "fossil_price_log"):
            save_fossil_price_log(fossil_payload, snakemake.output.fossil_price_log)
        save_learning_state(proposed_state, snakemake.output.state_proposed)

        logger.info("=" * 70)
        logger.info(f"Exogenous learning cost application completed for {year}")
        logger.info("=" * 70)
        return

    params_base = load_learning_params(get_legacy_params_path(learning_cfg))

    beta_cfg = learning_cfg.get("beta_adjustment", {})
    beta_adjustment_requested = beta_cfg.get("enabled", False)
    anchor_year = int(beta_cfg.get("anchor_year", 2020))
    anchor_state = pd.DataFrame(columns=["L_realized_GW"])
    anchor_state.attrs["anchor_source"] = ""
    anchor_state.attrs["anchor_network"] = ""
    force_base_for_horizon = (
        beta_adjustment_requested
        and use_base_learning_for_horizon(planning_horizons, year)
    )

    if force_base_for_horizon:
        logger.info(
            f"Ignoring learning-rate scenario adjustments for horizon {year}; "
            "the first two planning horizons always use base learning parameters."
        )
        anchor_state.attrs["anchor_source"] = "forced_base_first_two_horizons"
        params = apply_beta_adjustment(
            params_base,
            anchor_state,
            {"beta_adjustment": {"enabled": False}},
        )
    else:
        if beta_adjustment_requested:
            anchor_state = build_anchor_state(
                params_base,
                learning_cfg,
                global_scale_factors,
                anchor_year,
                anchor_network_path=anchor_network_path,
            )

        # Apply beta adjustments if enabled (recalibrates A to maintain anchor point)
        params = apply_beta_adjustment(params_base, anchor_state, learning_cfg)

    # Log learning curve approach
    if beta_adjustment_requested and not force_base_for_horizon:
        adjustment_summary = params.attrs.get("beta_adjustment_summary", {})
        logger.info(
            f"Using anchor-recalibrated learning parameters for "
            f"{adjustment_summary.get('adjusted_techs', [])}"
        )
        if get_previous_horizon(planning_horizons, year) == anchor_year:
            logger.info(
                f"{year} costs are still evaluated at anchor-year capacity {anchor_year}; "
                "visible cost differences may first appear in the next horizon"
            )
    elif beta_adjustment_requested and force_base_for_horizon:
        logger.info(
            f"Using BASE learning parameters A and β from learning_params.csv for {year} "
            "(scenario adjustments are deferred to later horizons)."
        )
    else:
        logger.info("Using BASE learning parameters A and β from learning_params.csv")

    if prev_network_path:
        logger.info(f"Previous network available: {prev_network_path}")
    else:
        logger.info("No previous network (first horizon: 2020)")
    
    logger.info(
        "Calculating learning-based costs "
        "(2020: historical 2015, 2025: historical 2020, 2030+: realized solved capacity from previous horizon)..."
    )
    
    learning_costs = calculate_learning_costs(
        params,
        learning_cfg,
        year,
        planning_horizons,
        prev_network_path,
        prior_state_payload,
        costs_file,
        global_scale_factors,
        wacc_dict,
    )

    cost_log_context = {
        **runtime_metadata,
        "planning_horizon": int(year),
        "lag_year": int(get_cost_lag_year(planning_horizons, year)),
        "anchor_year": int(anchor_year),
        "anchor_source": (
            anchor_state.attrs.get("anchor_source", "")
            if beta_cfg.get("enabled", False)
            else "disabled"
        ),
        "anchor_network": (
            anchor_state.attrs.get("anchor_network", "")
            if beta_cfg.get("enabled", False)
            else ""
        ),
    }
    for tech_costs in learning_costs.values():
        tech_costs.update(cost_log_context)
    proposed_state = {
        **runtime_metadata,
        "last_applied_year": int(year),
        "capacity_history": prior_state_payload.get("capacity_history", {}),
        "modeled_capacity_history": prior_state_payload.get("modeled_capacity_history", {}),
        "fossil_price_states": prior_state_payload.get("fossil_price_states", {}),
        "technology_states": {
            tech: {
                "capital_cost": float(values["capital_cost"]),
                "c_overnight": float(values["c_overnight"]),
                "cumulative_capacity_GW": float(values["cumulative_capacity_GW"]),
                "unit": values["unit"],
            }
            for tech, values in learning_costs.items()
        },
    }
    fossil_payload = prepare_fossil_payload_for_horizon(
        learning_cfg,
        runtime_metadata,
        year,
        proposed_state,
        learning_costs,
    )
    proposed_state["fossil_price_states"] = fossil_payload.get("states", {})

    logger.info(
        "battery_power remains on deterministic default costs from costs_%s.csv in legacy_curve runtime",
        year,
    )
    logger.info(
        "Legacy runtime uses %s update logic",
        runtime_metadata["runtime_conditioning"],
    )
    
    # Extract learning rates from params if beta adjustment was applied
    learning_rates = None
    if hasattr(params, 'attrs') and 'learning_rates' in params.attrs:
        learning_rates = params.attrs["learning_rates"]
    
    # Update network (single export with all metadata)
    tech_mapping = get_tech_mapping(learning_cfg)
    n, updates_log = update_network_costs(
        snakemake.input.network,
        learning_costs,
        tech_mapping,
        snakemake.output.network,
        learning_cfg,
        costs_file,
        learning_rates=learning_rates,
        fossil_payload=fossil_payload,
    )
    
    # Save cost log (includes costs, capacities, and all learning parameters)
    save_cost_log(learning_costs, snakemake.output.cost_log)
    if hasattr(snakemake.output, "fossil_price_log"):
        save_fossil_price_log(fossil_payload, snakemake.output.fossil_price_log)
    save_learning_state(
        proposed_state,
        snakemake.output.state_proposed,
    )
    
    logger.info("=" * 70)
    logger.info(f"Learning cost application completed for {year}")
    logger.info("=" * 70)


if __name__ == "__main__":
    if "snakemake" in globals():
        main(snakemake)
    else:
        from _helpers import mock_snakemake
        snakemake = mock_snakemake(
            "apply_learning_costs",
            simpl="",
            clusters="110",
            ll="copt",
            opts="1h",
            planning_horizons="2020",
            sopts="1h",
            configfile="/shared/share_cki25/energymodels/pypsa-earth/config.myopic.yaml",
            discountrate="0.071",
            demand="AB",
            h2export="10",
            learning_rate="base",
            learning_model="legacy_curve",
        )

        logger.warning("Running apply_learning_costs.py outside Snakemake!")

        main(snakemake)
