# -*- coding: utf-8 -*-
"""
Apply learning curve-based technology costs to PyPSA-Earth myopic networks.

This unified script handles the complete learning cost workflow:
1. Load Bayesian learning parameters (A, β) from learning_params.csv (same for all horizons)
2. Load cumulative capacity: historical or solved from previous planning horizon
3. Calculate costs using learning curve: c = A * L^(-β)
4. Update network costs in-memory
5. Save metadata and logs

Called once per horizon BEFORE solving.

WORKFLOW (Exogenous Learning):
- Flexible lag configuration via lag_periods parameter
- lag_periods=0: Immediate learning (NOT handled here, use learning.py SOS2/MILP)
- lag_periods=1: Lagged learning (handled here, exogenous LP)
  * 2020: Uses 2018 historical capacity (2-year historical lag)
  * 2025: Uses 2020 solved capacity 
  * 2030: Uses 2025 solved capacity
  * 2035: Uses 2030 solved capacity
  * And so on...

Mathematical Framework:
- Learning curve: c_k(L) = A_k * L^(-β_k)
- For 2020: L from historical CSV (2018 data)
- For 2025+: L from previous horizon's solved network
- Capital cost: (annuity + FOM) * investment, where FOM is decimal (e.g., 0.01578)

Global vs. Modeled Deployment:
- Learning curves use GLOBAL cumulative deployment (L) from historical data
- Model only captures fraction of global deployment (e.g., 5.63% for battery storage in 2020)
- Costs reflect global learning dynamics, applied uniformly to modeled capacity

Units Consistency:
- Power technologies (solar, wind, battery_power, electrolyser): MW/GW (unit='kW')
- Energy technologies (battery_energy, h2_energy): MWh/GWh (unit='kWh')
- Validation enforced to prevent mismatches

Created: 2025-11-06
Updated: 2025-12-18
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import yaml
from _helpers import configure_logging, create_logger
from forecast_deployment import (
    get_historical_datafile,
    load_historical_capacity,
)

# Battery Balance of System (BOS) multiplier
# Learning happens at cell/pack level (USD historical data), but model uses system costs (EUR)
# BOS_multiplier converts cell costs to full system costs including inverters, installation, etc.
# Calibrated to 2020: 246.7088 EUR/kWh (system) / (137 USD/kWh (cell) / 1.14 USD/EUR) = 2.054
BOS_multiplier = 246.7088 / (137 / 1.14)

# Global scaling factors for technologies where model covers only a fraction of global deployment
# Learning happens globally, so we scale modeled capacity to represent global deployment
# Values can be overridden in config.learning.yaml via:
# learning.global_scale_factors.<technology>
DEFAULT_GLOBAL_SCALE_FACTORS = {
    "battery_energy": 17.76,  # Model covers 66/1173 = 5.63% of global battery storage (2020)
    "battery_power": 17.76,   # Apply same global scaling to battery inverter learning
}

# Mapping of horizon year to historical capacity year for exogenous cost updates.
# This is separate from the anchor-year reconstruction used for beta adjustment.
COST_HISTORICAL_CAPACITY_YEARS = {
    2020: 2018,
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
        tech_mapping = learning_cfg.get("tech_mapping", {})
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

            hist_file = get_historical_datafile(tech)
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
    hist_file = get_historical_datafile(tech)
    if hist_file is None:
        raise ValueError(
            f"No historical data file configured for {tech}. "
            f"Check TECH_TO_DATAFILE mapping in forecast_deployment.py"
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


def extract_capacity_from_network(network_path, tech_mapping, global_scale_factors):
    """
    Extract realized installed capacity from solved network.
    
    Sums p_nom_opt (for generators, links) or e_nom_opt (for stores) by carrier.
    Converts carrier names to technology keys using tech_mapping.
    Applies global scaling factors where model covers only a fraction of deployment.
    
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
    
    # Apply global scaling factors for technologies that represent only a fraction of global deployment
    for tech in list(realized.keys()):
        if tech in global_scale_factors:
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
      * For 2020: Uses 2018 historical capacity (2-year historical lag)
      * For 2025: Uses 2020 solved capacity
      * For 2030: Uses 2025 solved capacity
      * And so on...
    
    Args:
        params: DataFrame with learning parameters (A, β) from learning_params.csv
        learning_cfg: Learning configuration dict (must include lag_periods)
        current_year: Current planning horizon year
        planning_horizons: List of all planning horizons
        prev_network_path: Path to previous solved network (for extracting capacity)
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

    # For 2025+, extract lagged realized capacities once to avoid repeated imports/log spam.
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
        tech_mapping = learning_cfg.get("tech_mapping", {})
        realized_capacity = extract_capacity_from_network(
            prev_network_path, tech_mapping, global_scale_factors
        )
    
    learning_costs = {}
    
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
        
        # Load cumulative capacity: historical (2020) or solved based on lag_year
        if current_year in COST_HISTORICAL_CAPACITY_YEARS:
            # For 2020: Use historical capacity data
            historical_year = COST_HISTORICAL_CAPACITY_YEARS[current_year]
            logger.info(f"    Using {historical_year} historical capacity (fixed 2-year historical lag)")
            
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
            
            # Use the solved capacity from lag_year
            cumulative_capacity = realized_capacity[tech]
            capacity_unit = "GWh" if tech in ENERGY_TECHS else "GW"
            logger.info(f"    Solved capacity from {lag_year}: {cumulative_capacity:.1f} {capacity_unit}")
        
        # Validate numerical parameters
        try:
            validate_learning_parameters(tech, A, beta, learning_rate, cumulative_capacity)
        except ValueError as e:
            logger.error(f"    Numerical validation failed: {e}")
            raise
        
        # Calculate overnight cost using learning curve: c = A * L^(-β)
        c_overnight_cell = A * (cumulative_capacity ** (-beta))
        
        # For batteries, apply BOS multiplier (learning at cell level, costs at system level)
        if tech == 'battery_energy':
            c_overnight = c_overnight_cell * BOS_multiplier
            logger.info(f"    Battery cell cost: {c_overnight_cell:.2f} EUR/{unit}, system cost (×{BOS_multiplier:.4f}): {c_overnight:.2f} EUR/{unit}")
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
    finance = learning_cfg["finance"]
    
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


def update_network_costs(network_path, learning_costs, tech_mapping, output_path, learning_cfg, costs_file, learning_rates=None):
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
                finance = learning_cfg["finance"]
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
    
    # Store learning rates in metadata if provided
    if learning_rates is not None:
        n.meta["learning_rates"] = convert_to_native(learning_rates)
        logger.info("Stored learning rates in network metadata")
    
    logger.info(f"Saving updated network to {output_path}")
    logger.info(f"  Updated {updates_count} carrier types across components")
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
        "unit",
        "c_overnight",
        "wacc_dict",
    ]
    ordered_columns = [col for col in preferred_columns if col in df.columns]
    remaining_columns = [col for col in df.columns if col not in ordered_columns]
    df = df[ordered_columns + remaining_columns]
    df.to_csv(output_file)


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

    Policy: always use base learning parameters for the first two planning horizons
    (e.g., 2020 and 2025 in a 5-year myopic setup).
    """
    horizons = sorted(planning_horizons)
    if len(horizons) < 2:
        return True
    return current_year in set(horizons[:2])


def main(snakemake):

    configure_logging(snakemake)
    
    year = int(snakemake.wildcards.planning_horizons)
    logger.info("=" * 70)
    logger.info(f"Applying learning-based costs for horizon {year}")
    logger.info("=" * 70)
    
    learning_cfg = load_config_learning(snakemake.input.learning_config)
    
    # Parse learning_rate wildcard and override beta_adjustment config
    if hasattr(snakemake.wildcards, 'learning_rate'):
        learning_rate_str = snakemake.wildcards.learning_rate
        logger.info(f"Learning rate wildcard detected: '{learning_rate_str}'")
        
        # Parse wildcard into beta adjustment config
        wildcard_beta_cfg = parse_learning_rate_wildcard(learning_rate_str)
        
        # Override the beta_adjustment section in learning_cfg
        learning_cfg["beta_adjustment"] = wildcard_beta_cfg
        logger.info("Beta adjustment config overridden by learning_rate wildcard")
    
    # Check if learning is enabled
    if not learning_cfg.get("enabled", False):
        logger.info("Learning is DISABLED - passing through network unchanged")
        # Just copy network without modifications
        n = pypsa.Network(snakemake.input.network)
        n.meta["learning_costs_applied"] = False
        n.export_to_netcdf(snakemake.output.network)
        
        # Create empty log file
        Path(snakemake.output.cost_log).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(snakemake.output.cost_log)
        return
    
    logger.info("Learning is ENABLED")
    global_scale_factors = get_global_scale_factors(learning_cfg)
    
    # Determine timestep
    planning_horizons = snakemake.params.planning_horizons
    timestep = get_timestep(planning_horizons, year)
    
    # NEW WORKFLOW: Apply learning costs for ALL horizons including 2020
    # 2020 uses 2018 historical data, 2025+ uses solved capacity from previous horizon
    logger.info(f"Timestep: {timestep} years since previous horizon (or first horizon if {year} == 2020)")
    
    # Load learning data
    params_base = load_learning_params(snakemake.input.params)

    prev_network_path = normalize_optional_input(snakemake.input.get('network_p', None))
    anchor_network_path = normalize_optional_input(
        snakemake.input.get('anchor_network', None)
    )

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
    
    logger.info("Calculating learning-based costs (2020: historical 2018, 2025+: solved capacity from previous horizon)...")
    
    # Get costs file for this year
    costs_file = snakemake.input.costs
    logger.info(f"Using costs file: {costs_file}")
    
    # Load regional WACCs if available
    wacc_dict = None
    if hasattr(snakemake.input, 'waccs') and snakemake.input.waccs:
        logger.info(f"Loading regional WACCs from {snakemake.input.waccs}")
        wacc_dict = load_country_waccs(snakemake.input.waccs)
        logger.info(f"  Loaded WACCs for {len(wacc_dict)} renewable technologies")
    else:
        logger.info("No regional WACC file provided - using global WACCs from config")
    
    learning_costs = calculate_learning_costs(
        params,
        learning_cfg,
        year,
        planning_horizons,
        prev_network_path,
        costs_file,
        global_scale_factors,
        wacc_dict,
    )

    cost_log_context = {
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
    
    # Extract learning rates from params if beta adjustment was applied
    learning_rates = None
    if hasattr(params, 'attrs') and 'learning_rates' in params.attrs:
        learning_rates = params.attrs["learning_rates"]
    
    # Update network (single export with all metadata)
    tech_mapping = learning_cfg["tech_mapping"]
    n, updates_log = update_network_costs(
        snakemake.input.network,
        learning_costs,
        tech_mapping,
        snakemake.output.network,
        learning_cfg,
        costs_file,
        learning_rates=learning_rates
    )
    
    # Save cost log (includes costs, capacities, and all learning parameters)
    save_cost_log(learning_costs, snakemake.output.cost_log)
    
    logger.info("=" * 70)
    logger.info(f"Exogenous learning cost application completed for {year}")
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
            learning_rate="base"
        )

        logger.warning("Running apply_learning_costs.py outside Snakemake!")

        main(snakemake)
