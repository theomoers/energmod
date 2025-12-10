# -*- coding: utf-8 -*-
"""
Apply learning curve-based technology costs to PyPSA-Earth myopic networks.

This unified script handles the complete learning cost workflow:
1. Load Bayesian learning parameters (A, β) from learning_params.csv (same for all horizons)
2. Forecast deployment using log-linear regression on historical + realized data
3. Calculate predicted costs: c = A * L^(-β)
4. Update network costs in-memory
5. Save metadata and logs

Called once per horizon BEFORE solving.

Mathematical Framework:
- Learning curve: c_k(L) = A_k * L^(-β_k)
- Forecast: log-linear regression with guardrails (theta, gamma)
- Predicted cumulative: L̂_t = L_{t-Δ} + ΔK̂_t
- Capital cost: (annuity + FOM) * investment, where FOM is decimal (e.g., 0.01578)

Created: 2025-11-06
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import yaml
from _helpers import configure_logging, create_logger
from forecast_deployment import (
    forecast_deployment_loglinear,
    get_historical_datafile,
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
    
    Format: tech.method-value or "base" for no adjustment
    Examples:
        - "base": No adjustment
        - "s.m1.2": Solar multiplier ×1.2
        - "w.o-0.1": Wind offset -0.1
        - "all.m1.5": All technologies multiplier ×1.5
        - "b.m0.8_w.m1.2": Battery multiplier ×0.8 AND wind multiplier ×1.2
    
    Tech codes:
        - s: solar_power
        - w: onwind_power (wind)
        - b: battery_energy
        - bi: battery_power (battery inverter)
        - e: electrolyser_power
        - h: h2_energy
        - all: all technologies
    
    Method codes:
        - m: multiplier (multiply base β)
        - o: offset (add to base β)
    
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
    
    for adj in adjustments:
        # Parse format: tech.method-value
        # E.g., "s.m1.2" -> tech="s", method="m", value="1.2"
        if '.' not in adj:
            logger.warning(f"Invalid learning rate format: {adj}, skipping")
            continue
        
        parts = adj.split('.')
        if len(parts) < 2:
            logger.warning(f"Invalid learning rate format: {adj}, skipping")
            continue
        
        tech_code = parts[0]
        method_value = '.'.join(parts[1:])  # Rejoin in case value has decimal point
        
        # Parse method and value (e.g., "m1.2" or "o-0.1")
        if method_value.startswith('m'):
            method = 'multiplier'
            value_str = method_value[1:]
        elif method_value.startswith('o'):
            method = 'offset'
            value_str = method_value[1:]
        else:
            logger.warning(f"Invalid method in {adj}, expected 'm' or 'o'")
            continue
        
        try:
            value = float(value_str)
        except ValueError:
            logger.warning(f"Invalid numeric value in {adj}: {value_str}")
            continue
        
        # Apply to appropriate tech
        if tech_code == 'all':
            global_method = method
            global_value = value
        elif tech_code in tech_map:
            tech_name = tech_map[tech_code]
            tech_specific[tech_name] = value
        else:
            logger.warning(f"Unknown tech code: {tech_code}")
    
    # Build config dict
    if global_method:
        config = {
            "enabled": True,
            "method": global_method,
            "multiplier": global_value if global_method == "multiplier" else 1.0,
            "offset": global_value if global_method == "offset" else 0.0,
            "tech_specific": tech_specific,
        }
    else:
        # Only tech-specific adjustments, default to multiplier method
        config = {
            "enabled": True,
            "method": "multiplier",
            "multiplier": 1.0,
            "offset": 0.0,
            "tech_specific": tech_specific,
        }
    
    logger.info(f"Parsed learning_rate wildcard '{learning_rate_str}':")
    logger.info(f"  Enabled: {config['enabled']}")
    logger.info(f"  Method: {config['method']}")
    if global_method:
        logger.info(f"  Global {config['method']}: {global_value}")
    if tech_specific:
        logger.info(f"  Tech-specific: {tech_specific}")
    
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


def load_investment_cost_from_costs(costs_file, tech_key):
    """
    Load investment (overnight) cost from cost CSV file.
    
    Args:
        costs_file: Path to costs CSV (e.g., costs_2020.csv)
        tech_key: Technology key (e.g., 'solar_power')
    
    Returns:
        Investment cost in EUR/kW or EUR/kWh
    
    Raises:
        FileNotFoundError: If cost file doesn't exist
        ValueError: If investment cost data not found for technology
    """
    if not Path(costs_file).exists():
        raise FileNotFoundError(f"Cost file not found: {costs_file}")
    
    # Map to cost file technology name
    cost_tech_name = TECH_TO_COST_NAME.get(tech_key)
    if cost_tech_name is None:
        raise ValueError(f"No cost file mapping for {tech_key} in TECH_TO_COST_NAME")
    
    try:
        df = pd.read_csv(costs_file)
        
        # Filter for this technology and investment parameter
        inv_data = df[(df['technology'] == cost_tech_name) & (df['parameter'] == 'investment')]
        
        if inv_data.empty:
            raise ValueError(
                f"No investment cost data found for technology '{cost_tech_name}' in {costs_file}. "
                f"Check that the cost file contains investment parameter for this technology."
            )
        
        inv_value = inv_data.iloc[0]['value']
        unit = inv_data.iloc[0]['unit']
        
        # Convert to EUR/kW or EUR/kWh
        if 'EUR/kW' in unit:
            inv_cost = inv_value
        elif 'USD/kW' in unit:
            # Assuming EUR/USD ~ 1.0 for simplicity, adjust if needed
            inv_cost = inv_value
        else:
            raise ValueError(
                f"Unexpected investment unit for {cost_tech_name}: {unit}. Expected 'EUR/kW' or similar."
            )
        
        logger.debug(f"  Loaded investment cost for {tech_key}: {inv_cost:.2f} {unit}")
        return inv_cost
        
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        raise ValueError(f"Error loading investment cost for {tech_key} from {costs_file}: {e}")


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
        return params_adjusted
    
    logger.info("Beta adjustment ENABLED")
    method = beta_cfg.get("method", "multiplier")
    multiplier = beta_cfg.get("multiplier", 1.0)
    offset = beta_cfg.get("offset", 0.0)
    tech_specific = beta_cfg.get("tech_specific", {})
    anchor_year = beta_cfg.get("anchor_year", 2020)
    
    logger.info(f"  Method: {method}")
    logger.info(f"  Global multiplier: {multiplier}")
    logger.info(f"  Global offset: {offset}")
    logger.info(f"  Anchor year: {anchor_year}")
    
    params_adjusted = params.copy()
    params_adjusted["beta_base"] = params_adjusted["beta"]  # Store original beta
    
    # Dictionary to store learning rates for metadata
    learning_rates = {}
    
    for tech in params_adjusted.index:
        A_base = params_adjusted.loc[tech, "A_over_unit"]
        beta_base = params_adjusted.loc[tech, "beta"]
        
        # Get anchor deployment
        if tech not in state_anchor.index:
            logger.warning(f"  {tech} not in anchor state, skipping beta adjustment")
            continue
        
        L_anchor = state_anchor.loc[tech, "L_realized_GW"]
        
        # Determine beta adjustment for this technology
        if tech in tech_specific:
            if method == "multiplier":
                beta_scenario = beta_base * tech_specific[tech]
                logger.info(f"  {tech}: β × {tech_specific[tech]:.2f} = {beta_scenario:.4f}")
            else:  # offset
                beta_scenario = beta_base + tech_specific[tech]
                logger.info(f"  {tech}: β + {tech_specific[tech]:.4f} = {beta_scenario:.4f}")
        else:
            if method == "multiplier":
                beta_scenario = beta_base * multiplier
            else:  # offset
                beta_scenario = beta_base + offset
            
            if multiplier != 1.0 or offset != 0.0:
                logger.info(f"  {tech}: β_base={beta_base:.4f} → β_scenario={beta_scenario:.4f}")
        
        # Calculate learning rates: LR = 1 - 2^(-β)
        lr_base = 1 - 2**(-beta_base)
        lr_scenario = 1 - 2**(-beta_scenario)
        
        # Print learning rates
        if beta_base != beta_scenario:
            logger.info(f"  {tech}: LR = {lr_scenario:.1%} (base: {lr_base:.1%})")
        
        # Store learning rates for metadata
        learning_rates[tech] = {
            "beta_base": float(beta_base),
            "beta_scenario": float(beta_scenario),
            "lr_base": float(lr_base),
            "lr_scenario": float(lr_scenario)
        }
        
        # Recalibrate A to maintain anchor point
        A_scenario = recalibrate_A_for_beta_scenario(
            A_base, beta_base, beta_scenario, L_anchor, tech
        )
        
        # Update parameters
        params_adjusted.loc[tech, "beta"] = beta_scenario
        params_adjusted.loc[tech, "A_over_unit"] = A_scenario
    
    # Store learning rates in the params_adjusted for passing to main
    params_adjusted.attrs["learning_rates"] = learning_rates
    
    return params_adjusted


def validate_learning_parameter_A(params, state_2020, costs_2020_file, learning_cfg, tolerance=0.15):
    """
    Validate that learning parameters predict 2020 investment costs correctly.
    
    Uses learning curve c = A × L^(-β) with 2020 deployment to predict costs,
    then compares with actual 2020 investment costs from cost CSV.
    
    Args:
        params: DataFrame with learning parameters (A_over_unit, beta, unit)
        state_2020: DataFrame with 2020 deployment state (L_realized_GW)
        costs_2020_file: Path to 2020 cost CSV file
        learning_cfg: Learning configuration dict
        tolerance: Acceptable relative difference (default 15%)
    
    Raises:
        AssertionError: If predicted cost differs from 2020 investment cost by more than tolerance
    """
    logger.info(f"Loading 2020 costs from {costs_2020_file}")
    
    for tech in params.index:
        A = params.loc[tech, "A_over_unit"]
        beta = params.loc[tech, "beta"]
        unit = params.loc[tech, "unit"]
        
        # Get 2020 cumulative deployment
        if tech not in state_2020.index:
            logger.warning(f"  {tech} not found in 2020 state, skipping validation")
            continue
        
        L_2020 = state_2020.loc[tech, "L_realized_GW"]
        
        # Load investment cost from 2020 cost file
        try:
            c_investment_2020 = load_investment_cost_from_costs(costs_2020_file, tech)
        except Exception as e:
            logger.warning(f"  Could not load investment cost for {tech} from 2020 costs: {e}")
            logger.warning(f"  Skipping validation for {tech}")
            continue
        
        # Calculate predicted cost using learning curve: c = A × L^(-β)
        c_predicted_2020 = A * (L_2020 ** (-beta))
        
        # Compare predicted cost with actual 2020 investment cost
        rel_diff = abs(c_predicted_2020 - c_investment_2020) / c_investment_2020
        
        # Check if beta was adjusted
        beta_base = params.loc[tech, "beta_base"] if "beta_base" in params.columns else beta
        beta_adjusted = beta != beta_base
        
        logger.info(f"  {tech}:")
        logger.info(f"    2020 deployment:    {L_2020:.1f} GW")
        if beta_adjusted:
            logger.info(f"    A parameter:        {A:.2f} EUR/{unit} (recalibrated)")
            logger.info(f"    β parameter:        {beta:.4f} (adjusted from {beta_base:.4f})")
        else:
            logger.info(f"    A parameter:        {A:.2f} EUR/{unit}")
            logger.info(f"    β parameter:        {beta:.4f}")
        logger.info(f"    Predicted cost:     {c_predicted_2020:.2f} EUR/{unit}")
        logger.info(f"    2020 investment:    {c_investment_2020:.2f} EUR/{unit}")
        logger.info(f"    Relative diff:      {rel_diff*100:.1f}%")
        
        if rel_diff > tolerance:
            raise AssertionError(
                f"Learning curve prediction for {tech} differs from 2020 investment cost by {rel_diff*100:.1f}%\n"
                f"  Predicted (A × L^(-β)): {c_predicted_2020:.2f} EUR/{unit}\n"
                f"  2020 investment cost:   {c_investment_2020:.2f} EUR/{unit}\n"
                f"  Parameters: A={A:.2f}, β={beta:.3f}, L_2020={L_2020:.1f} GW\n"
                f"  Tolerance = {tolerance*100:.1f}%\n"
                f"  This suggests learning parameters may not correctly represent 2020 costs. "
                f"Check learning parameter calibration or 2020 cost file."
            )
        else:
            logger.info(f"    PASS (within {tolerance*100:.0f}% tolerance)")


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


def load_learning_state(state_file):
    """
    Load realized deployment state from previous horizon.
    
    Args:
        state_file: Path to learning_state_{prev_year}.csv
    
    Returns:
        DataFrame with columns: L_realized_GW, K_realized_GW, K_prev_GW
    """
    logger.info(f"Loading learning state from {state_file}")
    state = pd.read_csv(state_file, index_col="technology")
    
    required_cols = ["L_realized_GW", "K_realized_GW", "K_prev_GW"]
    missing = set(required_cols) - set(state.columns)
    if missing:
        raise ValueError(f"Learning state missing columns: {missing}")
    
    logger.info(f"  Loaded state for {len(state)} technologies")
    return state


def load_realized_capacity_history(state_file_path, tech, current_year, planning_horizons):
    """
    Load realized capacity history from all previous learning state files.
    
    This accumulates realized capacity data points from all solved horizons
    up to (but not including) current_year to use for forecasting.
    
    Args:
        state_file_path: Path to the current state file (used as template for pattern)
        tech: Technology key
        current_year: Current planning horizon (exclude this year)
        planning_horizons: List of all planning horizons
    
    Returns:
        DataFrame with columns ['year', 'capacity_GW'] for all realized data points,
        or None if no previous data available
    """
    horizons = sorted([h for h in planning_horizons if h < current_year])
    
    if not horizons:
        logger.info(f"    No previous horizons for {tech} - using historical data only")
        return None
    
    realized_data = []
    
    # Extract the pattern from the state file path
    # E.g., learning_state_elec_s_110_lcopt_1h_1h_2025_0.071_AB_10export.csv
    # We need to replace the year (2025) with previous years
    state_path = Path(state_file_path)
    state_dir = state_path.parent
    state_basename = state_path.name
    
    for year in horizons:
        # Replace the year in the filename
        # Find and replace the current year pattern with the target year
        state_filename = state_basename.replace(f"_{current_year}_", f"_{year}_")
        state_file = state_dir / state_filename
        
        if not state_file.exists():
            logger.warning(f"    State file not found: {state_file}")
            continue
        
        state = pd.read_csv(state_file, index_col="technology")
        
        if tech not in state.index:
            logger.warning(f"    {tech} not in state file for {year}")
            continue
        
        # Extract realized capacity at this year
        K_realized = state.loc[tech, "K_realized_GW"]
        realized_data.append({'year': year, 'capacity_GW': K_realized})
    
    if not realized_data:
        return None
    
    df = pd.DataFrame(realized_data)
    logger.info(f"    Loaded {len(df)} realized data points for {tech}: "
                f"{df['year'].min():.0f}–{df['year'].max():.0f}")
    
    return df


def calculate_predicted_costs(params, state, learning_cfg, current_year, planning_horizons, state_file_path, costs_file, wacc_dict=None):
    """
    Calculate predicted costs for all technologies at current horizon.
    
    Uses log-linear (exponential) regression on full historical data plus
    realized data from previous model horizons.
    
    Args:
        params: DataFrame with learning parameters (A, β) from learning_params.csv
        state: DataFrame with previous deployment state (for base year reference)
        learning_cfg: Learning configuration dict
        current_year: Current planning horizon year
        planning_horizons: List of all planning horizons
        state_file_path: Path to state file (for pattern matching)
        costs_file: Path to cost CSV file for this year
        wacc_dict: Optional dict of regional WACCs {tech: {country: wacc}}
    
    Returns:
        Dict of {tech: {capital_cost, L_pred, dK_pred, ...}}
    """
    # Hardcoded deployment values for 2025 (known values)
    HARDCODED_2025_DEPLOYMENT = {
        'solar_power': 2130.64,  # GW
        'onwind_power': 1136.0,  # GW
        'battery_energy': 2400.0,  # GWh
    }
    
    guardrails_cfg = learning_cfg.get("guardrails", {})
    use_theta = guardrails_cfg.get("use_theta", True)
    use_gamma = guardrails_cfg.get("use_gamma", True)
    theta = guardrails_cfg.get("theta", 1.0)
    gamma_default = guardrails_cfg.get("gamma_default", 0.6)
    gamma_map = guardrails_cfg.get("gamma", {}) if use_gamma else {}
    
    # Determine base year (previous horizon or historical baseline)
    horizons = sorted([h for h in planning_horizons if h < current_year])
    base_year = horizons[-1] if horizons else 2023
    
    predictions = {}
    
    for tech in params.index:
        logger.info(f"  Processing {tech}...")
        
        # Get parameters
        A = params.loc[tech, "A_over_unit"]
        beta = params.loc[tech, "beta"]
        unit = params.loc[tech, "unit"]
        gamma = gamma_map.get(tech, gamma_default) if use_gamma else None  # Deployment cap per tech
        
        # Check if we have hardcoded 2025 data for this technology
        if current_year == 2025 and tech in HARDCODED_2025_DEPLOYMENT:
            L_pred = HARDCODED_2025_DEPLOYMENT[tech] * 1.1
            
            if tech not in state.index:
                raise ValueError(
                    f"Technology {tech} not found in state file but is required for 2025 hardcoded forecast. "
                    f"Check that learning state file contains {tech}."
                )
            
            K_base = state.loc[tech, "K_realized_GW"]
            L_base = state.loc[tech, "L_realized_GW"]
            dK_pred = L_pred - K_base
            
            logger.info(f"    Using HARDCODED 2025 deployment: L={L_pred:.1f}GW")
        else:
            # Get historical data file
            hist_file = get_historical_datafile(tech)
            if hist_file is None:
                raise ValueError(
                    f"No historical data file configured for {tech}. "
                    f"Check TECH_TO_DATAFILE mapping in forecast_deployment.py"
                )
            
            # Load realized capacity history from previous horizons
            realized_data = load_realized_capacity_history(
                state_file_path, tech, current_year, planning_horizons
            )
            
            # Forecast using log-linear regression
            try:
                forecast_results = forecast_deployment_loglinear(
                    historical_file=hist_file,
                    tech_name=tech,
                    target_year=current_year,
                    base_year=base_year,
                    realized_data=realized_data,
                    theta=theta,
                    gamma=gamma,
                    min_data_points=3,
                    use_theta=use_theta,
                    use_gamma=use_gamma
                )
                
                # Extract forecast values
                K_base = forecast_results['K_base_GW']
                L_pred = forecast_results['L_forecast_GW']
                dK_pred = forecast_results['dK_forecast_GW']
                L_base = K_base  # In this formulation, L = cumulative = K
                
            except Exception as e:
                logger.error(f"    Forecast failed for {tech}: {e}")
                raise ValueError(
                    f"Failed to forecast deployment for {tech}: {e}"
                ) from e
        
        # Calculate predicted overnight cost (EUR/kW or EUR/kWh)
        c_overnight_pred = A * (L_pred ** (-beta))
        
        # Convert to capital_cost (EUR/MW-yr or EUR/MWh-yr)
        # NOTE: For renewable technologies, we calculate a global average capital cost here
        # The actual regional WACCs will be applied in update_network_costs() on a per-bus basis
        capital_cost = convert_to_capital_cost(
            c_overnight_pred, tech, unit, learning_cfg, costs_file
        )
        
        # Store both adjusted and base parameters for transparency
        beta_base = params.loc[tech, "beta_base"] if "beta_base" in params.columns else beta
        
        predictions[tech] = {
            "capital_cost": capital_cost,
            "L_pred_GW": L_pred,
            "dK_pred_GW": dK_pred,
            "L_base_GW": L_base,
            "K_base_GW": K_base,
            "A": A,
            "beta": beta,
            "beta_base": beta_base,
            "beta_adjusted": beta != beta_base,
            "unit": unit,
            "c_overnight_pred": c_overnight_pred,
            "wacc_dict": wacc_dict,  # Store for per-bus calculation
        }
        
        # Determine capital cost unit for logging
        cap_cost_unit = "EUR/MW-yr" if unit == "kW" else "EUR/MWh-yr"
        
        logger.info(
            f"  {tech}: L_base={L_base:.1f}GW, L_pred={L_pred:.1f}GW (Δ={dK_pred:.1f}), "
            f"c_overnight={c_overnight_pred:.2f}EUR/{unit}, "
            f"capital_cost={capital_cost:.2f}{cap_cost_unit}"
        )
    
    return predictions


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


def update_network_costs(network_path, predictions, tech_mapping, output_path, learning_cfg, costs_file, learning_rates=None):
    """
    Update network component costs in-memory and save.
    
    Only updates technologies specified in predictions.
    Saves updated costs to n.meta for transparency.
    
    IMPORTANT: Checks unit dimension to prevent applying energy costs (EUR/MWh-yr) 
    to power components (generators/links) or vice versa.
    
    For renewable technologies (solar, onwind, offwind), applies regional WACCs on a per-bus basis
    by recalculating capital costs using the overnight cost from learning curve predictions
    and the region-specific WACC.
    
    Args:
        network_path: Path to input brownfield network
        predictions: Dict of predicted costs per technology
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
            if tech not in predictions:
                logger.warning(f" Carrier '{carrier}' in tech-mapping but not in predictions - skipping")
                continue
            
            # Check dimensional compatibility
            pred_unit = predictions[tech]["unit"]
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
            wacc_dict = predictions[tech].get("wacc_dict")
            is_renewable = tech in LEARNING_TO_WACC_TECH and wacc_dict is not None
            
            if is_renewable:
                # Apply regional WACCs on a per-bus basis
                wacc_tech = LEARNING_TO_WACC_TECH[tech]
                c_overnight = predictions[tech]["c_overnight_pred"]
                unit = predictions[tech]["unit"]
                
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
                new_cost = predictions[tech]["capital_cost"]
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
            if tech not in predictions:
                continue
            
            # Check dimensional compatibility
            pred_unit = predictions[tech]["unit"]
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
            
            new_cost = predictions[tech]["capital_cost"]
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
            if tech not in predictions:
                continue
            
            # Check dimensional compatibility
            pred_unit = predictions[tech]["unit"]
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
            
            new_cost = predictions[tech]["capital_cost"]
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
            if tech not in predictions:
                continue
            
            # Check dimensional compatibility
            pred_unit = predictions[tech]["unit"]
            if pred_unit != "kWh":
                logger.warning(
                    f"  Skipping stores[{carrier}]: technology '{tech}' has unit='{pred_unit}' "
                    f"but stores are ENERGY components requiring 'kWh'. Check tech_mapping."
                )
                continue
            
            mask = n.stores.carrier == carrier
            if not mask.any():
                continue
            
            new_cost = predictions[tech]["capital_cost"]
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
    n.meta["learning_predictions"] = convert_to_native(predictions)
    n.meta["learning_updates_log"] = convert_to_native(updates_log)
    
    # Store learning rates in metadata if provided
    if learning_rates is not None:
        n.meta["learning_rates"] = convert_to_native(learning_rates)
        logger.info("Stored learning rates in network metadata")
    
    logger.info(f"Saving updated network to {output_path}")
    logger.info(f"  Updated {updates_count} carrier types across components")
    n.export_to_netcdf(output_path)
    
    return n, updates_log


def save_cost_log(predictions, output_file):
    """Save predicted costs and forecasts to CSV for diagnostics."""
    logger.info(f"Saving cost log to {output_file}")
    
    # Ensure directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame.from_dict(predictions, orient="index")
    df.index.name = "technology"
    df.to_csv(output_file)


def save_prediction_state(predictions, output_file):
    """Save detailed prediction state for diagnostics."""
    logger.info(f"Saving prediction state to {output_file}")
    
    # Ensure directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    data = []
    for tech, pred in predictions.items():
        data.append({
            "technology": tech,
            "L_base_GW": pred["L_base_GW"],
            "K_base_GW": pred["K_base_GW"],
            "dK_pred_GW": pred["dK_pred_GW"],
            "L_pred_GW": pred["L_pred_GW"],
        })
    
    if data:
        pd.DataFrame(data).set_index("technology").to_csv(output_file)
    else:
        # No predictions - create empty DataFrame with correct columns
        pd.DataFrame(columns=["L_base_GW", "K_base_GW", "dK_pred_GW", "L_pred_GW"]).to_csv(output_file, index_label="technology")


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
        
        # Create empty log files
        Path(snakemake.output.cost_log).parent.mkdir(parents=True, exist_ok=True)
        Path(snakemake.output.pred_state).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(snakemake.output.cost_log)
        pd.DataFrame().to_csv(snakemake.output.pred_state)
        
        if snakemake.output.get("cost_params"):
            Path(snakemake.output.cost_params).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame().to_csv(snakemake.output.cost_params)
        return
    
    logger.info("Learning is ENABLED")
    
    # Determine timestep
    planning_horizons = snakemake.params.planning_horizons
    timestep = get_timestep(planning_horizons, year)
    
    if timestep == 0:
        logger.info(f"First horizon ({year}) - using pass-through costs (no learning)")
        n = pypsa.Network(snakemake.input.network)
        n.meta["learning_costs_applied"] = False
        n.meta["learning_first_horizon"] = True
        n.export_to_netcdf(snakemake.output.network)
        
        Path(snakemake.output.cost_log).parent.mkdir(parents=True, exist_ok=True)
        Path(snakemake.output.pred_state).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(snakemake.output.cost_log)
        pd.DataFrame().to_csv(snakemake.output.pred_state)
        
        if snakemake.output.get("cost_params"):
            Path(snakemake.output.cost_params).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame().to_csv(snakemake.output.cost_params)
        return
    
    logger.info(f"Timestep: {timestep} years since previous horizon")
    
    # Load learning data
    params_base = load_learning_params(snakemake.input.params)
    state = load_learning_state(snakemake.input.state)
    
    # Apply beta adjustments if enabled (recalibrates A to maintain anchor point)
    params = apply_beta_adjustment(params_base, state, learning_cfg)
    
    # Log learning curve approach
    beta_cfg = learning_cfg.get("beta_adjustment", {})
    if beta_cfg.get("enabled", False):
        logger.info("Using ADJUSTED learning parameters with 2020 anchor recalibration")
        logger.info(f"  Base parameters from: learning_params.csv")
        logger.info(f"  Beta adjustment: {beta_cfg.get('method', 'multiplier')}")
        logger.info(f"  A recalibrated to maintain {beta_cfg.get('anchor_year', 2020)} costs")
    else:
        logger.info("Using BASE learning parameters A and β from learning_params.csv")
        logger.info("No beta adjustment - using calibrated values directly")
    
    # Validate learning parameters against 2020 costs (only in 2025)
    if year == 2025:
        logger.info("="*70)
        logger.info("Validating learning parameters against 2020 baseline costs")
        beta_cfg = learning_cfg.get("beta_adjustment", {})
        if beta_cfg.get("enabled", False):
            logger.info("(Using adjusted β with recalibrated A to maintain 2020 anchor)")
        logger.info("="*70)
        try:
            validate_learning_parameter_A(
                params=params,
                state_2020=state,  # For 2025, state contains 2020 realized deployment
                costs_2020_file=snakemake.input.basecost,
                learning_cfg=learning_cfg,
                tolerance=0.15
            )
            logger.info("✓ Validation PASSED - Learning parameters correctly predict 2020 costs")
            if beta_cfg.get("enabled", False):
                logger.info("✓ Anchor recalibration working correctly")
        except AssertionError as e:
            logger.error(f"✗ Validation FAILED: {e}")
            raise
        logger.info("="*70)
    
    # Pass state file path for loading historical realized data
    state_file_path = snakemake.input.state
    
    logger.info("Calculating predicted costs using log-linear forecasting...")
    
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
    
    predictions = calculate_predicted_costs(
        params, state, learning_cfg, year, planning_horizons, state_file_path, costs_file, wacc_dict
    )
    
    # Extract learning rates from params if beta adjustment was applied
    learning_rates = None
    if hasattr(params, 'attrs') and 'learning_rates' in params.attrs:
        learning_rates = params.attrs["learning_rates"]
    
    # Update network (single export with all metadata)
    tech_mapping = learning_cfg["tech_mapping"]
    n, updates_log = update_network_costs(
        snakemake.input.network,
        predictions,
        tech_mapping,
        snakemake.output.network,
        learning_cfg,
        costs_file,
        learning_rates=learning_rates
    )
    
    # Save logs
    save_cost_log(predictions, snakemake.output.cost_log)
    save_prediction_state(predictions, snakemake.output.pred_state)
    
    # Save cost params to results for transparency
    cost_params_output = snakemake.output.get("cost_params")
    if cost_params_output:
        logger.info(f"Saving cost parameters to results: {cost_params_output}")
        save_cost_log(predictions, cost_params_output)
    
    logger.info("=" * 70)
    logger.info(f"Learning-based cost application completed for {year}")
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
