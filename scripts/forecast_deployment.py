# -*- coding: utf-8 -*-
"""
Log-linear (exponential) growth forecasting for renewable capacity deployment.

This module implements capacity forecasting using exponential regression on
full historical time series rather than simple 2-point CAGR.

Mathematical Framework:
- Log-linear model: ln(K_t) = a + b*t + ε
- Exponential form: K_t = exp(a) * exp(b*t)
- Growth rate: g = exp(b) - 1
- Forecast: K̂_t = exp(a + b*t)

The forecast is fitted on all available historical data up to (but not including)
the target year, then extended forward with guardrails (theta shrinkage, gamma cap).

Created: 2025-01-19
"""

import numpy as np
import pandas as pd
from scipy import stats
import logging
from typing import Dict, Tuple, Optional
from _helpers import create_logger

logger = create_logger(__name__)


def load_historical_capacity(
    data_file: str, 
    tech_name: str,
    capacity_col: str = "Cumulative production, GW"
) -> pd.DataFrame:
    """
    Load historical capacity data from CSV.
    
    Args:
        data_file: Path to historical data CSV (e.g., data/learning-data/solar.csv)
        tech_name: Technology name for logging
        capacity_col: Name of the capacity column in the CSV
    
    Returns:
        DataFrame with 'year' and 'capacity_GW' columns, sorted by year
    """
    logger.info(f"Loading historical data for {tech_name} from {data_file}")
    
    df = pd.read_csv(data_file)
    
    # Identify year column (flexible naming)
    year_col = next((col for col in df.columns if 'year' in col.lower()), None)
    if year_col is None:
        raise ValueError(f"No year column found in {data_file}")
    
    # Identify capacity column (flexible naming)
    cap_col = next(
        (col for col in df.columns if any(
            keyword in col.lower() 
            for keyword in ['cumulative', 'capacity', 'production']
        )), 
        None
    )
    if cap_col is None:
        raise ValueError(f"No capacity column found in {data_file}")
    
    # Extract and clean data
    hist_data = df[[year_col, cap_col]].copy()
    hist_data.columns = ['year', 'capacity_GW']
    hist_data = hist_data.dropna()
    hist_data = hist_data.sort_values('year').reset_index(drop=True)
    
    logger.info(f"  Loaded {len(hist_data)} historical data points "
                f"({hist_data['year'].min():.0f}–{hist_data['year'].max():.0f})")
    
    return hist_data


def merge_realized_data(
    historical: pd.DataFrame,
    realized: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Merge historical data with realized capacity from previous model horizons.
    
    Args:
        historical: DataFrame with historical capacity data (year, capacity_GW)
        realized: Optional DataFrame with realized capacity from model 
                  (year, capacity_GW). If None, returns historical only.
    
    Returns:
        Combined DataFrame, sorted by year, duplicates removed (keeps latest)
    """
    if realized is None or realized.empty:
        return historical.copy()
    
    # Combine and remove duplicates (prefer realized over historical)
    combined = pd.concat([historical, realized], ignore_index=True)
    combined = combined.sort_values('year')
    combined = combined.drop_duplicates(subset='year', keep='last')
    combined = combined.reset_index(drop=True)
    
    logger.info(f"  Merged data: {len(historical)} historical + {len(realized)} realized "
                f"= {len(combined)} total points")
    
    return combined


def fit_log_linear_growth(
    data: pd.DataFrame,
    max_year: Optional[int] = None,
    min_data_points: int = 3
) -> Tuple[float, float, float, float, Dict]:
    """
    Fit log-linear (exponential) growth model to capacity time series.
    
    Model: ln(K) = a + b*t
    Exponential form: K = exp(a) * exp(b*t)
    
    Args:
        data: DataFrame with 'year' and 'capacity_GW' columns
        max_year: If specified, only use data up to (but not including) this year
        min_data_points: Minimum number of data points required for fitting
    
    Returns:
        Tuple of (a, b, r_squared, growth_rate, diagnostics)
        - a: Intercept in log-linear model
        - b: Slope in log-linear model (annualized growth exponent)
        - r_squared: Goodness of fit (R²)
        - growth_rate: Implied annual growth rate (exp(b) - 1)
        - diagnostics: Dict with additional fit statistics
    """
    # Filter data if max_year specified
    if max_year is not None:
        fit_data = data[data['year'] < max_year].copy()
    else:
        fit_data = data.copy()
    
    # Check minimum data points
    if len(fit_data) < min_data_points:
        raise ValueError(
            f"Insufficient data points for fitting: {len(fit_data)} < {min_data_points}"
        )
    
    # Remove zeros/negatives before log transform
    fit_data = fit_data[fit_data['capacity_GW'] > 0].copy()
    
    if len(fit_data) < min_data_points:
        raise ValueError(
            f"Insufficient positive capacity data points: {len(fit_data)} < {min_data_points}"
        )
    
    # Log-linear regression: ln(K) vs year
    x = fit_data['year'].values
    y = np.log(fit_data['capacity_GW'].values)
    
    # Perform linear regression on log-transformed data
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Calculate derived quantities
    r_squared = r_value ** 2
    growth_rate = np.exp(slope) - 1.0
    
    # Diagnostics
    diagnostics = {
        'n_points': len(fit_data),
        'year_min': fit_data['year'].min(),
        'year_max': fit_data['year'].max(),
        'capacity_min_GW': fit_data['capacity_GW'].min(),
        'capacity_max_GW': fit_data['capacity_GW'].max(),
        'p_value': p_value,
        'std_err': std_err,
        'r_value': r_value,
    }
    
    logger.info(f"  Log-linear fit results:")
    logger.info(f"    Data range: {diagnostics['year_min']:.0f}–{diagnostics['year_max']:.0f} "
                f"({diagnostics['n_points']} points)")
    logger.info(f"    Capacity range: {diagnostics['capacity_min_GW']:.2f}–{diagnostics['capacity_max_GW']:.2f} GW")
    logger.info(f"    Growth rate: {growth_rate*100:.2f}% per year")
    logger.info(f"    R²: {r_squared:.4f}, p-value: {p_value:.2e}")
    
    return intercept, slope, r_squared, growth_rate, diagnostics


def forecast_capacity_exponential(
    a: float,
    b: float,
    base_year: int,
    target_year: int,
    K_base: float,
    theta: float = 1.0,
    gamma: Optional[float] = 1.0,
    use_theta: bool = True,
    use_gamma: bool = True,
) -> Tuple[float, float]:
    """
    Forecast future capacity using fitted exponential growth model.
    
    Base forecast: K̂_target = exp(a + b*target_year)
    With guardrails: 
        - Growth rate shrinkage: g_adj = θ * g
        - Deployment cap: ΔK ≤ γ * K_base
    
    Args:
        a: Log-linear intercept
        b: Log-linear slope (growth exponent)
        base_year: Reference year (last known data point)
        target_year: Year to forecast
        K_base: Realized capacity at base_year (GW)
        theta: Growth rate shrinkage factor (0 < θ ≤ 1)
        gamma: Deployment cap as fraction of K_base (None disables cap)
        use_theta: Whether to apply theta shrinkage
        use_gamma: Whether to apply gamma deployment caps
    
    Returns:
        Tuple of (K_forecast, dK_forecast)
        - K_forecast: Forecasted total capacity at target_year (GW)
        - dK_forecast: Forecasted capacity additions (GW)
    """
    years_forward = target_year - base_year
    
    if years_forward <= 0:
        raise ValueError(f"Target year {target_year} must be after base year {base_year}")
    
    # Raw exponential forecast
    K_raw = np.exp(a + b * target_year)
    dK_raw = K_raw - K_base
    
    # Apply guardrails selectively
    g_ann = np.exp(b) - 1.0  # Annual growth rate from slope
    if use_theta:
        g_adj = theta * g_ann
        K_guarded = K_base * (1 + g_adj) ** years_forward
    else:
        g_adj = g_ann
        K_guarded = K_raw
    dK_guarded = K_guarded - K_base
    
    if use_gamma and gamma is not None:
        dK_final = min(gamma * K_base, dK_guarded)
    else:
        dK_final = dK_guarded
    K_final = K_base + dK_final
    
    # 4. Ensure non-negative additions
    if dK_final < 0:
        logger.warning(f"Negative capacity addition ({dK_final:.2f} GW) - setting to zero")
        dK_final = 0.0
        K_final = K_base
    
    logger.info(f"  Forecast {base_year}→{target_year}: "
                f"K_base={K_base:.1f}GW → K_forecast={K_final:.1f}GW "
                f"(ΔK={dK_final:.1f}GW, raw={dK_raw:.1f}GW)")
    
    return K_final, dK_final


def forecast_deployment_loglinear(
    historical_file: str,
    tech_name: str,
    target_year: int,
    base_year: Optional[int] = None,
    realized_data: Optional[pd.DataFrame] = None,
    theta: float = 0.6,
    gamma: Optional[float] = 1.0,
    min_data_points: int = 3,
    use_theta: bool = True,
    use_gamma: bool = True,
) -> Dict[str, float]:
    """
    Complete deployment forecast workflow using log-linear regression.
    
    This is the main entry point for forecasting. It:
    1. Loads historical capacity data
    2. Merges with realized data from previous model horizons (if any)
    3. Fits log-linear growth model on data up to base_year
    4. Forecasts capacity at target_year with guardrails
    
    Args:
        historical_file: Path to historical capacity CSV
        tech_name: Technology name for logging
        target_year: Year to forecast (e.g., 2030)
        base_year: Last year of known data (auto-detected if None)
        realized_data: Optional DataFrame with (year, capacity_GW) from previous model runs
        theta: Growth rate shrinkage factor (default 0.6)
        gamma: Deployment cap as fraction of base capacity (default 1.0, None disables)
        min_data_points: Minimum data points required for fitting (default 3)
        use_theta: Whether to apply theta shrinkage
        use_gamma: Whether to apply gamma caps
    
    Returns:
        Dictionary with forecast results:
        - K_base_GW: Realized capacity at base_year
        - K_forecast_GW: Forecasted capacity at target_year
        - dK_forecast_GW: Forecasted capacity additions
        - growth_rate: Fitted annual growth rate (before shrinkage)
        - r_squared: Goodness of fit
        - diagnostics: Additional fit statistics
    """
    logger.info("=" * 70)
    logger.info(f"Forecasting deployment for {tech_name}: {base_year}→{target_year}")
    logger.info("=" * 70)
    
    # Load and merge data
    historical = load_historical_capacity(historical_file, tech_name)
    combined = merge_realized_data(historical, realized_data)
    
    # Auto-detect base year if not specified
    if base_year is None:
        base_year = int(combined[combined['year'] < target_year]['year'].max())
        logger.info(f"Auto-detected base year: {base_year}")
    
    # Filter out target year from combined data to avoid using realized target year data
    # This ensures we're forecasting TO target_year, not using target_year as an input
    combined_for_fit = combined[combined['year'] < target_year].copy()
    
    # Get base capacity (from the most recent year before target)
    K_base = combined_for_fit[combined_for_fit['year'] == base_year]['capacity_GW'].values
    if len(K_base) == 0:
        raise ValueError(f"No capacity data found for base year {base_year}")
    K_base = float(K_base[0])
    
    # Fit log-linear model (using data up to but NOT including target_year)
    a, b, r2, growth_rate, diagnostics = fit_log_linear_growth(
        combined_for_fit, 
        max_year=None,  # Already filtered above
        min_data_points=min_data_points
    )
    
    # Forecast with guardrails
    K_forecast, dK_forecast = forecast_capacity_exponential(
        a=a,
        b=b,
        base_year=base_year,
        target_year=target_year,
        K_base=K_base,
        theta=theta,
        gamma=gamma,
        use_theta=use_theta,
        use_gamma=use_gamma,
    )
    
    # Compile results
    results = {
        'K_base_GW': K_base,
        'K_forecast_GW': K_forecast,
        'dK_forecast_GW': dK_forecast,
        'L_forecast_GW': K_forecast,  # Cumulative = total capacity for this formulation
        'growth_rate': growth_rate,
        'r_squared': r2,
        'n_data_points': diagnostics['n_points'],
        'fit_year_min': diagnostics['year_min'],
        'fit_year_max': diagnostics['year_max'],
        'base_year': base_year,
        'target_year': target_year,
        'theta': theta,
        'gamma': gamma,
        'use_theta': use_theta,
        'use_gamma': use_gamma,
    }
    
    logger.info("=" * 70)
    logger.info(f"Forecast complete for {tech_name}")
    logger.info("=" * 70)
    
    return results


# Mapping from PyPSA technology keys to historical data files
TECH_TO_DATAFILE = {
    'solar_power': 'data/learning-data/historical/solar.csv',
    'onwind_power': 'data/learning-data/historical/wind.csv',
    'battery_energy': 'data/learning-data/historical/batteries.csv',
    'battery_power': 'data/learning-data/historical/batteries.csv',  # Same source, different unit
    'electrolyser_power': 'data/learning-data/historical/electrolyzers.csv',
    # 'h2_energy': None,  # No historical data available yet
}


def get_historical_datafile(tech_key: str) -> Optional[str]:
    """
    Get historical data file path for a given technology key.
    
    Args:
        tech_key: Technology key (e.g., 'solar_power', 'onwind_power')
    
    Returns:
        Path to historical data CSV, or None if not available
    """
    return TECH_TO_DATAFILE.get(tech_key)
