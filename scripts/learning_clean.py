"""
Rolling-Horizon Endogenous Learning for PyPSA-Earth
====================================================

This module implements endogenous technological learning for multi-period
capacity expansion using piecewise-linear approximation of experience curves.

MATHEMATICAL FOUNDATIONS
========================

1. Experience Curve (Wright's Law)
----------------------------------
The unit investment cost decreases as a power function of cumulative experience:

    c(E) = A x E^(-β)

where:
    - c(E) = specific overnight investment cost [EUR/kW] at cumulative experience E
    - E = cumulative global capacity [GW]
    - A = cost coefficient [EUR/kW] (cost when E = 1 GW)
    - β = learning exponent (dimensionless)

The learning rate (LR) relates to β:
    LR = 1 - 2^(-β)
    β = -\log_2(1 - LR)

Interpretation: LR = 0.20 means 20% cost reduction per doubling of capacity.

2. Cumulative Cost Curve (Integral of Experience Curve)
-------------------------------------------------------
Total overnight investment to build capacity from E₀ to E:

    TC(E) = \int_{E_0}^{E} c(x) dx
          = \int_{E_0}^{E} A x^{-\beta} dx
          = \left[\frac{A}{1-\beta}\right] \times \left[E^{1-\beta} - E_0^{1-\beta}\right]    \text{for } \beta \neq 1

Or equivalently (closed form):
    TC(E) = \left[\frac{1}{1-\beta}\right] \times \left[E \times c(E) - E_0 \times c(E_0)\right]

Properties:
    - TC(E₀) = 0  (no cost when no new capacity built)
    - TC(E) is concave (diminishing returns) and strictly increasing
    - Marginal cost dTC/dE = c(E) decreases with E (learning effect)

3. Unit Conventions
-------------------
    - E_points: cumulative capacity in GW (from historical data)
    - TC_points: cumulative overnight cost in EUR (integral from E_anchor to E)
    - Slopes m_j = \Delta TC/\Delta E: overnight cost per GW [EUR/GW]
    - PyPSA capacities: MW (converted to GW for learning: /1000)
    - inv_cost: overnight investment [EUR] for capacity built
    - capital_cost in PyPSA: annualized [EUR/MW/year]

    CRITICAL: The learning curve gives OVERNIGHT cost c(E) in EUR/kW.
    To add to PyPSA objective (which uses annual costs), we ANNUALIZE:
        annual_cost = overnight_cost x (annuity + FOM_rate)
    where annuity = r / (1 - (1+r)^(-n)) for discount rate r, lifetime n.

4. Piecewise Linear Approximation
---------------------------------
We discretize E into breakpoints {E₀, E₁, ..., Eₙ} and represent:

    E = \sum_j \delta_j x E_j           (convex combination)
    TC = \sum_j \delta_j x TC(E_j)      (interpolated total cost)

where \delta_j \geq 0, \sum_j \delta_j = 1, and SOS2 constraint ensures at most two 
adjacent \delta_j are nonzero.
The segment slopes (average cost in each segment):
    m_j = \frac{TC(E_{j+1}) - TC(E_j)}{E_{j+1} - E_j}  [EUR/GW]

5. SOS2 (Special Ordered Set Type 2) Constraints
------------------------------------------------
To enforce that at most two adjacent \delta's are nonzero:

    \sum_j z_j = 1              (exactly one segment j is "active")
    \delta_p \leq z_{p-1} + z_p          (\delta_p nonzero only if segment p-1 or p is active)
where z_j \in \{0,1\} indicates segment j is active.

6. Lagged Segment-Slope Pricing (lag_periods = 1)
-------------------------------------------------
For period t+5, the unit cost is based on experience E_end[t] (post-build in t):

    \Delta K_{\text{seg}}[t+5, j] \leq \text{Big-M} \times z_{\text{end}}[t, j]   (build priced at segment from E_end[t])
    \text{inv_cost}[t+5] = \sum_j m_j \times \Delta K_{\text{seg}}[t+5, j]  (overnight cost)

This ensures: cost in period t+5 depends on segment where E_end[t] lies.

7. Rolling Horizon Timing
-------------------------
For window [t, t+5]:
    - E_start[t] = committed experience entering period t (from history/prior window)
    - E_end[t] = E_start[t] + \frac{\Delta K[t]}{gf} = experience after building in t
    - E_start[t+5] = E_end[t] (state propagation within window)

Cost assignment with lag_periods=1:
    - Period t:   EXOGENOUS costs (based on E_start[t], known before optimization)
    - Period t+5: ENDOGENOUS costs (based on E_end[t], optimized)

WINDOW PROGRESSION
==================
Window 1 (2020, 2025): 
    - E_start[2020] = historical (e.g., 2018)
    - E_end[2020] \approx E_start[2020] (no extendable assets in 2020)
    - 2025 costs based on E_end[2020] -> effectively EXOGENOUS
    - Commit: 2020 build, persist E_end[2020]

Window 2 (2025, 2030):
    - E_start[2025] = E_end[2020] from Window 1 (persisted)
    - E_end[2025] = E_start[2025] + \Delta K[2025]/gf (OPTIMIZED)
    - 2030 costs based on E_end[2025] -> ENDOGENOUS
    - Commit: 2025 build, persist E_end[2025]

...and so on through 2050.

STATE PERSISTENCE
=================
After each window solve, we extract E_end[committed_year] and save it.
The next window loads this as E_previous[next_first_year].

Author: PyPSA-Earth Contributors
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

LinExpr = Union[float, "linopy.expressions.LinearExpression"]  # type: ignore[name-defined]


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass(frozen=True)
class PricingConfig:
    """
    Configuration for learning curve pricing mode.
    
    Attributes
    ----------
    mode : str
        "lagged_segment_slope" (recommended) or "immediate_delta_TC"
    lag_periods : int
        Number of periods of cost lag (1 = 5-year delay for 5-year steps)
    segments : int
        Number of piecewise-linear segments for experience curve
    global_factor : float
        Fraction of local build that counts toward global experience (1.0 = global market)
    """
    mode: str = "lagged_segment_slope"
    lag_periods: int = 1
    segments: int = 7
    global_factor: float = 1.0


@dataclass(frozen=True)
class FinanceConfig:
    """
    Financial parameters for annualizing overnight investment costs.
    
    The learning curve gives overnight cost c(E) in EUR/kW.
    To convert to annual cost for the objective function:
        annual_cost = overnight_cost x (annuity + FOM_rate)
    
    Attributes
    ----------
    wacc_default : float
        Weighted average cost of capital (discount rate)
    lifetime_default : int
        Default asset lifetime in years
    wacc_by_tech : dict
        Technology-specific WACC overrides
    lifetime_by_tech : dict
        Technology-specific lifetime overrides
    fom_by_tech : dict
        Fixed O&M as fraction of overnight CAPEX (e.g., 0.02 = 2%)
    """
    wacc_default: float = 0.07
    lifetime_default: int = 30
    wacc_by_tech: Optional[Dict[str, float]] = None
    lifetime_by_tech: Optional[Dict[str, int]] = None
    fom_by_tech: Optional[Dict[str, float]] = None


# =============================================================================
# MATHEMATICAL HELPER FUNCTIONS
# =============================================================================

def _annuity(n_years: int, r: float) -> float:
    """
    Compute annuity factor for converting overnight cost to annual payment.
    
    Formula: a = r / (1 - (1+r)^(-n))
    
    Parameters
    ----------
    n_years : int
        Asset lifetime in years
    r : float
        Discount rate (WACC)
    
    Returns
    -------
    float
        Annuity factor [1/year]
        
    Example
    -------
    For 30-year lifetime and 7% WACC:
        a = 0.07 / (1 - 1.07^(-30)) = 0.0806
        So overnight cost of 1000 EUR/kW -> 80.6 EUR/kW/year
    """
    if r <= 0:
        return 1.0 / max(n_years, 1)
    return r / (1.0 - (1.0 + r) ** (-n_years))


def _compute_segment_slopes(E_points: np.ndarray, TC_points: np.ndarray) -> np.ndarray:
    """
    Compute piecewise-linear segment slopes from breakpoints.
    
    The slope m_j = \Delta TC / \Delta E represents the average overnight cost [EUR/GW]
    for building capacity within segment j.
    
    Parameters
    ----------
    E_points : np.ndarray
        Experience breakpoints [GW], shape (n+1,)
    TC_points : np.ndarray
        Cumulative overnight cost at breakpoints [EUR], shape (n+1,)
    
    Returns
    -------
    np.ndarray
        Segment slopes [EUR/GW], shape (n,)
    """
    dE = np.diff(E_points)
    dTC = np.diff(TC_points)
    return dTC / np.maximum(dE, 1e-10)


def experience_curve(
    cumulative_capacity: np.ndarray,
    learning_rate: float,
    c0: float,
    initial_capacity: float = 1.0
) -> np.ndarray:
    """
    Calculate specific investment cost from experience curve.
    
    Formula: c(L) = c_0 \cdot (L / L_0)^{-\alpha}
    where \alpha = \log_2(1 / (1 - LR))
    
    Parameters
    ----------
    cumulative_capacity : float or np.ndarray
        Cumulative installed capacity L [MW or GW]
    learning_rate : float
        Learning rate (0-1), e.g., 0.20 for 20% cost reduction per doubling
    c0 : float
        Initial investment cost [EUR/MW or EUR/GW depending on capacity units]
    initial_capacity : float, default=1.0
        Reference cumulative capacity L₀ [MW or GW]
    
    Returns
    -------
    float or np.ndarray
        Specific investment cost c(L) [EUR/MW or EUR/GW]
    
    Notes
    -----
    Learning rate interpretation:
    - LR = 0.20 -> 20% cost reduction per doubling of capacity
    - LR = 0.15 -> 15% cost reduction per doubling
    - LR = 0.10 -> 10% cost reduction per doubling
    
    The learning index \alpha relates to LR by:
    - \alpha = -\log_2(1 - LR)
    - For LR=0.20: \alpha \approx 0.322
    
    Examples
    --------
    >>> c = experience_curve(1000, 0.20, 1000, 100)
    >>> # At 10x capacity (1000/100), with 20% LR:
    >>> # Doublings: log_2(10) \approx 3.32
    >>> # Cost multiplier: (1-0.20)^3.32 \approx 0.40
    >>> # c \approx 1000 * 0.40 = 400 EUR/MW
    """
    if learning_rate >= 1.0 or learning_rate < 0.0:
        raise ValueError(f"Learning rate must be in [0, 1), got {learning_rate}")
    
    alpha = np.log2(1.0 / (1.0 - learning_rate))
    capacity_ratio = cumulative_capacity / initial_capacity
    
    # Handle edge case where capacity_ratio might be very small
    capacity_ratio = np.maximum(capacity_ratio, 1e-10)
    
    return c0 / (capacity_ratio ** alpha)


def cumulative_cost_curve(
    cumulative_capacity: np.ndarray,
    learning_rate: float,
    c0: float,
    initial_capacity: float = 1.0,
    with_previous_TC: bool = True
) -> np.ndarray:
    """
    Calculate cumulative total investment cost (integral of experience curve).
    
    Formula: TC(L) = [1/(1-\alpha)] · [L·c(L) - L_0·c_0]
    where c(L) = c0 · (L/L_0)^(-\alpha) and \alpha = \log_2(1/(1-LR))
    
    This represents the total cost to build capacity from L_0 to L
    following the learning curve.
    
    Parameters
    ----------
    cumulative_capacity : float or np.ndarray
        Cumulative capacity L [MW or GW]
    learning_rate : float
        Learning rate (0-1)
    c0 : float
        Initial cost [EUR/MW or EUR/GW depending on capacity units]
    initial_capacity : float, default=1.0
        Reference capacity L_0 [MW or GW]
    with_previous_TC : bool, default=True
        If True, include baseline term (L₀·c₀) in calculation
    
    Returns
    -------
    float or np.ndarray
        Cumulative total cost TC(L) [EUR]
    
    Mathematical Derivation
    -----------------------
    Starting from experience curve: c(L) = c0(L/L_0)^(-\alpha)
    
    Total cost is the integral:
        TC(L)
            &= \int_{L_0}^{L} c(x)\, dx \\
            &= \int_{L_0}^{L} c_0 \left(\frac{x}{L_0}\right)^{-\alpha} dx \\
            &= c_0 L_0^{\alpha} \int_{L_0}^{L} x^{-\alpha} \, dx \\
            &= c_0 L_0^{\alpha}
            \left[ \frac{x^{1-\alpha}}{1-\alpha} \right]_{L_0}^{L} \\
            &= \frac{c_0 L_0^{\alpha}}{1-\alpha}
            \left( L^{1-\alpha} - L_0^{1-\alpha} \right) \\
            &= \frac{c_0 L_0}{1-\alpha}
            \left[ \left(\frac{L}{L_0}\right)^{1-\alpha} - 1 \right] \\
            &= \frac{1}{1-\alpha}
            \left[ c_0 L_0 \left(\frac{L}{L_0}\right)^{1-\alpha} - c_0 L_0 \right] \\
            &= \frac{1}{1-\alpha}
            \left[ L \, c_0 \left(\frac{L}{L_0}\right)^{-\alpha} - L_0 c_0 \right] \\
            &= \frac{1}{1-\alpha}
            \left[ L\, c(L) - L_0 c_0 \right]
    """
    if learning_rate >= 1.0 or learning_rate < 0.0:
        raise ValueError(f"Learning rate must be in [0, 1), got {learning_rate}")
    
    alpha = np.log2(1.0 / (1.0 - learning_rate))
    
    if abs(alpha - 1.0) < 1e-10:
        # Special case: α = 1 (logarithmic learning)
        # Use L'Hôpital's rule or direct logarithmic integral
        logger.warning(f"Learning rate {learning_rate} gives alpha = 1, using logarithmic approximation")
        TC = c0 * initial_capacity * np.log(cumulative_capacity / initial_capacity)
        return TC
    
    # Calculate c(L) at current capacity
    c_at_L = experience_curve(cumulative_capacity, learning_rate, c0, initial_capacity)
    
    # Cumulative cost formula: TC(L) = [1/(1-\alpha)] · [L·c(L) - L_0·c_0]
    if with_previous_TC:
        TC = (1.0 / (1.0 - alpha)) * (
            cumulative_capacity * c_at_L - initial_capacity * c0
        )
    else:
        # Incremental cost from L_0 to L (excluding baseline)
        TC = (1.0 / (1.0 - alpha)) * cumulative_capacity * c_at_L
    
    # Fix numerical precision: TC should be exactly 0 at L = L_0
    # Floating-point errors can cause small negative values (e.g., -1e-4)
    TC = np.where(np.isclose(cumulative_capacity, initial_capacity, rtol=1e-9), 0.0, TC)
    
    return TC


def _logspace_points(x_min: float, x_max: float, n_points: int) -> np.ndarray:
    """
    Generate logarithmically spaced points between x_min and x_max.
    
    Parameters
    ----------
    x_min : float
        Minimum value (must be > 0)
    x_max : float
        Maximum value
    n_points : int
        Number of points to generate
    
    Returns
    -------
    np.ndarray
        Array of logarithmically spaced points, shape (n_points,)
    """
    if x_min <= 0:
        raise ValueError(f"x_min must be positive for log spacing, got {x_min}")
    if x_max <= x_min:
        raise ValueError(f"x_max ({x_max}) must be > x_min ({x_min})")
    return np.logspace(np.log10(x_min), np.log10(x_max), n_points)


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================
def _get_periods(n) -> List[int]:
    """Extract investment periods from network, handling single-period case."""
    periods = list(getattr(n, "investment_periods", n.snapshots.get_level_values(0).unique()))
    periods = [int(p) for p in periods]
    if len(periods) < 1:
        raise ValueError(f"[learning] Need at least 1 investment period, got {periods}")
    if len(pd.Index(periods).unique()) != len(periods):
        raise ValueError(f"[learning] Duplicate investment periods: {periods}")
    return sorted(periods)


def _validate_learning_params(n, techs: List[str]) -> None:
    """
    Validate that required learning parameters are attached to network.
    
    Checks for:
    - n.learning_params exists
    - Required keys: interpolation, E_previous, parameters
    - For each tech: valid interpolation points (increasing E, non-decreasing TC)
    """
    if not hasattr(n, "learning_params"):
        raise ValueError("[learning] Network missing n.learning_params")
    lp = n.learning_params
    for k in ("interpolation", "E_previous", "parameters"):
        if k not in lp:
            raise ValueError(f"[learning] n.learning_params missing key '{k}'")

    for tech in techs:
        if tech not in lp["interpolation"]:
            raise ValueError(f"[learning] Missing interpolation for tech '{tech}'")
        if tech not in lp["E_previous"]:
            raise ValueError(f"[learning] Missing E_previous for tech '{tech}'")
        if tech not in lp["parameters"].index:
            raise ValueError(f"[learning] Missing parameters for tech '{tech}'")

        E_points, TC_points = lp["interpolation"][tech]
        if len(E_points) < 2:
            raise ValueError(f"[learning] {tech} needs >=2 interpolation points")
        if np.any(np.diff(E_points) <= 0):
            raise ValueError(f"[learning] {tech} E_points must be strictly increasing")
        if np.any(np.diff(TC_points) < -1e-9):
            raise ValueError(f"[learning] {tech} TC_points must be non-decreasing (cumulative cost)")


# =============================================================================
# COMPONENT AGGREGATION (ΔK from actual build decision variables)
# =============================================================================

def _period_mask(df: pd.DataFrame, period: int) -> pd.Series:
    if "build_year" not in df.columns:
        return pd.Series(True, index=df.index)
    return df["build_year"].astype(int) == int(period)


def _safe_zero_expr() -> float:
    return 0.0


def _sum_var(n, var_name: str, idx: pd.Index) -> LinExpr:
    if not hasattr(n, "model") or not hasattr(n.model, "variables") or var_name not in n.model.variables:
        return _safe_zero_expr()
    if len(idx) == 0:
        return _safe_zero_expr()
    v = n.model[var_name]
    return v.loc[idx].sum()


def get_capacity_build_expression(n, carriers: List[str], period: int) -> LinExpr:
    """
    Aggregate capacity build decision variables for given carriers and period.
    
    Converts from PyPSA units (MW/MWh) to learning units (GW/GWh).
    
    Parameters
    ----------
    n : pypsa.Network
        Network with model attached
    carriers : list
        PyPSA carrier names to aggregate (e.g., ["solar", "onwind"])
    period : int
        Investment period (build_year filter)
    
    Returns
    -------
    LinExpr
        Sum of capacity additions in GW (or GWh for stores)
        
    Unit Conversions
    ----------------
    - Generator/Link/StorageUnit p_nom: MW -> GW (/1000)
    - Store e_nom: MWh -> GWh (/1000)
    
    Note: Must match units used in E_points for interpolation!
    """
    total = _safe_zero_expr()

    # Generators: p_nom (MW)
    if hasattr(n, "generators") and not n.generators.empty:
        df = n.generators
        mask = df.p_nom_extendable & df.carrier.isin(carriers) & _period_mask(df, period)
        idx = df.index[mask]
        total = total + (_sum_var(n, "Generator-p_nom", idx) / 1000.0)

    # Links: p_nom (MW)
    if hasattr(n, "links") and not n.links.empty:
        df = n.links
        mask = df.p_nom_extendable & df.carrier.isin(carriers) & _period_mask(df, period)
        idx = df.index[mask]
        total = total + (_sum_var(n, "Link-p_nom", idx) / 1000.0)

    # StorageUnits: p_nom (MW)
    if hasattr(n, "storage_units") and not n.storage_units.empty:
        df = n.storage_units
        mask = df.p_nom_extendable & df.carrier.isin(carriers) & _period_mask(df, period)
        idx = df.index[mask]
        total = total + (_sum_var(n, "StorageUnit-p_nom", idx) / 1000.0)

    # Stores: e_nom (MWh) -> GWh
    if hasattr(n, "stores") and not n.stores.empty:
        df = n.stores
        mask = df.e_nom_extendable & df.carrier.isin(carriers) & _period_mask(df, period)
        idx = df.index[mask]
        total = total + (_sum_var(n, "Store-e_nom", idx) / 1000.0)

    return total


def zero_learning_capital_costs(
    n: pypsa.Network, techs: list, priced_periods: list, tech_to_carriers: dict
) -> None:
    """
    Zero out capital_cost for learning technologies in priced periods.
    
    This prevents PyPSA from automatically adding learning asset costs to the objective.
    Learning costs will be added explicitly via the inv_cost variable in the learning
    formulation.
    
    Parameters
    ----------
    n : pypsa.Network
        Network with components to modify
    techs : list
        Learning technologies (e.g., ['solar_power', 'onwind_power'])
    priced_periods : list
        Investment periods where learning costs are endogenous
    tech_to_carriers : dict
        Mapping from technology to carrier names
    """
    logger.info("")
    logger.info("[learning] Zeroing out capital_cost for learning assets:")
    
    for tech in techs:
        carriers = tech_to_carriers.get(tech, [tech])
        logger.info(f"  {tech} (carriers: {carriers}):")
        
        # Loop through priced periods to apply build_year filter
        for t in priced_periods:
            logger.info(f"    Period {t}:")
            
            # Generators
            if hasattr(n, "generators") and not n.generators.empty:
                mask = n.generators.p_nom_extendable & n.generators.carrier.isin(carriers) & _period_mask(n.generators, t)
                affected_gens = n.generators.index[mask]
                if len(affected_gens) > 0:
                    # Save original capital_cost for reference
                    if "capital_cost_original" not in n.generators.columns:
                        n.generators["capital_cost_original"] = n.generators["capital_cost"]
                    
                    original_costs = n.generators.loc[affected_gens, "capital_cost"].copy()
                    n.generators.loc[affected_gens, "capital_cost"] = 0.0
                    
                    logger.info(f"      Generators: {len(affected_gens)} affected")
                    logger.info(f"        Original capital_cost range: [{original_costs.min():,.2f}, {original_costs.max():,.2f}] EUR/MW/year")
                    logger.info(f"        Set to: 0.0 EUR/MW/year")
            
            # Links
            if hasattr(n, "links") and not n.links.empty:
                mask = n.links.p_nom_extendable & n.links.carrier.isin(carriers) & _period_mask(n.links, t)
                affected_links = n.links.index[mask]
                if len(affected_links) > 0:
                    if "capital_cost_original" not in n.links.columns:
                        n.links["capital_cost_original"] = n.links["capital_cost"]
                    
                    original_costs = n.links.loc[affected_links, "capital_cost"].copy()
                    n.links.loc[affected_links, "capital_cost"] = 0.0
                    
                    logger.info(f"      Links: {len(affected_links)} affected")
                    logger.info(f"        Original capital_cost range: [{original_costs.min():,.2f}, {original_costs.max():,.2f}] EUR/MW/year")
                    logger.info(f"        Set to: 0.0 EUR/MW/year")
            
            # StorageUnits
            if hasattr(n, "storage_units") and not n.storage_units.empty:
                mask = n.storage_units.p_nom_extendable & n.storage_units.carrier.isin(carriers) & _period_mask(n.storage_units, t)
                affected_su = n.storage_units.index[mask]
                if len(affected_su) > 0:
                    if "capital_cost_original" not in n.storage_units.columns:
                        n.storage_units["capital_cost_original"] = n.storage_units["capital_cost"]
                    
                    original_costs = n.storage_units.loc[affected_su, "capital_cost"].copy()
                    n.storage_units.loc[affected_su, "capital_cost"] = 0.0
                    
                    logger.info(f"      StorageUnits: {len(affected_su)} affected")
                    logger.info(f"        Original capital_cost range: [{original_costs.min():,.2f}, {original_costs.max():,.2f}] EUR/MW/year")
                    logger.info(f"        Set to: 0.0 EUR/MW/year")
            
            # Stores
            if hasattr(n, "stores") and not n.stores.empty:
                mask = n.stores.e_nom_extendable & n.stores.carrier.isin(carriers) & _period_mask(n.stores, t)
                affected_stores = n.stores.index[mask]
                if len(affected_stores) > 0:
                    if "capital_cost_original" not in n.stores.columns:
                        n.stores["capital_cost_original"] = n.stores["capital_cost"]
                    
                    original_costs = n.stores.loc[affected_stores, "capital_cost"].copy()
                    n.stores.loc[affected_stores, "capital_cost"] = 0.0
                    
                    logger.info(f"      Stores: {len(affected_stores)} affected")
                    logger.info(f"        Original capital_cost range: [{original_costs.min():,.2f}, {original_costs.max():,.2f}] EUR/MW/year")
                    logger.info(f"        Set to: 0.0 EUR/MW/year")
    
    logger.info("")
    logger.info("Capital costs zeroed. PyPSA will not add these to objective.")
    logger.info("Learning costs will be added explicitly via inv_cost variable.")
    logger.info("")


def build_TECH_TO_CARRIERS(learning_cfg: dict) -> Dict[str, List[str]]:
    base = {
        "solar_power": ["solar"],
        "onwind_power": ["onwind"],
        "offwind_power": ["offwind-ac", "offwind-dc"],
        "battery_power": ["battery inverter"],
        "battery_energy": ["battery"],
        "electrolyser_power": ["H2 Electrolysis"],
        "h2_energy": ["H2 Store"],
    }

    tm = learning_cfg.get("tech_mapping", {}) or {}
    if tm:
        override: Dict[str, List[str]] = {}
        for carrier, tech in tm.items():
            override.setdefault(tech, []).append(carrier)
        base.update(override)

    return base


# =============================================================================
# MAIN LEARNING INJECTION FUNCTION
# =============================================================================

def add_endogenous_learning_rolling_horizon_clean(n, snapshots, learning_cfg: dict) -> None:
    """
    Add endogenous learning constraints to a rolling-horizon multi-period network.
    
    This function implements the complete learning formulation:
    
    1. STATE VARIABLES (for all periods in window):
       - E_start[t]: experience entering period t [GW]
       - E_end[t]: experience after building in period t [GW]
       - DK[t]: capacity build in period t [GW]
       - inv_cost[t]: overnight investment cost [EUR]
    
    2. SOS2 INTERPOLATION (for E_end):
       - \delta[t,p]: convex combination weights (\sum\delta=1)
       - z[t,j]: segment indicator binaries (\sum z=1, SOS2 adjacency)
       - E_end[t] = \sum_p \delta[t,p] x E_p
    
    3. CONSTRAINTS:
       - E_start[first] = E_previous (anchored)
       - E_start[t+1] = E_end[t] (propagation)
       - DK[t] = gf x (E_end[t] - E_start[t])
       - DK[t] = \sum_j DK_seg[t,j] (segment decomposition)
       - DK_seg[t,j] \leq Big-M x z_end[t-1,j] (lagged gating)
       - inv_cost[t] = \sum_j m_j x DK_seg[t,j] (overnight cost)
    
    4. OBJECTIVE MODIFICATION (for endogenous periods only):
       - Remove: original capital_cost terms
       - Add: annualized learning cost = annuity x inv_cost
    
    Parameters
    ----------
    n : pypsa.Network
        Multi-period network with model attached
    snapshots : pd.Index
        Snapshots being optimized
    learning_cfg : dict
        Learning configuration (from config.learning.yaml)
    
    Timing
    ------
    For window [t, t+5]:
    - Period t: EXOGENOUS (costs from E_start[t], known)
    - Period t+5: ENDOGENOUS (costs from E_end[t], optimized)
    """

    if not learning_cfg.get("enabled", False):
        logger.info("[learning] Endogenous learning is disabled")
        return

    TECH_TO_CARRIERS = build_TECH_TO_CARRIERS(learning_cfg)

    lp = n.learning_params
    all_techs = list(lp.get("technologies", lp["interpolation"].keys()))
    techs = [t for t in all_techs if t in lp["interpolation"]]

    if not techs:
        logger.warning("[learning] enabled but no technologies found; skipping")
        return
    
    # Log which technologies are being excluded
    if len(all_techs) > len(techs):
        excluded = [t for t in all_techs if t not in techs]
        logger.info(f"[learning] TEMPORARY: Only implementing learning for {techs} (excluding {excluded})")

    periods = _get_periods(n)
    first_period = periods[0]
    priced_periods = periods[1:]       # only these get endogenous cost in objective
    state_periods = periods            # we model state for all periods

    pricing = PricingConfig(
        mode=(learning_cfg.get("pricing", {}) or {}).get("mode", "lagged_segment_slope"),
        lag_periods=int(learning_cfg.get("lag_periods", 1)),
        segments=int(learning_cfg.get("segments", lp.get("segments", 7))),
        global_factor=float(learning_cfg.get("global_factor", lp.get("global_factor", 1.0))),
    )

    if pricing.segments < 1:
        raise ValueError("[learning] segments must be >= 1")
    if pricing.global_factor <= 0:
        raise ValueError("[learning] global_factor must be > 0")
    if pricing.mode not in ("lagged_segment_slope", "immediate_delta_TC"):
        raise ValueError(f"[learning] Unknown pricing.mode={pricing.mode}")

    _validate_learning_params(n, techs)

    pts = list(range(pricing.segments + 1))
    segs = list(range(pricing.segments))

    techs_idx = pd.Index(techs, name="carrier")
    state_idx = pd.Index(state_periods, name="period")
    pts_idx = pd.Index(pts, name="segment_point")
    segs_idx = pd.Index(segs, name="segment")

    logger.info("=" * 80)
    logger.info("[learning] Rolling Horizon Endogenous Learning")
    logger.info("=" * 80)
    logger.info(f"[learning] techs={techs}")
    logger.info(f"[learning] window periods={periods} (commit {first_period}; price {priced_periods})")
    logger.info(f"[learning] pricing.mode={pricing.mode}, lag_periods={pricing.lag_periods}, segments={pricing.segments}, gf={pricing.global_factor}")
    
    # Note: Capital cost zeroing is now done in solve_rolling_horizon before n.optimize()
    # to ensure it happens just before optimization, not during extra_functionality

    # VARIABLES:

    # E_start[t], E_end[t]
    n.model.add_variables(lower=0.0, coords=[techs_idx, state_idx], name="learning_E_start")
    n.model.add_variables(lower=0.0, coords=[techs_idx, state_idx], name="learning_E_end")

    E_start = n.model["learning_E_start"]
    E_end = n.model["learning_E_end"]

    # SOS2 weights + segment binaries for E_end
    n.model.add_variables(lower=0.0, upper=1.0, coords=[techs_idx, state_idx, pts_idx], name="learning_delta_end")
    n.model.add_variables(binary=True, coords=[techs_idx, state_idx, segs_idx], name="learning_z_end")

    delta_end = n.model["learning_delta_end"]
    z_end = n.model["learning_z_end"]

    # Build and cost
    n.model.add_variables(lower=0.0, coords=[techs_idx, state_idx], name="learning_DK")
    n.model.add_variables(lower=0.0, coords=[techs_idx, state_idx], name="learning_inv_cost")

    DK = n.model["learning_DK"]
    inv_cost = n.model["learning_inv_cost"]

    # Optional TC_end if needed
    if pricing.mode == "immediate_delta_TC":
        n.model.add_variables(lower=0.0, coords=[techs_idx, state_idx], name="learning_TC_end")
        TC_end = n.model["learning_TC_end"]
    else:
        TC_end = None

    # Segment-disaggregated builds for lagged segment pricing
    if pricing.mode == "lagged_segment_slope":
        n.model.add_variables(lower=0.0, coords=[techs_idx, state_idx, segs_idx], name="learning_DK_seg")
        DK_seg = n.model["learning_DK_seg"]
    else:
        DK_seg = None

    # CONSTRAINTS:
    # 1) Anchor E_start[first_period] to previously committed history (E_previous[first_period]) from config or previously solved n
    for tech in techs:
        E_prev_map = lp["E_previous"][tech]
        if first_period not in E_prev_map:
            raise ValueError(f"[learning] {tech}: missing E_previous[{first_period}]")
        E0 = float(E_prev_map[first_period])
        if E0 <= 0:
            raise ValueError(f"[learning] {tech}: invalid E_previous[{first_period}]={E0} (must be > 0)")

        n.model.add_constraints(
            E_start.sel(carrier=tech, period=first_period) == E0, # E_start[tech, first_period] == E_previous[tech, first_period]
            name=f"learning_Estart_anchor_{tech}_{first_period}",
        )

    # 2) E_start within window
    for tech in techs:
        for i in range(1, len(state_periods)):
            t = state_periods[i]
            t_prev = state_periods[i - 1]
            n.model.add_constraints(
                E_start.sel(carrier=tech, period=t) == E_end.sel(carrier=tech, period=t_prev), # E_start[t_next] = E_end[t]
                name=f"learning_Estart_propagate_{tech}_{t}",
            )

    # 3) Link DK to actual build decision variables
    for tech in techs:
        carriers = TECH_TO_CARRIERS.get(tech, [tech])
        for t in state_periods:
            build_expr = get_capacity_build_expression(n, carriers=carriers, period=t) # in GW
            n.model.add_constraints( # DK[tech, t] = change in p_nom of tech in t (build_year == t )
                DK.sel(carrier=tech, period=t) == build_expr,
                name=f"learning_DK_buildlink_{tech}_{t}",
            )

    # 4) Experience transition within each period
    gf = pricing.global_factor
    for tech in techs:
        for t in state_periods:
            n.model.add_constraints( # DK[t] = gf * (E_end[t] - E_start[t])
                DK.sel(carrier=tech, period=t) == gf * (
                    E_end.sel(carrier=tech, period=t) - E_start.sel(carrier=tech, period=t)
                ),
                name=f"learning_DK_equals_gf_dE_{tech}_{t}",
            )
            # Monotonicity: E_end >= E_start (implied by DK>=0, but explicit might help solver tbd)
            n.model.add_constraints(
                E_end.sel(carrier=tech, period=t) >= E_start.sel(carrier=tech, period=t),
                name=f"learning_Eend_ge_Estart_{tech}_{t}",
            )

    # 5) SOS2 interpolation for E_end on all periods
    logger.info("[learning] Interpolation data for each technology:")
    for tech in techs:
        E_points, TC_points = lp["interpolation"][tech]

        E_arr = xr.DataArray(E_points, dims=["segment_point"], coords={"segment_point": pts})
        TC_arr = xr.DataArray(TC_points, dims=["segment_point"], coords={"segment_point": pts})

        E_min = float(E_points[0])
        E_max = float(E_points[-1])
        
        #points
        logger.info(f"  {tech}:")
        logger.info(f"    E_points (GW): {E_points}")
        if len(E_points) > 1:
            overnight_cost_points = np.diff(TC_points) / np.diff(E_points)# check if this is right
            logger.info(f"    Overnight cost per segment (EUR/kW): {overnight_cost_points / 1000000.0}") 
        logger.info(f"    TC_points (EUR): {TC_points}")
        logger.info(f"    E range: [{E_min:.4f}, {E_max:.4f}] GW, span = {E_max - E_min:.4f} GW")
        
        # slopes
        if len(E_points) > 1:
            slopes = (TC_points[1:] - TC_points[:-1]) / (E_points[1:] - E_points[:-1])
            logger.info(f"    Slopes (EUR/GW): min={float(np.min(slopes)):.4e}, max={float(np.max(slopes)):.4e}")
            logger.info(f"    Slopes (EUR/MW): min={float(np.min(slopes)/1000):.4e}, max={float(np.max(slopes)/1000):.4e}")

        for t in state_periods:
            d = delta_end.sel(carrier=tech, period=t)
            zt = z_end.sel(carrier=tech, period=t)

            # convex combination + exactly one segment
            n.model.add_constraints(d.sum("segment_point") == 1.0, name=f"learning_delta_end_sum_{tech}_{t}")
            n.model.add_constraints(zt.sum("segment") == 1.0, name=f"learning_z_end_one_{tech}_{t}")

            # SOS2 adjacency
            for p in pts:
                z_prev = zt.sel(segment=p - 1) if p > 0 else 0
                z_curr = zt.sel(segment=p) if p < pricing.segments else 0
                n.model.add_constraints(
                    d.sel(segment_point=p) <= z_prev + z_curr,
                    name=f"learning_SOS2_end_{tech}_{t}_p{p}",
                )

            # interpolate E_end
            n.model.add_constraints(
                E_end.sel(carrier=tech, period=t) == (d * E_arr).sum("segment_point"),
                name=f"learning_Eend_interp_{tech}_{t}",
            )

            # optional interpolate TC_end for immediate_delta_TC
            if pricing.mode == "immediate_delta_TC":
                n.model.add_constraints(
                    TC_end.sel(carrier=tech, period=t) == (d * TC_arr).sum("segment_point"),
                    name=f"learning_TCend_interp_{tech}_{t}",
                )

            for j in segs:
                E_lb = float(E_points[j])
                E_ub = float(E_points[j + 1])
                M_lb = max(E_lb - E_min, 0.0)
                M_ub = max(E_max - E_ub, 0.0)

                n.model.add_constraints(
                    E_end.sel(carrier=tech, period=t) >= E_lb - M_lb * (1 - zt.sel(segment=j)),
                    name=f"learning_Eend_lb_{tech}_{t}_{j}",
                )
                n.model.add_constraints(
                    E_end.sel(carrier=tech, period=t) <= E_ub + M_ub * (1 - zt.sel(segment=j)),
                    name=f"learning_Eend_ub_{tech}_{t}_{j}",
                )

    # 6) COSTS
    if pricing.mode == "lagged_segment_slope":
        # Price builds in period u using segment slope determined by z_end[u-1]
        # EXCEPT: First period uses EXACT learning curve (exogenous pricing)
        for tech in techs:
            E_points, TC_points = lp["interpolation"][tech]
            slopes = (TC_points[1:] - TC_points[:-1]) / (E_points[1:] - E_points[:-1])  # EUR per unit E (GW)
            m = xr.DataArray(slopes, dims=["segment"], coords={"segment": segs})

            # Big-M: max build possible within interpolation range
            K_big = float(max(E_points[-1] - E_points[0], 1.0))
            logger.info(f"Kbig is {K_big}")

            params_df = lp["parameters"]
            A_per_kW = float(params_df.loc[tech, "A_over_unit"])  # EUR/kW
            beta = float(params_df.loc[tech, "beta"])
            A_per_GW = A_per_kW * 1_000_000.0  # EUR/GW
            learning_rate = 1.0 - 2.0**(-beta)
            
            # Extract E_anchor for first period (from E_previous map)
            E_prev_map = lp["E_previous"][tech]
            E_anchor = float(E_prev_map[first_period])

            for i, t in enumerate(state_periods):
                n.model.add_constraints( # sum_j DK_seg[t,j] == DK[t]
                    DK_seg.sel(carrier=tech, period=t).sum("segment") == DK.sel(carrier=tech, period=t),
                    name=f"learning_DKseg_sum_{tech}_{t}",
                )

                if i == 0:
                    # First period: price using exact learning curve (exogenous)
                    # DK_seg variables exist but are not used for pricing
                    c0_per_GW = A_per_GW * (E_anchor ** (-beta))
                    c_exact = c0_per_GW  # E_start[first] == E_anchor by construction
                    
                    n.model.add_constraints(
                        inv_cost.sel(carrier=tech, period=t) == c_exact * DK.sel(carrier=tech, period=t),
                        name=f"learning_inv_cost_exact_{tech}_{t}",
                    )
                    logger.info(f"  {tech} period {t}: Using EXACT curve c(E={E_anchor:.2f})={c_exact/1e6:.4f} EUR/kW")
                    continue

                # Gate DK_seg[t,j] by previous period's z_end (lagged pricing)
                t_prev = state_periods[i - 1]
                for j in segs:
                    n.model.add_constraints(
                        DK_seg.sel(carrier=tech, period=t, segment=j) <= K_big * z_end.sel(carrier=tech, period=t_prev, segment=j),
                        name=f"learning_DKseg_gate_{tech}_{t}_{j}",
                    )
                
                # inv_cost via piecewise slopes
                n.model.add_constraints(
                    inv_cost.sel(carrier=tech, period=t) == (DK_seg.sel(carrier=tech, period=t) * m).sum("segment"),
                    name=f"learning_inv_cost_piecewise_{tech}_{t}",
                )

    elif pricing.mode == "immediate_delta_TC":
        # inv_cost[t] = TC_end[t] - TC_end[t-1], inv_cost[first]=0
        for tech in techs:
            n.model.add_constraints(
                inv_cost.sel(carrier=tech, period=first_period) == 0.0,
                name=f"learning_inv_cost_first_zero_{tech}_{first_period}",
            )
            for i in range(1, len(state_periods)):
                t = state_periods[i]
                t_prev = state_periods[i - 1]
                n.model.add_constraints(
                    inv_cost.sel(carrier=tech, period=t) == TC_end.sel(carrier=tech, period=t) - TC_end.sel(carrier=tech, period=t_prev),
                    name=f"learning_inv_cost_deltaTC_{tech}_{t}",
                )
    else:
        raise RuntimeError("unreachable pricing mode")


    # 7) Adjust objective:
    # - Remove original CAPEX for learning tech assets in PRICED periods only
    # - Add annualized learning-based inv_cost for priced periods only
    # - Need to annualize costs
    try:
        w = n.investment_period_weightings["objective"]
    except Exception:
        w = pd.Series(1.0, index=state_periods)

    finance_cfg = (learning_cfg.get("finance", {}) or {})
    fin = FinanceConfig(
        wacc_default=float((finance_cfg.get("wacc", {}) or {}).get("default", 0.07)),
        lifetime_default=int((finance_cfg.get("lifetime", {}) or {}).get("default", 30)),
        wacc_by_tech=(finance_cfg.get("wacc", {}) or None),
        lifetime_by_tech=(finance_cfg.get("lifetime", {}) or None),
        fom_by_tech=(finance_cfg.get("fom", {}) or None),
    )

    # ADd learning costs 
    learning_term = 0.0
    logger.info("[learning] Financial parameters for annualization:")
    for tech in techs:
        wacc = float((fin.wacc_by_tech or {}).get(tech, fin.wacc_default))
        lifetime = int((fin.lifetime_by_tech or {}).get(tech, fin.lifetime_default))
        fom_rate = float((fin.fom_by_tech or {}).get(tech, 0.0))
        
        annuity = _annuity(lifetime, wacc)
        annualization_factor = annuity + fom_rate
        
        logger.info(f"  {tech}: WACC={wacc:.2%}, lifetime={lifetime}y, "
                    f"annuity={annuity:.5f}, FOM={fom_rate:.2%}, "
                    f"total_factor={annualization_factor:.5f}")

        for t in priced_periods:
            wt = float(w.loc[t]) if hasattr(w, "loc") else float(w[t])
            # inv_cost[tech,t] is overnight EUR; multiply by factor for annual EUR
            logger.info(f"    Period {t}: weight={wt}, annualization_factor={annualization_factor:.5f}")
            learning_term = learning_term + wt * annualization_factor * inv_cost.sel(carrier=tech, period=t)

    # =============================================================================
    # ADD LEARNING COSTS TO OBJECTIVE
    # =============================================================================
    # Since we zeroed out capital_cost for learning assets, PyPSA's automatic
    # objective construction won't include them. Now we add learning costs.
    if priced_periods:
        # Log objective components for debugging
        try:
            orig_obj_const = float(n.model.objective.expression.const) if hasattr(n.model.objective.expression, 'const') else 0.0
            logger.info(f"[learning] OBJECTIVE MODIFICATION DEBUG:")
            logger.info(f"  Original objective constant term: {orig_obj_const:,.0f} EUR")
            logger.info(f"  Priced periods: {priced_periods}")
            logger.info(f"  Period weightings: {dict(w)}")
            
            # Count variables in learning term
            if hasattr(learning_term, 'vars'):
                logger.info(f"  learning_term has {len(learning_term.vars)} variable terms")
            
            # Try to evaluate the terms (might fail if they're symbolic)
            try:
                if hasattr(learning_term, 'const'):
                    learning_val = float(learning_term.const)
                    logger.info(f"  Learning term being added (constant): {learning_val:,.0f} EUR")
            except (AttributeError, TypeError):
                logger.info(f"  Learning term is symbolic (variable-dependent)")
        except Exception as e:
            logger.info(f"  Could not evaluate objective components: {e}")
        
        logger.info(f"[learning] Adding learning costs to objective")
        # ONLY ADD learning costs (don't subtract anything - capital_cost is already 0)
        n.model.objective.expression = n.model.objective.expression + learning_term
        
        logger.info(f"[learning] Objective modified for endogenous periods {priced_periods}:")
        logger.info(f"  + Added annualized learning costs for techs {techs}")
        logger.info(f"  Note: Original capital_cost for these techs was set to 0 before optimization")
    else:
        logger.info(f"[learning] No endogenous periods in this window, objective unchanged")

    logger.info("=" * 80)
    logger.info("[learning] Endogenous learning constraints added successfully")
    logger.info("=" * 80)
    

def extract_committed_learning_state(n_solved, committed_year: int) -> pd.DataFrame:
    """
    Extract committed experience E_end[committed_year] from solved model.
    
    This is the key function for state persistence between rolling windows.
    After solving window [t, t+5], extract E_end[t] and save it.
    The next window uses this as E_previous[t+5].
    
    Parameters
    ----------
    n_solved : pypsa.Network
        Solved network with model attached
    committed_year : int
        The period being committed (first period of the window)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - technology: learning tech name
        - year: committed year
        - E_end_committed_GW: cumulative experience after build
        - E_start_GW: cumulative experience before build
        - DK_GW: capacity built in this period
    """
    if not hasattr(n_solved, "model"):
        logger.warning("[learning] Cannot extract state: no model attached")
        return pd.DataFrame(columns=["technology", "year", "E_end_committed_GW", "E_start_GW", "DK_GW"])
    
    required_vars = ["learning_E_end", "learning_E_start", "learning_DK"]
    missing = [v for v in required_vars if v not in n_solved.model.variables]
    if missing:
        logger.warning(f"[learning] Cannot extract state: missing variables {missing}")
        return pd.DataFrame(columns=["technology", "year", "E_end_committed_GW", "E_start_GW", "DK_GW"])

    E_end = n_solved.model["learning_E_end"]
    E_start = n_solved.model["learning_E_start"]
    DK = n_solved.model["learning_DK"]
    
    techs = list(E_end.coords["carrier"].values)
    rows = []
    
    logger.info(f"[learning] Extracting committed state for year {committed_year}:")
    for tech in techs:
        try:
            e_end = float(E_end.sel(carrier=tech, period=int(committed_year)).solution)
            e_start = float(E_start.sel(carrier=tech, period=int(committed_year)).solution)
            dk = float(DK.sel(carrier=tech, period=int(committed_year)).solution)
            
            rows.append({
                "technology": tech, 
                "year": int(committed_year), 
                "E_end_committed_GW": e_end,
                "E_start_GW": e_start,
                "DK_GW": dk
            })
            logger.info(f"  {tech}: E_start={e_start:.4f} GW, ΔK={dk:.4f} GW, E_end={e_end:.4f} GW")
        except Exception as e:
            logger.warning(f"[learning] Failed to extract {tech}/{committed_year}: {e}")
            continue
    
    return pd.DataFrame(rows)


def load_committed_learning_state(state_dir: str, year: int) -> Dict[str, float]:
    """
    Load previously committed learning state from CSV file.
    
    This function loads E_end from a prior window to use as E_previous
    for the current window.
    
    Parameters
    ----------
    state_dir : str
        Directory containing learning state CSV files
    year : int
        The year whose committed state to load
    
    Returns
    -------
    dict
        Mapping from technology to committed E_end [GW]
        
    Example
    -------
    For window (2025, 2030), load state from 2020:
        state = load_committed_learning_state(state_dir, 2020)
        # state = {"solar_power": 486.5, "onwind_power": 650.2}
    """
    state_file = os.path.join(state_dir, f"learning_state_committed_{year}.csv")
    if not os.path.exists(state_file):
        logger.warning(f"[learning] State file not found: {state_file}")
        return {}
    
    df = pd.read_csv(state_file)
    result = {}
    
    logger.info(f"[learning] Loading committed state from {state_file}:")
    for _, row in df.iterrows():
        tech = row["technology"]
        E = row.get("E_end_committed_GW", row.get("E_end_committed", None))
        if E is None:
            logger.warning(f"[learning] No E_end found for {tech} in {state_file}")
            continue
        result[tech] = float(E)
        logger.info(f"  {tech}: E_end[{year}] = {E:.4f} GW")
    
    return result


def save_committed_learning_state(df_state: pd.DataFrame, output_dir: str, year: int) -> str:
    """
    Save committed learning state to CSV file.
    
    Parameters
    ----------
    df_state : pd.DataFrame
        State DataFrame from extract_committed_learning_state
    output_dir : str
        Directory to save to
    year : int
        Committed year (for filename)
    
    Returns
    -------
    str
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    state_path = os.path.join(output_dir, f"learning_state_committed_{year}.csv")
    df_state.to_csv(state_path, index=False)
    logger.info(f"[learning] Saved committed state to {state_path}")
    return state_path


# =============================================================================
# POST-SOLVE: VALIDATION
# =============================================================================

def validate_learning_solution(n_solved, tol: float = 1e-6) -> bool:
    """
    Validate solved learning model for constraint satisfaction.
    
    Checks:
    1. E_end within interpolation bounds [E_min, E_max]
    2. DK = gf x (E_end - E_start) (experience transition)
    3. E_start[t] = E_end[t-1] for t > first (state propagation)
    
    Parameters
    ----------
    n_solved : pypsa.Network
        Solved network with model attached
    tol : float
        Tolerance for constraint violations
    
    Returns
    -------
    bool
        True if all checks pass, False otherwise
    """
    if not hasattr(n_solved, "model"):
        logger.warning("[learning] Cannot validate: no model attached")
        return False
    if "learning_E_start" not in n_solved.model.variables or "learning_E_end" not in n_solved.model.variables:
        logger.warning("[learning] Cannot validate: missing E_start or E_end variables")
        return False
    if "learning_DK" not in n_solved.model.variables:
        logger.warning("[learning] Cannot validate: missing DK variable")
        return False

    periods = _get_periods(n_solved)
    cfg = getattr(n_solved, "config", {}).get("learning", {}) if hasattr(n_solved, "config") else {}
    gf = float(cfg.get("global_factor", 1.0))

    E_start = n_solved.model["learning_E_start"]
    E_end = n_solved.model["learning_E_end"]
    DK = n_solved.model["learning_DK"]

    max_res = 0.0
    issues = []
    
    logger.info("[learning] Validating solution...")
    
    for tech in list(E_end.coords["carrier"].values):
        E_points, _ = n_solved.learning_params["interpolation"][tech]
        emin, emax = float(E_points[0]), float(E_points[-1])
        
        for i, t in enumerate(periods):
            es = float(E_start.sel(carrier=tech, period=int(t)).solution)
            ee = float(E_end.sel(carrier=tech, period=int(t)).solution)
            dk = float(DK.sel(carrier=tech, period=int(t)).solution)
            
            # Check bounds
            if ee < emin - tol:
                issues.append(f"{tech} E_end[{t}]={ee:.6f} < E_min={emin:.6f}")
            if ee > emax + tol:
                issues.append(f"{tech} E_end[{t}]={ee:.6f} > E_max={emax:.6f}")

            # Check DK = gf × (E_end - E_start)
            expected_dk = gf * (ee - es)
            residual = abs(dk - expected_dk)
            max_res = max(max_res, residual)
            if residual > tol:
                issues.append(f"{tech} DK[{t}]={dk:.6f} != gf*(E_end-E_start)={expected_dk:.6f}")

            # Check state propagation
            if i > 0:
                ee_prev = float(E_end.sel(carrier=tech, period=int(periods[i - 1])).solution)
                prop_residual = abs(es - ee_prev)
                max_res = max(max_res, prop_residual)
                if prop_residual > tol:
                    issues.append(f"{tech} E_start[{t}]={es:.6f} != E_end[{periods[i-1]}]={ee_prev:.6f}")

    if issues:
        logger.warning(f"[learning] Validation FAILED with {len(issues)} issues:")
        for issue in issues[:10]:
            logger.warning(f"  - {issue}")
        if len(issues) > 10:
            logger.warning(f"  ... and {len(issues) - 10} more")
        return False
    else:
        logger.info(f"[learning] Validation PASSED. Max residual: {max_res:.2e}")
        return True
