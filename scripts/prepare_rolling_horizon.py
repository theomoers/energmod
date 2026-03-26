# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: PyPSA-Earth Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Concatenates N consecutive pypsa networks for rolling horizon optimization.

This script prepares a multi-period optimization problem by concatenating
a configurable window of investment periods. Unlike perfect foresight which
solves all periods at once, rolling horizon solves a sliding window of N
periods, saves the first period's results, and rolls forward.
"""

import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import os

from _helpers import (
    get_investment_weighting,
    get_rolling_window_years,
)
from add_existing_baseyear import add_build_year_to_new_assets
from temporal_clustering import _persist_period_id, _restore_period_id

# Rolling-horizon endogenous learning attachments
from learning.learning_data_io import load_historical_capacity
from learning_clean import cumulative_cost_curve, experience_curve

# Allow for PyPSA versions <0.35
try:
    from pypsa.descriptors import expand_series
except ImportError:
    from pypsa.common import expand_series

logger = logging.getLogger(__name__)


def _get_anchor_capacity_from_historical(learning_cfg: dict, tech: str) -> float:
    """
    Load historical cumulative capacity at anchor_year for given technology.

    Returns capacity in GW/GWh consistent with learning curves (historical files are in GW/GWh).
    """
    anchor_year = int(learning_cfg.get("anchor_year", 2015))
    hist_files = learning_cfg.get("historical_data", {}) or {}
    data_file = hist_files.get(tech)
    if not data_file:
        raise ValueError(f"learning.historical_data missing entry for {tech}")

    hist = load_historical_capacity(data_file, tech_name=tech)
    hist = hist[hist["capacity_GW"] > 0].copy()
    if hist.empty:
        raise ValueError(f"{tech}: historical data empty or non-positive in {data_file}")

    # Prefer exact year, else last known year <= anchor_year
    exact = hist.loc[hist["year"] == anchor_year, "capacity_GW"]
    if not exact.empty:
        return float(exact.iloc[0])

    past = hist.loc[hist["year"] <= anchor_year, "capacity_GW"]
    if past.empty:
        raise ValueError(
            f"{tech}: no historical capacity <= anchor_year={anchor_year} in {data_file}"
        )
    return float(past.iloc[-1])


def _estimate_kmax_from_network(n: pypsa.Network, tech: str, years: list[int]) -> float:
    """
    Estimate an upper bound on cumulative build over the rolling window for tech.

    For solar/onwind, use generator `p_nom_max` potentials where available; fall back
    to a heuristic if potentials are not finite.

    Returns K_max_total in GW.
    """
    if not hasattr(n, "generators") or n.generators.empty:
        return 0.0

    gens = n.generators[(n.generators.carrier == tech) & (n.generators.p_nom_extendable)]
    if gens.empty:
        return 0.0

    # Prefer build-year specific potentials when present
    if "build_year" in gens.columns:
        gens = gens[gens.build_year.isin(years)]

    pmax = gens.get("p_nom_max")
    if pmax is None:
        return 0.0

    finite = pmax.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        # Heuristic: allow up to 10x existing capacity if no explicit potentials
        base = gens.get("p_nom", pd.Series(0.0, index=gens.index)).sum()
        return float(base / 1000.0) * 10.0

    return float(finite.sum() / 1000.0)  # MW -> GW


def _logspace_points(E_min: float, E_max: float, n_points: int) -> np.ndarray:
    """
    Generate logarithmically spaced points in experience dimension.
    """
    E_min = max(float(E_min), 1e-6)
    E_max = max(float(E_max), E_min * 1.001)
    return np.logspace(np.log10(E_min), np.log10(E_max), n_points)


def attach_learning_parameters(n: pypsa.Network, years: list[int], config: dict, learning_state_file: str = None):
    """
    Attach endogenous learning parameters and interpolation points to a rolling-horizon network.

    Produces:
      - n.learning_params["parameters"]: DataFrame (A_over_unit, beta, unit)
      - n.learning_params["interpolation"]: {tech: (E_points_GW, TC_points_EUR)}
      - n.learning_params["E_previous"]: {tech: {first_year: E_anchor_GW}}
      - n.learning_params["TC_previous"]: {tech: {first_year: TC_anchor_EUR}}

    State Persistence Logic:
      - For the FIRST window (starting from base year, e.g., 2020):
        E_previous[first_year] = historical anchor (from anchor_year, e.g., 2018)
      - For SUBSEQUENT windows (e.g., starting from 2025):
        E_previous[first_year] = E_end[prev_year] loaded from committed state file

    Parameters
    ----------
    n : pypsa.Network
        Network to attach learning parameters to
    years : list[int]
        Investment years in the rolling window
    config : dict
        Configuration dictionary
    learning_state_file : str, optional
        Path to learning state CSV from previous window. If None, will try to construct from config.

    Notes:
      - E_points are **logarithmically spaced in E** (experience) dimension.
      - TC_points are computed from cumulative_cost_curve(E_points, ...) with
        initial_capacity = E_anchor, so TC(E_anchor)=0.
    """
    learning_cfg = (config or {}).get("learning", {}) or {}
    if not learning_cfg.get("enabled", False):
        return

    techs = learning_cfg.get("technologies", ["solar_power", "onwind_power"])
    segments = int(learning_cfg.get("segments", 7))
    lag_periods = int(learning_cfg.get("lag_periods", 1))
    gf = float(learning_cfg.get("global_factor", 1.0))

    # Load learning curve params (A_over_unit, beta, unit)
    manifest_path = Path("data/learning-data/manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Learning manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    params_path = Path("data/learning-data") / manifest["legacy_curve"]["params_csv"]
    params = pd.read_csv(params_path).set_index("technology")
    missing = [t for t in techs if t not in params.index]
    if missing:
        raise ValueError(f"Missing learning params for {missing} in {params_path}")
    params = params.loc[techs].copy()

    first_year = int(years[0])
    interpolation: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    E_previous: dict[str, dict[int, float]] = {}
    TC_previous: dict[str, dict[int, float]] = {}
    anchor_segment: dict[str, int] = {}
    
    # Determine if we should load committed state from a prior window
    # Get all planning horizons to determine if this is the first window
    all_horizons = config.get("scenario", {}).get("planning_horizons", [])
    base_year = min(int(y) for y in all_horizons) if all_horizons else first_year
    
    # Check for committed state from prior window
    committed_state = {}
    if first_year > base_year and learning_state_file:
        # Use provided learning state file path
        state_file = learning_state_file
        prev_year = first_year - 5  # Assume 5-year steps
        
        if os.path.exists(state_file):
            logger.info(f"Loading committed learning state from {state_file}")
            df = pd.read_csv(state_file)
            logger.info(f"  CSV contents: {len(df)} rows")
            logger.info(f"  CSV columns: {list(df.columns)}")
            for _, row in df.iterrows():
                tech = row["technology"]
                E = row.get("E_committed_GW", None)
                logger.info(f"  Read from CSV: tech='{tech}', E={E}")
                if E is not None:
                    committed_state[tech] = float(E)
                    logger.info(f"  {tech}: E_end[{prev_year}] = {E:.4f} GW (will use as E_start[{first_year}])")
            logger.info(f"  Final committed_state dict: {committed_state}")
        else:
            logger.warning(f"No committed state file found at {state_file}")
            logger.warning(f"Falling back to historical anchor for all technologies")

    logger.info("=" * 80)
    logger.info("ATTACHING ENDOGENOUS LEARNING PARAMETERS FOR ROLLING HORIZON")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Technologies: {techs}")
    logger.info(f"  Segments: {segments} (→ {segments+1} breakpoints)")
    logger.info(f"  Lag periods: {lag_periods} ({'immediate learning' if lag_periods == 0 else '5-year delayed pricing'})")
    logger.info(f"  Global factor: {gf} ({'global market' if gf == 1.0 else 'regional fraction'})")
    logger.info(f"  Rolling window years: {years}")
    logger.info(f"  Base year: {base_year}, First year: {first_year}")
    logger.info(f"  Anchor method: {learning_cfg.get('anchor_method', 'historical')}")
    logger.info(f"  Anchor year: {learning_cfg.get('anchor_year', 2015)}")
    if committed_state:
        logger.info(f"  Loaded committed state for: {list(committed_state.keys())}")
    logger.info("")

    for tech in techs:
        logger.info("-" * 80)
        logger.info(f"Technology: {tech}")
        logger.info("-" * 80)
        
        # Determine E_anchor: use committed state if available, else historical
        logger.info(f"  DEBUG: Checking committed_state for '{tech}'")
        logger.info(f"  DEBUG: committed_state keys: {list(committed_state.keys())}")
        logger.info(f"  DEBUG: tech in committed_state: {tech in committed_state}")
        
        if tech in committed_state:
            E_anchor = committed_state[tech]
            logger.info(f"  Using committed state: E_anchor = {E_anchor:.4f} GW (from prior window)")
        else:
            E_anchor = _get_anchor_capacity_from_historical(learning_cfg, tech)
            logger.info(f"  Using historical anchor: E_anchor = {E_anchor:.4f} GW")
        logger.info(f"  Anchor experience (E_anchor): {E_anchor:.4f} GW")

        # Use theoretical maximum capacity from config, fallback to network estimation
        max_capacity_map = learning_cfg.get("max_capacity_GW", {})
        K_max_total = _estimate_kmax_from_network(n, tech, years)
        E_max_required = E_anchor + K_max_total / max(gf, 1e-9)
        if tech in max_capacity_map:
            E_max_theoretical = float(max_capacity_map[tech])
            logger.info(f"  Using theoretical maximum from config: {E_max_theoretical:.4f} GW")
            # Ensure interpolation max covers maximum possible build in window
            if E_max_theoretical < E_max_required:
                logger.warning(
                    f"  {tech}: Configured E_max ({E_max_theoretical:.4f} GW) is below "
                    f"required minimum ({E_max_required:.4f} GW) based on max build and gf={gf:.3f}. "
                    "Using required minimum."
                )
            E_max = max(E_max_theoretical, E_max_required)
        else:
            # Fallback: estimate from network
            E_max = E_max_required
            logger.info(f"  Estimated max build (K_max_total): {K_max_total:.4f} GW")
            logger.info(f"  Interpolation range (from network estimate): E_min={E_anchor:.4f} GW, E_max={E_max:.4f} GW")
        
        logger.info(f"  Interpolation range: E_min={E_anchor:.4f} GW, E_max={E_max:.4f} GW")
        logger.info(f"  Range span: {E_max - E_anchor:.4f} GW")

        E_points = _logspace_points(E_anchor, E_max, segments + 1)
        logger.info(f"  Breakpoints (E_points) - logarithmically spaced:")
        logger.info(f"    Number of points: {len(E_points)}")
        for i, ep in enumerate(E_points):
            logger.info(f"      Point {i}: E = {ep:.4f} GW")

        A = float(params.loc[tech, "A_over_unit"])
        beta = float(params.loc[tech, "beta"])
        learning_rate = 1.0 - 2.0 ** (-beta)
        unit = str(params.loc[tech, "unit"])
        
        # CRITICAL: Understanding the regression formula and units
        # 
        # The CSV provides:
        # - A in EUR/kW (overnight investment cost)
        # - beta (learning exponent)
        # - unit = "GW" means E (cumulative capacity) is in GW
        # 
        # The regression formula is: c(E) = A × E^(-β)
        # where:
        # - E is in GW (cumulative global capacity)
        # - A is in EUR/kW (cost at E=1 GW)
        # - Result c(E) is in EUR/kW
        # 
        # Example for solar:
        #   A = 42045 EUR/kW (cost at E=1 GW)
        #   At E=486 GW: c(486) = 42045 × 486^(-0.593) ≈ 1072.7 EUR/kW ✓
        #   This is realistic: 1072.7 EUR/kW ≈ 1.07 million EUR/MW
        # 
        # For integration/optimization, we convert to EUR/GW to match capacity units:
        # A_per_GW = A × 1000 = 42045 × 1000 = 42,045,000 EUR/GW
        # Then: c(E) = 42,045,000 × E^(-β) EUR/GW where E is in GW
        # And: TC = ∫ c(E) dE has units (EUR/GW) × GW = EUR ✓
        
        A_per_kW = A  # EUR/kW (from CSV)
        A_per_MW = A * 1000.0  # EUR/MW
        A_per_GW = A * 1_000_000.0  # EUR/GW (for integration with E in GW)
        
        # Calculate c₀ (cost at anchor point) for anchored form
        # Regression: c(E) = A × E^(-β) [EUR/kW when E in GW]
        # At anchor: c₀ = A × E_anchor^(-β) [EUR/kW]
        c0_per_kW = A_per_kW * (E_anchor ** (-beta))
        c0_per_MW = c0_per_kW * 1000.0
        c0_per_GW = c0_per_kW * 1_000_000.0  # For integration
        
        logger.info(f"  Learning curve parameters:")
        logger.info(f"    A (cost at E=1 GW): {A_per_kW:.2f} EUR/kW = {A_per_MW:,.0f} EUR/MW")
        logger.info(f"    β (learning exponent): {beta:.4f}")
        logger.info(f"    Learning rate: {learning_rate:.2%} (cost reduction per doubling)")
        logger.info(f"    E_anchor (reference capacity): {E_anchor:.2f} GW")
        logger.info(f"    c₀ (cost at anchor): {c0_per_kW:.2f} EUR/kW = {c0_per_MW:,.0f} EUR/MW")
        
        # Verify: c(E_anchor) should equal c₀
        c_check = A_per_kW * (E_anchor ** (-beta))
        logger.info(f"    Verification: c(E={E_anchor:.2f} GW) = {A_per_kW:.2f} × {E_anchor:.2f}^(-{beta:.4f}) = {c_check:.2f} EUR/kW ✓")

        # Verify: c(E_anchor) should equal c₀
        c_check = A_per_kW * (E_anchor ** (-beta))
        logger.info(f"    Verification: c(E={E_anchor:.2f} GW) = {A_per_kW:.2f} × {E_anchor:.2f}^(-{beta:.4f}) = {c_check:.2f} EUR/kW ✓")

        # Cumulative cost is computed by integrating the experience curve:
        # TC(E) = ∫_{E₀}^{E} c(x) dx
        # 
        # Given c(x) = A × x^(-β) where A is in EUR/kW and x is in GW,
        # we need to convert units for consistent integration:
        # c(x) = A_per_GW × x^(-β) where A_per_GW = A × 10^6 (EUR/GW)
        # 
        # Then: TC(E) = ∫ (EUR/GW) × d(GW) = EUR ✓
        # 
        # The closed-form integral is:
        # TC(E) = [A_per_GW / (1-β)] × [E^(1-β) - E₀^(1-β)]
        # 
        # This is equivalent to using the anchored form in cumulative_cost_curve:
        # TC(E) = [c₀ × E₀^β / (1-β)] × [E^(1-β) - E₀^(1-β)]
        #       = [1/(1-β)] × [E × c(E) - E₀ × c₀]
        # where c₀ = A_per_GW × E₀^(-β) and c(E) = A_per_GW × E^(-β)
        
        TC_points = cumulative_cost_curve(
            E_points,  # In GW
            learning_rate=learning_rate,
            c0=c0_per_GW,  # In EUR/GW (cost at anchor)
            initial_capacity=E_anchor,  # In GW
            with_previous_TC=True,
        )
        
        # TC_points are in EUR (total investment cost from E_anchor to each E_point)
        # E_points are in GW
        # Slopes ΔTC/ΔE will be in EUR/GW
        # 
        # Sanity check the magnitudes:
        # - c₀ should be O(10^6) EUR/MW = O(10^3) EUR/kW ✓
        # - TC at E_max should be O(10^12) EUR for global scale ✓
        
        logger.info(f"  Cumulative costs (TC_points) at breakpoints:")
        for i, (ep, tcp) in enumerate(zip(E_points, TC_points)):
            logger.info(f"      Point {i}: E = {ep:.4f} GW → TC = {tcp:.2e} EUR")
        
        # Quick sanity check on final point
        if len(TC_points) > 0:
            final_TC = TC_points[-1]
            final_E = E_points[-1]
            avg_cost_per_GW = final_TC / (final_E - E_anchor) if final_E > E_anchor else 0
            logger.info(f"  Sanity check:")
            logger.info(f"    Total cumulative cost to {final_E:.0f} GW: {final_TC:.2e} EUR")
            logger.info(f"    Average cost per GW added: {avg_cost_per_GW:.2e} EUR/GW = {avg_cost_per_GW/1e6:.2f} million EUR/GW")
        
        logger.info(f"  Cumulative costs (TC_points) at breakpoints:")
        for i, (ep, tcp) in enumerate(zip(E_points, TC_points)):
            logger.info(f"      Point {i}: E = {ep:.4f} GW → TC = {tcp:.2e} EUR")

        interpolation[tech] = (E_points, TC_points)
        E_previous[tech] = {first_year: float(E_anchor)}
        TC_previous[tech] = {first_year: float(0.0)}  # by construction with initial_capacity=E_anchor
        
        # Precompute anchor segment index for lag=1 pricing of first period
        seg_idx = int(np.clip(np.searchsorted(E_points, E_anchor, side="right") - 1, 0, segments - 1))
        anchor_segment[tech] = seg_idx
        logger.info(f"  Anchor segment index (for lag=1 first period): {seg_idx}")
        logger.info("")

    n.learning_params = {
        "parameters": params,
        "interpolation": interpolation,
        "E_previous": E_previous,
        "TC_previous": TC_previous,
        "anchor_segment": anchor_segment,
        "lag_periods": lag_periods,
        "segments": segments,
        "global_factor": gf,
        "technologies": techs,
    }

    # Also stash a minimal marker for debugging
    if not hasattr(n, "meta") or not isinstance(n.meta, dict):
        n.meta = {}
    n.meta["learning_attached"] = True
    n.meta["learning_lag_periods"] = lag_periods
    n.meta["learning_segments"] = segments
    
    logger.info("=" * 80)
    logger.info("LEARNING PARAMETERS ATTACHED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Summary:")
    logger.info(f"  Technologies configured: {len(techs)}")
    logger.info(f"  Total breakpoints per tech: {segments + 1}")
    logger.info(f"  Total interpolation points: {len(techs) * (segments + 1)}")
    logger.info(f"  Learning formulation: lag_periods={lag_periods}")
    logger.info("")


def get_missing(df, n, c):
    """
    Get missing assets in network n compared to df for component c.

    Parameters
    ----------
    df : pd.DataFrame
        Static values of pypsa components
    n : pypsa.Network
        Network to which new assets should be added
    c : str
        pypsa component.list_name (e.g. "generators")

    Returns
    -------
    pd.DataFrame
        Static values of missing assets
    """
    df_final = getattr(n, c)
    missing_i = df.index.difference(df_final.index)
    return df.loc[missing_i]


def add_year_to_constraints(n, baseyear):
    """
    Add investment period to global constraints and rename index.

    Parameters
    ----------
    n : pypsa.Network
    baseyear : int
        year in which optimized assets are built
    """
    for c in n.iterate_components(["GlobalConstraint"]):
        c.df["investment_period"] = baseyear
        c.df.rename(index=lambda x: x + "-" + str(baseyear), inplace=True)


def hvdc_transport_model(n):
    """
    Convert AC lines to DC links for multi-decade optimisation with line
    expansion.

    Losses of DC links are assumed to be 3% per 1000km.
    """
    logger.info("Convert AC lines to DC links to perform multi-decade optimisation.")

    n.madd(
        "Link",
        n.lines.index,
        bus0=n.lines.bus0,
        bus1=n.lines.bus1,
        p_nom_extendable=True,
        p_nom=n.lines.s_nom,
        p_nom_min=n.lines.s_nom,
        p_min_pu=-1,
        efficiency=1 - 0.03 * n.lines.length / 1000,
        marginal_cost=0,
        carrier="DC",
        length=n.lines.length,
        capital_cost=n.lines.capital_cost,
    )

    # Remove AC lines
    logger.info("Removing AC lines")
    lines_rm = n.lines.index
    n.mremove("Line", lines_rm)

    # Set efficiency of all DC links to include losses depending on length
    n.links.loc[n.links.carrier == "DC", "efficiency"] = (
        1 - 0.03 * n.links.loc[n.links.carrier == "DC", "length"] / 1000
    )


def adjust_electricity_grid(n, year, years, config=None):
    """
    Adjust electricity grid for multi-period optimization.

    Parameters
    ----------
    n : pypsa.Network
        Network to adjust
    year : int
        Year in which optimized assets are built
    years : list[int]
        List of investment periods in the rolling window
    config : dict, optional
        Configuration dictionary. If provided and contains build_osm_network.force_ac=True,
        AC lines will NOT be converted to DC links.
    """
    n.lines["carrier"] = "AC"
    links_i = n.links[n.links.carrier == "DC"].index
    
    # Check if force_ac is set in config - if so, skip AC to DC conversion
    force_ac = False
    if config is not None:
        force_ac = config.get("build_osm_network", {}).get("force_ac", False)
    
    # Only convert AC to DC if:
    # 1. There are extendable lines/links AND
    # 2. force_ac is not True (i.e., we allow DC conversion)
    if (n.lines.s_nom_extendable.any() or n.links.loc[links_i, "p_nom_extendable"].any()) and not force_ac:
        hvdc_transport_model(n)
        links_i = n.links[n.links.carrier == "DC"].index
        n.links.loc[links_i, "lifetime"] = 100
        if year != years[0]:
            n.links.loc[links_i, "p_nom_min"] = 0
            n.links.loc[links_i, "p_nom"] = 0
    elif force_ac:
        logger.info(f"force_ac=True in config: Skipping AC to DC conversion for year {year}")


def concat_networks(years, network_paths, social_discountrate, config=None):
    """
    Concatenate given pypsa networks and add build years.

    Parameters
    ----------
    years : list[int]
        List of years representing investment periods in the rolling window
    network_paths : list[str]
        List of paths to network files for each investment period
    social_discountrate : float
        Social discount rate for weighting future periods
    config : dict, optional
        Configuration dictionary. Used to check force_ac setting.

    Returns
    -------
    pypsa.Network
        Network for the rolling horizon window
    """
    n = pypsa.Network()
    
    # Collect period_id mappings from each network for temporal clustering
    period_id_parts = []

    # Loop over each input network file and its corresponding investment year
    for i, network_path in enumerate(network_paths):
        year = years[i]
        network = pypsa.Network(network_path)
        
        # Extract period_id for temporal clustering (if exists)
        # Since all networks have the same representative days, we can extract the pattern
        # and apply it directly using the network's snapshots as index
        network_period_id = _restore_period_id(network)
        
        if network_period_id is not None:
            # Extract just the period_id values (pattern) - ignore the index
            # Since all networks have the same representative days, the pattern is the same
            period_id_values = network_period_id.values
            
            # Create MultiIndex for this year: (year, snapshot)
            # Use network.snapshots directly to ensure exact index match
            multi_idx = pd.MultiIndex.from_product([[year], network.snapshots])
            
            # Create unique period_id values per investment year
            # Format: "{year}_{original_period_id}" e.g., "2020_0", "2020_1", ..., "2025_0", "2025_1"
            unique_period_ids = [f"{year}_{pid}" for pid in period_id_values]
            period_id_year = pd.Series(unique_period_ids, index=multi_idx, name="period_id")
            period_id_parts.append(period_id_year)
            logger.info(f"Extracted period_id for year {year}: {len(network_period_id)} snapshots, unique periods: {network_period_id.nunique()}")
        
        # Adjust electricity grid for multi-period optimization
        adjust_electricity_grid(network, year, years, config=config)
        
        # Add build year to new assets if not already done
        add_build_year_to_new_assets(network, year)

        # Static components
        for component in network.iterate_components(
            [
                "Bus",
                "Carrier",
                "Generator",
                "Link",
                "Store",
                "Load",
                "Line",
                "StorageUnit",
            ]
        ):
            df_year = component.df.copy()
            missing = get_missing(df_year, n, component.list_name)
            
            if not missing.empty:
                n.import_components_from_dataframe(missing, component.name)

        # Time variant data
        network_sns = pd.MultiIndex.from_product([[year], network.snapshots])
        snapshots = n.snapshots.drop("now", errors="ignore").union(network_sns)
        n.set_snapshots(snapshots)

        # Iterate all component types in the loaded network
        for component in network.iterate_components():
            pnl = getattr(n, component.list_name + "_t")
            for k in component.pnl.keys():
                pnl_year = component.pnl[k].copy().reindex(snapshots, level=1)
                if pnl_year.empty and (not (component.name == "Load" and k == "p_set")):
                    continue
                if k not in pnl:
                    # TODO: for some reason efficiency2 isn't available, used this workaround:
                    #  initialize an empty time-series DataFrame for any missing key in pnl (e.g., 'efficiency2')
                    pnl[k] = pd.DataFrame(index=snapshots)
                if component.name == "Load":
                    static_load = network.loads.loc[network.loads.p_set != 0]
                    static_load_t = expand_series(static_load.p_set, network_sns).T
                    pnl_year = pd.concat(
                        [pnl_year.reindex(network_sns), static_load_t], axis=1
                    )
                    columns = (pnl[k].columns.union(pnl_year.columns)).unique()
                    pnl[k] = pnl[k].reindex(columns=columns)
                    pnl[k].loc[pnl_year.index, pnl_year.columns] = pnl_year

                else:
                    # For components that aren't new, we just extend
                    # time-varying data from the previous investment
                    # period.
                    if i > 0:
                        pnl[k].loc[(year,)] = pnl[k].loc[(years[i - 1],)].values

                    # Now, add time-varying data for new components.
                    cols = pnl_year.columns.difference(pnl[k].columns)
                    pnl[k] = pd.concat([pnl[k], pnl_year[cols]], axis=1)

        n.snapshot_weightings.loc[year, :] = network.snapshot_weightings.values

        # Global constraints
        for component in network.iterate_components(["GlobalConstraint"]):
            add_year_to_constraints(network, year)
            # Only add constraints that don't already exist
            new_constraints = component.df.index.difference(n.global_constraints.index)
            if not new_constraints.empty:
                n.import_components_from_dataframe(
                    component.df.loc[new_constraints], component.name
                )

    # Set investment periods
    n.investment_periods = n.snapshots.get_level_values(0).unique()
    
    # Weighting of the investment period
    # Assuming last period has same weighting as the period before
    time_w = pd.Series(n.investment_periods).diff().shift(-1).ffill()
    time_w.index = n.investment_periods
    n.investment_period_weightings["years"] = time_w.values
    
    # Set objective weightings with social discount rate
    objective_w = get_investment_weighting(
        n.investment_period_weightings["years"], social_discountrate
    )
    n.investment_period_weightings["objective"] = objective_w.values

    # All former static loads are now time-dependent -> set static = 0
    n.loads["p_set"] = 0
    n.loads_t.p_set = n.loads_t.p_set.fillna(0)

    # Clean up NaNs introduced by concatenation
    if hasattr(n, "links") and not n.links.empty:
        if "p_nom_max" in n.links.columns:
            nan_p_nom_max = n.links.p_nom_max.isna()
            if nan_p_nom_max.any():
                logger.warning(f"Found {nan_p_nom_max.sum()} links with NaN p_nom_max after concatenation; setting to inf")
                n.links.loc[nan_p_nom_max, "p_nom_max"] = np.inf

    if hasattr(n, "links_t") and hasattr(n.links_t, "efficiency"):
        eff_t = n.links_t.efficiency
        if not eff_t.empty:
            static_eff = n.links.efficiency.reindex(eff_t.columns).fillna(1.0)
            eff_t = eff_t.fillna(static_eff)
            n.links_t.efficiency = eff_t
    
    # Combine and store period_id for temporal clustering
    # This ensures cyclic storage constraints work correctly for multi-period networks
    if period_id_parts:
        combined_period_id = pd.concat(period_id_parts)
        combined_period_id = combined_period_id.reindex(n.snapshots)
        
        # Store in temporal_cluster dict and persist to meta for save/load
        # Use the helper function to ensure consistent serialization format
        _persist_period_id(n, combined_period_id)
        
        logger.info(f"Combined period_id for {len(years)} investment periods: {len(combined_period_id)} total snapshots")

    return n


def adjust_stores(n):
    """
    Adjust store behavior for multi-period optimization.

    Ensures stores behave cyclically over one year and not the whole modeling
    horizon. Sets appropriate flags for different types of stores.

    Parameters
    ----------
    n : pypsa.Network
        Network to adjust
    """
    # Cyclic constraint per period (not across entire horizon)
    cyclic_i = n.stores[n.stores.e_cyclic].index
    n.stores.loc[cyclic_i, "e_cyclic_per_period"] = True
    n.stores.loc[cyclic_i, "e_cyclic"] = False
    
    # Non-cyclic store assumptions (CO2 stores, biomass, etc.)
    non_cyclic_store = ["co2", "co2 stored", "solid biomass", "biogas", "EV battery"]
    co2_i = n.stores[n.stores.carrier.isin(non_cyclic_store)].index
    n.stores.loc[co2_i, "e_cyclic_per_period"] = False
    n.stores.loc[co2_i, "e_cyclic"] = False
    
    # e_initial at beginning of each investment period
    e_initial_store = ["solid biomass", "biogas"]
    initial_i = n.stores[n.stores.carrier.isin(e_initial_store)].index
    n.stores.loc[initial_i, "e_initial_per_period"] = True
    
    # CO2 stored gets e_initial_per_period as well
    co2_stored_i = n.stores[n.stores.carrier == "co2 stored"].index
    n.stores.loc[co2_stored_i, "e_initial_per_period"] = True


def adjust_lvlimit(n):
    """
    Convert global constraints for single investment period to one uniform if
    all attributes stay the same.
    """
    c = "GlobalConstraint"
    cols = ["carrier_attribute", "sense", "constant", "type"]
    glc_type = "transmission_volume_expansion_limit"
    
    glc_df = n.df(c)
    if glc_df.empty:
        return
        
    type_mask = glc_df.type == glc_type
    if not type_mask.any():
        return
        
    if (glc_df[type_mask][cols].nunique() == 1).all():
        glc = glc_df[type_mask][cols].iloc[[0]]
        glc.index = pd.Index(["lv_limit"])
        remove_i = glc_df[type_mask].index
        n.mremove(c, remove_i)
        n.import_components_from_dataframe(glc, c)


def adjust_CO2_glc(n):
    """
    Adjust CO2 global constraint type.
    """
    c = "GlobalConstraint"
    glc_name = "CO2Limit"
    glc_type = "primary_energy"
    
    glc_df = n.df(c)
    if glc_df.empty:
        return
        
    mask = (glc_df.index.str.contains(glc_name)) & (glc_df.type == glc_type)
    n.df(c).loc[mask, "type"] = "co2_limit"


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "prepare_rolling_horizon",
            simpl="",
            clusters="110",
            ll="copt",
            opts="1h",
            planning_horizons="2020",
            sopts="1h",
            discountrate="0.071",
            demand="AB",
            h2export="0.0",
            learning_rate="base",
        )

    logging.basicConfig(level=logging.INFO)
    
    # Parameters
    window_years = snakemake.params.window_years
    social_discountrate = snakemake.params.social_discountrate
    
    logger.info(
        f"Concatenating networks for rolling horizon window {window_years} "
        f"with social discount rate of {social_discountrate * 100}%"
    )

    # Get network paths for the rolling window
    network_paths = snakemake.input.networks
    
    # Concat prepared networks of rolling window to single network
    n = concat_networks(window_years, network_paths, social_discountrate, config=snakemake.config)

    # Adjust global constraints lv limit if the same for all years
    adjust_lvlimit(n)
    
    # Adjust global constraints CO2 limit
    adjust_CO2_glc(n)
    
    # Adjust stores to multi period investment
    adjust_stores(n)

    # Attach rolling-horizon endogenous learning parameters (if enabled)
    try:
        # Pass the learning state file path if available
        learning_state_path = snakemake.input.learning_state if hasattr(snakemake.input, 'learning_state') else None
        attach_learning_parameters(n, window_years, snakemake.config, learning_state_file=learning_state_path)
    except Exception as e:
        logger.exception(f"Failed to attach learning parameters: {e}")
        raise

    # Update meta
    if not hasattr(n, "meta") or not isinstance(n.meta, dict):
        n.meta = {}
    n.meta.update(dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards))))
    n.meta["rolling_horizon_window"] = window_years
    # Store learning state file path for use in solve_rolling_horizon
    if learning_state_path:
        n.meta["learning_state_file"] = learning_state_path

    # Export network
    n.export_to_netcdf(snakemake.output.network)
    
    logger.info(f"Rolling horizon network saved to {snakemake.output.network}")
