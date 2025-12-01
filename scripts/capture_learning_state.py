# -*- coding: utf-8 -*-
"""
Capture realized deployment from solved PyPSA network for learning curves.

This script runs AFTER solve_network_myopic to extract actual deployment
and update learning state for the next horizon iteration.

Extracts:
- K_realized: Total installed capacity at current horizon (from p_nom_opt/e_nom_opt)
- L_realized: Cumulative deployment (L_prev + ΔK_actual)
- K_prev: Previous horizon's capacity (for growth rate calculation)

Saves to: resources/learning/state/learning_state_{year}.csv

Created: 2025-11-06
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import yaml
from _helpers import configure_logging, create_logger

logger = create_logger(__name__)


def load_config_learning(config_file):
    """Load learning configuration from config.learning.yaml."""
    logger.info(f"Loading learning configuration from {config_file}")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config["learning"]


def load_previous_state(state_file):
    """
    Load previous learning state.
    
    Args:
        state_file: Path to learning_state_{prev_year}.csv
    
    Returns:
        DataFrame with columns: L_realized_GW, K_realized_GW, K_prev_GW
    """
    logger.info(f"Loading previous state from {state_file}")
    state = pd.read_csv(state_file, index_col="technology")
    
    required_cols = ["L_realized_GW", "K_realized_GW", "K_prev_GW"]
    missing = set(required_cols) - set(state.columns)
    if missing:
        raise ValueError(f"Previous state missing columns: {missing}")
    
    logger.info(f"  Loaded state for {len(state)} technologies")
    return state


def extract_realized_capacity(network_path, tech_mapping):
    """
    Extract realized installed capacity from solved network.
    
    Sums p_nom_opt (for generators, links) or e_nom_opt (for stores) by carrier.
    Converts carrier names to technology keys using tech_mapping.
    
    Args:
        network_path: Path to solved network
        tech_mapping: Dict mapping carriers to technologies
    
    Returns:
        Dict of {tech: capacity_GW}
    """
    logger.info(f"Loading solved network from {network_path}")
    n = pypsa.Network(network_path)
    
    # Reverse mapping: carrier → tech
    carrier_to_tech = {carrier: tech for carrier, tech in tech_mapping.items()}
    
    realized = {}
    
    # Extract from generators (p_nom_opt)
    if not n.generators.empty:
        gen_capacity = n.generators.groupby("carrier")["p_nom_opt"].sum() / 1e3  # MW -> GW
        for carrier, capacity in gen_capacity.items():
            if carrier in carrier_to_tech:
                tech = carrier_to_tech[carrier]
                realized[tech] = realized.get(tech, 0.0) + capacity
                logger.debug(f"  generators[{carrier}]: {capacity:.2f} GW -> {tech}")
    
    # Extract from storage_units (p_nom_opt)
    if not n.storage_units.empty:
        su_capacity = n.storage_units.groupby("carrier")["p_nom_opt"].sum() / 1e3
        for carrier, capacity in su_capacity.items():
            if carrier in carrier_to_tech:
                tech = carrier_to_tech[carrier]
                realized[tech] = realized.get(tech, 0.0) + capacity
                logger.debug(f"  storage_units[{carrier}]: {capacity:.2f} GW -> {tech}")
    
    # Extract from links (p_nom_opt)
    if not n.links.empty:
        link_capacity = n.links.groupby("carrier")["p_nom_opt"].sum() / 1e3
        for carrier, capacity in link_capacity.items():
            if carrier in carrier_to_tech:
                tech = carrier_to_tech[carrier]
                realized[tech] = realized.get(tech, 0.0) + capacity
                logger.debug(f"  links[{carrier}]: {capacity:.2f} GW -> {tech}")
    
    # Extract from stores (e_nom_opt) - energy capacity
    if not n.stores.empty:
        store_capacity = n.stores.groupby("carrier")["e_nom_opt"].sum() / 1e3  # MWh to GWh
        for carrier, capacity in store_capacity.items():
            if carrier in carrier_to_tech:
                tech = carrier_to_tech[carrier]
                # Note: For energy storage, capacity is in GWh not GW
                # But we treat it consistently as "deployment" in learning curves
                realized[tech] = realized.get(tech, 0.0) + capacity
                logger.debug(f"  stores[{carrier}]: {capacity:.2f} GWh -> {tech}")
    
    logger.info(f"  Extracted capacity for {len(realized)} technologies")
    for tech, cap in realized.items():
        logger.info(f"    {tech}: {cap:.2f} GW")
    
    return realized


def update_learning_state(prev_state, realized_capacity):
    """
    Update learning state with realized deployment.
    
    Creates new state DataFrame with:
    - K_realized: Current realized capacity
    - K_prev: Previous horizon's capacity (from prev_state.K_realized)
    - L_realized: Cumulative deployment (prev_state.L_realized + ΔK)
    
    Args:
        prev_state: Previous learning state DataFrame
        realized_capacity: Dict of {tech: capacity_GW}
    
    Returns:
        New state DataFrame
    """
    logger.info("Updating learning state with realized deployment")
    
    new_state = []
    
    for tech in prev_state.index:
        K_prev = prev_state.loc[tech, "K_realized_GW"]
        L_prev = prev_state.loc[tech, "L_realized_GW"]
        
        K_realized = realized_capacity.get(tech, K_prev)
        
        dK_actual = K_realized - K_prev
        
        L_realized = L_prev + dK_actual
        
        new_state.append({
            "technology": tech,
            "L_realized_GW": L_realized,
            "K_realized_GW": K_realized,
            "K_prev_GW": K_prev,
        })
        
        logger.info(
            f"  {tech}: K_prev={K_prev:.1f}GW -> K_realized={K_realized:.1f}GW "
            f"(ΔK={dK_actual:.1f}GW), L={L_realized:.1f}GW"
        )
    
    return pd.DataFrame(new_state).set_index("technology")


def compare_with_predictions(new_state, pred_state_file):
    """
    Compare realized deployment with predictions for diagnostics.
    
    Args:
        new_state: New learning state DataFrame
        pred_state_file: Path to prediction_state_{year}.csv
    
    Returns:
        Comparison DataFrame
    """
    if not Path(pred_state_file).exists():
        logger.warning(f"Prediction state file not found: {pred_state_file}")
        return None
    
    logger.info(f"Comparing with predictions from {pred_state_file}")
    pred_state = pd.read_csv(pred_state_file, index_col="technology")
    
    comparison = []
    
    for tech in new_state.index:
        if tech not in pred_state.index:
            continue
        
        K_realized = new_state.loc[tech, "K_realized_GW"]
        K_prev = new_state.loc[tech, "K_prev_GW"]
        dK_actual = K_realized - K_prev
        
        dK_pred = pred_state.loc[tech, "dK_pred_GW"]
        L_pred = pred_state.loc[tech, "L_pred_GW"]
        L_realized = new_state.loc[tech, "L_realized_GW"]
        
        # Calculate forecast error
        dK_error = dK_actual - dK_pred
        dK_error_pct = (dK_error / dK_pred * 100) if dK_pred > 0 else 0.0
        
        L_error = L_realized - L_pred
        L_error_pct = (L_error / L_pred * 100) if L_pred > 0 else 0.0
        
        comparison.append({
            "technology": tech,
            "dK_pred_GW": dK_pred,
            "dK_actual_GW": dK_actual,
            "dK_error_GW": dK_error,
            "dK_error_pct": dK_error_pct,
            "L_pred_GW": L_pred,
            "L_realized_GW": L_realized,
            "L_error_GW": L_error,
            "L_error_pct": L_error_pct,
        })
        
        logger.info(
            f"  {tech}: ΔK pred={dK_pred:.1f}GW, actual={dK_actual:.1f}GW "
            f"(error={dK_error_pct:+.1f}%)"
        )
    
    return pd.DataFrame(comparison).set_index("technology")


def save_state(state, output_file):
    """Save learning state to CSV."""
    logger.info(f"Saving learning state to {output_file}")
    
    # Ensure directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    state.to_csv(output_file)


def save_comparison(comparison, output_file):
    """Save forecast comparison to CSV."""
    if comparison is None or comparison.empty:
        logger.info("No comparison data to save")
        return
    
    logger.info(f"Saving forecast comparison to {output_file}")
    
    # Ensure directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    comparison.to_csv(output_file)


def main(snakemake):

    configure_logging(snakemake)
    
    year = int(snakemake.wildcards.planning_horizons)
    logger.info("=" * 70)
    logger.info(f"Capturing realized deployment for horizon {year}")
    logger.info("=" * 70)
    
    # Load configuration
    learning_cfg = load_config_learning(snakemake.input.learning_config)
    
    # Check if learning is enabled
    if not learning_cfg.get("enabled", False):
        logger.info("Learning is DISABLED - creating empty state file")
        
        # Create empty state file
        Path(snakemake.output.state).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(snakemake.output.state)
        
        if snakemake.output.get("comparison"):
            Path(snakemake.output.comparison).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame().to_csv(snakemake.output.comparison)
        return
    
    logger.info("Learning is ENABLED")
    
    # Determine if this is the first horizon
    planning_horizons = snakemake.params.planning_horizons
    horizons = sorted(planning_horizons)
    if year == horizons[0]:
        logger.info(f"First horizon ({year}) - no state to capture (using initialized state)")
        
        # Just copy the initial state file
        prev_state = load_previous_state(snakemake.input.prev_state)
        save_state(prev_state, snakemake.output.state)
        
        if snakemake.output.get("comparison"):
            Path(snakemake.output.comparison).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame().to_csv(snakemake.output.comparison)
        return
    
    # Load previous state
    prev_state = load_previous_state(snakemake.input.prev_state)
    
    # Extract realized capacity from solved network
    tech_mapping = learning_cfg["tech_mapping"]
    realized_capacity = extract_realized_capacity(snakemake.input.network, tech_mapping)
    
    # Update state
    new_state = update_learning_state(prev_state, realized_capacity)
    
    # Compare with predictions
    comparison = compare_with_predictions(new_state, snakemake.input.pred_state)
    
    # Save outputs
    save_state(new_state, snakemake.output.state)
    
    if snakemake.output.get("comparison"):
        save_comparison(comparison, snakemake.output.comparison)
    
    logger.info("=" * 70)
    logger.info(f"Deployment capture completed for {year}")
    logger.info("=" * 70)


if __name__ == "__main__":
    if "snakemake" in globals():
        main(snakemake)
    else:
        logger.error("This script must be run via Snakemake")
