"""
Plot learning curve results and forecast accuracy.

Creates visualizations of:
1. Cost trajectories over planning horizons
2. Forecast vs. realized deployment comparison
"""

import logging
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Add parent directory to path for _helpers import
sys.path.insert(0, str(Path(__file__).parent))
from _helpers import mock_snakemake

logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Mapping from learning tech names to cost file technology names
TECH_TO_COST_NAME = {
    'solar_power': 'solar',
    'onwind_power': 'onwind',
    'battery_power': 'battery inverter',
    'battery_energy': 'battery storage',
    'electrolyser_power': 'electrolysis',
    'h2_energy': None,  # No direct match in cost files
}


def load_2020_baseline_costs(costs_2020_path, tech_mapping):
    """
    Load 2020 baseline costs from costs_2020.csv.
    
    Args:
        costs_2020_path: Path to costs_2020.csv file
        tech_mapping: Dict mapping carriers to learning technology keys
    
    Returns:
        Dict of {tech_key: overnight_cost_EUR_per_kW}
    """
    if not Path(costs_2020_path).exists():
        logger.warning(f"2020 cost file not found: {costs_2020_path}")
        return {}
    
    logger.info(f"Loading 2020 baseline costs from {costs_2020_path}")
    
    # Read the costs file
    df = pd.read_csv(costs_2020_path)
    
    # Filter for investment parameter
    investment_df = df[df['parameter'] == 'investment'].copy()
    
    baseline_costs = {}
    
    for tech_key, cost_name in TECH_TO_COST_NAME.items():
        if cost_name is None:
            continue
            
        # Find matching technology
        tech_data = investment_df[investment_df['technology'] == cost_name]
        
        if tech_data.empty:
            logger.warning(f"  No 2020 cost data for {tech_key} (looking for '{cost_name}')")
            continue
        
        value = tech_data.iloc[0]['value']
        unit = tech_data.iloc[0]['unit']
        
        # Convert to EUR/kW
        if 'kW' in unit:
            cost_per_kw = value
        elif 'kWh' in unit:
            # For energy storage, keep as EUR/kWh (we'll handle this in plotting)
            cost_per_kw = value
        else:
            logger.warning(f"  Unexpected unit for {tech_key}: {unit}")
            continue
        
        baseline_costs[tech_key] = cost_per_kw
        logger.info(f"  {tech_key}: {cost_per_kw:.2f} {unit}")
    
    return baseline_costs


def plot_cost_trajectories(cost_logs, output_path, baseline_costs=None):
    """
    Plot cost trajectories for all technologies across planning horizons.
    
    Args:
        cost_logs: Dict mapping year -> DataFrame of cost log
        output_path: Path to save the figure
        baseline_costs: Dict of {tech_key: overnight_cost_EUR_per_kW} for 2020
    """
    if not cost_logs:
        logger.warning("No cost logs to plot")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    # Get all unique technologies from learning data (from DataFrame indices)
    all_techs = set()
    for df in cost_logs.values():
        if not df.empty:
            all_techs.update(df.index)
    
    if not all_techs:
        logger.warning("No technologies found in cost logs")
        return
    
    # Plot overnight costs over time
    for tech in sorted(all_techs):
        years = []
        overnight_costs = []
        
        # Add 2020 baseline if available AND we have learning data for this tech
        if baseline_costs and tech in baseline_costs:
            years.append(2020)
            overnight_costs.append(baseline_costs[tech])
        
        for year in sorted(cost_logs.keys()):
            df = cost_logs[year]
            if tech in df.index:
                tech_data = df.loc[tech]
                years.append(year)
                    
                if 'c_overnight_pred' in tech_data.index:
                    overnight_costs.append(tech_data['c_overnight_pred'])
                else:
                    overnight_costs.append(np.nan)
        
        if years:
            ax.plot(years, overnight_costs, marker='o', linewidth=2.5,
                    markersize=10, label=tech, alpha=0.8)
    
    ax.set_xlabel('Planning Horizon', fontsize=13, fontweight='bold')
    ax.set_ylabel('Overnight Cost [€/kW or €/kWh]', fontsize=13, fontweight='bold')
    ax.set_title('Technology Overnight Cost Trajectories', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved cost trajectories plot to {output_path}")


def plot_forecast_comparison(forecast_comparisons, output_path):
    """
    Plot forecast vs. realized deployment for all technologies.
    
    Args:
        forecast_comparisons: Dict mapping year -> DataFrame of forecast comparison
        output_path: Path to save the figure
    """
    if not forecast_comparisons:
        logger.warning("No forecast comparisons to plot")
        return
    
    # Combine all forecasts
    all_data = []
    for year, df in forecast_comparisons.items():
        if not df.empty:
            df_copy = df.copy()
            df_copy['target_year'] = year
            # Reset index to get technology as a column for easier plotting
            df_copy = df_copy.reset_index()
            all_data.append(df_copy)
    
    if not all_data:
        logger.warning("No forecast comparison data found")
        return
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Get unique technologies (now it's a column after reset_index)
    if 'technology' not in combined.columns:
        # Handle case where index name wasn't 'technology'
        combined.rename(columns={combined.columns[0]: 'technology'}, inplace=True)
    techs = combined['technology'].unique()
    n_techs = len(techs)
    
    # Create subplots
    n_cols = min(3, n_techs)
    n_rows = int(np.ceil(n_techs / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for idx, tech in enumerate(sorted(techs)):
        ax = axes[idx]
        tech_data = combined[combined['technology'] == tech].sort_values('target_year')
        
        years = tech_data['target_year'].values
        
        # Safely extract predicted and realized values
        predicted = tech_data['dK_pred_GW'].values if 'dK_pred_GW' in tech_data.columns else np.zeros(len(years))
        realized = tech_data['dK_actual_GW'].values if 'dK_actual_GW' in tech_data.columns else np.zeros(len(years))
        
        x = np.arange(len(years))
        width = 0.35
        
        ax.bar(x - width/2, predicted, width, label='Predicted', 
               alpha=0.8, color='skyblue', edgecolor='black')
        ax.bar(x + width/2, realized, width, label='Realized',
               alpha=0.8, color='coral', edgecolor='black')
        
        # Add error percentages as text
        for i, (pred, real) in enumerate(zip(predicted, realized)):
            if pred != 0:
                error_pct = 100 * (real - pred) / pred
                ax.text(i, max(pred, real) * 1.05, f'{error_pct:+.1f}%',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Target Year', fontsize=11, fontweight='bold')
        ax.set_ylabel('Deployment [GW]', fontsize=11, fontweight='bold')
        ax.set_title(f'{tech}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(years)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Hide empty subplots
    for idx in range(n_techs, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved forecast comparison plot to {output_path}")


def plot_cumulative_deployment(forecast_comparisons, output_path):
    """
    Plot cumulative deployment trajectories (L_realized_GW) over time.
    
    Args:
        forecast_comparisons: Dict mapping year -> DataFrame of forecast comparison
        output_path: Path to save the figure
    """
    if not forecast_comparisons:
        logger.warning("No forecast comparisons to plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Combine all data
    all_data = []
    for year, df in forecast_comparisons.items():
        if not df.empty:
            df_copy = df.copy()
            df_copy['target_year'] = year
            # Reset index to get technology as a column
            df_copy = df_copy.reset_index()
            all_data.append(df_copy)
    
    if not all_data:
        logger.warning("No cumulative deployment data found")
        return
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Ensure technology column exists
    if 'technology' not in combined.columns:
        combined.rename(columns={combined.columns[0]: 'technology'}, inplace=True)
    
    # Plot each technology
    for tech in sorted(combined['technology'].unique()):
        tech_data = combined[combined['technology'] == tech].sort_values('target_year')
        
        years = tech_data['target_year'].values
        
        # Safely extract cumulative deployment
        if 'L_realized_GW' not in tech_data.columns:
            logger.warning(f"L_realized_GW column missing for {tech}, skipping")
            continue
            
        cumulative = tech_data['L_realized_GW'].values
        
        ax.plot(years, cumulative, marker='o', linewidth=2.5, 
               markersize=10, label=tech, alpha=0.8)
    
    ax.set_xlabel('Planning Horizon', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative Deployment [GW]', fontsize=13, fontweight='bold')
    ax.set_title('Technology Cumulative Deployment Trajectories', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved cumulative deployment plot to {output_path}")


def main(snakemake):
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s'
    )
    
    # Load 2020 baseline costs
    costs_2020_path = snakemake.input.basecost
    baseline_costs = {}
    if costs_2020_path:
        tech_mapping = snakemake.params.get('tech_mapping', {})
        baseline_costs = load_2020_baseline_costs(costs_2020_path, tech_mapping)
    else:
        logger.warning("No costs_2020 file provided in snakemake inputs")
    
    # Load all cost logs
    cost_logs = {}
    for year, cost_log_path in zip(snakemake.params.planning_horizons, 
                                     snakemake.input.cost_logs):
        if Path(cost_log_path).exists():
            # Load with technology as index to match how it was saved
            df = pd.read_csv(cost_log_path, index_col=0)
            if not df.empty:
                cost_logs[year] = df
        else:
            logger.warning(f"Cost log not found: {cost_log_path}")
    
    # Load all forecast comparisons
    forecast_comparisons = {}
    for year, forecast_path in zip(snakemake.params.planning_horizons,
                                     snakemake.input.forecast_comparisons):
        if Path(forecast_path).exists():
            # Load with technology as index to match how it was saved
            df = pd.read_csv(forecast_path, index_col=0)
            if not df.empty:
                forecast_comparisons[year] = df
        else:
            logger.warning(f"Forecast comparison not found: {forecast_path}")
    
    # Create output directory
    output_dir = Path(snakemake.output.cost_trajectories).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    logger.info("Generating cost trajectory plots...")
    plot_cost_trajectories(cost_logs, snakemake.output.cost_trajectories, baseline_costs)
    
    logger.info("Generating forecast comparison plots...")
    plot_forecast_comparison(forecast_comparisons, snakemake.output.forecast_comparison)
    
    logger.info("Generating cumulative deployment plots...")
    plot_cumulative_deployment(forecast_comparisons, snakemake.output.cumulative_deployment)
    
    logger.info("All learning plots generated successfully")


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "plot_learning_results",
            simpl="",
            clusters="110",
            ll="copt",
            opts="1h",
            planning_horizons="2030",
            sopts="1h",
            configfile="/shared/share_cki25/energymodels/pypsa-earth/config.myopic.yaml",
            discountrate=0.071,
            demand="AB",
            h2export="0.0"
        )
    main(snakemake)
