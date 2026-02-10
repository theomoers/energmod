# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Created on Thu Jul 14 21:18:06 2022.

@author: user
"""

import logging
import os
from itertools import product

import pandas as pd
from _helpers import BASE_DIR, mock_snakemake, read_csv_nafix

_logger = logging.getLogger(__name__)


def calculate_end_values(df, base_year, planning_horizon, countries=None):
    """
    Calculate end values from CAGRs.

    If the dataframe has no explicit ``year`` column, assume a single
    CAGR applying over the full period from ``base_year`` to
    ``planning_horizon`` and compute::

        (1 + cagr) ** no_years

    If a ``year`` column is present (new specification), interpret the
    rows as year‑specific CAGRs and compute cumulative growth by
    multiplying the growth factors for each period:

    - 2025: use 2025 CAGR for 2020‑2025  -> (1 + cagr_2025) ** 5
    - 2030: use 2025 and 2030 CAGRs     -> (1 + cagr_2025) ** 5 * (1 + cagr_2030) ** 5
    - 2050: use all CAGRs up to 2050    -> product over all 5‑year periods
    """
    base_year = int(base_year)
    planning_horizon = int(planning_horizon)

    # Old behaviour: no year column, single CAGR over the whole period
    if "year" not in df.columns:
        no_years = planning_horizon - base_year
        return (1 + df) ** no_years

    # New behaviour: year-specific CAGRs
    if countries is None:
        countries = sorted(df["country"].unique())

    sector_cols = [c for c in df.columns if c not in ["country", "year"]]

    # Separate out DEFAULT rows (used as fallback for missing countries/years)
    has_default = (df["country"] == "DEFAULT").any()
    if has_default:
        default_rows = df[df["country"] == "DEFAULT"].set_index("year")
    else:
        default_rows = pd.DataFrame(columns=sector_cols)

    # All CAGR years available in the table
    all_cagr_years = sorted(df["year"].unique())

    results = {}

    for country in countries:
        country_rows = df[df["country"] == country].set_index("year")

        # Determine which CAGR years are relevant up to the planning horizon
        years_needed = [y for y in all_cagr_years if base_year < y <= planning_horizon]

        total = pd.Series(1.0, index=sector_cols, dtype=float)
        prev_year = base_year

        for y in years_needed:
            period_years = y - prev_year
            if period_years <= 0:
                continue

            if y in country_rows.index:
                row = country_rows.loc[y, sector_cols]
            elif has_default and y in default_rows.index:
                row = default_rows.loc[y, sector_cols]
            else:
                raise KeyError(
                    f"No CAGR data for country '{country}' and year {y}, "
                    "and no matching DEFAULT row."
                )

            period_growth = (1 + row) ** period_years
            total *= period_growth
            prev_year = y

        results[country] = total

    result_df = pd.DataFrame.from_dict(results, orient="index")
    result_df.index.name = "country"
    return result_df


def country_to_nodal(industrial_production, keys):
    # keys["country"] = keys.index.str[:2]  # TODO 2digit_3_digit adaptation needed

    nodal_production = pd.DataFrame(
        index=keys.index, columns=industrial_production.columns, dtype=float
    )

    countries = keys.country.unique()
    sectors = industrial_production.columns

    for country, sector in product(countries, sectors):
        buses = keys.index[keys.country == country]

        if sector not in keys.columns or keys[sector].sum() == 0:
            mapping = "gdp"
        else:
            mapping = sector

        key = keys.loc[buses, mapping]
        # print(sector)
        nodal_production.loc[buses, sector] = (
            industrial_production.at[country, sector] * key
        )

    return nodal_production


def apply_industry_electrification(industry_base_totals, config, demand_scenario, planning_horizon):
    """
    Apply carrier-specific electrification shares to industry demand.
    
    Shifts energy from fossil carriers (coal, gas) to electricity and hydrogen
    based on scenario-specific shares defined in config.sectorelectrification.yaml.
    
    Parameters
    ----------
    industry_base_totals : pd.DataFrame
        MultiIndex DataFrame with (country, carrier) as index and subsectors as columns
    config : dict
        Configuration dictionary containing industry_electrification parameters
    demand_scenario : str
        Demand scenario identifier (e.g., 'AB', 'NZ', 'AP', 'BU', 'DF')
    planning_horizon : str
        Planning year (e.g., '2020', '2030', '2050')
    
    Returns
    -------
    pd.DataFrame
        Modified industry_base_totals with adjusted carrier shares
    """
    
    # Check if industry electrification config exists
    if 'sector' not in config:
        _logger.warning("No 'sector' key in config. Using base carrier mix.")
        return industry_base_totals
    
    if 'industry_electrification' not in config['sector']:
        _logger.warning("No 'industry_electrification' in config['sector']. Using base carrier mix.")
        return industry_base_totals
    
    elec_config = config['sector']['industry_electrification']
    
    # Check if this scenario is the default scenario - if so, skip electrification
    default_scenario = elec_config.get('default_scenario', None)
    if default_scenario is not None and demand_scenario == default_scenario:
        _logger.info(f"Scenario {demand_scenario} is the default scenario. Skipping industry electrification.")
        return industry_base_totals
    
    result = industry_base_totals.copy()
    
    _logger.warning(f"Applying industry electrification for {demand_scenario}_{planning_horizon}")
    
    # Define subsector mappings to config parameters
    subsector_mapping = {
        'iron and steel': ('steel_electric_share', 'steel_hydrogen_share'),
        'non-metallic minerals': ('cement_electric_share', 'cement_hydrogen_share'),
        'chemical and petrochemical': ('chemicals_electric_share', 'chemicals_hydrogen_share'),
    }
    
    # List of other industries that use generic shares
    other_industries = [
        'non-ferrous metals', 'transport equipment', 'machinery',
        'mining and quarrying', 'food and tobacco', 'paper pulp and print',
        'wood and wood products', 'textile and leather', 'construction', 'other'
    ]
    
    # Get energy efficiency factors from config
    # Ratios of energy required for new technology vs conventional (coal/gas) processes
    # Values < 1.0 mean new process is more efficient (requires less energy)
    if 'efficiency_factors' in elec_config:
        efficiency_factors = elec_config['efficiency_factors']
    else:
        raise KeyError("Missing 'efficiency_factors' in sectorelectrification config.")
    
    scenario_key = f"{demand_scenario}_{planning_horizon}"
    
    countries = result.index.get_level_values(0).unique()
    
    for country in countries:
        # Process explicitly mapped subsectors (steel, cement, chemicals)
        for subsector, (elec_param, h2_param) in subsector_mapping.items():
            if subsector not in result.columns:
                continue
                
            # Get electrification shares from config
            elec_share = elec_config.get(elec_param, {}).get(scenario_key, 0.0)
            h2_share = elec_config.get(h2_param, {}).get(scenario_key, 0.0)
            conventional_share = 1.0 - elec_share - h2_share
            
            if conventional_share < 0:
                _logger.warning(
                    f"Electrification shares for {subsector} in {scenario_key} sum to > 1.0. "
                    f"Normalizing: elec={elec_share}, h2={h2_share}"
                )
                total = elec_share + h2_share
                elec_share /= total
                h2_share /= total
                conventional_share = 0.0
            
            # Get current fossil fuel consumption for this subsector
            if (country, 'coal') in result.index:
                coal_energy = result.loc[(country, 'coal'), subsector]
            else:
                coal_energy = 0.0
                
            if (country, 'gas') in result.index:
                gas_energy = result.loc[(country, 'gas'), subsector]
            else:
                gas_energy = 0.0
            
            total_fossil = coal_energy + gas_energy
            
            if total_fossil == 0:
                continue
            
            # Calculate new energy demands with efficiency adjustments
            subsector_group = 'other' if subsector not in efficiency_factors['electricity'] else subsector
            
            new_elec_energy = (
                total_fossil * elec_share * efficiency_factors['electricity'].get(subsector_group, 1.0)
            )
            new_h2_energy = (
                total_fossil * h2_share * efficiency_factors['hydrogen'].get(subsector_group, 1.0)
            )
            
            # Update carriers
            if (country, 'electricity') in result.index:
                result.loc[(country, 'electricity'), subsector] += new_elec_energy
            else:
                # Ensure electricity carrier exists
                result.loc[(country, 'electricity'), subsector] = new_elec_energy
            
            if (country, 'hydrogen') in result.index:
                result.loc[(country, 'hydrogen'), subsector] += new_h2_energy
            else:
                result.loc[(country, 'hydrogen'), subsector] = new_h2_energy
            
            # Reduce fossil fuels proportionally
            if (country, 'coal') in result.index:
                result.loc[(country, 'coal'), subsector] = coal_energy * conventional_share
            if (country, 'gas') in result.index:
                result.loc[(country, 'gas'), subsector] = gas_energy * conventional_share
        
        # Process other industries with generic shares
        elec_share_other = elec_config.get('other_industries_electric_share', {}).get(scenario_key, 0.0)
        h2_share_other = elec_config.get('other_industries_hydrogen_share', {}).get(scenario_key, 0.0)
        conventional_share_other = 1.0 - elec_share_other - h2_share_other
        
        if conventional_share_other < 0:
            _logger.warning(
                f"Other industries electrification shares in {scenario_key} sum to > 1.0. Normalizing."
            )
            total = elec_share_other + h2_share_other
            elec_share_other /= total
            h2_share_other /= total
            conventional_share_other = 0.0
        
        for subsector in other_industries:
            if subsector not in result.columns:
                continue
            
            # Get current fossil fuel consumption
            coal_energy = result.loc[(country, 'coal'), subsector] if (country, 'coal') in result.index else 0.0
            gas_energy = result.loc[(country, 'gas'), subsector] if (country, 'gas') in result.index else 0.0
            total_fossil = coal_energy + gas_energy
            
            if total_fossil == 0:
                continue
            
            # Apply generic efficiency factors
            new_elec_energy = (
                total_fossil * elec_share_other * efficiency_factors['electricity']['other']
            )
            new_h2_energy = (
                total_fossil * h2_share_other * efficiency_factors['hydrogen']['other']
            )
            
            # Update carriers
            if (country, 'electricity') in result.index:
                result.loc[(country, 'electricity'), subsector] += new_elec_energy
            else:
                result.loc[(country, 'electricity'), subsector] = new_elec_energy
            
            if (country, 'hydrogen') in result.index:
                result.loc[(country, 'hydrogen'), subsector] += new_h2_energy
            else:
                result.loc[(country, 'hydrogen'), subsector] = new_h2_energy
            
            # Reduce fossil fuels
            if (country, 'coal') in result.index:
                result.loc[(country, 'coal'), subsector] = coal_energy * conventional_share_other
            if (country, 'gas') in result.index:
                result.loc[(country, 'gas'), subsector] = gas_energy * conventional_share_other
    
    _logger.info(
        f"Applied industry electrification for scenario {scenario_key}: "
        f"Steel elec={elec_config.get('steel_electric_share', {}).get(scenario_key, 0.0):.2%}, "
        f"Steel H2={elec_config.get('steel_hydrogen_share', {}).get(scenario_key, 0.0):.2%}"
    )
    
    return result


if __name__ == "__main__":
    if "snakemake" not in globals():
        
        snakemake = mock_snakemake(
            "build_industry_demand",
            simpl="",
            network="elec",
            clusters="200",
            ll="copt",
            opts="3h",
            planning_horizons="2050",
            sopts="72h",
            configfile="/shared/share_cki25/energymodels/pypsa-earth/config.myopic.yaml",
            discountrate=0.071,
            demand="AB",
            h2export="10"
        )

    countries = snakemake.params.countries

    gadm_clustering = snakemake.params.alternative_clustering

    if snakemake.params.industry_demand:
        _logger.info(
            "Fetching custom industry demand data.. expecting file at 'data/custom/industry_demand_{0}_{1}.csv'".format(
                snakemake.wildcards["demand"], snakemake.wildcards["planning_horizons"]
            )
        )

        industry_demand = pd.read_csv(
            os.path.join(
                BASE_DIR,
                "data/custom/industry_demand_{0}_{1}.csv".format(
                    snakemake.wildcards["demand"],
                    snakemake.wildcards["planning_horizons"],
                ),
            ),
            index_col=[0, 1],
        )
        keys_path = snakemake.input.industrial_distribution_key

        dist_keys = pd.read_csv(
            keys_path, index_col=0, keep_default_na=False, na_values=[""]
        )
        production_base = pd.DataFrame(
            1, columns=industry_demand.columns, index=countries
        )
        nodal_keys = country_to_nodal(production_base, dist_keys)

        # Apply industry electrification shares to custom demand data
        industry_demand = apply_industry_electrification(
            industry_demand,
            snakemake.config,
            snakemake.wildcards['demand'],
            snakemake.wildcards['planning_horizons']
        )

        nodal_df = pd.DataFrame()

        for country in countries:
            nodal_production_tom_co = nodal_keys[
                nodal_keys.index.to_series().str.startswith(country)
            ]
            industry_base_totals_co = industry_demand.loc[country]
            # final energy consumption per node and industry (TWh/a)
            nodal_df_co = nodal_production_tom_co.dot(industry_base_totals_co.T)
            nodal_df = pd.concat([nodal_df, nodal_df_co])

    else:
        base_year = int(snakemake.params.base_year)
        planning_horizon = int(snakemake.wildcards.planning_horizons)

        cagr = read_csv_nafix(
            snakemake.input.industry_growth_cagr,
            index_col=None,
        )

        growth_factors = calculate_end_values(
            cagr, base_year, planning_horizon, countries=countries
        )

        industry_base_totals = read_csv_nafix(
            snakemake.input["base_industry_totals"], index_col=[0, 1]
        )

        # Construct a base production dataframe (all ones) with the same
        # shape as the growth_factors, then scale by the growth factors.
        production_base = growth_factors.copy()
        production_base.loc[:, :] = 1.0
        production_tom = production_base * growth_factors

        # non-used line; commented out
        # industry_totals = (production_tom * industry_base_totals).fillna(0)

        industry_util_factor = snakemake.params.industry_util_factor

        # Load distribution keys
        keys_path = snakemake.input.industrial_distribution_key

        dist_keys = pd.read_csv(
            keys_path, index_col=0, keep_default_na=False, na_values=[""]
        )

        # production of industries per node compared to current
        nodal_production_tom = country_to_nodal(production_tom, dist_keys)

        clean_industry_list = [
            "iron and steel",
            "chemical and petrochemical",
            "non-ferrous metals",
            "non-metallic minerals",
            "transport equipment",
            "machinery",
            "mining and quarrying",
            "food and tobacco",
            "paper pulp and print",
            "wood and wood products",
            "textile and leather",
            "construction",
            "other",
        ]

        emission_factors = {  # Based on JR data following PyPSA-EUR
            "iron and steel": 0.025,
            "chemical and petrochemical": 0.51,  # taken from HVC including process and feedstock
            "non-ferrous metals": 1.5,  # taken from Aluminum primary
            "non-metallic minerals": 0.542,  # taken for cement
            "transport equipment": 0,
            "machinery": 0,
            "mining and quarrying": 0,  # assumed
            "food and tobacco": 0,
            "paper pulp and print": 0,
            "wood and wood products": 0,
            "textile and leather": 0,
            "construction": 0,  # assumed
            "other": 0,
        }

        # fill industry_base_totals
        level_2nd = industry_base_totals.index.get_level_values(1).unique()
        mlv_index = pd.MultiIndex.from_product([countries, level_2nd])
        industry_base_totals = industry_base_totals.reindex(mlv_index, fill_value=0)

        geo_locs = pd.read_csv(
            snakemake.input.industrial_database,
            sep=",",
            header=0,
            keep_default_na=False,
            index_col=0,
        )
        geo_locs["capacity"] = pd.to_numeric(geo_locs.capacity)

        def match_technology(df):
            industry_mapping = {
                "Integrated steelworks": "iron and steel",
                "DRI + Electric arc": "iron and steel",
                "Electric arc": "iron and steel",
                "Cement": "non-metallic minerals",
                "HVC": "chemical and petrochemical",
                "Paper": "paper pulp and print",
            }

            df["industry"] = df["technology"].map(industry_mapping)
            return df

        # Calculating emissions

        # get the subset of countries that al
        countries_geo = geo_locs.index.unique().intersection(countries)
        geo_locs = match_technology(geo_locs).loc[countries_geo]

        aluminium_year = snakemake.params.aluminium_year
        AL = read_csv_nafix(
            os.path.join(BASE_DIR, "data/AL_production.csv"), index_col=0
        )
        # Filter data for the given year and countries
        AL_prod_tom = AL.query("Year == @aluminium_year and index in @countries_geo")[
            "production[ktons/a]"
        ]

        # Check if aluminum data is missing for any countries
        for country in countries_geo:
            if country not in AL_prod_tom.index:
                _logger.warning(
                    f"No aluminum production data found for {country}. Filled with 0.0."
                )

        # Reindex and fill missing values with 0.0
        AL_prod_tom = AL_prod_tom.reindex(countries_geo, fill_value=0.0)

        # Estimate emissions for aluminum production and converting from ktons to tons
        AL_emissions = AL_prod_tom * emission_factors["non-ferrous metals"] * 1000

        Steel_emissions = (
            geo_locs[geo_locs.industry == "iron and steel"]
            .groupby("country")
            .sum()
            .capacity
            * 1000
            * emission_factors["iron and steel"]
            * industry_util_factor
        )
        NMM_emissions = (
            geo_locs[geo_locs.industry == "non-metallic minerals"]
            .groupby("country")
            .sum()
            .capacity
            * 1000
            * emission_factors["non-metallic minerals"]
            * industry_util_factor
        )
        refinery_emissons = (
            geo_locs[geo_locs.industry == "chemical and petrochemical"]
            .groupby("country")
            .sum()
            .capacity
            * emission_factors["chemical and petrochemical"]
            * 0.136
            * 365
            * industry_util_factor
        )

        for country in countries:
            industry_base_totals.loc[(country, "process emissions"), :] = 0
            try:
                industry_base_totals.loc[
                    (country, "process emissions"), "non-metallic minerals"
                ] = NMM_emissions.loc[country]
            except KeyError:
                pass

            try:
                industry_base_totals.loc[
                    (country, "process emissions"), "iron and steel"
                ] = Steel_emissions.loc[country]
            except KeyError:
                pass
            try:
                industry_base_totals.loc[
                    (country, "process emissions"), "non-ferrous metals"
                ] = AL_emissions.loc[country]
            except KeyError:
                pass
            try:
                industry_base_totals.loc[
                    (country, "process emissions"), "chemical and petrochemical"
                ] = refinery_emissons.loc[country]
            except KeyError:
                pass
        industry_base_totals = industry_base_totals.sort_index()

        all_carriers = [
            "electricity",
            "gas",
            "coal",
            "oil",
            "hydrogen",
            "biomass",
            "low-temperature heat",
        ]

        # Fill missing carriers with 0s
        for country in countries:
            carriers_present = industry_base_totals.xs(country, level=0).index
            missing_carriers = set(all_carriers) - set(carriers_present)
            for carrier in missing_carriers:
                # Add the missing carrier with a value of 0
                industry_base_totals.loc[(country, carrier), :] = 0

        # temporary fix: merge other manufacturing, construction and non-fuel into other and drop the column
        other_cols = list(set(industry_base_totals.columns) - set(clean_industry_list))
        if len(other_cols) > 0:
            industry_base_totals["other"] += industry_base_totals[other_cols].sum(
                axis=1
            )
            industry_base_totals.drop(columns=other_cols, inplace=True)

        # Apply industry electrification shares
        industry_base_totals = apply_industry_electrification(
            industry_base_totals,
            snakemake.config,
            snakemake.wildcards['demand'],
            snakemake.wildcards['planning_horizons']
        )

        nodal_df = pd.DataFrame()

        for country in countries:
            nodal_production_tom_co = nodal_production_tom[
                nodal_production_tom.index.to_series().str.startswith(country)
            ]
            industry_base_totals_co = industry_base_totals.loc[country]
            # final energy consumption per node and industry (TWh/a)
            nodal_df_co = nodal_production_tom_co.dot(industry_base_totals_co.T)
            nodal_df = pd.concat([nodal_df, nodal_df_co])

    rename_sectors = {
        "elec": "electricity",
        "biomass": "solid biomass",
        "heat": "low-temperature heat",
    }
    nodal_df.rename(columns=rename_sectors, inplace=True)

    nodal_df.index.name = "MWh/a (tCO2/a)"

    nodal_df.to_csv(
        snakemake.output.industrial_energy_demand_per_node, float_format="%.2f"
    )
