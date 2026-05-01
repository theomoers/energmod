#!/apps/anaconda3/bin/python
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Adds existing power and heat generation capacities for initial planning
horizon.
"""

import logging
import os
import re
from types import SimpleNamespace

import country_converter as coco
import numpy as np
import pandas as pd
import powerplantmatching as pm
import pypsa
import xarray as xr
import validation as _validation_hooks

# from _helpers import (
#     configure_logging,
#     set_scenario_config,
#     update_config_from_wildcards,
# )
# from add_electricity import sanitize_carriers
from prepare_sector_network import define_spatial, prepare_costs  # , cluster_heat_buses
from _helpers import annuity

logger = logging.getLogger(__name__)
cc = coco.CountryConverter()
idx = pd.IndexSlice
spatial = SimpleNamespace()


# Centralized profile fallback hook (kept outside this core script for easier reversion).
if hasattr(_validation_hooks, "_replace_zero_profile_columns_with_nearest"):
    _profile_col_bus = _validation_hooks._profile_col_bus
    _replace_zero_profile_columns_with_nearest = _validation_hooks._replace_zero_profile_columns_with_nearest
    logger.info("Using centralized profile fallback hook from scripts/validation.py")


def load_country_waccs(wacc_path):
    """
    Load country-specific WACCs for renewable technologies.
    
    Parameters
    ----------
    wacc_path : str
        Path to the WACC CSV file
    
    Returns
    -------
    dict
        Nested dictionary: {technology: {country_code: wacc_decimal}}
    """
    try:
        wacc_df = pd.read_csv(wacc_path)
    except FileNotFoundError:
        logger.warning(f"WACC file not found at {wacc_path}. Using default discount rates.")
        return {}
    
    wacc_dict = {}
    for tech in ['solar', 'onwind', 'offwind']:
        if tech not in wacc_df.columns:
            continue
        wacc_dict[tech] = dict(zip(
            wacc_df['country'],
            wacc_df[tech] / 100.0  # Convert percentage to decimal
        ))
    
    return wacc_dict


def get_regional_capital_cost(bus, carrier, costs, wacc_dict, Nyears):
    """
    Calculate capital cost with regional WACC for renewable technologies.
    
    Parameters
    ----------
    bus : str
        Bus name (contains country code)
    carrier : str
        Technology carrier (solar, onwind, offwind-ac, offwind-dc)
    costs : pd.DataFrame
        Cost assumptions
    wacc_dict : dict
        Country-specific WACCs
    Nyears : float
        Number of years for annualization
    
    Returns
    -------
    float
        Capital cost in EUR/MW/year
    """
    # Map carriers to WACC technology names
    carrier_to_wacc_tech = {
        'solar': 'solar',
        'onwind': 'onwind',
        'offwind-ac': 'offwind',
        'offwind-dc': 'offwind',
    }
    
    # Check if this carrier uses regional WACC
    if carrier not in carrier_to_wacc_tech or not wacc_dict:
        # Use default fixed cost
        cost_key = carrier if carrier in costs.index else carrier.replace('-ac', '').replace('-dc', '')
        return costs.at[cost_key, "fixed"]
    
    wacc_tech = carrier_to_wacc_tech[carrier]
    if wacc_tech not in wacc_dict:
        cost_key = carrier if carrier in costs.index else carrier.replace('-ac', '').replace('-dc', '')
        return costs.at[cost_key, "fixed"]
    
    # Extract country code from bus name
    country_code = str(bus).split(" ")[0][:2]
    
    # Get country-specific WACC or use default
    country_waccs = wacc_dict[wacc_tech]
    if country_code in country_waccs:
        wacc = country_waccs[country_code]
        
        # Get cost parameters
        cost_key = carrier if carrier in costs.index else carrier.replace('-ac', '').replace('-dc', '')
        investment = costs.at[cost_key, 'investment']
        lifetime = costs.at[cost_key, 'lifetime']
        fom = costs.at[cost_key, 'FOM']
        
        # Recalculate capital cost with regional WACC
        annuity_factor = annuity(lifetime, wacc) + fom / 100.0
        return annuity_factor * investment * Nyears
    else:
        # Fallback to default
        cost_key = carrier if carrier in costs.index else carrier.replace('-ac', '').replace('-dc', '')
        return costs.at[cost_key, "fixed"]


def add_build_year_to_new_assets(n, baseyear):
    """
    Parameters
    ----------
    n : pypsa.Network
    baseyear : int
        year in which optimized assets are built
    """
    # Give assets with lifetimes and no build year the build year baseyear
    for c in n.iterate_components(["Link", "Generator", "Store"]):
        assets = c.df.index[(c.df.lifetime != np.inf) & (c.df.build_year == 0)]
        c.df.loc[assets, "build_year"] = baseyear

        # add -baseyear to name
        rename = pd.Series(c.df.index, c.df.index)
        rename[assets] += f"-{str(baseyear)}"
        c.df.rename(index=rename, inplace=True)

        # rename time-dependent
        selection = n.component_attrs[c.name].type.str.contains(
            "series"
        ) & n.component_attrs[c.name].status.str.contains("Input")
        for attr in n.component_attrs[c.name].index[selection]:
            c.pnl[attr] = c.pnl[attr].rename(columns=rename)


def add_existing_renewables(df_agg, n, costs, wacc_dict, Nyears):
    """
    Append existing renewables to the df_agg pd.DataFrame with the conventional
    power plants.
    """
    tech_map = {"solar": ("PV", "solar"),
                "onwind": ("Onshore", "onwind"),
                "offwind": ("Offshore", "offwind-ac")}  # map to network label

    countries = snakemake.config["countries"]
    irena = pm.data.IRENASTAT().powerplant.convert_country_to_alpha2()
    irena = irena.query("Country in @countries")
    irena = irena.groupby(["Technology", "Country", "Year"]).Capacity.sum()

    irena = irena.unstack().reset_index()
    
    # Create country-to-cluster mapping from busmap
    # This ensures capacities from merged countries are preserved
    n_pre_cluster = pypsa.Network(snakemake.input.n_pre_cluster)
    bm = pd.read_csv(snakemake.input.busmap)
    # ensure comparable types: convert busmap Bus ids to strings to match n.buses.index (which are strings)
    bm['Bus'] = bm['Bus'].astype(str)

    clustered = bm.set_index('Bus').reindex(n_pre_cluster.buses.index)['busmap'].values

    busmapping = pd.DataFrame({
        "bus": n_pre_cluster.buses.index,
        "country_before_clustering": n_pre_cluster.buses['country'].values,
        "clustered_country": clustered
    })

    country_to_cluster = busmapping.groupby('country_before_clustering')['clustered_country'].first().to_dict()
    
    # Also create a mapping from cluster bus to all buses that belong to it in the clustered network
    # We'll use this to distribute capacity
    cluster_to_buses = busmapping.groupby('clustered_country')['bus'].apply(set).to_dict()
    
    # Get AC/DC buses from the network
    elec_buses = n.buses.index[n.buses.carrier == "AC"].union(
        n.buses.index[n.buses.carrier == "DC"]
    )

    for carrier_key, (tech, carrier_label) in tech_map.items():
        df = (
            irena[irena.Technology.str.contains(tech)]
            .drop(columns=["Technology"])
            .set_index("Country")
            .reindex(countries, fill_value=0.0)
            .fillna(0.0)
        )
        df.columns = df.columns.astype(int)

        # calculate yearly differences
        df.insert(loc=0, value=0.0, column="1999")
        df = df.diff(axis=1).drop("1999", axis=1).clip(lower=0)
        
        # Aggregate capacity by cluster country (this preserves capacity from merged countries)
        df['cluster_country'] = df.index.map(country_to_cluster)
        
        # Check for countries in IRENA data but not in network
        missing_countries = df[df['cluster_country'].isna()].index.tolist()
        if missing_countries:
            missing_capacities = df.loc[missing_countries].drop(columns=['cluster_country'])
            total_missing = missing_capacities.sum(axis=1)
            logger.warning(f"Found {len(missing_countries)} countries in IRENA {carrier_key} data not in network mapping:")
            for country in missing_countries:
                total_cap = total_missing[country]
                if total_cap > 0:
                    logger.warning(f"  {country}: {total_cap:.1f} MW total capacity")
        
        # Drop countries not in mapping before grouping, and exclude cluster_country column from sum
        df_to_cluster = df[df['cluster_country'].notna()].copy()
        df_clustered = df_to_cluster.drop(columns=['cluster_country']).groupby(df_to_cluster['cluster_country']).sum()
        df_clustered = df_clustered.fillna(0.0)

        # distribute capacities among nodes according to capacity factor
        # weighting with nodal_fraction
        nodal_fraction = pd.Series(0.0, elec_buses)

        for country in n.buses.loc[elec_buses, "country"].unique():
            gens = n.generators.index[
                (n.generators.index.str[:2] == country)
                & (n.generators.carrier == carrier_label)
            ]
            if len(gens) == 0:
                continue
            cfs = n.generators_t.p_max_pu[gens].mean()
            if cfs.sum() <= 0:
                continue
            cfs_key = cfs / cfs.sum()
            nodal_fraction.loc[n.generators.loc[gens, "bus"]] = cfs_key.groupby(
                n.generators.loc[gens, "bus"]
            ).sum()

        # Use clustered capacity dataframe
        # For each bus, look up its country's cluster country to get total capacity
        # Then distribute among all buses of that country weighted by nodal_fraction
        nodal_df = pd.DataFrame(0.0, index=elec_buses, columns=df_clustered.columns)
        
        # Track which cluster countries have been distributed
        distributed_clusters = set()
        
        # Iterate over all cluster countries that have capacity, then find which buses belong to that cluster
        for cluster_country in df_clustered.index:
            # The cluster_country (e.g., "AE 0") IS a bus in the post-cluster network
            # Check if it exists as an elec bus
            if cluster_country in elec_buses:
                # This cluster country is itself a bus in the post-cluster network
                country_buses = [cluster_country]
            else:
                countries_for_cluster = [c for c, cc in country_to_cluster.items() if cc == cluster_country]
                raise ValueError(f"Cluster country {cluster_country} not found in post-cluster elec buses. Original countries mapping to it: {countries_for_cluster}")
            
            distributed_clusters.add(cluster_country)
            
            # Get nodal fractions for these buses and normalize them to sum to 1 within the country
            country_nodal_fraction = nodal_fraction.loc[country_buses]
            total_fraction = country_nodal_fraction.sum()
            
            # If total fraction is 0, distribute equally; otherwise normalize
            if total_fraction > 0:
                country_nodal_fraction = country_nodal_fraction / total_fraction
            else:
                country_nodal_fraction = pd.Series(1.0 / len(country_buses), index=country_buses)
            
            # Distribute cluster country capacity among country buses weighted by normalized nodal_fraction
            for bus in country_buses:
                nodal_df.loc[bus] = df_clustered.loc[cluster_country] * country_nodal_fraction.loc[bus]

        nodal_df = nodal_df.fillna(0.0)
        
        # Verify all cluster countries were distributed
        undistributed_clusters = set(df_clustered.index) - distributed_clusters
        if undistributed_clusters:
            undistributed_capacity = df_clustered.loc[list(undistributed_clusters)].sum().sum()
            error_msg = f"Found {len(undistributed_clusters)} cluster countries in IRENA {carrier_key} data not distributed to any network buses (total: {undistributed_capacity:.1f} MW):\n"
            for cluster in undistributed_clusters:
                total_cap = df_clustered.loc[cluster].sum()
                if total_cap > 0:
                    error_msg += f"  {cluster}: {total_cap:.1f} MW\n"
            raise ValueError(error_msg)

        for year in nodal_df.columns:
            for node in nodal_df.index:
                name = f"{node}-{carrier_key}-{year}"  # only for temp index key
                capacity = nodal_df.loc[node, year]
                if capacity > 0.0:
                    df_agg.at[name, "Fueltype"] = carrier_key
                    df_agg.at[name, "Capacity"] = capacity
                    df_agg.at[name, "DateIn"] = year
                    df_agg.at[name, "lifetime"] = costs.at[carrier_key, "lifetime"]
                    df_agg.at[name, "DateOut"] = (
                        year + costs.at[carrier_key, "lifetime"] - 1
                    )
                    df_agg.at[name, "cluster_bus"] = node


def add_power_capacities_installed_before_baseyear(n, grouping_years, costs, baseyear):
    """
    Parameters
    ----------
    n : pypsa.Network
    grouping_years :
        intervals to group existing capacities
    costs :
        to read lifetime to estimate YearDecomissioning
    baseyear : int
    """
    logger.debug(
        f"Adding power capacities installed before {baseyear} from powerplants.csv"
    )
    
    def get_cost_key(carrier_label):
        """
        Map carrier_label to the correct cost database key.
        For offwind-ac and offwind-dc, costs are accessed using "offwind" technology.
        """
        if carrier_label in ["offwind-ac", "offwind-dc"]:
            return "offwind"
        return carrier_label

    df_agg = pd.read_csv(snakemake.input.powerplants, index_col=0)

    rename_fuel = {
        "Hard Coal": "coal",
        "Lignite": "lignite",
        "Nuclear": "nuclear",
        "Oil": "oil",
        "OCGT": "OCGT",
        "CCGT": "CCGT",
        "Bioenergy": "biomass",
    }

    # Replace Fueltype "Natural Gas" with the respective technology (OCGT or CCGT)
    df_agg.loc[df_agg["Fueltype"] == "Natural Gas", "Fueltype"] = df_agg.loc[
        df_agg["Fueltype"] == "Natural Gas", "Technology"
    ]

    fueltype_to_drop = [
        "Hydro",
        "Wind",
        "Solar",
        "Geothermal",
        "Waste",
        "Other",
        "CCGT, Thermal",
    ]

    technology_to_drop = ["Pv", "Storage Technologies"]

    # drop unused fueltyps and technologies
    df_agg.drop(df_agg.index[df_agg.Fueltype.isin(fueltype_to_drop)], inplace=True)
    df_agg.drop(df_agg.index[df_agg.Technology.isin(technology_to_drop)], inplace=True)
    df_agg.Fueltype = df_agg.Fueltype.map(rename_fuel)

    # Intermediate fix for DateIn & DateOut
    # Fill missing DateIn
    # TODO: revise CHP
    biomass_i = df_agg.loc[df_agg.Fueltype == "biomass"].index
    if biomass_i.empty:
        mean = 0
    else:
        logger.info(f"Filling missing DateIn for biomass CHP with mean build year of existing biomass CHP plants")
        mean = df_agg.loc[biomass_i, "DateIn"].mean()
        logger.info(f"Mean build year for biomass CHP: {mean:.1f} based on {len(biomass_i) - df_agg.loc[biomass_i, 'DateIn'].isna().sum()} plants with valid data")
    df_agg.loc[biomass_i, "DateIn"] = df_agg.loc[biomass_i, "DateIn"].fillna(int(mean))
    # Fill missing DateOut
    dateout = (
        df_agg.loc[biomass_i, "DateIn"]
        + snakemake.params.costs["fill_values"]["lifetime"]
    )
    df_agg.loc[biomass_i, "DateOut"] = df_agg.loc[biomass_i, "DateOut"].fillna(dateout)

    # drop assets which are already phased out / decommissioned
    phased_out = df_agg[df_agg["DateOut"] < baseyear].index
    df_agg.drop(phased_out, inplace=True)

    # assign clustered bus
    busmap_s = pd.read_csv(snakemake.input.busmap_s, index_col=0).squeeze()
    busmap = pd.read_csv(snakemake.input.busmap, index_col=0).squeeze()

    inv_busmap = {}
    for k, v in busmap.items():
        inv_busmap[v] = inv_busmap.get(v, []) + [k]

    clustermaps = busmap_s.map(busmap)
    clustermaps.index = clustermaps.index.astype(int)

    df_agg["cluster_bus"] = df_agg.bus.map(clustermaps)

    # include renewables in df_agg
    add_existing_renewables(df_agg, n, costs, wacc_dict, Nyears)

    df_agg = df_agg[df_agg["DateIn"] <= baseyear]

    newer_assets = (df_agg.DateIn > max(grouping_years)).sum()
    if newer_assets:
        logger.warning(
            f"There are {newer_assets} assets with build year "
            f"after last power grouping year {max(grouping_years)}. "
            "These assets are dropped and not considered."
            "Consider to redefine the grouping years to keep them."
        )
        to_drop = df_agg[df_agg.DateIn > max(grouping_years)].index
        df_agg.drop(to_drop, inplace=True)

    df_agg["grouping_year"] = np.take(
        grouping_years, np.digitize(df_agg.DateIn, grouping_years, right=True)
    )

    # calculate (adjusted) remaining lifetime before phase-out (+1 because assuming
    # phase out date at the end of the year)
    df_agg["lifetime"] = df_agg.DateOut - df_agg["grouping_year"] + 1

    df = df_agg.pivot_table(
        index=["grouping_year", "Fueltype"],
        columns="cluster_bus",
        values="Capacity",
        aggfunc="sum",
    )

    lifetime = df_agg.pivot_table(
        index=["grouping_year", "Fueltype"],
        columns="cluster_bus",
        values="lifetime",
        aggfunc="mean",  # currently taken mean for clustering lifetimes
    )

    carrier = {
        "OCGT": "gas",
        "CCGT": "gas",
        "coal": "coal",
        "oil": "oil",
        "lignite": "lignite",
        "nuclear": "uranium",
        "biomass": "biomass",
    }

    for grouping_year, generator in df.index:
        # capacity is the capacity in MW at each node for this
        capacity = df.loc[grouping_year, generator]
        capacity = capacity[~capacity.isna()]
        capacity = capacity[
            capacity > snakemake.params.existing_capacities["threshold_capacity"]
        ]
        suffix = "-ac" if generator == "offwind" else ""
        carrier_label = generator + suffix
        name_suffix = f" {generator}{suffix}-{grouping_year}"
        asset_i = capacity.index + name_suffix

        if generator in ["solar", "onwind", "offwind"]:
            # For renewables, check existing capacity vs external data (irena) for this specific grouping_year
            # Only process if this grouping_year is at or before the baseyear
            # (future years don't have existing generators yet)
            carrier_match = (
                ["offwind-ac", "offwind-dc"]
                if generator == "offwind"
                else [carrier_label]
            )
            existing_renewable_gens = n.generators.index[
                (n.generators.build_year == grouping_year) & 
                (n.generators.carrier.isin(carrier_match))
            ]
            
            if not existing_renewable_gens.empty and grouping_year <= baseyear:
                logger.info(f"Found {len(existing_renewable_gens)} existing {generator} generators from {grouping_year}, comparing with external data")
                
                existing_capacity_by_bus = n.generators.loc[existing_renewable_gens].groupby('bus')['p_nom'].sum()
                
                buses_to_adjust = capacity.index.intersection(existing_capacity_by_bus.index)
                buses_to_add = capacity.index.difference(existing_capacity_by_bus.index)
                buses_to_remove = existing_capacity_by_bus.index.difference(capacity.index)
                
                # Scale down existing generators that don't have IRENA data (set to 0 instead of removing)
                if not buses_to_remove.empty:
                    gens_to_zero = existing_renewable_gens[n.generators.loc[existing_renewable_gens, 'bus'].isin(buses_to_remove)]
                    total_removed_capacity = n.generators.loc[gens_to_zero, 'p_nom'].sum()
                    logger.info(f"Setting {len(gens_to_zero)} existing {generator} generators to zero capacity (no IRENA data for year {grouping_year}, total: {total_removed_capacity:.1f} MW)")
                    n.generators.loc[gens_to_zero, 'p_nom'] = 0.0
                    n.generators.loc[gens_to_zero, 'p_nom_min'] = 0.0
                
                for bus in buses_to_adjust:
                    external_capacity = capacity[bus]
                    existing_capacity = existing_capacity_by_bus[bus]
                    
                    if existing_capacity != external_capacity:
                        # Scale existing generators at this bus to match IRENA data
                        scaling_factor = external_capacity / existing_capacity
                        gens_at_bus = existing_renewable_gens[n.generators.loc[existing_renewable_gens, 'bus'] == bus]
                        
                        logger.debug(f"Adjusting {generator} capacity at {bus} for year {grouping_year}: {existing_capacity:.1f} MW -> {external_capacity:.1f} MW (factor: {scaling_factor:.3f})")
                        n.generators.loc[gens_at_bus, 'p_nom'] *= scaling_factor
                        n.generators.loc[gens_at_bus, 'p_nom_min'] = n.generators.loc[gens_at_bus, 'p_nom']
                    
                    capacity = capacity.drop(bus)
                
                # Only add new generators for buses that don't exist yet for this year
                new_capacity = capacity[buses_to_add]
            else:
                # No existing generators for this grouping_year, add all as new
                new_capacity = capacity
                logger.info(f"No existing {generator} generators found for year {grouping_year}, will add {len(new_capacity)} new generators")
            
            cost_key = get_cost_key(carrier_label)
            marginal_cost = costs.at[cost_key, "VOM"]

            # Only add new generators if there's new capacity to add
            if not new_capacity.empty:
                if "m" in snakemake.wildcards.clusters:
                    for ind in new_capacity.index:
                        # existing capacities are split evenly among regions in every country
                        inv_ind = list(inv_busmap[ind])

                        # Build names once and reuse the exact list
                        names = [i + name_suffix for i in inv_ind]
                        
                        p_max_pu = n.generators_t.p_max_pu[
                            capacity.index + f" {generator}{suffix}-{baseyear}"
                        ]

                        bus_list = [ind] * len(names)
                        p_nom_each = new_capacity[ind] / max(1, len(inv_ind))
                        p_nom_list = [p_nom_each] * len(names)
                        
                        # Calculate capital costs with regional WACC for each bus
                        capital_cost_list = [
                            get_regional_capital_cost(bus, carrier_label, costs, wacc_dict, Nyears)
                            for bus in bus_list
                        ]

                        n.madd(
                            "Generator",
                            names,
                            bus=bus_list,
                            carrier=carrier_label,
                            p_nom=p_nom_list,
                            marginal_cost=marginal_cost,
                            capital_cost=capital_cost_list,
                            efficiency=costs.at[get_cost_key(carrier_label), "efficiency"],
                            p_max_pu=p_max_pu,
                            build_year=grouping_year,
                            lifetime=costs.at[get_cost_key(carrier_label), "lifetime"],
                            p_nom_extendable=baseyear_extendable,
                        )

                else:
                    # For non-clustered case, use existing generators as reference for p_max_pu
                    ref_cols = new_capacity.index + f" {generator}{suffix}-{baseyear}"
                    available_cols = ref_cols.intersection(n.generators_t.p_max_pu.columns)
                    missing_cols = ref_cols.difference(available_cols)
                    
                    if not missing_cols.empty:
                        logger.warning(f"Missing p_max_pu data for {len(missing_cols)} {generator} buses from baseyear {baseyear}")
                        logger.debug(f"Missing buses: {list(missing_cols)}")
                    
                    if available_cols.empty:
                        # Fall back to using any available generator of this type
                        fallback_cols = [c for c in n.generators_t.p_max_pu.columns if f" {generator}{suffix}-" in c]
                        if fallback_cols:
                            logger.info(f"Using fallback p_max_pu from {fallback_cols[0]} for {len(new_capacity)} {generator} generators")
                            p_max_pu = n.generators_t.p_max_pu[fallback_cols[:1]].copy()
                            p_max_pu = pd.concat([p_max_pu] * len(new_capacity), axis=1)
                            p_max_pu.columns = [bus + f" {generator}{suffix}-{grouping_year}" for bus in new_capacity.index]
                        else:
                            # Ultimate fallback: use 1.0 for all timesteps
                            logger.warning(f"No p_max_pu reference data found for {generator}, using 1.0 for all timesteps")
                            p_max_pu = pd.DataFrame(1.0, index=n.snapshots, 
                                                    columns=[bus + f" {generator}{suffix}-{grouping_year}" for bus in new_capacity.index])
                    else:
                        # Use available reference data
                        ref = n.generators_t.p_max_pu[available_cols]
                        p_max_pu = ref.copy()
                        # Map available columns to new capacity buses
                        bus_mapping = {col: col.replace(f" {generator}{suffix}-{baseyear}", "") for col in available_cols}
                        p_max_pu.columns = [bus + f" {generator}{suffix}-{grouping_year}" for bus in [bus_mapping[col] for col in available_cols]]
                        
                        # For missing buses, use fallback or 1.0
                        if not missing_cols.empty:
                            missing_buses = [col.replace(f" {generator}{suffix}-{baseyear}", "") for col in missing_cols]
                            if available_cols.size > 0:
                                # Use the first available column as template
                                template = n.generators_t.p_max_pu[available_cols[0]]
                                for bus in missing_buses:
                                    p_max_pu[bus + f" {generator}{suffix}-{grouping_year}"] = template.values
                            else:
                                # Use 1.0
                                for bus in missing_buses:
                                    p_max_pu[bus + f" {generator}{suffix}-{grouping_year}"] = 1.0

                    # Handle profiles that exist but are all-zero (e.g. JP/TW onwind),
                    # by borrowing the nearest non-zero profile from the same technology.
                    if generator in {"onwind", "solar", "offwind"}:
                        p_max_pu = _replace_zero_profile_columns_with_nearest(
                            n=n,
                            p_max_pu=p_max_pu,
                            generator=generator,
                            suffix=suffix,
                            source_year=baseyear,
                            target_year=grouping_year,
                        )
                    
                    names = [bus + name_suffix for bus in new_capacity.index]
                    bus_list = list(new_capacity.index)
                    
                    # Calculate regional capital costs for renewable technologies
                    capital_cost_list = [
                        get_regional_capital_cost(bus, carrier_label, costs, wacc_dict, Nyears)
                        for bus in bus_list
                    ]

                    n.madd(
                        "Generator",
                        names,
                        bus=bus_list,
                        carrier=carrier_label,
                        p_nom=list(new_capacity.values),
                        marginal_cost=marginal_cost,
                        capital_cost=capital_cost_list,
                        efficiency=costs.at[get_cost_key(carrier_label), "efficiency"],
                        p_max_pu=p_max_pu,
                        build_year=grouping_year,
                        lifetime=costs.at[get_cost_key(carrier_label), "lifetime"],
                        p_nom_extendable=baseyear_extendable,
                    )

        else:
            if carrier[generator] not in vars(spatial).keys():
                logger.debug(f"Carrier type {generator} not in spatial data, skipping")
                continue

            # Helper: construct fuel bus name correctly for spatial vs non-spatial fuels
            def fuel_bus(elec_bus, fuel):
                if "Earth" in vars(spatial)[fuel].locations:
                    return vars(spatial)[fuel].nodes[0]
                return f"{elec_bus} {fuel}"

            fuel_carrier = carrier[generator]
            bus_carrier = "solid biomass" if fuel_carrier == "biomass" else fuel_carrier

            # For spatial carriers, ensure we have bus names for all capacity locations
            if "Earth" not in vars(spatial)[fuel_carrier].locations:
                # Use country-level fuel buses instead of cluster-level
                required_bus0 = pd.Index([fuel_bus(loc, fuel_carrier) for loc in capacity.index])
                bus0 = required_bus0
            else:
                # For non-spatial carriers like uranium, use the spatial nodes
                bus0 = vars(spatial)[fuel_carrier].nodes

            # check for missing bus and create them
            missing_bus = pd.Index(bus0).difference(n.buses.index)
            if not missing_bus.empty:
                logger.info(f"Creating {len(missing_bus)} missing buses for {generator}: {list(missing_bus)}")
                if bus_carrier == "solid biomass" and bus_carrier not in n.carriers.index:
                    n.add("Carrier", bus_carrier)
                # Extract corresponding locations for the buses being added
                if "Earth" not in vars(spatial)[fuel_carrier].locations:
                    # For country-level fuel buses, extract country codes
                    bus_locations = [bus.split()[0] for bus in missing_bus]
                else:
                    # For non-spatial carriers, use the single Earth location
                    bus_locations = vars(spatial)[fuel_carrier].locations
                    logger.info(f"bus_locations: {bus_locations}")
                
                n.madd(
                    "Bus",
                    missing_bus,
                    carrier=bus_carrier,
                    location=bus_locations,
                    unit="MWh_el",
                )

            already_build = n.links.index.intersection(asset_i)
            new_build = asset_i.difference(n.links.index)
            lifetime_assets = lifetime.loc[grouping_year, generator].dropna()

            # Handle already built links - adjust capacity to match powerplant data
            if not already_build.empty:
                # Safer: preserve order of already_build and avoid regex surprises
                bases = already_build.str.replace(name_suffix, "", regex=False)
                target_fuel = (
                    capacity.reindex(bases).fillna(0.0) / costs.at[generator, "efficiency"]
                )

                # Find links of this generator type and grouping year that don't have powerplant data
                links_of_year = n.links.index[
                    (n.links.carrier == generator) & 
                    (n.links.build_year == grouping_year)
                ]
                
                links_without_data = links_of_year.difference(already_build)

                # Remove links that don't correspond to any external capacity data
                if not links_without_data.empty:
                    logger.debug(f"Removing {len(links_without_data)} {generator} links without external capacity data for year {grouping_year}")
                    n.mremove("Link", links_without_data)

                if not already_build.empty:
                    # Resize the already-built links to match the cohort
                    n.links.loc[already_build, "p_nom"] = target_fuel.to_numpy()

                    # Make the minimum equal to the cohort
                    n.links.loc[already_build, "p_nom_min"] = target_fuel.to_numpy()

                    n.links.loc[already_build, "p_nom_extendable"] = False

                    # Update lifetime on these to match the external data if available
                    if not lifetime_assets.empty:
                        bases = already_build.str.replace(name_suffix, "", regex=False)
                        mask = bases.isin(lifetime_assets.index)
                        if mask.any():
                            n.links.loc[already_build[mask], "lifetime"] = lifetime_assets.loc[bases[mask]].to_numpy()

            if not new_build.empty:
                new_capacity = capacity.loc[new_build.str.replace(name_suffix, "")]

                n.madd( # changed from https://github.com/pypsa-meets-earth/pypsa-earth/pull/1678/changes
                    "Link",
                    new_capacity.index,
                    suffix=name_suffix,
                    bus0=bus0,
                    bus1=new_capacity.index,
                    bus2="co2 atmosphere",
                    carrier=generator,
                    marginal_cost=costs.at[generator, "efficiency"]
                    * costs.at[generator, "VOM"],  # NB: VOM is per MWel
                    capital_cost=costs.at[generator, "efficiency"]
                    * costs.at[generator, "fixed"],  # NB: fixed cost is per MWel
                    p_nom=new_capacity / costs.at[generator, "efficiency"],
                    efficiency=costs.at[generator, "efficiency"],
                    efficiency2=costs.at[carrier[generator], "CO2 intensity"],
                    build_year=grouping_year,
                    lifetime=lifetime_assets.loc[new_capacity.index],
                )

        # check if existing capacities are larger than technical potential
        existing_large = n.generators[
            n.generators["p_nom_min"] > n.generators["p_nom_max"]
        ].index
        if len(existing_large):
            logger.warning(
                f"Existing capacities larger than technical potential for {existing_large},\
                           adjust technical potential to existing capacities"
            )
            n.generators.loc[existing_large, "p_nom_max"] = n.generators.loc[
                existing_large, "p_nom_min"
            ]

    thermal_carriers = set(carrier.values()) & set(n.links.carrier.unique())
    
    assert n.links.index[
        n.links.carrier.isin(thermal_carriers)
        & ~n.links.index.str.contains(r"-\d{4}$")
    ].empty, "Found thermal power plant links without proper year suffixes"


def add_heating_capacities_installed_before_baseyear(
    n,
    baseyear,
    grouping_years,
    ashp_cop,
    gshp_cop,
    time_dep_hp_cop,
    costs,
    default_lifetime,
):
    """
    Parameters
    ----------
    n : pypsa.Network
    baseyear : last year covered in the existing capacities database
    grouping_years : intervals to group existing capacities
        linear decommissioning of heating capacities from 2020 to 2045 is
        currently assumed heating capacities split between residential and
        services proportional to heating load in both 50% capacities
        in rural busess 50% in urban buses
    """
    logger.debug(f"Adding heating capacities installed before {baseyear}")

    existing_heating = pd.read_csv(
        snakemake.input.existing_heating_distribution, header=[0, 1], index_col=0
    )

    techs = existing_heating.columns.get_level_values(1).unique()

    for name in existing_heating.columns.get_level_values(0).unique():
        name_type = "central" if name == "urban central" else "decentral"

        nodes = pd.Index(n.buses.location[n.buses.index.str.contains(f"{name} heat")])

        if (name_type != "central") and options["electricity_distribution_grid"]:
            nodes_elec = nodes + " low voltage"
        else:
            nodes_elec = nodes

        heat_pump_type = "air" if "urban" in name else "ground"

        # Add heat pumps
        costs_name = f"decentral {heat_pump_type}-sourced heat pump"

        cop = {"air": ashp_cop, "ground": gshp_cop}

        if time_dep_hp_cop:
            efficiency = cop[heat_pump_type][nodes]
        else:
            efficiency = costs.at[costs_name, "efficiency"]

        for i, grouping_year in enumerate(grouping_years):
            if int(grouping_year) + default_lifetime <= int(baseyear):
                continue

            # installation is assumed to be linear for the past default_lifetime years
            ratio = (int(grouping_year) - int(grouping_years[i - 1])) / default_lifetime

            n.madd(
                "Link",
                nodes,
                suffix=f" {name} {heat_pump_type} heat pump-{grouping_year}",
                bus0=nodes_elec,
                bus1=nodes + " " + name + " heat",
                carrier=f"{name} {heat_pump_type} heat pump",
                efficiency=efficiency,
                capital_cost=costs.at[costs_name, "efficiency"]
                * costs.at[costs_name, "fixed"],
                p_nom=existing_heating.loc[nodes, (name, f"{heat_pump_type} heat pump")]
                * ratio
                / costs.at[costs_name, "efficiency"],
                build_year=int(grouping_year),
                lifetime=costs.at[costs_name, "lifetime"],
                p_nom_extendable=baseyear_extendable,
            )

            # add resistive heater, gas boilers and oil boilers
            n.madd(
                "Link",
                nodes,
                suffix=f" {name} resistive heater-{grouping_year}",
                bus0=nodes_elec,
                bus1=nodes + " " + name + " heat",
                carrier=name + " resistive heater",
                efficiency=costs.at[f"{name_type} resistive heater", "efficiency"],
                capital_cost=(
                    costs.at[f"{name_type} resistive heater", "efficiency"]
                    * costs.at[f"{name_type} resistive heater", "fixed"]
                ),
                p_nom=(
                    existing_heating.loc[nodes, (name, "resistive heater")]
                    * ratio
                    / costs.at[f"{name_type} resistive heater", "efficiency"]
                ),
                build_year=int(grouping_year),
                lifetime=costs.at[f"{name_type} resistive heater", "lifetime"],
                p_nom_extendable=baseyear_extendable,
            )

            n.madd(
                "Link",
                nodes,
                suffix=f" {name} gas boiler-{grouping_year}",
                bus0="EU gas" if "EU gas" in spatial.gas.nodes else nodes + " gas",
                bus1=nodes + " " + name + " heat",
                bus2="co2 atmosphere",
                carrier=name + " gas boiler",
                efficiency=costs.at[f"{name_type} gas boiler", "efficiency"],
                efficiency2=costs.at["gas", "CO2 intensity"],
                capital_cost=(
                    costs.at[f"{name_type} gas boiler", "efficiency"]
                    * costs.at[f"{name_type} gas boiler", "fixed"]
                ),
                p_nom=(
                    existing_heating.loc[nodes, (name, "gas boiler")]
                    * ratio
                    / costs.at[f"{name_type} gas boiler", "efficiency"]
                ),
                build_year=int(grouping_year),
                lifetime=costs.at[f"{name_type} gas boiler", "lifetime"],
                p_nom_extendable=baseyear_extendable,
            )

            n.madd(
                "Link",
                nodes,
                suffix=f" {name} oil boiler-{grouping_year}",
                bus0=spatial.oil.nodes,
                bus1=nodes + " " + name + " heat",
                bus2="co2 atmosphere",
                carrier=name + " oil boiler",
                efficiency=costs.at["decentral oil boiler", "efficiency"],
                efficiency2=costs.at["oil", "CO2 intensity"],
                capital_cost=costs.at["decentral oil boiler", "efficiency"]
                * costs.at["decentral oil boiler", "fixed"],
                p_nom=(
                    existing_heating.loc[nodes, (name, "oil boiler")]
                    * ratio
                    / costs.at["decentral oil boiler", "efficiency"]
                ),
                build_year=int(grouping_year),
                lifetime=costs.at[f"{name_type} gas boiler", "lifetime"],
                p_nom_extendable=baseyear_extendable,
            )

            # delete links with p_nom=nan corresponding to extra nodes in country
            n.mremove(
                "Link",
                [
                    index
                    for index in n.links.index.to_list()
                    if str(grouping_year) in index and np.isnan(n.links.p_nom[index])
                ],
            )

            # delete links with capacities below threshold
            threshold = snakemake.params.existing_capacities["threshold_capacity"]
            n.mremove(
                "Link",
                [
                    index
                    for index in n.links.index.to_list()
                    if str(grouping_year) in index and n.links.p_nom[index] < threshold
                ],
            )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_existing_baseyear",
            simpl="",
            clusters="110",
            ll="copt",
            opts="1h",
            planning_horizons="2020",
            sopts="1h",
            configfile="/shared/share_cki25/energymodels/pypsa-earth/config.myopic.yaml",
            discountrate="0.071",
            demand="AB",
            h2export="0.0"
        )

    # configure_logging(snakemake)
    # set_scenario_config(snakemake)

    # update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    options = snakemake.params.sector

    baseyear = snakemake.params.baseyear

    n = pypsa.Network(snakemake.input.network)

    baseyear_nonextendable = str(snakemake.params.extendability["baseyear_nonextendable"]).lower() == "true"
    baseyear_extendable = not baseyear_nonextendable

    # define spatial resolution of carriers
    spatial = define_spatial(n.buses[n.buses.carrier == "AC"].index, options)
    add_build_year_to_new_assets(n, baseyear)

    Nyears = n.snapshot_weightings.generators.sum() / 8760.0
    costs = prepare_costs(
        snakemake.input.costs,
        snakemake.config["costs"],
        snakemake.params.costs["output_currency"],
        snakemake.params.costs["fill_values"],
        Nyears,
        snakemake.params.costs["default_exchange_rate"],
        snakemake.params.costs["future_exchange_rate_strategy"],
        snakemake.params.costs["custom_future_exchange_rate"],
    )
    
    # Load regional WACCs for renewable capital cost calculations
    wacc_dict = load_country_waccs(snakemake.input.waccs)

    # Ensure all carriers used in this script are defined
    required_carriers = ["solar", "onwind", "offwind-ac", "gas", "coal", "oil", "lignite", "uranium", "biomass", "AC"]
    existing_carriers = set(n.carriers.index) if hasattr(n, 'carriers') and not n.carriers.empty else set()
    missing_carriers = set(required_carriers) - existing_carriers
    
    if missing_carriers:
        logger.info(f"Adding missing carrier definitions: {list(missing_carriers)}")
        for carrier in missing_carriers:
            if carrier == "AC":
                n.add("Carrier", carrier, co2_emissions=0, nice_name="AC")
            elif carrier in ["solar", "onwind", "offwind-ac"]:
                n.add("Carrier", carrier, co2_emissions=0, nice_name=carrier.title())
            elif carrier == "uranium":
                n.add("Carrier", carrier, co2_emissions=0, nice_name="Nuclear")
            else:
                # For fossil fuels, use default emissions if available in costs
                co2_emissions = costs.at[carrier, "CO2 intensity"] if carrier in costs.index else 0
                n.add("Carrier", carrier, co2_emissions=co2_emissions, nice_name=carrier.title())

    grouping_years_power = snakemake.params.existing_capacities["grouping_years_power"]
    grouping_years_heat = snakemake.params.existing_capacities["grouping_years_heat"]

    add_power_capacities_installed_before_baseyear(
        n, grouping_years_power, costs, baseyear
    )
    
    if not baseyear_extendable: # for myopic runs with baseyear <= 2020 (today)
        for c in n.iterate_components(["Generator", "StorageUnit"]):
            col = "p_nom_extendable"

            if col not in c.df.columns:
                c.df[col] = np.zeros(len(c.df), dtype=np.bool_)

            if "build_year" in c.df.columns:
                assets = c.df.index[c.df.build_year <= baseyear]

                if c.name == "Generator":
                    keep_extendable = ["OCGT", "CCGT", "coal", "biomass", "oil", "gas", "lignite"]
                    mask = assets.intersection(
                        c.df.index[~c.df.carrier.isin(keep_extendable)]
                    )
                    c.df.loc[mask, col] = False

                else:
                    c.df.loc[assets, col] = False

        # ensure boolean dtype
        c.df[col] = c.df[col].fillna(False).astype(bool)

        # WS2: In the calibrated 2020 baseyear only, freeze AC line expansion as well.
        # This keeps 2020 transmission as an operational baseline, while preserving
        # expansion behavior for later myopic horizons.
        if int(baseyear) == 2020 and hasattr(n, "lines") and not n.lines.empty:
            lines = n.lines
            if "s_nom_extendable" not in lines.columns:
                lines["s_nom_extendable"] = np.zeros(len(lines), dtype=np.bool_)

            if "build_year" in lines.columns:
                line_assets = lines.index[pd.to_numeric(lines.build_year, errors="coerce").fillna(0) <= int(baseyear)]
            else:
                logger.warning(
                    "Line component has no build_year column; applying 2020 non-extendability to all lines."
                )
                line_assets = lines.index

            if len(line_assets):
                lines.loc[line_assets, "s_nom_extendable"] = False
                if "s_nom_min" not in lines.columns:
                    lines["s_nom_min"] = 0.0
                lines.loc[line_assets, "s_nom_min"] = pd.to_numeric(
                    lines.loc[line_assets, "s_nom"], errors="coerce"
                ).fillna(0.0)
                lines["s_nom_extendable"] = lines["s_nom_extendable"].fillna(False).astype(bool)
                logger.info(
                    "In baseyear 2020: Set %d lines to s_nom_extendable=False and s_nom_min=s_nom.",
                    len(line_assets),
                )

        # WS2: Freeze DC transmission links in 2020 as well (transmission expansion starts after 2020).
        if int(baseyear) == 2020 and hasattr(n, "links") and not n.links.empty:
            links = n.links
            dc_mask = links.carrier.astype(str).eq("DC") if "carrier" in links.columns else pd.Series(False, index=links.index)
            if dc_mask.any():
                if "p_nom_extendable" not in links.columns:
                    links["p_nom_extendable"] = np.zeros(len(links), dtype=np.bool_)

                if "build_year" in links.columns:
                    build_year = pd.to_numeric(links.build_year, errors="coerce").fillna(0)
                    dc_assets = links.index[dc_mask & (build_year <= int(baseyear))]
                else:
                    logger.warning(
                        "Link component has no build_year column; applying 2020 non-extendability to all DC links."
                    )
                    dc_assets = links.index[dc_mask]

                if len(dc_assets):
                    links.loc[dc_assets, "p_nom_extendable"] = False
                    if "p_nom_min" not in links.columns:
                        links["p_nom_min"] = 0.0
                    links.loc[dc_assets, "p_nom_min"] = pd.to_numeric(
                        links.loc[dc_assets, "p_nom"], errors="coerce"
                    ).fillna(0.0)
                    links["p_nom_extendable"] = links["p_nom_extendable"].fillna(False).astype(bool)
                    logger.info(
                        "In baseyear 2020: Set %d DC links to p_nom_extendable=False and p_nom_min=p_nom.",
                        len(dc_assets),
                    )

        # WS2: Freeze battery capacity in the 2020 baseyear only; later horizons remain extendable.
        if int(baseyear) == 2020 and hasattr(n, "stores") and not n.stores.empty:
            stores = n.stores
            battery_store_mask = (
                stores.carrier.astype(str).eq("battery")
                if "carrier" in stores.columns
                else pd.Series(False, index=stores.index)
            )
            if battery_store_mask.any():
                if "e_nom_extendable" not in stores.columns:
                    stores["e_nom_extendable"] = np.zeros(len(stores), dtype=np.bool_)
                if "build_year" in stores.columns:
                    store_build_year = pd.to_numeric(stores.build_year, errors="coerce").fillna(0)
                    battery_store_assets = stores.index[
                        battery_store_mask & (store_build_year <= int(baseyear))
                    ]
                else:
                    logger.warning(
                        "Store component has no build_year column; applying 2020 non-extendability to all battery stores."
                    )
                    battery_store_assets = stores.index[battery_store_mask]

                if len(battery_store_assets):
                    stores.loc[battery_store_assets, "e_nom_extendable"] = False
                    if "e_nom_min" not in stores.columns:
                        stores["e_nom_min"] = 0.0
                    stores.loc[battery_store_assets, "e_nom_min"] = pd.to_numeric(
                        stores.loc[battery_store_assets, "e_nom"], errors="coerce"
                    ).fillna(0.0)
                    stores["e_nom_extendable"] = stores["e_nom_extendable"].fillna(False).astype(bool)
                    logger.info(
                        "In baseyear 2020: Set %d battery stores to e_nom_extendable=False and e_nom_min=e_nom.",
                        len(battery_store_assets),
                    )

        if int(baseyear) == 2020 and hasattr(n, "links") and not n.links.empty:
            links = n.links
            battery_link_carriers = {"battery charger", "battery discharger"}
            battery_link_mask = (
                links.carrier.astype(str).isin(battery_link_carriers)
                if "carrier" in links.columns
                else pd.Series(False, index=links.index)
            )
            if battery_link_mask.any():
                if "p_nom_extendable" not in links.columns:
                    links["p_nom_extendable"] = np.zeros(len(links), dtype=np.bool_)
                if "build_year" in links.columns:
                    link_build_year = pd.to_numeric(links.build_year, errors="coerce").fillna(0)
                    battery_link_assets = links.index[
                        battery_link_mask & (link_build_year <= int(baseyear))
                    ]
                else:
                    logger.warning(
                        "Link component has no build_year column; applying 2020 non-extendability to all battery links."
                    )
                    battery_link_assets = links.index[battery_link_mask]

                if len(battery_link_assets):
                    links.loc[battery_link_assets, "p_nom_extendable"] = False
                    if "p_nom_min" not in links.columns:
                        links["p_nom_min"] = 0.0
                    links.loc[battery_link_assets, "p_nom_min"] = pd.to_numeric(
                        links.loc[battery_link_assets, "p_nom"], errors="coerce"
                    ).fillna(0.0)
                    links["p_nom_extendable"] = links["p_nom_extendable"].fillna(False).astype(bool)
                    logger.info(
                        "In baseyear 2020: Set %d battery links to p_nom_extendable=False and p_nom_min=p_nom.",
                        len(battery_link_assets),
                    )

        if hasattr(_validation_hooks, "_freeze_baseyear_loophole_links"):
            _validation_hooks._freeze_baseyear_loophole_links(
                n,
                baseyear=baseyear,
                config=snakemake.config,
            )

        for c in n.iterate_components(["Generator", "Link", "StorageUnit"]):
            if "build_year" in c.df.columns:
                assets = c.df.index[c.df.build_year <= baseyear]
                c.df.loc[assets, "p_nom_min"] = c.df.loc[assets, "p_nom"]
            else:
                logger.warning(f"Component {c.name} has no build_year column, cannot set p_nom_min for existing assets")    

        logger.info(f"In baseyear {baseyear}: All existing assets set to p_nom_extendable/e_nom_extendable = False")
        logger.info(f"In baseyear {baseyear}: All existing assets set to p_nom_min = p_nom (and e_nom_min = e_nom for storage) to prevent capacity reduction")

    # Final cleanup: ensure generator capacities are not NaN
    if hasattr(n, "generators") and not n.generators.empty:
        gens = n.generators
        if "p_nom" in gens.columns:
            nan_p_nom = gens.p_nom.isna()
            if nan_p_nom.any():
                logger.warning(f"Found {nan_p_nom.sum()} generators with NaN p_nom; filling with 0 or p_nom_min")
                fallback = gens.loc[nan_p_nom, "p_nom_min"] if "p_nom_min" in gens.columns else pd.Series(0.0, index=gens.index)
                fallback = fallback.fillna(0.0)
                gens.loc[nan_p_nom, "p_nom"] = fallback.loc[nan_p_nom]

        if "p_nom_min" in gens.columns:
            nan_p_nom_min = gens.p_nom_min.isna()
            if nan_p_nom_min.any():
                if "p_nom_extendable" in gens.columns:
                    extendable = gens.p_nom_extendable.fillna(False)
                    gens.loc[nan_p_nom_min & extendable, "p_nom_min"] = 0.0
                    gens.loc[nan_p_nom_min & ~extendable, "p_nom_min"] = gens.loc[nan_p_nom_min & ~extendable, "p_nom"].fillna(0.0)
                else:
                    gens.loc[nan_p_nom_min, "p_nom_min"] = gens.loc[nan_p_nom_min, "p_nom"].fillna(0.0)

    # Final cleanup: ensure link capacities are not NaN
    if hasattr(n, "links") and not n.links.empty:
        links = n.links
        if "p_nom" in links.columns:
            nan_p_nom = links.p_nom.isna()
            if nan_p_nom.any():
                logger.warning(f"Found {nan_p_nom.sum()} links with NaN p_nom; filling with 0")
                links.loc[nan_p_nom, "p_nom"] = 0.0

        if "p_nom_min" in links.columns:
            nan_p_nom_min = links.p_nom_min.isna()
            if nan_p_nom_min.any():
                if "p_nom_extendable" in links.columns:
                    extendable = links.p_nom_extendable.fillna(False)
                    links.loc[nan_p_nom_min & extendable, "p_nom_min"] = 0.0
                    links.loc[nan_p_nom_min & ~extendable, "p_nom_min"] = links.loc[nan_p_nom_min & ~extendable, "p_nom"].fillna(0.0)
                else:
                    links.loc[nan_p_nom_min, "p_nom_min"] = links.loc[nan_p_nom_min, "p_nom"].fillna(0.0)

        if "p_nom_max" in links.columns:
            nan_p_nom_max = links.p_nom_max.isna()
            if nan_p_nom_max.any():
                logger.warning(f"Found {nan_p_nom_max.sum()} links with NaN p_nom_max; setting to inf")
                links.loc[nan_p_nom_max, "p_nom_max"] = np.inf

    # Fill missing time-varying link efficiency from static values
    if hasattr(n, "links_t") and hasattr(n.links_t, "efficiency"):
        eff_t = n.links_t.efficiency
        if not eff_t.empty:
            static_eff = n.links.efficiency.reindex(eff_t.columns)
            eff_t = eff_t.fillna(static_eff)
            n.links_t.efficiency = eff_t

    # TODO: not implemented in -sec yet
    # if options["heating"]:
    #     time_dep_hp_cop = options["time_dep_hp_cop"]
    #     ashp_cop = (
    #         xr.open_dataarray(snakemake.input.cop_air_total)
    #         .to_pandas()
    #         .reindex(index=n.snapshots)
    #     )
    #     gshp_cop = (
    #         xr.open_dataarray(snakemake.input.cop_soil_total)
    #         .to_pandas()
    #         .reindex(index=n.snapshots)
    #     )
    #     default_lifetime = snakemake.params.existing_capacities[
    #         "default_heating_lifetime"
    #     ]
    #     add_heating_capacities_installed_before_baseyear(
    #         n,
    #         baseyear,
    #         grouping_years_heat,
    #         ashp_cop,
    #         gshp_cop,
    #         time_dep_hp_cop,
    #         costs,
    #         default_lifetime,
    #     )

    # if options.get("cluster_heat_buses", False):
    #     cluster_heat_buses(n)

    # Preserve existing n.meta entries (e.g., temporal_cluster_period_id) before updating
    if not hasattr(n, 'meta') or n.meta is None:
        n.meta = {}
    n.meta.update(dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards))))

    # sanitize_carriers(n, snakemake.config)
    os.makedirs(os.path.dirname(snakemake.output[0]), exist_ok=True)

    n.export_to_netcdf(snakemake.output[0])
