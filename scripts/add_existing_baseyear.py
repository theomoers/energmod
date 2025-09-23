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
from types import SimpleNamespace

import country_converter as coco
import numpy as np
import pandas as pd
import powerplantmatching as pm
import pypsa
import xarray as xr

# from _helpers import (
#     configure_logging,
#     set_scenario_config,
#     update_config_from_wildcards,
# )
# from add_electricity import sanitize_carriers
from prepare_sector_network import define_spatial, prepare_costs  # , cluster_heat_buses

logger = logging.getLogger(__name__)
cc = coco.CountryConverter()
idx = pd.IndexSlice
spatial = SimpleNamespace()


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


def add_existing_renewables(df_agg, n, costs):
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

        # distribute capacities among nodes according to capacity factor
        # weighting with nodal_fraction
        elec_buses = n.buses.index[n.buses.carrier == "AC"].union(
            n.buses.index[n.buses.carrier == "DC"]
        )
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

        nodal_df = df.loc[n.buses.loc[elec_buses, "country"]]
        nodal_df.index = elec_buses
        nodal_df = nodal_df.multiply(nodal_fraction, axis=0)

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
        "Bioenergy": "urban central solid biomass CHP",
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
    biomass_i = df_agg.loc[df_agg.Fueltype == "urban central solid biomass CHP"].index
    if biomass_i.empty:
        mean = 0
    else:
        mean = df_agg.loc[biomass_i, "DateIn"].mean()
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
    add_existing_renewables(df_agg, n, costs)

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
        "urban central solid biomass CHP": "biomass",
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
            existing_renewable_gens = n.generators.index[
                (n.generators.build_year == grouping_year) & 
                (n.generators.carrier == carrier_label)
            ]
            
            if not existing_renewable_gens.empty:
                logger.info(f"Found {len(existing_renewable_gens)} existing {generator} generators from {grouping_year}, comparing with external data")
                
                existing_capacity_by_bus = n.generators.loc[existing_renewable_gens].groupby('bus')['p_nom'].sum()
                
                buses_to_adjust = capacity.index.intersection(existing_capacity_by_bus.index)
                buses_to_add = capacity.index.difference(existing_capacity_by_bus.index)
                
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
            
            # Get cost parameters from existing generators or fallback to cost database
            remaining_gens = n.generators.index[n.generators.carrier == carrier_label]
            if not remaining_gens.empty:
                capital_cost = n.generators.loc[remaining_gens, "capital_cost"].mean()
                marginal_cost = n.generators.loc[remaining_gens, "marginal_cost"].mean()
            else:
                # Fallback to cost database - use correct cost key for offshore wind
                cost_key = get_cost_key(carrier_label)
                capital_cost = costs.at[cost_key, "fixed"]
                marginal_cost = costs.at[cost_key, "VOM"]

            # Only add new generators if there's new capacity to add
            if not new_capacity.empty:
                if "m" in snakemake.wildcards.clusters:
                    for ind in new_capacity.index:
                        # existing capacities are split evenly among regions in every country
                        inv_ind = list(inv_busmap[ind])

                        # Build names once and reuse the exact list
                        names = [i + name_suffix for i in inv_ind]
                        
                        # for offshore the splitting only includes coastal regions
                        # Look for existing generators of the same type for p_max_pu reference
                        existing_reference_gens = n.generators.index[
                            (n.generators.carrier == carrier_label) & 
                            (n.generators.index.str.startswith(tuple(inv_ind)))
                        ]
                        
                        # Create p_max_pu using reference generators or default
                        if not existing_reference_gens.empty:
                            ref = n.generators_t.p_max_pu[existing_reference_gens].mean(axis=1)
                        else:
                            ref = pd.Series(1.0, index=n.snapshots)
                        
                        p_max_pu = pd.concat([ref.rename(nm) for nm in names], axis=1)

                        # Assert perfect alignment as suggested by friend
                        assert list(p_max_pu.columns) == names, "p_max_pu names must equal Generator names"
                        assert p_max_pu.index.equals(n.snapshots), "p_max_pu index must equal snapshots"
                        p_max_pu = p_max_pu.clip(lower=0, upper=1)

                        # Don't rely on broadcasting—pass lists as suggested
                        bus_list = [ind] * len(names)
                        p_nom_each = new_capacity[ind] / max(1, len(inv_ind))
                        p_nom_list = [p_nom_each] * len(names)

                        n.madd(
                            "Generator",
                            names,
                            bus=bus_list,
                            carrier=carrier_label,
                            p_nom=p_nom_list,
                            marginal_cost=marginal_cost,
                            capital_cost=capital_cost,
                            efficiency=costs.at[get_cost_key(carrier_label), "efficiency"],
                            p_max_pu=p_max_pu,
                            build_year=grouping_year,
                            lifetime=costs.at[get_cost_key(carrier_label), "lifetime"],
                            p_nom_extendable=baseyear_extendable,
                        )

                else:
                    # For non-clustered case, use existing generators as reference for p_max_pu
                    existing_reference_gens = n.generators.index[
                        (n.generators.carrier == carrier_label)
                    ]
                    
                    # Build names once and reuse the exact list
                    names = [bus + name_suffix for bus in new_capacity.index]
                    
                    # Create p_max_pu using reference generators or default
                    if not existing_reference_gens.empty:
                        ref = n.generators_t.p_max_pu[existing_reference_gens].mean(axis=1)
                    else:
                        ref = pd.Series(1.0, index=n.snapshots)
                    
                    p_max_pu = pd.concat([ref.rename(nm) for nm in names], axis=1)

                    # Assert perfect alignment as suggested by friend
                    assert list(p_max_pu.columns) == names, "p_max_pu names must equal Generator names"
                    assert p_max_pu.index.equals(n.snapshots), "p_max_pu index must equal snapshots"
                    p_max_pu = p_max_pu.clip(lower=0, upper=1)

                    n.madd(
                        "Generator",
                        names,
                        bus=new_capacity.index,
                        carrier=carrier_label,
                        p_nom=new_capacity,
                        marginal_cost=marginal_cost,
                        capital_cost=capital_cost,
                        efficiency=costs.at[get_cost_key(carrier_label), "efficiency"],
                        p_max_pu=p_max_pu,
                        build_year=grouping_year,
                        lifetime=costs.at[get_cost_key(carrier_label), "lifetime"],
                        p_nom_extendable=baseyear_extendable,
                    )

        else:
            if generator not in vars(spatial).keys():
                logger.debug(f"Carrier type {generator} not in spatial data, skipping")
                continue

            # Helper function to create country-level fuel buses
            def fuel_bus(elec_bus, fuel):
                return f"{n.buses.at[elec_bus, 'country']} {fuel}"

            # For spatial carriers, ensure we have bus names for all capacity locations
            if "Earth" not in vars(spatial)[carrier[generator]].locations:
                # Use country-level fuel buses instead of cluster-level
                required_bus0 = pd.Index([fuel_bus(loc, carrier[generator]) for loc in capacity.index])
                bus0 = required_bus0
            else:
                # For non-spatial carriers like uranium, use the spatial nodes
                bus0 = vars(spatial)[carrier[generator]].nodes

            # check for missing bus and create them
            missing_bus = pd.Index(bus0).difference(n.buses.index)
            if not missing_bus.empty:
                logger.info(f"Creating {len(missing_bus)} missing buses for {generator}: {list(missing_bus)}")
                # Extract corresponding locations for the buses being added
                if "Earth" not in vars(spatial)[carrier[generator]].locations:
                    # For country-level fuel buses, extract country codes
                    bus_locations = [bus.split()[0] for bus in missing_bus]  # country code
                else:
                    # For non-spatial carriers, use the single Earth location
                    bus_locations = vars(spatial)[carrier[generator]].locations
                    logger.info(f"bus_locations: {bus_locations}")
                
                n.madd(
                    "Bus",
                    missing_bus,
                    carrier=carrier[generator],
                    location=bus_locations,
                    unit="MWh_el",
                )

            already_build = n.links.index.intersection(asset_i)
            new_build = asset_i.difference(n.links.index)
            lifetime_assets = lifetime.loc[grouping_year, generator].dropna()

            # this is for the year 2020
            if not already_build.empty:
                n.links.loc[already_build, "p_nom_min"] = capacity.loc[
                    already_build.str.replace(name_suffix, "")
                ].values

            if not new_build.empty:
                new_capacity = capacity.loc[new_build.str.replace(name_suffix, "")]

                if generator != "urban central solid biomass CHP":
                    # Handle missing lifetime data by using available lifetime values or default from costs
                    available_lifetime_idx = new_capacity.index.intersection(lifetime_assets.index)
                    missing_lifetime_idx = new_capacity.index.difference(lifetime_assets.index)
                    
                    # For buses with available lifetime data
                    if not available_lifetime_idx.empty:
                        n.madd(
                            "Link",
                            available_lifetime_idx,
                            suffix=name_suffix,
                            bus0=[fuel_bus(loc, carrier[generator]) for loc in available_lifetime_idx],
                            bus1=available_lifetime_idx,
                            bus2="co2 atmosphere",
                            carrier=generator,
                            marginal_cost=costs.at[generator, "efficiency"]
                            * costs.at[generator, "VOM"],  # NB: VOM is per MWel
                            capital_cost=costs.at[generator, "efficiency"]
                            * costs.at[generator, "fixed"],  # NB: fixed cost is per MWel
                            p_nom=new_capacity.loc[available_lifetime_idx] / costs.at[generator, "efficiency"],
                            efficiency=costs.at[generator, "efficiency"],
                            efficiency2=costs.at[carrier[generator], "CO2 intensity"],
                            build_year=grouping_year,
                            lifetime=lifetime_assets.loc[available_lifetime_idx],
                            p_nom_extendable=baseyear_extendable,
                        )
                    
                    # For buses with missing lifetime data, use default lifetime from costs
                    if not missing_lifetime_idx.empty:
                        n.madd(
                            "Link",
                            missing_lifetime_idx,
                            suffix=name_suffix,
                            bus0=[fuel_bus(loc, carrier[generator]) for loc in missing_lifetime_idx],
                            bus1=missing_lifetime_idx,
                            bus2="co2 atmosphere",
                            carrier=generator,
                            marginal_cost=costs.at[generator, "efficiency"]
                            * costs.at[generator, "VOM"],  # NB: VOM is per MWel
                            capital_cost=costs.at[generator, "efficiency"]
                            * costs.at[generator, "fixed"],  # NB: fixed cost is per MWel
                            p_nom=new_capacity.loc[missing_lifetime_idx] / costs.at[generator, "efficiency"],
                            efficiency=costs.at[generator, "efficiency"],
                            efficiency2=costs.at[carrier[generator], "CO2 intensity"],
                            build_year=grouping_year,
                            lifetime=costs.at[generator, "lifetime"],
                            p_nom_extendable=baseyear_extendable,
                        )
                else:
                    key = "central solid biomass CHP"
                    # Handle missing lifetime data for biomass CHP
                    available_lifetime_idx = new_capacity.index.intersection(lifetime_assets.index)
                    missing_lifetime_idx = new_capacity.index.difference(lifetime_assets.index)
                    
                    # For buses with available lifetime data
                    if not available_lifetime_idx.empty:
                        n.madd(
                            "Link",
                            available_lifetime_idx,
                            suffix=name_suffix,
                            bus0=spatial.biomass.df.loc[available_lifetime_idx]["nodes"].values,
                            bus1=available_lifetime_idx,
                            bus2=available_lifetime_idx + " urban central heat",
                            carrier=generator,
                            p_nom=new_capacity.loc[available_lifetime_idx] / costs.at[key, "efficiency"],
                            capital_cost=costs.at[key, "fixed"]
                            * costs.at[key, "efficiency"],
                            marginal_cost=costs.at[key, "VOM"],
                            efficiency=costs.at[key, "efficiency"],
                            build_year=grouping_year,
                            efficiency2=costs.at[key, "efficiency-heat"],
                            lifetime=lifetime_assets.loc[available_lifetime_idx],
                            p_nom_extendable=baseyear_extendable,
                        )
                    
                    # For buses with missing lifetime data, use default lifetime from costs
                    if not missing_lifetime_idx.empty:
                        n.madd(
                            "Link",
                            missing_lifetime_idx,
                            suffix=name_suffix,
                            bus0=spatial.biomass.df.loc[missing_lifetime_idx]["nodes"].values,
                            bus1=missing_lifetime_idx,
                            bus2=missing_lifetime_idx + " urban central heat",
                            carrier=generator,
                            p_nom=new_capacity.loc[missing_lifetime_idx] / costs.at[key, "efficiency"],
                            capital_cost=costs.at[key, "fixed"]
                            * costs.at[key, "efficiency"],
                            marginal_cost=costs.at[key, "VOM"],
                            efficiency=costs.at[key, "efficiency"],
                            build_year=grouping_year,
                            efficiency2=costs.at[key, "efficiency-heat"],
                            lifetime=costs.at[key, "lifetime"],
                            p_nom_extendable=baseyear_extendable,
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
            clusters="200",
            ll="copt",
            opts="3h",
            planning_horizons="2020",
            sopts="72h",
            configfile="/shared/share_cki25/energymodels/pypsa-earth/config.myopic.yaml",
            discountrate="0.071",
            demand="AB",
            h2export="10"
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
        for c in n.iterate_components(["Generator"]):
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

        logger.info(f"In baseyear {baseyear}: All existing assets set to p_nom_extendable/e_nom_extendable = False")

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

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))

    # sanitize_carriers(n, snakemake.config)

    n.export_to_netcdf(snakemake.output[0])
