# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# -*- coding: utf-8 -*-
import logging
import os
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pypsa
import pytz
import ruamel.yaml
import xarray as xr
import validation as _validation_hooks
from _helpers import (
    BASE_DIR,
    create_dummy_data,
    create_network_topology,
    cycling_shift,
    locate_bus,
    locate_bus_alt_clust,
    mock_snakemake,
    override_component_attrs,
    prepare_costs,
    safe_divide,
    three_2_two_digits_country,
    two_2_three_digits_country,
    annuity
)
from learning.fuel_price_io import (
    get_fuel_price_by_node,
    load_country_fuel_prices_dict,
)
from prepare_transport_data import prepare_transport_data
from temporal_clustering import aggregate_snapshots

logger = logging.getLogger(__name__)

spatial = SimpleNamespace()
biomass_allocation = None


# Centralized validation/tuning hooks (kept outside this core script for easier reversion).
if hasattr(_validation_hooks, "align_country_electricity_demand_to_owid"):
    align_country_electricity_demand_to_owid = _validation_hooks.align_country_electricity_demand_to_owid
    apply_hydro_profile_fallback_and_diagnostics = (
        _validation_hooks.apply_hydro_profile_fallback_and_diagnostics
    )
    apply_renewable_profile_fallbacks = _validation_hooks.apply_renewable_profile_fallbacks
    align_country_hydro_reservoir_inflow_to_owid = _validation_hooks.align_country_hydro_reservoir_inflow_to_owid
    adjust_hydro = _validation_hooks.adjust_hydro
    align_country_onwind_profiles_to_owid = _validation_hooks.align_country_onwind_profiles_to_owid
    apply_country_wind_iteration_scaling = _validation_hooks.apply_country_wind_iteration_scaling
    apply_country_solar_iteration_scaling = _validation_hooks.apply_country_solar_iteration_scaling
    apply_country_hydro_iteration_scaling = _validation_hooks.apply_country_hydro_iteration_scaling
    apply_pris_nuclear_capacity_and_availability = (
        _validation_hooks.apply_pris_nuclear_capacity_and_availability
    )
    apply_gogpt_oil_capacity_fix = _validation_hooks.apply_gogpt_oil_capacity_fix
    apply_ember_bioenergy_capacity_fix = _validation_hooks.apply_ember_bioenergy_capacity_fix
    logger.info("Using centralized validation/tuning hooks from scripts/validation.py")
if hasattr(_validation_hooks, "apply_country_fuel_price_overrides"):
    apply_country_fuel_price_overrides = _validation_hooks.apply_country_fuel_price_overrides
if hasattr(_validation_hooks, "derive_post2020_structural_biomass_allocation"):
    derive_post2020_structural_biomass_allocation = (
        _validation_hooks.derive_post2020_structural_biomass_allocation
    )


def load_country_fuel_prices(fuelprices_path, investment_year, costs):
    """Load country-specific fuel prices from CSV for a given investment year."""
    fuel_price_dict = load_country_fuel_prices_dict(
        fuelprices_path=fuelprices_path,
        investment_year=investment_year,
        costs=costs,
    )
    logger.info(f"Loaded country-specific fuel prices for year {investment_year}")
    logger.info(f"  Oil prices: {len(fuel_price_dict.get('oil', {}))} countries")
    logger.info(f"  Gas prices: {len(fuel_price_dict.get('gas', {}))} countries")
    logger.info(f"  Coal prices: {len(fuel_price_dict.get('coal', {}))} countries")
    return fuel_price_dict


# Fossil price override hook is centralized in scripts/validation.py.
if hasattr(_validation_hooks, "apply_country_fuel_price_overrides"):
    apply_country_fuel_price_overrides = _validation_hooks.apply_country_fuel_price_overrides


def load_country_waccs(wacc_path, costs):
    """
    Load country-specific WACCs for renewable technologies.
    
    Parameters
    ----------
    wacc_path : str
        Path to the WACC CSV file
    costs : pd.DataFrame
        Default costs DataFrame for fallback values
    
    Returns
    -------
    dict
        Nested dictionary: {technology: {country_code: wacc_decimal}}
        Technologies: 'solar', 'onwind', 'offwind'
    """
    try:
        wacc_df = pd.read_csv(wacc_path)
    except FileNotFoundError:
        logger.warning(f"WACC file not found at {wacc_path}. Using global default discount rate.")
        return {}
    
    # Build nested dictionary: {tech: {country: wacc}}
    wacc_dict = {}
    for tech in ['solar', 'onwind', 'offwind']:
        if tech not in wacc_df.columns:
            logger.warning(f"Technology '{tech}' not found in WACC file. Using default.")
            continue
        
        # Convert percentage to decimal (e.g., 4.2 -> 0.042)
        wacc_dict[tech] = dict(zip(
            wacc_df['country'],
            wacc_df[tech] / 100.0
        ))
    
    logger.info(f"Loaded country-specific WACCs for renewable technologies")
    logger.info(f"  Solar WACC: {len(wacc_dict.get('solar', {}))} countries")
    logger.info(f"  Onwind WACC: {len(wacc_dict.get('onwind', {}))} countries")
    logger.info(f"  Offwind WACC: {len(wacc_dict.get('offwind', {}))} countries")
    
    return wacc_dict


def apply_regional_waccs(n, costs, wacc_dict, Nyears):
    """
    Apply country-specific WACCs to renewable generators and recalculate capital costs.
    
    Only applies to solar, onwind, offwind-ac, and offwind-dc technologies.
    Other technologies keep their default discount rate from costs.
    
    Parameters
    ----------
    n : pypsa.Network
        Network with generators to update
    costs : pd.DataFrame
        Cost assumptions with default discount rates
    wacc_dict : dict
        Country-specific WACCs from load_country_waccs()
    Nyears : float
        Number of years for annualization
    """
    
    if not wacc_dict:
        logger.info("No regional WACC data loaded. Using default discount rates.")
        return
    
    # Map carriers to WACC technology names
    carrier_to_wacc_tech = {
        'solar': 'solar',
        'onwind': 'onwind',
        'offwind-ac': 'offwind',
        'offwind-dc': 'offwind',
    }
    
    updated_generators = 0
    
    for carrier, wacc_tech in carrier_to_wacc_tech.items():
        if wacc_tech not in wacc_dict:
            continue
        
        # Get generators with this carrier
        gen_mask = n.generators.carrier == carrier
        if not gen_mask.any():
            continue
        
        country_waccs = wacc_dict[wacc_tech]
        
        # Get cost parameters for this technology
        if carrier not in costs.index:
            logger.warning(f"Carrier '{carrier}' not found in costs. Skipping WACC update.")
            continue
        
        investment = costs.at[carrier, 'investment']
        lifetime = costs.at[carrier, 'lifetime']
        fom = costs.at[carrier, 'FOM']
        default_discount = costs.at[carrier, 'discount rate']
        
        # Update each generator based on its country
        for gen_idx in n.generators.index[gen_mask]:
            # Extract country code from generator bus name (e.g., "NG 0" -> "NG")
            bus_name = n.generators.at[gen_idx, 'bus']
            country_code = str(bus_name).split(" ")[0][:2]
            
            # Get country-specific WACC or use default
            if country_code in country_waccs:
                wacc = country_waccs[country_code]
                
                # Recalculate capital cost with country-specific WACC
                annuity_factor = annuity(lifetime, wacc) + fom / 100.0
                new_capital_cost = annuity_factor * investment * Nyears
                
                n.generators.at[gen_idx, 'capital_cost'] = new_capital_cost
                updated_generators += 1
    
    logger.info(f"Applied regional WACCs to {updated_generators} renewable generators")
    logger.info(f"  Updated carriers: {', '.join(carrier_to_wacc_tech.keys())}")


import psutil, os
def print_memory(note=""):
    mem = psutil.Process(os.getpid()).memory_info().rss / 1e9
    print(f"[MEM] {note}: {mem:.2f} GB")


def add_emissions_from_weighted_energy(
    n,
    name: str,
    p_set_df, # DataFrame [snapshots x nodes], MW
    co2_intensity_t_per_MWh: float, # t/MWh (fuel energy basis)
    bus: str, # e.g. "co2 atmosphere"
    carrier: str, # e.g. "oil emissions"
    flat: bool = True, # flat = spread evenly; False = time-resolved
):
    """
    Adds a CO2 Load whose p_set is computed from weighted energy.
    Works for uniform or non-uniform snapshot weights (later for TSAM typical days).
    """
    w = n.snapshot_weightings.generators
    
    if isinstance(p_set_df, pd.DataFrame):
        P = p_set_df.sum(axis=1)
    else:
        P = p_set_df
        
    P = P.reindex(w.index).fillna(0.0)

    E_MWh = (P * w).sum()

    co2_total_t = float(E_MWh) * float(co2_intensity_t_per_MWh)

    if flat:
        # Spread total tonnes evenly as a constant rate (t/h) over the horizon
        total_hours = w.sum()
        co2_rate_t_per_h = 0.0 if total_hours == 0 else co2_total_t / total_hours

        # Add as scalar p_set
        n.add("Load", name, bus=bus, carrier=carrier, p_set=-co2_rate_t_per_h)

    else:
        # Time-resolved emissions (per-snapshot)
        co2_ts_t_per_h = -P * co2_intensity_t_per_MWh

        if name not in n.loads.index:
            n.add("Load", name, bus=bus, carrier=carrier)

        n.loads_t.p_set.loc[:, name] = co2_ts_t_per_h


def add_lifetime_wind_solar(n, costs):
    """
    Add lifetime for solar and wind generators.
    """
    for carrier in ["solar", "onwind", "offwind"]:
        gen_i = n.generators.index.str.contains(carrier)
        n.generators.loc[gen_i, "lifetime"] = costs.at[carrier, "lifetime"]


def add_carrier_buses(n, carrier, nodes=None, fuel_price_dict=None):
    """
    Add buses to connect e.g. coal, nuclear and oil plants.
    """

    if nodes is None:
        nodes = vars(spatial)[carrier].nodes
    location = vars(spatial)[carrier].locations

    # skip if carrier already exists
    if carrier in n.carriers.index:
        return

    if not isinstance(nodes, pd.Index):
        nodes = pd.Index(nodes)

    n.add("Carrier", carrier, co2_emissions=costs.at[carrier, "CO2 intensity"])

    n.madd("Bus", nodes, location=location, carrier=carrier)

    # initial fossil reserves
    e_initial = (snakemake.params.fossil_reserves).get(carrier, 0) * 1e6
    # capital cost could be corrected to e.g. 0.2 EUR/kWh * annuity and O&M
    n.madd(
        "Store",
        nodes + " Store",
        bus=nodes,
        e_nom_extendable=True,
        e_cyclic=True if e_initial == 0 else False,
        carrier=carrier,
        e_initial=e_initial,
    )

    # Use country-specific fuel prices for oil, gas, and coal
    if carrier in ['oil', 'gas', 'coal'] and fuel_price_dict is not None:
        marginal_cost = get_fuel_price_by_node(nodes, carrier, costs, fuel_price_dict)
    else:
        marginal_cost = costs.at[carrier, "fuel"]

    n.madd(
        "Generator",
        nodes,
        bus=nodes,
        p_nom_extendable=True,
        carrier=carrier,
        marginal_cost=marginal_cost,
    )


def ensure_biomass_resource_bus(n):
    """Ensure the finite biomass resource bus exists before biomass links are added."""
    if "solid biomass" not in n.carriers.index:
        n.add("Carrier", "solid biomass")

    if biomass_allocation is not None:
        carrier_name = "solid biomass power"
        if carrier_name not in n.carriers.index:
            n.add("Carrier", carrier_name)
        biomass_buses = pd.Index(spatial.biomass.power)
    else:
        carrier_name = "solid biomass"
        biomass_buses = pd.Index(spatial.biomass.nodes)

    biomass_buses_new = biomass_buses.difference(n.buses.index)
    if len(biomass_buses_new):
        n.madd(
            "Bus",
            biomass_buses_new,
            location=biomass_buses_new.to_series().map(
                dict(zip(biomass_buses, spatial.biomass.locations))
            ),
            carrier=carrier_name,
        )


def broadcast_madd_value(values, target_index, label):
    """Broadcast a single static value across madd rows while preserving aligned vectors."""
    if isinstance(values, str) or np.isscalar(values):
        return values

    target_len = len(pd.Index(target_index))
    seq = pd.Index(values)
    if len(seq) == 1:
        return seq[0]
    if len(seq) == target_len:
        return values
    raise ValueError(
        f"{label} length {len(seq)} does not match target length {target_len}"
    )


def match_geothermal_capacity_from_csv(n, baseyear, geothermal_csv_path, costs=None, 
                                        create_synthetic=False, 
                                        regions_shapefile=None):
    """
    Match geothermal capacity in the network to values from a CSV file by country.
    Optionally create synthetic generators for countries not in the network.
    
    Parameters
    ----------
    n : pypsa.Network
    baseyear : int
        Year to match capacities for
    geothermal_csv_path : str
        Path to CSV file with columns: Entity, Code, Year, Geothermal capacity (total)
    costs : pd.DataFrame, optional
        Cost data for generator parameters (required if create_synthetic=True)
    create_synthetic : bool, default True
        Whether to create synthetic generators for countries not in network
    regions_shapefile : str, optional
        Path to shapefile with bus regions for determining centroids
    """
    # Read geothermal capacity data
    try:
        geo_df = pd.read_csv(geothermal_csv_path)
    except FileNotFoundError:
        logger.warning(f"Geothermal capacity CSV not found at {geothermal_csv_path}, skipping geothermal capacity matching")
        return
    
    # Filter for the baseyear
    geo_df = geo_df[geo_df['Year'] == baseyear]
    
    if geo_df.empty:
        logger.warning(f"No geothermal capacity data found for year {baseyear}, skipping geothermal capacity matching")
        return
    
    # Get geothermal generators from network
    geo_gens = n.generators[n.generators.carrier == "geothermal"]
    
    
    # Extract country from generator bus names (format: "XX 0" where XX is 2-digit country code)
    if not geo_gens.empty:
        geo_gens_country = geo_gens.bus.str[:2]
        # Aggregate current geothermal capacity by country
        current_capacity_by_country = geo_gens.groupby(geo_gens_country)['p_nom'].sum()
    else:
        logger.info("No existing geothermal generators found in network")
        current_capacity_by_country = pd.Series(dtype=float)
    
    # Process each country in the CSV
    countries_matched = 0
    countries_not_in_network = []
    countries_with_synthetic = []
    
    for _, row in geo_df.iterrows():
        country_code_3 = row['Code']
        target_capacity_mw = row['Geothermal capacity (total)']
        
        # Skip rows without country codes (e.g., regional aggregates)
        if pd.isna(country_code_3) or country_code_3 == '':
            continue
        
        # Skip if capacity is zero or negligible
        if target_capacity_mw < 0.1:
            continue
        
        # Convert 3-digit to 2-digit country code
        try:
            country_code_2 = three_2_two_digits_country(country_code_3)
        except Exception as e:
            logger.warning(f"Could not convert country code {country_code_3} to 2-digit: {e}")
            continue
        
        # Check if country exists in network
        if country_code_2 not in current_capacity_by_country.index:
            countries_not_in_network.append((row['Entity'], country_code_3, country_code_2, target_capacity_mw))
            continue
        
        current_capacity = current_capacity_by_country[country_code_2]
        
        # Get all geothermal generators for this country
        country_geo_gens = geo_gens[geo_gens_country == country_code_2]
        
        if len(country_geo_gens) == 0:
            continue
        
        # Adjust p_nom to match target capacity, distributed proportionally
        if current_capacity > 0:
            scaling_factor = target_capacity_mw / current_capacity
            n.generators.loc[country_geo_gens.index, 'p_nom'] *= scaling_factor
            n.generators.loc[country_geo_gens.index, 'p_nom_min'] *= scaling_factor
            
            countries_matched += 1
        else:
            logger.warning(
                f"  {row['Entity']} ({country_code_2}): Current capacity is 0, cannot scale to {target_capacity_mw:.1f} MW"
            )

    
    # Create synthetic generators for countries not in network
    if create_synthetic and countries_not_in_network and costs is not None:
        
        # Try to load shapefile for centroids
        regions_gdf = None
        if regions_shapefile and os.path.exists(regions_shapefile):
            try:
                import geopandas as gpd
                regions_gdf = gpd.read_file(regions_shapefile)
                logger.info(f"Loaded regions shapefile with {len(regions_gdf)} regions")
            except Exception as e:
                logger.warning(f"Could not load regions shapefile: {e}. Will skip synthetic generator creation.")
                regions_gdf = None
        
        if regions_gdf is not None:
            for entity, code_3, code_2, capacity_mw in countries_not_in_network:
                # Find all regions for this country
                country_regions = regions_gdf[regions_gdf['country'] == code_2]
                
                if country_regions.empty:
                    logger.debug(f"No regions found for {entity} ({code_2}) in shapefile, skipping")
                    continue
                
                # Divide capacity evenly among regions
                n_regions = len(country_regions)
                capacity_per_region = capacity_mw / n_regions
                
                for idx, region in country_regions.iterrows():
                    bus_name = region['name']
                    
                    # Check if bus already exists
                    if bus_name not in n.buses.index:
                        # Create new AC bus with coordinates from shapefile
                        n.add("Bus",
                              bus_name,
                              carrier="AC",
                              country=code_2,
                              x=region['x'],
                              y=region['y'])
                        logger.debug(f"Created bus {bus_name} at ({region['x']:.3f}, {region['y']:.3f})")
                    
                    # Create geothermal generator
                    gen_name = f"{bus_name} geothermal-{baseyear}"
                    
                    n.add("Generator",
                          gen_name,
                          bus=bus_name,
                          carrier="geothermal",
                          p_nom=capacity_per_region,
                          p_nom_min=capacity_per_region,
                          marginal_cost=costs.at["geothermal", "VOM"],
                          capital_cost=costs.at["geothermal", "fixed"],
                          efficiency=costs.at["geothermal", "efficiency"],
                          build_year=baseyear,
                          lifetime=costs.at["geothermal", "lifetime"],
                          p_nom_extendable=False)
                
                countries_with_synthetic.append(entity)
                logger.info(
                    f"  {entity} ({code_2}): Created {n_regions} synthetic generator(s) "
                    f"with {capacity_per_region:.1f} MW each (total: {capacity_mw:.1f} MW)"
                )
        
        if countries_with_synthetic:
            logger.info(f"Created synthetic geothermal generators for {len(countries_with_synthetic)} countries")
    
    elif countries_not_in_network:
        logger.warning(
            f"Countries in CSV but not in network ({len(countries_not_in_network)}): "
            f"{', '.join([f'{e} ({c})' for e, c, _, _ in countries_not_in_network[:10]])}"
            + (f" and {len(countries_not_in_network) - 10} more" if len(countries_not_in_network) > 10 else "")
        )

def add_generation(
    n, costs, existing_capacities=0, existing_efficiencies=None, existing_nodes=None, fuel_price_dict=None
):
    """
    Adds conventional generation as specified in config.

    Args:
        n (network): PyPSA prenetwork
        costs (dataframe): _description_
        existing_capacities: dictionary containing installed capacities for conventional_generation technologies
        existing_efficiencies: dictionary containing efficiencies for conventional_generation technologies
        existing_nodes: dictionary containing nodes for conventional_generation technologies
        fuel_price_dict: dictionary of country-specific fuel prices

    Returns:
        _type_: _description_
    """

    logger.info("adding electricity generation")

    # Not required, because nodes are already defined in "nodes"
    # nodes = pop_layout.index

    fallback = {"OCGT": "gas"}
    conventionals = options.get("conventional_generation", fallback)

    for generator, carrier in conventionals.items():
        if generator == "biomass":
            ensure_biomass_resource_bus(n)
            if generator not in n.carriers.index:
                n.add("Carrier", generator, co2_emissions=0.0)
            carrier_nodes = broadcast_madd_value(
                spatial.biomass.power if biomass_allocation is not None else spatial.biomass.nodes,
                spatial.nodes,
                "biomass generation bus0",
            )
        else:
            add_carrier_buses(n, carrier, fuel_price_dict=fuel_price_dict)
            carrier_nodes = vars(spatial)[carrier].nodes
        link_names = spatial.nodes + " " + generator
        n.madd(
            "Link",
            link_names,
            bus0=carrier_nodes,
            bus1=spatial.nodes,
            bus2="co2 atmosphere",
            marginal_cost=costs.at[generator, "efficiency"]
            * costs.at[generator, "VOM"],  # NB: VOM is per MWel
            # NB: fixed cost is per MWel
            capital_cost=costs.at[generator, "efficiency"]
            * costs.at[generator, "fixed"],
            p_nom_extendable=(
                True
                if generator
                in snakemake.params.electricity.get("extendable_carriers", dict()).get(
                    "Generator", list()
                )
                else False
            ),
            p_nom=(
                (
                    existing_capacities[generator] / existing_efficiencies[generator]
                ).reindex(link_names, fill_value=0)
                if not existing_capacities == 0
                else 0
            ),  # NB: existing capacities are MWel
            carrier=generator,
            efficiency=(
                existing_efficiencies[generator].reindex(
                    link_names, fill_value=costs.at[generator, "efficiency"]
                )
                if existing_efficiencies is not None
                else costs.at[generator, "efficiency"]
            ),
            efficiency2=costs.at[carrier, "CO2 intensity"],
            lifetime=costs.at[generator, "lifetime"],
        )

        # remove newly added links that have no capacity and are not extendable
        to_remove = n.links.query(
            "carrier == @carrier & p_nom == 0 & not p_nom_extendable"
        ).index
        n.mremove("Link", to_remove)

        # set the "co2_emissions" of the carrier to 0, as emissions are accounted by link efficiency separately (efficiency to 'co2 atmosphere' bus)
        n.carriers.loc[carrier, "co2_emissions"] = 0


def H2_liquid_fossil_conversions(n, costs):
    """
    Function to add conversions between H2 and liquid fossil Carrier and bus is
    added in add_oil, which later on might be switched to add_generation.
    """
    logger.info("Adding H2 to liquid fossil conversions")
    n.madd(
        "Link",
        spatial.nodes + " Fischer-Tropsch",
        bus0=spatial.nodes + " H2",
        bus1=spatial.oil.nodes,
        bus2=spatial.co2.nodes,
        bus3=spatial.nodes,
        carrier="Fischer-Tropsch",
        efficiency=costs.at["Fischer-Tropsch", "efficiency"],
        capital_cost=costs.at["Fischer-Tropsch", "fixed"]
        * costs.at[
            "Fischer-Tropsch", "efficiency"
        ],  # Use efficiency to convert from EUR/MW_FT/a to EUR/MW_H2/a
        efficiency2=-costs.at["oil", "CO2 intensity"]
        * costs.at["Fischer-Tropsch", "efficiency"],
        efficiency3=-costs.at["Fischer-Tropsch", "electricity-input"]
        / costs.at["Fischer-Tropsch", "hydrogen-input"],
        p_nom_extendable=True,
        p_min_pu=options.get("min_part_load_fischer_tropsch", 0),
        lifetime=costs.at["Fischer-Tropsch", "lifetime"],
    )


def add_hydrogen(n, costs):
    "function to add hydrogen as an energy carrier with its conversion technologies from and to AC"
    logger.info("Adding hydrogen")

    n.add("Carrier", "H2")

    n.madd(
        "Bus",
        spatial.nodes + " H2",
        location=spatial.nodes,
        carrier="H2",
        x=n.buses.loc[list(spatial.nodes)].x.values,
        y=n.buses.loc[list(spatial.nodes)].y.values,
    )

    # Read hydrogen production technologies
    h2_techs = options["hydrogen"].get("production_technologies", [])

    # Dictionary containing distinct parameters of H2 production technologies
    tech_params = {
        "H2 Electrolysis": {
            "cost_name": "electrolysis",
            "bus0": spatial.nodes,
            "bus1": spatial.nodes + " grid H2",
            "efficiency": costs.at["electrolysis", "efficiency"],
        },
        "Alkaline electrolyzer large": {
            "cost_name": "Alkaline electrolyzer large size",
            "bus0": spatial.nodes,
            "bus1": spatial.nodes + " grid H2",
            "efficiency": 1
            / costs.at["Alkaline electrolyzer large size", "electricity-input"],
        },
        "Alkaline electrolyzer medium": {
            "cost_name": "Alkaline electrolyzer medium size",
            "bus0": spatial.nodes,
            "bus1": spatial.nodes + " grid H2",
            "efficiency": 1
            / costs.at["Alkaline electrolyzer medium size", "electricity-input"],
        },
        "Alkaline electrolyzer small": {
            "cost_name": "Alkaline electrolyzer small size",
            "bus0": spatial.nodes,
            "bus1": spatial.nodes + " grid H2",
            "efficiency": 1
            / costs.at["Alkaline electrolyzer small size", "electricity-input"],
        },
        "PEM electrolyzer": {
            "cost_name": "PEM electrolyzer small size",
            "bus0": spatial.nodes,
            "bus1": spatial.nodes + " grid H2",
            "efficiency": 1
            / costs.at["PEM electrolyzer small size", "electricity-input"],
        },
        "SOEC": {
            "cost_name": "SOEC",
            "bus0": spatial.nodes,
            "bus1": spatial.nodes + " grid H2",
            "efficiency": 1 / costs.at["SOEC", "electricity-input"],
        },
        "Solid biomass steam reforming": {
            "cost_name": "H2 production solid biomass steam reforming",
            "bus0": broadcast_madd_value(
                spatial.biomass.nodes,
                spatial.nodes,
                "solid biomass steam reforming bus0",
            ),
            "bus1": spatial.nodes + " green H2",
            "bus2": spatial.nodes,
            "bus3": "co2 atmosphere",
            "efficiency": 1
            / costs.at["H2 production solid biomass steam reforming", "wood-input"],
            "efficiency2": -costs.at[
                "H2 production solid biomass steam reforming", "electricity-input"
            ]
            / costs.at["H2 production solid biomass steam reforming", "wood-input"],
            "efficiency3": costs.at["solid biomass", "CO2 intensity"],
        },
        "Biomass gasification": {
            "cost_name": "H2 production biomass gasification",
            "bus0": broadcast_madd_value(
                spatial.biomass.nodes,
                spatial.nodes,
                "biomass gasification bus0",
            ),
            "bus1": spatial.nodes + " green H2",
            "bus2": spatial.nodes,
            "bus3": "co2 atmosphere",
            "efficiency": 1
            / costs.at["H2 production biomass gasification", "wood-input"],
            "efficiency2": -costs.at[
                "H2 production biomass gasification", "electricity-input"
            ]
            / costs.at["H2 production biomass gasification", "wood-input"],
            "efficiency3": costs.at["solid biomass", "CO2 intensity"],
        },
        "Biomass gasification CC": {
            "cost_name": "H2 production biomass gasification CC",
            "bus0": broadcast_madd_value(
                spatial.biomass.nodes,
                spatial.nodes,
                "biomass gasification CC bus0",
            ),
            "bus1": spatial.nodes + " green H2",
            "bus2": spatial.nodes,
            "bus3": "co2 atmosphere",
            "bus4": spatial.co2.nodes,
            "efficiency": 1
            / costs.at["H2 production biomass gasification CC", "wood-input"],
            "efficiency2": -costs.at[
                "H2 production biomass gasification CC", "electricity-input"
            ]
            / costs.at["H2 production biomass gasification CC", "wood-input"],
            "efficiency3": costs.at["solid biomass", "CO2 intensity"]
            * (1 - options["cc_fraction"]),
            "efficiency4": costs.at["solid biomass", "CO2 intensity"]
            * options["cc_fraction"],
        },
        "SMR": {
            "cost_name": "SMR",
            "bus0": spatial.gas.nodes,
            "bus1": spatial.nodes + " grey H2",
            "bus2": "co2 atmosphere",
            "efficiency": costs.at["SMR", "efficiency"],
            "efficiency2": costs.at["gas", "CO2 intensity"],
        },
        "SMR CC": {
            "cost_name": "SMR CC",
            "bus0": spatial.gas.nodes,
            "bus1": spatial.nodes + " blue H2",
            "bus2": "co2 atmosphere",
            "bus3": spatial.co2.nodes,
            "efficiency": costs.at["SMR CC", "efficiency"],
            "efficiency2": costs.at["gas", "CO2 intensity"]
            * (1 - options["cc_fraction"]),
            "efficiency3": costs.at["gas", "CO2 intensity"] * options["cc_fraction"],
        },
        "Natural gas steam reforming": {
            "cost_name": "H2 production natural gas steam reforming",
            "bus0": spatial.gas.nodes,
            "bus1": spatial.nodes + " grey H2",
            "bus2": spatial.nodes,
            "bus3": "co2 atmosphere",
            "efficiency": 1
            / costs.at["H2 production natural gas steam reforming", "gas-input"],
            "efficiency2": -costs.at[
                "H2 production natural gas steam reforming", "electricity-input"
            ]
            / costs.at["H2 production natural gas steam reforming", "gas-input"],
            "efficiency3": costs.at["gas", "CO2 intensity"],
        },
        "Natural gas steam reforming CC": {
            "cost_name": "H2 production natural gas steam reforming CC",
            "bus0": spatial.gas.nodes,
            "bus1": spatial.nodes + " blue H2",
            "bus2": spatial.nodes,
            "bus3": "co2 atmosphere",
            "bus4": spatial.co2.nodes,
            "efficiency": 1
            / costs.at["H2 production natural gas steam reforming CC", "gas-input"],
            "efficiency2": -costs.at[
                "H2 production natural gas steam reforming CC", "electricity-input"
            ]
            / costs.at["H2 production natural gas steam reforming CC", "gas-input"],
            "efficiency3": costs.at["gas", "CO2 intensity"]
            * (1 - options["cc_fraction"]),
            "efficiency4": costs.at["gas", "CO2 intensity"] * options["cc_fraction"],
        },
        "Coal gasification": {
            "cost_name": "H2 production coal gasification",
            "bus0": spatial.coal.nodes,
            "bus1": spatial.nodes + " grey H2",
            "bus2": spatial.nodes,
            "bus3": "co2 atmosphere",
            "efficiency": 1 / costs.at["H2 production coal gasification", "coal-input"],
            "efficiency2": -costs.at[
                "H2 production coal gasification", "electricity-input"
            ]
            / costs.at["H2 production coal gasification", "coal-input"],
            "efficiency3": costs.at["coal", "CO2 intensity"],
        },
        "Coal gasification CC": {
            "cost_name": "H2 production coal gasification CC",
            "bus0": spatial.coal.nodes,
            "bus1": spatial.nodes + " blue H2",
            "bus2": spatial.nodes,
            "bus3": "co2 atmosphere",
            "bus4": spatial.co2.nodes,
            "efficiency": 1
            / costs.at["H2 production coal gasification CC", "coal-input"],
            "efficiency2": -costs.at[
                "H2 production coal gasification CC", "electricity-input"
            ]
            / costs.at["H2 production coal gasification CC", "coal-input"],
            "efficiency3": costs.at["coal", "CO2 intensity"]
            * (1 - options["cc_fraction"]),
            "efficiency4": costs.at["coal", "CO2 intensity"] * options["cc_fraction"],
        },
        "Heavy oil partial oxidation": {
            "cost_name": "H2 production heavy oil partial oxidation",
            "bus0": spatial.oil.nodes,
            "bus1": spatial.nodes + " grey H2",
            "bus2": spatial.nodes,
            "bus3": "co2 atmosphere",
            "efficiency": 1
            / costs.at["H2 production heavy oil partial oxidation", "oil-input"],
            "efficiency2": -costs.at[
                "H2 production heavy oil partial oxidation", "electricity-input"
            ]
            / costs.at["H2 production heavy oil partial oxidation", "oil-input"],
            "efficiency3": costs.at["oil", "CO2 intensity"],
        },
    }

    if options["hydrogen"].get("hydrogen_colors", False):
        color_techs = {
            "grid H2": [
                "H2 Electrolysis",
                "Alkaline electrolyzer large",
                "Alkaline electrolyzer medium",
                "Alkaline electrolyzer small",
                "PEM electrolyzer",
                "SOEC",
            ],
            "green H2": [
                "Solid biomass steam reforming",
                "Biomass gasification",
                "Biomass gasification CC",
            ],
            "grey H2": [
                "SMR",
                "Natural gas steam reforming",
                "Coal gasification",
                "Heavy oil partial oxidation",
            ],
            "blue H2": [
                "SMR CC",
                "Natural gas steam reforming CC",
                "Coal gasification CC",
            ],
        }

        for color, techs in color_techs.items():
            if set(h2_techs) & set(techs):
                n.madd(
                    "Bus",
                    spatial.nodes + f" {color}",
                    location=spatial.nodes,
                    carrier=color,
                    x=n.buses.loc[list(spatial.nodes)].x.values,
                    y=n.buses.loc[list(spatial.nodes)].y.values,
                )
                n.madd(
                    "Link",
                    spatial.nodes + f" {color}",
                    bus0=spatial.nodes + f" {color}",
                    bus1=spatial.nodes + " H2",
                    p_nom_extendable=True,
                    carrier=color,
                    efficiency=1,
                    capital_cost=0,
                )

    # Add hydrogen production technologies
    for h2_tech in h2_techs:
        # Set H2 buses as production output if colors are not used
        params = tech_params[h2_tech]
        bus1 = (
            params["bus1"]
            if options["hydrogen"].get("hydrogen_colors", False)
            else spatial.nodes + " H2"
        )

        n.madd(
            "Link",
            spatial.nodes + " " + h2_tech,
            bus0=params["bus0"],
            bus1=bus1,
            bus2=params.get("bus2", None),
            bus3=params.get("bus3", None),
            bus4=params.get("bus4", None),
            p_nom_extendable=True,
            carrier=h2_tech,
            efficiency=params["efficiency"],
            efficiency2=params.get("efficiency2", 1.0),
            efficiency3=params.get("efficiency3", 1.0),
            efficiency4=params.get("efficiency4", 1.0),
            capital_cost=costs.at[params["cost_name"], "fixed"],
            lifetime=costs.at[params["cost_name"], "lifetime"],
        )

    n.madd(
        "Link",
        spatial.nodes + " H2 Fuel Cell",
        bus0=spatial.nodes + " H2",
        bus1=spatial.nodes,
        p_nom_extendable=True,
        carrier="H2 Fuel Cell",
        efficiency=costs.at["fuel cell", "efficiency"],
        # NB: fixed cost is per MWel
        capital_cost=costs.at["fuel cell", "fixed"]
        * costs.at["fuel cell", "efficiency"],
        lifetime=costs.at["fuel cell", "lifetime"],
    )

    cavern_nodes = pd.DataFrame()

    if snakemake.params.sector_options["hydrogen"]["underground_storage"]:
        if snakemake.params.h2_underground:
            custom_cavern = pd.read_csv(
                os.path.join(
                    BASE_DIR,
                    "data/custom/h2_underground_{0}_{1}.csv".format(
                        demand_sc, investment_year
                    ),
                )
            )
            # countries = n.buses.country.unique().to_list()
            countries = snakemake.params.countries
            custom_cavern = custom_cavern[custom_cavern.country.isin(countries)]

            cavern_nodes = n.buses[n.buses.country.isin(custom_cavern.country)]

            h2_pot = custom_cavern.set_index("id_region")["storage_cap_MWh"]

            h2_capital_cost = costs.at["hydrogen storage underground", "fixed"]

            # h2_pot.index = cavern_nodes.index

            # n.add("Carrier", "H2 UHS")

            n.madd(
                "Bus",
                nodes + " H2 UHS",
                location=nodes,
                carrier="H2 UHS",
                x=n.buses.loc[list(nodes)].x.values,
                y=n.buses.loc[list(nodes)].y.values,
            )

            n.madd(
                "Store",
                cavern_nodes.index + " H2 UHS",
                bus=cavern_nodes.index + " H2 UHS",
                e_nom_extendable=True,
                e_nom_max=h2_pot.values,
                e_cyclic=True,
                carrier="H2 UHS",
                capital_cost=h2_capital_cost,
            )

            n.madd(
                "Link",
                nodes + " H2 UHS charger",
                bus0=nodes + " H2",
                bus1=nodes + " H2 UHS",
                carrier="H2 UHS charger",
                # efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
                # capital_cost=costs.at["battery inverter", "fixed"],
                p_nom_extendable=True,
                # lifetime=costs.at["battery inverter", "lifetime"],
            )

            n.madd(
                "Link",
                nodes + " H2 UHS discharger",
                bus0=nodes + " H2 UHS",
                bus1=nodes + " H2",
                carrier="H2 UHS discharger",
                efficiency=1,
                # capital_cost=costs.at["battery inverter", "fixed"],
                p_nom_extendable=True,
                # lifetime=costs.at["battery inverter", "lifetime"],
            )

        else:
            h2_salt_cavern_potential = pd.read_csv(
                snakemake.input.h2_cavern, index_col=0
            ).squeeze()
            h2_cavern_ct = h2_salt_cavern_potential[~h2_salt_cavern_potential.isna()]
            cavern_nodes = n.buses[n.buses.country.isin(h2_cavern_ct.index)]

            h2_capital_cost = costs.at["hydrogen storage underground", "fixed"]

            # assumptions: weight storage potential in a country by population
            # TODO: fix with real geographic potentials
            # convert TWh to MWh with 1e6
            h2_pot = h2_cavern_ct.loc[cavern_nodes.country]
            h2_pot.index = cavern_nodes.index

            # distribute underground potential equally over all nodes #TODO change with real data
            s = pd.Series(h2_pot.index, index=h2_pot.index)
            country_codes = s.str[:2]
            code_counts = country_codes.value_counts()
            fractions = country_codes.map(code_counts).rdiv(1)
            h2_pot = h2_pot * fractions * 1e6

            # n.add("Carrier", "H2 UHS")

            n.madd(
                "Bus",
                nodes + " H2 UHS",
                location=nodes,
                carrier="H2 UHS",
                x=n.buses.loc[list(nodes)].x.values,
                y=n.buses.loc[list(nodes)].y.values,
            )

            n.madd(
                "Store",
                cavern_nodes.index + " H2 UHS",
                bus=cavern_nodes.index + " H2 UHS",
                e_nom_extendable=True,
                e_nom_max=h2_pot.values,
                e_cyclic=True,
                carrier="H2 UHS",
                capital_cost=h2_capital_cost,
            )

            n.madd(
                "Link",
                nodes + " H2 UHS charger",
                bus0=nodes,
                bus1=nodes + " H2 UHS",
                carrier="H2 UHS charger",
                # efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
                capital_cost=0,
                p_nom_extendable=True,
                # lifetime=costs.at["battery inverter", "lifetime"],
            )

            n.madd(
                "Link",
                nodes + " H2 UHS discharger",
                bus0=nodes,
                bus1=nodes + " H2 UHS",
                carrier="H2 UHS discharger",
                efficiency=1,
                capital_cost=0,
                p_nom_extendable=True,
                # lifetime=costs.at["battery inverter", "lifetime"],
            )

    # hydrogen stored overground (where not already underground)
    h2_capital_cost = costs.at[
        "hydrogen storage tank type 1 including compressor", "fixed"
    ]
    nodes_overground = nodes
    n.madd(
        "Store",
        nodes_overground + " H2 Store Tank",
        bus=nodes_overground + " H2",
        e_nom_extendable=True,
        e_cyclic=True,
        carrier="H2 Store Tank",
        capital_cost=h2_capital_cost,
    )

    # Hydrogen network:
    # -----------------
    def add_links_repurposed_H2_pipelines():
        n.madd(
            "Link",
            h2_links.index + " repurposed",
            bus0=h2_links.bus0.values + " H2",
            bus1=h2_links.bus1.values + " H2",
            p_min_pu=-1,
            p_nom_extendable=True,
            p_nom_max=h2_links.capacity.values
            * 0.8,  # https://gasforclimate2050.eu/wp-content/uploads/2020/07/2020_European-Hydrogen-Backbone_Report.pdf
            length=h2_links.length.values,
            capital_cost=costs.at["H2 (g) pipeline repurposed", "fixed"]
            * h2_links.length.values,
            carrier="H2 pipeline repurposed",
            lifetime=costs.at["H2 (g) pipeline repurposed", "lifetime"],
        )

    def add_links_new_H2_pipelines():
        n.madd(
            "Link",
            h2_links.index,
            bus0=h2_links.bus0.values + " H2",
            bus1=h2_links.bus1.values + " H2",
            p_min_pu=-1,
            p_nom_extendable=True,
            length=h2_links.length.values,
            capital_cost=costs.at["H2 (g) pipeline", "fixed"] * h2_links.length.values,
            carrier="H2 pipeline",
            lifetime=costs.at["H2 (g) pipeline", "lifetime"],
        )

    def add_links_elec_routing_new_H2_pipelines():
        attrs = ["bus0", "bus1", "length"]
        h2_links = pd.DataFrame(columns=attrs)

        candidates = pd.concat(
            {
                "lines": n.lines[attrs],
                "links": n.links.loc[n.links.carrier == "DC", attrs],
            }
        )

        for candidate in candidates.index:
            buses = [
                candidates.at[candidate, "bus0"],
                candidates.at[candidate, "bus1"],
            ]
            buses.sort()
            name = f"H2 pipeline {buses[0]} -> {buses[1]}"
            if name not in h2_links.index:
                h2_links.at[name, "bus0"] = buses[0]
                h2_links.at[name, "bus1"] = buses[1]
                h2_links.at[name, "length"] = candidates.at[candidate, "length"]

        n.madd(
            "Link",
            h2_links.index,
            bus0=h2_links.bus0.values + " H2",
            bus1=h2_links.bus1.values + " H2",
            p_min_pu=-1,
            p_nom_extendable=True,
            length=h2_links.length.values,
            capital_cost=costs.at["H2 (g) pipeline", "fixed"] * h2_links.length.values,
            carrier="H2 pipeline",
            lifetime=costs.at["H2 (g) pipeline", "lifetime"],
        )

    # Add H2 Links:
    if snakemake.params.sector_options["hydrogen"]["network"]:
        h2_links = pd.read_csv(snakemake.input.pipelines)

        def map_h2_node(x, gadm_clustering=False):
            try:
                iso3 = x[:3]
                iso2 = three_2_two_digits_country(iso3)
                return f"{iso2}_AC" if not gadm_clustering else f"{iso2}._AC"
            except Exception as e:
                logger.warning(f"Failed to map H2 pipeline node '{x}': {e}")
                return x
    
        original_bus0 = h2_links["bus0"].copy()
        original_bus1 = h2_links["bus1"].copy()
        h2_links["bus0"] = h2_links["bus0"].apply(map_h2_node, gadm_clustering=snakemake.params.alternative_clustering)
        h2_links["bus1"] = h2_links["bus1"].apply(map_h2_node, gadm_clustering=snakemake.params.alternative_clustering)
        
        logger.info(f"Mapped H2 pipeline buses from ISO3 format to ISO2_AC format:")
        logger.info(f"Example mappings: {original_bus0.iloc[0]} -> {h2_links['bus0'].iloc[0]}, "
                   f"{original_bus1.iloc[0]} -> {h2_links['bus1'].iloc[0]}")

        h2_links["buses_idx"] = (
            "H2 pipeline " + h2_links["bus0"] + " -> " + h2_links["bus1"]
        )

        h2_links = h2_links.groupby("buses_idx").agg(
            {"bus0": "first", "bus1": "first", "length": "mean", "capacity": "sum"}
        )

        if len(h2_links) > 0:
            if snakemake.params.sector_options["hydrogen"]["gas_network_repurposing"]:
                add_links_repurposed_H2_pipelines()
            if (
                snakemake.params.sector_options["hydrogen"]["network_routes"]
                == "greenfield"
            ):
                add_links_elec_routing_new_H2_pipelines()
            else:
                add_links_new_H2_pipelines()
        else:
            print(
                "No existing gas network; applying greenfield for H2 network"
            )  # TODO change to logger.info
            add_links_elec_routing_new_H2_pipelines()

        if snakemake.params.sector_options["hydrogen"]["hydrogen_colors"]:
            nuclear_gens_bus = n.generators[
                n.generators.carrier == "nuclear"
            ].bus.values
            buses_with_nuclear = n.buses.loc[nuclear_gens_bus]
            buses_with_nuclear_ind = n.buses.loc[nuclear_gens_bus].index

            # nn.add("Carrier", "nuclear electricity")
            # nn.add("Carrier", "pink H2")

            n.madd(
                "Bus",
                nuclear_gens_bus + " nuclear electricity",
                location=buses_with_nuclear_ind,
                carrier="nuclear electricity",
                x=buses_with_nuclear.x.values,
                y=buses_with_nuclear.y.values,
            )

            n.madd(
                "Bus",
                nuclear_gens_bus + " pink H2",
                location=buses_with_nuclear_ind,
                carrier="pink H2",
                x=buses_with_nuclear.x.values,
                y=buses_with_nuclear.y.values,
            )

            n.generators.loc[n.generators.carrier == "nuclear", "bus"] = (
                n.generators.loc[n.generators.carrier == "nuclear", "bus"]
                + " nuclear electricity"
            )

            n.madd(
                "Link",
                buses_with_nuclear_ind + " nuclear-to-grid",
                bus0=buses_with_nuclear_ind + " nuclear electricity",
                bus1=buses_with_nuclear_ind,
                carrier="nuclear-to-grid",
                capital_cost=0,
                p_nom_extendable=True,
                # lifetime=costs.at["battery inverter", "lifetime"],
            )

            n.madd(
                "Link",
                buses_with_nuclear_ind + " high-temp electrolysis",
                bus0=buses_with_nuclear_ind + " nuclear electricity",
                bus1=buses_with_nuclear_ind + " pink H2",
                carrier="high-temp electrolysis",
                # capital_cost=0,
                p_nom_extendable=True,
                efficiency=costs.at["electrolysis", "efficiency"] + 0.1,
                capital_cost=costs.at["electrolysis", "fixed"]
                + costs.at["electrolysis", "fixed"] * 0.1,
                lifetime=costs.at["electrolysis", "lifetime"],
            )

            n.madd(
                "Link",
                buses_with_nuclear_ind + " pink H2",
                bus0=buses_with_nuclear_ind + " pink H2",
                bus1=buses_with_nuclear_ind + " H2",
                carrier="pink H2",
                # efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
                capital_cost=0,
                p_nom_extendable=True,
                # lifetime=costs.at["battery inverter", "lifetime"],
            )


def define_spatial(nodes, options):
    """
    Namespace for spatial.

    Parameters
    ----------
    nodes : list-like
    """

    global spatial

    spatial.nodes = nodes

    # biomass

    spatial.biomass = SimpleNamespace()

    if options["biomass_transport"]:
        spatial.biomass.nodes = nodes + " solid biomass"
        spatial.biomass.locations = nodes
        spatial.biomass.power = nodes + " solid biomass power"
        spatial.biomass.buildings = nodes + " solid biomass buildings"
        spatial.biomass.industry_resource = nodes + " solid biomass industry resource"
        spatial.biomass.industry = nodes + " solid biomass for industry"
        spatial.biomass.industry_cc = nodes + " solid biomass for industry CC"
    else:
        spatial.biomass.nodes = ["Earth solid biomass"]
        spatial.biomass.locations = ["Earth"]
        spatial.biomass.power = ["Earth solid biomass power"]
        spatial.biomass.buildings = ["Earth solid biomass buildings"]
        spatial.biomass.industry_resource = ["Earth solid biomass industry resource"]
        spatial.biomass.industry = ["solid biomass for industry"]
        spatial.biomass.industry_cc = ["solid biomass for industry CC"]

    spatial.biomass.df = pd.DataFrame(vars(spatial.biomass), index=nodes)

    # co2

    spatial.co2 = SimpleNamespace()

    if options["co2_network"]:
        spatial.co2.nodes = nodes + " co2 stored"
        spatial.co2.locations = nodes
        spatial.co2.vents = nodes + " co2 vent"
        # spatial.co2.x = (n.buses.loc[list(nodes)].x.values,)
        # spatial.co2.y = (n.buses.loc[list(nodes)].y.values,)
    else:
        spatial.co2.nodes = ["co2 stored"]
        spatial.co2.locations = ["Earth"]
        spatial.co2.vents = ["co2 vent"]
        # spatial.co2.x = (0,)
        # spatial.co2.y = 0

    spatial.co2.df = pd.DataFrame(vars(spatial.co2), index=nodes)

    # oil

    spatial.oil = SimpleNamespace()

    if options["oil"]["spatial_oil"]:
        spatial.oil.nodes = nodes + " oil"
        spatial.oil.locations = nodes
    else:
        spatial.oil.nodes = ["Earth oil"]
        spatial.oil.locations = ["Earth"]

    # gas

    spatial.gas = SimpleNamespace()

    if options["gas"]["spatial_gas"]:
        spatial.gas.nodes = nodes + " gas"
        spatial.gas.locations = nodes
        spatial.gas.biogas = nodes + " biogas"
        spatial.gas.industry = nodes + " gas for industry"
        if options["cc"]:
            spatial.gas.industry_cc = nodes + " gas for industry CC"
        spatial.gas.biogas_to_gas = nodes + " biogas to gas"
    else:
        spatial.gas.nodes = ["Earth gas"]
        spatial.gas.locations = ["Earth"]
        spatial.gas.biogas = ["Earth biogas"]
        spatial.gas.industry = ["gas for industry"]
        if options["cc"]:
            spatial.gas.industry_cc = ["gas for industry CC"]
        spatial.gas.biogas_to_gas = ["Earth biogas to gas"]

    spatial.gas.df = pd.DataFrame(vars(spatial.gas), index=spatial.nodes)

    # coal

    spatial.coal = SimpleNamespace()

    if options["coal"]["spatial_coal"]:
        spatial.coal.nodes = nodes + " coal"
        spatial.coal.locations = nodes
        spatial.coal.industry = nodes + " coal for industry"
    else:
        spatial.coal.nodes = ["Earth coal"]
        spatial.coal.locations = ["Earth"]
        spatial.coal.industry = ["Earth coal for industry"]

    spatial.coal.df = pd.DataFrame(vars(spatial.coal), index=spatial.nodes)

    # lignite

    spatial.lignite = SimpleNamespace()

    if options["lignite"]["spatial_lignite"]:
        spatial.lignite.nodes = nodes + " lignite"
        spatial.lignite.locations = nodes
    else:
        spatial.lignite.nodes = ["Earth lignite"]
        spatial.lignite.locations = ["Earth"]

    spatial.lignite.df = pd.DataFrame(vars(spatial.lignite), index=spatial.nodes)

    return spatial


def add_biomass(n, costs):
    logger.info("adding biomass")

    # TODO get biomass potentials dataset and enable spatially resolved potentials

    # Get biomass and biogas potentials from config and convert from TWh to MWh
    biomass_pot = (
        snakemake.params.sector_options["solid_biomass_potential"] * 1e6
    )  # MWh
    biogas_pot = snakemake.params.sector_options["biogas_potential"] * 1e6  # MWh
    logger.info("Biomass and Biogas potential fetched from config")

    # Convert from total to nodal potentials,
    biogas_pot_spatial = biogas_pot / len(spatial.gas.biogas)
    logger.info("Biomass potentials spatially resolved equally across all nodes")

    if "biogas" not in n.carriers.index:
        n.add("Carrier", "biogas")
    if "solid biomass" not in n.carriers.index:
        n.add("Carrier", "solid biomass")
    if biomass_allocation is not None:
        for carrier_name in [
            "solid biomass power",
            "solid biomass buildings",
            "solid biomass industry resource",
        ]:
            if carrier_name not in n.carriers.index:
                n.add("Carrier", carrier_name)

    biogas_buses = pd.Index(spatial.gas.biogas)
    biogas_buses_new = biogas_buses.difference(n.buses.index)
    if len(biogas_buses_new):
        n.madd(
            "Bus",
            biogas_buses_new,
            location=biogas_buses_new.to_series().map(dict(zip(spatial.gas.biogas, spatial.biomass.locations))),
            carrier="biogas",
        )

    n.madd(
        "Store",
        spatial.gas.biogas,
        bus=spatial.gas.biogas,
        carrier="biogas",
        e_nom=biogas_pot_spatial,
        marginal_cost=costs.at["biogas", "fuel"],
        e_initial=biogas_pot_spatial,
    )

    if biomass_allocation is not None:
        biomass_bus_specs = [
            (
                pd.Index(spatial.biomass.power),
                "solid biomass power",
                float(biomass_allocation["power_required_twh"]) * 1e6,
            ),
            (
                pd.Index(spatial.biomass.buildings),
                "solid biomass buildings",
                float(biomass_allocation["buildings_required_twh"]) * 1e6,
            ),
            (
                pd.Index(spatial.biomass.industry_resource),
                "solid biomass industry resource",
                float(biomass_allocation["industry_required_twh"]) * 1e6,
            ),
        ]
    else:
        biomass_bus_specs = [
            (
                pd.Index(spatial.biomass.nodes),
                "solid biomass",
                biomass_pot,
            )
        ]

    for buses, carrier_name, total_nom_mwh in biomass_bus_specs:
        buses_new = buses.difference(n.buses.index)
        if len(buses_new):
            n.madd(
                "Bus",
                buses_new,
                location=buses_new.to_series().map(dict(zip(buses, spatial.biomass.locations))),
                carrier=carrier_name,
            )

        if len(buses) == 0 or total_nom_mwh <= 0.0:
            continue

        e_nom_spatial = total_nom_mwh / len(buses)
        n.madd(
            "Store",
            buses,
            bus=buses,
            carrier=carrier_name,
            e_nom=e_nom_spatial,
            marginal_cost=costs.at["solid biomass", "fuel"],
            e_initial=e_nom_spatial,
        )

    biomass_gen = "biomass EOP"
    n.madd(
        "Link",
        spatial.nodes + " biomass EOP",
        bus0=broadcast_madd_value(
            spatial.biomass.power if biomass_allocation is not None else spatial.biomass.nodes,
            spatial.nodes,
            "biomass EOP bus0",
        ),
        bus1=spatial.nodes,
        # bus2="co2 atmosphere",
        marginal_cost=costs.at[biomass_gen, "efficiency"]
        * costs.at[biomass_gen, "VOM"],  # NB: VOM is per MWel
        # NB: fixed cost is per MWel
        capital_cost=costs.at[biomass_gen, "efficiency"]
        * costs.at[biomass_gen, "fixed"],
        p_nom_extendable=True,
        carrier=biomass_gen,
        efficiency=costs.at[biomass_gen, "efficiency"],
        # efficiency2=costs.at["solid biomass", "CO2 intensity"],
        lifetime=costs.at[biomass_gen, "lifetime"],
    )

    n.madd(
        "Link",
        spatial.gas.biogas_to_gas,
        bus0=spatial.gas.biogas,
        bus1=spatial.gas.nodes,
        bus2="co2 atmosphere",
        carrier="biogas to gas",
        capital_cost=costs.loc["biogas upgrading", "fixed"],
        marginal_cost=costs.loc["biogas upgrading", "VOM"],
        efficiency2=-costs.at["gas", "CO2 intensity"],
        p_nom_extendable=True,
    )

    if options["biomass_transport"]:
        # TODO add biomass transport costs
        transport_costs = pd.read_csv(
            snakemake.input.biomass_transport_costs,
            index_col=0,
            keep_default_na=False,
        ).squeeze()

        # add biomass transport
        biomass_transport = create_network_topology(
            n, "biomass transport ", bidirectional=False
        )

        # costs
        countries_not_in_index = set(countries) - set(biomass_transport.index)
        if countries_not_in_index:
            logger.info(
                "No transport values found for {0}, using default value of {1}".format(
                    ", ".join(countries_not_in_index),
                    snakemake.params.sector_options["biomass_transport_default_cost"],
                )
            )

        bus0_costs = biomass_transport.bus0.apply(
            lambda x: transport_costs.get(
                x[:2], snakemake.params.sector_options["biomass_transport_default_cost"]
            )
        )
        bus1_costs = biomass_transport.bus1.apply(
            lambda x: transport_costs.get(
                x[:2], snakemake.params.sector_options["biomass_transport_default_cost"]
            )
        )
        biomass_transport["costs"] = pd.concat([bus0_costs, bus1_costs], axis=1).mean(
            axis=1
        )

        n.madd(
            "Link",
            biomass_transport.index,
            bus0=biomass_transport.bus0 + " solid biomass",
            bus1=biomass_transport.bus1 + " solid biomass",
            p_nom_extendable=False,
            p_nom=5e4, # 50 GW
            length=biomass_transport.length.values,
            marginal_cost=biomass_transport.costs * biomass_transport.length.values,
            capital_cost=1,
            carrier="solid biomass transport",
        )

    # n.madd(
    #         "Link",
    #         urban_central + " urban central solid biomass CHP",
    #         bus0=spatial.biomass.df.loc[urban_central, "nodes"].values,
    #         bus1=urban_central,
    #         bus2=urban_central + " urban central heat",
    #         carrier="urban central solid biomass CHP",
    #         p_nom_extendable=True,
    #         capital_cost=costs.at[key, "fixed"] * costs.at[key, "efficiency"],
    #         marginal_cost=costs.at[key, "VOM"],
    #         efficiency=costs.at[key, "efficiency"],
    #         efficiency2=costs.at[key, "efficiency-heat"],
    #         lifetime=costs.at[key, "lifetime"],
    #     )

    # AC buses with district heating
    urban_central = n.buses.index[n.buses.carrier == "urban central heat"]
    if not urban_central.empty and options["chp"]:
        urban_central = urban_central.str[: -len(" urban central heat")]

        key = "central solid biomass CHP"

        n.madd(
            "Link",
            urban_central + " urban central solid biomass CHP",
            bus0=(
                spatial.biomass.df.loc[urban_central, "power"].values
                if biomass_allocation is not None
                else spatial.biomass.df.loc[urban_central, "nodes"].values
            ),
            bus1=urban_central,
            bus2=urban_central + " urban central heat",
            carrier="urban central solid biomass CHP",
            p_nom_extendable=True,
            capital_cost=costs.at[key, "fixed"] * costs.at[key, "efficiency"],
            marginal_cost=costs.at[key, "VOM"],
            efficiency=costs.at[key, "efficiency"],
            efficiency2=costs.at[key, "efficiency-heat"],
            lifetime=costs.at[key, "lifetime"],
        )

        if snakemake.params.sector_options["cc"]:
            n.madd(
                "Link",
                urban_central + " urban central solid biomass CHP CC",
                bus0=(
                    spatial.biomass.df.loc[urban_central, "power"].values
                    if biomass_allocation is not None
                    else spatial.biomass.df.loc[urban_central, "nodes"].values
                ),
                bus1=urban_central,
                bus2=urban_central + " urban central heat",
                bus3="co2 atmosphere",
                bus4=spatial.co2.df.loc[urban_central, "nodes"].values,
                carrier="urban central solid biomass CHP CC",
                p_nom_extendable=True,
                capital_cost=costs.at[key, "fixed"] * costs.at[key, "efficiency"]
                + costs.at["biomass CHP capture", "fixed"]
                * costs.at["solid biomass", "CO2 intensity"],
                marginal_cost=costs.at[key, "VOM"],
                efficiency=costs.at[key, "efficiency"]
                - costs.at["solid biomass", "CO2 intensity"]
                * (
                    costs.at["biomass CHP capture", "electricity-input"]
                    + costs.at["biomass CHP capture", "compression-electricity-input"]
                ),
                efficiency2=costs.at[key, "efficiency-heat"]
                + costs.at["solid biomass", "CO2 intensity"]
                * (
                    costs.at["biomass CHP capture", "heat-output"]
                    + costs.at["biomass CHP capture", "compression-heat-output"]
                    - costs.at["biomass CHP capture", "heat-input"]
                ),
                efficiency3=-costs.at["solid biomass", "CO2 intensity"]
                * costs.at["biomass CHP capture", "capture_rate"],
                efficiency4=costs.at["solid biomass", "CO2 intensity"]
                * costs.at["biomass CHP capture", "capture_rate"],
                lifetime=costs.at[key, "lifetime"],
            )


def add_co2(n, costs, co2_network):
    "add carbon carrier, it's networks and storage units"

    # minus sign because opposite to how fossil fuels used:
    # CH4 burning puts CH4 down, atmosphere up
    n.add("Carrier", "co2", co2_emissions=-1.0)

    # this tracks CO2 in the atmosphere
    n.add(
        "Bus",
        "co2 atmosphere",
        location="Earth",  # TODO Ignoed by pypsa check
        carrier="co2",
    )

    # can also be negative
    n.add(
        "Store",
        "co2 atmosphere",
        e_nom_extendable=True,
        e_min_pu=-1,
        carrier="co2",
        bus="co2 atmosphere",
    )

    # this tracks CO2 stored, e.g. underground
    n.madd(
        "Bus",
        spatial.co2.nodes,
        location=spatial.co2.locations,
        carrier="co2 stored",
        # x=spatial.co2.x[0],
        # y=spatial.co2.y[0],
    )
    """
    co2_stored_x = n.buses.filter(like="co2 stored", axis=0).loc[:, "x"]
    co2_stored_y = n.buses.loc[n.buses[n.buses.carrier == "co2
    stored"].location].y.

    n.buses[n.buses.carrier == "co2 stored"].x = co2_stored_x.values
    n.buses[n.buses.carrier == "co2 stored"].y = co2_stored_y.values
    """

    n.madd(
        "Link",
        spatial.co2.vents,
        bus0=spatial.co2.nodes,
        bus1="co2 atmosphere",
        carrier="co2 vent",
        efficiency=1.0,
        p_nom_extendable=True,
    )

    n.madd(
        "Store",
        spatial.co2.nodes,
        e_nom_extendable=True,
        e_nom_max=np.inf,
        capital_cost=options["co2_sequestration_cost"],
        carrier="co2 stored",
        bus=spatial.co2.nodes,
    )

    if co2_network:

        logger.info("Adding CO2 network.")
        co2_links = create_network_topology(n, "CO2 pipeline ")

        cost_onshore = (
            (1 - co2_links.underwater_fraction)
            * costs.at["CO2 pipeline", "fixed"]
            * co2_links.length
        )
        cost_submarine = (
            co2_links.underwater_fraction
            * costs.at["CO2 submarine pipeline", "fixed"]
            * co2_links.length
        )
        capital_cost = cost_onshore + cost_submarine

        n.madd(
            "Link",
            co2_links.index,
            bus0=co2_links.bus0.values + " co2 stored",
            bus1=co2_links.bus1.values + " co2 stored",
            p_min_pu=-1,
            p_nom_extendable=True,
            length=co2_links.length.values,
            capital_cost=capital_cost.values,
            carrier="CO2 pipeline",
            lifetime=costs.at["CO2 pipeline", "lifetime"],
        )


def load_existing_battery_capacities(nodes):
    """
    Load existing battery capacities from CSV file and distribute to network nodes.
    
    Parameters
    ----------
    nodes : pd.Index
        Network nodes
    
    Returns
    -------
    dict
        Dictionary with 'e_nom' (energy capacity in MWh) and 'p_nom' (power capacity in MW) as pd.Series
    """
    battery_csv_path = snakemake.input.get("battery_capacities", None)

    try:
        battery_df = pd.read_csv(battery_csv_path, index_col=0)
    except FileNotFoundError:
        logger.warning(f"Battery capacity file not found at {battery_csv_path}. Setting all capacities to 0.")
        return {
            'e_nom': pd.Series(0.0, index=nodes),
            'p_nom': pd.Series(0.0, index=nodes)
        }
    
    # Initialize capacity series with zeros
    e_nom = pd.Series(0.0, index=nodes)
    p_nom = pd.Series(0.0, index=nodes)
    
    # Extract country codes from node names (format: "XX 0" where XX is 2-digit country code)
    node_countries = pd.Series({node: node.split()[0][:2] for node in nodes})
    
    # Count nodes per country
    nodes_per_country = node_countries.value_counts()
    
    # Distribute battery capacity equally among nodes in each country
    for country_code, node_count in nodes_per_country.items():
        if country_code in battery_df.index:
            # Get capacity for this country and divide equally among its nodes
            country_e_nom = battery_df.loc[country_code, 'capa_2020'] / node_count  
            country_p_nom = battery_df.loc[country_code, 'power_2020_MW'] / node_count 
            
            # Assign to all nodes in this country
            country_nodes = node_countries[node_countries == country_code].index
            e_nom.loc[country_nodes] = country_e_nom
            p_nom.loc[country_nodes] = country_p_nom
            
            logger.info(f"Added {battery_df.loc[country_code, 'capa_2020']:.1f} MWh (store) / {battery_df.loc[country_code, 'power_2020_MW']:.1f} MW (power) battery capacity for {country_code} ({node_count} nodes)")
        else:
            logger.debug(f"No battery data for country {country_code}")
    
    total_e = e_nom.sum()
    total_p = p_nom.sum()
    logger.info(f"Total existing battery capacity: {total_e:.1f} MWh energy, {total_p:.1f} MW power across {len(nodes)} nodes")
    
    return {'e_nom': e_nom, 'p_nom': p_nom}


def add_aviation(n, cost, gadm_clustering=False):
    # Load data required for aviation and navigation
    # TODO follow the same structure as land transport and heat

    all_aviation = ["total international aviation", "total domestic aviation"]

    aviation_ctry_TWh = energy_totals.loc[:, all_aviation].sum(axis=1)

    airports = pd.read_csv(snakemake.input.airports, keep_default_na=False)
    airports = airports[airports.country.isin(countries)]

    gadm_layer_id = snakemake.params.gadm_layer_id

    if gadm_clustering:
        airports = locate_bus_alt_clust(
            airports,
            countries,
            gadm_layer_id,
            snakemake.input.shapes_path,
            snakemake.params.alternative_clustering,
        ).set_index(f"gadm_{gadm_layer_id}")
    else:
        airports = locate_bus(
            airports,
            countries,
            gadm_layer_id,
            snakemake.input.shapes_path,
            snakemake.params.alternative_clustering,
        ).set_index(f"gadm_{gadm_layer_id}")

    airports["fraction_ctry"] = airports.groupby("country")["fraction"].transform(
        lambda s: s / s.sum() if s.sum() > 0 else 0.0
    )

    W = float(n.snapshot_weightings.generators.sum())  # total weighted hours
    SCALE = 1e6  # TWh -> MWh

    airports["country_TWh"] = airports["country"].map(aviation_ctry_TWh).fillna(0.0)
    airports["p_set"] = (airports["fraction_ctry"] * airports["country_TWh"] * SCALE / W).fillna(0.0)

    # Aggregate airports by index to handle potential duplicates
    airports_aggregated = airports.groupby(airports.index)["p_set"].sum().to_frame()

    ind = pd.DataFrame(index=n.buses.index[n.buses.carrier == "AC"])
    # keep only p_set for grouping; concat ensures all AC buses exist
    to_group = pd.concat([airports_aggregated[["p_set"]], ind], axis=1).fillna(0.0)
    p_set_by_bus = to_group.groupby(to_group.index)["p_set"].sum()

    n.madd(
        "Load",
        spatial.nodes,
        suffix=" kerosene for aviation",
        bus=spatial.oil.nodes,
        carrier="kerosene for aviation",
        p_set=p_set_by_bus,
    )

    bus_countries = n.buses.loc[n.buses.carrier == "AC", "country"]
    bus_energy_MWh = (p_set_by_bus * W).reindex(bus_countries.index).fillna(0.0)
    energy_by_country_MWh = bus_energy_MWh.groupby(bus_countries).sum()

    if snakemake.params.sector_options["international_bunkers"]:
        # Count all aviation (domestic + international)
        co2_MWh = float(energy_by_country_MWh.sum())
    else:
        # Apply domestic share per country
        domestic_to_total = (
            energy_totals["total domestic aviation"] /
            (energy_totals["total international aviation"] + energy_totals["total domestic aviation"])
        ).fillna(0.0)
        co2_MWh = float((energy_by_country_MWh * domestic_to_total.reindex(energy_by_country_MWh.index).fillna(0.0)).sum())

    # Use helper function for proper temporal weighting
    add_emissions_from_weighted_energy(
        n,
        "aviation oil emissions",
        p_set_by_bus,
        costs.at["oil", "CO2 intensity"],
        "co2 atmosphere",
        "oil emissions",
        flat=True
    )


def add_storage(n, costs):
    "function to add the different types of storage systems"
    logger.info("Add battery storage")

    n.add("Carrier", "battery")

    n.madd(
        "Bus",
        spatial.nodes + " battery",
        location=spatial.nodes,
        carrier="battery",
        x=n.buses.loc[list(spatial.nodes)].x.values,
        y=n.buses.loc[list(spatial.nodes)].y.values,
    )

    # Load existing battery capacities from CSV
    battery_data = load_existing_battery_capacities(spatial.nodes)
    
    n.madd(
        "Store",
        spatial.nodes + " battery",
        bus=spatial.nodes + " battery",
        e_cyclic=True,
        e_nom_extendable=True,
        e_nom=battery_data['e_nom'].values,
        e_nom_min=battery_data['e_nom'].values,
        carrier="battery",
        capital_cost=costs.at["battery storage", "fixed"],
        lifetime=costs.at["battery storage", "lifetime"],
        build_year=0,  # Will be set to baseyear in add_existing_baseyear.py
    )

    n.madd(
        "Link",
        spatial.nodes + " battery charger",
        bus0=spatial.nodes,
        bus1=spatial.nodes + " battery",
        carrier="battery charger",
        efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
        capital_cost=costs.at["battery inverter", "fixed"],
        p_nom_extendable=True,
        p_nom=battery_data['p_nom'].values,
        p_nom_min=battery_data['p_nom'].values,
        lifetime=costs.at["battery inverter", "lifetime"],
        build_year=0,  # Will be set to baseyear in add_existing_baseyear.py
    )

    n.madd(
        "Link",
        spatial.nodes + " battery discharger",
        bus0=spatial.nodes + " battery",
        bus1=spatial.nodes,
        carrier="battery discharger",
        efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
        marginal_cost=options["marginal_cost_storage"],
        p_nom_extendable=True,
        p_nom=battery_data['p_nom'].values,
        p_nom_min=battery_data['p_nom'].values,
        lifetime=costs.at["battery inverter", "lifetime"],
        build_year=0,  # Will be set to baseyear in add_existing_baseyear.py
    )


def h2_hc_conversions(n, costs):
    "function to add the conversion technologies between H2 and hydrocarbons"
    logger.info("Add H2 to hydrocarbon conversions")

    if options["methanation"]:
        n.madd(
            "Link",
            spatial.nodes,
            suffix=" Sabatier",
            bus0=spatial.nodes + " H2",
            bus1=spatial.gas.nodes,
            bus2=spatial.co2.nodes,
            p_nom_extendable=True,
            carrier="Sabatier",
            efficiency=costs.at["methanation", "efficiency"],
            efficiency2=-costs.at["methanation", "efficiency"]
            * costs.at["gas", "CO2 intensity"],
            # costs given per kW_gas
            capital_cost=costs.at["methanation", "fixed"]
            * costs.at["methanation", "efficiency"],
            lifetime=costs.at["methanation", "lifetime"],
        )

    if options["helmeth"]:
        n.madd(
            "Link",
            spatial.nodes,
            suffix=" helmeth",
            bus0=spatial.nodes,
            bus1=spatial.gas.nodes,
            bus2=spatial.co2.nodes,
            carrier="helmeth",
            p_nom_extendable=True,
            efficiency=costs.at["helmeth", "efficiency"],
            efficiency2=-costs.at["helmeth", "efficiency"]
            * costs.at["gas", "CO2 intensity"],
            capital_cost=costs.at["helmeth", "fixed"],
            lifetime=costs.at["helmeth", "lifetime"],
        )


def add_shipping(n, costs, gadm_clustering=False, fuel_price_dict=None):
    ports = pd.read_csv(
        snakemake.input.ports, index_col=None, keep_default_na=False
    ).squeeze()
    ports = ports[ports.country.isin(countries)]

    gadm_layer_id = snakemake.params.gadm_layer_id

    all_navigation = ["total international navigation", "total domestic navigation"]

    navigation_demand = energy_totals.loc[countries, all_navigation].sum(axis=1)

    efficiency = (
        options["shipping_average_efficiency"] / costs.at["fuel cell", "efficiency"]
    )

    # check whether item depends on investment year
    shipping_hydrogen_share = get(
        options["shipping_hydrogen_share"], demand_sc + "_" + str(investment_year)
    )

    ports = locate_bus(
        ports,
        countries,
        gadm_layer_id,
        snakemake.input.shapes_path,
        snakemake.params.alternative_clustering,
    ).set_index("gadm_{}".format(gadm_layer_id))

    ind = pd.DataFrame(n.buses.index[n.buses.carrier == "AC"])
    ind = ind.set_index(n.buses.index[n.buses.carrier == "AC"])

    ports["p_set"] = (
        shipping_hydrogen_share
        * ports["fraction"]
        * ports["country"].map(navigation_demand)
        * efficiency
        * 1e6
        / 8760
        # TODO double check the use of efficiency
    )  # TODO use real data here

    ports = pd.concat([ports, ind]).drop("Bus", axis=1)

    # ports = ports.fillna(0.0)
    ports = ports.groupby(ports.index).sum()

    if options["shipping_hydrogen_liquefaction"]:
        n.madd(
            "Bus",
            spatial.nodes,
            suffix=" H2 liquid",
            carrier="H2 liquid",
            location=spatial.nodes,
        )

        # link the H2 supply to liquified H2
        n.madd(
            "Link",
            spatial.nodes + " H2 liquefaction",
            bus0=spatial.nodes + " H2",
            bus1=spatial.nodes + " H2 liquid",
            carrier="H2 liquefaction",
            efficiency=costs.at["H2 liquefaction", "efficiency"],
            capital_cost=costs.at["H2 liquefaction", "fixed"],
            p_nom_extendable=True,
            lifetime=costs.at["H2 liquefaction", "lifetime"],
        )

        shipping_bus = spatial.nodes + " H2 liquid"
    else:
        shipping_bus = spatial.nodes + " H2"

    if not (
        snakemake.params.h2_policy["is_reference"]
        and snakemake.params.h2_policy["remove_h2_load"]
    ):
        n.madd(
            "Load",
            spatial.nodes,
            suffix=" H2 for shipping",
            bus=shipping_bus,
            carrier="H2 for shipping",
            p_set=ports["p_set"],
        )

    if shipping_hydrogen_share < 1:
        shipping_oil_share = 1 - shipping_hydrogen_share

        ports["p_set"] = (
            shipping_oil_share
            * ports["fraction"]
            * ports["country"].map(navigation_demand)
            * 1e6
            / 8760
        )

        n.madd(
            "Load",
            spatial.nodes,
            suffix=" shipping oil",
            bus=spatial.oil.nodes,
            carrier="shipping oil",
            p_set=ports["p_set"],
        )

        if snakemake.params.sector_options["international_bunkers"]:
            co2 = ports["p_set"].sum() * costs.at["oil", "CO2 intensity"]
        else:
            domestic_to_total = energy_totals["total domestic navigation"] / (
                energy_totals["total domestic navigation"]
                + energy_totals["total international navigation"]
            )

            co2 = (
                ports["p_set"].sum()
                * domestic_to_total
                * costs.at["oil", "CO2 intensity"]
            ).sum()

        n.add(
            "Load",
            "shipping oil emissions",
            bus="co2 atmosphere",
            carrier="shipping oil emissions",
            p_set=-co2,
        )

    if "oil" not in n.buses.carrier.unique():
        n.madd("Bus", spatial.oil.nodes, location=spatial.oil.locations, carrier="oil")
    if "oil" not in n.stores.carrier.unique():
        # could correct to e.g. 0.001 EUR/kWh * annuity and O&M
        n.madd(
            "Store",
            [oil_bus + " Store" for oil_bus in spatial.oil.nodes],
            bus=spatial.oil.nodes,
            e_nom_extendable=True,
            e_cyclic=True,
            carrier="oil",
        )

    if "oil" not in n.generators.carrier.unique():
        # Use country-specific fuel prices for oil
        if fuel_price_dict is not None and 'oil' in fuel_price_dict:
            oil_marginal_cost = get_fuel_price_by_node(spatial.oil.nodes, 'oil', costs, fuel_price_dict)
        else:
            oil_marginal_cost = costs.at["oil", "fuel"]
        
        n.madd(
            "Generator",
            spatial.oil.nodes,
            bus=spatial.oil.nodes,
            p_nom_extendable=True,
            carrier="oil",
            marginal_cost=oil_marginal_cost,
        )


def add_industry(n, costs):
    logger.info("adding industrial demand")
    # 1e6 to convert TWh to MWh

    # Get total weighted hours (replaces hardcoded 8760)
    W = n.snapshot_weightings.generators.sum()

    # industrial_demand.reset_index(inplace=True)

    # Add carrier Biomass

    n.madd(
        "Bus",
        spatial.biomass.industry,
        location=spatial.biomass.locations,
        carrier="solid biomass for industry",
    )

    if options["biomass_transport"]:
        p_set = (
            industrial_demand.loc[spatial.biomass.locations, "solid biomass"].rename(
                index=lambda x: x + " solid biomass for industry"
            )
            / W
        )
    else:
        p_set = industrial_demand["solid biomass"].sum() / W

    n.madd(
        "Load",
        spatial.biomass.industry,
        bus=spatial.biomass.industry,
        carrier="solid biomass for industry",
        p_set=p_set,
    )

    n.madd(
        "Link",
        spatial.biomass.industry,
        bus0=broadcast_madd_value(
            (
                spatial.biomass.industry_resource
                if biomass_allocation is not None
                else spatial.biomass.nodes
            ),
            spatial.biomass.industry,
            "solid biomass for industry bus0",
        ),
        bus1=spatial.biomass.industry,
        carrier="solid biomass for industry",
        p_nom_extendable=True,
        efficiency=1.0,
    )
    if snakemake.params.sector_options["cc"]:
        biomass_locations = pd.Index(spatial.biomass.locations)
        if biomass_locations.isin(spatial.co2.df.index).all():
            biomass_cc_bus3 = spatial.co2.df.loc[biomass_locations, "nodes"].values
        else:
            biomass_cc_bus3 = np.repeat(spatial.co2.nodes[0], len(spatial.biomass.industry_cc))

        n.madd(
            "Link",
            spatial.biomass.industry_cc,
            bus0=broadcast_madd_value(
                (
                    spatial.biomass.industry_resource
                    if biomass_allocation is not None
                    else spatial.biomass.nodes
                ),
                spatial.biomass.industry_cc,
                "solid biomass for industry CC bus0",
            ),
            bus1=spatial.biomass.industry,
            bus2="co2 atmosphere",
            bus3=biomass_cc_bus3,
            carrier="solid biomass for industry CC",
            p_nom_extendable=True,
            capital_cost=costs.at["cement capture", "fixed"]
            * costs.at["solid biomass", "CO2 intensity"],
            efficiency=0.9,  # TODO: make config option
            efficiency2=-costs.at["solid biomass", "CO2 intensity"]
            * costs.at["cement capture", "capture_rate"],
            efficiency3=costs.at["solid biomass", "CO2 intensity"]
            * costs.at["cement capture", "capture_rate"],
            lifetime=costs.at["cement capture", "lifetime"],
        )

    # CARRIER = FOSSIL GAS

    # nodes = pop_layout.index

    # industrial_demand['TWh/a (MtCO2/a)'] = industrial_demand['TWh/a (MtCO2/a)'].apply(
    #     lambda cocode: two_2_three_digits_country(cocode[:2]) + "." + cocode[3:])

    # industrial_demand.set_index("TWh/a (MtCO2/a)", inplace=True)

    # n.add("Bus", "gas for industry", location="Earth", carrier="gas for industry")
    n.madd(
        "Bus",
        spatial.gas.industry,
        location=spatial.gas.locations,
        carrier="gas for industry",
    )

    gas_demand = industrial_demand.loc[spatial.nodes, "gas"] / W

    if options["gas"]["spatial_gas"]:
        spatial_gas_demand = gas_demand.rename(index=lambda x: x + " gas for industry")
    else:
        spatial_gas_demand = gas_demand.sum()

    n.madd(
        "Load",
        spatial.gas.industry,
        bus=spatial.gas.industry,
        carrier="gas for industry",
        p_set=spatial_gas_demand,
    )

    n.madd(
        "Link",
        spatial.gas.industry,
        # bus0="Earth gas",
        bus0=spatial.gas.nodes,
        # bus1="gas for industry",
        bus1=spatial.gas.industry,
        bus2="co2 atmosphere",
        carrier="gas for industry",
        p_nom_extendable=True,
        efficiency=1.0,
        efficiency2=costs.at["gas", "CO2 intensity"],
    )
    if snakemake.params.sector_options["cc"]:
        n.madd(
            "Link",
            spatial.gas.industry_cc,
            # suffix=" gas for industry CC",
            # bus0="Earth gas",
            bus0=spatial.gas.nodes,
            bus1=spatial.gas.industry,
            bus2="co2 atmosphere",
            bus3=spatial.co2.nodes,
            carrier="gas for industry CC",
            p_nom_extendable=True,
            capital_cost=costs.at["cement capture", "fixed"]
            * costs.at["gas", "CO2 intensity"],
            efficiency=0.9,
            efficiency2=costs.at["gas", "CO2 intensity"]
            * (1 - costs.at["cement capture", "capture_rate"]),
            efficiency3=costs.at["gas", "CO2 intensity"]
            * costs.at["cement capture", "capture_rate"],
            lifetime=costs.at["cement capture", "lifetime"],
        )

    # CARRIER = COAL
    n.madd(
        "Bus",
        spatial.coal.industry,
        location=spatial.coal.locations,
        carrier="coal for industry",
    )

    coal_demand = industrial_demand.loc[spatial.nodes, "coal"] / W
    if options["coal"]["spatial_coal"]:
        spatial_coal_demand = coal_demand.rename(index=lambda x: x + " coal for industry")
    else:
        spatial_coal_demand = coal_demand.sum()

    n.madd(
        "Load",
        spatial.coal.industry,
        bus=spatial.coal.industry,
        carrier="coal for industry",
        p_set=spatial_coal_demand,
    )

    n.madd(
        "Link",
        spatial.coal.industry,
        bus0=spatial.coal.nodes,
        bus1=spatial.coal.industry,
        bus2="co2 atmosphere",
        carrier="coal for industry",
        p_nom_extendable=True,
        efficiency=1.0,
        efficiency2=costs.at["coal", "CO2 intensity"],
    )

    #################################################### CARRIER = HYDROGEN

    if not (
        snakemake.params.h2_policy["is_reference"]
        and snakemake.params.h2_policy["remove_h2_load"]
    ):
        n.madd(
            "Load",
            nodes,
            suffix=" H2 for industry",
            bus=nodes + " H2",
            carrier="H2 for industry",
            p_set=industrial_demand["hydrogen"].apply(lambda frac: frac / W),
        )

    # CARRIER = LIQUID HYDROCARBONS
    n.madd(
        "Load",
        spatial.nodes,
        suffix=" naphtha for industry",
        bus=spatial.oil.nodes,
        carrier="naphtha for industry",
        p_set=industrial_demand["oil"] / W,
    )

    #     #NB: CO2 gets released again to atmosphere when plastics decay or kerosene is burned
    #     #except for the process emissions when naphtha is used for petrochemicals, which can be captured with other industry process emissions
    #     #tco2 per hour
    # TODO kerosene for aviation should be added too but in the right func.
    co2_release = [" naphtha for industry"]
    # check land transport

    # Get p_set for industry oil (already a scalar per node, not time-varying)
    p_set_industry_oil = industrial_demand["oil"] / W
    
    # Use helper function for proper temporal weighting
    add_emissions_from_weighted_energy(
        n,
        "industry oil emissions",
        p_set_industry_oil,
        costs.at["oil", "CO2 intensity"],
        "co2 atmosphere",
        "industry oil emissions",
        flat=True
    )

    ########################################################### CARRIER = HEAT
    # TODO simplify bus expression
    p_set_ind_heat = industrial_demand.loc[spatial.nodes, "low-temperature heat"] / W
    
    n.madd(
        "Load",
        spatial.nodes,
        suffix=" low-temperature heat for industry",
        bus=[
            (
                node + " urban central heat"
                if node + " urban central heat" in n.buses.index
                else node + " services urban decentral heat"
            )
            for node in spatial.nodes
        ],
        carrier="low-temperature heat for industry",
        p_set=p_set_ind_heat,
    )

    ################################################## CARRIER = ELECTRICITY

    #     # remove today's industrial electricity demand by scaling down total electricity demand
    for ct in n.buses.country.dropna().unique():
        # TODO map onto n.bus.country
        # TODO make sure to check this one, should AC have carrier pf "electricity"?
        loads_i = n.loads.index[
            (n.loads.index.str[:2] == ct) & (n.loads.carrier == "AC")
        ]
        if n.loads_t.p_set.columns.intersection(loads_i).empty:
            continue

    # if not snakemake.config["custom_data"]["elec_demand"]:
    #     # if electricity demand is provided by pypsa-earth, the electricity used
    #     # in industry is included, and need to be removed from the default elec
    #     # demand here, and added as "industry electricity"
    #     factor = (
    #         1
    #         - industrial_demand.loc[loads_i, "current electricity"].sum()
    #         / n.loads_t.p_set[loads_i].sum().sum()
    #     )
    #     n.loads_t.p_set[loads_i] *= factor
    #     industrial_elec = industrial_demand["current electricity"].apply(
    #         lambda frac: frac / 8760
    #     )

    # else:
    industrial_elec = industrial_demand["electricity"] / W # converting to TWh per hour

    n.madd(
        "Load",
        spatial.nodes,
        suffix=" industry electricity",
        bus=spatial.nodes,
        carrier="industry electricity",
        p_set=industrial_elec,
    )

    n.add("Bus", "process emissions", location="Earth", carrier="process emissions")

    # this should be process emissions fossil+feedstock
    # then need load on atmosphere for feedstock emissions that are currently going to atmosphere via Link Fischer-Tropsch demand
    n.madd(
        "Load",
        spatial.nodes,
        suffix=" process emissions",
        bus="process emissions",
        carrier="process emissions",
        p_set=-(
            #    industrial_demand["process emission from feedstock"]+
            industrial_demand["process emissions"]
        )
        / W,
    )

    n.add(
        "Link",
        "process emissions",
        bus0="process emissions",
        bus1="co2 atmosphere",
        carrier="process emissions",
        p_nom_extendable=True,
        efficiency=1.0,
    )

    # assume enough local waste heat for CC
    if snakemake.params.sector_options["cc"]:
        n.madd(
            "Link",
            spatial.co2.locations,
            suffix=" process emissions CC",
            bus0="process emissions",
            bus1="co2 atmosphere",
            bus2=spatial.co2.nodes,
            carrier="process emissions CC",
            p_nom_extendable=True,
            capital_cost=costs.at["cement capture", "fixed"],
            efficiency=1 - costs.at["cement capture", "capture_rate"],
            efficiency2=costs.at["cement capture", "capture_rate"],
            lifetime=costs.at["cement capture", "lifetime"],
        )


def get(item, investment_year=None):
    """
    Check whether item depends on investment year.
    """
    if isinstance(item, dict):
        return item[investment_year]
    else:
        return item


"""
Missing data:
 - transport
 - aviation data
 - nodal_transport_data
 - cycling_shift
 - dsm_profile
 - avail_profile
"""


def add_land_transport(n, costs):
    """
    Function to add land transport to network.
    """
    # TODO options?

    logger.info("adding land transport")

    if options["dynamic_transport"]["enable"] == False:
        fuel_cell_share = get(
            options["land_transport_fuel_cell_share"],
            demand_sc + "_" + str(investment_year),
        )
        electric_share = get(
            options["land_transport_electric_share"],
            demand_sc + "_" + str(investment_year),
        )

    elif options["dynamic_transport"]["enable"] == True:
        fuel_cell_share = options["dynamic_transport"][
            "land_transport_fuel_cell_share"
        ][snakemake.wildcards.opts]
        electric_share = options["dynamic_transport"]["land_transport_electric_share"][
            snakemake.wildcards.opts
        ]

    ice_share = 1 - fuel_cell_share - electric_share

    logger.info("FCEV share: {}".format(fuel_cell_share))
    logger.info("EV share: {}".format(electric_share))
    logger.info("ICEV share: {}".format(ice_share))

    assert ice_share >= 0, "Error, more FCEV and EV share than 1."

    # Nodes are already defined, remove it from here
    # nodes = pop_layout.index

    if electric_share > 0:
        n.add("Carrier", "Li ion")

        n.madd(
            "Bus",
            spatial.nodes,
            location=spatial.nodes,
            suffix=" EV battery",
            carrier="Li ion",
            x=n.buses.loc[list(spatial.nodes)].x.values,
            y=n.buses.loc[list(spatial.nodes)].y.values,
        )

        p_set_base = (
            electric_share
            * (
                transport[spatial.nodes]
                + cycling_shift(transport[spatial.nodes], 1)
                + cycling_shift(transport[spatial.nodes], 2)
            )
            / 3
        )
        p_set = p_set_base

        n.madd(
            "Load",
            spatial.nodes,
            suffix=" land transport EV",
            bus=spatial.nodes + " EV battery",
            carrier="land transport EV",
            p_set=p_set,
        )

        p_nom = (
            nodal_transport_data["number cars"]
            * options.get("bev_charge_rate", 0.011)
            * electric_share
        )

        n.madd(
            "Link",
            spatial.nodes,
            suffix=" BEV charger",
            bus0=spatial.nodes,
            bus1=spatial.nodes + " EV battery",
            p_nom=p_nom,
            carrier="BEV charger",
            p_max_pu=avail_profile[spatial.nodes],
            efficiency=options.get("bev_charge_efficiency", 0.9),
            # These were set non-zero to find LU infeasibility when availability = 0.25
            # p_nom_extendable=True,
            # p_nom_min=p_nom,
            # capital_cost=1e6,  #i.e. so high it only gets built where necessary
        )

    if electric_share > 0 and options["v2g"]:
        n.madd(
            "Link",
            spatial.nodes,
            suffix=" V2G",
            bus1=spatial.nodes,
            bus0=spatial.nodes + " EV battery",
            p_nom=p_nom,
            carrier="V2G",
            p_max_pu=avail_profile[spatial.nodes],
            efficiency=options.get("bev_charge_efficiency", 0.9),
        )

    if electric_share > 0 and options["bev_dsm"]:
        e_nom = (
            nodal_transport_data["number cars"]
            * options.get("bev_energy", 0.05)
            * options["bev_availability"]
            * electric_share
        )

        n.madd(
            "Store",
            spatial.nodes,
            suffix=" battery storage",
            bus=spatial.nodes + " EV battery",
            carrier="battery storage",
            e_cyclic=True,
            e_nom=e_nom,
            e_max_pu=1,
            e_min_pu=dsm_profile[spatial.nodes],
        )

    if fuel_cell_share > 0:
        if not (
            snakemake.params.h2_policy["is_reference"]
            and snakemake.params.h2_policy["remove_h2_load"]
        ):
            p_set_fc_base = (
                fuel_cell_share
                / options["transport_fuel_cell_efficiency"]
                * transport[spatial.nodes]
            )
            
            n.madd(
                "Load",
                spatial.nodes,
                suffix=" land transport fuel cell",
                bus=spatial.nodes + " H2",
                carrier="land transport fuel cell",
                p_set=p_set_fc_base,
            )

    if ice_share > 0:
        if "oil" not in n.buses.carrier.unique():
            n.madd(
                "Bus", spatial.oil.nodes, location=spatial.oil.locations, carrier="oil"
            )
        ice_efficiency = options["transport_internal_combustion_efficiency"]

        n.madd(
            "Load",
            spatial.nodes,
            suffix=" land transport oil",
            bus=spatial.oil.nodes,
            carrier="land transport oil",
            p_set=ice_share / ice_efficiency * transport[spatial.nodes],
        )

        # Get p_set for land transport oil (already time-varying DataFrame)
        p_set_transport_oil = ice_share / ice_efficiency * transport[spatial.nodes]
        
        # Use helper function for proper temporal weighting
        add_emissions_from_weighted_energy(
            n,
            "land transport oil emissions",
            p_set_transport_oil,
            costs.at["oil", "CO2 intensity"],
            "co2 atmosphere",
            "land transport oil emissions",
            flat=True
        )


def create_nodes_for_heat_sector():
    # TODO pop_layout

    # rural are areas with low heating density and individual heating
    # urban are areas with high heating density
    # urban can be split into district heating (central) and individual heating (decentral)

    ct_urban = pop_layout.urban.groupby(pop_layout.ct).sum()
    # distribution of urban population within a country
    pop_layout["urban_ct_fraction"] = pop_layout.urban / pop_layout.ct.map(ct_urban.get)

    sectors = ["residential", "services"]

    h_nodes = {}
    urban_fraction = pop_layout.urban / pop_layout[["rural", "urban"]].sum(axis=1)

    for sector in sectors:
        h_nodes[sector + " rural"] = pop_layout.index
        h_nodes[sector + " urban decentral"] = pop_layout.index

    # maximum potential of urban demand covered by district heating
    central_fraction = options["district_heating"]["potential"]
    # district heating share at each node
    dist_fraction_node = (
        district_heat_share["district heat share"]
        * pop_layout["urban_ct_fraction"]
        / pop_layout["fraction"]
    )
    h_nodes["urban central"] = dist_fraction_node.index
    # if district heating share larger than urban fraction -> set urban
    # fraction to district heating share
    urban_fraction = pd.concat([urban_fraction, dist_fraction_node], axis=1).max(axis=1)
    # difference of max potential and today's share of district heating
    diff = (urban_fraction * central_fraction) - dist_fraction_node
    progress = get(options["district_heating"]["progress"], investment_year)
    dist_fraction_node += diff * progress
    # logger.info(
    #     "The current district heating share compared to the maximum",
    #     f"possible is increased by a progress factor of\n{progress}",
    #     "resulting in a district heating share of",  # "\n{dist_fraction_node}", #TODO fix district heat share
    # )

    return h_nodes, dist_fraction_node, urban_fraction


def add_heat(n, costs):
    # TODO options?
    # TODO pop_layout?

    logger.info("adding heat")

    sectors = ["residential", "services"]

    h_nodes, dist_fraction, urban_fraction = create_nodes_for_heat_sector()

    # NB: must add costs of central heating afterwards (EUR 400 / kWpeak, 50a, 1% FOM from Fraunhofer ISE)

    # exogenously reduce space heat demand
    if options["reduce_space_heat_exogenously"]:
        dE = get(options["reduce_space_heat_exogenously_factor"], investment_year)
        # print(f"assumed space heat reduction of {dE*100} %")
        for sector in sectors:
            heat_demand[sector + " space"] = (1 - dE) * heat_demand[sector + " space"]

    heat_systems = [
        "residential rural",
        "services rural",
        "residential urban decentral",
        "services urban decentral",
        "urban central",
    ]

    for name in heat_systems:
        name_type = "central" if name == "urban central" else "decentral"

        n.add("Carrier", name + " heat")

        n.madd(
            "Bus",
            h_nodes[name] + " {} heat".format(name),
            location=h_nodes[name],
            carrier=name + " heat",
        )

        ## Add heat load

        for sector in sectors:
            # heat demand weighting
            if "rural" in name:
                factor = 1 - urban_fraction[h_nodes[name]]
            elif "urban central" in name:
                factor = dist_fraction[h_nodes[name]]
            elif "urban decentral" in name:
                factor = urban_fraction[h_nodes[name]] - dist_fraction[h_nodes[name]]
            else:
                raise NotImplementedError(
                    f" {name} not in " f"heat systems: {heat_systems}"
                )

            if sector in name:
                heat_load = (
                    heat_demand[[sector + " water", sector + " space"]]
                    .groupby(level=1, axis=1)
                    .sum()[h_nodes[name]]
                    .multiply(factor)
                )

        if name == "urban central":
            heat_load = (
                heat_demand.groupby(level=1, axis=1)
                .sum()[h_nodes[name]]
                .multiply(
                    factor * (1 + options["district_heating"]["district_heating_loss"])
                )
            )

        n.madd(
            "Load",
            h_nodes[name],
            suffix=f" {name} heat",
            bus=h_nodes[name] + f" {name} heat",
            carrier=name + " heat",
            p_set=heat_load,
        )

        ## Add heat pumps

        heat_pump_type = "air" if "urban" in name else "ground"

        costs_name = f"{name_type} {heat_pump_type}-sourced heat pump"
        cop = {"air": ashp_cop, "ground": gshp_cop}
        efficiency = (
            cop[heat_pump_type][h_nodes[name]]
            if options["time_dep_hp_cop"]
            else costs.at[costs_name, "efficiency"]
        )

        n.madd(
            "Link",
            h_nodes[name],
            suffix=f" {name} {heat_pump_type} heat pump",
            bus0=h_nodes[name],
            bus1=h_nodes[name] + f" {name} heat",
            carrier=f"{name} {heat_pump_type} heat pump",
            efficiency=efficiency,
            capital_cost=costs.at[costs_name, "efficiency"]
            * costs.at[costs_name, "fixed"],
            p_nom_extendable=True,
            lifetime=costs.at[costs_name, "lifetime"],
        )

        if options["tes"]:
            n.add("Carrier", name + " water tanks")

            n.madd(
                "Bus",
                h_nodes[name] + f" {name} water tanks",
                location=h_nodes[name],
                carrier=name + " water tanks",
            )

            n.madd(
                "Link",
                h_nodes[name] + f" {name} water tanks charger",
                bus0=h_nodes[name] + f" {name} heat",
                bus1=h_nodes[name] + f" {name} water tanks",
                efficiency=costs.at["water tank charger", "efficiency"],
                carrier=name + " water tanks charger",
                p_nom_extendable=True,
            )

            n.madd(
                "Link",
                h_nodes[name] + f" {name} water tanks discharger",
                bus0=h_nodes[name] + f" {name} water tanks",
                bus1=h_nodes[name] + f" {name} heat",
                carrier=name + " water tanks discharger",
                efficiency=costs.at["water tank discharger", "efficiency"],
                p_nom_extendable=True,
            )

            if isinstance(options["tes_tau"], dict):
                tes_time_constant_days = options["tes_tau"][name_type]
            else:  # TODO add logger
                # logger.warning("Deprecated: a future version will require you to specify 'tes_tau' ",
                # "for 'decentral' and 'central' separately.")
                tes_time_constant_days = (
                    options["tes_tau"] if name_type == "decentral" else 180.0
                )

            # conversion from EUR/m^3 to EUR/MWh for 40 K diff and 1.17 kWh/m^3/K
            capital_cost = (
                costs.at[name_type + " water tank storage", "fixed"] / 0.00117 / 40
            )

            n.madd(
                "Store",
                h_nodes[name] + f" {name} water tanks",
                bus=h_nodes[name] + f" {name} water tanks",
                e_cyclic=True,
                e_nom_extendable=True,
                carrier=name + " water tanks",
                standing_loss=1 - np.exp(-1 / 24 / tes_time_constant_days),
                capital_cost=capital_cost,
                lifetime=costs.at[name_type + " water tank storage", "lifetime"],
            )

        if options["boilers"]:
            key = f"{name_type} resistive heater"

            n.madd(
                "Link",
                h_nodes[name] + f" {name} resistive heater",
                bus0=h_nodes[name],
                bus1=h_nodes[name] + f" {name} heat",
                carrier=name + " resistive heater",
                efficiency=costs.at[key, "efficiency"],
                capital_cost=costs.at[key, "efficiency"] * costs.at[key, "fixed"],
                p_nom_extendable=True,
                lifetime=costs.at[key, "lifetime"],
            )

            key = f"{name_type} gas boiler"

            n.madd(
                "Link",
                h_nodes[name] + f" {name} gas boiler",
                p_nom_extendable=True,
                bus0=spatial.gas.nodes,
                bus1=h_nodes[name] + f" {name} heat",
                bus2="co2 atmosphere",
                carrier=name + " gas boiler",
                efficiency=costs.at[key, "efficiency"],
                efficiency2=costs.at["gas", "CO2 intensity"],
                capital_cost=costs.at[key, "efficiency"] * costs.at[key, "fixed"],
                lifetime=costs.at[key, "lifetime"],
            )

        if options["solar_thermal"]:
            n.add("Carrier", name + " solar thermal")

            n.madd(
                "Generator",
                h_nodes[name],
                suffix=f" {name} solar thermal collector",
                bus=h_nodes[name] + f" {name} heat",
                carrier=name + " solar thermal",
                p_nom_extendable=True,
                capital_cost=costs.at[name_type + " solar thermal", "fixed"],
                p_max_pu=solar_thermal[h_nodes[name]],
                lifetime=costs.at[name_type + " solar thermal", "lifetime"],
            )

        if options["chp"] and name == "urban central":
            # add gas CHP; biomass CHP is added in biomass section
            n.madd(
                "Link",
                h_nodes[name] + " urban central gas CHP",
                bus0=spatial.gas.nodes,
                bus1=h_nodes[name],
                bus2=h_nodes[name] + " urban central heat",
                bus3="co2 atmosphere",
                carrier="urban central gas CHP",
                p_nom_extendable=True,
                capital_cost=costs.at["central gas CHP", "fixed"]
                * costs.at["central gas CHP", "efficiency"],
                marginal_cost=costs.at["central gas CHP", "VOM"],
                efficiency=costs.at["central gas CHP", "efficiency"],
                efficiency2=costs.at["central gas CHP", "efficiency"]
                / costs.at["central gas CHP", "c_b"],
                efficiency3=costs.at["gas", "CO2 intensity"],
                lifetime=costs.at["central gas CHP", "lifetime"],
            )
            if snakemake.params.sector_options["cc"]:
                n.madd(
                    "Link",
                    h_nodes[name] + " urban central gas CHP CC",
                    # bus0="Earth gas",
                    bus0=spatial.gas.nodes,
                    bus1=h_nodes[name],
                    bus2=h_nodes[name] + " urban central heat",
                    bus3="co2 atmosphere",
                    bus4=spatial.co2.df.loc[h_nodes[name], "nodes"].values,
                    carrier="urban central gas CHP CC",
                    p_nom_extendable=True,
                    capital_cost=costs.at["central gas CHP", "fixed"]
                    * costs.at["central gas CHP", "efficiency"]
                    + costs.at["biomass CHP capture", "fixed"]
                    * costs.at["gas", "CO2 intensity"],
                    marginal_cost=costs.at["central gas CHP", "VOM"],
                    efficiency=costs.at["central gas CHP", "efficiency"]
                    - costs.at["gas", "CO2 intensity"]
                    * (
                        costs.at["biomass CHP capture", "electricity-input"]
                        + costs.at[
                            "biomass CHP capture", "compression-electricity-input"
                        ]
                    ),
                    efficiency2=costs.at["central gas CHP", "efficiency"]
                    / costs.at["central gas CHP", "c_b"]
                    + costs.at["gas", "CO2 intensity"]
                    * (
                        costs.at["biomass CHP capture", "heat-output"]
                        + costs.at["biomass CHP capture", "compression-heat-output"]
                        - costs.at["biomass CHP capture", "heat-input"]
                    ),
                    efficiency3=costs.at["gas", "CO2 intensity"]
                    * (1 - costs.at["biomass CHP capture", "capture_rate"]),
                    efficiency4=costs.at["gas", "CO2 intensity"]
                    * costs.at["biomass CHP capture", "capture_rate"],
                    lifetime=costs.at["central gas CHP", "lifetime"],
                )

        if options["chp"] and options["micro_chp"] and name != "urban central":
            n.madd(
                "Link",
                h_nodes[name] + f" {name} micro gas CHP",
                p_nom_extendable=True,
                # bus0="Earth gas",
                bus0=spatial.gas.nodes,
                bus1=h_nodes[name],
                bus2=h_nodes[name] + f" {name} heat",
                bus3="co2 atmosphere",
                carrier=name + " micro gas CHP",
                efficiency=costs.at["micro CHP", "efficiency"],
                efficiency2=costs.at["micro CHP", "efficiency-heat"],
                efficiency3=costs.at["gas", "CO2 intensity"],
                capital_cost=costs.at["micro CHP", "fixed"],
                lifetime=costs.at["micro CHP", "lifetime"],
            )


def average_every_nhours(n, offset):
    # logger.info(f'Resampling the network to {offset}')
    m = n.copy(with_time=False)

    snapshot_weightings = n.snapshot_weightings.resample(offset.casefold()).sum()
    m.set_snapshots(snapshot_weightings.index)
    m.snapshot_weightings = snapshot_weightings

    for c in n.iterate_components():
        pnl = getattr(m, c.list_name + "_t")
        for k, df in c.pnl.items():
            if not df.empty:
                if c.list_name == "stores" and k == "e_max_pu":
                    pnl[k] = df.resample(offset.casefold()).min()
                elif c.list_name == "stores" and k == "e_min_pu":
                    pnl[k] = df.resample(offset.casefold()).max()
                else:
                    pnl[k] = df.resample(offset.casefold()).mean()

    return m


def add_dac(n, costs):
    heat_carriers = ["urban central heat", "services urban decentral heat"]
    heat_buses = n.buses.index[n.buses.carrier.isin(heat_carriers)]
    locations = n.buses.location[heat_buses]

    efficiency2 = -(
        costs.at["direct air capture", "electricity-input"]
        + costs.at["direct air capture", "compression-electricity-input"]
    )
    efficiency3 = -(
        costs.at["direct air capture", "heat-input"]
        - costs.at["direct air capture", "compression-heat-output"]
    )

    n.madd(
        "Link",
        heat_buses.str.replace(" heat", " DAC"),
        bus0="co2 atmosphere",
        bus1=spatial.co2.df.loc[locations, "nodes"].values,
        bus2=locations.values,
        bus3=heat_buses,
        carrier="DAC",
        capital_cost=costs.at["direct air capture", "fixed"],
        efficiency=1.0,
        efficiency2=efficiency2,
        efficiency3=efficiency3,
        p_nom_extendable=True,
        lifetime=costs.at["direct air capture", "lifetime"],
    )


def add_services(n, costs):
    temporal_resolution = n.snapshot_weightings.generators
    buses = spatial.nodes.intersection(n.loads_t.p_set.columns)

    profile_residential = normalize_by_country(
        n.loads_t.p_set[buses].reindex(columns=spatial.nodes, fill_value=0.0)
    ).fillna(0)

    p_set_elec = p_set_from_scaling(
        "services electricity", profile_residential, energy_totals, temporal_resolution
    )

    n.madd(
        "Load",
        spatial.nodes,
        suffix=" services electricity",
        bus=spatial.nodes,
        carrier="services electricity",
        p_set=p_set_elec,
    )
    p_set_biomass = p_set_from_scaling(
        "services biomass", profile_residential, energy_totals, temporal_resolution
    )

    n.madd(
        "Load",
        spatial.nodes,
        suffix=" services biomass",
        bus=broadcast_madd_value(
            (
                spatial.biomass.buildings
                if biomass_allocation is not None
                else spatial.biomass.nodes
            ),
            spatial.nodes,
            "services biomass bus",
        ),
        carrier="services biomass",
        p_set=p_set_biomass,
    )

    # co2 = (
    #     p_set_biomass.sum().sum() * costs.at["solid biomass", "CO2 intensity"]
    # ) / 8760

    # n.add(
    #     "Load",
    #     "services biomass emissions",
    #     bus="co2 atmosphere",
    #     carrier="biomass emissions",
    #     p_set=-co2,
    # )
    p_set_oil = p_set_from_scaling(
        "services oil", profile_residential, energy_totals, temporal_resolution
    )

    n.madd(
        "Load",
        spatial.nodes,
        suffix=" services oil",
        bus=spatial.oil.nodes,
        carrier="services oil",
        p_set=p_set_oil,
    )

    # Use helper function for proper temporal weighting
    add_emissions_from_weighted_energy(
        n,
        "services oil emissions",
        p_set_oil,
        costs.at["oil", "CO2 intensity"],
        "co2 atmosphere",
        "oil emissions",
        flat=True
    )

    p_set_gas = p_set_from_scaling(
        "services gas", profile_residential, energy_totals, temporal_resolution
    )

    n.madd(
        "Load",
        spatial.nodes,
        suffix=" services gas",
        bus=spatial.gas.nodes,
        carrier="services gas",
        p_set=p_set_gas,
    )

    # Use helper function for proper temporal weighting
    add_emissions_from_weighted_energy(
        n,
        "services gas emissions",
        p_set_gas,
        costs.at["gas", "CO2 intensity"],
        "co2 atmosphere",
        "gas emissions",
        flat=True
    )


def add_agriculture(n, costs):
    # Get total weighted hours (replaces hardcoded 8760)
    W = n.snapshot_weightings.generators.sum()
    
    p_set_agri_elec = nodal_energy_totals.loc[spatial.nodes, "agriculture electricity"] * 1e6 / W
    
    n.madd(
        "Load",
        spatial.nodes,
        suffix=" agriculture electricity",
        bus=spatial.nodes,
        carrier="agriculture electricity",
        p_set=p_set_agri_elec,
    )

    n.madd(
        "Load",
        spatial.nodes,
        suffix=" agriculture oil",
        bus=spatial.oil.nodes,
        carrier="agriculture oil",
        p_set=nodal_energy_totals.loc[spatial.nodes, "agriculture oil"] * 1e6 / W,
    )
    
    # Use robust emissions calculation
    p_set_agri_oil = nodal_energy_totals.loc[spatial.nodes, "agriculture oil"] * 1e6 / W
    add_emissions_from_weighted_energy(
        n,
        "agriculture oil emissions",
        p_set_agri_oil,
        costs.at["oil", "CO2 intensity"],
        "co2 atmosphere",
        "oil emissions"
    )


def normalize_by_country(df, droplevel=False):
    """
    Auxiliary function to normalize a dataframe by the country.

    If droplevel is False (default), the country level is added to the
    column index If droplevel is True, the original column format is
    preserved
    """
    ret = df.T.groupby(df.columns.str[:2]).apply(lambda x: x / x.sum().sum()).T
    if droplevel:
        return ret.droplevel(0, axis=1)
    else:
        return ret


def group_by_node(df, multiindex=False):
    """
    Auxiliary function to group a dataframe by the node name.
    """
    ret = df.T.groupby(df.columns.str.split(" ").str[0]).sum().T
    if multiindex:
        ret.columns = pd.MultiIndex.from_tuples(zip(ret.columns.str[:2], ret.columns))
    return ret


def _resolve_validation_repo_path(path_like):
    if hasattr(_validation_hooks, "_repo_path"):
        return _validation_hooks._repo_path(path_like)

    path = os.path.expanduser(str(path_like))
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def _get_year2025_country_capacity_upper_bounds(config, carriers):
    global_cfg = config.get("global_specific", {})
    year_cfg = global_cfg.get("year2025_capacity", {})
    if not year_cfg or not bool(year_cfg.get("year2025_capacity_constraint", False)):
        return pd.Series(dtype=float), None, None
    if not hasattr(_validation_hooks, "_irena_country_capacity_reference"):
        logger.warning(
            "Renewable nodal share cap skipped: validation helper _irena_country_capacity_reference is unavailable."
        )
        return pd.Series(dtype=float), None, None

    irena_csv = _resolve_validation_repo_path(
        year_cfg.get("irena_csv", "validation/data/irena_capacity_by_technology.csv")
    )
    if not os.path.exists(irena_csv):
        logger.warning(
            "Renewable nodal share cap skipped: IRENA capacity file not found at %s.",
            irena_csv,
        )
        return pd.Series(dtype=float), None, None

    target_year = int(year_cfg.get("year", 2025))
    reference_year = int(year_cfg.get("reference_year", target_year))
    fallback_to_latest = bool(year_cfg.get("fallback_to_latest_available", True))
    tolerance = float(year_cfg.get("tolerance", 0.10))
    upper_multiplier = year_cfg.get("upper_multiplier")
    upper_multiplier = (
        float(upper_multiplier) if upper_multiplier is not None else 1.0 + tolerance
    )

    constraint_technology_map = year_cfg.get(
        "irena_technology_by_constraint",
        year_cfg.get(
            "irena_technology_by_carrier",
            {
                "solar": ["PV"],
                "onwind": ["Onshore"],
            },
        ),
    )
    model_carriers_by_constraint = year_cfg.get(
        "model_carriers_by_constraint",
        {key: [key] for key in (constraint_technology_map or {}).keys()},
    )

    try:
        ref, used_reference_year = _validation_hooks._irena_country_capacity_reference(
            irena_csv=irena_csv,
            year=reference_year,
            carrier_technology_map=constraint_technology_map,
            fallback_to_latest=fallback_to_latest,
        )
    except Exception as exc:
        logger.warning(
            "Renewable nodal share cap skipped: unable to load IRENA country capacities from %s (%s).",
            irena_csv,
            exc,
        )
        return pd.Series(dtype=float), None, None

    if ref.empty:
        logger.warning(
            "Renewable nodal share cap skipped: no overlapping IRENA country-capacity rows found in %s.",
            irena_csv,
        )
        return pd.Series(dtype=float), None, None

    rows = []
    for row in ref.itertuples(index=False):
        for model_carrier in model_carriers_by_constraint.get(str(row.carrier), []):
            model_carrier = str(model_carrier)
            if model_carrier in carriers:
                rows.append(
                    (
                        str(row.country).upper(),
                        model_carrier,
                        max(float(row.reference_mw) * upper_multiplier, 0.0),
                    )
                )

    if not rows:
        return pd.Series(dtype=float), used_reference_year, upper_multiplier

    bounds = (
        pd.DataFrame(rows, columns=["country", "carrier", "upper_mw"])
        .drop_duplicates()
        .groupby(["country", "carrier"])["upper_mw"]
        .max()
    )
    return bounds, used_reference_year, upper_multiplier


def apply_renewable_nodal_share_caps(n, investment_year, config):
    global_cfg = config.get("global_specific", {})
    year_cfg = global_cfg.get("year2025_capacity", {})
    cap_cfg = year_cfg.get("nodal_distribution_limit", {})
    if not cap_cfg or not bool(cap_cfg.get("enable", False)):
        return

    target_year = int(year_cfg.get("year", 2025))
    if int(investment_year) != target_year:
        return

    logger.info(
        "Skipping prenetwork nodal share-cap mutation for %s; nodal_distribution_limit is enforced as a solve-time linear constraint.",
        investment_year,
    )


def apply_country_onwind_mean_cf_caps(n, investment_year):
    """Clip country-internal onwind CF outliers by scaling their full profile."""

    if int(investment_year) != 2020:
        logger.info(
            "Skipping country onwind mean-CF cap for %s; onwind CF caps are baseyear-only.",
            investment_year,
        )
        return 0

    if n.generators.empty or n.generators_t.p_max_pu.empty:
        return 0

    gen = n.generators.copy()
    gen["carrier"] = gen["carrier"].astype(str)
    gen = gen.loc[gen["carrier"].eq("onwind"), ["bus"]].copy()
    if gen.empty:
        return 0

    profiles = n.generators_t.p_max_pu.reindex(columns=gen.index)
    profiles = profiles.loc[:, profiles.notna().any(axis=0)]
    if profiles.empty:
        return 0
    gen = gen.loc[profiles.columns].copy()

    if "generators" in n.snapshot_weightings:
        weights = pd.to_numeric(
            n.snapshot_weightings["generators"], errors="coerce"
        ).reindex(n.snapshots)
        weights = weights.fillna(0.0)
    else:
        weights = pd.Series(1.0, index=n.snapshots)
    if float(weights.sum()) <= 0.0:
        weights = pd.Series(1.0, index=n.snapshots)

    bus_country = (
        n.buses["country"]
        if "country" in n.buses.columns
        else pd.Series(index=n.buses.index, dtype=object)
    )
    gen["country"] = gen["bus"].map(bus_country)
    fallback_country = gen["bus"].astype(str).str.split().str[0]
    gen["country"] = (
        gen["country"].fillna(fallback_country).astype(str).str.strip().str.upper()
    )
    gen = gen.loc[gen["country"].str.len().eq(2)].copy()
    if gen.empty:
        return 0

    profiles = profiles.loc[:, gen.index]
    mean_cf = profiles.mul(weights, axis=0).sum(axis=0) / float(weights.sum())
    gen["mean_cf"] = mean_cf.reindex(gen.index)
    gen["country_node_count"] = gen.groupby("country")["mean_cf"].transform("count")
    gen["country_median_cf"] = gen.groupby("country")["mean_cf"].transform("median")
    gen["country_cap_cf"] = 2.0 * gen["country_median_cf"]

    to_clip = gen.loc[
        gen["country_node_count"].ge(4)
        & gen["country_median_cf"].gt(0.0)
        & gen["mean_cf"].gt(gen["country_cap_cf"] + 1e-9)
    ].copy()
    if to_clip.empty:
        return 0

    scale = to_clip["country_cap_cf"] / to_clip["mean_cf"]
    n.generators_t.p_max_pu.loc[:, to_clip.index] = (
        n.generators_t.p_max_pu.loc[:, to_clip.index].mul(scale, axis=1).clip(lower=0.0)
    )

    sample = ", ".join(
        f"{name} {row.mean_cf:.3f}->{row.country_cap_cf:.3f} (country={row.country}, median={row.country_median_cf:.3f})"
        for name, row in to_clip.head(6).iterrows()
    )
    logger.info(
        "Applied country onwind mean-CF cap for %s: clipped_generators=%d, countries=%d, cap_rule='mean_cf <= 2.0 * country_median_cf' sample=%s",
        investment_year,
        len(to_clip),
        to_clip["country"].nunique(),
        sample,
    )
    return int(len(to_clip))


def normalize_and_group(df, multiindex=False):
    """
    Function to concatenate normalize_by_country and group_by_node.
    """
    return group_by_node(
        normalize_by_country(df, droplevel=True), multiindex=multiindex
    )


def p_set_from_scaling(col, scaling, energy_totals, nhours):
    """
    Function to create p_set from energy_totals, using the per-unit scaling
    dataframe.
    """
    result = 1e6 * scaling.div(nhours, axis=0).mul(energy_totals[col], level=0)
    
    # Only drop level if there are multiple levels in the column index
    if isinstance(result.columns, pd.MultiIndex) and result.columns.nlevels > 1:
        return result.droplevel(level=0, axis=1)
    else:
        return result


def add_residential(n, costs):
    # need to adapt for many countries #TODO

    # if snakemake.config["custom_data"]["heat_demand"]:
    # heat_demand_index=n.loads_t.p.filter(like='residential').filter(like='heat').dropna(axis=1).index
    # oil_res_index=n.loads_t.p.filter(like='residential').filter(like='oil').dropna(axis=1).index

    temporal_resolution = n.snapshot_weightings.generators

    heat_ind = (
        n.loads_t.p_set.filter(like="residential")
        .filter(like="heat")
        .dropna(axis=1)
        .columns
    )
    heat_shape_raw = normalize_by_country(n.loads_t.p_set[heat_ind])
    heat_shape = heat_shape_raw.rename(
        columns=n.loads.bus.map(n.buses.location), level=1
    )
    heat_shape = heat_shape.T.groupby(level=[0, 1]).sum().T

    res_heat_load = 1e6 * heat_shape_raw.mul(
        energy_totals["total residential space"]
        + energy_totals["total residential water"]
        - energy_totals["residential heat biomass"]
        - energy_totals["residential heat oil"]
        - energy_totals["residential heat gas"],
        level=0,
    ).droplevel(level=0, axis=1).div(temporal_resolution, axis=0)
    n.loads_t.p_set[heat_ind] = res_heat_load

    heat_oil_demand = p_set_from_scaling(
        "residential heat oil", heat_shape, energy_totals, temporal_resolution
    )
    heat_biomass_demand = p_set_from_scaling(
        "residential heat biomass", heat_shape, energy_totals, temporal_resolution
    )

    heat_gas_demand = p_set_from_scaling(
        "residential heat gas", heat_shape, energy_totals, temporal_resolution
    )

    res_index = spatial.nodes.intersection(n.loads_t.p_set.columns)
    profile_residential_raw = normalize_by_country(n.loads_t.p_set[res_index])
    profile_residential = profile_residential_raw.rename(
        columns=n.loads.bus.map(n.buses.location), level=1
    )
    profile_residential = profile_residential.T.groupby(level=[0, 1]).sum().T

    p_set_oil = (
        p_set_from_scaling(
            "residential oil", profile_residential, energy_totals, temporal_resolution
        )
        + heat_oil_demand
    )

    p_set_biomass = (
        p_set_from_scaling(
            "residential biomass",
            profile_residential,
            energy_totals,
            temporal_resolution,
        )
        + heat_biomass_demand
    )

    p_set_gas = (
        p_set_from_scaling(
            "residential gas", profile_residential, energy_totals, temporal_resolution
        )
        + heat_gas_demand
    )

    n.madd(
        "Load",
        spatial.nodes,
        suffix=" residential oil",
        bus=spatial.oil.nodes,
        carrier="residential oil",
        p_set=p_set_oil,
    )

    # Use helper function for proper temporal weighting
    add_emissions_from_weighted_energy(
        n,
        "residential oil emissions",
        p_set_oil,
        costs.at["oil", "CO2 intensity"],
        "co2 atmosphere",
        "oil emissions",
        flat=True
    )
    n.madd(
        "Load",
        spatial.nodes,
        suffix=" residential biomass",
        bus=broadcast_madd_value(
            (
                spatial.biomass.buildings
                if biomass_allocation is not None
                else spatial.biomass.nodes
            ),
            spatial.nodes,
            "residential biomass bus",
        ),
        carrier="residential biomass",
        p_set=p_set_biomass,
    )

    n.madd(
        "Load",
        spatial.nodes,
        suffix=" residential gas",
        bus=spatial.gas.nodes,
        carrier="residential gas",
        p_set=p_set_gas,
    )

    # Use helper function for proper temporal weighting
    add_emissions_from_weighted_energy(
        n,
        "residential gas emissions",
        p_set_gas,
        costs.at["gas", "CO2 intensity"],
        "co2 atmosphere",
        "gas emissions",
        flat=True
    )

    for country in countries:
        rem_heat_demand = (
            energy_totals.loc[country, "total residential space"]
            + energy_totals.loc[country, "total residential water"]
            - energy_totals.loc[country, "residential heat biomass"]
            - energy_totals.loc[country, "residential heat oil"]
            - energy_totals.loc[country, "residential heat gas"]
        )

        heat_buses = (n.loads_t.p_set.filter(regex="heat").filter(like=country)).columns

        safe_division = safe_divide(
            n.loads_t.p_set.filter(like=country)[heat_buses],
            n.loads_t.p_set.filter(like=country)[heat_buses].sum().sum(),
        )
        n.loads_t.p_set.loc[:, heat_buses] = np.where(
            safe_division.notna(),
            (safe_division * rem_heat_demand * 1e6).div(temporal_resolution, axis=0),
            0.0,
        )

    # Revise residential electricity demand
    buses = n.buses[n.buses.carrier == "AC"].index.intersection(n.loads_t.p_set.columns)

    profile_pu = normalize_by_country(n.loads_t.p_set[buses]).fillna(0)
    res_elec_load = p_set_from_scaling(
        "electricity residential", profile_pu, energy_totals, temporal_resolution
    )
    n.loads_t.p_set.loc[:, buses] = res_elec_load


def add_electricity_distribution_grid(n, costs):
    logger.info("Adding electricity distribution network")
    nodes = pop_layout.index

    n.madd(
        "Bus",
        nodes + " low voltage",
        location=nodes,
        carrier="low voltage",
        unit="MWh_el",
    )

    n.madd(
        "Link",
        nodes + " electricity distribution grid",
        bus0=nodes,
        bus1=nodes + " low voltage",
        p_nom_extendable=True,
        p_min_pu=-1,
        carrier="electricity distribution grid",
        efficiency=1,
        lifetime=costs.at["electricity distribution grid", "lifetime"],
        capital_cost=costs.at["electricity distribution grid", "fixed"],
    )

    # deduct distribution losses from electricity demand as these are included in total load
    # https://nbviewer.org/github/Open-Power-System-Data/datapackage_timeseries/blob/2020-10-06/main.ipynb
    if (
        efficiency := options["transmission_efficiency"]
        .get("electricity distribution grid", {})
        .get("efficiency_static")
    ):
        logger.info(
            f"Deducting distribution losses from electricity demand: {np.around(100*(1-efficiency), decimals=2)}%"
        )
        n.loads_t.p_set.loc[:, n.loads.carrier == "AC"] *= efficiency

    # move AC loads to low voltage buses
    ac_loads = n.loads.index[n.loads.carrier == "AC"]
    n.loads.loc[ac_loads, "bus"] += " low voltage"

    # move industry, rail transport, agriculture and services electricity to low voltage
    loads = n.loads.index[n.loads.carrier.str.contains("electricity")]
    n.loads.loc[loads, "bus"] += " low voltage"

    bevs = n.links.index[n.links.carrier == "BEV charger"]
    n.links.loc[bevs, "bus0"] += " low voltage"

    v2gs = n.links.index[n.links.carrier == "V2G"]
    n.links.loc[v2gs, "bus1"] += " low voltage"

    hps = n.links.index[n.links.carrier.str.contains("heat pump")]
    n.links.loc[hps, "bus0"] += " low voltage"

    rh = n.links.index[n.links.carrier.str.contains("resistive heater")]
    n.links.loc[rh, "bus0"] += " low voltage"

    mchp = n.links.index[n.links.carrier.str.contains("micro gas")]
    n.links.loc[mchp, "bus1"] += " low voltage"

    if options.get("solar_rooftop", False):
        logger.info("Adding solar rooftop technology")
        # set existing solar to cost of utility cost rather the 50-50 rooftop-utility
        solar = n.generators.index[n.generators.carrier == "solar"]
        n.generators.loc[solar, "capital_cost"] = costs.at["solar-utility", "fixed"]
        pop_solar = pop_layout.total.rename(index=lambda x: x + " solar")

        # add max solar rooftop potential assuming 0.1 kW/m2 and 20 m2/person,
        # i.e. 2 kW/person (population data is in thousands of people) so we get MW
        potential = 0.1 * 20 * pop_solar

        # For solar generators without population data, use average population
        missing_solar = solar.difference(potential.index)
        if len(missing_solar) > 0:
            avg_population = pop_layout.total.mean()
            print(f"No. of missing solar generators: {len(missing_solar)}")
            print(f"Warning: Solar generators without population data: {missing_solar.tolist()}")
            print(f"Using average population ({avg_population:.1f}k people) for missing solar generators")
            
            # Add missing entries with average population
            for solar_gen in missing_solar:
                potential[solar_gen] = 0.1 * 20 * avg_population

        n.madd(
            "Generator",
            solar,
            suffix=" rooftop",
            bus=n.generators.loc[solar, "bus"] + " low voltage",
            carrier="solar rooftop",
            p_nom_extendable=True,
            p_nom_max=potential.loc[solar],
            marginal_cost=n.generators.loc[solar, "marginal_cost"],
            capital_cost=costs.at["solar-rooftop", "fixed"],
            efficiency=n.generators.loc[solar, "efficiency"],
            p_max_pu=n.generators_t.p_max_pu[solar],
            lifetime=costs.at["solar-rooftop", "lifetime"],
        )

    if options.get("home_battery", False):
        logger.info("Adding home battery technology")
        n.add("Carrier", "home battery")

        n.madd(
            "Bus",
            nodes + " home battery",
            location=nodes,
            carrier="home battery",
            unit="MWh_el",
        )

        n.madd(
            "Store",
            nodes + " home battery",
            bus=nodes + " home battery",
            location=nodes,
            e_cyclic=True,
            e_nom_extendable=True,
            carrier="home battery",
            capital_cost=costs.at["home battery storage", "fixed"],
            lifetime=costs.at["battery storage", "lifetime"],
        )

        n.madd(
            "Link",
            nodes + " home battery charger",
            bus0=nodes + " low voltage",
            bus1=nodes + " home battery",
            carrier="home battery charger",
            efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
            capital_cost=costs.at["home battery inverter", "fixed"],
            p_nom_extendable=True,
            lifetime=costs.at["battery inverter", "lifetime"],
        )

        n.madd(
            "Link",
            nodes + " home battery discharger",
            bus0=nodes + " home battery",
            bus1=nodes + " low voltage",
            carrier="home battery discharger",
            efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
            marginal_cost=options["marginal_cost_storage"],
            p_nom_extendable=True,
            lifetime=costs.at["battery inverter", "lifetime"],
        )


# def add_co2limit(n, Nyears=1.0, limit=0.0):
#     print("Adding CO2 budget limit as per unit of 1990 levels of", limit)

#     countries = n.buses.country.dropna().unique()

#     sectors = emission_sectors_from_opts(opts)

#     # convert Mt to tCO2
#     co2_totals = 1e6 * pd.read_csv(snakemake.input.co2_totals_name, index_col=0)

#     co2_limit = co2_totals.loc[countries, sectors].sum().sum()

#     co2_limit *= limit * Nyears

#     n.add(
#         "GlobalConstraint",
#         "CO2Limit",
#         carrier_attribute="co2_emissions",
#         sense="<=",
#         constant=co2_limit,
#     )


def add_custom_water_cost(n):
    for country in countries:
        water_costs = pd.read_csv(
            os.path.join(
                BASE_DIR,
                "resources/custom_data/{}_water_costs.csv".format(country),
                sep=",",
                index_col=0,
            )
        )
        water_costs = water_costs.filter(like=country, axis=0).loc[spatial.nodes]
        electrolysis_links = n.links.filter(like=country, axis=0).filter(
            like="lectrolysis", axis=0
        )

        elec_index = n.links[
            (n.links.carrier == "H2 Electrolysis")
            & (n.links.bus0.str.contains(country))
        ].index
        n.links.loc[elec_index, "marginal_cost"] = water_costs.values
        # n.links.filter(like=country, axis=0).filter(like='lectrolysis', axis=0)["marginal_cost"] = water_costs.values
        # n.links.filter(like=country, axis=0).filter(like='lectrolysis', axis=0).apply(lambda x: water_costs[x.index], axis=0)
        # print(n.links.filter(like=country, axis=0).filter(like='lectrolysis', axis=0).marginal_cost)


def add_rail_transport(n, costs):
    # Get total weighted hours (replaces hardcoded 8760)
    W = n.snapshot_weightings.generators.sum()
    
    p_set_elec = nodal_energy_totals.loc[spatial.nodes, "electricity rail"]
    p_set_oil = (nodal_energy_totals.loc[spatial.nodes, "total rail"]) - p_set_elec

    n.madd(
        "Load",
        spatial.nodes,
        suffix=" rail transport oil",
        bus=spatial.oil.nodes,
        carrier="rail transport oil",
        p_set=p_set_oil * 1e6 / W,
    )

    p_set_rail_elec = p_set_elec * 1e6 / W
    
    n.madd(
        "Load",
        spatial.nodes,
        suffix=" rail transport electricity",
        bus=spatial.nodes,
        carrier="rail transport electricity",
        p_set=p_set_rail_elec,
    )


def get_capacities_from_elec(n, carriers, component):
    """
    Gets capacities and efficiencies for {carrier} in n.{component} that were
    previously assigned in add_electricity.
    """
    component_list = ["generators", "storage_units", "links", "stores"]
    component_dict = {name: getattr(n, name) for name in component_list}
    e_nom_carriers = ["stores"]
    nom_col = {x: "e_nom" if x in e_nom_carriers else "p_nom" for x in component_list}
    eff_col = "efficiency"

    capacity_dict = {}
    efficiency_dict = {}
    node_dict = {}
    for carrier in carriers:
        capacity_dict[carrier] = component_dict[component].query("carrier in @carrier")[
            nom_col[component]
        ]
        efficiency_dict[carrier] = component_dict[component].query(
            "carrier in @carrier"
        )[eff_col]
        node_dict[carrier] = component_dict[component].query("carrier in @carrier")[
            "bus"
        ]

    return capacity_dict, efficiency_dict, node_dict


def remove_elec_base_techs(n):
    """
    Remove conventional generators (e.g. OCGT, oil) build in electricity-only network,
    since they're re-added here using links.
    """
    conventional_generators = options.get("conventional_generation", {})
    to_remove = pd.Index(conventional_generators.keys())
    # remove only conventional_generation carriers present in the network
    to_remove = pd.Index(
        snakemake.params.electricity.get("conventional_carriers", [])
    ).intersection(to_remove)

    if to_remove.empty:
        return

    logger.info(f"Removing Generators with carrier {list(to_remove)}")
    names = n.generators.index[n.generators.carrier.isin(to_remove)]
    for name in names:
        n.remove("Generator", name)
    n.carriers.drop(to_remove, inplace=True, errors="ignore")


def remove_carrier_related_components(n, carriers_to_drop):
    """
    Removes carrier related components, such as "Carrier", "Generator", "Link", "Store", and "Storage Unit"
    """
    # remove carriers
    n.carriers.drop(carriers_to_drop, inplace=True, errors="ignore")

    # remove buses, generators, stores, and storage units with carrier to remote
    for c in n.iterate_components(["Bus", "Generator", "Store", "StorageUnit"]):
        logger.info(f"Removing {c.list_name} with carrier {list(carriers_to_drop)}")
        names = c.df.index[c.df.carrier.isin(carriers_to_drop)]
        if c.name == "Bus":
            buses_to_remove = names
        n.mremove(c.name, names)

    # remove links connected to buses that were removed
    links_to_remove = n.links.query(
        "bus0 in @buses_to_remove or bus1 in @buses_to_remove or bus2 in @buses_to_remove or bus3 in @buses_to_remove or bus4 in @buses_to_remove"
    ).index
    logger.info(
        f"Removing links with carrier {list(n.links.loc[links_to_remove].carrier.unique())}"
    )
    n.mremove("Link", links_to_remove)


def add_direct_electric_loads(n, energy_totals, columns, temporal_resolution):
    """
    Adds direct electricity Loads on the AC bus for each column in `columns`.
    The time profile defaults to the normalized AC load profile by country.
    """
    # Get buses that have existing AC load profiles, but use all spatial.nodes for assignment
    existing_buses = spatial.nodes.intersection(n.loads_t.p_set.columns)

    if len(existing_buses) > 0:
        profile_residential = normalize_by_country(
            n.loads_t.p_set[existing_buses].reindex(columns=spatial.nodes, fill_value=0.0)
        ).fillna(0)
    else:
        # Create default profile if no buses have loads yet
        default_profile = pd.DataFrame(index=n.snapshots, columns=spatial.nodes, data=1.0)
        profile_residential = normalize_by_country(default_profile).fillna(0)

    for col in columns:
        if col not in energy_totals.columns:
            continue
        p_set = p_set_from_scaling(col, profile_residential, energy_totals, temporal_resolution)
        n.madd(
            "Load",
            spatial.nodes,                     # use spatial nodes 
            suffix=f" {col}",
            bus=spatial.nodes,                 # AC buses
            carrier=col,
            p_set=p_set,
        )


if __name__ == "__main__":
    if "snakemake" not in globals():
        # from helper import mock_snakemake #TODO remove func from here to helper script
        snakemake = mock_snakemake(
            "prepare_sector_network",
            simpl="",
            clusters="200",
            ll="copt",
            opts="3h",
            planning_horizons="2020",
            sopts="72h",
            configfile="/shared/share_cki25/energymodels/pypsa-earth/config.myopic.yaml",
            discountrate=0.071,
            demand="AB",
        )

    # Load population layout
    pop_layout = pd.read_csv(snakemake.input.clustered_pop_layout, index_col=0)

    # Load all sector wildcards
    options = snakemake.params.sector_options

    # Load input network
    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)

    # Fetch the country list from the network
    # countries = list(n.buses.country.unique())
    countries = snakemake.params.countries
    # Locate all the AC buses
    nodes = n.buses[
        n.buses.carrier == "AC"
    ].index  # TODO if you take nodes from the index of buses of n it's more than pop_layout
    # clustering of regions must be double checked.. refer to regions onshore

    # Add location. TODO: move it into pypsa-earth
    n.buses.location = n.buses.index

    # Set carrier of AC loads
    existing_nodes = [node for node in nodes if node in n.loads.index]
    if len(existing_nodes) < len(nodes):
        print(
            f"Warning: For {len(nodes) - len(existing_nodes)} of {len(nodes)} nodes there were no load nodes found in network and were skipped."
        )
    n.loads.loc[existing_nodes, "carrier"] = "AC"

    Nyears = n.snapshot_weightings.generators.sum() / 8760

    # Fetch wildcards
    investment_year = int(snakemake.wildcards.planning_horizons[-4:])
    demand_sc = snakemake.wildcards.demand  # loading the demand scenario wildcard

    # Prepare the costs dataframe
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

    # Load country-specific fuel prices
    fuel_price_dict = load_country_fuel_prices(
        snakemake.input.fuelprices,
        investment_year,
        costs
    )
    fuel_price_dict = apply_country_fuel_price_overrides(
        fuel_price_dict,
        investment_year,
        costs,
        snakemake.config,
    )

    # Load country-specific WACCs for renewable technologies
    wacc_dict = load_country_waccs(
        snakemake.input.waccs,
        costs
    )

    # Define spatial for biomass and co2. They require the same spatial definition
    elec_nodes = set(n.buses.index[n.buses.carrier == "AC"])
    pop_layout = pop_layout[pop_layout.index.isin(elec_nodes)]

    spatial = define_spatial(pop_layout.index, options)

    if snakemake.params.foresight in ["myopic", "perfect"]:
        add_lifetime_wind_solar(n, costs)

    nodal_energy_totals = pd.read_csv(
        snakemake.input.nodal_energy_totals,
        index_col=0,
        keep_default_na=False,
        na_values=[""],
    )
    energy_totals = pd.read_csv(
        snakemake.input.energy_totals,
        index_col=0,
        keep_default_na=False,
        na_values=[""],
    )
    # Get the data required for land transport
    # TODO Leon, This contains transport demand, right? if so let's change it to transport_demand?
    transport = pd.read_csv(
        snakemake.input.transport, index_col=0, parse_dates=True
    ).reindex(columns=spatial.nodes, fill_value=0.0)

    avail_profile = pd.read_csv(
        snakemake.input.avail_profile, index_col=0, parse_dates=True
    )
    dsm_profile = pd.read_csv(
        snakemake.input.dsm_profile, index_col=0, parse_dates=True
    )
    nodal_transport_data = pd.read_csv(  # TODO This only includes no. of cars, change name to something descriptive?
        snakemake.input.nodal_transport_data, index_col=0
    )

    # Load data required for the heat sector
    heat_demand = pd.read_csv(
        snakemake.input.heat_demand, index_col=0, header=[0, 1], parse_dates=True
    ).fillna(0)
    # Ground-sourced heatpump coefficient of performance
    gshp_cop = pd.read_csv(
        snakemake.input.gshp_cop, index_col=0, parse_dates=True
    )  # only needed with heat dep. hp cop allowed from config
    # TODO add option heat_dep_hp_cop to the config

    # Air-sourced heatpump coefficient of performance
    ashp_cop = pd.read_csv(
        snakemake.input.ashp_cop, index_col=0, parse_dates=True
    )  # only needed with heat dep. hp cop allowed from config

    # Solar thermal availability profiles
    solar_thermal = pd.read_csv(
        snakemake.input.solar_thermal, index_col=0, parse_dates=True
    )
    gshp_cop = pd.read_csv(snakemake.input.gshp_cop, index_col=0, parse_dates=True)

    # Share of district heating at each node
    district_heat_share = pd.read_csv(snakemake.input.district_heat_share, index_col=0)
    district_heat_share = district_heat_share[district_heat_share.index.isin(elec_nodes)]

    # Load data required for aviation and navigation
    # TODO follow the same structure as land transport and heat

    # Load industry demand data
    industrial_demand = pd.read_csv(
        snakemake.input.industrial_demand, index_col=0, header=0
    )  # * 1e6

    biomass_allocation = derive_post2020_structural_biomass_allocation(
        investment_year=investment_year,
        config=snakemake.config,
        physical_total_twh=float(snakemake.params.sector_options["solid_biomass_potential"]),
        energy_totals=energy_totals,
        industrial_demand=industrial_demand,
    )

    ##########################################################################
    ############## Functions adding different carrires and sectors ###########
    ##########################################################################

    # read existing installed capacities of generators
    if options.get("keep_existing_capacities", False): # default in config is true
        existing_capacities, existing_efficiencies, existing_nodes = (
            get_capacities_from_elec(
                n,
                carriers=options.get("conventional_generation").keys(),
                component="generators",
            )
        )
    else:
        existing_capacities, existing_efficiencies, existing_nodes = 0, None, None

    add_co2(n, costs, options["co2_network"])  # TODO add costs

    # remove conventional generators built in elec-only model
    remove_elec_base_techs(n)

    add_generation(n, costs, existing_capacities, existing_efficiencies, existing_nodes, fuel_price_dict)

    # remove H2 and battery technologies added in elec-only model
    remove_carrier_related_components(n, carriers_to_drop=["H2", "battery"])

    add_hydrogen(n, costs)  # TODO add costs

    add_storage(n, costs)

    H2_liquid_fossil_conversions(n, costs)

    h2_hc_conversions(n, costs)
    add_heat(n, costs)
    add_biomass(n, costs)

    add_industry(n, costs)

    add_shipping(n, costs, gadm_clustering=snakemake.params.alternative_clustering, fuel_price_dict=fuel_price_dict)

    # Add_aviation runs with dummy data
    add_aviation(n, costs, gadm_clustering=snakemake.params.alternative_clustering)

    # prepare_transport_data(n)

    add_land_transport(n, costs)

    # if snakemake.config["custom_data"]["transport_demand"]:
    add_rail_transport(n, costs)

    # if snakemake.config["custom_data"]["custom_sectors"]:
    add_agriculture(n, costs)
    add_residential(n, costs)
    add_services(n, costs)

    # Add direct electric loads for other electricity categories
    _extra_cols = [
        "other electricity",
    ]
    add_direct_electric_loads(n, energy_totals, _extra_cols, n.snapshot_weightings.generators)

    if options.get("electricity_distribution_grid", False):
        add_electricity_distribution_grid(n, costs)

    sopts = snakemake.wildcards.sopts.split("-")

    m = None  # Initialize m to handle case where no pattern matches
    downsampled = False
    for o in sopts:
        m = re.match(r"^\d+h$", o, re.IGNORECASE)
        if m is not None and m.group(0).lower() != "1h":
            logger.info("Applying snapshot averaging...")
            n = average_every_nhours(n, m.group(0))
            downsampled = True
            break

    temporal_cfg = snakemake.params.get("temporal_clustering", {}) or {}
    tc_activated = bool(temporal_cfg.get("activate", False))
    if tc_activated:
        if downsampled:
            raise ValueError("Temporal clustering (TSAM) and snapshot averaging cannot be applied together. Please choose only one temporal reduction method.")
        else:
            logger.info("Applying temporal clustering (TSAM) during prepare_sector_network...")
            logger.info(f"TSAM parameters: n_periods={temporal_cfg.get('n_periods', 10)}, hours={temporal_cfg.get('hours', 24)}, method={temporal_cfg.get('clusterMethod', 'hierarchical')}")
            
            # Check if this is baseyear or if we should load existing clustering
            baseyear = snakemake.params.planning_horizons_baseyear
            current_year = investment_year
            tsam_input_path = snakemake.input.get("tsam_clustering", [])
            predef_cluster_order = None
            predef_cluster_centers = None
            
            if current_year != baseyear and tsam_input_path:
                import pickle
                try:
                    with open(tsam_input_path[0], "rb") as f:
                        tsam_data = pickle.load(f)
                    predef_cluster_order = tsam_data.get("predefClusterOrder", None)
                    predef_cluster_centers = tsam_data.get("predefClusterCenterIndices", None)
                    logger.info(f"Loaded TSAM clustering from baseyear ({baseyear}): {len(predef_cluster_order) if predef_cluster_order is not None else 0} days predefined, {len(predef_cluster_centers) if predef_cluster_centers is not None else 0} cluster centers")
                except Exception as e:
                    logger.warning(f"Failed to load TSAM clustering from {tsam_input_path[0]}: {e}")
                    logger.warning("Proceeding with independent clustering (may cause snapshot mismatch!)")
            
            # When using predefClusterOrder, disable extremePeriodMethod 
            # (TSAM's implementation doesn't properly support combining these features)
            extreme_period_method = temporal_cfg.get("extremePeriodMethod", "None")
            if predef_cluster_order is not None and extreme_period_method != "None":
                logger.warning(f"Disabling extremePeriodMethod ({extreme_period_method}) when using predefClusterOrder from baseyear")
                logger.warning("This ensures consistent snapshots across all planning horizons")
                extreme_period_method = "None"
            
            # Pass predefClusterCenterIndices to TSAM via temporal_cfg temporarily
            # (aggregate_snapshots doesn't have this parameter, but TSAM TimeSeriesAggregation does)
            if predef_cluster_centers is not None:
                temporal_cfg_with_centers = temporal_cfg.copy()
                temporal_cfg_with_centers["predefClusterCenterIndices"] = predef_cluster_centers
            else:
                temporal_cfg_with_centers = temporal_cfg
            
            aggregate_snapshots(
                n,
                n_periods=temporal_cfg.get("n_periods", 10),
                hours=temporal_cfg.get("hours", 24),
                normed=temporal_cfg.get("normed", True),
                solver=temporal_cfg.get("solver", "glpk"),
                extremePeriodMethod=extreme_period_method,
                clusterMethod=temporal_cfg.get("clusterMethod", "hierarchical"),
                predefClusterOrder=predef_cluster_order,
                predefClusterCenterIndices=predef_cluster_centers,
                overwrite_time_dfs=temporal_cfg.get("overwrite_time_dfs", False),
            )
            logger.info(f"TSAM aggregation complete. Network now has {len(n.snapshots)} snapshots; period_id persisted.")
            
            # Save clustering output file (required by Snakemake)
            import pickle
            import os
            tsam_output_path = snakemake.output.tsam_clustering
            os.makedirs(os.path.dirname(tsam_output_path), exist_ok=True)
            
            if current_year == baseyear and hasattr(n, "cluster_order"):
                # Baseyear: save full clustering data with clusterOrder and clusterCenterIndices
                predef_order = list(n.cluster_order)
                predef_centers = list(n.cluster_centers) if hasattr(n, "cluster_centers") and n.cluster_centers is not None else None
                
                tsam_data = {
                    "predefClusterOrder": predef_order,
                    "predefClusterCenterIndices": predef_centers,
                    "n_periods": temporal_cfg.get("n_periods", 10),
                    "hours": temporal_cfg.get("hours", 24),
                    "snapshots": n.snapshots.tolist(),
                    "snapshot_weightings": n.snapshot_weightings.to_dict(),
                }
                
                with open(tsam_output_path, "wb") as f:
                    pickle.dump(tsam_data, f)
                logger.info(f"Saved TSAM clustering to {tsam_output_path} with clusterOrder and clusterCenterIndices for reuse in future horizons")
            elif current_year == baseyear:
                logger.warning("Could not save TSAM clustering - n.cluster_order not found")
            else:
                # Non-baseyear: create placeholder file (clustering was loaded from baseyear)
                tsam_data = {
                    "note": f"Clustering loaded from baseyear {baseyear}",
                    "snapshots": n.snapshots.tolist(),
                    "snapshot_weightings": n.snapshot_weightings.to_dict(),
                }
                with open(tsam_output_path, "wb") as f:
                    pickle.dump(tsam_data, f)
                logger.info(f"Created placeholder TSAM clustering file at {tsam_output_path} (used baseyear clustering)")

    # Match exogenous electricity demand (Load components) to OWID in baseyear.
    # Endogenous electricity consumption through Links (e.g. heat pumps/electrolysis)
    # is intentionally not scaled here.
    align_country_electricity_demand_to_owid(n, investment_year, snakemake.config)
    # Patch hydro profile/inflow pathologies after upstream attachment but before hydro scaling.
    apply_hydro_profile_fallback_and_diagnostics(
        n,
        investment_year,
        snakemake.config,
        output_network_path=getattr(snakemake.output, "network", None),
    )
    # Patch missing/all-zero wind/solar/offshore profiles in every planning year
    # without reapplying baseyear country-level CF scaling.
    apply_renewable_profile_fallbacks(n, investment_year, snakemake.config)
    align_country_hydro_reservoir_inflow_to_owid(n, investment_year, snakemake.config)
    # Optional per-country iterative hydro overrides (reservoir+ror jointly).
    apply_country_hydro_iteration_scaling(n, investment_year, snakemake.config)
    # Scale onshore wind availability profiles against OWID wind electricity
    # using baseline conversion ratios in baseyear.
    align_country_onwind_profiles_to_owid(n, investment_year, snakemake.config)
    # Optional per-country iterative wind overrides written by calibration wrapper.
    apply_country_wind_iteration_scaling(n, investment_year, snakemake.config)
    # Optional per-country iterative solar overrides written by calibration wrapper.
    apply_country_solar_iteration_scaling(n, investment_year, snakemake.config)
    # Historical capacity validation for nuclear, oil, and bioenergy is applied
    # after add_existing_baseyear/add_brownfield so downstream stock edits cannot undo it.
    # Clip within-country onwind availability outliers before capacity constraints.
    apply_country_onwind_mean_cf_caps(n, investment_year)
    # Legacy compatibility hook; the actual nodal_distribution_limit is now enforced
    # as a solve-time linear constraint in solve_network/validation.
    apply_renewable_nodal_share_caps(n, investment_year, snakemake.config)

    # TODO add co2 limit here, if necessary
    # co2_limit_pu = eval(sopts[0][5:])
    # co2_limit = co2_limit_pu *
    # # Add co2 limit
    # co2_limit = 1e9
    # n.add(
    #     "GlobalConstraint",
    #     "CO2Limit",
    #     carrier_attribute="co2_emissions",
    #     sense="<=",
    #     constant=co2_limit,
    # )

    if options["dac"]:
        add_dac(n, costs)

    if snakemake.params.water_costs:
        add_custom_water_cost(n)

    adjust_hydro(n, investment_year, snakemake.config)
    
    # Match geothermal capacity to CSV data by country
    geothermal_csv = snakemake.input.geothermal_capacity

    regions_shapefile = snakemake.input.shapes_path
    
    match_geothermal_capacity_from_csv(
        n, 2020, geothermal_csv, 
        costs=costs,
        regions_shapefile=regions_shapefile
    ) # based on owid data
    

    # Apply country-specific WACCs to ALL renewable generators (must be last to catch all generators)
    logger.info("Applying regional WACCs to all renewable generators...")
    apply_regional_waccs(n, costs, wacc_dict, Nyears)

    if biomass_allocation is not None:
        if not isinstance(getattr(n, "meta", None), dict):
            n.meta = {}
        n.meta["post2020_structural_biomass"] = biomass_allocation.to_dict()

    n.export_to_netcdf(snakemake.output.network)
