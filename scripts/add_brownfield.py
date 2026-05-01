# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Prepares brownfield data from previous planning horizon.
"""

import logging

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from add_existing_baseyear import add_build_year_to_new_assets
from solve_network import apply_optional_sector_clustering
import solve_network as solve_network_module

# from pypsa.clustering.spatial import normed_or_uniform

logger = logging.getLogger(__name__)
idx = pd.IndexSlice


def _validate_imported_capacity_transfer(n, c, attr):
    """
    Validate that imported brownfield assets kept nominal capacity and build_year.
    """
    list_name = n.components[c.name]["list_name"]
    target_df = getattr(n, list_name)
    imported_idx = c.df.index.intersection(target_df.index)

    if len(imported_idx) != len(c.df.index):
        missing = c.df.index.difference(imported_idx)
        raise ValueError(
            f"Brownfield import missing {len(missing)} {c.name} assets after merge; "
            f"sample: {missing[:10].tolist()}"
        )

    expected_nom = float(c.df.loc[imported_idx, f"{attr}_nom"].fillna(0.0).sum())
    actual_nom = float(target_df.loc[imported_idx, f"{attr}_nom"].fillna(0.0).sum())
    if not np.isclose(expected_nom, actual_nom, rtol=1e-6, atol=1e-6):
        raise ValueError(
            f"Brownfield import capacity mismatch for {c.name}: "
            f"expected {expected_nom}, got {actual_nom}"
        )

    if "build_year" in c.df.columns and "build_year" in target_df.columns:
        expected_year = pd.to_numeric(c.df.loc[imported_idx, "build_year"], errors="coerce")
        actual_year = pd.to_numeric(target_df.loc[imported_idx, "build_year"], errors="coerce")
        mismatch = ~np.isclose(
            expected_year.fillna(-1.0).to_numpy(),
            actual_year.fillna(-1.0).to_numpy(),
            rtol=0.0,
            atol=0.0,
        )
        if mismatch.any():
            bad = imported_idx[mismatch][:10].tolist()
            raise ValueError(
                f"Brownfield import changed build_year for {c.name}; sample assets: {bad}"
            )


def _reroute_legacy_biomass_power_links(n):
    """Connect imported biomass-electric links to the structural power biomass store."""
    cfg = snakemake.config.get("global_specific", {}).get("post2020_structural_biomass", {}) or {}
    if not cfg.get("enable", False):
        return

    if "solid biomass power" not in set(n.buses.carrier):
        return

    biomass_power_carriers = {
        "biomass",
        "biomass EOP",
        "urban central solid biomass CHP",
        "urban central solid biomass CHP CC",
    }
    link_mask = n.links.carrier.isin(biomass_power_carriers)
    if not link_mask.any():
        return

    ac_output = pd.Series(False, index=n.links.index)
    for port in range(1, 5):
        bus_col = f"bus{port}"
        if bus_col in n.links.columns:
            ac_output |= n.links[bus_col].map(n.buses.carrier).fillna("").eq("AC")

    bus0 = n.links.loc[link_mask & ac_output, "bus0"].astype(str)
    old_biomass_bus = (
        bus0.map(n.buses.carrier).fillna("").eq("solid biomass")
        | bus0.eq("Earth solid biomass")
        | bus0.str.endswith(" solid biomass")
    )
    candidate_links = bus0.index[old_biomass_bus]
    if candidate_links.empty:
        return

    def power_bus_name(name):
        if name.endswith(" solid biomass"):
            return name[: -len(" solid biomass")] + " solid biomass power"
        return name + " power"

    target_bus = bus0.loc[candidate_links].map(power_bus_name)
    existing_target = target_bus.isin(n.buses.index)
    if not existing_target.all():
        missing = target_bus.loc[~existing_target].unique()[:10].tolist()
        logger.warning(
            "Could not reroute %d biomass power links to structural biomass power stores; "
            "missing target buses sample: %s",
            int((~existing_target).sum()),
            missing,
        )

    reroute_links = target_bus.index[existing_target]
    if reroute_links.empty:
        return

    n.links.loc[reroute_links, "bus0"] = target_bus.loc[reroute_links].values
    logger.info(
        "Rerouted %d biomass electricity links from solid biomass to solid biomass power.",
        len(reroute_links),
    )


def add_brownfield(n, n_p, year):
    logger.info(f"Preparing brownfield for the year {year}")

    # electric transmission grid set optimised capacities of previous as minimum
    n.lines.s_nom_min = n_p.lines.s_nom_opt
    dc_i = n.links[n.links.carrier == "DC"].index
    n.links.loc[dc_i, "p_nom_min"] = n_p.links.loc[dc_i, "p_nom_opt"]

    # Update p_nom_min/e_nom_min for extendable generators, links, and stores with build_year=0 (e.g., CCGT, batteries)
    # to prevent capacity from shrinking relative to previous horizon
    for c_name in ["Generator", "Link", "Store"]:
        if c_name == "Generator":
            component_n = n.generators
            component_n_p = n_p.generators
            attr = "p"
        elif c_name == "Link":
            component_n = n.links
            component_n_p = n_p.links
            attr = "p"
        else:  # Store
            component_n = n.stores
            component_n_p = n_p.stores
            attr = "e"
        
        # Find extendable assets with build_year=0 that exist in both networks
        # Exclude tracking assets (infinite lifetime)
        extendable_zero_build = component_n.index[
            (component_n.build_year == 0) & 
            (component_n[f"{attr}_nom_extendable"] == True) &
            (component_n.lifetime != np.inf)
        ]
        common_assets = extendable_zero_build.intersection(component_n_p.index)
        
        if not common_assets.empty:
            # Set p_nom_min/e_nom_min to the optimized capacity from previous horizon
            component_n.loc[common_assets, f"{attr}_nom_min"] = component_n_p.loc[common_assets, f"{attr}_nom_opt"].values
            logger.info(f"Updated {attr}_nom_min for {len(common_assets)} extendable {c_name}s with build_year=0 from previous horizon")

    for c in n_p.iterate_components(["Link", "Generator", "Store"]):
        attr = "e" if c.name == "Store" else "p"

        # first, remove generators, links and stores that track
        # CO2 or global EU values since these are already in n
        n_p.mremove(c.name, c.df.index[c.df.lifetime == np.inf])

        # remove assets whose build_year + lifetime < year
        n_p.mremove(c.name, c.df.index[c.df.build_year + c.df.lifetime < year])

        # remove assets if their optimized nominal capacity is lower than a threshold
        # since CHP heat Link is proportional to CHP electric Link, make sure threshold is compatible
        chp_heat = c.df.index[
            (c.df[f"{attr}_nom_extendable"] & c.df.index.str.contains("urban central"))
            & c.df.index.str.contains("CHP")
            & c.df.index.str.contains("heat")
        ]
        
        # Identify battery-related assets to exclude from threshold removal
        battery_assets = c.df.index[c.df.carrier.str.contains("battery", case=False, na=False)]

        threshold = snakemake.params.threshold_capacity

        if not chp_heat.empty:
            threshold_chp_heat = (
                threshold
                * c.df.efficiency[chp_heat.str.replace("heat", "electric")].values
                * c.df.p_nom_ratio[chp_heat.str.replace("heat", "electric")].values
                / c.df.efficiency[chp_heat].values
            )
            n_p.mremove(
                c.name,
                chp_heat[c.df.loc[chp_heat, f"{attr}_nom_opt"] < threshold_chp_heat],
            )

        # Remove assets below threshold, but exclude CHP heat and battery assets
        n_p.mremove(
            c.name,
            c.df.index[
                (c.df[f"{attr}_nom_extendable"] & ~c.df.index.isin(chp_heat) & ~c.df.index.isin(battery_assets))
                & (c.df[f"{attr}_nom_opt"] < threshold)
            ],
        )

        # copy over assets but fix their capacity
        c.df[f"{attr}_nom"] = c.df[f"{attr}_nom_opt"]
        c.df[f"{attr}_nom_extendable"] = False

        n.import_components_from_dataframe(c.df, c.name)
        _validate_imported_capacity_transfer(n, c, attr)

        # copy time-dependent
        selection = n.component_attrs[c.name].type.str.contains(
            "series"
        ) & n.component_attrs[c.name].status.str.contains("Input")
        
        # Check for snapshot alignment before importing time series
        if not n.snapshots.equals(n_p.snapshots):
            logger.warning(
                f"Snapshot mismatch detected when importing {c.name} time series from previous horizon!\n"
                f"  Current network has {len(n.snapshots)} snapshots: {n.snapshots[0]} to {n.snapshots[-1]}\n"
                f"  Previous network has {len(n_p.snapshots)} snapshots: {n_p.snapshots[0]} to {n_p.snapshots[-1]}\n"
                f"  This will cause PyPSA to fill missing values with defaults, corrupting capacity factors and other time series.\n"
                f"  Skipping time-dependent data import for {c.name}. Check temporal_clustering configuration."
            )
            # Skip time series import to avoid corruption
            continue
        
        for tattr in n.component_attrs[c.name].index[selection]:
            n.import_series_from_dataframe(c.pnl[tattr], c.name, tattr)

        # deal with gas network
        pipe_carrier = ["gas pipeline"]
        if snakemake.params.H2_retrofit:
            # drop capacities of previous year to avoid duplicating
            to_drop = n.links.carrier.isin(pipe_carrier) & (n.links.build_year != year)
            n.mremove("Link", n.links.loc[to_drop].index)

            # subtract the already retrofitted from today's gas grid capacity
            h2_retrofitted_fixed_i = n.links[
                (n.links.carrier == "H2 pipeline retrofitted")
                & (n.links.build_year != year)
            ].index
            gas_pipes_i = n.links[n.links.carrier.isin(pipe_carrier)].index
            CH4_per_H2 = 1 / snakemake.params.H2_retrofit_capacity_per_CH4
            fr = "H2 pipeline retrofitted"
            to = "gas pipeline"
            # today's pipe capacity
            pipe_capacity = n.links.loc[gas_pipes_i, "p_nom"]
            # already retrofitted capacity from gas -> H2
            already_retrofitted = (
                n.links.loc[h2_retrofitted_fixed_i, "p_nom"]
                .rename(lambda x: x.split("-2")[0].replace(fr, to))
                .groupby(level=0)
                .sum()
            )
            remaining_capacity = (
                pipe_capacity
                - CH4_per_H2
                * already_retrofitted.reindex(index=pipe_capacity.index).fillna(0)
            )
            n.links.loc[gas_pipes_i, "p_nom"] = remaining_capacity
        else:
            new_pipes = n.links.carrier.isin(pipe_carrier) & (
                n.links.build_year == year
            )
            n.links.loc[new_pipes, "p_nom"] = 0.0
            n.links.loc[new_pipes, "p_nom_min"] = 0.0

    _reroute_legacy_biomass_power_links(n)


def disable_grid_expansion_if_limit_hit(n):
    """
    Check if transmission expansion limit is already reached; then turn off.

    In particular, this function checks if the total transmission
    capital cost or volume implied by s_nom_min and p_nom_min are
    numerically close to the respective global limit set in
    n.global_constraints. If so, the nominal capacities are set to the
    minimum and extendable is turned off; the corresponding global
    constraint is then dropped.
    """
    cols = {"cost": "capital_cost", "volume": "length"}
    for limit_type in ["cost", "volume"]:
        glcs = n.global_constraints.query(
            f"type == 'transmission_expansion_{limit_type}_limit'"
        )

        for name, glc in glcs.iterrows():
            total_expansion = (
                (
                    n.lines.query("s_nom_extendable")
                    .eval(f"s_nom_min * {cols[limit_type]}")
                    .sum()
                )
                + (
                    n.links.query("carrier == 'DC' and p_nom_extendable")
                    .eval(f"p_nom_min * {cols[limit_type]}")
                    .sum()
                )
            ).sum()

            # Allow small numerical differences
            if np.abs(glc.constant - total_expansion) / glc.constant < 1e-6:
                logger.info(
                    f"Transmission expansion {limit_type} is already reached, disabling expansion and limit"
                )
                extendable_acs = n.lines.query("s_nom_extendable").index
                n.lines.loc[extendable_acs, "s_nom_extendable"] = False
                n.lines.loc[extendable_acs, "s_nom"] = n.lines.loc[
                    extendable_acs, "s_nom_min"
                ]

                extendable_dcs = n.links.query(
                    "carrier == 'DC' and p_nom_extendable"
                ).index
                n.links.loc[extendable_dcs, "p_nom_extendable"] = False
                n.links.loc[extendable_dcs, "p_nom"] = n.links.loc[
                    extendable_dcs, "p_nom_min"
                ]

                n.global_constraints.drop(name, inplace=True)


def adjust_battery_capacity_2025(n, year):
    """
    Adjust battery capacity for assets with build_year=2025 based on CSV data.
    
    Uses capa_2025 for Store energy capacity and power_2025_MW for Link power capacity.
    Follows the same distribution logic as in prepare_sector_network.py.
    
    Parameters
    ----------
    n : pypsa.Network
        Network to adjust
    year : int
        Current planning horizon year
    """
    battery_csv_path = snakemake.input.get("battery_capacities", None)
    
    if battery_csv_path is None:
        logger.warning("No battery capacity CSV path provided in snakemake.input")
        return
    
    try:
        battery_df = pd.read_csv(battery_csv_path, index_col=0)
    except FileNotFoundError:
        logger.warning(f"Battery capacity file not found at {battery_csv_path}")
    
    # Check if 2025 columns exist
    if 'capa_2025' not in battery_df.columns or 'power_2025_MW' not in battery_df.columns:
        logger.warning("Battery CSV missing capa_2025 or power_2025_MW columns")
        return
    
    # Get battery stores and links with build_year=2025
    battery_stores = n.stores[
        (n.stores.carrier == "battery") & (n.stores.build_year == year)
    ]
    battery_chargers = n.links[
        (n.links.carrier == "battery charger") & (n.links.build_year == year)
    ]
    battery_dischargers = n.links[
        (n.links.carrier == "battery discharger") & (n.links.build_year == year)
    ]
    
    if battery_stores.empty:
        logger.info(f"No battery stores with build_year={year} found")
        return
    
    # Validate that chargers and dischargers exist
    if battery_chargers.empty or battery_dischargers.empty:
        logger.warning(f"Found {len(battery_stores)} battery stores but missing chargers or dischargers for year {year}")
        return
    
    # Extract nodes from battery store names 
    # Format can be "XX 0 battery" or "XX 0-2025 battery" -> extract "XX 0"
    nodes = battery_stores.index.str.replace(r"-\d{4} battery$", "", regex=True).str.replace(" battery", "")
    
    # Extract country codes from node names (format: "XX 0" -> "XX")
    node_countries = pd.Series({node: node.split()[0][:2] for node in nodes})
    
    # Count nodes per country
    nodes_per_country = node_countries.value_counts()
    
    # Distribute 2025 battery capacity equally among nodes in each country
    total_e_adjusted = 0.0
    total_p_adjusted = 0.0
    
    for country_code, node_count in nodes_per_country.items():
        if country_code in battery_df.index:
            # Get 2025 capacity for this country and divide equally among its nodes
            country_e_nom = battery_df.loc[country_code, 'capa_2025'] / node_count  
            country_p_nom = battery_df.loc[country_code, 'power_2025_MW'] / node_count 
            
            # Get nodes for this country
            country_nodes = node_countries[node_countries == country_code].index
            
            # Build actual store/link names from the battery_stores/links indices
            # Get the actual names that exist in the network
            country_store_names = battery_stores.index[nodes.isin(country_nodes)]
            country_charger_names = battery_chargers.index[
                battery_chargers.index.str.replace(r"-\d{4} battery charger$", "", regex=True).str.replace(" battery charger", "").isin(country_nodes)
            ]
            country_discharger_names = battery_dischargers.index[
                battery_dischargers.index.str.replace(r"-\d{4} battery discharger$", "", regex=True).str.replace(" battery discharger", "").isin(country_nodes)
            ]
            
            # Update Store energy capacity (e_nom and e_nom_min)
            n.stores.loc[country_store_names, 'e_nom'] = country_e_nom
            n.stores.loc[country_store_names, 'e_nom_min'] = country_e_nom
            
            # Update Link power capacity for chargers (p_nom and p_nom_min)
            n.links.loc[country_charger_names, 'p_nom'] = country_p_nom
            n.links.loc[country_charger_names, 'p_nom_min'] = country_p_nom
            
            # Update Link power capacity for dischargers (p_nom and p_nom_min)
            n.links.loc[country_discharger_names, 'p_nom'] = country_p_nom
            n.links.loc[country_discharger_names, 'p_nom_min'] = country_p_nom
            
            total_e_adjusted += battery_df.loc[country_code, 'capa_2025']
            total_p_adjusted += battery_df.loc[country_code, 'power_2025_MW']
            
            logger.info(f"Adjusted {country_code}: {battery_df.loc[country_code, 'capa_2025']:.1f} MWh (store) / {battery_df.loc[country_code, 'power_2025_MW']:.1f} MW (power) for {node_count} nodes")
        else:
            logger.debug(f"No 2025 battery data for country {country_code}")
    
    logger.info(f"Total battery capacity adjusted for {year}: {total_e_adjusted:.1f} MWh energy, {total_p_adjusted:.1f} MW power")


# def adjust_renewable_profiles(n, input_profiles, params, year):
#     """
#     Adjusts renewable profiles according to the renewable technology specified,
#     using the latest year below or equal to the selected year.
#     """

#     # spatial clustering
#     cluster_busmap = pd.read_csv(snakemake.input.cluster_busmap, index_col=0).squeeze()
#     simplify_busmap = pd.read_csv(
#         snakemake.input.simplify_busmap, index_col=0
#     ).squeeze()
#     clustermaps = simplify_busmap.map(cluster_busmap)
#     clustermaps.index = clustermaps.index.astype(str)

#     # temporal clustering
#     dr = pd.date_range(**params["snapshots"], freq="h")
#     snapshotmaps = (
#         pd.Series(dr, index=dr).where(lambda x: x.isin(n.snapshots), pd.NA).ffill()
#     )

#     for carrier in params["carriers"]:
#         if carrier == "hydro":
#             continue
#         with xr.open_dataset(getattr(input_profiles, "profile_" + carrier)) as ds:
#             if ds.indexes["bus"].empty or "year" not in ds.indexes:
#                 continue

#             closest_year = max(
#                 (y for y in ds.year.values if y <= year), default=min(ds.year.values)
#             )

#             p_max_pu = (
#                 ds["profile"]
#                 .sel(year=closest_year)
#                 .transpose("time", "bus")
#                 .to_pandas()
#             )

#             # spatial clustering
#             weight = ds["weight"].sel(year=closest_year).to_pandas()
#             weight = weight.groupby(clustermaps).transform(normed_or_uniform)
#             p_max_pu = (p_max_pu * weight).T.groupby(clustermaps).sum().T
#             p_max_pu.columns = p_max_pu.columns + f" {carrier}"

#             # temporal_clustering
#             p_max_pu = p_max_pu.groupby(snapshotmaps).mean()

#             # replace renewable time series
#             n.generators_t.p_max_pu.loc[:, p_max_pu.columns] = p_max_pu


if __name__ == "__main__":
    if "snakemake" not in globals():

        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_brownfield",
            simpl="",
            clusters="110",
            ll="copt",
            opts="1h",
            planning_horizons="2025",
            sopts="1h",
            configfile="/shared/share_cki25/energymodels/pypsa-earth/config.myopic.yaml",
            discountrate=0.071,
            demand="AB",
            h2export="0.0"
        )

    logger.info(f"Preparing brownfield from the file {snakemake.input.network_p}")

    year = int(snakemake.wildcards.planning_horizons)

    n = pypsa.Network(snakemake.input.network)

    # TODO
    # adjust_renewable_profiles(n, snakemake.input, snakemake.params, year)

    add_build_year_to_new_assets(n, year)

    # Ensure brownfield merge uses the same clustered topology as the previous solved horizon.
    # apply_optional_sector_clustering() resolves default universal busmap path via
    # solve_network module globals; pass through this rule's snakemake context so both
    # solve_network and add_brownfield derive the same scenario-specific path.
    solve_network_module.snakemake = snakemake
    buses_before_clustering = len(n.buses)
    n = apply_optional_sector_clustering(n, snakemake.config)
    if len(n.buses) != buses_before_clustering:
        logger.info(
            "Applied additional sector clustering in add_brownfield before merge: buses %d -> %d.",
            buses_before_clustering,
            len(n.buses),
        )
    if not isinstance(getattr(n, "meta", None), dict):
        n.meta = {}
    n.meta.update({"brownfield_input_clustered_before_merge": True})

    n_p = pypsa.Network(snakemake.input.network_p)

    add_brownfield(n, n_p, year)

    # Reset capacity of current year assets to 0 and make them extendable.
    # The brownfield constraint will be enforced via imported assets from previous year.
    for c in n.iterate_components(["Link", "Generator", "Store"]):
        attr = "e" if c.name == "Store" else "p"
        current_year_assets = c.df.index[c.df.build_year == year]
        
        if not current_year_assets.empty:
            # Identify battery assets (Stores with carrier "battery" or Links with "battery charger"/"battery discharger")
            battery_mask = c.df.carrier.str.contains("battery", case=False, na=False)
            battery_current = current_year_assets[battery_mask[current_year_assets]]
            
            # Other assets (including biomass and biogas) are reset to zero and re-opened
            # for endogenous replacement in the current horizon.
            other_current = current_year_assets[~battery_mask[current_year_assets]]
            
            # Reset battery stores to 0 and make extendable
            if not battery_current.empty:
                c.df.loc[battery_current, f"{attr}_nom"] = 0
                c.df.loc[battery_current, f"{attr}_nom_min"] = 0
                c.df.loc[battery_current, f"{attr}_nom_extendable"] = True
                logger.info(f"Reset {len(battery_current)} {c.name} battery assets with build_year={year} to {attr}_nom=0, {attr}_nom_min=0, {attr}_nom_extendable=True")
            
            # Reset other assets to 0 and make extendable
            if not other_current.empty:
                c.df.loc[other_current, f"{attr}_nom"] = 0
                c.df.loc[other_current, f"{attr}_nom_min"] = 0
                c.df.loc[other_current, f"{attr}_nom_extendable"] = True
                logger.info(f"Reset {len(other_current)} {c.name} assets with build_year={year} to {attr}_nom=0, {attr}_nom_min=0, {attr}_nom_extendable=True")

    

    # Make geothermal and nuclear generators extendable to allow capacity expansion in future years
    #geothermal_gens = n.generators.index[n.generators.carrier == "geothermal"]
    #if not geothermal_gens.empty:
    #    n.generators.loc[geothermal_gens, "p_nom_extendable"] = True
    #    logger.info(f"Set {len(geothermal_gens)} geothermal generators to p_nom_extendable=True")
    
    #nuclear_gens = n.generators.index[n.generators.carrier == "nuclear"]
    #if not nuclear_gens.empty:
    #    n.generators.loc[nuclear_gens, "p_nom_extendable"] = True
    #    logger.info(f"Set {len(nuclear_gens)} nuclear generators to p_nom_extendable=True")

    for carrier in ['coal', 'gas', 'oil']:
        fuel_gens = n.generators.index[n.generators.carrier == carrier]
        
        if not fuel_gens.empty:
            n.generators.loc[fuel_gens, "p_nom"] = n.generators.loc[fuel_gens, "p_nom_min"]
            logger.info(f"Set {len(fuel_gens)} {carrier} generators' p_nom to p_nom_min")

    # adjust battery capacity with build year 2025 based on csv
    if year == 2025:
        adjust_battery_capacity_2025(n, year)

    disable_grid_expansion_if_limit_hit(n)

    n.meta.update(dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards))))
    
    n.export_to_netcdf(snakemake.output[0])
