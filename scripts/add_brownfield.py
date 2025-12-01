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

# from pypsa.clustering.spatial import normed_or_uniform

logger = logging.getLogger(__name__)
idx = pd.IndexSlice


def add_brownfield(n, n_p, year):
    logger.info(f"Preparing brownfield for the year {year}")

    # electric transmission grid set optimised capacities of previous as minimum
    n.lines.s_nom_min = n_p.lines.s_nom_opt
    dc_i = n.links[n.links.carrier == "DC"].index
    n.links.loc[dc_i, "p_nom_min"] = n_p.links.loc[dc_i, "p_nom_opt"]

    # Update p_nom_min for extendable generators and links with build_year=0 (e.g., CCGT)
    # to prevent capacity from shrinking relative to previous horizon
    for c_name in ["Generator", "Link"]:
        if c_name == "Generator":
            component_n = n.generators
            component_n_p = n_p.generators
        else:
            component_n = n.links
            component_n_p = n_p.links
        
        # Find extendable assets with build_year=0 that exist in both networks
        extendable_zero_build = component_n.index[
            (component_n.build_year == 0) & 
            (component_n.p_nom_extendable == True)
        ]
        common_assets = extendable_zero_build.intersection(component_n_p.index)
        
        if not common_assets.empty:
            # Set p_nom_min to the optimized capacity from previous horizon
            component_n.loc[common_assets, "p_nom_min"] = component_n_p.loc[common_assets, "p_nom_opt"].values
            logger.info(f"Updated p_nom_min for {len(common_assets)} extendable {c_name}s with build_year=0 from previous horizon")

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

        n_p.mremove( # remove assets below threshold
            c.name,
            c.df.index[
                (c.df[f"{attr}_nom_extendable"] & ~c.df.index.isin(chp_heat))
                & (c.df[f"{attr}_nom_opt"] < threshold)
            ],
        )

        # copy over assets but fix their capacity
        c.df[f"{attr}_nom"] = c.df[f"{attr}_nom_opt"]
        c.df[f"{attr}_nom_extendable"] = False

        n.import_components_from_dataframe(c.df, c.name)

        # copy time-dependent
        selection = n.component_attrs[c.name].type.str.contains(
            "series"
        ) & n.component_attrs[c.name].status.str.contains("Input")
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
            planning_horizons="2030",
            sopts="1h",
            configfile="/shared/share_cki25/energymodels/pypsa-earth/config.myopic.yaml",
            discountrate=0.071,
            demand="AB",
            h2export="10"
        )

    logger.info(f"Preparing brownfield from the file {snakemake.input.network_p}")

    year = int(snakemake.wildcards.planning_horizons)

    n = pypsa.Network(snakemake.input.network)

    # TODO
    # adjust_renewable_profiles(n, snakemake.input, snakemake.params, year)

    add_build_year_to_new_assets(n, year)

    n_p = pypsa.Network(snakemake.input.network_p)

    add_brownfield(n, n_p, year)

    # Reset capacity of current year assets to 0 and make them extendable
    # The brownfield constraint will be enforced via imported assets from previous year
    # Biomass/biogas assets: reset capacity to 0 but keep them non-extendable (resource-limited)
    for c in n.iterate_components(["Link", "Generator", "Store"]):
        attr = "e" if c.name == "Store" else "p"
        current_year_assets = c.df.index[c.df.build_year == year]
        
        if not current_year_assets.empty:
            # Identify biomass/biogas assets
            biomass_biogas_mask = c.df.carrier.str.contains("biomass|biogas", case=False, na=False)
            biomass_biogas_current = current_year_assets[biomass_biogas_mask[current_year_assets]]
            other_current = current_year_assets[~biomass_biogas_mask[current_year_assets]]
            
            # Reset biomass/biogas to 0 but keep non-extendable
            if not biomass_biogas_current.empty:
                c.df.loc[biomass_biogas_current, f"{attr}_nom"] = 0
                c.df.loc[biomass_biogas_current, f"{attr}_nom_min"] = 0
                #c.df.loc[biomass_biogas_current, f"{attr}_nom_extendable"] = False
                logger.info(f"Reset {len(biomass_biogas_current)} {c.name} biomass/biogas assets with build_year={year} to {attr}_nom=0, {attr}_nom_min=0, {attr}_nom_extendable=False")
            
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

    disable_grid_expansion_if_limit_hit(n)

    n.meta.update(dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards))))
    
    n.export_to_netcdf(snakemake.output[0])
