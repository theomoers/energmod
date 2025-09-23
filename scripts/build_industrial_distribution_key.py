# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Build industrial distribution keys from hotmaps database.
"""

import logging
import os
import uuid
from distutils.version import StrictVersion
from itertools import product

import geopandas as gpd
import pandas as pd
from _helpers import locate_bus, locate_bus_alt_clust, three_2_two_digits_country
from shapely.geometry import Point

logger = logging.getLogger(__name__)
gpd_version = StrictVersion(gpd.__version__)


def build_nodal_distribution_key(
    industrial_database, regions, industry, countries, alternative_clustering=False
):  # returns percentage of co2 emissions
    """
    Build nodal distribution keys for each sector.
    """

    # countries = regions["name"].str[:2].unique()

    keys = pd.DataFrame(index=regions.name, columns=industry, dtype=float)

    pop = pd.read_csv(
        snakemake.input.clustered_pop_layout,
        index_col=0,
        keep_default_na=False,
        na_values=[""],
    )

    gdp = pd.read_csv(
        snakemake.input.clustered_gdp_layout,
        index_col=0,
        keep_default_na=False,
        na_values=[""],
    )

    # pop["country"] = pop.index.str[:2]
    if alternative_clustering:
        # For alternative clustering (one node per country), each country node gets 1.0
        keys["population"] = 1.0
        keys["gdp"] = 1.0
    else:
        # For standard clustering, distribute proportionally within country
        pop_country = pop.index.str[:2]
        gdp_country = gdp.index.str[:2]
        
        pop_normalized = pop.groupby(pop_country)["total"].transform(lambda x: x / x.sum())
        keys["population"] = pop_normalized.values
        
        gdp_normalized = gdp.groupby(gdp_country)["total"].transform(lambda x: x / x.sum())
        keys["gdp"] = gdp_normalized.values

    for tech, country in product(industry, countries):
        if alternative_clustering:
            # For alternative clustering, only one region per country
            regions_ct = regions.name[regions.name.str.startswith(country + '.')]
            
            if len(regions_ct) == 0:
                logger.warning(f"No regions found for country {country} with pattern '{country}.'")
                logger.warning(f"Sample regions: {regions.name[:5].tolist()}")
                regions_ct = regions.name[regions.name.str.contains(country)]
                if len(regions_ct) == 0:
                    logger.warning(f"No regions found even with broader search for {country}")
                    continue
        else:
            regions_ct = regions.name[regions.name.str.contains(country)]

        if len(regions_ct) == 0:
            logger.warning(f"Skipping {tech} for {country} - no matching regions found")
            continue

        facilities = industrial_database.query(
            "country == @country and industry == @tech"
        )
        # TODO adapt for facilities with production values not emissions
        if not facilities.empty:
            if alternative_clustering:
                # For alternative clustering, assign all to the single country node
                if len(regions_ct) > 0:
                    key = pd.Series([1.0], index=regions_ct)
                else:
                    logger.warning(f"Cannot assign capacity for {tech} in {country} - no valid regions")
                    continue
            else:
                # For standard clustering, distribute based on actual capacity
                indicator = facilities["capacity"]
                if indicator.sum() == 0:
                    key = pd.Series(1 / len(facilities), facilities.index)
                else:
                    # TODO: strong assumption
                    # indicator = indicator.fillna(0)
                    key = indicator / indicator.sum()
                # For standard clustering, distribute across GADM regions
                key = (
                    key.groupby(facilities.index).sum().reindex(regions_ct, fill_value=0.0)
                )
        else:
            if alternative_clustering:
                # For alternative clustering with no facilities, still assign 1.0 to the country node
                if len(regions_ct) > 0:
                    key = pd.Series([1.0], index=regions_ct)
                else:
                    continue
            else:
                key = keys.loc[regions_ct, "gdp"]

        keys.loc[regions_ct, tech] = key
    keys["country"] = pop["ct"]
    return keys


def match_technology(df):
    industry_mapping = {
        "Integrated steelworks": "iron and steel",
        "DRI + Electric arc": "iron and steel",
        "Electric arc": "iron and steel",
        "Cement": "non-metallic minerals",
        "HVC": "chemical and petrochemical",
        "Paper": "paper pulp and print",
        "Aluminium": "non-ferrous metals",
    }

    df["industry"] = df["technology"].map(industry_mapping)
    return df


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake 

        snakemake = mock_snakemake(
            "build_industrial_distribution_key",
            simpl="",
            network="elec",
            clusters="200",
            ll="copt",
            opts="3h",
            planning_horizons="2020",
            sopts="72h",
            configfile="/shared/share_cki25/energymodels/pypsa-earth/config.myopic.yaml",
            discountrate=0.071,
            demand="AB",
            h2export="10"
        )

    regions = gpd.read_file(snakemake.input.regions_onshore)
    shapes_path = snakemake.input.shapes_path

    gadm_layer_id = snakemake.params.gadm_layer_id
    countries = snakemake.params.countries
    gadm_clustering = snakemake.params.alternative_clustering

    # countries = ["EG", "BH"]

    if regions["name"][0][
        :3
    ].isalpha():  # TODO clean later by changing all codes to 2 letters
        regions["name"] = regions["name"].apply(
            lambda name: three_2_two_digits_country(name[:3]) + name[3:]
        )

    if snakemake.params.industry_database:
        logger.info(
            "Using custom industry database from 'data/custom/industrial_database.csv' instead of default"
        )
        geo_locs = pd.read_csv(
            "data/custom/industrial_database.csv",
            sep=",",
            header=0,
            keep_default_na=False,  # , index_col=0
        )
        geo_locs["industry"] = geo_locs["technology"]
    else:
        logger.info("Using default industry database")
        geo_locs = pd.read_csv(
            snakemake.input.industrial_database,
            sep=",",
            header=0,
            keep_default_na=False,  # , index_col=0
        )
        geo_locs = geo_locs[geo_locs["country"].isin(countries)]
        geo_locs["capacity"] = pd.to_numeric(geo_locs.capacity)

        # Call the function to add the "industry" column
        df_with_industry = match_technology(geo_locs)

    geo_locs.capacity = pd.to_numeric(geo_locs.capacity)

    geo_locs = geo_locs[geo_locs.quality != "nonexistent"]

    industry = geo_locs.industry.unique()

    if gadm_clustering:
        industrial_database = locate_bus_alt_clust(
            geo_locs[geo_locs.quality != "unavailable"],
            countries,
            gadm_layer_id,
            shapes_path,
            gadm_clustering,
        ).set_index("gadm_" + str(gadm_layer_id))
    else:
        industrial_database = locate_bus(
            geo_locs[geo_locs.quality != "unavailable"],
            countries,
            gadm_layer_id,
            shapes_path,
            gadm_clustering,
        ).set_index("gadm_" + str(gadm_layer_id))

    keys = build_nodal_distribution_key(
        industrial_database, regions, industry, countries, gadm_clustering
    )

    keys.to_csv(snakemake.output.industrial_distribution_key)
