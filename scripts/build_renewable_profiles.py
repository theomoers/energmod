#!/apps/anaconda3/bin/python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# -*- coding: utf-8 -*-
"""
Calculates for each network node the (i) installable capacity (based on land-
use), (ii) the available generation time series (based on weather data), and
(iii) the average distance from the node for onshore wind, AC-connected
offshore wind, DC-connected offshore wind and solar PV generators. For hydro
generators, it calculates the expected inflows. In addition for offshore wind
it calculates the fraction of the grid connection which is under water.

Relevant settings
-----------------

.. code:: yaml

    snapshots:

    atlite:
        nprocesses:

    renewable:
        {technology}:
            cutout:
            copernicus:
                grid_codes:
                distance:
                distance_grid_codes:
            natura:
            max_depth:
            max_shore_distance:
            min_shore_distance:
            capacity_per_sqkm:
            correction_factor:
            potential:
            min_p_max_pu:
            clip_p_max_pu:
            resource:
            clip_min_inflow:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`snapshots_cf`, :ref:`atlite_cf`, :ref:`renewable_cf`

Inputs
------

- ``data/copernicus/PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif``: `Copernicus Land Service <https://land.copernicus.eu/global/products/lc>`_ inventory on 23 land use classes (e.g. forests, arable land, industrial, urban areas) based on UN-FAO classification. See `Table 4 in the PUM <https://land.copernicus.eu/global/sites/cgls.vito.be/files/products/CGLOPS1_PUM_LC100m-V3_I3.4.pdf>`_ for a list of all classes.

    .. image:: /img/copernicus.png
        :width: 33 %

- ``data/gebco/GEBCO_2021_TID.nc``: A `bathymetric <https://en.wikipedia.org/wiki/Bathymetry>`_ data set with a global terrain model for ocean and land at 15 arc-second intervals by the `General Bathymetric Chart of the Oceans (GEBCO) <https://www.gebco.net/data_and_products/gridded_bathymetry_data/>`_.

    .. image:: /img/gebco_2021_grid_image.jpg
        :width: 50 %

    **Source:** `GEBCO <https://www.gebco.net/data_and_products/images/gebco_2019_grid_image.jpg>`_

- ``resources/natura.tiff``: confer :ref:`natura`
- ``resources/offshore_shapes.geojson``: confer :ref:`shapes`
- ``resources/.geojson``: (if not offshore wind), confer :ref:`busregions`
- ``resources/regions_offshore.geojson``: (if offshore wind), :ref:`busregions`
- ``"cutouts/" + config["renewable"][{technology}]['cutout']``: :ref:`cutout`
- ``networks/base.nc``: :ref:`base`

Outputs
-------

- ``resources/profile_{technology}.nc``, except hydro technology, with the following structure

    ===================  ==========  =========================================================
    Field                Dimensions  Description
    ===================  ==========  =========================================================
    profile              bus, time   the per unit hourly availability factors for each node
    -------------------  ----------  ---------------------------------------------------------
    weight               bus         sum of the layout weighting for each node
    -------------------  ----------  ---------------------------------------------------------
    p_nom_max            bus         maximal installable capacity at the node (in MW)
    -------------------  ----------  ---------------------------------------------------------
    potential            y, x        layout of generator units at cutout grid cells inside the
                                     Voronoi cell (maximal installable capacity at each grid
                                     cell multiplied by capacity factor)
    -------------------  ----------  ---------------------------------------------------------
    average_distance     bus         average distance of units in the Voronoi cell to the
                                     grid node (in km)
    -------------------  ----------  ---------------------------------------------------------
    underwater_fraction  bus         fraction of the average connection distance which is
                                     under water (only for offshore)
    ===================  ==========  =========================================================

- ``resources/profile_hydro.nc`` for the hydro technology
    ===================  ================  ========================================================
    Field                Dimensions        Description
    ===================  ================  ========================================================
    inflow               plant, time       Inflow to the state of charge (in MW),
                                           e.g. due to river inflow in hydro reservoir.
    ===================  ================  ========================================================

    - **profile**

    .. image:: /img/profile_ts.png
        :width: 33 %
        :align: center

    - **p_nom_max**

    .. image:: /img/p_nom_max_hist.png
        :width: 33 %
        :align: center

    - **potential**

    .. image:: /img/potential_heatmap.png
        :width: 33 %
        :align: center

    - **average_distance**

    .. image:: /img/distance_hist.png
        :width: 33 %
        :align: center

    - **underwater_fraction**

    .. image:: /img/underwater_hist.png
        :width: 33 %
        :align: center

Description
-----------

This script leverages on atlite function to derivate hourly time series for an entire year
for solar, wind (onshore and offshore), and hydro data.

This script functions at two main spatial resolutions: the resolution of the
network nodes and their `Voronoi cells
<https://en.wikipedia.org/wiki/Voronoi_diagram>`_, and the resolution of the
cutout grid cells for the weather data. Typically the weather data grid is
finer than the network nodes, so we have to work out the distribution of
generators across the grid cells within each Voronoi cell. This is done by
taking account of a combination of the available land at each grid cell and the
capacity factor there.

This uses the Copernicus land use data,
Natura2000 nature reserves and GEBCO bathymetry data.

.. image:: /img/eligibility.png
    :width: 50 %
    :align: center

To compute the layout of generators in each node's Voronoi cell, the
installable potential in each grid cell is multiplied with the capacity factor
at each grid cell. This is done since we assume more generators are installed
at cells with a higher capacity factor.

.. image:: /img/offwinddc-gridcell.png
    :width: 50 %
    :align: center

.. image:: /img/offwindac-gridcell.png
    :width: 50 %
    :align: center

.. image:: /img/onwind-gridcell.png
    :width: 50 %
    :align: center

.. image:: /img/solar-gridcell.png
    :width: 50 %
    :align: center

This layout is then used to compute the generation availability time series
from the weather data cutout from ``atlite``.

Two methods are available to compute the maximal installable potential for the
node (`p_nom_max`): ``simple`` and ``conservative``:

- ``simple`` adds up the installable potentials of the individual grid cells.
  If the model comes close to this limit, then the time series may slightly
  overestimate production since it is assumed the geographical distribution is
  proportional to capacity factor.

- ``conservative`` ascertains the nodal limit by increasing capacities
  proportional to the layout until the limit of an individual grid cell is
  reached.
"""
import functools
import os
import time
from math import isnan

import atlite
import country_converter as coco
import geopandas as gpd
import numpy as np
import pandas as pd
import progressbar as pgb
import xarray as xr
from _helpers import BASE_DIR, configure_logging, create_logger
from add_electricity import load_powerplants
from dask.distributed import Client
from pypsa.geo import haversine
from shapely.geometry import LineString, Point, box

cc = coco.CountryConverter()

logger = create_logger(__name__)


COPERNICUS_CRS = "EPSG:4326"
GEBCO_CRS = "EPSG:4326"
PPL_CRS = "EPSG:4326"


def check_cutout_match(cutout, geodf):
    cutout_box = box(*cutout.bounds)
    region_box = box(*regions.total_bounds)

    assert not region_box.intersection(cutout_box).is_empty, (
        "The requested region is completely out of the cutout area.\n\r"
        "Check please the provided cutout.\n\r"
        "More details on cutout generation are available in docs:\n\r"
        "https://pypsa-earth.readthedocs.io/en/latest/tutorial.html\n\r"
    )

    if not region_box.covered_by(cutout_box):
        logger.warning(
            "Weather data does not fully cover the requester region.\n\r"
            "It's recommended to check the provided cutout.\n\r"
            "More details on cutout generation are available in docs:\n\r"
            "https://pypsa-earth.readthedocs.io/en/latest/tutorial.html"
        )


def get_eia_annual_hydro_generation(fn, countries):
    # in billion kWh/a = TWh/a
    df = pd.read_csv(fn, skiprows=1, index_col=1, na_values=[" ", "--"]).iloc[1:, 1:]
    df.index = df.index.str.strip()

    df.loc["Germany"] = df.filter(like="Germany", axis=0).sum()
    df.loc["Serbia"] += df.loc["Kosovo"]
    df = df.loc[~df.index.str.contains("Former")]
    df.drop(["World", "Germany, West", "Germany, East"], inplace=True)

    df.index = cc.convert(df.index, to="iso2")
    df.index.name = "countries"

    # Filter countries to only include those that exist in the EIA data
    available_countries = [c for c in countries if c in df.index]
    missing_countries = [c for c in countries if c not in df.index]
    
    if missing_countries:
        logger.warning(
            f"The following countries are missing from EIA hydro data and will be skipped: {missing_countries}"
        )
    
    if not available_countries:
        logger.warning("No countries found in EIA hydro data. Returning empty dataframe.")
        return pd.DataFrame()
    
    df = df.T[available_countries] * 1e6  # in MWh/a
    df.index = df.index.astype(int)

    return df


def get_hydro_capacities_annual_hydro_generation(fn, countries, year):
    hydro_stats = (
        pd.read_csv(
            fn,
            comment="#",
            keep_default_na=False,
            na_values=["-", ""],
            index_col=0,
        )
        .rename({"Country": "countries"}, axis=1)
        .set_index("countries")
    )
    hydro_prod_by_country = (
        hydro_stats[hydro_stats.index.isin(countries)][
            ["InflowHourlyAvg[GWh]"]
        ].transpose()
        * 1e3
        * 8760
    )  # change unit to MWh/y
    hydro_prod_by_country.index = pd.Index([year])

    return hydro_prod_by_country


def check_cutout_completness(cf):
    """
    Check if a cutout contains missed values.

    That may be the case due to some issues with accessibility of ERA5
    data See for details
    https://confluence.ecmwf.int/display/CUSF/Missing+data+in+ERA5T
    Returns share of cutout cells with missed data
    """
    n_missed_cells = pd.isnull(cf).sum()
    n_cells = len(np.ndarray.flatten(cf.data))
    share_missed_cells = 100 * (n_missed_cells / n_cells)
    if share_missed_cells > 0:
        logger.warning(
            f"A provided cutout contains missed data:\r\n content of {share_missed_cells:2.1f}% all cutout cells is lost"
        )
    return share_missed_cells


def estimate_bus_loss(data_column, tech):
    """
    Calculated share of buses with data loss due to flaws in the cutout data.

    Returns share of the buses with missed data
    """
    n_weights_initial = len(data_column)
    n_lost_weights = pd.isnull(data_column).sum()
    share_missed_buses = 100 * (n_lost_weights / n_weights_initial)
    if share_missed_buses >= 30:
        recommend_msg = "\r\nYou may want to re-generate the cutout"
    else:
        recommend_msg = ""

    if n_lost_weights > 0:
        logger.warning(
            f"Missed cutout cells have resulted in data loss:\r\n for {tech} {share_missed_buses:2.1f}% buses overall {recommend_msg}"
        )
    return share_missed_buses


def filter_cutout_region(cutout, regions):
    """
    Filter the cutout to focus on the region of interest.
    """
    # filter cutout regions to focus on the region of interest
    minx, miny, maxx, maxy = regions.total_bounds
    minx, maxx = max(-180.0, minx - cutout.dx), min(180.0, maxx + cutout.dx)
    miny, maxy = max(-90.0, miny - cutout.dy), min(90.0, maxy + cutout.dy)
    cutout.data = cutout.data.sel(x=slice(minx, maxx), y=slice(miny, maxy))
    return cutout


def rescale_hydro(plants, runoff, normalize_using_yearly, normalization_year, redistribute_all_countries=False):
    """
    Function used to rescale the inflows of the hydro capacities to match
    country statistics.

    Parameters
    ----------
    plants : DataFrame
        Run-of-river plants orf dams with lon, lat, countries, installed_hydro columns.
        Countries and installed_hydro column are only used with normalize_using_yearly
        installed_hydro column shall be a boolean vector specifying whether that plant
        is currently installed and used to normalize the inflows
    runoff : xarray object
        Runoff at each bus
    normalize_using_yearly : DataFrame
        Dataframe that specifies for every country the total hydro production
    normalization_year : int
        Year used for normalization
    redistribute_all_countries : bool, optional
        If True, redistributes generation from countries without plants to countries with plants
        proportionally according to their modeled runoff weights. Default is False.
    """

    if plants.empty or plants.installed_hydro.any() == False:
        logger.info("No bus has installed hydro plants, ignoring normalization.")
        return runoff

    years_statistics = normalize_using_yearly.index
    if isinstance(years_statistics, pd.DatetimeIndex):
        years_statistics = years_statistics.year
    else:
        years_statistics = years_statistics.astype(int)

    years_statistics = years_statistics.unique()

    if normalization_year not in set(years_statistics):
        logger.warning(
            f"Missing hydro statistics for year {normalization_year}; no normalization performed."
        )
    else:
        # Log initial state
        unique_plant_countries = plants.countries.unique()
        # Filter out NaN values before converting to list/set to avoid type comparison errors
        unique_plant_countries = [c for c in unique_plant_countries if pd.notna(c)]
        eia_countries = normalize_using_yearly.columns.tolist()
        plants_with_hydro = plants[plants.installed_hydro == True]
        
        logger.info(f"Hydro rescaling for year {normalization_year}:")
        logger.info(f"  Total plants: {len(plants)}")
        logger.info(f"  Plants with installed hydro: {len(plants_with_hydro)}")
        logger.info(f"  Countries with plants: {len(unique_plant_countries)} - {sorted(unique_plant_countries)}")
        logger.info(f"  Countries with EIA data: {len(eia_countries)} - {sorted(eia_countries)}")
        
        # Skip rescaling if only Germany ('DE') has EIA data
        if len(eia_countries) == 1 and eia_countries[0] == 'DE':
            logger.warning("Skipping hydro rescaling: only 'DE' has EIA data, insufficient for multi-country rescaling")
            logger.warning("Returning unscaled runoff data")
            return runoff
        
        # Identify countries with EIA data but no plants
        countries_with_eia_no_plants = set(eia_countries) - set(unique_plant_countries)
        countries_with_plants_no_eia = set(unique_plant_countries) - set(eia_countries)
        
        if countries_with_eia_no_plants:
            logger.warning(f"Countries with EIA data but no plants: {sorted(countries_with_eia_no_plants)}")
            total_missing_generation = normalize_using_yearly.loc[normalization_year, list(countries_with_eia_no_plants)].sum()
            logger.warning(f"  Total missing generation from these countries: {total_missing_generation:.2f} MWh/a")
        
        if countries_with_plants_no_eia:
            logger.warning(f"Countries with plants but no EIA data: {sorted(countries_with_plants_no_eia)}")
        # get buses that have installed hydro capacity to be used to compute
        # the normalization
        normalization_buses = plants[plants.installed_hydro == True].index

        # check nans
        share_nans = float(runoff.isnull().sum() / runoff.shape[0] / runoff.shape[1])
        if share_nans > 1e-4:
            logger.warning(
                "Share of NaN values in hydro cutout: {:.2f}%".format(100 * share_nans)
            )

        # average yearly runoff by plant
        yearlyavg_runoff_by_plant = (
            runoff.rename("runoff").mean("time", skipna=True).to_dataframe()
        ) * 8760.0

        yearlyavg_runoff_by_plant["country"] = plants.loc[
            yearlyavg_runoff_by_plant.index, "countries"
        ]

        # group runoff by country
        grouped_runoffs = (
            yearlyavg_runoff_by_plant.loc[normalization_buses].groupby("country").sum()
        )

        # common country indices
        common_countries = normalize_using_yearly.columns.intersection(
            grouped_runoffs.index
        )

        logger.info(f"  Common countries (with both EIA data and plants): {len(common_countries)} - {sorted(common_countries)}")

        # Prepare EIA data for scaling - potentially redistributed
        eia_data_for_scaling = normalize_using_yearly.loc[normalization_year].copy()
        
        if redistribute_all_countries and countries_with_eia_no_plants:
            logger.info("Redistributing generation from countries without plants to countries with plants...")
            logger.info("  Using modelled runoff weights for proportional redistribution")
            
            # Get generation from countries without plants
            missing_generation = eia_data_for_scaling[list(countries_with_eia_no_plants)].sum()
            
            # Get countries with plants that also have EIA data for proportional redistribution
            recipient_countries = list(common_countries)
            
            if recipient_countries:
                # Calculate proportions based on modelled runoff weights instead of EIA generation
                recipient_runoff = grouped_runoffs.runoff[recipient_countries]
                total_recipient_runoff = recipient_runoff.sum()
                
                if total_recipient_runoff > 0:
                    # Redistribute proportionally based on runoff weights
                    redistribution_factors = recipient_runoff / total_recipient_runoff
                    redistributed_amount = missing_generation * redistribution_factors
                    
                    logger.info(f"  Redistributing {missing_generation:.2f} MWh/a from {len(countries_with_eia_no_plants)} countries")
                    logger.info(f"  To {len(recipient_countries)} countries based on runoff weights")
                    
                    # Add redistributed generation to recipient countries
                    eia_data_for_scaling[recipient_countries] += redistributed_amount
                    
                    # Log redistribution details
                    for country in recipient_countries:
                        original = normalize_using_yearly.loc[normalization_year, country]
                        new_value = eia_data_for_scaling[country]
                        runoff_weight = recipient_runoff[country]
                        redistrib_factor = redistribution_factors[country]
                        logger.info(f"    {country}: {original:.2f} -> {new_value:.2f} MWh/a (+{new_value-original:.2f}) [runoff: {runoff_weight:.2f}, factor: {redistrib_factor:.3f}]")
                else:
                    logger.warning("Cannot redistribute: total runoff of recipient countries is zero")
            else:
                logger.warning("Cannot redistribute: no recipient countries with both plants and EIA data")

        tot_common_yearly = np.nansum(eia_data_for_scaling[common_countries])
        tot_common_runoff = np.nansum(grouped_runoffs.runoff[common_countries])
        
        logger.info(f"  Total EIA generation (common countries): {tot_common_yearly:.2f} MWh/a")
        logger.info(f"  Total simulated runoff (common countries): {tot_common_runoff:.2f} MWh/a")

        # define default_factor. When nan values, used the default 1.0
        default_factor = 1.0
        if not (
            isnan(tot_common_yearly)
            or isnan(tot_common_runoff)
            or tot_common_runoff <= 0.0
        ):
            default_factor = tot_common_yearly / tot_common_runoff

        def create_scaling_factor(
            eia_data, grouped_runoffs, c_bus, default_value=1.0
        ):
            if c_bus in eia_data.index and c_bus in grouped_runoffs.index:
                # normalization in place
                scaling_factor = np.nansum(eia_data[c_bus]) / grouped_runoffs.runoff[c_bus]
                logger.debug(f"    Scaling factor for {c_bus}: {scaling_factor:.4f} (EIA: {eia_data[c_bus]:.2f}, Runoff: {grouped_runoffs.runoff[c_bus]:.2f})")
                return scaling_factor
            elif c_bus not in eia_data.index:
                # data not available in the normalization procedure
                logger.debug(f"    No EIA data for {c_bus}, using default factor: {default_value}")
                return default_value
            elif c_bus not in grouped_runoffs.index:
                # no hydro inflows available for the country
                logger.debug(f"    No runoff data for {c_bus}, using default factor: {default_value}")
                return default_value

        unique_countries = plants.countries.unique()
        missing_countries_normalization = np.setdiff1d(
            unique_countries, eia_data_for_scaling.index
        )
        missing_countries_grouped_runoff = np.setdiff1d(
            unique_countries, grouped_runoffs.index
        )

        if missing_countries_normalization.size != 0:
            logger.warning(
                f"Missing countries in the EIA dataframe: "
                + ", ".join(missing_countries_normalization)
                + ". Default value used"
            )

        if missing_countries_grouped_runoff.size != 0:
            logger.warning(
                f"Missing installed plants in: "
                + ", ".join(missing_countries_grouped_runoff)
                + ". Default value used"
            )

        logger.info(f"  Global scaling default factor: {default_factor:.4f}")
        logger.info("  Creating scaling factors by country:")

        # matrix used to scale the runoffs
        scaling_matrix = xr.DataArray(
            [
                create_scaling_factor(
                    eia_data_for_scaling,
                    grouped_runoffs,
                    c_bus,
                    default_factor,
                )
                * np.ones(runoff.time.shape)
                for c_bus in plants.countries
            ],
            coords=dict(
                plant=plants.index.values,
                time=runoff.time.values,
            ),
        )

        # Check all buses to be in the final dataset
        missing_buses = plants.index.difference(scaling_matrix.plant)
        if len(missing_buses) > 0:
            logger.warning(f"Missing hydro inflows for buses: {missing_buses}")

        # Apply scaling
        runoff *= scaling_matrix
        
        # Log final results
        final_total_generation = (runoff.mean("time") * 8760.0).sum().values
        logger.info(f"  Final scaled total generation: {final_total_generation:.2f} MWh/a")
        
        if redistribute_all_countries and countries_with_eia_no_plants:
            original_eia_total = normalize_using_yearly.loc[normalization_year].sum()
            logger.info(f"  Original total EIA generation: {original_eia_total:.2f} MWh/a")
            logger.info(f"  Generation coverage: {(final_total_generation/original_eia_total)*100:.1f}%")

    return runoff


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_renewable_profiles",
            technology="hydro",
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
    configure_logging(snakemake)

    pgb.streams.wrap_stderr()
    countries = snakemake.params.countries
    paths = snakemake.input
    nprocesses = int(snakemake.threads)
    noprogress = not snakemake.config["enable"]["progress_bar"]
    config = snakemake.params.renewable[snakemake.wildcards.technology]
    resource = config["resource"]
    correction_factor = config.get("correction_factor", 1.0)
    p_nom_max_meth = config.get("potential", "conservative")

    # crs
    geo_crs = snakemake.params.crs["geo_crs"]
    area_crs = snakemake.params.crs["area_crs"]

    if isinstance(config.get("copernicus", {}), list):
        config["copernicus"] = {"grid_codes": config["copernicus"]}

    if correction_factor != 1.0:
        logger.info(f"correction_factor is set as {correction_factor}")
    regions = gpd.read_file(paths.regions)  # .set_index("name").rename_axis("bus")

    assert not regions.empty, (
        f"List of regions in {snakemake.input.regions} is empty, please "
        "disable the corresponding renewable technology"
    )

    # Remove faulty offshore shapes before availability matrix calculation
    if snakemake.wildcards.technology.startswith("offwind") and snakemake.params.global_specific.get("remove_faulty_offshore_shapes", False):
        logger.info(f"Original regions length: {len(regions)}")
        to_remove = [545, 1343]
        logger.info(f"Removing faulty offshore shapes with indices: {to_remove}. Not relevant for energy model (polar regions)")
        logger.info(f"Removing regions: {regions.iloc[to_remove][['name', 'geometry']]}")
        regions = regions.drop(to_remove).reset_index(drop=True)
        logger.info(f"New regions length: {len(regions)}")

    # do not pull up, set_index does not work if geo dataframe is empty
    regions = regions.set_index("name").rename_axis("bus")

    if nprocesses > 1:
        client = Client(n_workers=nprocesses, threads_per_worker=1)
    else:
        client = None

    cutout = atlite.Cutout(paths["cutout"])

    check_cutout_match(cutout=cutout, geodf=regions)

    if not snakemake.wildcards.technology.startswith("hydro"):
        # the region should be restricted for non-hydro technologies, as the hydro potential is calculated across hydrobasins which may span beyond the region of the country
        cutout = filter_cutout_region(cutout, regions)

    buses = regions.index

    func = getattr(cutout, resource.pop("method"))
    resource["dask_kwargs"] = {"scheduler": client}

    # filter plants for hydro
    if snakemake.wildcards.technology.startswith("hydro"):
        country_shapes = gpd.read_file(paths.country_shapes)
        hydrobasins_path = os.path.join(BASE_DIR, resource["hydrobasins"])
        resource["hydrobasins"] = hydrobasins_path
        hydrobasins = gpd.read_file(hydrobasins_path)
        ppls = load_powerplants("/shared/share_cki25/energymodels/pypsa-earth/permstorage/110nodes_1h_7ph/resources/powerplants.csv") # ppls = load_powerplants(snakemake.input.powerplants)

        all_hydro_ppls = ppls[ppls.carrier == "hydro"]

        # select hydro units within hydrobasins
        hgdf = gpd.GeoDataFrame(
            all_hydro_ppls,
            index=all_hydro_ppls.index,
            geometry=gpd.points_from_xy(all_hydro_ppls.lon, all_hydro_ppls.lat),
            crs=PPL_CRS,
        ).to_crs(hydrobasins.crs)
        temp_gdf = gpd.sjoin(hgdf, hydrobasins, predicate="within", how="left")

        hydro_ppls = pd.DataFrame(
            hgdf.loc[temp_gdf.index_right.dropna().index].drop(columns="geometry")
        )

        bus_notin_hydrobasins = list(
            set(all_hydro_ppls.index).difference(set(hydro_ppls.index))
        )

        resource["plants"] = hydro_ppls.rename(columns={"country": "countries"})[
            ["lon", "lat", "countries"]
        ]

        # TODO: possibly revise to account for non-existent hydro powerplants
        resource["plants"]["installed_hydro"] = [
            True for bus_id in resource["plants"].index
        ]

        # get normalization before executing runoff
        normalization = None
        if ("normalization" in config) and isinstance(config["normalization"], dict):
            normalization = config.pop("normalization")

        # check if there are hydro powerplants
        if resource["plants"].empty:
            # when no powerplants are available save an empty file
            xr.DataArray(
                dims=["plant", "time"], coords={"plant": []}, name="inflow"
            ).to_netcdf(snakemake.output.profile)
        else:
            # otherwise perform the calculations
            inflow = correction_factor * func(capacity_factor=True, **resource)

            if "clip_min_inflow" in config:
                inflow = inflow.where(inflow >= config["clip_min_inflow"], 0)

            # check if normalization field belongs to the settings and it is not false
            if normalization:
                method = normalization["method"]
                norm_year = normalization.get("year", int(inflow.time[0].dt.year))
                if method == "hydro_capacities":
                    path_hydro_capacities = snakemake.input.hydro_capacities
                    normalize_using_yearly = (
                        get_hydro_capacities_annual_hydro_generation(
                            path_hydro_capacities, countries, norm_year
                        )
                    )

                elif method == "eia":
                    path_eia_stats = snakemake.input.eia_hydro_generation
                    normalize_using_yearly = get_eia_annual_hydro_generation(
                        path_eia_stats, countries
                    )

                redistribute_hydro = snakemake.params.get("redistribute_hydro_over_countries", False)
                
                inflow = rescale_hydro(
                    resource["plants"], inflow, normalize_using_yearly, norm_year, 
                    redistribute_hydro
                )
                logger.info(
                    f"Hydro normalization method '{method}' on year-statistics {norm_year}"
                )
            else:
                logger.info("No hydro normalization")

            inflow *= config.get("multiplier", 1.0)

            # add zero values for out of hydrobasins elements
            if len(bus_notin_hydrobasins) > 0:
                regions_notin = all_hydro_ppls.loc[
                    bus_notin_hydrobasins, ["lon", "lat", "country"]
                ]
                logger.warning(
                    f"Buses {bus_notin_hydrobasins} are not contained into hydrobasins."
                    f"Setting empty time series. Bus location:\n{regions_notin}"
                )

                # initialize empty DataArray and append to inflow
                notin_data = xr.DataArray(
                    np.zeros(
                        (len(bus_notin_hydrobasins), inflow.coords["time"].shape[0])
                    ),
                    dims=("plant", "time"),
                    coords=dict(
                        plant=bus_notin_hydrobasins,
                        time=inflow.coords["time"],
                    ),
                )
                inflow = xr.concat([inflow, notin_data], dim="plant")

            inflow.rename("inflow").to_netcdf(snakemake.output.profile)
    else:
        capacity_per_sqkm = config["capacity_per_sqkm"]

        excluder = atlite.ExclusionContainer(crs=area_crs, res=100)

        if "natura" in config and config["natura"]:
            excluder.add_raster(paths.natura, nodata=0, allow_no_overlap=True)

        if "copernicus" in config and config["copernicus"]:
            copernicus = config["copernicus"]
            excluder.add_raster(
                paths.copernicus,
                codes=copernicus["grid_codes"],
                invert=True,
                crs=COPERNICUS_CRS,
            )
            if "distance" in copernicus and config["copernicus"]["distance"] > 0:
                excluder.add_raster(
                    paths.copernicus,
                    codes=copernicus["distance_grid_codes"],
                    buffer=copernicus["distance"],
                    crs=COPERNICUS_CRS,
                )

        if "max_depth" in config:
            # lambda not supported for atlite + multiprocessing
            # use named function np.greater with partially frozen argument instead
            # and exclude areas where: -max_depth > grid cell depth
            func_depth = functools.partial(np.greater, -config["max_depth"])
            excluder.add_raster(
                paths.gebco, codes=func_depth, crs=GEBCO_CRS, nodata=-1000
            )

        if "min_shore_distance" in config:
            buffer = config["min_shore_distance"]
            excluder.add_geometry(paths.country_shapes, buffer=buffer)

        if "max_shore_distance" in config:
            buffer = config["max_shore_distance"]
            excluder.add_geometry(paths.country_shapes, buffer=buffer, invert=True)

        kwargs = dict(nprocesses=nprocesses, disable_progressbar=noprogress)
        if noprogress:
            logger.info("Calculate landuse availabilities...")
            start = time.time()
            availability = cutout.availabilitymatrix(regions, excluder, **kwargs)

            duration = time.time() - start
            logger.info(f"Completed availability calculation ({duration:2.2f}s)")
        else:
            availability = cutout.availabilitymatrix(regions, excluder, **kwargs)
        area = cutout.grid.to_crs(area_crs).area / 1e6
        area = xr.DataArray(
            area.values.reshape(cutout.shape), [cutout.coords["y"], cutout.coords["x"]]
        )

        potential = capacity_per_sqkm * availability.sum("bus") * area

        capacity_factor = correction_factor * func(capacity_factor=True, **resource)
        layout = capacity_factor * area * capacity_per_sqkm

        n_cells_lost = check_cutout_completness(capacity_factor)

        profile, capacities = func(
            matrix=availability.stack(spatial=["y", "x"]),
            layout=layout,
            index=buses,
            per_unit=True,
            return_capacity=True,
            **resource,
        )

        logger.info(f"Calculating maximal capacity per bus (method '{p_nom_max_meth}')")
        if p_nom_max_meth == "simple":
            p_nom_max = capacity_per_sqkm * availability @ area
        elif p_nom_max_meth == "conservative":
            max_cap_factor = capacity_factor.where(availability != 0).max(["x", "y"])
            p_nom_max = capacities / max_cap_factor
        else:
            raise AssertionError(
                'Config key `potential` should be one of "simple" '
                f'(default) or "conservative", not "{p_nom_max_meth}"'
            )

        logger.info("Calculate average distances.")
        layoutmatrix = (layout * availability).stack(spatial=["y", "x"])

        coords = cutout.grid[["x", "y"]]
        bus_coords = regions[["x", "y"]]

        average_distance = []
        centre_of_mass = []
        for bus in buses:
            row = layoutmatrix.sel(bus=bus).data
            nz_b = (row != 0) & (~pd.isnull(row))
            row = row[nz_b]
            co = coords[nz_b]
            distances = haversine(bus_coords.loc[bus], co)
            average_distance.append((distances * (row / row.sum())).sum())
            centre_of_mass.append(co.values.T @ (row / row.sum()))

        average_distance = xr.DataArray(average_distance, [buses])
        centre_of_mass = xr.DataArray(centre_of_mass, [buses, ("spatial", ["x", "y"])])

        ds = xr.merge(
            [
                (correction_factor * profile).rename("profile"),
                capacities.rename("weight"),
                p_nom_max.rename("p_nom_max"),
                potential.rename("potential"),
                average_distance.rename("average_distance"),
            ]
        )

        if n_cells_lost > 0:
            estimate_bus_loss(
                data_column=ds.weight, tech=snakemake.wildcards.technology
            )

        if snakemake.wildcards.technology.startswith("offwind"):
            logger.info("Calculate underwater fraction of connections.")
            offshore_shape = gpd.read_file(paths["offshore_shapes"]).union_all()
            underwater_fraction = []
            for bus in buses:
                p = centre_of_mass.sel(bus=bus).data
                line = LineString([tuple(p), tuple(regions.loc[bus, ["x", "y"]])])
                frac = line.intersection(offshore_shape).length / line.length
                underwater_fraction.append(frac)

            ds["underwater_fraction"] = xr.DataArray(underwater_fraction, [buses])

        # select only buses with some capacity and minimal capacity factor
        ds = ds.sel(
            bus=(
                (ds["profile"].mean("time") >= config.get("min_p_max_pu", 0.0))
                & (ds["p_nom_max"] >= config.get("min_p_nom_max", 0.0))
            )
        )

        if "clip_p_max_pu" in config:
            min_p_max_pu = config["clip_p_max_pu"]
            ds["profile"] = ds["profile"].where(ds["profile"] >= min_p_max_pu, 0)

        ds.to_netcdf(snakemake.output.profile)

    if client is not None:
        client.shutdown()
