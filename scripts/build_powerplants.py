# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# -*- coding: utf-8 -*-
"""
Retrieves conventional powerplant capacities and locations from `powerplantmatching <https://github.com/FRESNA/powerplantmatching>`_, assigns these to buses and creates a ``.csv`` file. It is possible to amend the powerplant database with custom entries provided in ``data/custom_powerplants.csv``.

Relevant Settings
-----------------

.. code:: yaml

    electricity:
      powerplants_filter:
      custom_powerplants:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`electricity`

Inputs
------

- ``networks/base.nc``: confer :ref:`base`.
- ``data/custom_powerplants.csv``: custom powerplants in the same format as `powerplantmatching <https://github.com/FRESNA/powerplantmatching>`_ provides or as OSM extractor generates

Outputs
-------

- ``resource/powerplants.csv``: A list of conventional power plants (i.e. neither wind nor solar) with fields for name, fuel type, technology, country, capacity in MW, duration, commissioning year, retrofit year, latitude, longitude, and dam information as documented in the `powerplantmatching README <https://github.com/FRESNA/powerplantmatching/blob/master/README.md>`_; additionally it includes information on the closest substation/bus in ``networks/base.nc``.

    .. image:: /img/powerplantmatching.png
        :width: 30 %

    **Source:** `powerplantmatching on GitHub <https://github.com/FRESNA/powerplantmatching>`_

Description
-----------

The configuration options ``electricity: powerplants_filter`` and ``electricity: custom_powerplants`` can be used to control whether data should be retrieved from the original powerplants database or from custom amendments. These specify `pandas.query <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html>`_ commands.

1. Adding all powerplants from custom:

    .. code:: yaml

        powerplants_filter: false
        custom_powerplants: true

2. Replacing powerplants in e.g. Germany by custom data:

    .. code:: yaml

        powerplants_filter: Country not in ['Germany']
        custom_powerplants: true

    or

    .. code:: yaml

        powerplants_filter: Country not in ['Germany']
        custom_powerplants: Country in ['Germany']

3. Adding additional built year constraints:

    .. code:: yaml

        powerplants_filter: Country not in ['Germany'] and YearCommissioned <= 2015
        custom_powerplants: YearCommissioned <= 2015

Format required for the custom_powerplants.csv should be similar to the powerplantmatching format with some additional considerations:

Columns required: [id, Name, Fueltype, Technology, Set, Country, Capacity, Efficiency, DateIn, DateRetrofit, DateOut, lat, lon, Duration, Volume_Mm3, DamHeight_m, StorageCapacity_MWh, EIC, projectID]

Tagging considerations for columns in the file:

- FuelType: 'Natural Gas' has to be tagged either as 'OCGT', 'CCGT'
- Technology: 'Reservoir' has to be set as 'ror' if hydro powerplants are to be considered as 'Generators' and not 'StorageUnits'
- Country:  Country name has to be defined with its alpha2 code ('NG' for Nigeria,'BO' for Bolivia, 'FR' for France, etc.

The following assumptions were done to map custom OSM-extracted power plants with powerplantmatching format.

1. The benchmark PPM keys values were taken as follows:
        'Fueltype': ['Hydro', 'Hard Coal', 'Natural Gas', 'Lignite', 'Nuclear', 'Oil', 'Bioenergy'
            'Wind', 'Geothermal', 'Solar', 'Waste', 'Other']

        'Technology': ['Reservoir', 'Pumped Storage', 'Run-Of-River', 'Steam Turbine', 'CCGT', 'OCGT'
            'Pv', 'CCGT, Thermal', 'Offshore', 'Storage Technologies']

        'Set': ['Store', 'PP', 'CHP']

2. OSM-extracted features were mapped into PPM ones using a (quite arbitrary) set of rules:
        'coal': 'Hard Coal'
        'wind_turbine': 'Onshore',
        'horizontal_axis' : 'Onshore',
        'vertical_axis' : 'Offhore',
        'nuclear': 'Steam Turbine'
3. All hydro OSM-extracted objects were interpreted as generation technologies, although ["Run-Of-River", "Pumped Storage", "Reservoir"] in PPM can belong to 'Storage Technologies', too.
4. OSM extraction was supposed to be ignoring non-generation features like CHP and Natural Gas storage (in contrast to PPM).
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import powerplantmatching as pm
import pypsa
import yaml
from _helpers import (
    configure_logging,
    create_logger,
    country_name_2_two_digits,
    locate_bus,
    locate_bus_alt_clust,
    read_csv_nafix,
    to_csv_nafix,
    two_digits_2_name_country,
)
from scipy.spatial import cKDTree as KDTree
from shapely.geometry import Point

logger = create_logger(__name__)


FOSSIL_CAPACITY_FUELTYPE_MAP = {
    "Hard Coal": "coal",
    "Lignite": "coal",
    "Coal": "coal",
    "CCGT": "gas",
    "OCGT": "gas",
    "Natural Gas": "gas",
    "Oil": "oil",
}


def _repo_path(path):
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(os.getcwd(), path)


def _parse_tracker_year(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    return pd.to_numeric(text[:4], errors="coerce")


def _tracker_operating_mask(df, year):
    status = df["Status"].astype(str).str.strip().str.lower()
    start_year = df["Start year"].apply(_parse_tracker_year)
    retired_year = df["Retired year"].apply(_parse_tracker_year)
    return (
        status.eq("operating")
        & start_year.fillna(-np.inf).le(year)
        & (retired_year.isna() | retired_year.gt(year))
    )


def _gem_country_to_iso2(country):
    country = str(country).strip()
    if not country:
        return np.nan
    special = {
        "Kosovo": "XK",
        "Türkiye": "TR",
        "Turkey": "TR",
        "Russia": "RU",
        "Vietnam": "VN",
        "Iran": "IR",
        "Syria": "SY",
        "Laos": "LA",
        "Bolivia": "BO",
        "Venezuela": "VE",
        "Tanzania": "TZ",
        "Moldova": "MD",
        "South Korea": "KR",
        "North Korea": "KP",
        "Taiwan": "TW",
    }
    if country in special:
        return special[country]
    iso2 = country_name_2_two_digits(country)
    if not isinstance(iso2, str) or iso2.upper() in {"NOT FOUND", "NAN"}:
        return np.nan
    return iso2.upper()


def _load_gem_coal_reference(path, year):
    coal = pd.read_csv(path, encoding="utf-8-sig")
    required = {"Country/Area", "Capacity (MW)", "Status", "Start year", "Retired year"}
    missing = required - set(coal.columns)
    if missing:
        raise ValueError(f"Missing required GEM coal columns in {path}: {sorted(missing)}")

    coal = coal.loc[_tracker_operating_mask(coal, year)].copy()
    country_lookup = {c: _gem_country_to_iso2(c) for c in coal["Country/Area"].dropna().unique()}
    coal["country"] = coal["Country/Area"].map(country_lookup)
    coal["reference_mw"] = pd.to_numeric(
        coal["Capacity (MW)"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    ).fillna(0.0)
    coal["validation_tech"] = "coal"
    return (
        coal.loc[coal["country"].notna()]
        .groupby(["country", "validation_tech"], as_index=False)["reference_mw"]
        .sum()
    )


def _load_gem_oil_gas_reference(path, year):
    gogpt = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "Country/Area",
        "Fuel",
        "Fuel classification?",
        "Capacity (MW)",
        "Status",
        "Start year",
        "Retired year",
    }
    missing = required - set(gogpt.columns)
    if missing:
        raise ValueError(
            f"Missing required GEM oil/gas columns in {path}: {sorted(missing)}"
        )

    gogpt = gogpt.loc[_tracker_operating_mask(gogpt, year)].copy()
    classification = gogpt["Fuel classification?"].astype(str).str.lower()
    fuel = gogpt["Fuel"].astype(str).str.lower()
    gas = classification.str.contains("gas", na=False) | fuel.str.contains(
        "fossil gas|natural gas|lng", na=False, regex=True
    )
    # Dual gas/liquid units are counted as gas to avoid double-counting.
    oil = (
        classification.str.contains("oil only|liquid only", na=False)
        | fuel.str.contains("diesel|fuel oil", na=False, regex=True)
    ) & ~gas

    gogpt["validation_tech"] = pd.Series(pd.NA, index=gogpt.index, dtype="object")
    gogpt.loc[gas, "validation_tech"] = "gas"
    gogpt.loc[oil, "validation_tech"] = "oil"
    gogpt = gogpt.loc[gogpt["validation_tech"].notna()].copy()
    country_lookup = {c: _gem_country_to_iso2(c) for c in gogpt["Country/Area"].dropna().unique()}
    gogpt["country"] = gogpt["Country/Area"].map(country_lookup)
    gogpt["reference_mw"] = pd.to_numeric(
        gogpt["Capacity (MW)"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    ).fillna(0.0)
    return (
        gogpt.loc[gogpt["country"].notna()]
        .groupby(["country", "validation_tech"], as_index=False)["reference_mw"]
        .sum()
    )


def _load_fossil_capacity_reference(config):
    year = int(config.get("year", 2020))
    coal_path = _repo_path(config.get("coal_csv"))
    oil_gas_path = _repo_path(config.get("oil_gas_csv"))
    if not coal_path or not oil_gas_path:
        raise ValueError("fossil_capacity_alignment requires coal_csv and oil_gas_csv")

    reference = pd.concat(
        [
            _load_gem_coal_reference(coal_path, year),
            _load_gem_oil_gas_reference(oil_gas_path, year),
        ],
        ignore_index=True,
    )
    carriers = set(config.get("carriers", ["coal", "gas", "oil"]))
    reference = reference.loc[reference["validation_tech"].isin(carriers)].copy()
    return (
        reference.groupby(["country", "validation_tech"], as_index=False)["reference_mw"]
        .sum()
        .assign(year=year)
    )


def _powerplant_active_mask(ppl, year):
    date_in = pd.to_numeric(ppl.get("DateIn", np.nan), errors="coerce")
    date_out = pd.to_numeric(ppl.get("DateOut", np.nan), errors="coerce")
    return date_in.fillna(-np.inf).le(year) & (date_out.isna() | date_out.ge(year))


def align_fossil_powerplant_capacities(ppl, config):
    alignment = config.get("global_specific", {}).get("fossil_capacity_alignment", {})
    if not alignment or not alignment.get("enable", False):
        return ppl

    year = int(alignment.get("year", 2020))
    carriers = set(alignment.get("carriers", ["coal", "gas", "oil"]))
    min_reference_mw = float(alignment.get("min_reference_mw", 1.0))
    min_model_mw = float(alignment.get("min_model_mw", 1.0))
    scale_min = alignment.get("scale_min", 0.0)
    scale_max = alignment.get("scale_max", None)
    scale_min = None if scale_min is None else float(scale_min)
    scale_max = None if scale_max is None else float(scale_max)
    scale_down_existing = bool(alignment.get("scale_down_existing", True))
    scale_down_existing_carriers = set(
        str(c) for c in alignment.get("scale_down_existing_carriers", [])
    )

    scaled = ppl.copy()
    scaled["Capacity"] = pd.to_numeric(scaled["Capacity"], errors="coerce").fillna(0.0)
    scaled["Country"] = scaled["Country"].astype(str).str.upper().str.strip()
    scaled["validation_tech"] = scaled["Fueltype"].map(FOSSIL_CAPACITY_FUELTYPE_MAP)
    scaled["is_active_alignment_year"] = _powerplant_active_mask(scaled, year)

    active = scaled.loc[
        scaled["is_active_alignment_year"] & scaled["validation_tech"].isin(carriers)
    ].copy()
    model = (
        active.groupby(["Country", "validation_tech"], as_index=False)["Capacity"]
        .sum()
        .rename(columns={"Country": "country", "Capacity": "model_before_mw"})
    )

    reference = _load_fossil_capacity_reference(alignment)
    reference = reference.loc[reference["reference_mw"].ge(min_reference_mw)].copy()

    audit = model.merge(
        reference[["country", "validation_tech", "reference_mw"]],
        on=["country", "validation_tech"],
        how="outer",
    )
    audit["model_before_mw"] = audit["model_before_mw"].fillna(0.0)
    audit["reference_mw"] = audit["reference_mw"].fillna(0.0)
    audit["raw_factor"] = np.where(
        audit["model_before_mw"].ge(min_model_mw),
        audit["reference_mw"] / audit["model_before_mw"],
        np.nan,
    )
    audit["factor"] = audit["raw_factor"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    if scale_min is not None:
        audit["factor"] = audit["factor"].clip(lower=scale_min)
    if scale_max is not None:
        audit["factor"] = audit["factor"].clip(upper=scale_max)

    has_model = audit["model_before_mw"].ge(min_model_mw)
    has_reference = audit["reference_mw"].ge(min_reference_mw)
    would_downscale = has_model & has_reference & audit["raw_factor"].lt(1.0)
    allow_downscale = scale_down_existing | audit["validation_tech"].isin(scale_down_existing_carriers)
    preserved_downscale = would_downscale & ~allow_downscale
    audit.loc[preserved_downscale, "factor"] = 1.0

    audit["policy"] = np.where(
        ~has_model,
        "skip_missing_model_capacity",
        np.where(
            ~has_reference,
            "skip_missing_reference",
            np.where(
                preserved_downscale,
                "kept_existing_above_reference",
                np.where(audit["factor"].eq(1.0), "unchanged", "scaled"),
            ),
        ),
    )

    factor_lookup = audit.set_index(["country", "validation_tech"])["factor"].to_dict()
    factors = pd.Series(
        [
            factor_lookup.get((country, tech), 1.0)
            for country, tech in zip(scaled["Country"], scaled["validation_tech"])
        ],
        index=scaled.index,
        dtype=float,
    )
    do_scale = scaled["is_active_alignment_year"] & scaled["validation_tech"].isin(carriers)
    scaled.loc[do_scale, "Capacity"] = scaled.loc[do_scale, "Capacity"] * factors.loc[do_scale]

    after = (
        scaled.loc[scaled["is_active_alignment_year"] & scaled["validation_tech"].isin(carriers)]
        .groupby(["Country", "validation_tech"], as_index=False)["Capacity"]
        .sum()
        .rename(columns={"Country": "country", "Capacity": "model_after_mw"})
    )
    audit = audit.merge(after, on=["country", "validation_tech"], how="left")
    audit["model_after_mw"] = audit["model_after_mw"].fillna(0.0)
    audit["year"] = year
    audit["pct_error_after"] = np.where(
        audit["reference_mw"].abs().gt(0),
        100.0 * (audit["model_after_mw"] - audit["reference_mw"]) / audit["reference_mw"],
        np.nan,
    )
    audit = audit.sort_values(["validation_tech", "country"])

    audit_dir = alignment.get("audit_dir")
    if audit_dir:
        audit_dir = _repo_path(audit_dir)
        os.makedirs(audit_dir, exist_ok=True)
        audit_path = os.path.join(
            audit_dir,
            f"fossil_powerplant_capacity_alignment_{year}.csv",
        )
        audit.to_csv(audit_path, index=False)
        logger.info("Wrote fossil powerplant capacity alignment audit to %s", audit_path)

    scaled_count = int((audit["policy"] == "scaled").sum())
    preserved_count = int((audit["policy"] == "kept_existing_above_reference").sum())
    biggest = audit.loc[audit["policy"] == "scaled"].copy()
    biggest["distance_from_one"] = (biggest["factor"] - 1.0).abs()
    logger.info(
        "Aligned fossil powerplant capacities for %s country/fuel groups in %s; preserved %s groups above reference",
        scaled_count,
        year,
        preserved_count,
    )
    if not biggest.empty:
        logger.info(
            "Largest fossil capacity scale factors:\n%s",
            biggest.sort_values("distance_from_one", ascending=False)
            .head(10)[["country", "validation_tech", "model_before_mw", "reference_mw", "factor"]]
            .round(3)
            .to_string(index=False),
        )

    return scaled.drop(columns=["validation_tech", "is_active_alignment_year"])


def convert_osm_to_pm(filepath_ppl_osm, filepath_ppl_pm):
    if os.stat(filepath_ppl_osm).st_size == 0:
        return to_csv_nafix(pd.DataFrame(), filepath_ppl_pm, index=False)

    add_ppls = read_csv_nafix(filepath_ppl_osm, index_col=0, dtype={"bus": "str"})

    custom_ppls_coords = gpd.GeoSeries.from_wkt(add_ppls["geometry"])
    add_ppls = (
        add_ppls.rename(
            columns={
                "name": "Name",
                "tags.generator:source": "Fueltype",
                "tags.generator:type": "Technology",
                "tags.power": "Set",
                "power_output_MW": "Capacity",
            }
        )
        .replace(
            dict(
                Fueltype={
                    "nuclear": "Nuclear",
                    "wind": "Wind",
                    "hydro": "Hydro",
                    "tidal": "Other",
                    "wave": "Other",
                    "geothermal": "Geothermal",
                    "solar": "Solar",
                    # "Hard Coal" follows defaults of PPM
                    "coal": "Hard Coal",
                    "gas": "Natural Gas",
                    "biomass": "Bioenergy",
                    "biofuel": "Bioenergy",
                    "biogas": "Bioenergy",
                    "oil": "Oil",
                    "diesel": "Oil",
                    "gasoline": "Oil",
                    "waste": "Waste",
                    "osmotic": "Other",
                    "wave": "Other",
                    # approximation
                    # TODO: this shall be improved, one entry shall be Oil and the otherone gas
                    "gas;oil": "Oil",
                    "steam": "Natural Gas",
                    "waste_heat": "Other",
                },
                Technology={
                    "combined_cycle": "CCGT",
                    "gas_turbine": "OCGT",
                    "steam_turbine": "Steam Turbine",
                    "reciprocating_engine": "Combustion Engine",
                    # a very strong assumption
                    "wind_turbine": "Onshore",
                    "horizontal_axis": "Onshore",
                    "vertical_axis": "Offhore",
                    "solar_photovoltaic_panel": "Pv",
                },
                Set={"generator": "PP", "plant": "PP"},
            )
        )
        .assign(
            Country=lambda df: df.Country.map(two_digits_2_name_country),
            # Name=lambda df: "OSM_"
            # + df.Country.astype(str)
            # + "_"
            # + df.id.astype(str)
            # + "-"
            # + df.Name.astype(str),
            Efficiency="",
            Duration="",
            Volume_Mm3="",
            DamHeight_m="",
            StorageCapacity_MWh="",
            DateIn="",
            DateRetrofit="",
            DateMothball="",
            DateOut="",
            lat=custom_ppls_coords.y,
            lon=custom_ppls_coords.x,
            EIC=lambda df: df.id,
            projectID=lambda df: "OSM" + df.id.astype(str),
        )
        .dropna(subset=["Fueltype"])
    )

    # All Hydro objects can be interpreted by PPM as Storages, too
    # However, everything extracted from OSM seems to belong
    # to power plants with "tags.power" == "generator" only
    osm_ppm_df = pd.DataFrame(
        data={
            "osm_method": ["run-of-the-river", "water-pumped-storage", "water-storage"],
            "ppm_technology": ["Run-Of-River", "Pumped Storage", "Reservoir"],
        }
    )
    for i in osm_ppm_df.index:
        add_ppls.loc[
            add_ppls["tags.generator:method"] == osm_ppm_df.loc[i, "osm_method"],
            "Technology",
        ] = osm_ppm_df.loc[i, "ppm_technology"]

    # originates from osm::"tags.generator:source"
    add_ppls.loc[add_ppls["Fueltype"] == "Nuclear", "Technology"] = "Steam Turbine"

    # PMM contains data on NG, batteries and hydro storages
    # trying to catch some of them...
    # originates from osm::"tags.generator:source"
    add_ppls.loc[add_ppls["Fueltype"] == "battery", "Set"] = "Store"
    # originates from osm::tags.generator:type
    add_ppls.loc[add_ppls["Technology"] == "battery storage", "Set"] = "Store"

    add_ppls = add_ppls.replace(dict(Fueltype={"battery": "Other"})).drop(
        columns=["tags.generator:method", "geometry", "Area", "id"],
        errors="ignore",
    )

    to_csv_nafix(add_ppls, filepath_ppl_pm, index=False)

    return add_ppls


def add_custom_powerplants(ppl, inputs, config):
    if "custom_powerplants" not in config["electricity"]:
        return ppl

    custom_ppl_query = config["electricity"]["custom_powerplants"]
    if not custom_ppl_query:
        return ppl
    add_ppls = read_csv_nafix(
        inputs.custom_powerplants, index_col=0, dtype={"bus": "str"}
    )

    if custom_ppl_query == "merge":
        return pd.concat(
            [ppl, add_ppls], sort=False, ignore_index=True, verify_integrity=True
        )
    elif custom_ppl_query == "replace":
        return add_ppls


def replace_natural_gas_technology(df: pd.DataFrame):
    """
    Maps and replaces gas technologies in the powerplants.csv onto model
    compliant carriers.
    """
    mapping = {
        "Steam Turbine": "CCGT",
        "Combustion Engine": "OCGT",
        "NG": "CCGT",
        "Ng": "CCGT",
        "NG/FO": "OCGT",
        "Ng/Fo": "OCGT",
        "NG/D": "OCGT",
        "LNG": "OCGT",
        "CCGT/D": "CCGT",
        "CCGT/FO": "CCGT",
        "LCCGT": "CCGT",
        "CCGT/Fo": "CCGT",
    }
    fueltype = df["Fueltype"] == "Natural Gas"
    df.loc[fueltype, "Technology"] = (
        df.loc[fueltype, "Technology"].replace(mapping).fillna("CCGT")
    )
    unique_tech_with_ng = df.loc[fueltype, "Technology"].unique()
    unknown_techs = np.setdiff1d(unique_tech_with_ng, ["CCGT", "OCGT"])
    if len(unknown_techs) > 0:
        df.loc[fueltype, "Technology"] = df.loc[fueltype, "Technology"].replace(
            {t: "CCGT" for t in unknown_techs}
        )
    df["Fueltype"] = np.where(fueltype, df["Technology"], df["Fueltype"])
    return df


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake
        
        snakemake = mock_snakemake(
            "build_powerplants",
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

    with open(snakemake.input.pm_config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    filepath_osm_ppl = snakemake.input.osm_powerplants
    filepath_osm2pm_ppl = snakemake.output.powerplants_osm2pm

    n = pypsa.Network(snakemake.input.base_network)
    countries_codes = n.buses.country.unique()
    countries_names = list(map(two_digits_2_name_country, countries_codes))

    config["target_countries"] = countries_names

    if (
        "EXTERNAL_DATABASE"
        in config["matching_sources"] + config["fully_included_sources"]
    ):
        if "EXTERNAL_DATABASE" not in config:
            logger.error(
                "Missing configuration EXTERNAL_DATABASE in powerplantmatching config yaml\n\t"
                "Please check file configs/powerplantmatching_config.yaml"
            )
        logger.info("Parsing OSM generator data to powerplantmatching format")
        config["EXTERNAL_DATABASE"]["fn"] = os.path.join(
            os.getcwd(), filepath_osm2pm_ppl
        )
    else:
        # create an empty file
        with open(filepath_osm2pm_ppl, "w"):
            pass

    # specify the main query for filtering powerplants
    ppl_query = snakemake.params.powerplants_filter
    if isinstance(ppl_query, str):
        config["main_query"] = ppl_query
    else:
        config["main_query"] = ""

    if snakemake.config["electricity"]["custom_powerplants"] != "replace":
        ppl = (
            pm.powerplants(from_url=False, update=True, config_update=config)
            .powerplant.fill_missing_decommissioning_years()
            .query('Fueltype not in ["Solar", "Wind"] and Country in @countries_names')
            .powerplant.convert_country_to_alpha2()
            .pipe(replace_natural_gas_technology)
        )
    else:
        ppl = pd.DataFrame()

    ppl = add_custom_powerplants(ppl, snakemake.input, snakemake.config).query(
        ppl_query
    )  # add carriers from own powerplant files

    ppl = ppl.reset_index(drop=True)
    ppl = align_fossil_powerplant_capacities(ppl, snakemake.config)

    cntries_without_ppl = [c for c in countries_codes if c not in ppl.Country.unique()]

    for c in countries_codes:
        substation_i = n.buses.query("substation_lv and country == @c").index
        kdtree = KDTree(n.buses.loc[substation_i, ["x", "y"]].values)
        ppl_i = ppl.query("Country == @c").index

        tree_i = kdtree.query(ppl.loc[ppl_i, ["lon", "lat"]].values)[1]
        ppl.loc[ppl_i, "bus"] = substation_i.append(pd.Index([np.nan]))[tree_i]

    if cntries_without_ppl:
        logger.warning(f"No powerplants known in: {', '.join(cntries_without_ppl)}")

    bus_null_b = ppl["bus"].isnull()
    if bus_null_b.any():
        logger.warning(f"Couldn't find close bus for {bus_null_b.sum()} powerplants")

    if snakemake.params.alternative_clustering:
        gadm_layer_id = snakemake.params.gadm_layer_id
        country_list = snakemake.params.countries
        geo_crs = snakemake.params.geo_crs

        ppl = locate_bus_alt_clust(
            ppl.rename(columns={"lon": "x", "lat": "y", "Country": "country"}),
            country_list,
            gadm_layer_id,
            snakemake.input.gadm_shapes,
            snakemake.params.alternative_clustering,
            col_out="region_id",
        ).rename(columns={"x": "lon", "y": "lat", "country": "Country"})

    ppl.to_csv(snakemake.output.powerplants)
