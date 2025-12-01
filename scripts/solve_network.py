# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# -*- coding: utf-8 -*-
"""
Solves linear optimal power flow for a network iteratively while updating
reactances.

Relevant Settings
-----------------

.. code:: yaml

    solving:
        tmpdir:
        options:
            formulation:
            clip_p_max_pu:
            load_shedding:
            noisy_costs:
            nhours:
            min_iterations:
            max_iterations:
            skip_iterations:
            track_iterations:
        solver:
            name:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`electricity_cf`, :ref:`solving_cf`, :ref:`plotting_cf`

Inputs
------

- ``networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc``: confer :ref:`prepare`

Outputs
-------

- ``results/networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc``: Solved PyPSA network including optimisation results

    .. image:: /img/results.png
        :width: 40 %

Description
-----------

Total annual system costs are minimised with PyPSA. The full formulation of the
linear optimal power flow (plus investment planning)
is provided in the
`documentation of PyPSA <https://pypsa.readthedocs.io/en/latest/optimal_power_flow.html#linear-optimal-power-flow>`_.
The optimization is based on the :func:`network.optimize` function.
Additionally, some extra constraints specified in :mod:`prepare_network` and :mod:`solve_network` are added.

Solving the network in multiple iterations is motivated through the dependence of transmission line capacities and impedances on values of corresponding flows.
As lines are expanded their electrical parameters change, which renders the optimisation bilinear even if the power flow
equations are linearized.
To retain the computational advantage of continuous linear programming, a sequential linear programming technique
is used, where in between iterations the line impedances are updated.
Details (and errors introduced through this heuristic) are discussed in the paper

- Fabian Neumann and Tom Brown. `Heuristics for Transmission Expansion Planning in Low-Carbon Energy System Models <https://arxiv.org/abs/1907.10548>`_), *16th International Conference on the European Energy Market*, 2019. `arXiv:1907.10548 <https://arxiv.org/abs/1907.10548>`_.

.. warning::
    Capital costs of existing network components are not included in the objective function,
    since for the optimisation problem they are just a constant term (no influence on optimal result).

    Therefore, these capital costs are not included in ``network.objective``!

    If you want to calculate the full total annual system costs add these to the objective value.

.. tip::
    The rule :mod:`solve_all_networks` runs
    for all ``scenario`` s in the configuration file
    the rule :mod:`solve_network`.
"""
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from _helpers import configure_logging, create_logger, override_component_attrs
from linopy import merge
from temporal_clustering import aggregate_snapshots, temporal_aggregation_storage_constraints
from pypsa.descriptors import get_switchable_as_dense as get_as_dense
from pypsa.optimization.abstract import optimize_transmission_expansion_iteratively
#from pypsa.optimization.optimize import optimize

logger = create_logger(__name__)
pypsa.pf.logger.setLevel(logging.WARNING)

def _safe_solver_log(smk):
    """Return a usable log filename if present (named 'solver' or first log), else None."""
    lf = None
    if hasattr(smk, "log") and smk.log is not None:
        # if rule used a single unnamed path: log: "path.log"
        if isinstance(smk.log, (str, os.PathLike)):
            return str(smk.log)

        # prefer a named log 'solver'
        lf = getattr(smk.log, "solver", None)

        # fall back to first item if it's a sequence
        if lf is None:
            try:
                lf = smk.log[0]
            except Exception:
                pass

        # if someone provided a list/tuple, take the first string
        if isinstance(lf, (list, tuple)) and lf:
            lf = lf[0]

    return str(lf) if lf else None


def prepare_network(n, solve_opts):
    if "clip_p_max_pu" in solve_opts:
        for df in (
            n.generators_t.p_max_pu,
            n.generators_t.p_min_pu,
            n.storage_units_t.inflow,
        ):
            df.where(df > solve_opts["clip_p_max_pu"], other=0.0, inplace=True)

    if "lv_limit" in n.global_constraints.index:
        n.line_volume_limit = n.global_constraints.at["lv_limit", "constant"]
        n.line_volume_limit_dual = n.global_constraints.at["lv_limit", "mu"]

    if solve_opts.get("load_shedding"):
        n.add("Carrier", "Load")
        n.madd(
            "Generator",
            n.buses.index,
            " load",
            bus=n.buses.index,
            carrier="Load",
            sign=1,
            marginal_cost=solve_opts.get("load_shedding") * 1000,  # convert to Eur/MWh
            p_nom=1e12,
        )

    if solve_opts.get("noisy_costs"):
        for t in n.iterate_components():
            # if 'capital_cost' in t.df:
            #    t.df['capital_cost'] += 1e1 + 2.*(np.random.random(len(t.df)) - 0.5)
            if "marginal_cost" in t.df:
                np.random.seed(174)
                t.df["marginal_cost"] += 1e-2 + 2e-3 * (
                    np.random.random(len(t.df)) - 0.5
                )

        for t in n.iterate_components(["Line", "Link"]):
            np.random.seed(123)
            t.df["capital_cost"] += (
                1e-1 + 2e-2 * (np.random.random(len(t.df)) - 0.5)
            ) * t.df["length"]

    if solve_opts.get("nhours"):
        nhours = solve_opts["nhours"]
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760.0 / nhours

    if snakemake.config["foresight"] == "myopic":
        add_land_use_constraint(n)

    return n


def add_CCL_constraints(n, config):
    """
    Add CCL (country & carrier limit) constraint to the network.

    Add minimum and maximum levels of generator nominal capacity per carrier
    for individual countries. Opts and path for agg_p_nom_minmax.csv must be defined
    in config.yaml. Default file is available at data/agg_p_nom_minmax.csv.

    Parameters
    ----------
    n : pypsa.Network
    config : dict

    Example
    -------
    scenario:
        opts: [Co2L-CCL-24H]
    electricity:
        agg_p_nom_limits: data/agg_p_nom_minmax.csv
    """
    agg_p_nom_limits = config["electricity"].get("agg_p_nom_limits")

    try:
        agg_p_nom_minmax = pd.read_csv(agg_p_nom_limits, index_col=list(range(2)))
    except IOError:
        logger.exception(
            "Need to specify the path to a .csv file containing "
            "aggregate capacity limits per country in "
            "config['electricity']['agg_p_nom_limit']."
        )
    logger.info(
        "Adding per carrier generation capacity constraints for " "individual countries"
    )

    gen_country = n.generators.bus.map(n.buses.country)
    capacity_variable = n.model["Generator-p_nom"]

    lhs = []
    ext_carriers = n.generators.query("p_nom_extendable").carrier.unique()
    for c in ext_carriers:
        ext_carrier = n.generators.query("p_nom_extendable and carrier == @c")
        country_grouper = (
            ext_carrier.bus.map(n.buses.country)
            .rename_axis("Generator-ext")
            .rename("country")
        )
        ext_carrier_per_country = capacity_variable.loc[
            country_grouper.index
        ].groupby_sum(country_grouper)
        lhs.append(ext_carrier_per_country)
    lhs = merge(lhs, dim=pd.Index(ext_carriers, name="carrier"))

    min_matrix = agg_p_nom_minmax["min"].to_xarray().unstack().reindex_like(lhs)
    max_matrix = agg_p_nom_minmax["max"].to_xarray().unstack().reindex_like(lhs)

    n.model.add_constraints(
        lhs >= min_matrix, name="agg_p_nom_min", mask=min_matrix.notnull()
    )
    n.model.add_constraints(
        lhs <= max_matrix, name="agg_p_nom_max", mask=max_matrix.notnull()
    )


def add_EQ_constraints(n, o, scaling=1e-1):
    """
    Add equity constraints to the network.

    Currently this is only implemented for the electricity sector only.

    Opts must be specified in the config.yaml.

    Parameters
    ----------
    n : pypsa.Network
    o : str

    Example
    -------
    scenario:
        opts: [Co2L-EQ0.7-24h]

    Require each country or node to on average produce a minimal share
    of its total electricity consumption itself. Example: EQ0.7c demands each country
    to produce on average at least 70% of its consumption; EQ0.7 demands
    each node to produce on average at least 70% of its consumption.
    """
    float_regex = "[0-9]*\.?[0-9]+"
    level = float(re.findall(float_regex, o)[0])
    if o[-1] == "c":
        ggrouper = n.generators.bus.map(n.buses.country)
        lgrouper = n.loads.bus.map(n.buses.country)
        sgrouper = n.storage_units.bus.map(n.buses.country)
    else:
        ggrouper = n.generators.bus
        lgrouper = n.loads.bus
        sgrouper = n.storage_units.bus
    load = (
        n.snapshot_weightings.generators
        @ n.loads_t.p_set.groupby(lgrouper, axis=1).sum()
    )
    inflow = (
        n.snapshot_weightings.stores
        @ n.storage_units_t.inflow.groupby(sgrouper, axis=1).sum()
    )
    inflow = inflow.reindex(load.index).fillna(0.0)
    rhs = scaling * (level * load - inflow)
    dispatch_variable = n.model["Generator-p"]
    lhs_gen = (
        (dispatch_variable * (n.snapshot_weightings.generators * scaling))
        .groupby(ggrouper.to_xarray())
        .sum()
        .sum("snapshot")
    )
    # the current formulation implies that the available hydro power is (inflow - spillage)
    # it implies efficiency_dispatch is 1 which is not quite general
    # see https://github.com/pypsa-meets-earth/pypsa-earth/issues/1245 for possible improvements
    if not n.storage_units_t.inflow.empty:
        spillage_variable = n.model["StorageUnit-spill"]
        lhs_spill = (
            (spillage_variable * (-n.snapshot_weightings.stores * scaling))
            .groupby_sum(sgrouper)
            .groupby(sgrouper.to_xarray())
            .sum()
            .sum("snapshot")
        )
        lhs = lhs_gen + lhs_spill
    else:
        lhs = lhs_gen
    n.model.add_constraints(lhs >= rhs, name="equity_min")


def add_BAU_constraints(n, config):
    """
    Add a per-carrier minimal overall capacity.

    BAU_mincapacities and opts must be adjusted in the config.yaml.

    Parameters
    ----------
    n : pypsa.Network
    config : dict

    Example
    -------
    scenario:
        opts: [Co2L-BAU-24h]
    electricity:
        BAU_mincapacities:
            solar: 0
            onwind: 0
            OCGT: 100000
            offwind-ac: 0
            offwind-dc: 0
    Which sets minimum expansion across all nodes e.g. in Europe to 100GW.
    OCGT bus 1 + OCGT bus 2 + ... > 100000
    """
    mincaps = pd.Series(config["electricity"]["BAU_mincapacities"])
    p_nom = n.model["Generator-p_nom"]
    ext_i = n.generators.query("p_nom_extendable == True")

    if ext_i.empty:
        return
    ext_carrier_i = xr.DataArray(ext_i.carrier.rename_axis("Generator-ext"))
    lhs = p_nom.groupby(ext_carrier_i).sum()
    rhs = mincaps[lhs.indexes["carrier"]].rename_axis("carrier")
    n.model.add_constraints(lhs >= rhs, name="bau_mincaps")


def add_SAFE_constraints(n, config):
    """
    Add a capacity reserve margin of a certain fraction above the peak demand.
    Renewable generators and storage do not contribute. Ignores network.

    Parameters
    ----------
        n : pypsa.Network
        config : dict

    Example
    -------
    config.yaml requires to specify opts:

    scenario:
        opts: [Co2L-SAFE-24h]
    electricity:
        SAFE_reservemargin: 0.1
    Which sets a reserve margin of 10% above the peak demand.
    """
    peakdemand = n.loads_t.p_set.sum(axis=1).max()
    margin = 1.0 + config["electricity"]["SAFE_reservemargin"]
    reserve_margin = peakdemand * margin
    conventional_carriers = config["electricity"]["conventional_carriers"]
    ext_gens_i = n.generators.query("(carrier in @conventional_carriers) and p_nom_extendable").index

    capacity_variable = n.model["Generator-p_nom"]
    p_nom = n.model["Generator-p_nom"].loc[ext_gens_i]
    lhs = p_nom.sum()
    exist_conv_caps = n.generators.query("not p_nom_extendable and (carrier in @conventional_carriers)").p_nom.sum()

    rhs = reserve_margin - exist_conv_caps
    n.model.add_constraints(lhs >= rhs, name="safe_mintotalcap")


def add_operational_reserve_margin_constraint(n, sns, config):
    """
    Build reserve margin constraints based on the formulation
    as suggested in GenX
    https://energy.mit.edu/wp-content/uploads/2017/10/Enhanced-Decision-Support-for-a-Changing-Electricity-Landscape.pdf
    It implies that the reserve margin also accounts for optimal
    dispatch of distributed energy resources (DERs) and demand response
    which is a novel feature of GenX.
    """
    reserve_config = config["electricity"]["operational_reserve"]
    EPSILON_LOAD = reserve_config["epsilon_load"]
    EPSILON_VRES = reserve_config["epsilon_vres"]
    CONTINGENCY = reserve_config["contingency"]

    # Reserve Variables
    n.model.add_variables(
        0, np.inf, coords=[sns, n.generators.index], name="Generator-r"
    )
    reserve = n.model["Generator-r"]
    summed_reserve = reserve.sum("Generator")
    lhs = summed_reserve


    # Share of extendable renewable capacities
    ext_i = n.generators.query("p_nom_extendable").index
    vres_i = n.generators_t.p_max_pu.columns

    if not ext_i.empty and not vres_i.empty:
        active = vres_i.intersection(ext_i)
        if len(active) > 0:
            capacity_factor = n.generators_t.p_max_pu[active]
            p_nom_vres = (
                n.model["Generator-p_nom"]
                .loc[active]
                .rename({"Generator-ext": "Generator"})
            )
            lhs = lhs + (
                p_nom_vres * (-EPSILON_VRES * xr.DataArray(capacity_factor))
            ).sum("Generator")


    # Total demand per t
    demand = get_as_dense(n, "Load", "p_set").sum(axis=1)

    # VRES potential of non extendable generators
    capacity_factor = n.generators_t.p_max_pu[vres_i.difference(ext_i)]
    renewable_capacity = n.generators.p_nom[vres_i.difference(ext_i)]
    potential = (capacity_factor * renewable_capacity).sum(axis=1)

    # Right-hand-side
    rhs = EPSILON_LOAD * demand + EPSILON_VRES * potential + CONTINGENCY

    n.model.add_constraints(lhs >= rhs, name="reserve_margin")


def update_capacity_constraint(n):
    gen_i = n.generators.index
    ext_i = n.generators.query("p_nom_extendable").index
    fix_i = n.generators.query("not p_nom_extendable").index

    dispatch = n.model["Generator-p"]
    reserve = n.model["Generator-r"]

    capacity_fixed = n.generators.p_nom[fix_i]

    p_max_pu = get_as_dense(n, "Generator", "p_max_pu")

    lhs = dispatch + reserve

    # TODO check if `p_max_pu[ext_i]` is safe for empty `ext_i` and drop if cause in case
    if not ext_i.empty:
        capacity_variable = n.model["Generator-p_nom"].rename(
            {"Generator-ext": "Generator"}
        )
        lhs = dispatch + reserve - capacity_variable * xr.DataArray(p_max_pu[ext_i])

    rhs = (p_max_pu[fix_i] * capacity_fixed).reindex(columns=gen_i, fill_value=0)

    n.model.add_constraints(lhs <= rhs, name="gen_updated_capacity_constraint")


def add_operational_reserve_margin(n, sns, config):
    """
    Parameters
    ----------
        n : pypsa.Network
        sns: pd.DatetimeIndex
        config : dict

    Example:
    --------
    config.yaml requires to specify operational_reserve:
    operational_reserve: # like https://genxproject.github.io/GenX/dev/core/#Reserves
        activate: true
        epsilon_load: 0.02 # percentage of load at each snapshot
        epsilon_vres: 0.02 # percentage of VRES at each snapshot
        contingency: 400000 # MW
    """

    add_operational_reserve_margin_constraint(n, sns, config)

    update_capacity_constraint(n)


def add_battery_constraints(n):
    """
    Add constraint ensuring that charger = discharger, i.e.
    1 * charger_size - efficiency * discharger_size = 0
    """
    if not n.links.p_nom_extendable.any():
        return

    discharger_bool = n.links.index.str.contains("battery discharger")
    charger_bool = n.links.index.str.contains("battery charger")

    dischargers_ext = n.links[discharger_bool].query("p_nom_extendable").index
    chargers_ext = n.links[charger_bool].query("p_nom_extendable").index

    eff = n.links.efficiency[dischargers_ext].values
    lhs = (
        n.model["Link-p_nom"].loc[chargers_ext]
        - n.model["Link-p_nom"].loc[dischargers_ext] * eff
    )

    n.model.add_constraints(lhs == 0, name="Link-charger_ratio")


def add_RES_constraints(n, res_share, config):
    """
    The constraint ensures that a predefined share of power is generated
    by renewable sources

    Parameters
    ----------
        n : pypsa.Network
        res_share: float
        config : dict
    """

    logger.warning(
        "The add_RES_constraints() is still work in progress. "
        "Unexpected results might be incurred, particularly if "
        "temporal clustering is applied or if an unexpected change of technologies "
        "is subject to future improvements."
    )

    renew_techs = config["electricity"]["renewable_carriers"]

    charger = ["H2 electrolysis", "battery charger"]
    discharger = ["H2 fuel cell", "battery discharger"]

    ren_gen = n.generators.query("carrier in @renew_techs")
    ren_stores = n.storage_units.query("carrier in @renew_techs")
    ren_charger = n.links.query("carrier in @charger")
    ren_discharger = n.links.query("carrier in @discharger")

    gens_i = ren_gen.index
    stores_i = ren_stores.index
    charger_i = ren_charger.index
    discharger_i = ren_discharger.index

    # weights as xarray
    w_gen = xr.DataArray(n.snapshot_weightings.generators, dims=["snapshot"])
    w_sto = xr.DataArray(n.snapshot_weightings.stores,     dims=["snapshot"])

    # --- LHS: total renewable energy (MWh), annual scalar ---
    lhs = 0

    if len(gens_i) > 0:
        lhs_gen = (n.model["Generator-p"].loc[:, gens_i] * w_gen).sum()
        lhs = lhs + lhs_gen

    if len(stores_i) > 0:
        lhs_dispatch = (n.model["StorageUnit-p_dispatch"].loc[:, stores_i] * w_sto).sum()
        lhs_store    = (n.model["StorageUnit-p_store"   ].loc[:, stores_i] * w_sto).sum()
        lhs = lhs + lhs_dispatch - lhs_store

    if len(charger_i) > 0 or len(discharger_i) > 0:
        charge    = n.model["Link-p"].loc[:, charger_i]    if len(charger_i)    > 0 else None
        discharge = n.model["Link-p"].loc[:, discharger_i] if len(discharger_i) > 0 else None

        if charge is not None and charge.size > 0:
            eta_ch = xr.DataArray(n.links.loc[charger_i, "efficiency"].fillna(1.0),
                                  coords=[charger_i], dims=["Link"])
            lhs = lhs - (charge * eta_ch * w_sto).sum()

        if discharge is not None and discharge.size > 0:
            eta_dc = xr.DataArray(n.links.loc[discharger_i, "efficiency"].fillna(1.0),
                                  coords=[discharger_i], dims=["Link"])
            lhs = lhs + (discharge * eta_dc * w_sto).sum()

    # --- RHS: required renewable share of net consumption (annual scalar) ---
    # Load energy (MWh):
    total_load = 0.0
    if not n.loads_t.p_set.empty:
        total_load = float((n.loads_t.p_set.sum(axis=1) * n.snapshot_weightings["generators"]).sum())

    # Storage inflow energy (MWh) subtracts from net load, following add_EQ_constraints logic:
    inflow_energy = 0.0
    if not n.storage_units_t.inflow.empty:
        inflow_energy = float((n.storage_units_t.inflow.sum(axis=1) * n.snapshot_weightings["stores"]).sum())

    rhs = res_share * (total_load - inflow_energy)

    n.model.add_constraints(lhs >= rhs, name="res_share")



def add_land_use_constraint(n):
    if "m" in snakemake.wildcards.clusters:
        _add_land_use_constraint_m(n)
    else:
        _add_land_use_constraint(n)


def _add_land_use_constraint(n):
    # warning: this will miss existing offwind which is not classed AC-DC and has carrier 'offwind'

    for carrier in ["solar", "onwind", "offwind-ac", "offwind-dc"]:
        existing = (
            n.generators.loc[n.generators.carrier == carrier, "p_nom"]
            .groupby(n.generators.bus.map(n.buses.location))
            .sum()
        )
        existing.index += " " + carrier + "-" + snakemake.wildcards.planning_horizons
        
        # Find intersection and missing entries
        valid_idx = existing.index.intersection(n.generators.index)
        missing_idx = existing.index.difference(n.generators.index)
        
        if len(missing_idx) > 0:
            logger.warning(f"Land use constraint for {carrier}: {len(missing_idx)} generators not found in network")
            logger.debug(f"Missing generators: {list(missing_idx)}")
        
        # Only apply constraint to existing generators
        if len(valid_idx) > 0:
            n.generators.loc[valid_idx, "p_nom_max"] -= existing.loc[valid_idx]

    # Fix pandas FutureWarning: use direct assignment instead of inplace
    n.generators["p_nom_max"] = n.generators["p_nom_max"].clip(lower=0)


def _add_land_use_constraint_m(n):
    # if generators clustering is lower than network clustering, land_use accounting is at generators clusters

    planning_horizons = snakemake.config["scenario"]["planning_horizons"]
    grouping_years = snakemake.config["existing_capacities"]["grouping_years"]
    current_horizon = snakemake.wildcards.planning_horizons

    for carrier in ["solar", "onwind", "offwind-ac", "offwind-dc"]:
        existing = n.generators.loc[n.generators.carrier == carrier, "p_nom"]
        ind = list(
            set(
                [
                    i.split(sep=" ")[0] + " " + i.split(sep=" ")[1]
                    for i in existing.index
                ]
            )
        )

        previous_years = [
            str(y)
            for y in planning_horizons + grouping_years
            if y < int(snakemake.wildcards.planning_horizons)
        ]

        for p_year in previous_years:
            ind2 = [
                i for i in ind if i + " " + carrier + "-" + p_year in existing.index
            ]
            sel_current = [i + " " + carrier + "-" + current_horizon for i in ind2]
            sel_p_year = [i + " " + carrier + "-" + p_year for i in ind2]
            
            # Find intersection and missing entries
            valid_current = pd.Index(sel_current).intersection(n.generators.index)
            missing_current = pd.Index(sel_current).difference(n.generators.index)
            
            if len(missing_current) > 0:
                logger.warning(f"Land use constraint for {carrier} year {p_year}: {len(missing_current)} generators not found in network")
                logger.debug(f"Missing generators: {list(missing_current)}")
            
            # Only apply constraint to existing generators
            if len(valid_current) > 0:
                # Map valid current generators back to their corresponding previous year generators
                valid_p_year = [sel_p_year[sel_current.index(c)] for c in valid_current]
                n.generators.loc[valid_current, "p_nom_max"] -= existing.loc[
                    valid_p_year
                ].rename(lambda x: x[:-4] + current_horizon)

    # Fix pandas FutureWarning: use direct assignment instead of inplace
    n.generators["p_nom_max"] = n.generators["p_nom_max"].clip(lower=0)


def add_h2_network_cap(n, cap):
    h2_network = n.links.loc[n.links.carrier == "H2 pipeline"]
    if h2_network.index.empty:
        return
    h2_network_cap = n.model["Link-p_nom"]
    h2_network_cap_index = h2_network_cap.indexes["Link-ext"]
    subset_index = h2_network.index.intersection(h2_network_cap_index)
    diff_index = h2_network_cap_index.difference(subset_index)
    if len(diff_index) > 0:
        logger.warning(
            f"Impossible to set a limit for H2 pipelines extension for the following links: {diff_index}"
        )
    lhs = (
        h2_network_cap.loc[subset_index] * h2_network.loc[subset_index, "length"]
    ).sum()
    rhs = cap * 1000
    n.model.add_constraints(lhs <= rhs, name="h2_network_cap")


def H2_export_yearly_constraint(n):
    res = [
        "csp",
        "rooftop-solar",
        "solar",
        "onwind",
        "onwind2",
        "offwind",
        "offwind2",
        "ror",
    ]
    res_index = n.generators.loc[n.generators.carrier.isin(res)].index

    weightings = pd.DataFrame(
        np.outer(n.snapshot_weightings["generators"], [1.0] * len(res_index)),
        index=n.snapshots,
        columns=res_index,
    )
    capacity_variable = n.model["Generator-p"]
    # annual RES energy (MWh)
    res = (weightings * capacity_variable.loc[:, res_index].to_pandas()).sum().sum()


    load_ind = n.loads[n.loads.carrier == "AC"].index.intersection(
        n.loads_t.p_set.columns
    )

    load = (
        n.loads_t.p_set[load_ind].sum(axis=1) * n.snapshot_weightings["generators"]
    ).sum()

    h2_export = n.loads.loc["H2 export load"].p_set * 8760

    lhs = res

    include_country_load = snakemake.config["policy_config"]["yearly"][
        "re_country_load"
    ]

    if include_country_load:
        elec_efficiency = (
            n.links.filter(like="Electrolysis", axis=0).loc[:, "efficiency"].mean()
        )
        rhs = (
            h2_export * (1 / elec_efficiency) + load
        )  # 0.7 is approximation of electrloyzer efficiency # TODO obtain value from network
    else:
        rhs = h2_export * (1 / 0.7)

    n.model.add_constraints(lhs >= rhs, name="H2ExportConstraint-RESproduction")


def monthly_constraints(n, n_ref):
    res_techs = [
        "csp",
        "rooftop-solar",
        "solar",
        "onwind",
        "onwind2",
        "offwind",
        "offwind2",
        "ror",
    ]
    allowed_excess = snakemake.config["policy_config"]["hydrogen"]["allowed_excess"]

    res_index = n.generators.loc[n.generators.carrier.isin(res_techs)].index

    weightings = pd.DataFrame(
        np.outer(n.snapshot_weightings["generators"], [1.0] * len(res_index)),
        index=n.snapshots,
        columns=res_index,
    )
    capacity_variable = n.model["Generator-p"]

    # single line sum
    res = (weightings * capacity_variable.loc[:, res_index].to_pandas()).sum(axis=1)
    res = res.groupby(res.index.month).sum()

    link_p = n.model["Link-p"]
    elec_i = n.links.index[n.links.index.str.contains("H2 Electrolysis")]
    electrolysis = link_p.loc[:, elec_i]  # xarray [snapshot, Link]

    w_gen = xr.DataArray(n.snapshot_weightings["generators"], dims=["snapshot"])
    elec_eff = xr.DataArray(
        n.links.loc[elec_i, "efficiency"],
        coords=[elec_i],
        dims=["Link"],
    )

    elec_input = (-allowed_excess) * (electrolysis * w_gen * elec_eff).sum("Link")
    elec_input = elec_input.to_pandas().groupby(elec_input.index.month).sum()


    if snakemake.config["policy_config"]["hydrogen"]["additionality"]:
        # ref RES by month (MWh)
        res_ref = (n_ref.generators_t.p[res_index] * weightings)
        res_ref = res_ref.groupby(n_ref.generators_t.p.index.month).sum().sum(axis=1)

        # ref electrolysis input by month (MWh); p0 at electrolyzers is negative on the power side
        elec_cols_ref = n_ref.links_t.p0.columns[n_ref.links_t.p0.columns.str.contains("H2 Electrolysis")]
        if len(elec_cols_ref) > 0:
            elec_series_ref = n_ref.links_t.p0[elec_cols_ref].sum(axis=1)  # negative
            elec_input_ref = ( -elec_series_ref * n_ref.snapshot_weightings["generators"] )
            elec_input_ref = elec_input_ref.groupby(elec_input_ref.index.month).sum()
        else:
            elec_input_ref = pd.Series(0.0, index=res_ref.index)


        for i in range(len(res.index)):
            lhs = res.iloc[i] + elec_input.iloc[i]
            rhs = res_ref.iloc[i] + elec_input_ref.iloc[i]
            n.model.add_constraints(
                lhs >= rhs, name=f"RESconstraints_{i}-REStarget_{i}"
            )


    else:
        for i in range(len(res.index)):
            lhs = res.iloc[i] + elec_input.iloc[i]

            n.model.add_constraints(
                lhs >= 0.0, name=f"RESconstraints_{i}-REStarget_{i}"
            )

    # else:
    #     logger.info("ignoring H2 export constraint as wildcard is set to 0")


def add_chp_constraints(n):
    electric_bool = (
        n.links.index.str.contains("urban central")
        & n.links.index.str.contains("CHP")
        & n.links.index.str.contains("electric")
    )
    heat_bool = (
        n.links.index.str.contains("urban central")
        & n.links.index.str.contains("CHP")
        & n.links.index.str.contains("heat")
    )

    electric = n.links.index[electric_bool]
    heat = n.links.index[heat_bool]

    electric_ext = n.links[electric_bool].query("p_nom_extendable").index
    heat_ext = n.links[heat_bool].query("p_nom_extendable").index

    electric_fix = n.links[electric_bool].query("~p_nom_extendable").index
    heat_fix = n.links[heat_bool].query("~p_nom_extendable").index

    p = n.model["Link-p"]  # dimension: [time, link]

    # output ratio between heat and electricity and top_iso_fuel_line for extendable
    if not electric_ext.empty:
        p_nom = n.model["Link-p_nom"]

        lhs = (
            p_nom.loc[electric_ext]
            * (n.links.p_nom_ratio * n.links.efficiency)[electric_ext].values
            - p_nom.loc[heat_ext] * n.links.efficiency[heat_ext].values
        )
        n.model.add_constraints(lhs == 0, name="chplink-fix_p_nom_ratio")

        rename = {"Link-ext": "Link"}
        lhs = (
            p.loc[:, electric_ext]
            + p.loc[:, heat_ext]
            - p_nom.rename(rename).loc[electric_ext]
        )
        n.model.add_constraints(lhs <= 0, name="chplink-top_iso_fuel_line_ext")

    # top_iso_fuel_line for fixed
    if not electric_fix.empty:
        lhs = p.loc[:, electric_fix] + p.loc[:, heat_fix]
        rhs = n.links.p_nom[electric_fix]
        n.model.add_constraints(lhs <= rhs, name="chplink-top_iso_fuel_line_fix")

    # back-pressure
    if not electric.empty and not heat.empty:
        lhs = (
            p.loc[:, heat] * (n.links.efficiency[heat] * n.links.c_b[electric].values)
            - p.loc[:, electric] * n.links.efficiency[electric]
        )
        n.model.add_constraints(lhs <= 0, name="chplink-backpressure")



def add_co2_sequestration_limit(n, sns):
    co2_stores = n.stores.loc[n.stores.carrier == "co2 stored"].index

    if co2_stores.empty:
        return

    vars_final_co2_stored = n.model["Store-e"].loc[sns[-1], co2_stores]

    lhs = (1 * vars_final_co2_stored).sum()
    rhs = (
        n.config["sector"].get("co2_sequestration_potential", 5) * 1e6
    )  # TODO change 200 limit (Europe)

    name = "co2_sequestration_limit"

    n.model.add_constraints(lhs <= rhs, name=f"GlobalConstraint-{name}")


def set_h2_colors(n):
    blue_h2 = n.model["Link-p"].loc[
        n.links.index[n.links.index.str.contains("blue H2")]
    ]

    pink_h2 = n.model["Link-p"].loc[
        n.links.index[n.links.index.str.contains("pink H2")]
    ]

    fuelcell_ind = n.loads[n.loads.carrier == "land transport fuel cell"].index

    other_ind = n.loads[
        (n.loads.carrier == "H2 for industry")
        | (n.loads.carrier == "H2 for shipping")
        | (n.loads.carrier == "H2")
    ].index

    load_fuelcell = (
        n.loads_t.p_set[fuelcell_ind].sum(axis=1) * n.snapshot_weightings["generators"]
    ).sum()

    load_other_h2 = (n.loads_t.p_set[other_ind].sum(axis=1) * n.snapshot_weightings["generators"]).sum()


    load_h2 = load_fuelcell + load_other_h2

    weightings_blue = pd.DataFrame(
        np.outer(n.snapshot_weightings["generators"], [1.0] * len(blue_h2.columns)),
        index=n.snapshots,
        columns=blue_h2.columns,
    )

    weightings_pink = pd.DataFrame(
        np.outer(n.snapshot_weightings["generators"], [1.0] * len(pink_h2.columns)),
        index=n.snapshots,
        columns=pink_h2.columns,
    )

    total_blue = (weightings_blue * blue_h2).sum().sum()

    total_pink = (weightings_pink * pink_h2).sum().sum()

    rhs_blue = load_h2 * snakemake.config["sector"]["hydrogen"]["blue_share"]
    rhs_pink = load_h2 * snakemake.config["sector"]["hydrogen"]["pink_share"]

    n.model.add_constraints(total_blue == rhs_blue, name="blue_h2_share")

    n.model.add_constraints(total_pink == rhs_pink, name="pink_h2_share")


def add_existing(n):
    if snakemake.wildcards["planning_horizons"] == "2050":
        directory = (
            "results/"
            + "Existing_capacities/"
            + snakemake.config["run"].replace("2050", "2030")
        )
        n_name = (
            snakemake.input.network.split("/")[-1]
            .replace(str(snakemake.config["scenario"]["clusters"][0]), "")
            .replace(str(snakemake.config["costs"]["discountrate"][0]), "")
            .replace("_presec", "")
            .replace(".nc", ".csv")
        )
        df = pd.read_csv(directory + "/electrolyzer_caps_" + n_name, index_col=0)
        existing_electrolyzers = df.p_nom_opt.values

        h2_index = n.links[n.links.carrier == "H2 Electrolysis"].index
        n.links.loc[h2_index, "p_nom_min"] = existing_electrolyzers

        # n_name = snakemake.input.network.split("/")[-1].replace(str(snakemake.config["scenario"]["clusters"][0]), "").\
        #     replace(".nc", ".csv").replace(str(snakemake.config["costs"]["discountrate"][0]), "")
        df = pd.read_csv(directory + "/res_caps_" + n_name, index_col=0)

        for tech in snakemake.config["custom_data"]["renewables"]:
            # df = pd.read_csv(snakemake.config["custom_data"]["existing_renewables"], index_col=0)
            existing_res = df.loc[tech]
            existing_res.index = existing_res.index.str.apply(lambda x: x + tech)
            tech_index = n.generators[n.generators.carrier == tech].index
            n.generators.loc[tech_index, tech] = existing_res


def add_lossy_bidirectional_link_constraints(n: pypsa.Network) -> None:
    """
    Ensures that the two links simulating a bidirectional_link are extended the same amount.
    """
    if not n.links.p_nom_extendable.any() or "reversed" not in n.links.columns:
        return

    n.links["reversed"] = n.links.reversed.fillna(0).astype(bool)
    carriers = n.links.loc[n.links.reversed, "carrier"].unique()

    forward_i = n.links.query(
        "carrier in @carriers and ~reversed and p_nom_extendable"
    ).index

    def get_backward_i(forward_i):
        return pd.Index([
            (re.sub(r"-(\d{4})$", r"-reversed-\1", s)
             if re.search(r"-\d{4}$", s) else s + "-reversed")
            for s in forward_i
        ])

    # --- replacement block starts here ---
    links_p_nom = n.model["Link-p_nom"]
    var_index   = links_p_nom.indexes["Link-ext"]

    pairs = []
    for fwd in forward_i:
        bwd = get_backward_i(pd.Index([fwd]))[0]
        if fwd in var_index and bwd in var_index:
            pairs.append((fwd, bwd))

    if not pairs:
        return

    subset_forward  = pd.Index([f for f, _ in pairs])
    subset_backward = pd.Index([b for _, b in pairs])

    lhs = links_p_nom.loc[subset_backward] - links_p_nom.loc[subset_forward]
    # --- replacement block ends here ---

    n.model.add_constraints(lhs == 0, name="Link-bidirectional_sync")


def _generator_output_energy_by_buscarrier(n, bus_carrier="AC"):
    """
    Annual generator output (MWh) grouped by GENERATOR carrier, but only from
    generators whose bus has carrier == bus_carrier (e.g. "AC").
    """
    # pick only generators connected to the requested bus carrier
    ac_gen_i = n.generators.index[
        n.generators.bus.map(n.buses.carrier).fillna("").eq(bus_carrier)
    ]
    if len(ac_gen_i) == 0:
        return xr.DataArray([], dims=["carrier"])

    # Use integer-based indexing for temporal clustering compatibility
    p_g_full = n.model["Generator-p"]
    gen_idx = p_g_full.indexes.get("Generator", pd.Index([]))
    gen_mask = gen_idx.isin(ac_gen_i)
    p_g = p_g_full.isel(Generator=gen_mask)                            # [snapshot, Generator]
    
    # Get the actual generator indices after filtering
    filtered_gen_i = gen_idx[gen_mask]
    
    # snapshot_weightings.generators is a Series (column from DataFrame)
    # Explicitly provide coordinates to avoid timestamp/int comparison warnings
    w = xr.DataArray(
        n.snapshot_weightings.generators.values,
        coords=[n.snapshots],
        dims=["snapshot"]
    )
    g_car = n.generators.loc[filtered_gen_i, "carrier"].rename_axis("Generator").to_xarray()

    # MWh by generator carrier
    return (p_g * w).sum("snapshot").groupby(g_car).sum("Generator")


def _link_output_energy_by_buscarrier(n, bus_carrier="AC"):
    """
    Annual link *output-side* energy (MWh) grouped by LINK carrier, restricted
    to links whose OUTPUT bus (bus1) sits on a bus with carrier == bus_carrier.
    Energy at the output side is p[t,link] * efficiency[link].
    """
    if n.links.empty:
        return xr.DataArray([], dims=["carrier"])

    out_bus_carrier = n.links.bus1.map(n.buses.carrier).fillna("")
    link_i = n.links.index[out_bus_carrier.eq(bus_carrier)]
    if len(link_i) == 0:
        return xr.DataArray([], dims=["carrier"])

    # Use integer-based indexing for temporal clustering compatibility
    p_l_full = n.model["Link-p"]
    link_idx = p_l_full.indexes.get("Link", pd.Index([]))
    link_mask = link_idx.isin(link_i)
    p_l = p_l_full.isel(Link=link_mask)                                # [snapshot, Link]
    
    # Get the actual link indices after filtering
    filtered_link_i = link_idx[link_mask]
    
    eta  = xr.DataArray(n.links.loc[filtered_link_i, "efficiency"].fillna(1.0),
                        coords=[filtered_link_i], dims=["Link"])
    # snapshot_weightings.generators is a Series (column from DataFrame)
    # Explicitly provide coordinates to avoid timestamp/int comparison warnings
    w = xr.DataArray(
        n.snapshot_weightings.generators.values,
        coords=[n.snapshots],
        dims=["snapshot"]
    )
    lcar = n.links.loc[filtered_link_i, "carrier"].rename_axis("Link").to_xarray()

    # MWh by link carrier at AC output
    return (p_l * eta * w).sum("snapshot").groupby(lcar).sum("Link")

def _storageunit_output_energy_by_buscarrier(n, bus_carrier="AC"):
    """
    Annual StorageUnit *output-side* energy (MWh) grouped by STORAGE UNIT carrier,
    restricted to storage units whose bus sits on a bus with carrier == bus_carrier.
    Output is the electric dispatch variable p_dispatch (already AC-side).
    Weighted with snapshot_weightings.stores (consistent with PyPSA stats).
    """
    if n.storage_units.empty:
        return xr.DataArray([], dims=["carrier"])

    ac_su_i = n.storage_units.index[
        n.storage_units.bus.map(n.buses.carrier).fillna("").eq(bus_carrier)
    ]
    if len(ac_su_i) == 0:
        return xr.DataArray([], dims=["carrier"])

    # Use integer-based indexing for temporal clustering compatibility
    p_su_full = n.model["StorageUnit-p_dispatch"]
    su_idx = p_su_full.indexes.get("StorageUnit", pd.Index([]))
    su_mask = su_idx.isin(ac_su_i)
    p_su = p_su_full.isel(StorageUnit=su_mask)                         # [snapshot, StorageUnit]
    
    # Get the actual storage unit indices after filtering
    filtered_su_i = su_idx[su_mask]
    
    # snapshot_weightings.stores is a Series (column from DataFrame)
    # Explicitly provide coordinates to avoid timestamp/int comparison warnings
    w = xr.DataArray(
        n.snapshot_weightings.stores.values,
        coords=[n.snapshots],
        dims=["snapshot"]
    )
    su_car = n.storage_units.loc[filtered_su_i, "carrier"].rename_axis("StorageUnit").to_xarray()

    # MWh by storage-unit carrier at AC output
    return (p_su * w).sum("snapshot").groupby(su_car).sum("StorageUnit")


def add_baseyear_generation_band(n, planning_year, config):
    global_cfg = config.get("global_specific", {})
    cfg = global_cfg.get("baseyear_generation", {})
    if not cfg or not cfg.get("baseyear_generation_constraint", False):
        return

    baseyear = str(cfg.get("year", 2020))
    if str(planning_year) != baseyear:
        logger.info(f"Skipping baseyear generation constraints for {planning_year} (configured for {baseyear})")
        return

    logger.info(f"Adding baseyear generation constraints for {planning_year}")

    tol   = float(cfg.get("tolerance", 0.05))
    units = str(cfg.get("units", "TWh")).lower()
    unit_scale = {"mwh":1.0, "gwh":1e3, "twh":1e6}.get(units, 1e6)

    carriers_map     = cfg.get("carriers_map", {})
    targets          = cfg.get("targets", {})
    link_bus_carrier = cfg.get("link_bus_carrier", "AC") # should be "AC"
    gen_bus_carrier  = cfg.get("gen_bus_carrier",  "AC") # allows override; default "AC"

    gen_E  = _generator_output_energy_by_buscarrier(n, bus_carrier=gen_bus_carrier) # [carrier] MWh
    link_E = _link_output_energy_by_buscarrier(n, bus_carrier=link_bus_carrier) # [carrier] MWh
    su_E = _storageunit_output_energy_by_buscarrier(n, bus_carrier=gen_bus_carrier) # [carrier] MWh

    def _sum_tokens(tokens):
        toks = tokens if isinstance(tokens, (list, tuple)) else [tokens]
        pieces = []

        gen_list  = list(gen_E.indexes.get("carrier", []))  if gen_E.size  else []
        link_list = list(link_E.indexes.get("carrier", [])) if link_E.size else []
        su_list   = list(su_E.indexes.get("carrier", []))   if su_E.size   else []

        for t in toks:
            if isinstance(t, str) and t.startswith("re:"):
                pat = re.compile(t[3:])
                g = [c for c in gen_list  if pat.search(c)]
                l = [c for c in link_list if pat.search(c)]
                s = [c for c in su_list   if pat.search(c)]
                if g: pieces.append(gen_E.sel(carrier=g).sum("carrier"))
                if l: pieces.append(link_E.sel(carrier=l).sum("carrier"))
                if s: pieces.append(su_E.sel(carrier=s).sum("carrier"))
            else:
                if t in gen_list:  pieces.append(gen_E.sel(carrier=t))
                if t in link_list: pieces.append(link_E.sel(carrier=t))
                if t in su_list:   pieces.append(su_E.sel(carrier=t))

        if not pieces:
            return None
        out = pieces[0]
        for p in pieces[1:]:
            out = out + p
        return out

    for alias, target in targets.items():
        if isinstance(target, str) and target.upper().startswith("X"):
            logger.info(f"Skipping {alias} (placeholder target '{target}')")
            continue

        tokens = carriers_map.get(alias, [alias])  # e.g., ["coal"] or ["coal","lignite"]
        lhs = _sum_tokens(tokens)
        if lhs is None:
            logger.warning(f"No carriers matched for alias '{alias}' with tokens {tokens}")
            continue

        lower = float(target) * (1.0 - tol) * unit_scale
        upper = float(target) * (1.0 + tol) * unit_scale

        logger.info(f"{alias}: {lower/unit_scale:.2f} ≤ AC-side energy ≤ {upper/unit_scale:.2f} {units.upper()} (tokens={tokens})")
        n.model.add_constraints(lhs >= lower, name=f"baseyear_energy_min__{alias}")
        n.model.add_constraints(lhs <= upper, name=f"baseyear_energy_max__{alias}")


def add_year2025_generation_band(n, planning_year, config):
    """
    Add 2025 generation constraints similar to baseyear constraints.
    Constrains annual generation by carrier to match 2025 targets +/- tolerance.
    """
    global_cfg = config.get("global_specific", {})
    cfg = global_cfg.get("year2025_generation", {})
    if not cfg or not cfg.get("year2025_generation_constraint", False):
        return

    target_year = str(cfg.get("year", 2025))
    if str(planning_year) != target_year:
        logger.info(f"Skipping 2025 generation constraints for {planning_year} (configured for {target_year})")
        return

    logger.info(f"Adding 2025 generation constraints for {planning_year}")

    tol   = float(cfg.get("tolerance", 0.15))
    units = str(cfg.get("units", "TWh")).lower()
    unit_scale = {"mwh":1.0, "gwh":1e3, "twh":1e6}.get(units, 1e6)

    carriers_map     = cfg.get("carriers_map", {})
    targets          = cfg.get("targets", {})
    link_bus_carrier = cfg.get("link_bus_carrier", "AC")
    gen_bus_carrier  = cfg.get("gen_bus_carrier",  "AC")

    gen_E  = _generator_output_energy_by_buscarrier(n, bus_carrier=gen_bus_carrier) # [carrier] MWh
    link_E = _link_output_energy_by_buscarrier(n, bus_carrier=link_bus_carrier) # [carrier] MWh
    su_E = _storageunit_output_energy_by_buscarrier(n, bus_carrier=gen_bus_carrier) # [carrier] MWh

    def _sum_tokens(tokens):
        toks = tokens if isinstance(tokens, (list, tuple)) else [tokens]
        pieces = []

        gen_list  = list(gen_E.indexes.get("carrier", []))  if gen_E.size  else []
        link_list = list(link_E.indexes.get("carrier", [])) if link_E.size else []
        su_list   = list(su_E.indexes.get("carrier", []))   if su_E.size   else []

        for t in toks:
            if isinstance(t, str) and t.startswith("re:"):
                pat = re.compile(t[3:])
                g = [c for c in gen_list  if pat.search(c)]
                l = [c for c in link_list if pat.search(c)]
                s = [c for c in su_list   if pat.search(c)]
                if g: pieces.append(gen_E.sel(carrier=g).sum("carrier"))
                if l: pieces.append(link_E.sel(carrier=l).sum("carrier"))
                if s: pieces.append(su_E.sel(carrier=s).sum("carrier"))
            else:
                if t in gen_list:  pieces.append(gen_E.sel(carrier=t))
                if t in link_list: pieces.append(link_E.sel(carrier=t))
                if t in su_list:   pieces.append(su_E.sel(carrier=t))

        if not pieces:
            return None
        out = pieces[0]
        for p in pieces[1:]:
            out = out + p
        return out

    for alias, target in targets.items():
        if isinstance(target, str) and target.upper().startswith("X"):
            logger.info(f"Skipping {alias} (placeholder target '{target}')")
            continue

        tokens = carriers_map.get(alias, [alias])  # e.g., ["coal"] or ["coal","lignite"]
        lhs = _sum_tokens(tokens)
        if lhs is None:
            logger.warning(f"No carriers matched for alias '{alias}' with tokens {tokens}")
            continue

        lower = float(target) * (1.0 - tol) * unit_scale
        upper = float(target) * (1.0 + tol) * unit_scale

        logger.info(f"{alias}: {lower/unit_scale:.2f} ≤ AC-side energy ≤ {upper/unit_scale:.2f} {units.upper()} (tokens={tokens})")
        n.model.add_constraints(lhs >= lower, name=f"year2025_energy_min__{alias}")
        n.model.add_constraints(lhs <= upper, name=f"year2025_energy_max__{alias}")


def add_year2025_capacity_targets(n, planning_year, config):
    """
    Add 2025 capacity targets for solar and onwind technologies.
    Constrains installed capacity (p_nom) by carrier to match 2025 targets +/- tolerance.
    """
    global_cfg = config.get("global_specific", {})
    cfg = global_cfg.get("year2025_capacity", {})
    if not cfg or not cfg.get("year2025_capacity_constraint", False):
        return

    target_year = str(cfg.get("year", 2025))
    if str(planning_year) != target_year:
        logger.info(f"Skipping 2025 capacity constraints for {planning_year} (configured for {target_year})")
        return

    logger.info(f"Adding 2025 capacity constraints for {planning_year}")

    tol   = float(cfg.get("tolerance", 0.15))
    units = str(cfg.get("units", "GW")).lower()
    unit_scale = {"mw":1.0, "gw":1e3, "tw":1e6}.get(units, 1e3)

    targets = cfg.get("targets", {})

    # Get extendable generators by carrier
    ext_gens = n.generators.query("p_nom_extendable")
    
    if ext_gens.empty:
        logger.warning("No extendable generators found for capacity constraints")
        return

    # Get the capacity variable
    p_nom = n.model["Generator-p_nom"]
    
    for carrier, target in targets.items():
        if isinstance(target, str) and target.upper().startswith("X"):
            logger.info(f"Skipping {carrier} (placeholder target '{target}')")
            continue

        # Find generators matching this carrier
        carrier_gens = ext_gens[ext_gens.carrier == carrier].index
        
        if len(carrier_gens) == 0:
            logger.warning(f"No extendable generators found for carrier '{carrier}'")
            continue

        # Sum of p_nom for this carrier
        lhs = p_nom.loc[carrier_gens].sum()
        
        # Add existing non-extendable capacity
        existing_capacity = n.generators.query(
            "carrier == @carrier and not p_nom_extendable"
        ).p_nom.sum()
        
        lower = float(target) * (1.0 - tol) * unit_scale - existing_capacity
        upper = float(target) * (1.0 + tol) * unit_scale - existing_capacity

        logger.info(
            f"{carrier}: {(lower + existing_capacity)/unit_scale:.2f} ≤ total capacity ≤ "
            f"{(upper + existing_capacity)/unit_scale:.2f} {units.upper()} "
            f"(existing: {existing_capacity/unit_scale:.2f} {units.upper()}, "
            f"new: {lower/unit_scale:.2f}-{upper/unit_scale:.2f} {units.upper()})"
        )
        
        n.model.add_constraints(lhs >= lower, name=f"year2025_capacity_min__{carrier}")
        n.model.add_constraints(lhs <= upper, name=f"year2025_capacity_max__{carrier}")


def extra_functionality(n, snapshots):
    """
    Collects supplementary constraints which will be passed to
    ``pypsa.linopf.network_lopf``.

    If you want to enforce additional custom constraints, this is a good location to add them.
    The arguments ``opts`` and ``snakemake.config`` are expected to be attached to the network.
    """
    opts = n.opts
    config = n.config
    if "BAU" in opts and n.generators.p_nom_extendable.any():
        add_BAU_constraints(n, config)
    if "SAFE" in opts and n.generators.p_nom_extendable.any():
        add_SAFE_constraints(n, config)
    if "CCL" in opts and n.generators.p_nom_extendable.any():
        add_CCL_constraints(n, config)
    reserve = config["electricity"].get("operational_reserve", {})
    if reserve.get("activate"):
        add_operational_reserve_margin(n, snapshots, config)
    for o in opts:
        if "RES" in o:
            res_share = float(re.findall("[0-9]*\.?[0-9]+$", o)[0])
            add_RES_constraints(n, res_share, config)
    for o in opts:
        if "EQ" in o:
            add_EQ_constraints(n, o)

    add_battery_constraints(n)
    add_lossy_bidirectional_link_constraints(n)

    # Temporal clustering storage constraints
    # Check if temporal clustering is active via period_id in n.temporal_cluster
    tc_config = config.get("temporal_clustering", {})
    
    if tc_config.get("activate", False):
        logger.info("Adding temporal aggregation storage constraints (Kotzur et al. 2018)")
        
        # Get optional parameters from config (with defaults for Pyomo parity)
        
        use_dt_in_intra = tc_config.get("use_dt_in_intra", False)
        include_inflow_in_intra = tc_config.get("include_inflow_in_intra", False)
        
        logger.info(f"  use_dt_in_intra: {use_dt_in_intra}")
        logger.info(f"  include_inflow_in_intra: {include_inflow_in_intra}")
        
        temporal_aggregation_storage_constraints(
            n,
            use_dt_in_intra=use_dt_in_intra,
            include_inflow_in_intra=include_inflow_in_intra,
        )
    else:
        logger.info("No temporal clustering detected, skipping temporal storage constraints")


    if snakemake.config["sector"]["chp"]:
        logger.info("setting CHP constraints")
        add_chp_constraints(n)

    if (
        snakemake.config["policy_config"]["hydrogen"]["temporal_matching"]
        == "h2_yearly_matching"
    ):
        if snakemake.config["policy_config"]["hydrogen"]["additionality"] == True:
            logger.info(
                "additionality is currently not supported for yearly constraints, proceeding without additionality"
            )
        logger.info("setting h2 export to yearly greenness constraint")
        H2_export_yearly_constraint(n)

    elif (
        snakemake.config["policy_config"]["hydrogen"]["temporal_matching"]
        == "h2_monthly_matching"
    ):
        if not snakemake.config["policy_config"]["hydrogen"]["is_reference"]:
            logger.info("setting h2 export to monthly greenness constraint")
            n_ref_local = getattr(n, "n_ref", None)
            if n_ref_local is not None:
                monthly_constraints(n, n_ref_local)
            else:
                logger.warning("monthly matching requested but n_ref not provided; skipping")


    elif (
        snakemake.config["policy_config"]["hydrogen"]["temporal_matching"]
        == "no_res_matching"
    ):
        logger.info("no h2 export constraint set")

    else:
        raise ValueError(
            'H2 export constraint is invalid, check config["policy_config"]'
        )

    if snakemake.config["sector"]["hydrogen"]["network"]:
        if snakemake.config["sector"]["hydrogen"]["network_limit"]:
            add_h2_network_cap(
                n, snakemake.config["sector"]["hydrogen"]["network_limit"]
            )

    if snakemake.config["sector"]["hydrogen"]["set_color_shares"]:
        logger.info("setting H2 color mix")
        set_h2_colors(n)

    add_baseyear_generation_band(
        n,
        planning_year=snakemake.wildcards.planning_horizons,
        config=n.config if hasattr(n, "config") else snakemake.config,
    )

    add_year2025_generation_band(
        n,
        planning_year=snakemake.wildcards.planning_horizons,
        config=n.config if hasattr(n, "config") else snakemake.config,
    )

    add_co2_sequestration_limit(n, snapshots)

    # Add 2025 capacity targets
    add_year2025_capacity_targets(
        n,
        planning_year=snakemake.wildcards.planning_horizons,
        config=n.config if hasattr(n, "config") else snakemake.config,
    )

    logger.info('Model after adding extra functionality:')
    logger.info(n.model)


def solve_network(n, config, solving, **kwargs):
    set_of_options = solving["solver"]["options"]
    cf_solving = solving["options"]

    kwargs["solver_options"] = (
        solving["solver_options"][set_of_options] if set_of_options else {}
    )
    kwargs["solver_name"] = solving["solver"]["name"]
    kwargs["extra_functionality"] = extra_functionality

    skip_iterations = cf_solving.get("skip_iterations", False)
    if not n.lines.s_nom_extendable.any():
        skip_iterations = True
        logger.info("No expandable lines found. Skipping iterative solving.")

    # add to network for extra_functionality
    n.config = config
    if not hasattr(n, "opts"):
        if "snakemake" in globals():
            n.opts = snakemake.wildcards.opts.split("-")
        else:
            n.opts = globals().get("opts", [])



    logger.info(f"Network has {len(n.buses)} buses, {len(n.generators)} generators")
    logger.info(f"Network has {len(n.loads)} loads, {len(n.stores)} stores, {len(n.links)} links")
    valid_buses = set(n.buses.index)
    
    gen_bad_buses = n.generators[~n.generators.bus.isin(valid_buses)]
    if len(gen_bad_buses) > 0:
        logger.warning(f"Found {len(gen_bad_buses)} generators with invalid buses: {gen_bad_buses.index.tolist()}")
        logger.warning(f"Invalid generator buses: {gen_bad_buses.bus.unique().tolist()}")
    
    load_bad_buses = n.loads[~n.loads.bus.isin(valid_buses)]
    if len(load_bad_buses) > 0:
        logger.warning(f"Found {len(load_bad_buses)} loads with invalid buses: {load_bad_buses.index.tolist()}")
        logger.warning(f"Invalid load buses: {load_bad_buses.bus.unique().tolist()}")
    
    store_bad_buses = n.stores[~n.stores.bus.isin(valid_buses)]
    if len(store_bad_buses) > 0:
        logger.warning(f"Found {len(store_bad_buses)} stores with invalid buses: {store_bad_buses.index.tolist()}")
        logger.warning(f"Invalid store buses: {store_bad_buses.bus.unique().tolist()}")
    
    link_bad_bus0 = n.links[~n.links.bus0.isin(valid_buses)]
    link_bad_bus1 = n.links[~n.links.bus1.isin(valid_buses)]
    if len(link_bad_bus0) > 0:
        logger.warning(f"Found {len(link_bad_bus0)} links with invalid bus0: {link_bad_bus0.index.tolist()}")
        logger.warning(f"Invalid link bus0 buses: {link_bad_bus0.bus0.unique().tolist()}")
    if len(link_bad_bus1) > 0:
        logger.warning(f"Found {len(link_bad_bus1)} links with invalid bus1: {link_bad_bus1.index.tolist()}")
        logger.warning(f"Invalid link bus1 buses: {link_bad_bus1.bus1.unique().tolist()}")
    
    gen_capacity_issues = n.generators.query("p_nom_max < p_nom_min")
    if len(gen_capacity_issues) > 0:
        logger.warning(f"Found {len(gen_capacity_issues)} generators with p_nom_max < p_nom_min")
        for idx, gen in gen_capacity_issues.iterrows():
            logger.warning(f"  {idx}: p_nom_min={gen.p_nom_min}, p_nom_max={gen.p_nom_max}")

        # Fix the issue by setting p_nom_max to accommodate the minimum requirement
        logger.info(f"Fixing {len(gen_capacity_issues)} generators with p_nom_max < p_nom_min")
        for idx, gen in gen_capacity_issues.iterrows():
            logger.info(f"  {idx}: p_nom_min={gen.p_nom_min}, p_nom_max={gen.p_nom_max} -> setting p_nom_max = p_nom_min * 2")
        n.generators.loc[gen_capacity_issues.index, "p_nom_max"] = n.generators.loc[gen_capacity_issues.index, "p_nom_min"] * 2

    zero_cost_gens = n.generators.query("marginal_cost <= 0 and p_nom_extendable == True")
    if len(zero_cost_gens) > 0:
        logger.warning(f"Found {len(zero_cost_gens)} extendable generators with zero/negative marginal costs")
        logger.warning(f"Zero cost generators: {zero_cost_gens.index.tolist()}")
    
    if 'carrier' in n.buses.columns:
        bus_carriers = n.buses.carrier.value_counts()
        logger.info(f"Bus carriers: {bus_carriers.to_dict()}")
        
        h2_buses = n.buses[n.buses.carrier.str.contains('H2', na=False)]
        if len(h2_buses) > 0:
            logger.info(f"Found {len(h2_buses)} H2 buses: {h2_buses.index.tolist()}")
            
            h2_loads = n.loads[n.loads.carrier.str.contains('H2', na=False)] if not n.loads.empty else pd.DataFrame()
            h2_stores = n.stores[n.stores.carrier.str.contains('H2', na=False)] if not n.stores.empty else pd.DataFrame()
            h2_links = n.links[n.links.carrier.str.contains('H2', na=False)] if not n.links.empty else pd.DataFrame()
            
            logger.info(f"H2 components: {len(h2_loads)} loads, {len(h2_stores)} stores, {len(h2_links)} links")
    

    logger.info("Performing preemptive network cleanup...")
    buses_to_keep = set(n.buses.index)
    
    # Clean generators with undefined buses
    generators_with_bad_buses = n.generators[~n.generators.bus.isin(buses_to_keep)]
    if len(generators_with_bad_buses) > 0:
        logger.warning(f"Removing {len(generators_with_bad_buses)} generators with undefined buses before solving")
        logger.info(f"Generators with bad buses:\n{generators_with_bad_buses}")
        n.generators = n.generators.drop(generators_with_bad_buses.index)
        
        # Clean time series data for removed generators
        for attr in ['p_max_pu', 'p_min_pu']:
            if hasattr(n.generators_t, attr):
                attr_data = getattr(n.generators_t, attr)
                cols_to_remove = generators_with_bad_buses.index.intersection(attr_data.columns)
                if len(cols_to_remove) > 0:
                    setattr(n.generators_t, attr, attr_data.drop(columns=cols_to_remove))
    
    # Clean loads with undefined buses
    loads_with_bad_buses = n.loads[~n.loads.bus.isin(buses_to_keep)]
    if len(loads_with_bad_buses) > 0:
        logger.warning(f"Removing {len(loads_with_bad_buses)} loads with undefined buses before solving")
        logger.info(f"Loads with bad buses:\n{loads_with_bad_buses}")
        n.loads = n.loads.drop(loads_with_bad_buses.index)
        
        if hasattr(n.loads_t, 'p_set'):
            cols_to_remove = loads_with_bad_buses.index.intersection(n.loads_t.p_set.columns)
            if len(cols_to_remove) > 0:
                n.loads_t.p_set = n.loads_t.p_set.drop(columns=cols_to_remove)
    
    # Clean stores with undefined buses
    stores_with_bad_buses = n.stores[~n.stores.bus.isin(buses_to_keep)]
    if len(stores_with_bad_buses) > 0:
        logger.warning(f"Removing {len(stores_with_bad_buses)} stores with undefined buses before solving")
        logger.info(f"Stores with bad buses:\n{stores_with_bad_buses}")
        n.stores = n.stores.drop(stores_with_bad_buses.index)
    
    links_with_bad_bus0 = n.links[~n.links.bus0.isin(buses_to_keep)]
    links_with_bad_bus1 = n.links[~n.links.bus1.isin(buses_to_keep)]
    links_with_bad_buses = links_with_bad_bus0.index.union(links_with_bad_bus1.index)
    
    if len(links_with_bad_buses) > 0:
        logger.warning(f"Removing {len(links_with_bad_buses)} links with undefined buses before solving")
        logger.info(f"Links with bad buses:\n{n.links.loc[links_with_bad_buses]}")
        n.links = n.links.drop(links_with_bad_buses)
        
        for attr in ["p0", "p1", "p", "r", "p_min_pu", "p_max_pu", "efficiency", "marginal_cost"]:
            if hasattr(n.links_t, attr):
                attr_data = getattr(n.links_t, attr)
                if hasattr(attr_data, "columns"):
                    cols_to_remove = links_with_bad_buses.intersection(attr_data.columns)
                    if len(cols_to_remove) > 0:
                        setattr(n.links_t, attr, attr_data.drop(columns=cols_to_remove))

    if skip_iterations:
        logger.info("Solving network without transmission expansion iterations...")
        try:
            status, condition = n.optimize(**kwargs)
            logger.info(f"Initial solve result: status='{status}', condition='{condition}'")
        except Exception as e:
            logger.error(f"Optimization failed with exception: {e}")
            status, condition = "error", str(e)
            raise RuntimeError(f"Optimization failed with exception: {e}") from e
        
        logger.info(f"Saving linopy model to {snakemake.output.lpfile.replace('.lp', '.nc')}")
        try:
            n.model.to_netcdf(snakemake.output.lpfile.replace('.lp', '.nc'))
            logger.info("Linopy model saved successfully")
        except Exception as e:
            logger.warning(f"Could not save linopy model: {e}")
        
        logger.info(f"Saving LP file to {snakemake.output.lpfile}")
        try:
            n.model.to_file(snakemake.output.lpfile)
            logger.info("LP file saved successfully")
        except Exception as e:
            logger.warning(f"Could not save LP file: {e}")
    else:
        logger.info("Solving network with transmission expansion iterations...")
        kwargs["track_iterations"] = cf_solving.get("track_iterations", False)
        kwargs["min_iterations"]   = cf_solving.get("min_iterations", 4)
        kwargs["max_iterations"]   = cf_solving.get("max_iterations", 6)

        try:
            kwargs_iter = dict(kwargs)
            kwargs_iter.pop("log_fn", None)
            status, condition = optimize_transmission_expansion_iteratively(n, **kwargs_iter)

            logger.info(f"Iterative solve result: status='{status}', condition='{condition}'")
        except Exception as e:
            logger.error(f"Iterative optimization failed with exception: {e}")
            status, condition = "error", str(e)
            raise RuntimeError(f"Iterative optimization failed with exception: {e}") from e
        
        try:
            n.model.to_netcdf(snakemake.output.lpfile.replace('.lp', '.nc'))
            logger.info("Linopy model saved successfully")
        except Exception as e:
            logger.warning(f"Could not save linopy model: {e}")
        
        logger.info(f"Saving LP file to {snakemake.output.lpfile}")
        try:
            n.model.to_file(snakemake.output.lpfile)
            logger.info("LP file saved successfully")
        except Exception as e:
            logger.warning(f"Could not save LP file: {e}")
        
    if "infeasible" in condition or "unbounded" in condition or status != 'ok':
        logger.error(f"Solver status: {status}")
        logger.error(f"Termination condition: {condition}")
        
        # Remove components with undefined buses first
        buses_to_keep = set(n.buses.index)
        
        # Clean generators with undefined buses
        generators_with_bad_buses = n.generators.query("bus not in @buses_to_keep")
        if len(generators_with_bad_buses) > 0:
            logger.info(f"Removing {len(generators_with_bad_buses)} generators with undefined buses")
            logger.info(f"Bad generator buses: {generators_with_bad_buses.bus.unique().tolist()}")
            logger.info(f"Generators being removed: {generators_with_bad_buses.index.tolist()}")
            n.generators = n.generators.drop(generators_with_bad_buses.index)
            
            # Clean time series data for removed generators
            for attr in ['p_max_pu', 'p_min_pu']:
                if hasattr(n.generators_t, attr):
                    attr_data = getattr(n.generators_t, attr)
                    cols_to_remove = generators_with_bad_buses.index.intersection(attr_data.columns)
                    if len(cols_to_remove) > 0:
                        logger.info(f"Cleaning {attr} time series for {len(cols_to_remove)} generators")
                        setattr(n.generators_t, attr, attr_data.drop(columns=cols_to_remove))
        
        # Clean loads with undefined buses
        loads_with_bad_buses = n.loads.query("bus not in @buses_to_keep")
        if len(loads_with_bad_buses) > 0:
            logger.info(f"Removing {len(loads_with_bad_buses)} loads with undefined buses")
            logger.info(f"Bad load buses: {loads_with_bad_buses.bus.unique().tolist()}")
            logger.info(f"Loads being removed: {loads_with_bad_buses.index.tolist()}")
            n.loads = n.loads.drop(loads_with_bad_buses.index)
            
            # Clean time series data for removed loads
            if hasattr(n.loads_t, 'p_set'):
                cols_to_remove = loads_with_bad_buses.index.intersection(n.loads_t.p_set.columns)
                if len(cols_to_remove) > 0:
                    logger.info(f"Cleaning p_set time series for {len(cols_to_remove)} loads")
                    n.loads_t.p_set = n.loads_t.p_set.drop(columns=cols_to_remove)
        
        # Clean stores with undefined buses
        stores_with_bad_buses = n.stores.query("bus not in @buses_to_keep")
        if len(stores_with_bad_buses) > 0:
            logger.info(f"Removing {len(stores_with_bad_buses)} stores with undefined buses")
            logger.info(f"Bad store buses: {stores_with_bad_buses.bus.unique().tolist()}")
            logger.info(f"Stores being removed: {stores_with_bad_buses.index.tolist()}")
            n.stores = n.stores.drop(stores_with_bad_buses.index)
        
        # Clean links with undefined buses
        links_with_bad_bus0 = n.links.query("bus0 not in @buses_to_keep")
        links_with_bad_bus1 = n.links.query("bus1 not in @buses_to_keep")
        links_with_bad_buses = links_with_bad_bus0.index.union(links_with_bad_bus1.index)
        
        if len(links_with_bad_buses) > 0:
            logger.info(f"Removing {len(links_with_bad_buses)} links with undefined buses")
            logger.info(f"Links being removed: {links_with_bad_buses.tolist()}")
            
            for link_id in links_with_bad_buses[:10]:
                link = n.links.loc[link_id]
                logger.info(f"  {link_id}: bus0={link.bus0}, bus1={link.bus1}, carrier={link.carrier}")
            
            n.links = n.links.drop(links_with_bad_buses)
            
            # Clean time series data for removed links
            # Clean time series data for removed links (explicit safe list)
            for attr_name in ["p0", "p1", "p", "r", "p_min_pu", "p_max_pu", "efficiency", "marginal_cost"]:
                if hasattr(n.links_t, attr_name):
                    attr_data = getattr(n.links_t, attr_name)
                    if hasattr(attr_data, "columns"):
                        cols_to_remove = links_with_bad_buses.intersection(attr_data.columns)
                        if len(cols_to_remove) > 0:
                            logger.info(f"Cleaning {attr_name} time series for {len(cols_to_remove)} links")
                            setattr(n.links_t, attr_name, attr_data.drop(columns=cols_to_remove))
        
        # Fix generator expansion limits that cause infeasibility
        problematic_gens = n.generators.query("p_nom_max < p_nom_min")
        if len(problematic_gens) > 0:
            logger.info(f"Fixing {len(problematic_gens)} generators with p_nom_max < p_nom_min")
            for idx, gen in problematic_gens.iterrows():
                logger.info(f"  {idx}: p_nom_min={gen.p_nom_min}, p_nom_max={gen.p_nom_max} -> setting p_nom_max = p_nom_min * 2")
                logger.info(f"  {idx}: p_nom_min={gen.p_nom_min} -> p_nom_max={gen.p_nom_min * 2}")
            n.generators.loc[problematic_gens.index, "p_nom_max"] = n.generators.loc[problematic_gens.index, "p_nom_min"] * 2
        
        ## Check for generators with zero or negative costs
        #zero_cost_gens = n.generators.query("marginal_cost <= 0 and p_nom_max > 0")
        #if len(zero_cost_gens) > 0:
        #    logger.info(f"Setting minimum marginal cost for {len(zero_cost_gens)} generators with zero/negative costs")
        #    for idx, gen in zero_cost_gens.iterrows():
        #        logger.info(f"  {idx}: marginal_cost={gen.marginal_cost} -> 0.001")
        #    n.generators.loc[zero_cost_gens.index, "marginal_cost"] = 0.001
        
        # Ensure all extendable generators have reasonable limits
        extendable_gens = n.generators.query("p_nom_extendable == True")
        problematic_extendable = extendable_gens.query("p_nom_max <= p_nom_min or p_nom_max <= 0")
        if len(problematic_extendable) > 0:
            logger.info(f"Fixing {len(problematic_extendable)} extendable generators with problematic limits")
            # Set a reasonable maximum for problematic extendable generators
            n.generators.loc[problematic_extendable.index, "p_nom_max"] = 1e6  # Large but finite limit
            for idx, gen in problematic_extendable.iterrows():
                logger.info(f"  {idx}: p_nom_max set to 1e6 MW")
        
        ## Check for NaN values in critical columns
        #critical_cols = ['marginal_cost', 'p_nom_min', 'p_nom_max']
        #for col in critical_cols:
        #    if col in n.generators.columns:
        #        nan_gens = n.generators[n.generators[col].isna()]
        #        if len(nan_gens) > 0:
        #            logger.info(f"Fixing {len(nan_gens)} generators with NaN values in {col}")
        #            if col == 'marginal_cost':
        #                n.generators.loc[nan_gens.index, col] = 0.001
        #            elif col in ['p_nom_min', 'p_nom_max']:
        #                n.generators.loc[nan_gens.index, col] = 0
        

        logger.info(f"Removed generators: {len(generators_with_bad_buses) if 'generators_with_bad_buses' in locals() else 0}")
        logger.info(f"Removed loads: {len(loads_with_bad_buses) if 'loads_with_bad_buses' in locals() else 0}")
        logger.info(f"Removed stores: {len(stores_with_bad_buses) if 'stores_with_bad_buses' in locals() else 0}")
        logger.info(f"Removed links: {len(links_with_bad_buses) if 'links_with_bad_buses' in locals() else 0}")
        logger.info(f"Fixed generator capacity limits: {len(problematic_gens) if 'problematic_gens' in locals() else 0}")
        logger.info(f"Fixed zero-cost generators: {len(zero_cost_gens) if 'zero_cost_gens' in locals() else 0}")
        
        logger.info("Retrying solve with cleaned network...")

        logger.info("Setting numeric focus parameters for solver...")

        robust_solver_options = {
                "NumericFocus": 3,
                "Method": 2,  # barrier
                "Crossover": 0,
                "BarHomogeneous": 1,
                "BarConvTol": 1e-3,
                "FeasibilityTol": 1e-3,
                "OptimalityTol": 1e-3,
                "Presolve": 2,
                "Aggregate": 1,
                "Threads": 12,
                "Seed": 123
            }
        
        logger.info(f"Using robust solver options: {robust_solver_options}")
        kwargs_robust = kwargs.copy()
        kwargs_robust["solver_options"] = robust_solver_options


        try:
            if skip_iterations:
                status, condition = n.optimize(**kwargs_robust)
            else:
                status, condition = optimize_transmission_expansion_iteratively(n, **kwargs_robust)
                
            logger.info(f"Retry result: status='{status}', condition='{condition}'")
            
            if status != "ok":
                logger.error(f"Retry still failed: {status} / {condition}")
                raise RuntimeError(f"Solving status '{status}' with termination condition '{condition}' after retry")
                
        except Exception as e:
            logger.error(f"Retry failed with exception: {e}")
            raise RuntimeError(f"Retry failed with exception: {e}") from e
        
        # If we reach here, retry was successful
        return n

    return n


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_network_myopic",
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

    opts = snakemake.wildcards.opts.split("-")
    solve_opts = snakemake.config["solving"]["options"]

    is_sector_coupled = "sopts" in snakemake.wildcards.keys()

    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)

    if snakemake.params.augmented_line_connection.get("add_to_snakefile"):
        n.lines.loc[n.lines.index.str.contains("new"), "s_nom_min"] = (
            snakemake.params.augmented_line_connection.get("min_expansion")
        )

    if (
        snakemake.config["custom_data"]["add_existing"]
        and snakemake.wildcards.planning_horizons == "2050"
        and is_sector_coupled
    ):
        add_existing(n)

    if (
        snakemake.config["policy_config"]["hydrogen"]["additionality"]
        and not snakemake.config["policy_config"]["hydrogen"]["is_reference"]
        and snakemake.config["policy_config"]["hydrogen"]["temporal_matching"]
        != "no_res_matching"
        and is_sector_coupled
    ):
        n_ref_path = snakemake.config["policy_config"]["hydrogen"]["path_to_ref"]
        n_ref = pypsa.Network(n_ref_path)
    else:
        n_ref = None

    # attach to network so extra_functionality can access safely
    # (works whether additionality/monthly is on or off)
    # Will be replaced when we rebuild/modify n later in this file.
    # We set it again just before solve, after we have the final 'n'.

    n = prepare_network(n, solve_opts)
    # ensure monthly constraints can access reference network if present
    n.n_ref = n_ref

    # ---- TSAM: aggregate to typical periods (ONLY if enabled) ----
    skip = True
    if skip:
        logger.info("Temporal clustering in prepare_sector_network")
    else:
        temporal_cfg = snakemake.config.get("temporal_clustering", {})
        if temporal_cfg.get("activate", False):
            # Guard against accidentally using both nhours and TSAM
            if solve_opts.get("nhours"):
                logger.warning("Both 'nhours' and TSAM requested. Proceeding with TSAM; ignoring nhours downsampling semantics.")
            
            logger.info("Applying temporal clustering (TSAM) after network preparation...")
            logger.info(f"TSAM parameters: n_periods={temporal_cfg.get('n_periods', 10)}, hours={temporal_cfg.get('hours', 24)}, method={temporal_cfg.get('clusterMethod', 'hierarchical')}")
            
            aggregate_snapshots(
                n,
                n_periods=temporal_cfg.get("n_periods", 10),
                hours=temporal_cfg.get("hours", 24),
                normed=temporal_cfg.get("normed", True),
                solver=temporal_cfg.get("solver", "glpk"),
                extremePeriodMethod=temporal_cfg.get("extremePeriodMethod", "None"),
                clusterMethod=temporal_cfg.get("clusterMethod", "hierarchical"),
                predefClusterOrder=None,
                overwrite_time_dfs=temporal_cfg.get("overwrite_time_dfs", False),
            )
            logger.info(f"TSAM aggregation complete. Network now has {len(n.snapshots)} snapshots; period_id persisted.")

    n = solve_network(
        n,
        config=snakemake.config,
        solving=snakemake.params.solving,
        log_fn=_safe_solver_log(snakemake),
    )
    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
    logger.info(f"Objective constant: {n.objective_constant}")
