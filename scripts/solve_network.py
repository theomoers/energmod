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
import pypsa.clustering.spatial as pypsa_spatial
import xarray as xr
import yaml
import validation as _validation_hooks
from _helpers import (
    configure_logging,
    create_logger,
    override_component_attrs,
    three_2_two_digits_country,
)
from linopy import merge
from pypsa.clustering.spatial import (
    DEFAULT_ONE_PORT_STRATEGIES,
    get_clustering_from_busmap,
)
from temporal_clustering import add_kotzur_storage_constraints
from pypsa.descriptors import get_switchable_as_dense as get_as_dense
from pypsa.optimization.abstract import optimize_transmission_expansion_iteratively
#from apply_build_constraints import add_build_rate_constraints
#from pypsa.optimization.optimize import optimize

logger = create_logger(__name__)
pypsa.pf.logger.setLevel(logging.WARNING)


# Baseyear generation validation helpers moved to scripts/validation.py


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


def _repo_path(path_like):
    p = Path(path_like)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[1] / p
    return p.resolve()


def _zscore_columns(df):
    out = df.copy()
    for col in out.columns:
        s = out[col].astype(float)
        std = float(s.std(ddof=0))
        if std > 0:
            out[col] = (s - float(s.mean())) / std
        else:
            out[col] = 0.0
    return out.fillna(0.0)


def _get_bus_country_for_clustering(n):
    country = (
        n.buses["country"]
        if "country" in n.buses.columns
        else pd.Series("", index=n.buses.index)
    )
    location = (
        n.buses["location"]
        if "location" in n.buses.columns
        else pd.Series("", index=n.buses.index)
    )
    country = country.fillna("").astype(str).str.strip().str.upper()
    location = location.fillna("").astype(str).str.strip().str.upper()
    bus_name = n.buses.index.to_series(index=n.buses.index).astype(str).str.strip().str.upper()

    def _extract_cc(series):
        return series.str.extract(r"^([A-Z]{2})(?:\b|\s|[-_])", expand=False).fillna("")

    inferred = _extract_cc(location)
    country = country.where(country != "", inferred)

    inferred_name = _extract_cc(bus_name)
    country = country.where(country != "", inferred_name)

    return country.fillna("")


def _country_electric_load_profiles(n, countries):
    valid_loads = n.loads.index[n.loads.bus.isin(n.buses.index)]
    if len(valid_loads) == 0:
        return pd.DataFrame(0.0, index=n.snapshots, columns=countries)

    load_bus = n.loads.loc[valid_loads, "bus"]
    bus_country = _get_bus_country_for_clustering(n)
    load_country = load_bus.map(bus_country)
    load_bus_carrier = load_bus.map(n.buses.carrier)
    ac_loads = load_bus_carrier[load_bus_carrier == "AC"].index

    if len(ac_loads) == 0 or "p_set" not in n.loads_t:
        return pd.DataFrame(0.0, index=n.snapshots, columns=countries)

    load_ts = n.loads_t.p_set.reindex(columns=ac_loads).fillna(0.0)
    country_profiles = load_ts.T.groupby(load_country.loc[ac_loads]).sum().T
    return country_profiles.reindex(columns=countries, fill_value=0.0)


def _ac_bus_load_profiles(n, ac_buses):
    ac_buses = pd.Index(ac_buses)
    valid_loads = n.loads.index[n.loads.bus.isin(ac_buses)]
    if len(valid_loads) == 0 or "p_set" not in n.loads_t:
        return pd.DataFrame(0.0, index=n.snapshots, columns=ac_buses)

    load_ts = n.loads_t.p_set.reindex(columns=valid_loads).fillna(0.0)
    load_bus = n.loads.loc[valid_loads, "bus"]
    bus_profiles = load_ts.T.groupby(load_bus).sum().T
    return bus_profiles.reindex(columns=ac_buses, fill_value=0.0)


def _parse_manual_country_clusters(mapping_file):
    with open(mapping_file, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Manual clustering file is empty: {mapping_file}")

    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = enumerate(raw, start=1)
    else:
        raise TypeError(
            f"Manual clustering must be dict/list in {mapping_file}; got {type(raw)}"
        )

    country_to_cluster = {}
    for cluster_id, countries in items:
        if countries is None:
            continue
        if isinstance(countries, str):
            countries = [countries]
        if not isinstance(countries, (list, tuple, set)):
            raise TypeError(
                f"Cluster entry {cluster_id} must be list/tuple/set/string, got {type(countries)}"
            )

        for country in countries:
            cc = str(country).strip().upper()
            if cc in {"", "NONE", "NAN"}:
                continue
            if cc in country_to_cluster and country_to_cluster[cc] != str(cluster_id):
                raise ValueError(
                    f"Country '{cc}' appears in multiple clusters "
                    f"({country_to_cluster[cc]} and {cluster_id}) in {mapping_file}"
                )
            country_to_cluster[cc] = str(cluster_id)

    if not country_to_cluster:
        raise ValueError(f"No valid country assignments found in {mapping_file}")

    return pd.Series(country_to_cluster, name="cluster")


def _build_ac_bus_feature_frame(n, ac_buses, pca_components, random_state):
    ac_buses = pd.Index(ac_buses)
    profiles = _ac_bus_load_profiles(n, ac_buses)
    annual_demand = profiles.sum(axis=0).reindex(ac_buses).fillna(0.0).clip(lower=0.0)

    geo = n.buses.reindex(ac_buses)[["x", "y"]].rename(columns={"x": "geo_x", "y": "geo_y"})

    gen_bus = n.generators.bus
    gen_carrier = n.generators.carrier.astype(str).str.lower()
    p_nom_max = n.generators.p_nom_max.where(
        np.isfinite(n.generators.p_nom_max), np.nan
    )
    gen_cap = p_nom_max.fillna(n.generators.p_nom).fillna(0.0).clip(lower=0.0)
    on_ac = gen_bus.isin(ac_buses)
    gen_bus = gen_bus[on_ac]
    gen_carrier = gen_carrier[on_ac]
    gen_cap = gen_cap[on_ac]

    wind = gen_cap[gen_carrier.str.contains("wind", na=False)].groupby(gen_bus).sum()
    solar = gen_cap[gen_carrier.str.contains("solar", na=False)].groupby(gen_bus).sum()
    hydro_gen = gen_cap[gen_carrier.isin(["hydro", "ror"])].groupby(gen_bus).sum()
    fossil = gen_cap[
        gen_carrier.str.contains("coal|lignite|oil|gas|ocgt|ccgt", regex=True, na=False)
    ].groupby(gen_bus).sum()

    su_bus = n.storage_units.bus
    su_carrier = n.storage_units.carrier.astype(str).str.lower()
    su_cap = n.storage_units.p_nom.fillna(0.0).clip(lower=0.0)
    su_on_ac = su_bus.isin(ac_buses)
    su_bus = su_bus[su_on_ac]
    su_carrier = su_carrier[su_on_ac]
    su_cap = su_cap[su_on_ac]
    hydro_su = su_cap[su_carrier.isin(["hydro", "phs"])].groupby(su_bus).sum()
    hydro = hydro_gen.add(hydro_su, fill_value=0.0)

    installed = gen_cap.groupby(gen_bus).sum()
    renewable = gen_cap[
        gen_carrier.str.contains("wind|solar|hydro|ror", regex=True, na=False)
    ].groupby(gen_bus).sum()
    capmix_renew = renewable / installed.replace(0.0, np.nan)
    capmix_fossil = fossil / installed.replace(0.0, np.nan)

    pcs = pd.DataFrame(index=ac_buses)
    if len(profiles.columns) > 0 and len(profiles.index) > 0 and pca_components > 0:
        shape = profiles.divide(
            profiles.mean(axis=0).replace(0.0, np.nan), axis=1
        ).fillna(0.0)
        X = shape.T.reindex(index=ac_buses).fillna(0.0).to_numpy(dtype=float)
        X_std = X.std(axis=0)
        X = (X - X.mean(axis=0)) / np.where(X_std > 0, X_std, 1.0)
        max_components = min(pca_components, X.shape[0], X.shape[1])
        if max_components > 0:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=max_components, random_state=random_state)
            scores = pca.fit_transform(X)
            pcs = pd.DataFrame(
                scores,
                index=ac_buses,
                columns=[f"load_pc{i+1}" for i in range(max_components)],
            )

    feat = pd.DataFrame(index=ac_buses)
    feat = feat.join(geo)
    feat["res_wind"] = wind.reindex(ac_buses).fillna(0.0)
    feat["res_solar"] = solar.reindex(ac_buses).fillna(0.0)
    feat["res_hydro"] = hydro.reindex(ac_buses).fillna(0.0)
    feat["fossil_proxy"] = fossil.reindex(ac_buses).fillna(0.0)
    feat["capmix_renew_share"] = capmix_renew.reindex(ac_buses).fillna(0.0)
    feat["capmix_fossil_share"] = capmix_fossil.reindex(ac_buses).fillna(0.0)
    if not pcs.empty:
        feat = feat.join(pcs)

    feat = _zscore_columns(feat.fillna(0.0))
    return feat, annual_demand


def _build_ac_bus_cf_pca_features(
    n,
    ac_buses,
    random_state,
    n_components=3,
    by_tech=True,
    tech_groups=None,
):
    ac_buses = pd.Index(ac_buses)
    n_components = int(n_components)
    if n_components <= 0:
        return pd.DataFrame(index=ac_buses)
    if "p_max_pu" not in n.generators_t:
        return pd.DataFrame(index=ac_buses)

    gens = n.generators[n.generators.bus.isin(ac_buses)].copy()
    if gens.empty:
        return pd.DataFrame(index=ac_buses)

    p_nom_max = gens.p_nom_max.where(np.isfinite(gens.p_nom_max), np.nan)
    weights = p_nom_max.fillna(gens.p_nom).fillna(0.0).clip(lower=0.0)
    if float(weights.sum()) <= 0:
        weights[:] = 1.0

    cf_dense = get_as_dense(n, "Generator", "p_max_pu", inds=gens.index).fillna(0.0)
    carrier = gens.carrier.astype(str).str.lower()

    if tech_groups is None:
        tech_groups = {
            "solar": ["solar"],
            "onwind": ["onwind"],
            "offwind": ["offwind"],
            "hydro": ["hydro", "ror"],
        }

    if by_tech:
        groups = []
        for name, tokens in tech_groups.items():
            if isinstance(tokens, str):
                tokens = [tokens]
            if not tokens:
                continue
            mask = pd.Series(False, index=carrier.index)
            for token in tokens:
                t = str(token).strip().lower()
                if t == "":
                    continue
                mask = mask | carrier.str.contains(t, na=False)
            idx = carrier.index[mask]
            if len(idx) > 0:
                groups.append((str(name), idx))
    else:
        groups = [("all", carrier.index)]

    if not groups:
        return pd.DataFrame(index=ac_buses)

    all_features = pd.DataFrame(index=ac_buses)
    for group_name, gen_idx in groups:
        g_weights = weights.reindex(gen_idx).fillna(0.0)
        if float(g_weights.sum()) <= 0:
            continue

        g_cf = cf_dense.reindex(columns=gen_idx).fillna(0.0)
        g_bus = gens.loc[gen_idx, "bus"]

        weighted = g_cf.mul(g_weights, axis=1)
        bus_weighted_sum = weighted.T.groupby(g_bus).sum().T
        bus_weight_sum = g_weights.groupby(g_bus).sum()
        bus_cf = bus_weighted_sum.div(bus_weight_sum.replace(0.0, np.nan), axis=1).fillna(0.0)
        bus_cf = bus_cf.reindex(columns=ac_buses, fill_value=0.0)

        X = bus_cf.T.to_numpy(dtype=float)
        if X.shape[0] == 0 or X.shape[1] == 0:
            continue
        X_std = X.std(axis=0)
        X = (X - X.mean(axis=0)) / np.where(X_std > 0, X_std, 1.0)
        max_components = min(n_components, X.shape[0], X.shape[1])
        if max_components <= 0:
            continue

        from sklearn.decomposition import PCA

        pca = PCA(n_components=max_components, random_state=random_state)
        scores = pca.fit_transform(X)
        cols = [f"cfpc_{group_name}_{i+1}" for i in range(max_components)]
        group_features = pd.DataFrame(scores, index=ac_buses, columns=cols)
        all_features = all_features.join(group_features, how="left")

    return _zscore_columns(all_features.fillna(0.0)) if not all_features.empty else all_features


def _build_country_feature_frame(n, countries, pca_components, random_state):
    profiles = _country_electric_load_profiles(n, countries)
    annual_demand = profiles.sum(axis=0).reindex(countries).fillna(0.0).clip(lower=0.0)

    bus_country = _get_bus_country_for_clustering(n)
    ac_buses = n.buses[n.buses.carrier == "AC"]
    if ac_buses.empty:
        ac_buses = n.buses.copy()
    geo = ac_buses.assign(_country=bus_country.loc[ac_buses.index]).groupby("_country")[["x", "y"]].mean().reindex(countries)
    geo.columns = ["geo_x", "geo_y"]

    gen_country = n.generators.bus.map(bus_country)
    gen_carrier = n.generators.carrier.astype(str).str.lower()
    p_nom_max = n.generators.p_nom_max.where(
        np.isfinite(n.generators.p_nom_max), np.nan
    )
    gen_cap = p_nom_max.fillna(n.generators.p_nom).fillna(0.0).clip(lower=0.0)

    wind = gen_cap[gen_carrier.str.contains("wind", na=False)].groupby(gen_country).sum()
    solar = gen_cap[gen_carrier.str.contains("solar", na=False)].groupby(gen_country).sum()
    hydro_gen = gen_cap[gen_carrier.isin(["hydro", "ror"])].groupby(gen_country).sum()
    fossil = gen_cap[
        gen_carrier.str.contains("coal|lignite|oil|gas|ocgt|ccgt", regex=True, na=False)
    ].groupby(gen_country).sum()

    su_country = n.storage_units.bus.map(bus_country)
    su_carrier = n.storage_units.carrier.astype(str).str.lower()
    su_cap = n.storage_units.p_nom.fillna(0.0).clip(lower=0.0)
    hydro_su = su_cap[su_carrier.isin(["hydro", "phs"])].groupby(su_country).sum()

    hydro = hydro_gen.add(hydro_su, fill_value=0.0)

    installed = n.generators.p_nom.fillna(0.0).clip(lower=0.0).groupby(gen_country).sum()
    renewable = gen_cap[
        gen_carrier.str.contains("wind|solar|hydro|ror", regex=True, na=False)
    ].groupby(gen_country).sum()

    capmix_renew = renewable / installed.replace(0.0, np.nan)
    capmix_fossil = fossil / installed.replace(0.0, np.nan)

    pcs = pd.DataFrame(index=countries)
    if len(profiles.columns) > 0 and len(profiles.index) > 0 and pca_components > 0:
        shape = profiles.divide(
            profiles.mean(axis=0).replace(0.0, np.nan), axis=1
        ).fillna(0.0)
        X = shape.T.reindex(index=countries).fillna(0.0).to_numpy(dtype=float)
        X_std = X.std(axis=0)
        X = (X - X.mean(axis=0)) / np.where(X_std > 0, X_std, 1.0)
        max_components = min(pca_components, X.shape[0], X.shape[1])
        if max_components > 0:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=max_components, random_state=random_state)
            scores = pca.fit_transform(X)
            pcs = pd.DataFrame(
                scores,
                index=countries,
                columns=[f"load_pc{i+1}" for i in range(max_components)],
            )

    feat = pd.DataFrame(index=countries)
    feat = feat.join(geo)
    feat["res_wind"] = wind.reindex(countries).fillna(0.0)
    feat["res_solar"] = solar.reindex(countries).fillna(0.0)
    feat["res_hydro"] = hydro.reindex(countries).fillna(0.0)
    feat["fossil_proxy"] = fossil.reindex(countries).fillna(0.0)
    feat["capmix_renew_share"] = capmix_renew.reindex(countries).fillna(0.0)
    feat["capmix_fossil_share"] = capmix_fossil.reindex(countries).fillna(0.0)
    if not pcs.empty:
        feat = feat.join(pcs)

    feat = _zscore_columns(feat.fillna(0.0))
    return feat, annual_demand


def _build_ac_bus_adjacency(n, ac_buses):
    ac_buses = pd.Index(ac_buses)
    ac_set = set(ac_buses)
    adj = {b: set() for b in ac_buses}

    def _add_edges(df):
        for b0, b1 in df.itertuples(index=False, name=None):
            if b0 not in ac_set or b1 not in ac_set or b0 == b1:
                continue
            adj[b0].add(b1)
            adj[b1].add(b0)

    _add_edges(n.lines[["bus0", "bus1"]])
    _add_edges(n.links[["bus0", "bus1"]])
    return adj


def _build_country_adjacency(n, countries):
    countries = pd.Index(countries)
    bus_country = _get_bus_country_for_clustering(n)
    adj = {c: set() for c in countries}
    ac_buses = set(n.buses.index[n.buses.carrier == "AC"])

    def _add_edges(df):
        for b0, b1 in df.itertuples(index=False, name=None):
            if b0 not in ac_buses or b1 not in ac_buses:
                continue
            c0 = bus_country.get(b0)
            c1 = bus_country.get(b1)
            if pd.isna(c0) or pd.isna(c1) or c0 == c1:
                continue
            if c0 in adj and c1 in adj:
                adj[c0].add(c1)
                adj[c1].add(c0)

    _add_edges(n.lines[["bus0", "bus1"]])
    _add_edges(n.links[["bus0", "bus1"]])
    return adj


def _connected_components(nodes, adjacency):
    nodes_set = set(nodes)
    seen = set()
    components = []
    for start in sorted(nodes_set):
        if start in seen:
            continue
        stack = [start]
        comp = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            comp.add(cur)
            for nxt in sorted(adjacency.get(cur, set()), reverse=True):
                if nxt in nodes_set and nxt not in seen:
                    stack.append(nxt)
        components.append(comp)
    return components


def _enforce_node_cluster_contiguity(labels, features, nodes, adjacency, max_iter=10):
    node_labels = pd.Series(labels, index=nodes)
    for _ in range(max_iter):
        changed = False
        unique_labels = sorted(pd.unique(node_labels.values))
        centroids = features.groupby(node_labels).mean()

        for label in unique_labels:
            members = sorted(node_labels[node_labels == label].index.tolist())
            if len(members) <= 1:
                continue
            comps = _connected_components(members, adjacency)
            if len(comps) <= 1:
                continue

            comps_sorted = sorted(
                (sorted(comp) for comp in comps),
                key=lambda comp: (-len(comp), comp[0]),
            )
            for comp in comps_sorted[1:]:
                neighbor_labels = set()
                for node in comp:
                    neighbors = sorted(adjacency.get(node, set()))
                    if neighbors:
                        neighbor_labels.update(node_labels.loc[neighbors].tolist())
                neighbor_labels.discard(label)
                if not neighbor_labels:
                    neighbor_labels = set(unique_labels) - {label}
                if not neighbor_labels:
                    continue

                comp_center = features.loc[comp].mean().to_numpy(dtype=float)
                best_label = min(
                    sorted(neighbor_labels),
                    key=lambda l: np.linalg.norm(
                        comp_center - centroids.loc[l].to_numpy(dtype=float)
                    ),
                )
                node_labels.loc[comp] = best_label
                changed = True

        if not changed:
            break

    return node_labels.reindex(nodes).to_numpy()


def _enforce_country_cluster_contiguity(labels, features, countries, adjacency, max_iter=10):
    country_labels = pd.Series(labels, index=countries)
    for _ in range(max_iter):
        changed = False
        unique_labels = sorted(pd.unique(country_labels.values))
        centroids = features.groupby(country_labels).mean()

        for label in unique_labels:
            members = sorted(country_labels[country_labels == label].index.tolist())
            if len(members) <= 1:
                continue
            comps = _connected_components(members, adjacency)
            if len(comps) <= 1:
                continue

            comps_sorted = sorted(
                (sorted(comp) for comp in comps),
                key=lambda comp: (-len(comp), comp[0]),
            )
            largest = comps_sorted[0]
            for comp in comps_sorted[1:]:
                if comp == largest:
                    continue
                neighbor_labels = set()
                for c in comp:
                    neighbors = sorted(adjacency.get(c, set()))
                    if neighbors:
                        neighbor_labels.update(country_labels.loc[neighbors].tolist())
                neighbor_labels.discard(label)
                if not neighbor_labels:
                    neighbor_labels = set(unique_labels) - {label}
                if not neighbor_labels:
                    continue

                comp_center = features.loc[comp].mean().to_numpy(dtype=float)
                best_label = min(
                    sorted(neighbor_labels),
                    key=lambda l: np.linalg.norm(
                        comp_center - centroids.loc[l].to_numpy(dtype=float)
                    ),
                )
                country_labels.loc[comp] = best_label
                changed = True

        if not changed:
            break

    return country_labels.reindex(countries).to_numpy()


def _extract_ac_anchor_bus(bus_name):
    if not isinstance(bus_name, str):
        return None
    m = re.match(r"^([A-Z]{2}\s+\d+)\b", bus_name.strip().upper())
    if not m:
        return None
    return m.group(1)


def _vintage_token_numeric(value, integer=False):
    if pd.isna(value):
        return "nan", np.nan
    try:
        v = float(value)
    except Exception:
        return "nan", np.nan
    if np.isinf(v):
        return "inf", np.inf
    if integer:
        vint = int(round(v))
        return str(vint), float(vint)
    return f"{v:.12g}", v


def _decoded_vintage_label(base, build_year, lifetime, include_lifetime=False):
    base = str(base)
    suffix = ""
    if np.isfinite(build_year) and float(build_year) > 0:
        suffix = f"-{int(round(float(build_year)))}"
    label = f"{base}{suffix}"
    if include_lifetime and np.isfinite(lifetime) and float(lifetime) > 0:
        label = f"{label}_l{int(round(float(lifetime)))}"
    return label


def _replace_encoded_token(name, token_to_meta, include_lifetime=False):
    text = str(name)
    for token, (base, build_year, lifetime) in token_to_meta.items():
        if token in text:
            repl = _decoded_vintage_label(
                base, build_year, lifetime, include_lifetime=include_lifetime
            )
            return text.replace(token, repl), (base, build_year, lifetime)
    return text, None


def _sanitize_vintage_encoded_component_names(n, token_to_meta, components):
    if not token_to_meta:
        return

    processed = set()
    for component in components:
        if component in processed or component not in n.components:
            continue
        processed.add(component)
        list_name = n.components[component]["list_name"]
        df = getattr(n, list_name, None)
        if df is None or df.empty:
            continue

        rename_map = {}
        meta_by_old = {}
        for old in df.index.astype(str):
            if "__vintage_y" not in old:
                continue
            new, meta = _replace_encoded_token(old, token_to_meta, include_lifetime=False)
            if new != old:
                rename_map[old] = new
                meta_by_old[old] = meta

        if not rename_map:
            continue

        # Resolve collisions deterministically by adding lifetime token first.
        inverse = {}
        for old, new in rename_map.items():
            inverse.setdefault(new, []).append(old)

        for new, olds in inverse.items():
            if len(olds) <= 1:
                continue
            for i, old in enumerate(sorted(olds)):
                alt, meta = _replace_encoded_token(old, token_to_meta, include_lifetime=True)
                if alt in rename_map.values() and rename_map.get(old) != alt:
                    alt = f"{alt}__v{i}"
                rename_map[old] = alt
                meta_by_old[old] = meta

        # Rename static index
        df.rename(index=rename_map, inplace=True)

        # Rename associated time series columns
        pnl = getattr(n, f"{list_name}_t")
        for attr, data in pnl.items():
            if data.empty or not hasattr(data, "columns"):
                continue
            cols_map = {c: rename_map[c] for c in data.columns if c in rename_map}
            if cols_map:
                pnl[attr] = data.rename(columns=cols_map)


def _encode_vintage_carriers_for_clustering(n, components):
    """
    Temporarily encode build_year/lifetime into carrier names before clustering.

    PyPSA default aggregation strategies can reset build_year/lifetime for one-port
    components. Encoding vintages into the grouping key preserves vintage classes.
    """
    carrier_backups = {}
    token_to_meta = {}
    processed = set()

    for component in components:
        if component in processed or component not in n.components:
            continue
        processed.add(component)
        list_name = n.components[component]["list_name"]
        df = getattr(n, list_name, None)
        if df is None or df.empty or "carrier" not in df.columns:
            continue
        if "build_year" not in df.columns and "lifetime" not in df.columns:
            continue

        carrier_backups[list_name] = df["carrier"].copy()
        carriers = df["carrier"].fillna("").astype(str)
        build_year = (
            pd.to_numeric(df["build_year"], errors="coerce")
            if "build_year" in df.columns
            else pd.Series(np.nan, index=df.index)
        )
        lifetime = (
            pd.to_numeric(df["lifetime"], errors="coerce")
            if "lifetime" in df.columns
            else pd.Series(np.nan, index=df.index)
        )

        encoded = carriers.copy()
        for idx in df.index:
            base = carriers.at[idx]
            y_token, y_value = _vintage_token_numeric(build_year.at[idx], integer=True)
            l_token, l_value = _vintage_token_numeric(lifetime.at[idx], integer=False)
            token = f"{base}__vintage_y{y_token}_l{l_token}"
            encoded.at[idx] = token
            token_to_meta[token] = (base, y_value, l_value)

        df.loc[:, "carrier"] = encoded.values

    return carrier_backups, token_to_meta


def _restore_encoded_carriers(n, carrier_backups):
    for list_name, original in carrier_backups.items():
        df = getattr(n, list_name, None)
        if df is None or df.empty or "carrier" not in df.columns:
            continue
        keep = original.index.intersection(df.index)
        if len(keep):
            df.loc[keep, "carrier"] = original.loc[keep].values


def _decode_vintage_carriers_after_clustering(n, token_to_meta, components):
    if not token_to_meta:
        return

    processed = set()
    for component in components:
        if component in processed or component not in n.components:
            continue
        processed.add(component)
        list_name = n.components[component]["list_name"]
        df = getattr(n, list_name, None)
        if df is None or df.empty or "carrier" not in df.columns:
            continue

        carriers = df["carrier"].fillna("").astype(str)
        mask = carriers.isin(token_to_meta)
        if not mask.any():
            continue

        meta = carriers.loc[mask].map(token_to_meta)
        decoded_carrier = [m[0] for m in meta.values]
        decoded_build_year = [m[1] for m in meta.values]
        decoded_lifetime = [m[2] for m in meta.values]

        df.loc[mask, "carrier"] = decoded_carrier
        if "build_year" in df.columns:
            df.loc[:, "build_year"] = pd.to_numeric(df["build_year"], errors="coerce")
            df.loc[mask, "build_year"] = np.asarray(decoded_build_year, dtype=float)
        if "lifetime" in df.columns:
            df.loc[:, "lifetime"] = pd.to_numeric(df["lifetime"], errors="coerce")
            df.loc[mask, "lifetime"] = np.asarray(decoded_lifetime, dtype=float)


def _cluster_tokens_by_dominant_country(ac_cluster_ids, bus_country, ac_weights=None):
    ac_cluster_ids = ac_cluster_ids.astype(str)
    ac_buses = pd.Index(ac_cluster_ids.index)

    if ac_weights is None:
        w = pd.Series(1.0, index=ac_buses)
    else:
        w = ac_weights.reindex(ac_buses).fillna(0.0).clip(lower=0.0)
        if float(w.sum()) <= 0:
            w[:] = 1.0

    df = pd.DataFrame(
        {
            "cluster": ac_cluster_ids.reindex(ac_buses).values,
            "country": bus_country.reindex(ac_buses).fillna("").astype(str).values,
            "weight": w.values,
        },
        index=ac_buses,
    )

    cluster_totals = df.groupby("cluster")["weight"].sum()
    cluster_country_weights = (
        df.query("country != ''").groupby(["cluster", "country"])["weight"].sum()
    )

    dominant_country = {}
    for cluster in sorted(pd.unique(df["cluster"])):
        if (
            isinstance(cluster_country_weights.index, pd.MultiIndex)
            and cluster in cluster_country_weights.index.get_level_values(0)
        ):
            cw = cluster_country_weights.xs(cluster, level=0)
            max_w = float(cw.max())
            top = sorted(cw[cw == max_w].index.astype(str).tolist())[0]
            dominant_country[cluster] = top
        else:
            dominant_country[cluster] = "ZZ"

    dominant_series = pd.Series(dominant_country)
    cluster_tokens = {}
    for country in sorted(dominant_series.unique()):
        clusters = dominant_series[dominant_series == country].index.tolist()
        clusters = sorted(
            clusters,
            key=lambda c: (
                -float(cluster_totals.get(c, 0.0)),
                str(c),
            ),
        )
        for i, c in enumerate(clusters):
            cluster_tokens[c] = f"{country} {i}"

    return pd.Series(cluster_tokens, name="cluster_token")


def _ac_cluster_to_busmap(n, ac_cluster_ids, ac_weights=None):
    ac_cluster_ids = ac_cluster_ids.astype(str)
    ac_buses = pd.Index(ac_cluster_ids.index)
    buses = n.buses.index.astype(str)
    bus_country = _get_bus_country_for_clustering(n).astype(str).str.strip()
    carriers = n.buses.carrier.fillna("unknown").astype(str)

    keep_identity = bus_country == ""
    keep_count = int(keep_identity.sum())
    if keep_count > 0:
        logger.info(
            "Preserving %d global buses without country code as identity buses (not clustered).",
            keep_count,
        )

    if ac_weights is None:
        ac_weights = pd.Series(1.0, index=ac_buses)
    else:
        ac_weights = ac_weights.reindex(ac_buses).fillna(0.0)
        if float(ac_weights.sum()) <= 0:
            ac_weights[:] = 1.0

    cluster_tokens = _cluster_tokens_by_dominant_country(
        ac_cluster_ids=ac_cluster_ids,
        bus_country=bus_country,
        ac_weights=ac_weights,
    )
    ac_bus_tokens = ac_cluster_ids.reindex(ac_buses).map(cluster_tokens)
    country_default_token = _country_default_tokens(n, ac_bus_tokens)

    mapped = pd.Series(index=n.buses.index, dtype=object, name="busmap")
    mapped.loc[ac_buses] = (
        ac_bus_tokens.astype(str) + " " + carriers.reindex(ac_buses)
    )

    for b in n.buses.index:
        if b in ac_buses:
            continue
        if keep_identity.get(b, False):
            mapped.at[b] = b
            continue

        anchor = _extract_ac_anchor_bus(str(b))
        cluster_id = None
        if anchor in ac_cluster_ids.index:
            cluster_id = ac_cluster_ids.at[anchor]
        else:
            loc = str(n.buses.at[b, "location"]).strip()
            if loc in ac_cluster_ids.index:
                cluster_id = ac_cluster_ids.at[loc]
        token = None
        if cluster_id is not None:
            token = cluster_tokens.get(str(cluster_id))
        if token is None and bus_country.get(b, "") in country_default_token:
            token = country_default_token[bus_country.at[b]]

        if token is None or str(token).strip() == "":
            mapped.at[b] = b
        else:
            mapped.at[b] = f"{token} {carriers.at[b]}"

    return mapped


def _ac_bus_kmeans_mapping(n, cfg):
    ac_buses = pd.Index(n.buses.index[n.buses.carrier == "AC"])
    if len(ac_buses) == 0:
        raise ValueError("No AC buses available for kmeans clustering.")

    n_clusters = int(cfg.get("n_clusters", 90))
    random_state = int(cfg.get("random_state", 42))
    pca_components = int(cfg.get("pca_components", 5))
    if n_clusters <= 0:
        raise ValueError(f"n_clusters must be positive, got {n_clusters}")
    if n_clusters > len(ac_buses):
        raise ValueError(
            f"Requested {n_clusters} clusters for {len(ac_buses)} AC buses."
        )

    features, annual_demand = _build_ac_bus_feature_frame(
        n, ac_buses, pca_components, random_state
    )
    include_cf_features = bool(cfg.get("include_p_max_pu_features", False))
    if include_cf_features:
        cf_features = _build_ac_bus_cf_pca_features(
            n=n,
            ac_buses=ac_buses,
            random_state=random_state,
            n_components=int(cfg.get("p_max_pu_components", 3)),
            by_tech=bool(cfg.get("p_max_pu_by_tech", True)),
            tech_groups=cfg.get("p_max_pu_tech_groups"),
        )
        if not cf_features.empty:
            features = features.join(cf_features, how="left")
            logger.info(
                "Added %d p_max_pu-based PCA feature columns for AC-bus clustering.",
                len(cf_features.columns),
            )
        else:
            logger.warning(
                "include_p_max_pu_features=true but no p_max_pu PCA features were produced."
            )
    features = _zscore_columns(features.fillna(0.0))

    feature_weights = {
        "geography": 1.0,
        "load_pca": 2.0,
        "resource": 1.0,
        "fossil": 1.0,
        "capmix": 0.5,
        "cf_timeseries": 1.0,
    }
    feature_weights.update(cfg.get("feature_weights", {}))

    weighted_features = features.copy()
    load_pc_cols = [c for c in weighted_features.columns if c.startswith("load_pc")]
    grouped_cols = {
        "geography": [c for c in ["geo_x", "geo_y"] if c in weighted_features.columns],
        "load_pca": load_pc_cols,
        "resource": [
            c
            for c in ["res_wind", "res_solar", "res_hydro"]
            if c in weighted_features.columns
        ],
        "fossil": [c for c in ["fossil_proxy"] if c in weighted_features.columns],
        "capmix": [
            c
            for c in ["capmix_renew_share", "capmix_fossil_share"]
            if c in weighted_features.columns
        ],
        "cf_timeseries": [
            c for c in weighted_features.columns if c.startswith("cfpc_")
        ],
    }
    for group, cols in grouped_cols.items():
        if not cols:
            continue
        weighted_features.loc[:, cols] = (
            weighted_features.loc[:, cols] * float(feature_weights[group])
        )

    sample_weight = annual_demand.reindex(ac_buses).fillna(0.0).clip(lower=0.0)
    if float(sample_weight.sum()) <= 0:
        sample_weight[:] = 1.0
    else:
        positive = sample_weight[sample_weight > 0]
        floor = float(positive.min()) if len(positive) else 1.0
        sample_weight = sample_weight.where(sample_weight > 0, floor)

    from sklearn.cluster import KMeans

    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=int(cfg.get("n_init", 50)),
        max_iter=int(cfg.get("max_iter", 1000)),
    )
    labels = km.fit_predict(
        weighted_features.to_numpy(dtype=float), sample_weight=sample_weight.values
    )

    if bool(cfg.get("enforce_contiguity", True)):
        adjacency = _build_ac_bus_adjacency(n, ac_buses)
        labels = _enforce_node_cluster_contiguity(
            labels, weighted_features, ac_buses, adjacency
        )

    unique_after = len(pd.unique(labels))
    if unique_after != n_clusters:
        logger.warning(
            "Contiguity post-processing changed effective AC-bus cluster count from %d to %d.",
            n_clusters,
            unique_after,
        )

    ac_cluster = pd.Series(labels, index=ac_buses).map(lambda x: f"{int(x) + 1:03d}")
    return ac_cluster, features, sample_weight


def _country_kmeans_mapping(n, countries, cfg):
    n_clusters = int(cfg.get("n_clusters", 90))
    random_state = int(cfg.get("random_state", 42))
    pca_components = int(cfg.get("pca_components", 5))
    if n_clusters <= 0:
        raise ValueError(f"n_clusters must be positive, got {n_clusters}")
    if n_clusters > len(countries):
        raise ValueError(
            f"Requested {n_clusters} clusters for {len(countries)} countries."
        )

    features, annual_demand = _build_country_feature_frame(
        n, countries, pca_components, random_state
    )

    feature_weights = {
        "geography": 1.0,
        "load_pca": 2.0,
        "resource": 1.0,
        "fossil": 1.0,
        "capmix": 0.5,
    }
    feature_weights.update(cfg.get("feature_weights", {}))

    weighted_features = features.copy()
    load_pc_cols = [c for c in weighted_features.columns if c.startswith("load_pc")]
    grouped_cols = {
        "geography": [c for c in ["geo_x", "geo_y"] if c in weighted_features.columns],
        "load_pca": load_pc_cols,
        "resource": [
            c
            for c in ["res_wind", "res_solar", "res_hydro"]
            if c in weighted_features.columns
        ],
        "fossil": [c for c in ["fossil_proxy"] if c in weighted_features.columns],
        "capmix": [
            c
            for c in ["capmix_renew_share", "capmix_fossil_share"]
            if c in weighted_features.columns
        ],
    }
    for group, cols in grouped_cols.items():
        if not cols:
            continue
        weighted_features.loc[:, cols] = (
            weighted_features.loc[:, cols] * float(feature_weights[group])
        )

    sample_weight = annual_demand.reindex(countries).fillna(0.0).clip(lower=0.0)
    if float(sample_weight.sum()) <= 0:
        sample_weight[:] = 1.0
    else:
        positive = sample_weight[sample_weight > 0]
        floor = float(positive.min()) if len(positive) else 1.0
        sample_weight = sample_weight.where(sample_weight > 0, floor)

    from sklearn.cluster import KMeans

    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=int(cfg.get("n_init", 50)),
        max_iter=int(cfg.get("max_iter", 1000)),
    )
    labels = km.fit_predict(weighted_features.to_numpy(dtype=float), sample_weight=sample_weight.values)

    if bool(cfg.get("enforce_contiguity", True)):
        adjacency = _build_country_adjacency(n, countries)
        labels = _enforce_country_cluster_contiguity(
            labels, weighted_features, countries, adjacency
        )

    unique_after = len(pd.unique(labels))
    if unique_after != n_clusters:
        logger.warning(
            "Contiguity post-processing changed effective country-cluster count from %d to %d.",
            n_clusters,
            unique_after,
        )

    country_to_cluster = pd.Series(labels, index=countries).map(
        lambda x: f"{int(x) + 1:03d}"
    )
    return country_to_cluster, features, sample_weight


def _country_to_busmap(n, country_to_cluster):
    bus_country = _get_bus_country_for_clustering(n).astype(str).str.strip()
    known_countries = set([c for c in bus_country.unique() if c != ""])
    missing = sorted(known_countries - set(country_to_cluster.index))
    if missing:
        raise ValueError(
            "Country-cluster mapping missing countries: "
            + ", ".join(missing[:20])
            + ("..." if len(missing) > 20 else "")
        )

    def _cluster_label_by_dominant_country():
        countries = country_to_cluster.index.astype(str)
        profiles = _country_electric_load_profiles(n, countries)
        annual = profiles.sum(axis=0).reindex(countries).fillna(0.0)

        if float(annual.sum()) <= 0:
            # Fallback: use AC bus counts if load profiles are empty.
            bus_counts = (
                bus_country[bus_country != ""]
                .value_counts()
                .reindex(countries)
                .fillna(0.0)
            )
            weights = bus_counts
        else:
            weights = annual

        labels = {}
        for cluster_id in country_to_cluster.unique():
            members = country_to_cluster[country_to_cluster == cluster_id].index
            if len(members) == 0:
                continue
            w = weights.reindex(members).fillna(0.0)
            if float(w.max()) <= 0:
                w = pd.Series(1.0, index=members)
            top_country = w.idxmax()
            labels[str(cluster_id)] = str(top_country)
        return labels

    # Preserve global buses (no country code, e.g. Earth buses) one-to-one.
    # They are system-wide carriers and should not be merged by country clustering.
    keep_identity = bus_country == ""
    keep_count = int(keep_identity.sum())
    if keep_count > 0:
        logger.info(
            "Preserving %d global buses without country code as identity buses (not clustered).",
            keep_count,
        )

    cluster_ids = bus_country.map(country_to_cluster).astype(str)
    cluster_labels = _cluster_label_by_dominant_country()
    cluster_names = cluster_ids.map(lambda cid: cluster_labels.get(str(cid), str(cid)))
    carriers = n.buses.carrier.fillna("unknown").astype(str)
    mapped = pd.Series(index=n.buses.index, dtype=object, name="busmap")
    mapped.loc[~keep_identity] = (
        cluster_names.loc[~keep_identity] + " " + carriers.loc[~keep_identity]
    )
    mapped.loc[keep_identity] = n.buses.index[keep_identity]
    return mapped


def _max_abs_series_err(a, b):
    if len(a) == 0 and len(b) == 0:
        return 0.0, 0.0
    x = a.reindex(a.index.union(b.index), fill_value=0.0)
    y = b.reindex(a.index.union(b.index), fill_value=0.0)
    err = float(np.abs(x - y).max())
    scale = float(max(np.abs(x).max(), np.abs(y).max(), 0.0))
    return err, scale


def _check_component_series_index_consistency(n):
    for comp in n.all_components:
        if comp not in n.components:
            continue
        list_name = n.components[comp]["list_name"]
        if not hasattr(n, f"{list_name}_t"):
            continue
        static_df = getattr(n, list_name, None)
        pnl = getattr(n, f"{list_name}_t")
        if static_df is None:
            continue
        static_index = set(static_df.index)
        for attr, df in pnl.items():
            if df.empty:
                continue
            missing = sorted(set(df.columns) - static_index)
            if missing:
                raise ValueError(
                    f"{comp}_t.{attr} has columns not present in {comp}: "
                    + ", ".join(missing[:10])
                    + ("..." if len(missing) > 10 else "")
                )


def _prune_component_time_series(n):
    """Drop time-series columns that are not present in the static component tables."""
    for comp in n.all_components:
        if comp not in n.components:
            continue
        list_name = n.components[comp]["list_name"]
        if not hasattr(n, f"{list_name}_t"):
            continue
        static_df = getattr(n, list_name, None)
        pnl = getattr(n, f"{list_name}_t")
        if static_df is None:
            continue
        static_index = static_df.index
        for attr, df in pnl.items():
            if df.empty:
                continue
            missing = sorted(set(df.columns) - set(static_index))
            if missing:
                pnl[attr] = df.drop(columns=missing, errors="ignore")
                logger.warning(
                    "Dropped %d %s_t.%s columns not present in %s",
                    len(missing),
                    comp,
                    attr,
                    comp,
                )


def _ensure_carriers_defined(n):
    """Ensure all carrier names referenced by components exist in n.carriers."""
    carrier_sources = [
        ("buses", "carrier"),
        ("generators", "carrier"),
        ("loads", "carrier"),
        ("links", "carrier"),
        ("stores", "carrier"),
        ("storage_units", "carrier"),
        ("lines", "carrier"),
        ("sub_networks", "carrier"),
    ]

    carriers = set()
    for table, col in carrier_sources:
        if hasattr(n, table):
            df = getattr(n, table)
            if hasattr(df, "columns") and col in df.columns:
                carriers.update(df[col].dropna().astype(str).unique().tolist())

    carriers = {c for c in carriers if c != ""}
    if not carriers:
        return

    if not hasattr(n, "carriers") or n.carriers is None:
        n.carriers = pd.DataFrame(index=pd.Index([], name="Carrier"))

    existing = set(n.carriers.index.astype(str))
    missing = sorted(carriers - existing)
    if not missing:
        return

    new_rows = pd.DataFrame(index=pd.Index(missing, name=n.carriers.index.name or "Carrier"))
    for col in n.carriers.columns:
        if pd.api.types.is_numeric_dtype(n.carriers[col]):
            new_rows[col] = 0.0
        else:
            new_rows[col] = ""

    if "nice_name" in new_rows.columns:
        new_rows["nice_name"] = new_rows.index.astype(str)

    n.carriers = pd.concat([n.carriers, new_rows], axis=0)
    logger.info("Added %d missing carriers to n.carriers", len(missing))


def _validate_one_port_conservation(
    n_before,
    n_after,
    components,
    validation_rtol=1e-6,
    validation_atol=1e-6,
):
    for comp in components:
        list_name = n_before.components[comp]["list_name"]
        df_before = getattr(n_before, list_name).copy()
        df_after = getattr(n_after, list_name).copy()

        if comp != "Generator":
            df_before = df_before[df_before.bus.isin(n_before.buses.index)]
            df_after = df_after[df_after.bus.isin(n_after.buses.index)]

        if df_before.empty and df_after.empty:
            continue

        cap_col = "p_nom" if "p_nom" in df_before.columns else ("e_nom" if "e_nom" in df_before.columns else None)

        # Static columns
        for col, strategy in DEFAULT_ONE_PORT_STRATEGIES.items():
            if col not in df_before.columns or col not in df_after.columns:
                continue
            if strategy == "sum":
                a = float(df_before[col].fillna(0.0).sum())
                b = float(df_after[col].fillna(0.0).sum())
                if not np.isclose(a, b, rtol=validation_rtol, atol=validation_atol):
                    raise ValueError(
                        f"{comp}.{col} sum mismatch after clustering: {a} vs {b}"
                    )
            elif strategy == "capacity_weighted_average" and cap_col is not None:
                a = float((df_before[col].fillna(0.0) * df_before[cap_col].fillna(0.0)).sum())
                b = float((df_after[col].fillna(0.0) * df_after[cap_col].fillna(0.0)).sum())
                tol = validation_atol + validation_rtol * max(abs(a), abs(b), 1.0)
                if abs(a - b) > tol:
                    raise ValueError(
                        f"{comp}.{col} capacity-weighted total mismatch: {a} vs {b}"
                    )

        # Time-varying columns
        pnl_before = getattr(n_before, f"{list_name}_t")
        pnl_after = getattr(n_after, f"{list_name}_t")
        for attr, strategy in DEFAULT_ONE_PORT_STRATEGIES.items():
            if attr not in pnl_before or attr not in pnl_after:
                continue
            df_t_before = pnl_before[attr]
            df_t_after = pnl_after[attr]
            if df_t_before.empty and df_t_after.empty:
                continue
            if strategy == "sum":
                pre_cols = df_t_before.columns.intersection(df_before.index)
                post_cols = df_t_after.columns.intersection(df_after.index)
                s_before = df_t_before[pre_cols].fillna(0.0).sum(axis=1)
                s_after = df_t_after[post_cols].fillna(0.0).sum(axis=1)
                err, scale = _max_abs_series_err(s_before, s_after)
                tol = validation_atol + validation_rtol * max(scale, 1.0)
                if err > tol:
                    raise ValueError(
                        f"{comp}_t.{attr} sum mismatch: max abs error {err} > {tol}"
                    )
            elif strategy == "capacity_weighted_average" and cap_col is not None:
                pre_cols = df_t_before.columns.intersection(df_before.index)
                post_cols = df_t_after.columns.intersection(df_after.index)
                s_before = df_t_before[pre_cols].fillna(0.0).mul(
                    df_before.loc[pre_cols, cap_col].fillna(0.0), axis=1
                ).sum(axis=1)
                s_after = df_t_after[post_cols].fillna(0.0).mul(
                    df_after.loc[post_cols, cap_col].fillna(0.0), axis=1
                ).sum(axis=1)
                err, scale = _max_abs_series_err(s_before, s_after)
                tol = validation_atol + validation_rtol * max(scale, 1.0)
                if err > tol:
                    raise ValueError(
                        f"{comp}_t.{attr} capacity-weighted mismatch: max abs error {err} > {tol}"
                    )


def _validate_cluster_aggregation(
    n_before,
    n_after,
    aggregate_one_ports,
    validation_rtol=1e-6,
    validation_atol=1e-6,
):
    if not n_before.snapshots.equals(n_after.snapshots):
        raise ValueError("Snapshots changed during additional sector clustering.")

    # Sanity check all component time-series tables reference existing static assets.
    _check_component_series_index_consistency(n_after)

    # Validate conservation for all aggregated one-port components + generators.
    validate_components = sorted(set(aggregate_one_ports + ["Generator"]))
    _validate_one_port_conservation(
        n_before=n_before,
        n_after=n_after,
        components=validate_components,
        validation_rtol=validation_rtol,
        validation_atol=validation_atol,
    )

    logger.info(
        "Additional clustering validation passed for components: %s",
        ", ".join(validate_components),
    )


def _write_cluster_outputs(output_dir, country_to_cluster, busmap, features, settings):
    output_dir.mkdir(parents=True, exist_ok=True)
    country_to_cluster.rename("cluster").to_csv(output_dir / "country_to_cluster.csv")
    busmap.rename("clustered_bus").to_csv(output_dir / "busmap.csv")
    if features is not None:
        features.to_csv(output_dir / "country_features.csv")
    with open(output_dir / "settings.yaml", "w") as f:
        yaml.safe_dump(settings, f, sort_keys=True)


def _load_busmap_series(path):
    df = pd.read_csv(path, index_col=0)
    if df.empty or len(df.columns) == 0:
        raise ValueError(f"Busmap file is empty: {path}")
    col = "clustered_bus" if "clustered_bus" in df.columns else df.columns[0]
    s = df[col].astype(str)
    s.index = s.index.astype(str)
    s.name = "busmap"
    return s


def _extract_cluster_token(mapped_bus_name, carrier):
    if pd.isna(mapped_bus_name):
        return None
    mapped = str(mapped_bus_name)
    carrier = str(carrier)
    suffix = f" {carrier}"
    if mapped.endswith(suffix):
        token = mapped[: -len(suffix)].strip()
        return token if token else None
    return None


def _country_default_tokens(n, ac_tokens):
    ac_tokens = ac_tokens.dropna()
    if ac_tokens.empty:
        return {}
    profiles = _ac_bus_load_profiles(n, ac_tokens.index)
    weights = profiles.sum(axis=0).reindex(ac_tokens.index).fillna(0.0).clip(lower=0.0)
    if float(weights.sum()) <= 0:
        weights[:] = 1.0
    bus_country = _get_bus_country_for_clustering(n).reindex(ac_tokens.index).fillna("")
    by_country_cluster = (
        pd.DataFrame(
            {
                "country": bus_country.values,
                "token": ac_tokens.values,
                "weight": weights.values,
            },
            index=ac_tokens.index,
        )
        .query("country != ''")
        .groupby(["country", "token"])["weight"]
        .sum()
    )
    defaults = {}
    if len(by_country_cluster) > 0:
        for country in by_country_cluster.index.get_level_values(0).unique():
            w = by_country_cluster.xs(country, level=0)
            defaults[str(country)] = str(w.idxmax())
    return defaults


def _nearest_token_by_distance(n, source_bus, candidate_buses, ac_tokens):
    candidate_buses = pd.Index(candidate_buses).intersection(ac_tokens.index)
    candidate_buses = candidate_buses[ac_tokens.reindex(candidate_buses).notna()]
    if len(candidate_buses) == 0:
        return None

    src_xy = n.buses.loc[source_bus, ["x", "y"]]
    try:
        src_x = float(src_xy["x"])
        src_y = float(src_xy["y"])
    except Exception:
        return str(ac_tokens.reindex(candidate_buses).iloc[0])
    if not np.isfinite(src_x) or not np.isfinite(src_y):
        return str(ac_tokens.reindex(candidate_buses).iloc[0])

    cand_xy = n.buses.loc[candidate_buses, ["x", "y"]].astype(float)
    finite = np.isfinite(cand_xy["x"].values) & np.isfinite(cand_xy["y"].values)
    if not finite.any():
        return str(ac_tokens.reindex(candidate_buses).iloc[0])
    cand_xy = cand_xy.iloc[np.where(finite)[0]]
    dist2 = (cand_xy["x"] - src_x) ** 2 + (cand_xy["y"] - src_y) ** 2
    nearest = dist2.idxmin()
    token = ac_tokens.get(nearest)
    return str(token) if pd.notna(token) else None


def _extend_universal_busmap(n, busmap):
    busmap = busmap.copy()
    busmap.index = busmap.index.astype(str)
    all_buses = n.buses.index.astype(str)
    missing = all_buses.difference(busmap.index)
    if len(missing) == 0:
        return busmap, 0

    bus_country = _get_bus_country_for_clustering(n).astype(str).str.strip()
    carriers = n.buses.carrier.fillna("unknown").astype(str)
    keep_identity = bus_country == ""

    ac_buses = pd.Index(n.buses.index[n.buses.carrier == "AC"]).astype(str)
    covered_ac = ac_buses.intersection(busmap.index)
    ac_tokens = covered_ac.to_series(index=covered_ac).map(
        lambda b: _extract_cluster_token(busmap.at[b], "AC")
    )
    ac_tokens = ac_tokens.dropna()
    country_default_token = _country_default_tokens(n, ac_tokens)

    # First map missing AC buses, then non-AC buses can anchor to them.
    missing_ac = [b for b in missing if carriers.get(b, "") == "AC"]
    missing_non_ac = [b for b in missing if carriers.get(b, "") != "AC"]

    def _pick_token_for_bus(b):
        anchor = _extract_ac_anchor_bus(str(b))
        if anchor in busmap.index:
            token = _extract_cluster_token(busmap.at[anchor], "AC")
            if token:
                return token
        loc = str(n.buses.at[b, "location"]).strip()
        if loc in busmap.index:
            token = _extract_cluster_token(busmap.at[loc], "AC")
            if token:
                return token

        country = bus_country.get(b, "")
        if country in country_default_token:
            same_country_ac = ac_tokens.index[bus_country.reindex(ac_tokens.index) == country]
            if len(same_country_ac) > 0:
                token = _nearest_token_by_distance(n, b, same_country_ac, ac_tokens)
                if token:
                    return token
            return country_default_token[country]

        token = _nearest_token_by_distance(n, b, ac_tokens.index, ac_tokens)
        return token

    added = 0
    for b in missing_ac:
        if keep_identity.get(b, False):
            busmap.at[b] = b
            added += 1
            continue
        token = _pick_token_for_bus(b)
        if token is None:
            busmap.at[b] = b
        else:
            busmap.at[b] = f"{token} AC"
            ac_tokens.loc[b] = token
            country = bus_country.get(b, "")
            if country and country not in country_default_token:
                country_default_token[country] = token
        added += 1

    for b in missing_non_ac:
        if keep_identity.get(b, False):
            busmap.at[b] = b
            added += 1
            continue
        token = _pick_token_for_bus(b)
        if token is None:
            busmap.at[b] = b
        else:
            busmap.at[b] = f"{token} {carriers.at[b]}"
        added += 1

    busmap = busmap.reindex(all_buses)
    busmap.name = "busmap"
    return busmap, added


def _ensure_pypsa_aggregateoneport_compat():
    """
    Patch PyPSA's aggregateoneport for pandas versions where Series.groupby(axis=...)
    is not supported anymore.
    """
    if getattr(pypsa_spatial, "_cki_aggregateoneport_compat", False):
        return

    def _aggregateoneport_compat(
        n,
        busmap,
        component,
        carriers=None,
        buses=None,
        with_time=True,
        custom_strategies=None,
    ):
        if custom_strategies is None:
            custom_strategies = {}

        c = component
        df = n.df(c)
        attrs = n.components[c]["attrs"]

        if "carrier" in df.columns:
            if carriers is None:
                carriers = df.carrier.unique()
            to_aggregate = df.carrier.isin(carriers)
        else:
            to_aggregate = pd.Series(True, index=df.index)

        if buses is not None:
            to_aggregate |= df.bus.isin(buses)

        df = df[to_aggregate]
        df = df.assign(bus=df.bus.map(busmap))

        output_columns = attrs.index[attrs.static & attrs.status.str.startswith("Output")]
        columns = [col for col in df.columns if col not in output_columns]

        strategies = {**pypsa_spatial.DEFAULT_ONE_PORT_STRATEGIES, **custom_strategies}
        static_strategies = pypsa_spatial.align_strategies(strategies, columns, c)

        grouper = [df.bus, df.carrier] if "carrier" in df.columns else df.bus

        uniform_weights = pd.Series(1.0, index=df.index).groupby(grouper).transform(
            pypsa_spatial.normed_or_uniform
        )
        capacity = df.columns.intersection({"p_nom", "e_nom"})
        capacity_weights = uniform_weights
        if len(capacity):
            capacity_weights = df[capacity[0]].groupby(grouper).transform(
                pypsa_spatial.normed_or_uniform
            )

        weights = uniform_weights
        if "weight" in df.columns:
            weights = df.weight.groupby(grouper).transform(pypsa_spatial.normed_or_uniform)

        for col, strategy in static_strategies.items():
            if strategy == "weighted_average":
                df[col] = df[col] * weights
                static_strategies[col] = "sum"
            elif strategy == "capacity_weighted_average":
                df[col] = df[col] * capacity_weights
                static_strategies[col] = "sum"
            elif strategy == "weighted_min":
                df["p_nom_max"] /= weights
                static_strategies[col] = "min"

        aggregated = df.groupby(grouper).agg(static_strategies)
        aggregated.index = pypsa_spatial.flatten_multiindex(aggregated.index).rename(c)

        non_aggregated = n.df(c)[~to_aggregate]
        non_aggregated = non_aggregated.assign(bus=non_aggregated.bus.map(busmap))

        df = pd.concat([aggregated, non_aggregated], sort=False)
        df.fillna(attrs.default, inplace=True)

        pnl = {}
        if with_time:
            dynamic_strategies = pypsa_spatial.align_strategies(strategies, n.pnl(c), c)
            for attr, data in n.pnl(c).items():
                if data.empty:
                    pnl[attr] = data
                    continue
                strategy = dynamic_strategies[attr]
                data = n.get_switchable_as_dense(c, attr)
                aggregated = data.loc[:, to_aggregate]

                if strategy == "weighted_average":
                    aggregated = aggregated * weights
                    aggregated = aggregated.T.groupby(grouper).sum().T
                elif strategy == "capacity_weighted_average":
                    aggregated = aggregated * capacity_weights
                    aggregated = aggregated.T.groupby(grouper).sum().T
                elif strategy == "weighted_min":
                    aggregated = aggregated / weights
                    aggregated = aggregated.T.groupby(grouper).min().T
                else:
                    aggregated = aggregated.T.groupby(grouper).agg(strategy).T
                aggregated.columns = pypsa_spatial.flatten_multiindex(aggregated.columns).rename(c)

                non_aggregated = data.loc[:, ~to_aggregate]
                pnl[attr] = pd.concat([aggregated, non_aggregated], axis=1, sort=False)

                # filter out static values
                if attr in df:
                    is_static = (pnl[attr] == df[attr]).all()
                    pnl[attr] = pnl[attr].loc[:, ~is_static]

        return df, pnl

    pypsa_spatial.aggregateoneport = _aggregateoneport_compat
    pypsa_spatial._cki_aggregateoneport_compat = True


def _ensure_pypsa_nodal_balance_busname_compat():
    """
    Ensure nodal-balance helper receives a named bus index ("Bus").

    Some PyPSA/Linopy combinations can pass unnamed bus subsets to
    define_nodal_balance_constraints, which then fails on rhs.rename(Bus=...).
    """
    try:
        import pypsa.optimization.constraints as pypsa_constraints
        import pypsa.optimization.optimize as pypsa_optimize
    except Exception:
        return

    if getattr(pypsa_constraints, "_cki_nodal_balance_busname_compat", False):
        return

    def _define_nodal_balance_constraints_compat(
        n,
        sns,
        transmission_losses=0,
        buses=None,
        suffix="",
    ):
        # Same logic as upstream, but enforce a stable Bus dimension name.
        m = n.model
        if buses is None:
            buses = n.buses.index
        buses = pd.Index(buses)
        if buses.name is None:
            buses = buses.rename("Bus")

        args = [
            ["Generator", "p", "bus", 1],
            ["Store", "p", "bus", 1],
            ["StorageUnit", "p_dispatch", "bus", 1],
            ["StorageUnit", "p_store", "bus", -1],
            ["Line", "s", "bus0", -1],
            ["Line", "s", "bus1", 1],
            ["Transformer", "s", "bus0", -1],
            ["Transformer", "s", "bus1", 1],
            ["Link", "p", "bus0", -1],
            ["Link", "p", "bus1", pypsa_constraints.get_as_dense(n, "Link", "efficiency", sns)],
        ]

        if not n.links.empty:
            for i in pypsa_constraints.additional_linkports(n):
                eff = pypsa_constraints.get_as_dense(n, "Link", f"efficiency{i}", sns)
                args.append(["Link", "p", f"bus{i}", eff])

        if transmission_losses:
            args.extend(
                [
                    ["Line", "loss", "bus0", -0.5],
                    ["Line", "loss", "bus1", -0.5],
                    ["Transformer", "loss", "bus0", -0.5],
                    ["Transformer", "loss", "bus1", -0.5],
                ]
            )

        exprs = []
        for c, attr, column, sign in args:
            if n.df(c).empty:
                continue
            if "sign" in n.df(c):
                sign = sign * n.df(c).sign

            expr = pypsa_constraints.DataArray(sign) * m[f"{c}-{attr}"]
            cbuses = n.df(c)[column][lambda ds: ds.isin(buses)].rename("Bus")

            if column in ["bus" + i for i in pypsa_constraints.additional_linkports(n)]:
                cbuses = cbuses[cbuses != ""]

            expr = expr.sel({c: cbuses.index})
            if expr.size:
                exprs.append(expr.groupby(cbuses).sum())

        lhs = pypsa_constraints.merge(exprs, join="outer").reindex(Bus=buses)
        rhs_df = (
            (-pypsa_constraints.get_as_dense(n, "Load", "p_set", sns) * n.loads.sign)
            .T.groupby(n.loads.bus)
            .sum()
            .T.reindex(columns=buses, fill_value=0)
        )
        rhs_df.index.name = "snapshot"
        rhs_df.columns.name = "Bus"
        rhs = pypsa_constraints.DataArray(rhs_df)

        empty_nodal_balance = (lhs.vars == -1).all("_term")
        if empty_nodal_balance.any():
            if (empty_nodal_balance & (rhs != 0)).any().item():
                raise ValueError("Empty LHS with non-zero RHS in nodal balance constraint.")
            mask = ~empty_nodal_balance
        else:
            mask = None

        if suffix:
            lhs = lhs.rename(Bus=f"Bus{suffix}")
            rhs = rhs.rename(Bus=f"Bus{suffix}")
            if mask is not None:
                mask = mask.rename(Bus=f"Bus{suffix}")
        n.model.add_constraints(lhs, "=", rhs, name=f"Bus{suffix}-nodal_balance", mask=mask)

    pypsa_constraints.define_nodal_balance_constraints = _define_nodal_balance_constraints_compat
    # create_model imports the function into optimize module namespace
    pypsa_optimize.define_nodal_balance_constraints = _define_nodal_balance_constraints_compat
    pypsa_constraints._cki_nodal_balance_busname_compat = True


def _normalize_clustered_link_buses(clustered, busmap):
    if clustered.links.empty:
        return

    mapping = busmap.to_dict()
    bus_cols = [c for c in clustered.links.columns if re.fullmatch(r"bus\d*", str(c))]
    if not bus_cols:
        return

    def _map_bus(value):
        if pd.isna(value):
            return ""
        if not isinstance(value, str):
            value = str(value)
        if value == "":
            return ""
        return mapping.get(value, value)

    for col in bus_cols:
        clustered.links[col] = clustered.links[col].map(_map_bus)
        invalid = (~clustered.links[col].isin(clustered.buses.index)) & (clustered.links[col] != "")
        if invalid.any():
            clustered.links.loc[invalid, col] = ""

    # Remove invalid intra-cluster links after terminal remapping.
    invalid_primary = (
        (clustered.links.bus0 == "")
        | (clustered.links.bus1 == "")
        | (clustered.links.bus0 == clustered.links.bus1)
    )
    if invalid_primary.any():
        dropped = int(invalid_primary.sum())
        clustered.mremove("Link", clustered.links.index[invalid_primary])
        logger.warning(
            "Dropped %d links with invalid clustered bus0/bus1 after additional clustering.",
            dropped,
        )


def apply_optional_sector_clustering(n, config):
    raw_cfg = config.get(
        "additional_sector_clustering",
        config.get("solving", {}).get("additional_sector_clustering", None),
    )
    if raw_cfg is None:
        return n

    if isinstance(raw_cfg, int):
        cfg = {"enable": True, "mode": "kmeans", "n_clusters": int(raw_cfg)}
    elif isinstance(raw_cfg, dict):
        cfg = dict(raw_cfg)
    else:
        logger.warning(
            "Ignoring additional_sector_clustering because config type is unsupported: %s",
            type(raw_cfg),
        )
        return n

    if not bool(cfg.get("enable", False)):
        return n
    if getattr(n, "_additional_sector_clustered", False):
        return n
    meta = getattr(n, "meta", {})
    if isinstance(meta, dict) and meta.get("additional_sector_clustered", False):
        n._additional_sector_clustered = 1
        return n

    mode = str(cfg.get("mode", "kmeans")).lower()
    output_dir = None
    try:
        if "snakemake" in globals() and hasattr(snakemake, "output") and snakemake.output:
            solved_path = Path(snakemake.output[0]).resolve()
            output_dir = solved_path.parent.parent / "additional_sector_clustering"
    except Exception:
        output_dir = None

    if output_dir is None:
        output_dir = _repo_path(cfg.get("output_directory", "results/additional_sector_clustering"))
    universal_busmap_file = cfg.get(
        "universal_busmap_file", str(output_dir / "universal_busmap.csv")
    )
    universal_busmap_path = _repo_path(universal_busmap_file)
    reuse_universal_busmap = bool(cfg.get("reuse_universal_busmap", True))
    overwrite_universal_busmap = bool(cfg.get("overwrite_universal_busmap", False))
    write_universal_busmap = bool(cfg.get("write_universal_busmap", True))

    country_to_cluster = None
    ac_bus_to_cluster = None
    features = None
    sample_weight = None
    busmap_source = "computed"

    if reuse_universal_busmap and universal_busmap_path.exists() and not overwrite_universal_busmap:
        busmap = _load_busmap_series(universal_busmap_path)
        missing = n.buses.index.difference(busmap.index)
        if len(missing) > 0:
            busmap, added = _extend_universal_busmap(n, busmap)
            logger.warning(
                "Universal busmap missing %d buses; deterministically extended mapping without changing existing assignments.",
                added,
            )
            if write_universal_busmap:
                universal_busmap_path.parent.mkdir(parents=True, exist_ok=True)
                busmap.rename("clustered_bus").to_csv(universal_busmap_path)
                logger.info(
                    "Updated universal additional clustering busmap at %s",
                    universal_busmap_path,
                )
        busmap = busmap.reindex(n.buses.index)
        busmap_source = "universal"
        logger.info("Using universal additional clustering busmap from %s", universal_busmap_path)
    else:
        logger.info("Applying additional sector clustering in solve_network.py (mode=%s)", mode)

        if mode == "manual":
            bus_country = _get_bus_country_for_clustering(n)
            countries = pd.Index(sorted([c for c in bus_country.unique() if c != ""]))
            if len(countries) == 0:
                logger.warning("No countries found on buses; skipping additional sector clustering.")
                return n
            manual_file = _repo_path(cfg.get("manual_mapping_file", "configs/global.cluster.yaml"))
            country_to_cluster = _parse_manual_country_clusters(manual_file).reindex(countries)
            missing = country_to_cluster[country_to_cluster.isna()].index.tolist()
            if missing:
                missing_policy = str(cfg.get("manual_missing_policy", "singleton")).lower()
                if missing_policy == "error":
                    raise ValueError(
                        "Manual cluster mapping does not cover all countries. Missing: "
                        + ", ".join(missing[:20])
                        + ("..." if len(missing) > 20 else "")
                    )
                if missing_policy == "singleton":
                    for c in missing:
                        country_to_cluster.loc[c] = f"auto_{c}"
                    logger.warning(
                        "Manual mapping missing %d countries (%s). "
                        "Assigned deterministic singleton clusters via manual_missing_policy=singleton.",
                        len(missing),
                        ", ".join(missing[:20]) + ("..." if len(missing) > 20 else ""),
                    )
                else:
                    raise ValueError(
                        f"Unknown manual_missing_policy '{missing_policy}'. Use 'singleton' or 'error'."
                    )
            country_to_cluster = country_to_cluster.astype(str)
            logger.info(
                "Loaded manual country clusters from %s with %d target clusters.",
                manual_file,
                country_to_cluster.nunique(),
            )
        elif mode in {"kmeans", "algorithm"}:
            ac_bus_to_cluster, features, sample_weight = _ac_bus_kmeans_mapping(
                n, cfg
            )
            busmap = _ac_cluster_to_busmap(n, ac_bus_to_cluster, sample_weight)
            logger.info(
                "Computed weighted AC-bus k-means mapping with %d clusters (seed=%s).",
                ac_bus_to_cluster.nunique(),
                cfg.get("random_state", 42),
            )
        else:
            raise ValueError(
                f"Unknown additional sector clustering mode '{mode}'. Use 'manual' or 'kmeans'."
            )

        if mode == "manual":
            busmap = _country_to_busmap(n, country_to_cluster)
        if write_universal_busmap:
            universal_busmap_path.parent.mkdir(parents=True, exist_ok=True)
            busmap.rename("clustered_bus").to_csv(universal_busmap_path)
            logger.info(
                "Wrote universal additional clustering busmap to %s",
                universal_busmap_path,
            )

    bus_strategies = {
        "v_nom": "first",
        "x": "mean",
        "y": "mean",
        "lon": "mean",
        "lat": "mean",
        "country": "first",
        "carrier": "first",
        "sub_network": "first",
        "location": "first",
        "type": "first",
        "unit": "first",
        "control": "first",
        "tag_substation": "first",
        "tag_area": "first",
    }
    line_strategies = {
        # Aggregated lines can combine multiple spatial metadata entries.
        # Keep deterministic representative values instead of requiring equality.
        "geometry": "first",
        "bounds": "first",
        "carrier": "first",
        "type": "first",
        "sub_network": "first",
    }
    aggregate_one_ports = cfg.get("aggregate_one_ports")
    if aggregate_one_ports is None:
        aggregate_one_ports = [c for c in n.one_port_components if c != "Generator"]
    if isinstance(aggregate_one_ports, str):
        aggregate_one_ports = [aggregate_one_ports]
    aggregate_one_ports = [c for c in aggregate_one_ports if c in n.one_port_components]

    # Preserve vintage metadata when one-port aggregation is enabled.
    raw_one_port_strategies = cfg.get("one_port_strategies", {})
    if not isinstance(raw_one_port_strategies, dict):
        raw_one_port_strategies = {}
    one_port_strategies = {}
    for comp in aggregate_one_ports:
        comp_strategies = dict(raw_one_port_strategies.get(comp, {}))
        comp_strategies.setdefault("build_year", "first")
        comp_strategies.setdefault("lifetime", "first")
        one_port_strategies[comp] = comp_strategies
    raw_generator_strategies = cfg.get("generator_strategies", {})
    if not isinstance(raw_generator_strategies, dict):
        raw_generator_strategies = {}
    generator_strategies = dict(raw_generator_strategies)
    generator_strategies.setdefault("build_year", "first")
    generator_strategies.setdefault("lifetime", "first")

    vintage_components = ["Generator"] + aggregate_one_ports
    carrier_backups, token_to_meta = _encode_vintage_carriers_for_clustering(
        n, vintage_components
    )

    _ensure_pypsa_aggregateoneport_compat()
    try:
        clustering = get_clustering_from_busmap(
            n,
            busmap,
            aggregate_generators_weighted=True,
            aggregate_one_ports=aggregate_one_ports,
            line_length_factor=float(cfg.get("line_length_factor", 1.0)),
            scale_link_capital_costs=False,
            bus_strategies=bus_strategies,
            one_port_strategies=one_port_strategies,
            generator_strategies=generator_strategies,
            line_strategies=line_strategies,
        )
    finally:
        _restore_encoded_carriers(n, carrier_backups)

    clustered = clustering.network
    _decode_vintage_carriers_after_clustering(clustered, token_to_meta, vintage_components)
    _sanitize_vintage_encoded_component_names(clustered, token_to_meta, vintage_components)
    _normalize_clustered_link_buses(clustered, busmap)
    _ensure_carriers_defined(clustered)
    _prune_component_time_series(clustered)

    # Keep critical attributes used later in solve flow.
    for attr in (
        "n_ref",
        "build_rate_limits",
        "build_rate_scenario",
        "build_rate_target_year",
        "config",
        "opts",
        "temporal_cluster",
    ):
        if hasattr(n, attr):
            setattr(clustered, attr, getattr(n, attr))

    if bool(cfg.get("validate_aggregation", True)):
        _validate_cluster_aggregation(
            n,
            clustered,
            aggregate_one_ports=aggregate_one_ports,
            validation_rtol=float(cfg.get("validation_rtol", 1e-6)),
            validation_atol=float(cfg.get("validation_atol", 1e-6)),
        )

    if bool(cfg.get("write_mapping", True)):
        settings = {
            "mode": mode,
            "n_clusters": int(
                country_to_cluster.nunique()
                if country_to_cluster is not None
                else ac_bus_to_cluster.nunique() if ac_bus_to_cluster is not None else 0
            ),
            "random_state": int(cfg.get("random_state", 42)),
            "pca_components": int(cfg.get("pca_components", 5)),
            "include_p_max_pu_features": bool(cfg.get("include_p_max_pu_features", False)),
            "p_max_pu_components": int(cfg.get("p_max_pu_components", 3)),
            "p_max_pu_by_tech": bool(cfg.get("p_max_pu_by_tech", True)),
            "p_max_pu_tech_groups": cfg.get("p_max_pu_tech_groups"),
            "feature_weights": cfg.get("feature_weights", {}),
            "busmap_source": busmap_source,
            "universal_busmap_file": str(universal_busmap_path),
        }
        if country_to_cluster is not None:
            _write_cluster_outputs(
                output_dir=output_dir,
                country_to_cluster=country_to_cluster.sort_index(),
                busmap=busmap.sort_index(),
                features=features.sort_index() if isinstance(features, pd.DataFrame) else None,
                settings=settings,
            )
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            if ac_bus_to_cluster is not None:
                ac_bus_to_cluster.sort_index().rename("cluster").to_csv(
                    output_dir / "ac_bus_to_cluster.csv"
                )
            busmap.sort_index().rename("clustered_bus").to_csv(output_dir / "busmap.csv")
            if isinstance(features, pd.DataFrame):
                features.sort_index().to_csv(output_dir / "ac_bus_features.csv")
            with open(output_dir / "settings.yaml", "w") as f:
                yaml.safe_dump(settings, f, sort_keys=True)
        logger.info("Wrote additional clustering mapping artifacts to %s", output_dir)

    # Use int instead of bool to keep netCDF export happy (netCDF4 can't store bool attrs)
    clustered._additional_sector_clustered = 1
    if not isinstance(getattr(clustered, "meta", None), dict):
        clustered.meta = {}
    clustered.meta.update(
        {
            "additional_sector_clustered": True,
            "additional_sector_clustering_mode": mode,
            "additional_sector_clustering_busmap_source": busmap_source,
            "additional_sector_clustering_busmap_file": str(universal_busmap_path),
        }
    )
    logger.info(
        "Additional sector clustering reduced buses %d -> %d, generators %d -> %d, loads %d -> %d.",
        len(n.buses),
        len(clustered.buses),
        len(n.generators),
        len(clustered.generators),
        len(n.loads),
        len(clustered.loads),
    )
    return clustered


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

    # Only add land use constraints if snakemake is available (normal workflow)
    # For rolling horizon/perfect foresight, land use constraints are handled separately
    try:
        if snakemake.config["foresight"] == "myopic":
            add_land_use_constraint(n)
    except (NameError, KeyError):
        # snakemake not available (e.g., called from rolling horizon script)
        # Land use constraints will be handled separately
        pass

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
    float_regex = r"[0-9]*\.?[0-9]+"
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


# Baseyear OWID country-generation constraint implementation moved to scripts/validation.py

if hasattr(_validation_hooks, "add_baseyear_generation_band"):
    add_baseyear_generation_band = _validation_hooks.add_baseyear_generation_band
    logger.info("Using centralized baseyear generation constraint hook from scripts/validation.py")


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
            res_share = float(re.findall(r"[0-9]*\.?[0-9]+$", o)[0])
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
        skipping = False
        if not skipping:
            logger.info("Adding temporal aggregation storage constraints (Kotzur et al. 2018)")
            
            # Get parameters from config
            hours_per_period = tc_config.get("hours", 24)
            
            # Carrier filtering (which storage types get Kotzur constraints)
            su_carriers = ("phs", "hydro")  # StorageUnits
            store_carriers = ("battery", "battery storage", "h2", "h2 store tank")  # Stores
            
            logger.info(f"  hours_per_period: {hours_per_period}")
            logger.info(f"  StorageUnit carriers: {su_carriers}")
            logger.info(f"  Store carriers: {store_carriers}")
            
            add_kotzur_storage_constraints(
                n,
                hours_per_period=hours_per_period,
                su_carriers=su_carriers,
                store_carriers=store_carriers,
            )
        else:
            logger.info("Skipping storage constraints.")
    else:
        logger.info("No temporal clustering detected, skipping temporal storage constraints")


    if config["sector"]["chp"]:
        logger.info("setting CHP constraints")
        add_chp_constraints(n)

    if (
        config["policy_config"]["hydrogen"]["temporal_matching"]
        == "h2_yearly_matching"
    ):
        if config["policy_config"]["hydrogen"]["additionality"] == True:
            logger.info(
                "additionality is currently not supported for yearly constraints, proceeding without additionality"
            )
        logger.info("setting h2 export to yearly greenness constraint")
        H2_export_yearly_constraint(n)

    elif (
        config["policy_config"]["hydrogen"]["temporal_matching"]
        == "h2_monthly_matching"
    ):
        if not config["policy_config"]["hydrogen"]["is_reference"]:
            logger.info("setting h2 export to monthly greenness constraint")
            n_ref_local = getattr(n, "n_ref", None)
            if n_ref_local is not None:
                monthly_constraints(n, n_ref_local)
            else:
                logger.warning("monthly matching requested but n_ref not provided; skipping")


    elif (
        config["policy_config"]["hydrogen"]["temporal_matching"]
        == "no_res_matching"
    ):
        logger.info("no h2 export constraint set")

    else:
        raise ValueError(
            'H2 export constraint is invalid, check config["policy_config"]'
        )

    if config["sector"]["hydrogen"]["network"]:
        if config["sector"]["hydrogen"]["network_limit"]:
            add_h2_network_cap(
                n, config["sector"]["hydrogen"]["network_limit"]
            )

    if config["sector"]["hydrogen"]["set_color_shares"]:
        logger.info("setting H2 color mix")
        set_h2_colors(n)

    # Get planning_year - use first investment period for multi-period networks
    try:
        planning_year = snakemake.wildcards.planning_horizons
    except (NameError, AttributeError):
        # For rolling horizon/perfect foresight, use first investment period
        if hasattr(n, "investment_periods") and len(n.investment_periods) > 0:
            planning_year = str(n.investment_periods[0])
        else:
            # Fallback: use first snapshot year if available
            if hasattr(n.snapshots, "levels") and len(n.snapshots.levels) > 0:
                planning_year = str(n.snapshots.levels[0][0])
            else:
                planning_year = None

    if planning_year is not None:
        add_baseyear_generation_band(
            n,
            planning_year=planning_year,
            config=config,
        )

        add_year2025_generation_band(
            n,
            planning_year=planning_year,
            config=config,
        )

        # Add 2025 capacity targets
        add_year2025_capacity_targets(
            n,
            planning_year=planning_year,
            config=config,
        )

    add_co2_sequestration_limit(n, snapshots)
    
    # Add build rate constraints (if build_rate_limits attached to network)
    #add_build_rate_constraints(n, snapshots)


def solve_network(n, config, solving, **kwargs):
    set_of_options = solving["solver"]["options"]
    cf_solving = solving["options"]

    kwargs["solver_options"] = (
        solving["solver_options"][set_of_options] if set_of_options else {}
    )
    kwargs["solver_options"]["DualReductions"] = 0
    logger.info("Added DualReductions=0 to force infeasible/unbounded determination")
    kwargs["solver_name"] = solving["solver"]["name"]
    kwargs["extra_functionality"] = extra_functionality

    _ensure_pypsa_nodal_balance_busname_compat()

    # add to network for extra_functionality
    n.config = config
    if not hasattr(n, "opts"):
        if "snakemake" in globals():
            n.opts = snakemake.wildcards.opts.split("-")
        else:
            n.opts = globals().get("opts", [])

    n = apply_optional_sector_clustering(n, config)

    skip_iterations = cf_solving.get("skip_iterations", False)
    if not n.lines.s_nom_extendable.any():
        skip_iterations = True
        logger.info("No expandable lines found. Skipping iterative solving.")



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
            h2_loads = n.loads[n.loads.carrier.str.contains('H2', na=False)] if not n.loads.empty else pd.DataFrame()
            h2_stores = n.stores[n.stores.carrier.str.contains('H2', na=False)] if not n.stores.empty else pd.DataFrame()
            h2_links = n.links[n.links.carrier.str.contains('H2', na=False)] if not n.links.empty else pd.DataFrame()
    

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
        
        # Conditionally save LP files based on config
        save_lpfile = solving.get("save_lpfile", False)
        if save_lpfile and hasattr(snakemake, 'output') and hasattr(snakemake.output, 'lpfile'):
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
        elif save_lpfile:
            logger.warning("LP file saving is enabled but lpfile output not defined in Snakefile")
        else:
            logger.info("LP file saving is disabled (set solving.save_lpfile: true to enable)")
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
        
        # Conditionally save LP files based on config
        save_lpfile = solving.get("save_lpfile", False)
        if save_lpfile and hasattr(snakemake, 'output') and hasattr(snakemake.output, 'lpfile'):
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
        elif save_lpfile:
            logger.warning("LP file saving is enabled but lpfile output not defined in Snakefile")
        else:
            logger.info("LP file saving is disabled (set solving.save_lpfile: true to enable)")
        
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

        cpu_threads = os.cpu_count() or 1
        base_solver_opts = kwargs.get("solver_options", {}) or {}
        configured_threads = (
            base_solver_opts.get("Threads")
            if "Threads" in base_solver_opts
            else base_solver_opts.get("threads", cpu_threads)
        )
        try:
            retry_threads = int(configured_threads)
        except (TypeError, ValueError):
            retry_threads = cpu_threads
        retry_threads = max(1, min(retry_threads, cpu_threads))

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
                "Threads": retry_threads,
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
            opts="1h",
            planning_horizons="2020",
            sopts="1h",
            configfile="config.myopic.yaml",
            discountrate="0.071",
            demand="AB",
            h2export="0.0"
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

    n = solve_network(
        n,
        config=snakemake.config,
        solving=snakemake.params.solving,
        log_fn=_safe_solver_log(snakemake),
    )
    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
    logger.info(f"Objective constant: {n.objective_constant}")
