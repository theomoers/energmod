#!/usr/bin/env python3
"""
Validate a solved PyPSA network against IRENA/IAEA/GEM capacities and OWID energy balances.

The script compares:
1. Installed/optimized electric capacity (MW) by country and aggregated technology group.
2. Electricity generation balance (TWh) by country and source categories.
3. Non-electric fossil energy use (TWh) by country and fuel.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pypsa
import yaml
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import yaml
except ImportError:  # pragma: no cover - runtime environment normally provides PyYAML
    yaml = None

REPO_ROOT = Path(__file__).parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from _helpers import (
    country_name_2_two_digits,
    read_osm_config,
    three_2_two_digits_country,
    two_2_three_digits_country,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_VALIDATION_GATE_CONFIG_PATH = REPO_ROOT / "validation" / "config.validation_gate.yaml"


def _repo_relative_path_str(path: Path) -> str:
    """Prefer repo-relative paths in validation artifacts/logs."""
    path = Path(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        pass
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except (ValueError, OSError, RuntimeError):
        return str(path)


DEMAND_METRIC_DEFINITIONS = {
    "electricity_demand": "load + link (excludes storage charging and Store withdrawals)",
    "electricity_demand_total_ac_withdrawal": "load + link + storage charging + Store withdrawals",
    "electricity_demand_load_component": "AC Load withdrawals only",
    "electricity_demand_link_component": "AC withdrawals attributed to Links",
    "electricity_demand_storage_charging": "AC withdrawals used for StorageUnit charging",
    "electricity_demand_store_component": "AC withdrawals used for Store charging",
}

VALIDATION_GATE_DOMAIN_SPECS = {
    "capacity": {
        "metric_col": "validation_tech",
        "model_col": "capacity_mw",
        "reference_col": "reference_mw",
        "abs_error_col": "abs_error_mw",
        "ape_col": "ape_pct",
        "unit": "MW",
    },
    "electricity_balance": {
        "metric_col": "metric",
        "model_col": "model_twh",
        "reference_col": "reference_twh",
        "abs_error_col": "abs_error_twh",
        "ape_col": "ape_pct",
        "unit": "TWh",
    },
    "electricity_demand": {
        "metric_col": "metric",
        "model_col": "model_twh",
        "reference_col": "reference_twh",
        "abs_error_col": "abs_error_twh",
        "ape_col": "ape_pct",
        "unit": "TWh",
    },
    "fossil_non_electric": {
        "metric_col": "metric",
        "model_col": "model_twh",
        "reference_col": "reference_twh",
        "abs_error_col": "abs_error_twh",
        "ape_col": "ape_pct",
        "unit": "TWh",
    },
}


# IRENA technology categories -> validation categories (non-fossil electric techs).
IRENA_TO_VALIDATION_TECH = {
    "PV": "solar",
    "Onshore": "onshore_wind",
    "Offshore": "offshore_wind",
    "Renewable hydropower": "hydro",
    "Pumped Storage": "hydro",
    "Mixed Hydro Plants": "hydro",
    "Geothermal": "geothermal",
    "Nuclear": "nuclear",
    "Biogas": "bioenergy",
    "Liquid biofuels": "bioenergy",
    "Solid biofuels": "bioenergy",
    "Renewable municipal waste": "bioenergy",
}

IRENA_FOSSIL_AGGREGATE_TECHS = {
    "Natural gas",
    "Coal and peat",
    "Oil",
    "Fossil fuels n.e.s.",
    "Other non-renewable energy",
}


# Model carrier -> validation categories for installed capacity
MODEL_CAPACITY_TECH_MAP = {
    "solar": "solar",
    "onwind": "onshore_wind",
    "offwind-ac": "offshore_wind",
    "offwind-dc": "offshore_wind",
    "ror": "hydro",
    "hydro": "hydro",
    "PHS": "hydro",
    "geothermal": "geothermal",
    "nuclear": "nuclear",
    "OCGT": "gas",
    "CCGT": "gas",
    "urban central gas CHP": "gas",
    "urban central gas CHP CC": "gas",
    "coal": "coal",
    "lignite": "coal",
    "oil": "oil",
    "biomass": "bioenergy",
    "urban central solid biomass CHP": "bioenergy",
    "urban central solid biomass CHP CC": "bioenergy",
}

FOSSIL_BUS_CARRIERS = {"gas", "oil", "coal", "lignite"}
FOSSIL_VALIDATION_TECHS = {"gas", "oil", "coal"}
MODEL_FOSSIL_TECHS_FOR_AGGREGATE = {"gas", "coal", "oil"}
VALIDATION_TECH_FOSSIL_AGGREGATED = "fossil_aggregated"


# Fallback hints for unmapped model carriers (used only when a clear match exists).
OTHER_RENEWABLE_HINTS = ("csp", "marine", "wave", "tidal")
OTHER_NON_RENEWABLE_HINTS = ("fossil", "peat", "oil shale", "non-renewable")


# Model carrier -> OWID electricity columns
MODEL_ENERGY_METRIC_MAP = {
    "solar": "solar_electricity",
    "onwind": "wind_electricity",
    "offwind-ac": "wind_electricity",
    "offwind-dc": "wind_electricity",
    "ror": "hydro_electricity",
    "hydro": "hydro_electricity",
    "geothermal": "other_renewable_electricity",
    "nuclear": "nuclear_electricity",
    "OCGT": "gas_electricity",
    "CCGT": "gas_electricity",
    "urban central gas CHP": "gas_electricity",
    "urban central gas CHP CC": "gas_electricity",
    "coal": "coal_electricity",
    "lignite": "coal_electricity",
    "oil": "oil_electricity",
    "biomass": "biofuel_electricity",
    "urban central solid biomass CHP": "biofuel_electricity",
    "urban central solid biomass CHP CC": "biofuel_electricity",
}


OWID_ENERGY_COLUMNS = [
    "coal_electricity",
    "gas_electricity",
    "oil_electricity",
    "biofuel_electricity",
    "hydro_electricity",
    "nuclear_electricity",
    "solar_electricity",
    "wind_electricity",
    "other_renewable_electricity",
    "fossil_electricity",
    "renewables_electricity",
    "low_carbon_electricity",
    "electricity_generation",
]

OWID_NON_ELECTRIC_COLUMNS = [
    "oil_consumption",
    "gas_consumption",
    "coal_consumption",
    "oil_electricity",
    "gas_electricity",
    "coal_electricity",
]

OWID_ELECTRICITY_DEMAND_COLUMN = "electricity_demand"
ELECTRICITY_TECH_METRICS = [
    "coal_electricity",
    "gas_electricity",
    "oil_electricity",
    "biofuel_electricity",
    "hydro_electricity",
    "nuclear_electricity",
    "solar_electricity",
    "wind_electricity",
    "other_renewable_electricity",
]

FOSSIL_BUS_CARRIER_TO_NON_ELEC_METRIC = {
    "oil": "oil_non_electric_consumption",
    "gas": "gas_non_electric_consumption",
    "coal": "coal_non_electric_consumption",
    "lignite": "coal_non_electric_consumption",
}

IAEA_COUNTRY_ALIASES = {
    "UAE": "United Arab Emirates",
}


EXTENDABLE_COMPONENT_SPECS = {
    "Generator": {
        "list_name": "generators",
        "extendable_col": "p_nom_extendable",
        "nom_col": "p_nom",
        "nom_opt_col": "p_nom_opt",
        "nom_min_col": "p_nom_min",
        "bus_cols": ["bus"],
    },
    "Link": {
        "list_name": "links",
        "extendable_col": "p_nom_extendable",
        "nom_col": "p_nom",
        "nom_opt_col": "p_nom_opt",
        "nom_min_col": "p_nom_min",
        "bus_cols": ["bus0", "bus1", "bus2", "bus3", "bus4"],
    },
    "StorageUnit": {
        "list_name": "storage_units",
        "extendable_col": "p_nom_extendable",
        "nom_col": "p_nom",
        "nom_opt_col": "p_nom_opt",
        "nom_min_col": "p_nom_min",
        "bus_cols": ["bus"],
    },
    "Store": {
        "list_name": "stores",
        "extendable_col": "e_nom_extendable",
        "nom_col": "e_nom",
        "nom_opt_col": "e_nom_opt",
        "nom_min_col": "e_nom_min",
        "bus_cols": ["bus"],
    },
    "Line": {
        "list_name": "lines",
        "extendable_col": "s_nom_extendable",
        "nom_col": "s_nom",
        "nom_opt_col": "s_nom_opt",
        "nom_min_col": "s_nom_min",
        "bus_cols": ["bus0", "bus1"],
    },
    "Transformer": {
        "list_name": "transformers",
        "extendable_col": "s_nom_extendable",
        "nom_col": "s_nom",
        "nom_opt_col": "s_nom_opt",
        "nom_min_col": "s_nom_min",
        "bus_cols": ["bus0", "bus1"],
    },
}

BASEYEAR_EXTENDABILITY_AUDIT_COLUMNS = [
    "year",
    "component",
    "asset",
    "carrier",
    "country",
    "country_context",
    "extendable_attr",
    "nominal_attr",
    "nominal_value",
    "nominal_opt_attr",
    "nominal_opt_value",
    "nominal_min_attr",
    "nominal_min_value",
    "build_year",
    "lifetime",
    "lifetime_is_inf",
    "is_zero_capacity_placeholder",
    "classification",
    "gate_bucket",
    "gate_blocking",
    "matched_rule_id",
    "matched_rule_rationale",
]

# Treat tiny optimized capacities as numerical noise for template placeholders.
EXTENDABILITY_PLACEHOLDER_ABS_TOL = 1e-3
BATTERY_BASEYEAR_CARRIERS_BY_COMPONENT = {
    "Link": {"battery charger", "battery discharger"},
    "Store": {"battery"},
    "StorageUnit": {"battery"},
}


@lru_cache(maxsize=1)
def _country_to_continent_lookup() -> dict[str, str]:
    """
    Build ISO2->continent lookup from regions_definition_config.yaml.
    """
    try:
        world_iso = read_osm_config("world_iso")
    except Exception as exc:
        LOGGER.warning("Could not load world_iso mapping for continent rollups: %s", exc)
        return {}

    lookup: dict[str, str] = {}
    if isinstance(world_iso, dict):
        for continent, entries in world_iso.items():
            if not isinstance(entries, dict):
                continue
            for iso2 in entries.keys():
                if isinstance(iso2, str) and len(iso2) == 2:
                    lookup[iso2] = str(continent)
    return lookup

def _capacity_column(df: pd.DataFrame) -> str:
    if "p_nom_opt" in df.columns and df["p_nom_opt"].notna().any():
        return "p_nom_opt"
    return "p_nom"


def _bus_country_lookup(n: pypsa.Network) -> pd.Series:
    """
    Build a robust bus->country lookup.

    Some sectoral fuel buses do not populate `buses.country`; for these,
    infer ISO2 from `buses.location` or bus name prefix.
    """
    country = (
        n.buses["country"].replace("", np.nan)
        if "country" in n.buses.columns
        else pd.Series("", index=n.buses.index, dtype=object)
    )
    location_series = (
        n.buses["location"]
        if "location" in n.buses.columns
        else pd.Series("", index=n.buses.index, dtype=object)
    )
    location_iso2 = location_series.astype(str).str.extract(r"^([A-Z]{2})\b")[0]
    index_iso2 = n.buses.index.astype(str).to_series(index=n.buses.index).str.extract(r"^([A-Z]{2})\b")[0]
    return country.fillna(location_iso2).fillna(index_iso2).fillna("")


def _normalize_allowlist_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    return [str(value)]


def _load_baseyear_extendability_allowlist(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Baseyear extendability allowlist not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rules_raw = data.get("rules", []) if isinstance(data, dict) else data
    if not isinstance(rules_raw, list):
        raise ValueError(
            f"Invalid extendability allowlist format in {path}; expected a top-level 'rules' list."
        )

    normalized: list[dict] = []
    for i, rule in enumerate(rules_raw, start=1):
        if not isinstance(rule, dict):
            raise ValueError(f"Allowlist rule #{i} in {path} must be a mapping")
        normalized.append(
            {
                "id": str(rule.get("id", f"rule_{i}")),
                "classification": str(rule.get("classification", "allowed")),
                "rationale": str(rule.get("rationale", "")),
                "component": set(_normalize_allowlist_list(rule.get("component"))),
                "carrier": set(_normalize_allowlist_list(rule.get("carrier"))),
                "name": set(_normalize_allowlist_list(rule.get("name"))),
                "component_regex": rule.get("component_regex"),
                "carrier_regex": rule.get("carrier_regex"),
                "name_regex": rule.get("name_regex"),
                "extendable_attr": set(_normalize_allowlist_list(rule.get("extendable_attr"))),
                "lifetime_is_inf": rule.get("lifetime_is_inf"),
                "build_year_eq": rule.get("build_year_eq"),
            }
        )
    return normalized


def _allowlist_rule_matches_extendability_row(row: pd.Series, rule: dict) -> bool:
    for field in ["component", "carrier", "name", "extendable_attr"]:
        allowed = rule.get(field) or set()
        row_field = "asset" if field == "name" else field
        if allowed and str(row.get(row_field, "")) not in allowed:
            return False

    if rule.get("component_regex") and not re.search(str(rule["component_regex"]), str(row.get("component", ""))):
        return False
    if rule.get("carrier_regex") and not re.search(str(rule["carrier_regex"]), str(row.get("carrier", ""))):
        return False
    if rule.get("name_regex") and not re.search(str(rule["name_regex"]), str(row.get("asset", ""))):
        return False

    if rule.get("lifetime_is_inf") is not None:
        if bool(row.get("lifetime_is_inf", False)) != bool(rule["lifetime_is_inf"]):
            return False

    if rule.get("build_year_eq") is not None:
        row_year = pd.to_numeric(pd.Series([row.get("build_year")]), errors="coerce").iloc[0]
        try:
            rule_year = float(rule["build_year_eq"])
        except Exception:
            return False
        if not pd.notna(row_year) or not np.isclose(float(row_year), rule_year, rtol=0.0, atol=0.0):
            return False

    return True


def _first_non_empty_country_context(df: pd.DataFrame, bus_cols: list[str], bus_country: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if df.empty:
        out["country"] = pd.Series(dtype=object)
        out["country_context"] = pd.Series(dtype=object)
        return out

    mapped_cols = []
    for col in bus_cols:
        if col in df.columns:
            mapped = df[col].map(bus_country).fillna("").astype(str)
            mapped_cols.append(mapped.rename(col))

    if not mapped_cols:
        out["country"] = ""
        out["country_context"] = ""
        return out

    mapped_df = pd.concat(mapped_cols, axis=1)
    non_empty = mapped_df.replace("", np.nan)
    out["country"] = non_empty.bfill(axis=1).iloc[:, 0].fillna("")

    def _row_country_context(row: pd.Series) -> str:
        vals = sorted({str(v) for v in row.tolist() if isinstance(v, str) and v})
        return "|".join(vals)

    out["country_context"] = mapped_df.apply(_row_country_context, axis=1)
    return out


def _empty_baseyear_extendability_audit() -> pd.DataFrame:
    return pd.DataFrame(columns=BASEYEAR_EXTENDABILITY_AUDIT_COLUMNS)


def _classify_extendability_gate_buckets(audit_df: pd.DataFrame, year: int) -> pd.DataFrame:
    if audit_df.empty:
        out = audit_df.copy()
        if "is_zero_capacity_placeholder" not in out.columns:
            out["is_zero_capacity_placeholder"] = pd.Series(dtype=bool)
        if "gate_bucket" not in out.columns:
            out["gate_bucket"] = pd.Series(dtype=object)
        if "gate_blocking" not in out.columns:
            out["gate_blocking"] = pd.Series(dtype=bool)
        return out

    out = audit_df.copy()
    nominal = pd.to_numeric(out["nominal_value"], errors="coerce").fillna(0.0).abs()
    nominal_opt = pd.to_numeric(out["nominal_opt_value"], errors="coerce").fillna(0.0).abs()
    build_year = pd.to_numeric(out["build_year"], errors="coerce")
    build_year_is_baseyear = build_year.notna() & build_year.eq(float(year))
    zero_placeholder = (
        out["classification"].eq("unexpected")
        & out["component"].isin(["Link", "Store"])
        & build_year_is_baseyear
        & (nominal <= EXTENDABILITY_PLACEHOLDER_ABS_TOL)
        & (nominal_opt <= EXTENDABILITY_PLACEHOLDER_ABS_TOL)
    )
    carrier_norm = out["carrier"].fillna("").astype(str).str.lower()
    battery_placeholder = pd.Series(False, index=out.index)
    for component, carriers in BATTERY_BASEYEAR_CARRIERS_BY_COMPONENT.items():
        battery_placeholder |= out["component"].eq(component) & carrier_norm.isin(set(carriers))
    zero_placeholder &= ~battery_placeholder
    out["is_zero_capacity_placeholder"] = zero_placeholder.astype(bool)

    out["gate_bucket"] = "unexpected_blocking"
    out.loc[out["classification"].eq("allowed"), "gate_bucket"] = "allowed"
    out.loc[
        out["classification"].eq("unexpected") & out["is_zero_capacity_placeholder"],
        "gate_bucket",
    ] = "unexpected_zero_placeholder_nonblocking"
    out["gate_blocking"] = out["gate_bucket"].eq("unexpected_blocking")
    return out


def _baseyear_extendability_audit(
    n: pypsa.Network,
    year: int,
    allowlist_rules: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bus_country = _bus_country_lookup(n)
    audit_parts: list[pd.DataFrame] = []

    for component_name, spec in EXTENDABLE_COMPONENT_SPECS.items():
        if not hasattr(n, spec["list_name"]):
            continue
        df = getattr(n, spec["list_name"])
        extendable_col = spec["extendable_col"]
        if extendable_col not in df.columns:
            continue
        extendable_mask = df[extendable_col].fillna(False).astype(bool)
        if not extendable_mask.any():
            continue

        sub = df.loc[extendable_mask].copy()
        ctx = _first_non_empty_country_context(sub, spec["bus_cols"], bus_country)

        carrier = (
            sub["carrier"].astype(str)
            if "carrier" in sub.columns
            else pd.Series("", index=sub.index, dtype=object)
        )
        if component_name == "Line":
            carrier = carrier.replace("", "AC")
        elif component_name == "Transformer":
            carrier = carrier.replace("", "transformer")

        nominal = (
            pd.to_numeric(sub.get(spec["nom_col"]), errors="coerce")
            if spec["nom_col"] in sub.columns
            else pd.Series(np.nan, index=sub.index)
        )
        nominal_opt = (
            pd.to_numeric(sub.get(spec["nom_opt_col"]), errors="coerce")
            if spec["nom_opt_col"] in sub.columns
            else pd.Series(np.nan, index=sub.index)
        )
        nominal_min = (
            pd.to_numeric(sub.get(spec["nom_min_col"]), errors="coerce")
            if spec["nom_min_col"] in sub.columns
            else pd.Series(np.nan, index=sub.index)
        )
        build_year = (
            pd.to_numeric(sub.get("build_year"), errors="coerce")
            if "build_year" in sub.columns
            else pd.Series(np.nan, index=sub.index)
        )
        lifetime = (
            pd.to_numeric(sub.get("lifetime"), errors="coerce")
            if "lifetime" in sub.columns
            else pd.Series(np.nan, index=sub.index)
        )

        audit = pd.DataFrame(index=sub.index)
        audit["year"] = int(year)
        audit["component"] = component_name
        audit["asset"] = sub.index.astype(str)
        audit["carrier"] = carrier.fillna("").astype(str)
        audit["country"] = ctx["country"].fillna("")
        audit["country_context"] = ctx["country_context"].fillna("")
        audit["extendable_attr"] = extendable_col
        audit["nominal_attr"] = spec["nom_col"]
        audit["nominal_value"] = nominal
        audit["nominal_opt_attr"] = spec["nom_opt_col"]
        audit["nominal_opt_value"] = nominal_opt
        audit["nominal_min_attr"] = spec["nom_min_col"]
        audit["nominal_min_value"] = nominal_min
        audit["build_year"] = build_year
        audit["lifetime"] = lifetime
        audit["lifetime_is_inf"] = np.isinf(lifetime.fillna(np.nan))
        audit["classification"] = "unexpected"
        audit["matched_rule_id"] = ""
        audit["matched_rule_rationale"] = ""

        for idx, row in audit.iterrows():
            for rule in allowlist_rules:
                if _allowlist_rule_matches_extendability_row(row, rule):
                    audit.at[idx, "classification"] = str(rule.get("classification", "allowed"))
                    audit.at[idx, "matched_rule_id"] = str(rule.get("id", ""))
                    audit.at[idx, "matched_rule_rationale"] = str(rule.get("rationale", ""))
                    break

        audit_parts.append(audit.reset_index(drop=True))

    if audit_parts:
        audit_df = pd.concat(audit_parts, ignore_index=True)
        audit_df = _classify_extendability_gate_buckets(audit_df, year=year)
        audit_df = audit_df.loc[:, BASEYEAR_EXTENDABILITY_AUDIT_COLUMNS].copy()
        audit_df = audit_df.sort_values(
            ["gate_bucket", "classification", "component", "carrier", "country", "asset"],
            ascending=[True, True, True, True, True, True],
            ignore_index=True,
        )
    else:
        audit_df = _empty_baseyear_extendability_audit()

    summary = pd.DataFrame(
        [
            {
                "year": int(year),
                "total_extendable_assets": int(len(audit_df)),
                "allowed_assets": int((audit_df["classification"] == "allowed").sum()),
                "unexpected_assets": int((audit_df["classification"] == "unexpected").sum()),
                "unexpected_blocking_assets": int(
                    (audit_df.get("gate_blocking", pd.Series(False, index=audit_df.index))).sum()
                ),
                "unexpected_zero_placeholder_nonblocking_assets": int(
                    (
                        audit_df.get(
                            "is_zero_capacity_placeholder",
                            pd.Series(False, index=audit_df.index),
                        )
                    ).sum()
                ),
                "status": (
                    "fail"
                    if bool(
                        audit_df.get("gate_blocking", pd.Series(False, index=audit_df.index)).any()
                    )
                    else "pass"
                ),
            }
        ]
    )
    return audit_df, summary


def _safe_iso3_to_iso2(code: str) -> str | np.nan:
    if not isinstance(code, str) or len(code) != 3:
        return np.nan
    try:
        iso2 = three_2_two_digits_country(code)
    except Exception:
        return np.nan
    if not isinstance(iso2, str) or len(iso2) != 2:
        return np.nan
    return iso2


def _link_electric_capacity(
    n: pypsa.Network, exclude_bus0_carriers: set[str] | None = None
) -> pd.DataFrame:
    cap_col = _capacity_column(n.links)
    link_df = n.links.copy()
    bus0_carrier = link_df["bus0"].map(n.buses.carrier)
    bus_country = _bus_country_lookup(n)

    pieces = []
    for port in [1, 2, 3, 4]:
        bus_col = f"bus{port}"
        if bus_col not in link_df.columns:
            continue

        eff_col = "efficiency" if port == 1 else f"efficiency{port}"
        if eff_col not in link_df.columns:
            continue

        buses = link_df[bus_col]
        bus_carrier = buses.map(n.buses.carrier)
        mask = buses.notna() & buses.ne("") & bus_carrier.eq("AC")
        if exclude_bus0_carriers:
            mask &= ~bus0_carrier.isin(exclude_bus0_carriers)
        if not mask.any():
            continue

        sub = link_df.loc[mask, [cap_col, eff_col, "carrier", bus_col]].copy()
        sub["country"] = sub[bus_col].map(bus_country).fillna("")
        sub["component"] = "Link"
        sub["capacity_mw"] = sub[cap_col].astype(float) * sub[eff_col].astype(float).abs()
        sub = sub.loc[sub["capacity_mw"] > 0, ["component", "country", "carrier", "capacity_mw"]]
        pieces.append(sub)

    if not pieces:
        return pd.DataFrame(columns=["component", "country", "carrier", "capacity_mw"])

    return pd.concat(pieces, ignore_index=True)


def _fossil_electric_link_carriers(n: pypsa.Network) -> set[str]:
    """
    Return link carriers that consume fossil fuel (bus0) and produce electricity (any AC output port).
    """
    links = n.links.copy()
    links["bus0_carrier"] = links["bus0"].map(n.buses.carrier)
    has_ac_output = pd.Series(False, index=links.index)

    for port in [1, 2, 3, 4]:
        bus_col = f"bus{port}"
        if bus_col not in links.columns:
            continue
        has_ac_output |= links[bus_col].map(n.buses.carrier).eq("AC")

    mask = links["bus0_carrier"].isin(FOSSIL_BUS_CARRIERS) & has_ac_output
    return set(links.loc[mask, "carrier"].dropna().unique())


def _infer_other_capacity_bucket(carrier: str) -> str | None:
    c = str(carrier).lower()
    if any(k in c for k in OTHER_RENEWABLE_HINTS):
        return "other_renewables"
    if any(k in c for k in OTHER_NON_RENEWABLE_HINTS):
        return "other_non_renewables"
    return None


def _model_capacity_by_country_tech(
    n: pypsa.Network,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gen_col = _capacity_column(n.generators)
    su_col = _capacity_column(n.storage_units)
    bus_country = _bus_country_lookup(n)

    gen = n.generators.copy()
    gen = gen.loc[gen.bus.map(n.buses.carrier).eq("AC")].copy()
    gen["component"] = "Generator"
    gen["country"] = gen.bus.map(bus_country).fillna("")
    gen["capacity_mw"] = gen[gen_col].astype(float)
    gen = gen.loc[:, ["component", "country", "carrier", "capacity_mw"]]
    gen = gen.loc[gen.capacity_mw > 0]

    su = n.storage_units.copy()
    su = su.loc[su.bus.map(n.buses.carrier).eq("AC")].copy()
    su["component"] = "StorageUnit"
    su["country"] = su.bus.map(bus_country).fillna("")
    su["capacity_mw"] = su[su_col].astype(float)
    su = su.loc[:, ["component", "country", "carrier", "capacity_mw"]]
    su = su.loc[su.capacity_mw > 0]

    links = _link_electric_capacity(n)

    model_raw = pd.concat([gen, su, links], ignore_index=True)
    model_raw = model_raw.loc[model_raw.country.ne("")]
    # Exclude load-shedding pseudo-generators from capacity validation diagnostics.
    model_raw = model_raw.loc[
        ~((model_raw["component"] == "Generator") & (model_raw["carrier"] == "Load"))
    ].copy()

    model_raw["validation_tech"] = model_raw["carrier"].map(MODEL_CAPACITY_TECH_MAP)
    # Fossil capacity validation must use conversion links, not fuel-supply generators.
    model_raw = model_raw.loc[
        ~(
            model_raw["component"].ne("Link")
            & model_raw["validation_tech"].isin(FOSSIL_VALIDATION_TECHS)
        )
    ].copy()

    model_raw["direct_validation_tech"] = model_raw["validation_tech"]
    model_raw["inferred_validation_tech"] = np.where(
        model_raw["direct_validation_tech"].isna(),
        model_raw["carrier"].apply(_infer_other_capacity_bucket),
        np.nan,
    )
    model_raw["final_validation_tech"] = model_raw["direct_validation_tech"].fillna(
        model_raw["inferred_validation_tech"]
    )
    model_raw["mapping_status"] = np.where(
        model_raw["direct_validation_tech"].notna(),
        "mapped_direct",
        np.where(
            model_raw["inferred_validation_tech"].notna(),
            "mapped_inferred",
            "unmapped",
        ),
    )
    model_raw.loc[model_raw["final_validation_tech"].eq("oil"), "mapping_status"] = (
        "mapped_direct_oil_aggregated_only"
    )

    mapping_diag = (
        model_raw.groupby(
            [
                "component",
                "carrier",
                "mapping_status",
                "direct_validation_tech",
                "inferred_validation_tech",
                "final_validation_tech",
            ],
            dropna=False,
            as_index=False,
        )["capacity_mw"]
        .sum()
        .sort_values("capacity_mw", ascending=False)
    )

    mapped = model_raw.loc[model_raw.validation_tech.notna()].copy()
    mapped = (
        mapped.groupby(["country", "validation_tech"], as_index=False)["capacity_mw"]
        .sum()
    )

    fossil_agg = mapped.loc[
        mapped["validation_tech"].isin(MODEL_FOSSIL_TECHS_FOR_AGGREGATE)
    ].copy()
    if not fossil_agg.empty:
        fossil_agg = (
            fossil_agg.groupby("country", as_index=False)["capacity_mw"].sum()
        )
        fossil_agg["validation_tech"] = VALIDATION_TECH_FOSSIL_AGGREGATED
        mapped = pd.concat([mapped, fossil_agg], ignore_index=True)
        mapped = (
            mapped.groupby(["country", "validation_tech"], as_index=False)["capacity_mw"]
            .sum()
        )

    # Oil is validated only through the aggregated fossil category.
    mapped = mapped.loc[mapped["validation_tech"] != "oil"].copy()

    unmapped = model_raw.loc[model_raw.validation_tech.isna()].copy()
    unmapped["inferred_validation_tech"] = unmapped["carrier"].apply(_infer_other_capacity_bucket)

    inferred = unmapped.loc[unmapped["inferred_validation_tech"].notna()].copy()
    if not inferred.empty:
        inferred = (
            inferred.groupby(["country", "inferred_validation_tech"], as_index=False)["capacity_mw"]
            .sum()
            .rename(columns={"inferred_validation_tech": "validation_tech"})
        )
        mapped = pd.concat([mapped, inferred], ignore_index=True)
        mapped = mapped.groupby(["country", "validation_tech"], as_index=False)["capacity_mw"].sum()

    unmapped = unmapped.loc[unmapped["inferred_validation_tech"].isna()].copy()
    unmapped = (
        unmapped.groupby(["component", "carrier"], as_index=False)["capacity_mw"]
        .sum()
        .sort_values("capacity_mw", ascending=False)
    )

    return mapped, unmapped, mapping_diag


def _irena_capacity_by_country_tech(irena_csv: Path, year: int) -> pd.DataFrame:
    irena = pd.read_csv(irena_csv)
    year_col = str(year)
    if year_col not in irena.columns:
        raise ValueError(f"Year column '{year_col}' not found in {irena_csv}")

    ref = irena[["Technology", "Country", year_col]].copy()
    ref = ref.rename(columns={"Country": "country", year_col: "reference_mw"})
    ref["country"] = ref["country"].astype(str).str.strip()
    ref["reference_mw"] = pd.to_numeric(ref["reference_mw"], errors="coerce").fillna(0.0)
    ref["validation_tech"] = ref["Technology"].map(IRENA_TO_VALIDATION_TECH)
    ref = ref.loc[ref.validation_tech.notna()].copy()
    ref = ref.groupby(["country", "validation_tech"], as_index=False)["reference_mw"].sum()

    return ref


def _irena_fossil_aggregated_capacity_by_country(irena_csv: Path, year: int) -> pd.DataFrame:
    irena = pd.read_csv(irena_csv)
    year_col = str(year)
    if year_col not in irena.columns:
        raise ValueError(f"Year column '{year_col}' not found in {irena_csv}")

    ref = irena[["Technology", "Country", year_col]].copy()
    ref = ref.rename(columns={"Country": "country", year_col: "reference_mw"})
    ref["country"] = ref["country"].astype(str).str.strip()
    ref["reference_mw"] = pd.to_numeric(ref["reference_mw"], errors="coerce").fillna(0.0)
    ref = ref.loc[ref["Technology"].isin(IRENA_FOSSIL_AGGREGATE_TECHS)].copy()
    ref["validation_tech"] = VALIDATION_TECH_FOSSIL_AGGREGATED
    ref = ref.groupby(["country", "validation_tech"], as_index=False)["reference_mw"].sum()
    return ref


def _irena_single_tech_capacity_by_country(irena_csv: Path, year: int, tech: str) -> pd.DataFrame:
    irena = pd.read_csv(irena_csv)
    year_col = str(year)
    if year_col not in irena.columns:
        raise ValueError(f"Year column '{year_col}' not found in {irena_csv}")
    ref = irena.loc[irena["Technology"] == tech, ["Country", year_col]].copy()
    ref = ref.rename(columns={"Country": "country", year_col: "reference_mw"})
    ref["country"] = ref["country"].astype(str).str.strip()
    ref["reference_mw"] = pd.to_numeric(ref["reference_mw"], errors="coerce").fillna(0.0)
    return ref.groupby("country", as_index=False)["reference_mw"].sum()


def _nuclear_reference_with_iaea_fallback(irena_csv: Path, iaea_csv: Path, year: int) -> pd.DataFrame:
    """
    Nuclear reference prioritization:
    1) Use IAEA where available and positive.
    2) If IAEA is missing/zero and IRENA has positive value, use IRENA.
    """
    iaea = _iaea_nuclear_capacity_by_country(iaea_csv).rename(
        columns={"reference_mw": "iaea_mw"}
    )
    irena = _irena_single_tech_capacity_by_country(irena_csv, year, "Nuclear").rename(
        columns={"reference_mw": "irena_mw"}
    )
    irena["validation_tech"] = "nuclear"
    merged = iaea.merge(irena, on=["country", "validation_tech"], how="outer")
    merged["iaea_mw"] = pd.to_numeric(merged["iaea_mw"], errors="coerce").fillna(0.0)
    merged["irena_mw"] = pd.to_numeric(merged["irena_mw"], errors="coerce").fillna(0.0)
    merged["reference_mw"] = np.where(
        (merged["iaea_mw"] <= 0.0) & (merged["irena_mw"] > 0.0),
        merged["irena_mw"],
        merged["iaea_mw"],
    )
    fallback_rows = int(((merged["iaea_mw"] <= 0.0) & (merged["irena_mw"] > 0.0)).sum())
    if fallback_rows:
        LOGGER.info(
            "Nuclear fallback: used IRENA for %d country entries where IAEA is missing/zero.",
            fallback_rows,
        )
    return merged.loc[:, ["country", "validation_tech", "reference_mw"]]


@lru_cache(maxsize=1024)
def _safe_country_name_to_iso2(name: str) -> str | np.nan:
    if not isinstance(name, str):
        return np.nan
    clean = name.strip()
    if not clean:
        return np.nan
    alias = IAEA_COUNTRY_ALIASES.get(clean.upper())
    candidates = [clean, clean.title()]
    if alias:
        candidates = [alias] + candidates
    for candidate in candidates:
        try:
            iso2 = country_name_2_two_digits(candidate)
        except Exception:
            continue
        if isinstance(iso2, str) and len(iso2) == 2:
            return iso2
    return np.nan


def _gem_clean_capacity_by_country_tech(gem_clean_csv: Path, year: int) -> pd.DataFrame:
    gem = pd.read_csv(gem_clean_csv, encoding="utf-8-sig")
    required = {"country", "validation_tech", "reference_mw"}
    missing = sorted(required - set(gem.columns))
    if missing:
        raise ValueError(
            f"Missing required columns in {gem_clean_csv}: {', '.join(missing)}"
        )

    if "year" in gem.columns:
        gem = gem.loc[pd.to_numeric(gem["year"], errors="coerce") == year].copy()

    gem["country"] = gem["country"].astype(str).str.strip().str.upper()
    gem["validation_tech"] = gem["validation_tech"].astype(str).str.strip()
    gem["reference_mw"] = pd.to_numeric(gem["reference_mw"], errors="coerce").fillna(0.0)
    gem = gem.loc[gem["validation_tech"].isin(["gas", "coal"])].copy()
    gem = gem.groupby(["country", "validation_tech"], as_index=False)["reference_mw"].sum()
    return gem


def _iaea_nuclear_capacity_by_country(iaea_csv: Path) -> pd.DataFrame:
    iaea = pd.read_csv(iaea_csv, encoding="utf-8-sig")
    country_col = next((c for c in iaea.columns if "country" in str(c).lower()), None)
    capacity_col = next((c for c in iaea.columns if "capacity" in str(c).lower()), None)
    if country_col is None or capacity_col is None:
        raise ValueError(f"Could not find country/capacity columns in {iaea_csv}")

    ref = iaea[[country_col, capacity_col]].copy()
    ref = ref.rename(columns={country_col: "country_name", capacity_col: "reference_mw"})
    ref["country"] = ref["country_name"].apply(_safe_country_name_to_iso2)
    ref["reference_mw"] = pd.to_numeric(ref["reference_mw"], errors="coerce").fillna(0.0)
    ref = ref.loc[ref["country"].notna()].copy()
    ref["validation_tech"] = "nuclear"
    ref = ref.groupby(["country", "validation_tech"], as_index=False)["reference_mw"].sum()
    return ref


def _capacity_reference_by_country_tech(
    irena_csv: Path,
    iaea_csv: Path,
    gem_clean_csv: Path,
    year: int,
) -> pd.DataFrame:
    irena_ref = _irena_capacity_by_country_tech(irena_csv, year)
    irena_ref = irena_ref.loc[
        ~irena_ref["validation_tech"].isin(["nuclear", "gas", "coal", "oil"])
    ].copy()

    nuclear_ref = _nuclear_reference_with_iaea_fallback(irena_csv, iaea_csv, year)
    if not gem_clean_csv.exists():
        raise FileNotFoundError(
            f"Cleaned GEM reference not found at {gem_clean_csv}. "
            "Run validation/scripts/prepare_gem_capacity_reference.py first."
        )
    gem_ref = _gem_clean_capacity_by_country_tech(gem_clean_csv, year)
    gem_gas_ref = gem_ref.loc[gem_ref["validation_tech"] == "gas"].copy()
    gem_coal_ref = gem_ref.loc[gem_ref["validation_tech"] == "coal"].copy()
    if gem_coal_ref.empty:
        LOGGER.warning(
            "No coal rows found in cleaned GEM file %s; falling back to IRENA 'Coal and peat'.",
            gem_clean_csv,
        )
        coal = _irena_single_tech_capacity_by_country(irena_csv, year, "Coal and peat")
        coal["validation_tech"] = "coal"
        gem_coal_ref = coal

    irena_fossil_agg = _irena_fossil_aggregated_capacity_by_country(irena_csv, year)

    ref = pd.concat(
        [irena_ref, nuclear_ref, gem_gas_ref, gem_coal_ref, irena_fossil_agg], ignore_index=True
    )
    return ref.groupby(["country", "validation_tech"], as_index=False)["reference_mw"].sum()


def _capacity_comparison(model_cap: pd.DataFrame, ref_cap: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cmp_df = model_cap.merge(
        ref_cap,
        on=["country", "validation_tech"],
        how="outer",
    ).fillna({"capacity_mw": 0.0, "reference_mw": 0.0})

    cmp_df["error_mw"] = cmp_df["capacity_mw"] - cmp_df["reference_mw"]
    cmp_df["abs_error_mw"] = cmp_df["error_mw"].abs()
    cmp_df["ape_pct"] = np.where(
        cmp_df["reference_mw"] > 0,
        cmp_df["abs_error_mw"] / cmp_df["reference_mw"] * 100.0,
        np.nan,
    )

    summary = cmp_df.groupby("validation_tech", as_index=False).agg(
        countries=("country", "nunique"),
        model_total_mw=("capacity_mw", "sum"),
        reference_total_mw=("reference_mw", "sum"),
        mae_mw=("abs_error_mw", "mean"),
        rmse_mw=("error_mw", lambda s: float(np.sqrt(np.mean(np.square(s))))),
        mean_ape_pct=("ape_pct", "mean"),
        wape_pct=(
            "abs_error_mw",
            lambda s: (
                float(s.sum())
                / max(
                    cmp_df.loc[s.index, "reference_mw"].sum(),
                    1e-9,
                )
                * 100.0
            ),
        ),
    )

    return cmp_df, summary.sort_values("wape_pct", ascending=False)


def _model_electricity_balance(n: pypsa.Network) -> pd.DataFrame:
    eb = n.statistics.energy_balance(
        bus_carrier="AC",
        aggregate_time="sum",
        aggregate_groups="sum",
        groupby=n.statistics.groupers.get_country_and_carrier,
        nice_names=False,
    )
    eb = eb.rename("energy_mwh").reset_index()
    eb = eb.loc[eb.component.isin(["Generator", "StorageUnit", "Link"])].copy()
    eb = eb.loc[(eb.country != "") & (eb.energy_mwh > 0)].copy()
    eb["metric"] = eb["carrier"].map(MODEL_ENERGY_METRIC_MAP)
    eb = eb.loc[eb.metric.notna()].copy()

    model = eb.groupby(["country", "metric"], as_index=False)["energy_mwh"].sum()
    model["model_twh"] = model["energy_mwh"] / 1e6
    model = model.drop(columns=["energy_mwh"])

    wide = model.pivot(index="country", columns="metric", values="model_twh").fillna(0.0)
    for col in [
        "coal_electricity",
        "gas_electricity",
        "oil_electricity",
        "biofuel_electricity",
        "hydro_electricity",
        "nuclear_electricity",
        "solar_electricity",
        "wind_electricity",
        "other_renewable_electricity",
    ]:
        if col not in wide.columns:
            wide[col] = 0.0

    wide["fossil_electricity"] = (
        wide["coal_electricity"] + wide["gas_electricity"] + wide["oil_electricity"]
    )
    wide["renewables_electricity"] = (
        wide["hydro_electricity"]
        + wide["solar_electricity"]
        + wide["wind_electricity"]
        + wide["biofuel_electricity"]
        + wide["other_renewable_electricity"]
    )
    wide["low_carbon_electricity"] = wide["renewables_electricity"] + wide["nuclear_electricity"]
    wide["electricity_generation"] = wide["fossil_electricity"] + wide["low_carbon_electricity"]

    out = wide.reset_index().melt(
        id_vars="country",
        value_vars=OWID_ENERGY_COLUMNS,
        var_name="metric",
        value_name="model_twh",
    )
    return out


def _model_electricity_mapping_diagnostics(n: pypsa.Network) -> pd.DataFrame:
    eb = n.statistics.energy_balance(
        bus_carrier="AC",
        aggregate_time="sum",
        aggregate_groups="sum",
        groupby=n.statistics.groupers.get_country_and_carrier,
        nice_names=False,
    )
    eb = eb.rename("energy_mwh").reset_index()
    eb = eb.loc[eb.component.isin(["Generator", "StorageUnit", "Link"])].copy()
    eb = eb.loc[(eb.country != "") & (eb.energy_mwh > 0)].copy()
    eb["metric"] = eb["carrier"].map(MODEL_ENERGY_METRIC_MAP)
    eb["mapping_status"] = np.where(eb["metric"].notna(), "mapped_direct", "unmapped")
    eb["energy_twh"] = eb["energy_mwh"] / 1e6
    out = (
        eb.groupby(
            ["component", "carrier", "mapping_status", "metric"],
            dropna=False,
            as_index=False,
        )["energy_twh"]
        .sum()
        .sort_values("energy_twh", ascending=False)
    )
    return out


def _renewable_zero_profile_diagnostics(n: pypsa.Network) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect renewable assets with non-zero capacity but zero annual available energy.

    This catches cases like all-zero `p_max_pu` profile columns (e.g. JP/TW onwind).
    """
    bus_country = _bus_country_lookup(n)
    weights = n.snapshot_weightings["generators"]
    weight_sum = float(weights.sum())
    rows = []

    vre_carriers = {"solar", "onwind", "offwind-ac", "offwind-dc", "ror"}
    if not n.generators.empty:
        gen = n.generators.loc[n.generators["carrier"].astype(str).isin(vre_carriers)].copy()
        if not gen.empty:
            cap_col = _capacity_column(gen)
            cap_mw = pd.to_numeric(gen[cap_col], errors="coerce").fillna(
                pd.to_numeric(gen.get("p_nom"), errors="coerce").fillna(0.0)
            )
            gen["capacity_mw"] = cap_mw
            gen["country"] = gen["bus"].map(bus_country).fillna("")
            ts_cols = n.generators_t.p_max_pu.columns.intersection(gen.index)
            static_cols = gen.index.difference(ts_cols)
            availability_hours = pd.Series(0.0, index=gen.index, dtype=float)
            if len(ts_cols) > 0:
                availability_hours.loc[ts_cols] = (
                    n.generators_t.p_max_pu.reindex(columns=ts_cols)
                    .fillna(0.0)
                    .mul(weights, axis=0)
                    .sum(axis=0)
                    .astype(float)
                )
            if len(static_cols) > 0:
                static_pmax = (
                    pd.to_numeric(gen.loc[static_cols, "p_max_pu"], errors="coerce")
                    .fillna(0.0)
                    .clip(lower=0.0)
                )
                availability_hours.loc[static_cols] = static_pmax * weight_sum

            gen["available_twh"] = availability_hours.mul(gen["capacity_mw"]).div(1e6)
            gen["profile_all_zero"] = (gen["capacity_mw"] > 0.0) & (gen["available_twh"] <= 1e-9)
            gen_out = gen.reset_index()
            gen_out = gen_out.rename(columns={gen_out.columns[0]: "asset"})
            rows.extend(
                gen_out.assign(component="Generator")[[
                    "component",
                    "asset",
                    "country",
                    "carrier",
                    "capacity_mw",
                    "available_twh",
                    "profile_all_zero",
                ]]
                .to_dict("records")
            )

    if not n.storage_units.empty:
        su = n.storage_units.loc[n.storage_units["carrier"].astype(str).eq("hydro")].copy()
        if not su.empty:
            cap_col = _capacity_column(su)
            su["capacity_mw"] = pd.to_numeric(su[cap_col], errors="coerce").fillna(
                pd.to_numeric(su.get("p_nom"), errors="coerce").fillna(0.0)
            )
            su["country"] = su["bus"].map(bus_country).fillna("")
            inflow_cols = n.storage_units_t.inflow.columns.intersection(su.index)
            inflow_twh = pd.Series(0.0, index=su.index, dtype=float)
            if len(inflow_cols) > 0:
                inflow_twh.loc[inflow_cols] = (
                    n.storage_units_t.inflow.reindex(columns=inflow_cols)
                    .fillna(0.0)
                    .mul(weights, axis=0)
                    .sum(axis=0)
                    .astype(float)
                    / 1e6
                )
            su["available_twh"] = inflow_twh
            su["profile_all_zero"] = (su["capacity_mw"] > 0.0) & (su["available_twh"] <= 1e-9)
            su_out = su.reset_index()
            su_out = su_out.rename(columns={su_out.columns[0]: "asset"})
            rows.extend(
                su_out.assign(component="StorageUnit", carrier="hydro_reservoir_inflow")[[
                    "component",
                    "asset",
                    "country",
                    "carrier",
                    "capacity_mw",
                    "available_twh",
                    "profile_all_zero",
                ]]
                .to_dict("records")
            )

    asset_diag = pd.DataFrame(
        rows,
        columns=[
            "component",
            "asset",
            "country",
            "carrier",
            "capacity_mw",
            "available_twh",
            "profile_all_zero",
        ],
    )
    if asset_diag.empty:
        summary = pd.DataFrame(
            columns=[
                "country",
                "carrier",
                "assets",
                "zero_profile_assets",
                "capacity_mw",
                "zero_profile_capacity_mw",
                "available_twh",
                "zero_profile_share_assets_pct",
            ]
        )
        return asset_diag, summary

    asset_diag["country"] = asset_diag["country"].fillna("")
    summary = (
        asset_diag.groupby(["country", "carrier"], as_index=False)
        .agg(
            assets=("asset", "count"),
            zero_profile_assets=("profile_all_zero", "sum"),
            capacity_mw=("capacity_mw", "sum"),
            zero_profile_capacity_mw=(
                "capacity_mw",
                lambda s: float(s[asset_diag.loc[s.index, "profile_all_zero"]].sum()),
            ),
            available_twh=("available_twh", "sum"),
        )
    )
    summary["zero_profile_share_assets_pct"] = np.where(
        summary["assets"] > 0,
        100.0 * summary["zero_profile_assets"] / summary["assets"],
        np.nan,
    )
    summary = summary.sort_values(
        ["zero_profile_assets", "zero_profile_capacity_mw"], ascending=False
    )
    return asset_diag.sort_values(["profile_all_zero", "capacity_mw"], ascending=[False, False]), summary


def _load_nan_diagnostics(n: pypsa.Network) -> pd.DataFrame:
    """Detect NaNs in AC load time series used for demand constraints."""
    if n.loads.empty or not hasattr(n.loads_t, "p_set"):
        return pd.DataFrame(
            columns=[
                "component",
                "asset",
                "country",
                "carrier",
                "nan_cells",
                "nan_snapshots",
                "max_abs_p_set_mw",
            ]
        )

    bus_country = _bus_country_lookup(n)
    bus_carrier = (
        n.buses["carrier"] if "carrier" in n.buses.columns else pd.Series("", index=n.buses.index, dtype=object)
    )
    loads = n.loads.copy()
    loads["country"] = loads["bus"].map(bus_country).fillna("")
    loads["bus_carrier"] = loads["bus"].map(bus_carrier).fillna("")
    loads = loads.loc[(loads["country"] != "") & (loads["bus_carrier"] == "AC")].copy()
    if loads.empty:
        return pd.DataFrame(
            columns=[
                "component",
                "asset",
                "country",
                "carrier",
                "nan_cells",
                "nan_snapshots",
                "max_abs_p_set_mw",
            ]
        )

    cols = n.loads_t.p_set.columns.intersection(loads.index)
    if len(cols) == 0:
        return pd.DataFrame(
            columns=[
                "component",
                "asset",
                "country",
                "carrier",
                "nan_cells",
                "nan_snapshots",
                "max_abs_p_set_mw",
            ]
        )

    p_set = n.loads_t.p_set.reindex(columns=cols)
    nan_mask = p_set.isna()
    nan_cells = nan_mask.sum(axis=0)
    hits = nan_cells.loc[nan_cells > 0]
    if hits.empty:
        return pd.DataFrame(
            columns=[
                "component",
                "asset",
                "country",
                "carrier",
                "nan_cells",
                "nan_snapshots",
                "max_abs_p_set_mw",
            ]
        )

    nan_snapshots = nan_mask.loc[:, hits.index].sum(axis=0)
    max_abs = p_set.loc[:, hits.index].abs().max(axis=0, skipna=True).fillna(0.0)
    out = loads.loc[hits.index].copy()
    if "carrier" not in out.columns:
        out["carrier"] = ""
    out["component"] = "Load"
    out["nan_cells"] = pd.to_numeric(hits, errors="coerce").fillna(0).astype(int)
    out["nan_snapshots"] = pd.to_numeric(nan_snapshots, errors="coerce").fillna(0).astype(int)
    out["max_abs_p_set_mw"] = pd.to_numeric(max_abs, errors="coerce").fillna(0.0)
    out = out.reset_index().rename(columns={out.index.name or "index": "asset"})
    return out[["component", "asset", "country", "carrier", "nan_cells", "nan_snapshots", "max_abs_p_set_mw"]]


def _hydro_missing_inflow_diagnostics(n: pypsa.Network) -> pd.DataFrame:
    """Detect hydro reservoir storage units with capacity but no inflow time series column."""
    if n.storage_units.empty:
        return pd.DataFrame(columns=["component", "asset", "country", "carrier", "capacity_mw"])

    bus_country = _bus_country_lookup(n)
    su = n.storage_units.loc[n.storage_units["carrier"].astype(str).eq("hydro")].copy()
    if su.empty:
        return pd.DataFrame(columns=["component", "asset", "country", "carrier", "capacity_mw"])

    cap_col = _capacity_column(su)
    su["capacity_mw"] = pd.to_numeric(su[cap_col], errors="coerce").fillna(
        pd.to_numeric(su.get("p_nom"), errors="coerce").fillna(0.0)
    )
    su["country"] = su["bus"].map(bus_country).fillna("")
    su = su.loc[(su["country"] != "") & (su["capacity_mw"] > 0.0)].copy()
    if su.empty:
        return pd.DataFrame(columns=["component", "asset", "country", "carrier", "capacity_mw"])

    inflow_cols = set(n.storage_units_t.inflow.columns)
    missing = su.loc[~su.index.isin(inflow_cols)].copy()
    if missing.empty:
        return pd.DataFrame(columns=["component", "asset", "country", "carrier", "capacity_mw"])
    missing = missing.reset_index().rename(columns={missing.index.name or "index": "asset"})
    missing["component"] = "StorageUnit"
    missing["carrier"] = "hydro_reservoir_inflow_missing_column"
    return missing[["component", "asset", "country", "carrier", "capacity_mw"]]


def _guardrail_status_rank(status: str) -> int:
    return {"pass": 0, "warn": 1, "fail": 2}.get(str(status), 2)


def _workflow_guardrail_artifacts(
    n: pypsa.Network,
    zero_profile_assets: pd.DataFrame,
    *,
    zero_profile_hit_severity: str,
    hydro_missing_inflow_hit_severity: str,
    hydro_zero_ror_hit_severity: str,
    hydro_zero_reservoir_inflow_hit_severity: str,
    nan_load_hit_severity: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    def add_check(
        *,
        check_id: str,
        check_group: str,
        description: str,
        severity_if_hit: str,
        detail: pd.DataFrame | None,
        hit_count_col: str | None = None,
        capacity_col: str | None = None,
        message: str = "",
    ) -> None:
        d = pd.DataFrame() if detail is None else detail.copy()
        if d.empty:
            status = "pass"
            hit_count = 0
            affected_assets = 0
            affected_capacity_mw = 0.0
        else:
            status = severity_if_hit
            if hit_count_col and hit_count_col in d.columns:
                hit_count = int(pd.to_numeric(d[hit_count_col], errors="coerce").fillna(0).sum())
            else:
                hit_count = int(len(d))
            affected_assets = int(d["asset"].nunique()) if "asset" in d.columns else int(len(d))
            affected_capacity_mw = (
                float(pd.to_numeric(d[capacity_col], errors="coerce").fillna(0.0).sum())
                if capacity_col and capacity_col in d.columns
                else 0.0
            )
            for _, row in d.iterrows():
                detail_rows.append(
                    {
                        "check_id": check_id,
                        "check_group": check_group,
                        "status": status,
                        "severity_if_hit": severity_if_hit,
                        "component": row.get("component", ""),
                        "asset": row.get("asset", ""),
                        "country": row.get("country", ""),
                        "carrier": row.get("carrier", ""),
                        "metric_name": (
                            "nan_cells"
                            if check_id == "nan_load_timeseries"
                            else "capacity_mw"
                            if ("capacity_mw" in d.columns)
                            else "count"
                        ),
                        "metric_value": (
                            float(row.get("nan_cells", 0))
                            if check_id == "nan_load_timeseries"
                            else float(row.get("capacity_mw", 0.0))
                            if "capacity_mw" in d.columns
                            else 1.0
                        ),
                        "secondary_metric_name": (
                            "nan_snapshots" if check_id == "nan_load_timeseries" else ""
                        ),
                        "secondary_metric_value": (
                            float(row.get("nan_snapshots", 0))
                            if check_id == "nan_load_timeseries"
                            else np.nan
                        ),
                        "message": message,
                    }
                )

        summary_rows.append(
            {
                "check_id": check_id,
                "check_group": check_group,
                "description": description,
                "status": status,
                "severity_if_hit": severity_if_hit,
                "status_rank": _guardrail_status_rank(status),
                "hit_count": hit_count,
                "affected_assets": affected_assets,
                "affected_capacity_mw": affected_capacity_mw,
                "message": message,
            }
        )

    zero_hits = zero_profile_assets.loc[zero_profile_assets.get("profile_all_zero", False)].copy()
    if not isinstance(zero_hits, pd.DataFrame):
        zero_hits = pd.DataFrame(columns=zero_profile_assets.columns)

    add_check(
        check_id="renewable_zero_profile_assets",
        check_group="profiles",
        description="Renewable assets with non-zero capacity but zero annual available energy.",
        severity_if_hit=zero_profile_hit_severity,
        detail=zero_hits,
        capacity_col="capacity_mw",
        message="See renewable_zero_profile_assets.csv for full per-asset diagnostics.",
    )

    hydro_missing = _hydro_missing_inflow_diagnostics(n)
    add_check(
        check_id="hydro_missing_reservoir_inflow_columns",
        check_group="profiles",
        description="Hydro reservoir storage units with capacity but missing inflow columns.",
        severity_if_hit=hydro_missing_inflow_hit_severity,
        detail=hydro_missing,
        capacity_col="capacity_mw",
    )

    hydro_zero_ror = zero_hits.loc[zero_hits.get("carrier", pd.Series(dtype=object)).eq("ror")].copy()
    add_check(
        check_id="hydro_zero_ror_profiles",
        check_group="profiles",
        description="Run-of-river generators with capacity but zero available energy profile.",
        severity_if_hit=hydro_zero_ror_hit_severity,
        detail=hydro_zero_ror,
        capacity_col="capacity_mw",
    )

    hydro_zero_reservoir = zero_hits.loc[
        zero_hits.get("carrier", pd.Series(dtype=object)).eq("hydro_reservoir_inflow")
    ].copy()
    add_check(
        check_id="hydro_zero_reservoir_inflow_profiles",
        check_group="profiles",
        description="Hydro reservoir inflow series with capacity but zero annual inflow.",
        severity_if_hit=hydro_zero_reservoir_inflow_hit_severity,
        detail=hydro_zero_reservoir,
        capacity_col="capacity_mw",
    )

    nan_loads = _load_nan_diagnostics(n)
    add_check(
        check_id="nan_load_timeseries",
        check_group="demand",
        description="NaNs detected in AC load p_set time series.",
        severity_if_hit=nan_load_hit_severity,
        detail=nan_loads,
        hit_count_col="nan_cells",
        message="NaN load values can silently distort demand validation and solver feasibility.",
    )

    summary_df = pd.DataFrame(summary_rows).sort_values(["status_rank", "check_id"], ascending=[False, True])
    detail_df = pd.DataFrame(
        detail_rows,
        columns=[
            "check_id",
            "check_group",
            "status",
            "severity_if_hit",
            "component",
            "asset",
            "country",
            "carrier",
            "metric_name",
            "metric_value",
            "secondary_metric_name",
            "secondary_metric_value",
            "message",
        ],
    )
    overall_status = "pass"
    if not summary_df.empty:
        max_rank = int(summary_df["status_rank"].max())
        overall_status = {0: "pass", 1: "warn", 2: "fail"}.get(max_rank, "fail")
    status_obj = {
        "schema_version": "workflow_guardrails_v1",
        "overall_status": overall_status,
        "checks_total": int(len(summary_df)),
        "checks_fail": int((summary_df["status"] == "fail").sum()) if not summary_df.empty else 0,
        "checks_warn": int((summary_df["status"] == "warn").sum()) if not summary_df.empty else 0,
        "checks_pass": int((summary_df["status"] == "pass").sum()) if not summary_df.empty else 0,
        "detail_rows": int(len(detail_df)),
    }
    return summary_df, detail_df, status_obj


def _write_workflow_guardrail_artifacts(
    output_dir: Path,
    *,
    network_path: Path,
    year: int,
    summary_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    status_obj: dict[str, object],
) -> dict[str, object]:
    summary_path = output_dir / "workflow_guardrail_summary.csv"
    detail_path = output_dir / "workflow_guardrail_detail.csv"
    status_path = output_dir / "workflow_guardrail_status.json"

    _round_for_csv(summary_df).to_csv(summary_path, index=False)
    _round_for_csv(detail_df).to_csv(detail_path, index=False)

    enriched = dict(status_obj)
    enriched.update(
        {
            "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
            "network": str(network_path),
            "year": int(year),
            "artifacts": {
                "summary_csv": str(summary_path.name),
                "detail_csv": str(detail_path.name),
            },
        }
    )
    status_path.write_text(json.dumps(enriched, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return enriched


def _owid_electricity_balance(
    owid_csv: Path, year: int, iso3_filter: set[str] | None = None
) -> pd.DataFrame:
    usecols = ["year", "iso_code"] + OWID_ENERGY_COLUMNS + ["other_renewable_exc_biofuel_electricity"]
    owid = pd.read_csv(owid_csv, usecols=lambda c: c in set(usecols))
    owid = owid.loc[owid["year"] == year].copy()
    if iso3_filter is not None:
        owid = owid.loc[owid["iso_code"].isin(iso3_filter)].copy()
    owid["country"] = owid["iso_code"].apply(_safe_iso3_to_iso2)
    owid = owid.loc[owid.country.notna()].copy()

    for col in OWID_ENERGY_COLUMNS:
        if col in owid.columns:
            owid[col] = pd.to_numeric(owid[col], errors="coerce")

    # Prefer OWID's non-biofuel renewable split when available.
    if "other_renewable_exc_biofuel_electricity" in owid.columns:
        owid["other_renewable_electricity"] = owid["other_renewable_exc_biofuel_electricity"]

    keep_cols = [c for c in OWID_ENERGY_COLUMNS if c in owid.columns]
    owid = owid.groupby("country", as_index=False)[keep_cols].sum(min_count=1)
    out = owid.melt(
        id_vars="country",
        value_vars=keep_cols,
        var_name="metric",
        value_name="reference_twh",
    )
    return out


def _model_non_electric_fossil_balance(n: pypsa.Network) -> pd.DataFrame:
    bus_country = _bus_country_lookup(n)
    electric_fossil_links = _fossil_electric_link_carriers(n)

    w = n.statistics.withdrawal(
        aggregate_time="sum",
        aggregate_groups="sum",
        groupby=n.statistics.groupers.get_bus_and_carrier_and_bus_carrier,
        nice_names=False,
    )
    w = w.rename("energy_mwh").reset_index()
    w["country"] = w["bus"].map(bus_country).fillna("")

    w = w.loc[
        w["country"].ne("")
        & w["bus_carrier"].isin(FOSSIL_BUS_CARRIERS)
        & w["energy_mwh"].gt(0.0)
    ].copy()

    is_non_electric = (
        w["component"].eq("Load")
        | (w["component"].eq("Link") & ~w["carrier"].isin(electric_fossil_links))
    )
    non_electric = w.loc[is_non_electric].copy()
    non_electric["metric"] = non_electric["bus_carrier"].map(FOSSIL_BUS_CARRIER_TO_NON_ELEC_METRIC)
    non_electric = non_electric.loc[non_electric["metric"].notna()].copy()
    non_electric = (
        non_electric.groupby(["country", "metric"], as_index=False)["energy_mwh"]
        .sum()
        .rename(columns={"energy_mwh": "model_twh"})
    )
    non_electric["model_twh"] = non_electric["model_twh"] / 1e6
    return non_electric


def _owid_non_electric_fossil_balance(
    owid_csv: Path, year: int, iso3_filter: set[str] | None = None
) -> pd.DataFrame:
    usecols = ["year", "iso_code"] + OWID_NON_ELECTRIC_COLUMNS
    owid = pd.read_csv(owid_csv, usecols=lambda c: c in set(usecols))
    owid = owid.loc[owid["year"] == year].copy()
    if iso3_filter is not None:
        owid = owid.loc[owid["iso_code"].isin(iso3_filter)].copy()
    owid["country"] = owid["iso_code"].apply(_safe_iso3_to_iso2)
    owid = owid.loc[owid.country.notna()].copy()

    for col in OWID_NON_ELECTRIC_COLUMNS:
        if col in owid.columns:
            owid[col] = pd.to_numeric(owid[col], errors="coerce")

    agg_cols = [c for c in OWID_NON_ELECTRIC_COLUMNS if c in owid.columns]
    owid = owid.groupby("country", as_index=False)[agg_cols].sum(min_count=1)
    owid["oil_non_electric_consumption"] = (
        owid["oil_consumption"].fillna(0.0) - owid["oil_electricity"].fillna(0.0)
    ).clip(lower=0.0)
    owid["gas_non_electric_consumption"] = (
        owid["gas_consumption"].fillna(0.0) - owid["gas_electricity"].fillna(0.0)
    ).clip(lower=0.0)
    owid["coal_non_electric_consumption"] = (
        owid["coal_consumption"].fillna(0.0) - owid["coal_electricity"].fillna(0.0)
    ).clip(lower=0.0)

    keep_cols = [
        "oil_non_electric_consumption",
        "gas_non_electric_consumption",
        "coal_non_electric_consumption",
    ]
    out = owid.melt(
        id_vars="country",
        value_vars=keep_cols,
        var_name="metric",
        value_name="reference_twh",
    )
    return out


def _model_electricity_demand_breakdown(n: pypsa.Network) -> pd.DataFrame:
    bus_country = _bus_country_lookup(n)
    w = n.statistics.withdrawal(
        aggregate_time="sum",
        aggregate_groups="sum",
        groupby=n.statistics.groupers.get_bus_and_carrier_and_bus_carrier,
        nice_names=False,
    )
    w = w.rename("energy_mwh").reset_index()
    w["country"] = w["bus"].map(bus_country).fillna("")
    ac = w.loc[
        w["bus_carrier"].eq("AC")
        & w["country"].ne("")
        & w["energy_mwh"].gt(0.0)
    ].copy()
    if ac.empty:
        return pd.DataFrame(columns=["country", "metric", "model_twh"])

    by_country_component_twh = ac.groupby(["country", "component"], as_index=False)["energy_mwh"].sum()
    by_country_component_twh["model_twh"] = by_country_component_twh["energy_mwh"] / 1e6
    by_country_component_twh = by_country_component_twh.drop(columns=["energy_mwh"])

    pivot = (
        by_country_component_twh.pivot(index="country", columns="component", values="model_twh")
        .fillna(0.0)
        .rename_axis(None, axis=1)
    )
    for col in ["Load", "Link", "StorageUnit", "Store"]:
        if col not in pivot.columns:
            pivot[col] = 0.0

    out = pd.DataFrame(index=pivot.index)
    out["electricity_demand_load_component"] = pivot["Load"]
    out["electricity_demand_link_component"] = pivot["Link"]
    out["electricity_demand_storage_charging"] = pivot["StorageUnit"]
    out["electricity_demand_store_component"] = pivot["Store"]
    out["electricity_demand_total_ac_withdrawal"] = (
        out["electricity_demand_load_component"]
        + out["electricity_demand_link_component"]
        + out["electricity_demand_storage_charging"]
        + out["electricity_demand_store_component"]
    )
    out["electricity_demand"] = (
        out["electricity_demand_load_component"] + out["electricity_demand_link_component"]
    )
    out = out.reset_index().rename(columns={"index": "country"})
    out = out.melt(id_vars="country", var_name="metric", value_name="model_twh")
    return out


def _model_electricity_demand(n: pypsa.Network) -> pd.DataFrame:
    breakdown = _model_electricity_demand_breakdown(n)
    if breakdown.empty:
        return pd.DataFrame(columns=["country", "metric", "model_twh"])
    return breakdown.loc[breakdown["metric"] == OWID_ELECTRICITY_DEMAND_COLUMN].copy()


def _owid_electricity_demand(
    owid_csv: Path, year: int, iso3_filter: set[str] | None = None
) -> pd.DataFrame:
    usecols = ["year", "iso_code", OWID_ELECTRICITY_DEMAND_COLUMN]
    owid = pd.read_csv(owid_csv, usecols=lambda c: c in set(usecols))
    if OWID_ELECTRICITY_DEMAND_COLUMN not in owid.columns:
        LOGGER.warning(
            "OWID column '%s' not found in %s; skipping electricity demand validation",
            OWID_ELECTRICITY_DEMAND_COLUMN,
            owid_csv,
        )
        return pd.DataFrame(columns=["country", "metric", "reference_twh"])

    owid = owid.loc[owid["year"] == year].copy()
    if iso3_filter is not None:
        owid = owid.loc[owid["iso_code"].isin(iso3_filter)].copy()

    owid["country"] = owid["iso_code"].apply(_safe_iso3_to_iso2)
    owid = owid.loc[owid.country.notna()].copy()
    owid[OWID_ELECTRICITY_DEMAND_COLUMN] = pd.to_numeric(
        owid[OWID_ELECTRICITY_DEMAND_COLUMN], errors="coerce"
    ).fillna(0.0)

    grouped = (
        owid.groupby("country", as_index=False)[OWID_ELECTRICITY_DEMAND_COLUMN]
        .sum(min_count=1)
        .rename(columns={OWID_ELECTRICITY_DEMAND_COLUMN: "reference_twh"})
    )
    grouped["metric"] = OWID_ELECTRICITY_DEMAND_COLUMN
    return grouped[["country", "metric", "reference_twh"]]


def _energy_comparison(model_energy: pd.DataFrame, ref_energy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cmp_df = model_energy.merge(ref_energy, on=["country", "metric"], how="outer")
    cmp_df["model_twh"] = cmp_df["model_twh"].fillna(0.0)
    cmp_df["reference_twh"] = cmp_df["reference_twh"].fillna(0.0)
    cmp_df["error_twh"] = cmp_df["model_twh"] - cmp_df["reference_twh"]
    cmp_df["abs_error_twh"] = cmp_df["error_twh"].abs()
    cmp_df["ape_pct"] = np.where(
        cmp_df["reference_twh"] > 0,
        cmp_df["abs_error_twh"] / cmp_df["reference_twh"] * 100.0,
        np.nan,
    )

    summary = cmp_df.groupby("metric", as_index=False).agg(
        countries=("country", "nunique"),
        model_total_twh=("model_twh", "sum"),
        reference_total_twh=("reference_twh", "sum"),
        mae_twh=("abs_error_twh", "mean"),
        rmse_twh=("error_twh", lambda s: float(np.sqrt(np.mean(np.square(s))))),
        mean_ape_pct=("ape_pct", "mean"),
        wape_pct=(
            "abs_error_twh",
            lambda s: (
                float(s.sum())
                / max(
                    cmp_df.loc[s.index, "reference_twh"].sum(),
                    1e-9,
                )
                * 100.0
            ),
        ),
    )

    return cmp_df, summary.sort_values("wape_pct", ascending=False)


def _capacity_comparison_rollups(cap_cmp: pd.DataFrame) -> pd.DataFrame:
    cont_lookup = _country_to_continent_lookup()

    country = cap_cmp[["country", "validation_tech", "capacity_mw", "reference_mw"]].copy()
    country["geo_level"] = "country"
    country["geography"] = country["country"]

    continent = cap_cmp[["country", "validation_tech", "capacity_mw", "reference_mw"]].copy()
    continent["geo_level"] = "continent"
    continent["geography"] = continent["country"].map(cont_lookup).fillna("Unknown")
    continent = (
        continent.groupby(["geo_level", "geography", "validation_tech"], as_index=False)[
            ["capacity_mw", "reference_mw"]
        ].sum()
    )

    global_df = (
        cap_cmp.groupby("validation_tech", as_index=False)[["capacity_mw", "reference_mw"]]
        .sum()
        .assign(geo_level="global", geography="Global")
    )

    roll = pd.concat(
        [
            country[["geo_level", "geography", "validation_tech", "capacity_mw", "reference_mw"]],
            continent[["geo_level", "geography", "validation_tech", "capacity_mw", "reference_mw"]],
            global_df[["geo_level", "geography", "validation_tech", "capacity_mw", "reference_mw"]],
        ],
        ignore_index=True,
    )
    roll["error_mw"] = roll["capacity_mw"] - roll["reference_mw"]
    roll["abs_error_mw"] = roll["error_mw"].abs()
    roll["ape_pct"] = np.where(
        roll["reference_mw"] > 0,
        roll["abs_error_mw"] / roll["reference_mw"] * 100.0,
        np.nan,
    )
    roll["divergence_pct"] = _relative_divergence_pct(roll["capacity_mw"], roll["reference_mw"])
    return roll


def _energy_comparison_rollups(energy_cmp: pd.DataFrame) -> pd.DataFrame:
    cont_lookup = _country_to_continent_lookup()

    country = energy_cmp[["country", "metric", "model_twh", "reference_twh"]].copy()
    country["geo_level"] = "country"
    country["geography"] = country["country"]

    continent = energy_cmp[["country", "metric", "model_twh", "reference_twh"]].copy()
    continent["geo_level"] = "continent"
    continent["geography"] = continent["country"].map(cont_lookup).fillna("Unknown")
    continent = (
        continent.groupby(["geo_level", "geography", "metric"], as_index=False)[
            ["model_twh", "reference_twh"]
        ].sum()
    )

    global_df = (
        energy_cmp.groupby("metric", as_index=False)[["model_twh", "reference_twh"]]
        .sum()
        .assign(geo_level="global", geography="Global")
    )

    roll = pd.concat(
        [
            country[["geo_level", "geography", "metric", "model_twh", "reference_twh"]],
            continent[["geo_level", "geography", "metric", "model_twh", "reference_twh"]],
            global_df[["geo_level", "geography", "metric", "model_twh", "reference_twh"]],
        ],
        ignore_index=True,
    )
    roll["error_twh"] = roll["model_twh"] - roll["reference_twh"]
    roll["abs_error_twh"] = roll["error_twh"].abs()
    roll["ape_pct"] = np.where(
        roll["reference_twh"] > 0,
        roll["abs_error_twh"] / roll["reference_twh"] * 100.0,
        np.nan,
    )
    roll["divergence_pct"] = _relative_divergence_pct(roll["model_twh"], roll["reference_twh"])
    return roll


def _coverage_by_group(
    cmp_df: pd.DataFrame,
    *,
    group_col: str,
    model_col: str,
    reference_col: str,
) -> pd.DataFrame:
    rows = []
    for key, group in cmp_df.groupby(group_col):
        model_pos = set(group.loc[group[model_col] > 0, "country"])
        ref_pos = set(group.loc[group[reference_col] > 0, "country"])
        overlap = model_pos & ref_pos
        rows.append(
            {
                group_col: key,
                "model_positive_countries": len(model_pos),
                "reference_positive_countries": len(ref_pos),
                "overlap_positive_countries": len(overlap),
                "model_only_positive_countries": len(model_pos - ref_pos),
                "reference_only_positive_countries": len(ref_pos - model_pos),
                "coverage_vs_reference_pct": (
                    len(overlap) / max(len(ref_pos), 1) * 100.0
                ),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["coverage_vs_reference_pct", "reference_positive_countries"],
        ascending=[True, False],
    )


def _coverage_by_country(
    cmp_df: pd.DataFrame,
    *,
    group_col: str,
    model_col: str,
    reference_col: str,
) -> pd.DataFrame:
    rows = []
    for country, group in cmp_df.groupby("country"):
        model_pos = set(group.loc[group[model_col] > 0, group_col])
        ref_pos = set(group.loc[group[reference_col] > 0, group_col])
        overlap = model_pos & ref_pos
        rows.append(
            {
                "country": country,
                "model_positive_groups": len(model_pos),
                "reference_positive_groups": len(ref_pos),
                "overlap_positive_groups": len(overlap),
                "model_only_positive_groups": len(model_pos - ref_pos),
                "reference_only_positive_groups": len(ref_pos - model_pos),
                "coverage_vs_reference_pct": (
                    len(overlap) / max(len(ref_pos), 1) * 100.0
                ),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["coverage_vs_reference_pct", "reference_positive_groups"],
        ascending=[True, False],
    )


def _relative_divergence_pct(model: pd.Series, reference: pd.Series) -> pd.Series:
    denom = np.maximum(
        np.maximum(model.abs().astype(float), reference.abs().astype(float)),
        1e-9,
    )
    return (model.astype(float) - reference.astype(float)).abs() / denom * 100.0


def _flag_summary_by_group(
    cmp_df: pd.DataFrame,
    *,
    group_col: str,
    model_col: str,
    reference_col: str,
    abs_error_col: str,
    divergence_col: str,
    flag_col: str,
    ape_col: str,
) -> pd.DataFrame:
    rows = []
    for group_value, group in cmp_df.groupby(group_col):
        reference_total = float(group[reference_col].sum())
        rows.append(
            {
                group_col: group_value,
                "entries": int(len(group)),
                "flagged_entries": int(group[flag_col].sum()),
                "flag_rate_pct": float(group[flag_col].mean() * 100.0) if len(group) else 0.0,
                "model_total": float(group[model_col].sum()),
                "reference_total": reference_total,
                "wape_pct": (
                    float(group[abs_error_col].sum()) / max(reference_total, 1e-9) * 100.0
                ),
                "median_divergence_pct": float(group[divergence_col].median()),
                "p90_divergence_pct": float(group[divergence_col].quantile(0.90)),
                "mean_ape_pct": float(group[ape_col].mean(skipna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values(["flag_rate_pct", "wape_pct"], ascending=False)


def _round_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Round exported CSV values for readability.

    Rules:
    - TWh/MW/GW and error magnitude columns: 1 decimal
    - Percentage/rate columns: 2 decimals
    - Integer count columns are left untouched
    """
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in {"entries", "countries", "flagged_entries"}:
            continue
        lower = col.lower()
        if "pct" in lower or "rate" in lower:
            out[col] = out[col].round(2)
            continue
        if any(
            token in lower
            for token in [
                "_twh",
                "_mw",
                "_gw",
                "model_total",
                "reference_total",
                "mae",
                "rmse",
                "error",
                "divergence",
            ]
        ):
            out[col] = out[col].round(1)
            continue
        out[col] = out[col].round(3)
    return out


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _pct_or_nan(numerator: int | float, denominator: int | float) -> float:
    if denominator is None or float(denominator) <= 0:
        return np.nan
    return float(numerator) / float(denominator) * 100.0


def _wape_or_nan(abs_error_total: float, reference_total: float) -> float:
    if reference_total <= 0:
        return np.nan
    return float(abs_error_total) / float(reference_total) * 100.0


def _load_validation_gate_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Validation gate config not found: {path}")

    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        cfg = yaml.safe_load(text)
    else:
        try:
            cfg = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "PyYAML is not installed and validation gate config is not JSON-compatible YAML."
            ) from exc

    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Validation gate config must be a mapping at top level: {path}")

    metric_rules = cfg.get("metric_rules", [])
    guardrail_rules = cfg.get("guardrail_rules", [])
    if not isinstance(metric_rules, list):
        raise ValueError("'metric_rules' must be a list in validation gate config")
    if not isinstance(guardrail_rules, list):
        raise ValueError("'guardrail_rules' must be a list in validation gate config")

    cfg["metric_rules"] = metric_rules
    cfg["guardrail_rules"] = guardrail_rules

    merged_defs = dict(DEMAND_METRIC_DEFINITIONS)
    user_defs = cfg.get("demand_metric_definitions", {})
    if isinstance(user_defs, dict):
        merged_defs.update({str(k): str(v) for k, v in user_defs.items()})
    cfg["demand_metric_definitions"] = merged_defs
    return cfg


def _demand_metric_definitions_table(definitions: dict[str, str]) -> pd.DataFrame:
    preferred_order = [
        "electricity_demand",
        "electricity_demand_total_ac_withdrawal",
        "electricity_demand_load_component",
        "electricity_demand_link_component",
        "electricity_demand_storage_charging",
        "electricity_demand_store_component",
    ]
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    for metric in preferred_order:
        if metric in definitions:
            rows.append({"metric": metric, "definition": str(definitions[metric])})
            seen.add(metric)
    for metric in sorted(definitions):
        if metric in seen:
            continue
        rows.append({"metric": str(metric), "definition": str(definitions[metric])})
    return pd.DataFrame(rows, columns=["metric", "definition"])


def _evaluate_metric_gate_rule(
    rule: dict[str, Any],
    *,
    cmp_frames_by_domain: dict[str, pd.DataFrame],
    demand_metric_definitions: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gate_item = str(rule.get("metric", ""))
    domain = str(rule.get("domain", "")).strip()
    gate_class = str(rule.get("gate_class", "secondary")).strip().lower() or "secondary"
    required_for_overall = _coerce_bool(
        rule.get("required_for_overall_gate"),
        default=(gate_class == "blocking"),
    )
    enabled = _coerce_bool(rule.get("enabled"), default=True)

    row: dict[str, Any] = {
        "row_type": "metric",
        "gate_item": gate_item,
        "domain": domain,
        "gate_class": gate_class,
        "required_for_overall_gate": required_for_overall,
        "evaluation_status": "not_evaluated",
        "status": "not_evaluated",
        "pass": None,
        "metric_definition": demand_metric_definitions.get(gate_item, "") if domain == "electricity_demand" else "",
        "notes": str(rule.get("notes", "") or ""),
        "unit": "",
        "model_total": np.nan,
        "reference_total": np.nan,
        "global_wape_pct_actual": np.nan,
        "global_wape_pct_threshold": _coerce_optional_float(rule.get("global_wape_threshold_pct")),
        "global_wape_pass": None,
        "country_ape_threshold_pct": _coerce_optional_float(rule.get("country_ape_threshold_pct")),
        "country_pass_rate_pct_threshold": _coerce_optional_float(
            rule.get("country_pass_rate_threshold_pct")
        ),
        "country_pass_rate_required": _coerce_bool(rule.get("country_pass_rate_required"), default=False),
        "country_pass_rate_all_pct": np.nan,
        "country_pass_rate_all_num": np.nan,
        "country_pass_rate_all_den": np.nan,
        "country_pass_rate_material_pct": np.nan,
        "country_pass_rate_material_num": np.nan,
        "country_pass_rate_material_den": np.nan,
        "country_pass_rate_pass": None,
        "materiality_threshold": _coerce_optional_float(rule.get("materiality_threshold")),
        "materiality_unit": str(rule.get("materiality_unit", "") or ""),
        "metric_gate_pass": None,
        "actual_total_zero_profile_assets": np.nan,
        "actual_total_zero_profile_capacity_mw": np.nan,
        "warn_if_total_zero_profile_assets_gt": np.nan,
        "fail_if_total_zero_profile_assets_gt": np.nan,
        "warn_if_total_zero_profile_capacity_mw_gt": np.nan,
        "fail_if_total_zero_profile_capacity_mw_gt": np.nan,
    }
    country_rows: list[dict[str, Any]] = []

    if not enabled:
        row["notes"] = (row["notes"] + " " if row["notes"] else "") + "Rule disabled in config."
        return row, country_rows

    if not gate_item:
        row["notes"] = (row["notes"] + " " if row["notes"] else "") + "Missing 'metric' in config rule."
        return row, country_rows

    spec = VALIDATION_GATE_DOMAIN_SPECS.get(domain)
    if spec is None:
        row["notes"] = (
            (row["notes"] + " " if row["notes"] else "")
            + f"Unsupported domain '{domain}'."
        )
        return row, country_rows

    cmp_df = cmp_frames_by_domain.get(domain)
    if cmp_df is None:
        row["notes"] = (row["notes"] + " " if row["notes"] else "") + "No comparison table available."
        return row, country_rows

    metric_col = str(spec["metric_col"])
    model_col = str(spec["model_col"])
    reference_col = str(spec["reference_col"])
    abs_error_col = str(spec["abs_error_col"])
    ape_col = str(spec["ape_col"])
    unit = str(spec["unit"])
    row["unit"] = unit
    if not row["materiality_unit"]:
        row["materiality_unit"] = unit

    if metric_col not in cmp_df.columns:
        row["notes"] = (
            (row["notes"] + " " if row["notes"] else "")
            + f"Comparison table for domain '{domain}' is missing '{metric_col}'."
        )
        return row, country_rows

    subset = cmp_df.loc[cmp_df[metric_col].astype(str) == gate_item].copy()
    if subset.empty:
        row["notes"] = (row["notes"] + " " if row["notes"] else "") + "Metric not present in comparison output."
        return row, country_rows

    model_total = float(subset[model_col].sum())
    reference_total = float(subset[reference_col].sum())
    abs_error_total = float(subset[abs_error_col].sum())
    wape_actual = _wape_or_nan(abs_error_total, reference_total)
    row["model_total"] = model_total
    row["reference_total"] = reference_total
    row["global_wape_pct_actual"] = wape_actual

    global_wape_threshold = row["global_wape_pct_threshold"]
    if global_wape_threshold is not None and np.isfinite(wape_actual):
        row["global_wape_pass"] = bool(wape_actual <= global_wape_threshold)

    ape_threshold = row["country_ape_threshold_pct"]
    materiality_threshold = row["materiality_threshold"]
    if materiality_threshold is not None and materiality_threshold < 0:
        materiality_threshold = 0.0
        row["materiality_threshold"] = materiality_threshold

    eligible_all = subset.loc[pd.to_numeric(subset[reference_col], errors="coerce").fillna(0.0) > 0].copy()
    if materiality_threshold is None:
        eligible_material = eligible_all.copy()
    else:
        eligible_material = eligible_all.loc[eligible_all[reference_col] >= materiality_threshold].copy()

    all_den = int(len(eligible_all))
    material_den = int(len(eligible_material))
    row["country_pass_rate_all_den"] = all_den
    row["country_pass_rate_material_den"] = material_den

    if ape_threshold is not None and all_den > 0:
        all_pass_mask = pd.to_numeric(eligible_all[ape_col], errors="coerce") <= float(ape_threshold)
        all_num = int(all_pass_mask.fillna(False).sum())
        row["country_pass_rate_all_num"] = all_num
        row["country_pass_rate_all_pct"] = _pct_or_nan(all_num, all_den)
    if ape_threshold is not None and material_den > 0:
        material_pass_mask = pd.to_numeric(eligible_material[ape_col], errors="coerce") <= float(ape_threshold)
        material_num = int(material_pass_mask.fillna(False).sum())
        row["country_pass_rate_material_num"] = material_num
        row["country_pass_rate_material_pct"] = _pct_or_nan(material_num, material_den)

    country_pass_required = bool(row["country_pass_rate_required"])
    country_pass_rate_threshold = row["country_pass_rate_pct_threshold"]
    if country_pass_required:
        if country_pass_rate_threshold is None:
            row["notes"] = (
                (row["notes"] + " " if row["notes"] else "")
                + "Country pass rate is required but threshold is missing."
            )
        elif ape_threshold is None:
            row["notes"] = (
                (row["notes"] + " " if row["notes"] else "")
                + "Country pass rate is required but country APE threshold is missing."
            )
        elif material_den <= 0:
            row["notes"] = (
                (row["notes"] + " " if row["notes"] else "")
                + "Country pass rate is required but no material reference countries are available."
            )
        elif np.isfinite(float(row["country_pass_rate_material_pct"])):
            row["country_pass_rate_pass"] = bool(
                float(row["country_pass_rate_material_pct"]) >= float(country_pass_rate_threshold)
            )

    required_check_results: list[bool | None] = []
    if global_wape_threshold is not None:
        required_check_results.append(
            bool(row["global_wape_pass"]) if row["global_wape_pass"] is not None else None
        )
    if country_pass_required:
        required_check_results.append(
            bool(row["country_pass_rate_pass"]) if row["country_pass_rate_pass"] is not None else None
        )

    if required_check_results:
        if any(result is False for result in required_check_results):
            row["metric_gate_pass"] = False
        elif any(result is None for result in required_check_results):
            row["metric_gate_pass"] = None
        else:
            row["metric_gate_pass"] = True

    if row["metric_gate_pass"] is True:
        row["evaluation_status"] = "evaluated"
        row["status"] = "pass"
        row["pass"] = True
    elif row["metric_gate_pass"] is False:
        row["evaluation_status"] = "evaluated"
        row["status"] = "fail"
        row["pass"] = False
    elif required_check_results:
        row["evaluation_status"] = "partial"
        row["status"] = "partial"
        row["pass"] = None
    else:
        row["evaluation_status"] = "partial"
        row["status"] = "info"
        row["pass"] = None

    ref_values = pd.to_numeric(subset[reference_col], errors="coerce").fillna(0.0)
    ape_values = pd.to_numeric(subset[ape_col], errors="coerce")
    for idx, r in subset.iterrows():
        ref_val = float(ref_values.loc[idx])
        ape_val = ape_values.loc[idx]
        is_reference_positive = bool(ref_val > 0.0)
        is_material = bool(is_reference_positive and (materiality_threshold is None or ref_val >= materiality_threshold))
        counts_all = bool(is_reference_positive and ape_threshold is not None and pd.notna(ape_val))
        counts_material = bool(counts_all and is_material)
        passes_ape = None
        if counts_all:
            passes_ape = bool(float(ape_val) <= float(ape_threshold))
        country_rows.append(
            {
                "domain": domain,
                "gate_class": gate_class,
                "required_for_overall_gate": required_for_overall,
                "metric": gate_item,
                "country": str(r.get("country", "")),
                "unit": unit,
                "model_value": float(r.get(model_col, 0.0)),
                "reference_value": ref_val,
                "abs_error_value": float(r.get(abs_error_col, 0.0)),
                "ape_pct": (float(ape_val) if pd.notna(ape_val) else np.nan),
                "country_ape_threshold_pct": ape_threshold if ape_threshold is not None else np.nan,
                "passes_country_ape_threshold": passes_ape,
                "is_reference_positive": is_reference_positive,
                "is_material_country": is_material,
                "materiality_threshold": (
                    materiality_threshold if materiality_threshold is not None else np.nan
                ),
                "materiality_unit": row["materiality_unit"],
                "counts_toward_all_country_pass_rate": counts_all,
                "counts_toward_material_country_pass_rate": counts_material,
                "metric_definition": row["metric_definition"],
            }
        )

    country_rows.sort(
        key=lambda rec: (
            str(rec.get("metric", "")),
            -float(rec.get("abs_error_value", 0.0)),
            str(rec.get("country", "")),
        )
    )
    return row, country_rows


def _evaluate_guardrail_rule(
    rule: dict[str, Any],
    *,
    zero_profile_summary: pd.DataFrame,
) -> dict[str, Any]:
    guardrail = str(rule.get("guardrail", ""))
    gate_class = str(rule.get("gate_class", "guardrail")).strip().lower() or "guardrail"
    required_for_overall = _coerce_bool(rule.get("required_for_overall_gate"), default=False)
    enabled = _coerce_bool(rule.get("enabled"), default=True)

    row: dict[str, Any] = {
        "row_type": "guardrail",
        "gate_item": guardrail,
        "domain": "guardrail",
        "gate_class": gate_class,
        "required_for_overall_gate": required_for_overall,
        "evaluation_status": "not_evaluated",
        "status": "not_evaluated",
        "pass": None,
        "metric_definition": "",
        "notes": str(rule.get("notes", "") or ""),
        "unit": "",
        "model_total": np.nan,
        "reference_total": np.nan,
        "global_wape_pct_actual": np.nan,
        "global_wape_pct_threshold": np.nan,
        "global_wape_pass": None,
        "country_ape_threshold_pct": np.nan,
        "country_pass_rate_pct_threshold": np.nan,
        "country_pass_rate_required": False,
        "country_pass_rate_all_pct": np.nan,
        "country_pass_rate_all_num": np.nan,
        "country_pass_rate_all_den": np.nan,
        "country_pass_rate_material_pct": np.nan,
        "country_pass_rate_material_num": np.nan,
        "country_pass_rate_material_den": np.nan,
        "country_pass_rate_pass": None,
        "materiality_threshold": np.nan,
        "materiality_unit": "",
        "metric_gate_pass": None,
        "actual_total_zero_profile_assets": np.nan,
        "actual_total_zero_profile_capacity_mw": np.nan,
        "warn_if_total_zero_profile_assets_gt": _coerce_optional_float(
            rule.get("warn_if_total_zero_profile_assets_gt")
        ),
        "fail_if_total_zero_profile_assets_gt": _coerce_optional_float(
            rule.get("fail_if_total_zero_profile_assets_gt")
        ),
        "warn_if_total_zero_profile_capacity_mw_gt": _coerce_optional_float(
            rule.get("warn_if_total_zero_profile_capacity_mw_gt")
        ),
        "fail_if_total_zero_profile_capacity_mw_gt": _coerce_optional_float(
            rule.get("fail_if_total_zero_profile_capacity_mw_gt")
        ),
    }

    if not enabled:
        row["notes"] = (row["notes"] + " " if row["notes"] else "") + "Rule disabled in config."
        return row
    if guardrail != "renewable_zero_profile_assets":
        row["notes"] = (
            (row["notes"] + " " if row["notes"] else "")
            + f"Unsupported guardrail '{guardrail}' (WS1 implementation)."
        )
        return row

    total_zero_assets = 0.0
    total_zero_cap = 0.0
    if not zero_profile_summary.empty:
        if "zero_profile_assets" in zero_profile_summary.columns:
            total_zero_assets = float(
                pd.to_numeric(zero_profile_summary["zero_profile_assets"], errors="coerce")
                .fillna(0.0)
                .sum()
            )
        if "zero_profile_capacity_mw" in zero_profile_summary.columns:
            total_zero_cap = float(
                pd.to_numeric(zero_profile_summary["zero_profile_capacity_mw"], errors="coerce")
                .fillna(0.0)
                .sum()
            )

    row["actual_total_zero_profile_assets"] = total_zero_assets
    row["actual_total_zero_profile_capacity_mw"] = total_zero_cap
    row["evaluation_status"] = "evaluated"

    fail_hit = False
    warn_hit = False
    fail_assets_gt = row["fail_if_total_zero_profile_assets_gt"]
    fail_cap_gt = row["fail_if_total_zero_profile_capacity_mw_gt"]
    warn_assets_gt = row["warn_if_total_zero_profile_assets_gt"]
    warn_cap_gt = row["warn_if_total_zero_profile_capacity_mw_gt"]

    thresholds_defined = any(
        threshold is not None
        for threshold in [fail_assets_gt, fail_cap_gt, warn_assets_gt, warn_cap_gt]
    )
    if fail_assets_gt is not None and total_zero_assets > float(fail_assets_gt):
        fail_hit = True
    if fail_cap_gt is not None and total_zero_cap > float(fail_cap_gt):
        fail_hit = True
    if warn_assets_gt is not None and total_zero_assets > float(warn_assets_gt):
        warn_hit = True
    if warn_cap_gt is not None and total_zero_cap > float(warn_cap_gt):
        warn_hit = True

    if not thresholds_defined:
        row["status"] = "info"
        row["pass"] = None
        row["notes"] = (row["notes"] + " " if row["notes"] else "") + "No thresholds configured."
    elif fail_hit:
        row["status"] = "fail"
        row["pass"] = False
    elif warn_hit:
        row["status"] = "warn"
        row["pass"] = False
    else:
        row["status"] = "pass"
        row["pass"] = True

    return row


def _rollup_metric_group_status(metric_rows: pd.DataFrame) -> str:
    if metric_rows.empty:
        return "not_applicable"
    if metric_rows["status"].astype(str).eq("fail").any():
        return "fail"
    if metric_rows["evaluation_status"].astype(str).isin(["partial", "not_evaluated"]).any():
        return "incomplete"
    return "pass"


def _rollup_guardrail_status(guardrail_rows: pd.DataFrame) -> str:
    if guardrail_rows.empty:
        return "not_applicable"
    if guardrail_rows["status"].astype(str).eq("fail").any():
        return "fail"
    if guardrail_rows["evaluation_status"].astype(str).isin(["partial", "not_evaluated"]).any():
        return "incomplete"
    if guardrail_rows["status"].astype(str).eq("warn").any():
        return "warn"
    return "pass"


def _required_guardrails_gate_status(guardrail_rows: pd.DataFrame) -> str:
    required = guardrail_rows.loc[guardrail_rows["required_for_overall_gate"].fillna(False)].copy()
    if required.empty:
        return "not_applicable"
    if required["status"].astype(str).eq("fail").any():
        return "fail"
    if required["evaluation_status"].astype(str).isin(["partial", "not_evaluated"]).any():
        return "incomplete"
    if required["status"].astype(str).eq("warn").any():
        return "warn"
    return "pass"


def _evaluate_validation_gate(
    *,
    gate_config: dict[str, Any],
    cap_cmp: pd.DataFrame,
    elec_cmp: pd.DataFrame,
    demand_cmp: pd.DataFrame,
    non_elec_cmp: pd.DataFrame,
    zero_profile_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cmp_frames_by_domain = {
        "capacity": cap_cmp,
        "electricity_balance": elec_cmp,
        "electricity_demand": demand_cmp,
        "fossil_non_electric": non_elec_cmp,
    }
    demand_defs = dict(gate_config.get("demand_metric_definitions", DEMAND_METRIC_DEFINITIONS))

    summary_rows: list[dict[str, Any]] = []
    country_rows: list[dict[str, Any]] = []

    for raw_rule in gate_config.get("metric_rules", []):
        if not isinstance(raw_rule, dict):
            continue
        metric_row, metric_country_rows = _evaluate_metric_gate_rule(
            raw_rule,
            cmp_frames_by_domain=cmp_frames_by_domain,
            demand_metric_definitions=demand_defs,
        )
        summary_rows.append(metric_row)
        country_rows.extend(metric_country_rows)

    for raw_rule in gate_config.get("guardrail_rules", []):
        if not isinstance(raw_rule, dict):
            continue
        summary_rows.append(
            _evaluate_guardrail_rule(
                raw_rule,
                zero_profile_summary=zero_profile_summary,
            )
        )

    summary_df = pd.DataFrame(summary_rows)
    country_df = pd.DataFrame(country_rows)

    if summary_df.empty:
        summary_df = pd.DataFrame(
            columns=[
                "row_type",
                "gate_item",
                "domain",
                "gate_class",
                "required_for_overall_gate",
                "evaluation_status",
                "status",
                "pass",
            ]
        )

    if country_df.empty:
        country_df = pd.DataFrame(
            columns=[
                "domain",
                "gate_class",
                "required_for_overall_gate",
                "metric",
                "country",
                "unit",
                "model_value",
                "reference_value",
                "abs_error_value",
                "ape_pct",
                "country_ape_threshold_pct",
                "passes_country_ape_threshold",
                "is_reference_positive",
                "is_material_country",
                "materiality_threshold",
                "materiality_unit",
                "counts_toward_all_country_pass_rate",
                "counts_toward_material_country_pass_rate",
                "metric_definition",
            ]
        )

    metric_rows = summary_df.loc[summary_df["row_type"].astype(str) == "metric"].copy()
    blocking_metric_rows = metric_rows.loc[metric_rows["required_for_overall_gate"].fillna(False)].copy()
    secondary_metric_rows = metric_rows.loc[~metric_rows["required_for_overall_gate"].fillna(False)].copy()
    guardrail_rows = summary_df.loc[summary_df["row_type"].astype(str) == "guardrail"].copy()

    blocking_metrics_status = _rollup_metric_group_status(blocking_metric_rows)
    secondary_metrics_status = _rollup_metric_group_status(secondary_metric_rows)
    guardrail_reporting_status = _rollup_guardrail_status(guardrail_rows)
    required_guardrails_status = _required_guardrails_gate_status(guardrail_rows)

    overall_gate_status = "pass"
    if blocking_metrics_status in {"fail", "incomplete"}:
        overall_gate_status = blocking_metrics_status
    elif required_guardrails_status in {"fail", "incomplete"}:
        overall_gate_status = required_guardrails_status

    overall = {
        "row_type": "overall",
        "gate_item": "overall_validation_gate",
        "domain": "overall",
        "gate_class": "overall",
        "required_for_overall_gate": True,
        "evaluation_status": "evaluated",
        "status": overall_gate_status,
        "pass": (overall_gate_status == "pass"),
        "metric_definition": "",
        "notes": "",
        "unit": "",
        "model_total": np.nan,
        "reference_total": np.nan,
        "global_wape_pct_actual": np.nan,
        "global_wape_pct_threshold": np.nan,
        "global_wape_pass": None,
        "country_ape_threshold_pct": np.nan,
        "country_pass_rate_pct_threshold": np.nan,
        "country_pass_rate_required": False,
        "country_pass_rate_all_pct": np.nan,
        "country_pass_rate_all_num": np.nan,
        "country_pass_rate_all_den": np.nan,
        "country_pass_rate_material_pct": np.nan,
        "country_pass_rate_material_num": np.nan,
        "country_pass_rate_material_den": np.nan,
        "country_pass_rate_pass": None,
        "materiality_threshold": np.nan,
        "materiality_unit": "",
        "metric_gate_pass": None,
        "actual_total_zero_profile_assets": np.nan,
        "actual_total_zero_profile_capacity_mw": np.nan,
        "warn_if_total_zero_profile_assets_gt": np.nan,
        "fail_if_total_zero_profile_assets_gt": np.nan,
        "warn_if_total_zero_profile_capacity_mw_gt": np.nan,
        "fail_if_total_zero_profile_capacity_mw_gt": np.nan,
        "overall_blocking_metrics_status": blocking_metrics_status,
        "overall_secondary_metrics_status": secondary_metrics_status,
        "overall_guardrail_reporting_status": guardrail_reporting_status,
        "overall_required_guardrails_status": required_guardrails_status,
        "overall_gate_status": overall_gate_status,
        "configured_metric_rules": int(len([r for r in gate_config.get("metric_rules", []) if isinstance(r, dict)])),
        "configured_guardrail_rules": int(
            len([r for r in gate_config.get("guardrail_rules", []) if isinstance(r, dict)])
        ),
        "blocking_metric_count": int(len(blocking_metric_rows)),
        "blocking_metric_fail_count": int(blocking_metric_rows["status"].astype(str).eq("fail").sum())
        if not blocking_metric_rows.empty
        else 0,
        "blocking_metric_incomplete_count": int(
            blocking_metric_rows["evaluation_status"].astype(str).isin(["partial", "not_evaluated"]).sum()
        )
        if not blocking_metric_rows.empty
        else 0,
        "secondary_metric_count": int(len(secondary_metric_rows)),
        "secondary_metric_fail_count": int(secondary_metric_rows["status"].astype(str).eq("fail").sum())
        if not secondary_metric_rows.empty
        else 0,
        "guardrail_count": int(len(guardrail_rows)),
        "guardrail_fail_count": int(guardrail_rows["status"].astype(str).eq("fail").sum())
        if not guardrail_rows.empty
        else 0,
        "guardrail_warn_count": int(guardrail_rows["status"].astype(str).eq("warn").sum())
        if not guardrail_rows.empty
        else 0,
        "guardrail_incomplete_count": int(
            guardrail_rows["evaluation_status"].astype(str).isin(["partial", "not_evaluated"]).sum()
        )
        if not guardrail_rows.empty
        else 0,
    }

    for field in [
        "overall_blocking_metrics_status",
        "overall_secondary_metrics_status",
        "overall_guardrail_reporting_status",
        "overall_required_guardrails_status",
        "overall_gate_status",
    ]:
        summary_df[field] = overall[field]

    summary_df = pd.concat([summary_df, pd.DataFrame([overall])], ignore_index=True, sort=False)

    if not country_df.empty:
        country_df = country_df.sort_values(["domain", "metric", "country"]).reset_index(drop=True)

    metric_pass_rate_cols = [
        "gate_item",
        "domain",
        "gate_class",
        "required_for_overall_gate",
        "status",
        "evaluation_status",
        "unit",
        "global_wape_pct_actual",
        "global_wape_pct_threshold",
        "global_wape_pass",
        "country_ape_threshold_pct",
        "country_pass_rate_pct_threshold",
        "country_pass_rate_required",
        "country_pass_rate_all_pct",
        "country_pass_rate_all_num",
        "country_pass_rate_all_den",
        "country_pass_rate_material_pct",
        "country_pass_rate_material_num",
        "country_pass_rate_material_den",
        "country_pass_rate_pass",
        "materiality_threshold",
        "materiality_unit",
        "metric_definition",
        "notes",
    ]
    metric_pass_rates_df = summary_df.loc[summary_df["row_type"].astype(str) == "metric"].copy()
    metric_pass_rates_df = metric_pass_rates_df.reindex(columns=metric_pass_rate_cols)

    summary_col_order = [
        "row_type",
        "gate_item",
        "domain",
        "gate_class",
        "required_for_overall_gate",
        "evaluation_status",
        "status",
        "pass",
        "unit",
        "model_total",
        "reference_total",
        "global_wape_pct_actual",
        "global_wape_pct_threshold",
        "global_wape_pass",
        "country_ape_threshold_pct",
        "country_pass_rate_pct_threshold",
        "country_pass_rate_required",
        "country_pass_rate_all_pct",
        "country_pass_rate_all_num",
        "country_pass_rate_all_den",
        "country_pass_rate_material_pct",
        "country_pass_rate_material_num",
        "country_pass_rate_material_den",
        "country_pass_rate_pass",
        "materiality_threshold",
        "materiality_unit",
        "metric_gate_pass",
        "actual_total_zero_profile_assets",
        "actual_total_zero_profile_capacity_mw",
        "warn_if_total_zero_profile_assets_gt",
        "fail_if_total_zero_profile_assets_gt",
        "warn_if_total_zero_profile_capacity_mw_gt",
        "fail_if_total_zero_profile_capacity_mw_gt",
        "overall_blocking_metrics_status",
        "overall_secondary_metrics_status",
        "overall_guardrail_reporting_status",
        "overall_required_guardrails_status",
        "overall_gate_status",
        "configured_metric_rules",
        "configured_guardrail_rules",
        "blocking_metric_count",
        "blocking_metric_fail_count",
        "blocking_metric_incomplete_count",
        "secondary_metric_count",
        "secondary_metric_fail_count",
        "guardrail_count",
        "guardrail_fail_count",
        "guardrail_warn_count",
        "guardrail_incomplete_count",
        "metric_definition",
        "notes",
    ]
    summary_df = summary_df.reindex(columns=summary_col_order)

    country_col_order = [
        "domain",
        "gate_class",
        "required_for_overall_gate",
        "metric",
        "country",
        "unit",
        "model_value",
        "reference_value",
        "abs_error_value",
        "ape_pct",
        "country_ape_threshold_pct",
        "passes_country_ape_threshold",
        "is_reference_positive",
        "is_material_country",
        "materiality_threshold",
        "materiality_unit",
        "counts_toward_all_country_pass_rate",
        "counts_toward_material_country_pass_rate",
        "metric_definition",
    ]
    country_df = country_df.reindex(columns=country_col_order)
    return summary_df, country_df, metric_pass_rates_df, overall


def _df_to_json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    return json.loads(df.to_json(orient="records"))


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, np.generic):
        return _json_sanitize(value.item())
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value
    return value


def _write_validation_gate_outputs(
    *,
    output_dir: Path,
    gate_summary: pd.DataFrame,
    gate_country_status: pd.DataFrame,
    gate_metric_pass_rates: pd.DataFrame,
    gate_overall: dict[str, Any],
    gate_config: dict[str, Any],
    gate_config_path: Path,
) -> None:
    _round_for_csv(gate_summary).to_csv(output_dir / "validation_gate_summary.csv", index=False)
    _round_for_csv(gate_metric_pass_rates).to_csv(
        output_dir / "validation_gate_metric_pass_rates.csv", index=False
    )
    _round_for_csv(gate_country_status).to_csv(
        output_dir / "validation_gate_country_metric_status.csv", index=False
    )

    json_payload = {
        "config_path": _repo_relative_path_str(gate_config_path),
        "config_version": gate_config.get("version"),
        "overall": _json_sanitize(gate_overall),
        "summary_rows": _df_to_json_records(_round_for_csv(gate_summary)),
        "artifacts": {
            "summary_csv": "validation_gate_summary.csv",
            "metric_pass_rates_csv": "validation_gate_metric_pass_rates.csv",
            "country_metric_status_csv": "validation_gate_country_metric_status.csv",
            "demand_metric_definitions_csv": "electricity_demand_metric_definitions.csv",
        },
    }
    (output_dir / "validation_gate_summary.json").write_text(
        json.dumps(_json_sanitize(json_payload), indent=2, sort_keys=False),
        encoding="utf-8",
    )


def _mapping_rules_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    cap_rules = (
        pd.DataFrame(
            {
                "model_carrier": list(MODEL_CAPACITY_TECH_MAP.keys()),
                "validation_tech": list(MODEL_CAPACITY_TECH_MAP.values()),
            }
        )
        .sort_values(["validation_tech", "model_carrier"])
        .reset_index(drop=True)
    )
    energy_rules = (
        pd.DataFrame(
            {
                "model_carrier": list(MODEL_ENERGY_METRIC_MAP.keys()),
                "owid_metric": list(MODEL_ENERGY_METRIC_MAP.values()),
            }
        )
        .sort_values(["owid_metric", "model_carrier"])
        .reset_index(drop=True)
    )
    return cap_rules, energy_rules


def _write_compact_validation_outputs(
    cap_cmp: pd.DataFrame,
    elec_cmp: pd.DataFrame,
    output_dir: Path,
    *,
    capacity_flag_threshold_pct: float,
    electricity_flag_threshold_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cap_focus = cap_cmp.copy()
    cap_focus["divergence_pct"] = _relative_divergence_pct(
        cap_focus["capacity_mw"], cap_focus["reference_mw"]
    )
    cap_focus["flag"] = cap_focus["divergence_pct"] > float(capacity_flag_threshold_pct)

    cap_summary_focus = _flag_summary_by_group(
        cap_focus,
        group_col="validation_tech",
        model_col="capacity_mw",
        reference_col="reference_mw",
        abs_error_col="abs_error_mw",
        divergence_col="divergence_pct",
        flag_col="flag",
        ape_col="ape_pct",
    )
    cap_flags = cap_focus.loc[cap_focus["flag"]].sort_values(
        ["divergence_pct", "abs_error_mw"], ascending=False
    )

    elec_focus = elec_cmp.loc[elec_cmp["metric"].isin(ELECTRICITY_TECH_METRICS)].copy()
    elec_focus["divergence_pct"] = _relative_divergence_pct(
        elec_focus["model_twh"], elec_focus["reference_twh"]
    )
    elec_focus["flag"] = elec_focus["divergence_pct"] > float(electricity_flag_threshold_pct)

    elec_summary_focus = _flag_summary_by_group(
        elec_focus,
        group_col="metric",
        model_col="model_twh",
        reference_col="reference_twh",
        abs_error_col="abs_error_twh",
        divergence_col="divergence_pct",
        flag_col="flag",
        ape_col="ape_pct",
    )
    elec_flags = elec_focus.loc[elec_focus["flag"]].sort_values(
        ["divergence_pct", "abs_error_twh"], ascending=False
    )

    _round_for_csv(cap_summary_focus).to_csv(
        output_dir / "capacity_focus_summary_by_technology.csv", index=False
    )
    _round_for_csv(cap_flags).to_csv(
        output_dir
        / f"capacity_flags_country_technology_gt{int(capacity_flag_threshold_pct)}pct.csv",
        index=False,
    )
    cap_matrix = cap_focus.pivot_table(
        index="country",
        columns="validation_tech",
        values="divergence_pct",
        aggfunc="first",
        fill_value=np.nan,
    ).reset_index()
    _round_for_csv(cap_matrix).to_csv(
        output_dir / "capacity_divergence_matrix_country_technology_pct.csv", index=False
    )

    _round_for_csv(elec_summary_focus).to_csv(
        output_dir / "electricity_balance_focus_summary_by_technology.csv", index=False
    )
    _round_for_csv(elec_flags).to_csv(
        output_dir
        / f"electricity_balance_flags_country_technology_gt{int(electricity_flag_threshold_pct)}pct.csv",
        index=False,
    )
    elec_matrix = elec_focus.pivot_table(
        index="country",
        columns="metric",
        values="divergence_pct",
        aggfunc="first",
        fill_value=np.nan,
    ).reset_index()
    _round_for_csv(elec_matrix).to_csv(
        output_dir / "electricity_balance_divergence_matrix_country_technology_pct.csv", index=False
    )

    return cap_summary_focus, cap_flags, elec_summary_focus, elec_flags


def _overall_wape(cmp_df: pd.DataFrame, abs_error_col: str, reference_col: str) -> float:
    return float(cmp_df[abs_error_col].sum() / max(cmp_df[reference_col].sum(), 1e-9) * 100.0)


def _status_from_threshold(value: float, target: float) -> str:
    if value <= target:
        return "GOOD"
    if value <= target * 1.2:
        return "BORDERLINE"
    return "NOT_GOOD"


def _save_barh(
    series: pd.Series,
    *,
    title: str,
    xlabel: str,
    output_png: Path,
    color: str = "#3b82f6",
) -> None:
    data = series.dropna()
    if data.empty:
        return
    data = data.sort_values(ascending=True)
    fig_h = max(3.0, 0.34 * len(data) + 1.2)
    fig, ax = plt.subplots(figsize=(9.5, fig_h))
    ax.barh(data.index.astype(str), data.values, color=color, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def _save_divergence_heatmap(
    cmp_df: pd.DataFrame,
    *,
    row_col: str,
    category_col: str,
    divergence_col: str,
    abs_error_col: str,
    top_n_rows: int,
    title: str,
    output_png: Path,
) -> None:
    if cmp_df.empty:
        return

    row_rank = (
        cmp_df.groupby(row_col, as_index=False)[abs_error_col]
        .sum()
        .sort_values(abs_error_col, ascending=False)
    )
    rows = row_rank[row_col].head(top_n_rows).tolist()
    if not rows:
        return

    pivot = (
        cmp_df.loc[cmp_df[row_col].isin(rows), [row_col, category_col, divergence_col]]
        .pivot_table(index=row_col, columns=category_col, values=divergence_col, aggfunc="first")
        .reindex(rows)
    )
    if pivot.empty:
        return

    arr = pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(arr)

    fig_w = max(8.0, 0.65 * len(pivot.columns) + 4.0)
    fig_h = max(6.0, 0.28 * len(pivot.index) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(masked, cmap="RdYlGn_r", vmin=0.0, vmax=100.0, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.astype(str), rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(str))
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("Absolute divergence (%)")
    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def _write_dashboard(
    *,
    cap_cmp: pd.DataFrame,
    cap_summary: pd.DataFrame,
    cap_flags: pd.DataFrame,
    elec_cmp: pd.DataFrame,
    elec_summary: pd.DataFrame,
    elec_flags: pd.DataFrame,
    demand_cmp: pd.DataFrame,
    output_dir: Path,
    capacity_target_pct: float = 20.0,
    electricity_target_pct: float = 10.0,
    demand_target_pct: float = 10.0,
    demand_metric_definitions: dict[str, str] | None = None,
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    cap_focus = cap_cmp.copy()
    if "divergence_pct" not in cap_focus.columns:
        cap_focus["divergence_pct"] = _relative_divergence_pct(
            cap_focus["capacity_mw"], cap_focus["reference_mw"]
        )

    elec_focus = elec_cmp.loc[elec_cmp["metric"].isin(ELECTRICITY_TECH_METRICS)].copy()
    if "divergence_pct" not in elec_focus.columns:
        elec_focus["divergence_pct"] = _relative_divergence_pct(
            elec_focus["model_twh"], elec_focus["reference_twh"]
        )

    cap_overall_wape = _overall_wape(cap_cmp, "abs_error_mw", "reference_mw")
    elec_overall_wape = _overall_wape(elec_focus, "abs_error_twh", "reference_twh")
    demand_overall_wape = _overall_wape(demand_cmp, "abs_error_twh", "reference_twh")

    cap_status = _status_from_threshold(cap_overall_wape, capacity_target_pct)
    elec_status = _status_from_threshold(elec_overall_wape, electricity_target_pct)
    demand_status = _status_from_threshold(demand_overall_wape, demand_target_pct)
    demand_defs = demand_metric_definitions or DEMAND_METRIC_DEFINITIONS

    cap_share = (
        cap_cmp.groupby("validation_tech", as_index=False)["abs_error_mw"]
        .sum()
        .sort_values("abs_error_mw", ascending=False)
    )
    cap_share["contribution_pct"] = (
        cap_share["abs_error_mw"] / max(cap_share["abs_error_mw"].sum(), 1e-9) * 100.0
    )
    elec_share = (
        elec_focus.groupby("metric", as_index=False)["abs_error_twh"]
        .sum()
        .sort_values("abs_error_twh", ascending=False)
    )
    elec_share["contribution_pct"] = (
        elec_share["abs_error_twh"] / max(elec_share["abs_error_twh"].sum(), 1e-9) * 100.0
    )

    cap_country_share = (
        cap_cmp.groupby("country", as_index=False)["abs_error_mw"]
        .sum()
        .sort_values("abs_error_mw", ascending=False)
    )
    cap_country_share["contribution_pct"] = (
        cap_country_share["abs_error_mw"] / max(cap_country_share["abs_error_mw"].sum(), 1e-9) * 100.0
    )
    elec_country_share = (
        elec_focus.groupby("country", as_index=False)["abs_error_twh"]
        .sum()
        .sort_values("abs_error_twh", ascending=False)
    )
    elec_country_share["contribution_pct"] = (
        elec_country_share["abs_error_twh"] / max(elec_country_share["abs_error_twh"].sum(), 1e-9) * 100.0
    )

    _round_for_csv(cap_share).to_csv(output_dir / "painpoints_capacity_by_technology.csv", index=False)
    _round_for_csv(elec_share).to_csv(
        output_dir / "painpoints_electricity_balance_by_technology.csv", index=False
    )
    _round_for_csv(cap_country_share).to_csv(output_dir / "painpoints_capacity_by_country.csv", index=False)
    _round_for_csv(elec_country_share).to_csv(
        output_dir / "painpoints_electricity_balance_by_country.csv", index=False
    )

    _save_barh(
        cap_summary.sort_values("wape_pct", ascending=False).set_index("validation_tech")["wape_pct"],
        title="Capacity Validation WAPE by Technology",
        xlabel="WAPE (%)",
        output_png=plots_dir / "capacity_wape_by_technology.png",
        color="#1f77b4",
    )
    _save_barh(
        elec_summary.loc[elec_summary["metric"].isin(ELECTRICITY_TECH_METRICS)]
        .sort_values("wape_pct", ascending=False)
        .set_index("metric")["wape_pct"],
        title="Electricity Balance WAPE by Technology",
        xlabel="WAPE (%)",
        output_png=plots_dir / "electricity_balance_wape_by_technology.png",
        color="#d62728",
    )
    _save_barh(
        cap_flags.sort_values("abs_error_mw", ascending=False)
        .head(20)
        .assign(label=lambda d: d["country"] + " | " + d["validation_tech"])
        .set_index("label")["abs_error_mw"]
        / 1000.0,
        title="Top 20 Capacity Painpoints (Absolute Error)",
        xlabel="Absolute error (GW)",
        output_png=plots_dir / "capacity_top20_abs_error_gw.png",
        color="#9467bd",
    )
    _save_barh(
        elec_flags.sort_values("abs_error_twh", ascending=False)
        .head(20)
        .assign(label=lambda d: d["country"] + " | " + d["metric"])
        .set_index("label")["abs_error_twh"],
        title="Top 20 Electricity Balance Painpoints (Absolute Error)",
        xlabel="Absolute error (TWh)",
        output_png=plots_dir / "electricity_balance_top20_abs_error_twh.png",
        color="#2ca02c",
    )
    _save_divergence_heatmap(
        cap_focus,
        row_col="country",
        category_col="validation_tech",
        divergence_col="divergence_pct",
        abs_error_col="abs_error_mw",
        top_n_rows=40,
        title="Capacity Divergence Heatmap (Top 40 Countries by Absolute Error)",
        output_png=plots_dir / "capacity_divergence_heatmap_top40_countries.png",
    )
    _save_divergence_heatmap(
        elec_focus,
        row_col="country",
        category_col="metric",
        divergence_col="divergence_pct",
        abs_error_col="abs_error_twh",
        top_n_rows=40,
        title="Electricity Balance Divergence Heatmap (Top 40 Countries by Absolute Error)",
        output_png=plots_dir / "electricity_balance_divergence_heatmap_top40_countries.png",
    )

    status_lines = [
        f"- Capacity (country-tech) overall WAPE: **{cap_overall_wape:.2f}%** (target <= {capacity_target_pct:.1f}%) -> **{cap_status}**",
        f"- Electricity balance (country-tech) overall WAPE: **{elec_overall_wape:.2f}%** (target <= {electricity_target_pct:.1f}%) -> **{elec_status}**",
        f"- Electricity demand overall WAPE: **{demand_overall_wape:.2f}%** (target <= {demand_target_pct:.1f}%) -> **{demand_status}**",
    ]

    cap_top = _round_for_csv(
        cap_flags.sort_values("abs_error_mw", ascending=False).head(10).loc[
            :,
            ["country", "validation_tech", "capacity_mw", "reference_mw", "abs_error_mw", "divergence_pct"],
        ]
    )
    elec_top = _round_for_csv(
        elec_flags.sort_values("abs_error_twh", ascending=False).head(10).loc[
            :,
            ["country", "metric", "model_twh", "reference_twh", "abs_error_twh", "divergence_pct"],
        ]
    )

    md = []
    md.append("# Validation Dashboard")
    md.append("")
    md.append("## Are We Good?")
    md.extend(status_lines)
    md.append("")
    md.append("## Electricity Demand Metric Definitions")
    md.append(
        f"- `electricity_demand`: {demand_defs.get('electricity_demand', DEMAND_METRIC_DEFINITIONS['electricity_demand'])}"
    )
    md.append(
        "- `electricity_demand_total_ac_withdrawal`: "
        + demand_defs.get(
            "electricity_demand_total_ac_withdrawal",
            DEMAND_METRIC_DEFINITIONS["electricity_demand_total_ac_withdrawal"],
        )
    )
    md.append("")
    md.append("## Biggest Painpoints")
    md.append("- Capacity by technology contribution: `painpoints_capacity_by_technology.csv`")
    md.append("- Electricity balance by technology contribution: `painpoints_electricity_balance_by_technology.csv`")
    md.append("- Capacity by country contribution: `painpoints_capacity_by_country.csv`")
    md.append("- Electricity balance by country contribution: `painpoints_electricity_balance_by_country.csv`")
    md.append("")
    md.append("## Top 10 Capacity Painpoints (> threshold flags)")
    md.append(cap_top.to_markdown(index=False))
    md.append("")
    md.append("## Top 10 Electricity Balance Painpoints (> threshold flags)")
    md.append(elec_top.to_markdown(index=False))
    md.append("")
    md.append("## Plots")
    md.append("- `plots/capacity_wape_by_technology.png`")
    md.append("- `plots/electricity_balance_wape_by_technology.png`")
    md.append("- `plots/capacity_top20_abs_error_gw.png`")
    md.append("- `plots/electricity_balance_top20_abs_error_twh.png`")
    md.append("- `plots/capacity_divergence_heatmap_top40_countries.png`")
    md.append("- `plots/electricity_balance_divergence_heatmap_top40_countries.png`")
    md.append("")
    (output_dir / "validation_dashboard.md").write_text("\n".join(md), encoding="utf-8")


def _print_summary(
    cap_cmp: pd.DataFrame,
    cap_summary: pd.DataFrame,
    elec_energy_cmp: pd.DataFrame,
    elec_energy_summary: pd.DataFrame,
    non_elec_cmp: pd.DataFrame,
    non_elec_summary: pd.DataFrame,
    demand_cmp: pd.DataFrame,
    demand_summary: pd.DataFrame,
) -> None:
    total_capacity_wape = (
        cap_cmp["abs_error_mw"].sum() / max(cap_cmp["reference_mw"].sum(), 1e-9) * 100.0
    )
    total_elec_wape = (
        elec_energy_cmp["abs_error_twh"].sum()
        / max(elec_energy_cmp["reference_twh"].sum(), 1e-9)
        * 100.0
    )
    total_non_elec_wape = (
        non_elec_cmp["abs_error_twh"].sum()
        / max(non_elec_cmp["reference_twh"].sum(), 1e-9)
        * 100.0
    )
    total_demand_wape = (
        demand_cmp["abs_error_twh"].sum()
        / max(demand_cmp["reference_twh"].sum(), 1e-9)
        * 100.0
    )

    LOGGER.info("Capacity validation summary (sorted by WAPE):")
    for _, row in cap_summary.iterrows():
        LOGGER.info(
            "  %-22s model=%10.1f MW ref=%10.1f MW wape=%7.2f%% mean_ape=%7.2f%%",
            row["validation_tech"],
            row["model_total_mw"],
            row["reference_total_mw"],
            row["wape_pct"],
            row["mean_ape_pct"],
        )
    LOGGER.info("  [overall] weighted error (capacity) ~= %.2f%%", total_capacity_wape)

    LOGGER.info("Electricity balance validation summary (sorted by WAPE):")
    for _, row in elec_energy_summary.iterrows():
        LOGGER.info(
            "  %-25s model=%10.1f TWh ref=%10.1f TWh wape=%7.2f%% mean_ape=%7.2f%%",
            row["metric"],
            row["model_total_twh"],
            row["reference_total_twh"],
            row["wape_pct"],
            row["mean_ape_pct"],
        )
    LOGGER.info("  [overall] weighted error (electricity balance) ~= %.2f%%", total_elec_wape)

    LOGGER.info("Non-electric fossil energy summary (sorted by WAPE):")
    for _, row in non_elec_summary.iterrows():
        LOGGER.info(
            "  %-31s model=%10.1f TWh ref=%10.1f TWh wape=%7.2f%% mean_ape=%7.2f%%",
            row["metric"],
            row["model_total_twh"],
            row["reference_total_twh"],
            row["wape_pct"],
            row["mean_ape_pct"],
        )
    LOGGER.info("  [overall] weighted error (non-electric fossil) ~= %.2f%%", total_non_elec_wape)

    LOGGER.info("Electricity demand validation summary (sorted by WAPE):")
    for _, row in demand_summary.iterrows():
        LOGGER.info(
            "  %-25s model=%10.1f TWh ref=%10.1f TWh wape=%7.2f%% mean_ape=%7.2f%%",
            row["metric"],
            row["model_total_twh"],
            row["reference_total_twh"],
            row["wape_pct"],
            row["mean_ape_pct"],
        )
    LOGGER.info("  [overall] weighted error (electricity demand) ~= %.2f%%", total_demand_wape)


def parse_args() -> argparse.Namespace:
    default_network = (
        "results/Global_200/postnetworks/elec_s_200_ec_lcopt_1h_1h_2020_0.071_AB_0.0export_base.nc"
    )
    parser = argparse.ArgumentParser(
        description="Validate solved network capacities and energy balances by country."
    )
    parser.add_argument("--network", type=Path, default=Path(default_network), help="Path to solved .nc network")
    parser.add_argument(
        "--irena-csv",
        type=Path,
        default=REPO_ROOT / "validation" / "data" / "irena_capacity_by_technology.csv",
        help="Path to IRENA capacity CSV",
    )
    parser.add_argument(
        "--iaea-csv",
        type=Path,
        default=REPO_ROOT / "validation" / "data" / "Nuclear_capacity_IEAE.csv",
        help="Path to IAEA nuclear capacity CSV",
    )
    parser.add_argument(
        "--gem-clean-csv",
        type=Path,
        default=REPO_ROOT / "validation" / "data" / "gem_capacity_reference_2020.csv",
        help="Path to cleaned GEM capacity CSV from prepare_gem_capacity_reference.py",
    )
    parser.add_argument(
        "--owid-csv",
        type=Path,
        default=REPO_ROOT / "validation" / "data" / "owid-energy-data.csv",
        help="Path to OWID energy CSV",
    )
    parser.add_argument("--year", type=int, default=2020, help="Validation year")
    parser.add_argument(
        "--baseyear-extendability-allowlist",
        type=Path,
        default=REPO_ROOT / "validation" / "config.baseyear_extendability_allowlist.yaml",
        help="YAML allowlist for 2020 extendable non-physical/accounting assets.",
    )
    parser.add_argument(
        "--skip-baseyear-extendability-check",
        action="store_true",
        help="Skip the 2020 extendability audit/hard-fail check.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "validation" / "results",
        help="Directory for validation outputs",
    )
    parser.add_argument(
        "--capacity-flag-threshold-pct",
        type=float,
        default=20.0,
        help="Flag capacity country-technology entries above this absolute divergence percent",
    )
    parser.add_argument(
        "--electricity-flag-threshold-pct",
        type=float,
        default=10.0,
        help="Flag electricity balance country-technology entries above this absolute divergence percent",
    )
    parser.add_argument(
        "--csv-output-profile",
        choices=["minimal", "full"],
        default="minimal",
        help="CSV output volume. 'minimal' keeps essential comparison/flags/summaries; 'full' writes all diagnostics.",
    )
    parser.add_argument(
        "--validation-gate-config",
        type=Path,
        default=DEFAULT_VALIDATION_GATE_CONFIG_PATH,
        help="Path to validation gate config (YAML or JSON-compatible YAML).",
    )
    parser.add_argument(
        "--guardrails-only",
        action="store_true",
        help="Run workflow guardrail diagnostics only and emit guardrail artifacts without full validation outputs.",
    )
    parser.add_argument(
        "--fail-on-guardrail-fail",
        action="store_true",
        help="Exit non-zero if any workflow guardrail check is classified as fail.",
    )
    parser.add_argument(
        "--guardrail-zero-profile-hit-severity",
        choices=["pass", "warn", "fail"],
        default="fail",
        help="Severity to assign when renewable zero-profile assets are detected.",
    )
    parser.add_argument(
        "--guardrail-hydro-missing-inflow-hit-severity",
        choices=["pass", "warn", "fail"],
        default="fail",
        help="Severity to assign when hydro reservoir inflow columns are missing.",
    )
    parser.add_argument(
        "--guardrail-hydro-zero-ror-hit-severity",
        choices=["pass", "warn", "fail"],
        default="fail",
        help="Severity to assign when run-of-river assets have zero profiles.",
    )
    parser.add_argument(
        "--guardrail-hydro-zero-reservoir-inflow-hit-severity",
        choices=["pass", "warn", "fail"],
        default="fail",
        help="Severity to assign when hydro reservoir inflow profiles are all zero.",
    )
    parser.add_argument(
        "--guardrail-nan-load-hit-severity",
        choices=["pass", "warn", "fail"],
        default="fail",
        help="Severity to assign when NaNs are found in AC load p_set time series.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gate_config = _load_validation_gate_config(args.validation_gate_config)
    demand_metric_definitions = dict(
        gate_config.get("demand_metric_definitions", DEMAND_METRIC_DEFINITIONS)
    )

    LOGGER.info("Loading network: %s", args.network)
    n = pypsa.Network(args.network)

    baseyear_extendability_audit = _empty_baseyear_extendability_audit()
    baseyear_extendability_summary = pd.DataFrame(
        [
            {
                "year": int(args.year),
                "total_extendable_assets": 0,
                "allowed_assets": 0,
                "unexpected_assets": 0,
                "unexpected_blocking_assets": 0,
                "unexpected_zero_placeholder_nonblocking_assets": 0,
                "status": (
                    "skipped"
                    if args.skip_baseyear_extendability_check or args.year != 2020
                    else "not_run"
                ),
            }
        ]
    )
    extendability_gate_failed = False
    extendability_check_enabled = not args.skip_baseyear_extendability_check and args.year == 2020
    if extendability_check_enabled:
        allowlist_rules = _load_baseyear_extendability_allowlist(args.baseyear_extendability_allowlist)
        baseyear_extendability_audit, baseyear_extendability_summary = _baseyear_extendability_audit(
            n=n,
            year=args.year,
            allowlist_rules=allowlist_rules,
        )
        baseyear_extendability_summary["allowlist_path"] = _repo_relative_path_str(
            args.baseyear_extendability_allowlist
        )
        unexpected_extendable = baseyear_extendability_audit.loc[
            baseyear_extendability_audit["classification"] == "unexpected"
        ].copy()
        unexpected_blocking_extendable = baseyear_extendability_audit.loc[
            baseyear_extendability_audit.get("gate_blocking", pd.Series(False, index=baseyear_extendability_audit.index))
        ].copy()
        nonblocking_zero_placeholders = baseyear_extendability_audit.loc[
            baseyear_extendability_audit.get(
                "is_zero_capacity_placeholder",
                pd.Series(False, index=baseyear_extendability_audit.index),
            )
        ].copy()
        extendability_gate_failed = not unexpected_blocking_extendable.empty
        if extendability_gate_failed:
            LOGGER.error(
                "2020 extendability audit found %d blocking unexpected extendable assets "
                "(plus %d non-blocking zero-cap placeholders; see baseyear_extendability_audit.csv).",
                len(unexpected_blocking_extendable),
                len(nonblocking_zero_placeholders),
            )
        else:
            LOGGER.info(
                "2020 extendability audit passed: blocking unexpected=0 "
                "(non-blocking zero-cap placeholders=%d, total unexpected=%d, total extendable=%d).",
                len(nonblocking_zero_placeholders),
                len(unexpected_extendable),
                len(baseyear_extendability_audit),
            )
    else:
        baseyear_extendability_summary["allowlist_path"] = _repo_relative_path_str(
            args.baseyear_extendability_allowlist
        )

    network_countries = set(n.buses.country.dropna())
    network_countries.discard("")
    network_iso3 = {
        two_2_three_digits_country(c) for c in network_countries if isinstance(c, str) and len(c) == 2
    }
    network_iso3 = {c for c in network_iso3 if isinstance(c, str) and len(c) == 3}

    zero_profile_assets, zero_profile_summary = _renewable_zero_profile_diagnostics(n)
    guardrail_summary, guardrail_detail, guardrail_status = _workflow_guardrail_artifacts(
        n,
        zero_profile_assets,
        zero_profile_hit_severity=args.guardrail_zero_profile_hit_severity,
        hydro_missing_inflow_hit_severity=args.guardrail_hydro_missing_inflow_hit_severity,
        hydro_zero_ror_hit_severity=args.guardrail_hydro_zero_ror_hit_severity,
        hydro_zero_reservoir_inflow_hit_severity=args.guardrail_hydro_zero_reservoir_inflow_hit_severity,
        nan_load_hit_severity=args.guardrail_nan_load_hit_severity,
    )
    guardrail_status = _write_workflow_guardrail_artifacts(
        args.output_dir,
        network_path=args.network,
        year=args.year,
        summary_df=guardrail_summary,
        detail_df=guardrail_detail,
        status_obj=guardrail_status,
    )
    LOGGER.info(
        "Workflow guardrails: overall=%s (fail=%d, warn=%d, pass=%d).",
        guardrail_status["overall_status"],
        guardrail_status["checks_fail"],
        guardrail_status["checks_warn"],
        guardrail_status["checks_pass"],
    )
    if guardrail_status["overall_status"] == "fail":
        zero_hits = zero_profile_assets.loc[zero_profile_assets["profile_all_zero"]].copy()
        if not zero_hits.empty:
            LOGGER.warning(
                "Detected %d renewable assets with non-zero capacity but zero annual available energy (see renewable_zero_profile_assets.csv and workflow_guardrail_summary.csv).",
                len(zero_hits),
            )

    if args.guardrails_only:
        LOGGER.info(
            "Guardrails-only mode complete. Wrote artifacts to %s",
            _repo_relative_path_str(args.output_dir),
        )
        if args.fail_on_guardrail_fail and guardrail_status["overall_status"] == "fail":
            raise SystemExit(2)
        return

    model_cap, unmapped_cap, cap_mapping_diag = _model_capacity_by_country_tech(n)
    model_cap = model_cap.loc[model_cap.country.isin(network_countries)].copy()
    ref_cap = _capacity_reference_by_country_tech(
        args.irena_csv,
        args.iaea_csv,
        args.gem_clean_csv,
        args.year,
    )
    ref_cap = ref_cap.loc[ref_cap.country.isin(network_countries)].copy()

    active_capacity_techs = set(model_cap["validation_tech"].unique())
    dropped_ref_techs = sorted(set(ref_cap["validation_tech"].unique()) - active_capacity_techs)
    ref_cap = ref_cap.loc[ref_cap["validation_tech"].isin(active_capacity_techs)].copy()
    if dropped_ref_techs:
        LOGGER.info(
            "Skipping reference-only capacity categories not represented in model carriers: %s",
            ", ".join(dropped_ref_techs),
        )

    cap_cmp, cap_summary = _capacity_comparison(model_cap, ref_cap)
    cap_rollups = _capacity_comparison_rollups(cap_cmp)
    cap_coverage_by_tech = _coverage_by_group(
        cap_cmp,
        group_col="validation_tech",
        model_col="capacity_mw",
        reference_col="reference_mw",
    )
    cap_coverage_by_country = _coverage_by_country(
        cap_cmp,
        group_col="validation_tech",
        model_col="capacity_mw",
        reference_col="reference_mw",
    )

    model_elec = _model_electricity_balance(n)
    model_elec = model_elec.loc[model_elec.country.isin(network_countries)].copy()
    elec_mapping_diag = _model_electricity_mapping_diagnostics(n)
    elec_mapping_diag = elec_mapping_diag.loc[elec_mapping_diag["carrier"].notna()].copy()
    ref_elec = _owid_electricity_balance(args.owid_csv, args.year, iso3_filter=network_iso3)
    ref_elec = ref_elec.loc[ref_elec.country.isin(network_countries)].copy()
    elec_cmp, elec_summary = _energy_comparison(model_elec, ref_elec)
    elec_rollups = _energy_comparison_rollups(elec_cmp)
    elec_tech_cmp = elec_cmp.loc[elec_cmp["metric"].isin(ELECTRICITY_TECH_METRICS)].copy()
    elec_coverage_by_metric = _coverage_by_group(
        elec_tech_cmp,
        group_col="metric",
        model_col="model_twh",
        reference_col="reference_twh",
    )
    elec_coverage_by_country = _coverage_by_country(
        elec_tech_cmp,
        group_col="metric",
        model_col="model_twh",
        reference_col="reference_twh",
    )

    model_non_elec = _model_non_electric_fossil_balance(n)
    model_non_elec = model_non_elec.loc[model_non_elec.country.isin(network_countries)].copy()
    ref_non_elec = _owid_non_electric_fossil_balance(
        args.owid_csv, args.year, iso3_filter=network_iso3
    )
    ref_non_elec = ref_non_elec.loc[ref_non_elec.country.isin(network_countries)].copy()
    non_elec_cmp, non_elec_summary = _energy_comparison(model_non_elec, ref_non_elec)

    model_demand_breakdown = _model_electricity_demand_breakdown(n)
    model_demand_breakdown = model_demand_breakdown.loc[
        model_demand_breakdown.country.isin(network_countries)
    ].copy()
    model_demand = _model_electricity_demand(n)
    model_demand = model_demand.loc[model_demand.country.isin(network_countries)].copy()
    ref_demand = _owid_electricity_demand(args.owid_csv, args.year, iso3_filter=network_iso3)
    ref_demand = ref_demand.loc[ref_demand.country.isin(network_countries)].copy()
    demand_cmp, demand_summary = _energy_comparison(model_demand, ref_demand)
    demand_rollups = _energy_comparison_rollups(demand_cmp)
    demand_components = model_demand_breakdown.loc[
        model_demand_breakdown["metric"].isin(
            [
                "electricity_demand_load_component",
                "electricity_demand_link_component",
                "electricity_demand_storage_charging",
                "electricity_demand_store_component",
                "electricity_demand_total_ac_withdrawal",
            ]
        )
    ].copy()
    gate_summary, gate_country_status, gate_metric_pass_rates, gate_overall = _evaluate_validation_gate(
        gate_config=gate_config,
        cap_cmp=cap_cmp,
        elec_cmp=elec_cmp,
        demand_cmp=demand_cmp,
        non_elec_cmp=non_elec_cmp,
        zero_profile_summary=zero_profile_summary,
    )

    write_full_csvs = args.csv_output_profile == "full"

    _demand_metric_definitions_table(demand_metric_definitions).to_csv(
        args.output_dir / "electricity_demand_metric_definitions.csv",
        index=False,
    )
    _round_for_csv(cap_cmp).to_csv(
        args.output_dir / "capacity_comparison_country_technology.csv", index=False
    )
    _round_for_csv(elec_cmp).to_csv(args.output_dir / "energy_comparison_country_metric.csv", index=False)
    _round_for_csv(elec_summary).to_csv(args.output_dir / "energy_summary_by_metric.csv", index=False)
    _round_for_csv(demand_cmp).to_csv(
        args.output_dir / "electricity_demand_comparison_country_metric.csv", index=False
    )
    _round_for_csv(demand_summary).to_csv(
        args.output_dir / "electricity_demand_summary_by_metric.csv", index=False
    )
    _round_for_csv(demand_components).to_csv(
        args.output_dir / "electricity_demand_components_country_metric.csv", index=False
    )
    _round_for_csv(cap_mapping_diag).to_csv(
        args.output_dir / "capacity_mapping_diagnostics_by_carrier.csv", index=False
    )
    _round_for_csv(elec_mapping_diag).to_csv(
        args.output_dir / "electricity_mapping_diagnostics_by_carrier.csv", index=False
    )
    _round_for_csv(zero_profile_assets).to_csv(
        args.output_dir / "renewable_zero_profile_assets.csv", index=False
    )
    _round_for_csv(zero_profile_summary).to_csv(
        args.output_dir / "renewable_zero_profile_country_carrier_summary.csv", index=False
    )
    _round_for_csv(baseyear_extendability_audit).to_csv(
        args.output_dir / "baseyear_extendability_audit.csv", index=False
    )
    _round_for_csv(baseyear_extendability_summary).to_csv(
        args.output_dir / "baseyear_extendability_summary.csv", index=False
    )
    _write_validation_gate_outputs(
        output_dir=args.output_dir,
        gate_summary=gate_summary,
        gate_country_status=gate_country_status,
        gate_metric_pass_rates=gate_metric_pass_rates,
        gate_overall=gate_overall,
        gate_config=gate_config,
        gate_config_path=args.validation_gate_config,
    )

    if write_full_csvs:
        _round_for_csv(cap_summary).to_csv(args.output_dir / "capacity_summary_by_technology.csv", index=False)
        _round_for_csv(non_elec_cmp).to_csv(
            args.output_dir / "fossil_non_electric_energy_comparison_country_metric.csv", index=False
        )
        _round_for_csv(non_elec_summary).to_csv(
            args.output_dir / "fossil_non_electric_energy_summary_by_metric.csv", index=False
        )
        _round_for_csv(unmapped_cap).to_csv(
            args.output_dir / "unmapped_electricity_carriers_capacity.csv", index=False
        )
        _round_for_csv(cap_rollups).to_csv(
            args.output_dir / "capacity_comparison_rollups_country_continent_global.csv",
            index=False,
        )
        _round_for_csv(cap_coverage_by_tech).to_csv(
            args.output_dir / "capacity_coverage_by_technology.csv", index=False
        )
        _round_for_csv(cap_coverage_by_country).to_csv(
            args.output_dir / "capacity_coverage_by_country.csv", index=False
        )
        _round_for_csv(elec_rollups).to_csv(
            args.output_dir / "electricity_balance_rollups_country_continent_global.csv",
            index=False,
        )
        _round_for_csv(elec_coverage_by_metric).to_csv(
            args.output_dir / "electricity_balance_coverage_by_metric.csv", index=False
        )
        _round_for_csv(elec_coverage_by_country).to_csv(
            args.output_dir / "electricity_balance_coverage_by_country.csv", index=False
        )
        _round_for_csv(model_demand_breakdown).to_csv(
            args.output_dir / "electricity_demand_breakdown_country_metric.csv", index=False
        )
        _round_for_csv(demand_rollups).to_csv(
            args.output_dir / "electricity_demand_rollups_country_continent_global.csv",
            index=False,
        )
        cap_rules, energy_rules = _mapping_rules_tables()
        _round_for_csv(cap_rules).to_csv(args.output_dir / "mapping_rules_capacity.csv", index=False)
        _round_for_csv(energy_rules).to_csv(
            args.output_dir / "mapping_rules_electricity_balance.csv", index=False
        )

    (
        cap_summary_focus,
        cap_flags,
        elec_summary_focus,
        elec_flags,
    ) = _write_compact_validation_outputs(
        cap_cmp,
        elec_cmp,
        args.output_dir,
        capacity_flag_threshold_pct=args.capacity_flag_threshold_pct,
        electricity_flag_threshold_pct=args.electricity_flag_threshold_pct,
    )
    _write_dashboard(
        cap_cmp=cap_cmp,
        cap_summary=cap_summary,
        cap_flags=cap_flags,
        elec_cmp=elec_cmp,
        elec_summary=elec_summary,
        elec_flags=elec_flags,
        demand_cmp=demand_cmp,
        output_dir=args.output_dir,
        capacity_target_pct=args.capacity_flag_threshold_pct,
        electricity_target_pct=args.electricity_flag_threshold_pct,
        demand_target_pct=args.electricity_flag_threshold_pct,
        demand_metric_definitions=demand_metric_definitions,
    )

    LOGGER.info("Wrote validation outputs to %s", _repo_relative_path_str(args.output_dir))
    LOGGER.info(
        "Workflow guardrail status artifact: %s",
        _repo_relative_path_str(args.output_dir / "workflow_guardrail_status.json"),
    )
    LOGGER.info("CSV output profile: %s", args.csv_output_profile)
    LOGGER.info(
        "Validation gate (%s): overall=%s | blocking_metrics=%s | secondary_metrics=%s | guardrails=%s",
        args.validation_gate_config,
        gate_overall.get("overall_gate_status", "unknown"),
        gate_overall.get("overall_blocking_metrics_status", "unknown"),
        gate_overall.get("overall_secondary_metrics_status", "unknown"),
        gate_overall.get("overall_guardrail_reporting_status", "unknown"),
    )
    LOGGER.info(
        "Capacity flags (>%.1f%% divergence): %d / %d entries",
        args.capacity_flag_threshold_pct,
        len(cap_flags),
        len(cap_cmp),
    )
    LOGGER.info(
        "Electricity-balance flags (>%.1f%% divergence): %d / %d entries (technology metrics only)",
        args.electricity_flag_threshold_pct,
        len(elec_flags),
        len(elec_cmp.loc[elec_cmp.metric.isin(ELECTRICITY_TECH_METRICS)]),
    )
    if not cap_summary_focus.empty:
        LOGGER.info(
            "Top capacity flag rates by technology:\n%s",
            cap_summary_focus.head(8).to_string(index=False),
        )
    if not elec_summary_focus.empty:
        LOGGER.info(
            "Top electricity-balance flag rates by technology:\n%s",
            elec_summary_focus.head(8).to_string(index=False),
        )
    if not unmapped_cap.empty:
        LOGGER.info(
            "Unmapped electricity-output carriers (not compared to IRENA):\n%s",
            unmapped_cap.head(20).to_string(index=False),
        )
    if not cap_coverage_by_tech.empty:
        LOGGER.info(
            "Lowest capacity coverage vs reference (technology):\n%s",
            _round_for_csv(cap_coverage_by_tech.head(8)).to_string(index=False),
        )
    if not elec_coverage_by_metric.empty:
        LOGGER.info(
            "Lowest electricity balance coverage vs reference (metric):\n%s",
            _round_for_csv(elec_coverage_by_metric.head(8)).to_string(index=False),
        )
    if not demand_components.empty:
        global_demand_split = (
            demand_components.groupby("metric", as_index=False)["model_twh"]
            .sum()
            .sort_values("model_twh", ascending=False)
        )
        LOGGER.info(
            "Electricity demand component split (global, TWh):\n%s",
            _round_for_csv(global_demand_split).to_string(index=False),
        )
    LOGGER.info(
        "Electricity demand metric definitions: electricity_demand = %s; electricity_demand_total_ac_withdrawal = %s",
        demand_metric_definitions.get("electricity_demand", DEMAND_METRIC_DEFINITIONS["electricity_demand"]),
        demand_metric_definitions.get(
            "electricity_demand_total_ac_withdrawal",
            DEMAND_METRIC_DEFINITIONS["electricity_demand_total_ac_withdrawal"],
        ),
    )

    LOGGER.info(
        "Non-electric OWID reference is computed as fuel consumption minus fuel electricity output."
    )
    _print_summary(
        cap_cmp,
        cap_summary,
        elec_cmp,
        elec_summary,
        non_elec_cmp,
        non_elec_summary,
        demand_cmp,
        demand_summary,
    )
    if args.fail_on_guardrail_fail and guardrail_status["overall_status"] == "fail":
        raise SystemExit(2)

    if extendability_check_enabled and extendability_gate_failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
