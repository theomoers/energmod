#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iteratively tune biomass/biogas potential inputs (config values) for baseyear runs.

What this script tunes:
- `sector.solid_biomass_potential` (TWh/a)
- `sector.biogas_potential` (TWh/a)

Why separate: these are config inputs, not per-country profile scalers.
By default the script tunes solid biomass only against OWID `biofuel_electricity`
(global + country diagnostics) and keeps biogas potential fixed. Use `--mode joint`
to scale both together.

Notes on model mapping:
- `biofuel_electricity` in validation mainly reflects solid-biomass electricity / CHP.
- `biogas` is upgraded into the gas system (`biogas to gas`) and usually affects gas
  dispatch rather than OWID `biofuel_electricity` directly.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import ruamel.yaml
from tuner_guardrails import (
    raise_if_simulated_failure,
    restore_from_last_good,
    sync_last_good_from_mutable,
)

LOG = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _helpers import three_2_two_digits_country  # noqa: E402

BIOFUEL_ELEC_CARRIERS = {
    "biomass",
    "urban central solid biomass CHP",
    "urban central solid biomass CHP CC",
}
SOLID_BIOMASS_INPUT_LINK_CARRIERS = {
    "biomass EOP",
    "urban central solid biomass CHP",
    "urban central solid biomass CHP CC",
}


def _safe_iso3_to_iso2(code):
    if not isinstance(code, str) or len(code) != 3:
        return np.nan
    try:
        iso2 = three_2_two_digits_country(code)
    except Exception:
        return np.nan
    if isinstance(iso2, str) and len(iso2) == 2:
        return iso2
    return np.nan


def _repo_path(path_like):
    p = Path(path_like)
    return p if p.is_absolute() else REPO_ROOT / p


def _run_cmd(cmd, cwd, env=None, dry_run=False):
    LOG.info("RUN: %s", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _run_snakemake(args):
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp")
    env.setdefault("XDG_CACHE_HOME", "/tmp")

    if args.unlock_first:
        unlock_cmd = [
            "snakemake",
            "--unlock",
            "--configfile",
            *args.configfiles,
            "--rerun-trigger",
            "mtime",
        ]
        _run_cmd(unlock_cmd, REPO_ROOT, env=env, dry_run=args.dry_run)

    cmd = [
        "snakemake",
        "-j",
        str(args.snakemake_cores),
        args.network_target,
        "--configfile",
        *args.configfiles,
        "--rerun-trigger",
        "mtime",
        "--forcerun",
        *args.force_rules,
    ]
    _run_cmd(cmd, REPO_ROOT, env=env, dry_run=args.dry_run)


def _load_owid_biofuel_reference(owid_csv, year):
    usecols = {"year", "iso_code", "biofuel_electricity"}
    owid = pd.read_csv(owid_csv, usecols=lambda c: c in usecols)
    owid = owid.loc[owid["year"] == int(year)].copy()
    owid["country"] = owid["iso_code"].apply(_safe_iso3_to_iso2)
    owid = owid.loc[owid["country"].notna()].copy()
    owid["biofuel_electricity"] = pd.to_numeric(owid["biofuel_electricity"], errors="coerce").fillna(0.0)
    return owid.groupby("country")["biofuel_electricity"].sum(min_count=1).sort_index()


def _load_model_biofuel_electricity_by_country(network_path):
    n = pypsa.Network(network_path)
    eb = n.statistics.energy_balance(
        bus_carrier="AC",
        aggregate_time="sum",
        aggregate_groups="sum",
        groupby=n.statistics.groupers.get_country_and_carrier,
        nice_names=False,
    )
    eb = eb.rename("energy_mwh").reset_index()
    eb = eb.loc[(eb.country != "") & (eb.energy_mwh > 0)].copy()

    bio = (
        eb.loc[(eb.component.isin(["Generator", "Link"])) & (eb.carrier.isin(BIOFUEL_ELEC_CARRIERS))]
        .groupby("country")["energy_mwh"]
        .sum()
        / 1e6
    )

    # Diagnostics: biomass/biogas capacities and actual link input/output energy.
    stores_diag = {
        "solid_biomass_store_use_twh": np.nan,
        "biogas_store_use_twh": np.nan,
        "solid_biomass_store_capacity_twh": np.nan,
        "biogas_store_capacity_twh": np.nan,
        "biogas_to_gas_twh": np.nan,
    }
    if not n.stores.empty:
        store_carriers = n.stores["carrier"].astype(str)
        for carrier, key_prefix in [("solid biomass", "solid_biomass"), ("biogas", "biogas")]:
            idx = n.stores.index[store_carriers.eq(carrier)]
            if len(idx) == 0:
                continue
            e_nom_col = "e_nom_opt" if "e_nom_opt" in n.stores.columns and n.stores.loc[idx, "e_nom_opt"].notna().any() else "e_nom"
            e_cap = pd.to_numeric(n.stores.loc[idx, e_nom_col], errors="coerce").fillna(0.0).sum()
            stores_diag[f"{key_prefix}_store_capacity_twh"] = float(e_cap) / 1e6

    try:
        weights = n.snapshot_weightings["generators"]
        if not n.links.empty and not n.links_t.p0.empty:
            solid_idx = n.links.index[n.links["carrier"].astype(str).isin(SOLID_BIOMASS_INPUT_LINK_CARRIERS)]
            solid_cols = n.links_t.p0.columns.intersection(solid_idx)
            if len(solid_cols) > 0:
                solid_in_mwh = (
                    n.links_t.p0.reindex(columns=solid_cols)
                    .fillna(0.0)
                    .clip(lower=0.0)
                    .mul(weights, axis=0)
                    .sum()
                    .sum()
                )
                stores_diag["solid_biomass_store_use_twh"] = float(solid_in_mwh) / 1e6

            biogas_idx = n.links.index[n.links["carrier"].astype(str).eq("biogas to gas")]
            biogas_cols = n.links_t.p0.columns.intersection(biogas_idx)
            if len(biogas_cols) > 0:
                biogas_in_mwh = (
                    n.links_t.p0.reindex(columns=biogas_cols)
                    .fillna(0.0)
                    .clip(lower=0.0)
                    .mul(weights, axis=0)
                    .sum()
                    .sum()
                )
                stores_diag["biogas_store_use_twh"] = float(biogas_in_mwh) / 1e6
                if not n.links_t.p1.empty:
                    biogas_out_mwh = (
                        (-n.links_t.p1.reindex(columns=biogas_cols))
                        .fillna(0.0)
                        .clip(lower=0.0)
                        .mul(weights, axis=0)
                        .sum()
                        .sum()
                    )
                    stores_diag["biogas_to_gas_twh"] = float(biogas_out_mwh) / 1e6
    except Exception:
        pass

    out = pd.DataFrame({"biofuel_twh": bio}).fillna(0.0).sort_index()
    return out, stores_diag


def _compute_metrics(model_df, target_df, min_ref_country_twh):
    comp = target_df.join(model_df[["biofuel_twh"]], how="left").fillna(0.0)
    comp["error_twh"] = comp["biofuel_twh"] - comp["target_biofuel_twh"]
    comp["abs_error_twh"] = comp["error_twh"].abs()
    comp["ape_pct"] = np.where(
        comp["target_biofuel_twh"] > 0,
        100.0 * comp["abs_error_twh"] / comp["target_biofuel_twh"],
        np.nan,
    )

    valid = comp["target_biofuel_twh"] >= float(min_ref_country_twh)
    comp_valid = comp.loc[valid].copy()
    if comp_valid.empty:
        country_wape = np.nan
    else:
        denom = comp_valid["target_biofuel_twh"].sum()
        country_wape = 100.0 * comp_valid["abs_error_twh"].sum() / denom if denom > 0 else np.nan

    model_total = float(comp["biofuel_twh"].sum())
    target_total = float(comp["target_biofuel_twh"].sum())
    global_err = (100.0 * abs(model_total - target_total) / target_total) if target_total > 0 else np.nan
    return {
        "country_wape_pct": country_wape,
        "biofuel_model_twh": model_total,
        "biofuel_target_twh": target_total,
        "global_abs_pct": global_err,
        "countries_eval": int(valid.sum()),
    }, comp


def _compute_adjustment(target, model, min_adjust, max_adjust):
    if target > 0 and model > 0:
        ratio = target / model
    elif target > 0 and model <= 0:
        ratio = max_adjust
    elif target <= 0 and model > 0:
        ratio = min_adjust
    else:
        ratio = 1.0
    return float(np.clip(ratio, min_adjust, max_adjust))


def _yaml_load(path):
    yaml = ruamel.yaml.YAML(typ="safe")
    with open(path, "r") as f:
        data = yaml.load(f) or {}
    return data


def _get_nested(d, keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _write_biomass_overlay(path, solid_twh, biogas_twh):
    path.parent.mkdir(parents=True, exist_ok=True)
    yaml = ruamel.yaml.YAML()
    yaml.preserve_quotes = True
    data = {}
    if path.exists():
        with open(path, "r") as f:
            data = yaml.load(f) or {}
    if not isinstance(data, dict):
        data = {}
    sector = data.get("sector")
    if not isinstance(sector, dict):
        sector = {}
        data["sector"] = sector
    sector["solid_biomass_potential"] = float(solid_twh)
    sector["biogas_potential"] = float(biogas_twh)
    with open(path, "w") as f:
        yaml.dump(data, f)


def parse_args():
    p = argparse.ArgumentParser(description="Tune biomass/biogas potential config inputs for baseyear validation.")
    p.add_argument(
        "--network-target",
        default="results/Global_200/postnetworks/elec_s_200_ec_lcopt_1h_1h_2020_0.071_AB_0.0export_base.nc",
    )
    p.add_argument(
        "--configfiles",
        nargs="+",
        default=[
            "config.myopic.yaml",
            "config.learning.yaml",
            "validation/config.iteration_common.yaml",
        ],
    )
    p.add_argument("--base-config", default="config.myopic.yaml")
    p.add_argument("--overlay-config", default="validation/config.iteration_common.yaml")
    p.add_argument(
        "--last-good-overlay-config",
        default="validation/config.iteration_common_biomass_last_good.yaml",
        help="Rollback snapshot of the last known-good biomass tuning overlay.",
    )
    p.add_argument("--owid-csv", default="validation/data/owid-energy-data.csv")
    p.add_argument("--year", type=int, default=2020)
    p.add_argument("--iterations", type=int, default=6)
    p.add_argument("--mode", choices=["solid-only", "joint"], default="solid-only")
    p.add_argument("--history-csv", default="validation/results_compare/biomass_tuning_history.csv")
    p.add_argument(
        "--country-detail-csv",
        default="validation/results_compare/biomass_tuning_country_detail.csv",
    )
    p.add_argument("--country-min-ref-twh", type=float, default=1.0)
    p.add_argument("--target-global-error-pct", type=float, default=2.0)
    p.add_argument("--target-country-wape-pct", type=float, default=None)
    p.add_argument("--adjust-alpha", type=float, default=0.7)
    p.add_argument("--min-adjust", type=float, default=0.5)
    p.add_argument("--max-adjust", type=float, default=2.0)
    p.add_argument("--min-factor", type=float, default=0.1)
    p.add_argument("--max-factor", type=float, default=10.0)
    p.add_argument("--snakemake-cores", type=int, default=1)
    p.add_argument(
        "--force-rules",
        nargs="+",
        default=[
            "prepare_sector_network",
            "add_export",
            "add_existing_baseyear",
            "apply_learning_costs",
            "solve_network_myopic",
        ],
    )
    p.add_argument("--unlock-first", action="store_true")
    p.add_argument(
        "--simulate-post-write-failure",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    network_target = _repo_path(args.network_target)
    owid_csv = _repo_path(args.owid_csv)
    base_config = _repo_path(args.base_config)
    overlay_config = _repo_path(args.overlay_config)
    last_good_overlay_config = _repo_path(args.last_good_overlay_config)
    history_csv = _repo_path(args.history_csv)
    country_detail_csv = _repo_path(args.country_detail_csv)

    if not owid_csv.exists():
        raise FileNotFoundError(f"OWID CSV not found: {owid_csv}")
    if not base_config.exists():
        raise FileNotFoundError(f"Base config not found: {base_config}")

    ref_biofuel_twh = _load_owid_biofuel_reference(owid_csv, args.year)
    if ref_biofuel_twh.empty:
        raise RuntimeError(f"No OWID biofuel_electricity reference found for year {args.year} in {owid_csv}")
    targets = pd.DataFrame({"target_biofuel_twh": ref_biofuel_twh.clip(lower=0.0)})

    base_cfg = _yaml_load(base_config)
    base_solid = float(_get_nested(base_cfg, ["sector", "solid_biomass_potential"], 100.0))
    base_biogas = float(_get_nested(base_cfg, ["sector", "biogas_potential"], 40.0))

    factor_solid = 1.0
    factor_biogas = 1.0
    if overlay_config.exists():
        try:
            ov = _yaml_load(overlay_config)
            ov_solid = _get_nested(ov, ["sector", "solid_biomass_potential"], None)
            ov_biogas = _get_nested(ov, ["sector", "biogas_potential"], None)
            if ov_solid is not None:
                factor_solid = float(ov_solid) / max(base_solid, 1e-9)
            if ov_biogas is not None:
                factor_biogas = float(ov_biogas) / max(base_biogas, 1e-9)
            LOG.info("Loaded current biomass overlay %s (solid=%.3fx, biogas=%.3fx)", overlay_config, factor_solid, factor_biogas)
        except Exception as exc:
            LOG.warning("Could not parse existing overlay %s: %s", overlay_config, exc)

    history_csv.parent.mkdir(parents=True, exist_ok=True)
    country_detail_csv.parent.mkdir(parents=True, exist_ok=True)
    last_good_overlay_config.parent.mkdir(parents=True, exist_ok=True)

    if not sync_last_good_from_mutable(overlay_config, last_good_overlay_config):
        _write_biomass_overlay(
            last_good_overlay_config,
            base_solid * factor_solid,
            base_biogas * factor_biogas,
        )
        LOG.info("Seeded last-good biomass overlay: %s", last_good_overlay_config)
    else:
        LOG.info(
            "Seeded last-good biomass overlay from current mutable file: %s",
            last_good_overlay_config,
        )

    if not network_target.exists():
        LOG.info("Initial solved network missing, running first solve.")
        _write_biomass_overlay(overlay_config, base_solid * factor_solid, base_biogas * factor_biogas)
        _run_snakemake(args)

    history_rows = []
    detail_last = None
    for i in range(1, args.iterations + 1):
        if not network_target.exists():
            raise FileNotFoundError(f"Solved network not found after solve: {network_target}")

        model_biofuel, stores_diag = _load_model_biofuel_electricity_by_country(network_target)
        metrics, detail = _compute_metrics(model_biofuel, targets, args.country_min_ref_twh)
        detail_last = detail.reset_index().rename(columns={"index": "country"})

        row = {"iteration": i}
        row.update(metrics)
        row.update(stores_diag)
        row["solid_biomass_potential_twh"] = base_solid * factor_solid
        row["biogas_potential_twh"] = base_biogas * factor_biogas
        row["solid_biomass_factor"] = factor_solid
        row["biogas_factor"] = factor_biogas
        if pd.notna(row.get("solid_biomass_store_capacity_twh")) and row["solid_biomass_store_capacity_twh"] > 0:
            row["solid_biomass_store_utilization_pct"] = 100.0 * row.get("solid_biomass_store_use_twh", np.nan) / row["solid_biomass_store_capacity_twh"]
        else:
            row["solid_biomass_store_utilization_pct"] = np.nan
        if pd.notna(row.get("biogas_store_capacity_twh")) and row["biogas_store_capacity_twh"] > 0:
            row["biogas_store_utilization_pct"] = 100.0 * row.get("biogas_store_use_twh", np.nan) / row["biogas_store_capacity_twh"]
        else:
            row["biogas_store_utilization_pct"] = np.nan
        history_rows.append(row)

        LOG.info(
            "Iteration %d: biofuel=%.1f/%.1f TWh (global_err=%.2f%%), country_wape=%.2f%%, solid_biomass_potential=%.1f TWh, biogas_potential=%.1f TWh, solid_use=%.1f TWh, biogas_use=%.1f TWh, biogas_to_gas=%.1f TWh",
            i,
            metrics["biofuel_model_twh"],
            metrics["biofuel_target_twh"],
            metrics["global_abs_pct"],
            metrics["country_wape_pct"],
            row["solid_biomass_potential_twh"],
            row["biogas_potential_twh"],
            row.get("solid_biomass_store_use_twh", np.nan),
            row.get("biogas_store_use_twh", np.nan),
            row.get("biogas_to_gas_twh", np.nan),
        )

        global_ok = np.isnan(metrics["global_abs_pct"]) or metrics["global_abs_pct"] <= float(args.target_global_error_pct)
        country_ok = (
            True
            if args.target_country_wape_pct is None
            else (np.isnan(metrics["country_wape_pct"]) or metrics["country_wape_pct"] <= float(args.target_country_wape_pct))
        )
        if global_ok and country_ok:
            LOG.info("Converged at iteration %d.", i)
            break

        adj = _compute_adjustment(
            target=metrics["biofuel_target_twh"],
            model=metrics["biofuel_model_twh"],
            min_adjust=args.min_adjust,
            max_adjust=args.max_adjust,
        )
        adj = float(np.power(adj, args.adjust_alpha))
        factor_solid = float(np.clip(factor_solid * adj, args.min_factor, args.max_factor))
        if args.mode == "joint":
            factor_biogas = float(np.clip(factor_biogas * adj, args.min_factor, args.max_factor))

        _write_biomass_overlay(overlay_config, base_solid * factor_solid, base_biogas * factor_biogas)
        LOG.info(
            "Wrote biomass overlay %s: solid_biomass_potential=%.3f TWh, biogas_potential=%.3f TWh",
            overlay_config,
            base_solid * factor_solid,
            base_biogas * factor_biogas,
        )
        try:
            raise_if_simulated_failure(args.simulate_post_write_failure, "biomass tuner")
            _run_snakemake(args)
        except Exception as exc:
            restored = restore_from_last_good(overlay_config, last_good_overlay_config)
            LOG.error(
                "Biomass solve/update failed after writing overlay. Rolled back to last good overlay at %s (restored=%s, snapshot=%s). Error: %s",
                overlay_config,
                restored,
                last_good_overlay_config,
                exc,
            )
            raise
        sync_last_good_from_mutable(overlay_config, last_good_overlay_config)

    history = pd.DataFrame(history_rows)
    history.to_csv(history_csv, index=False, float_format="%.4f")
    if detail_last is not None:
        detail_last.to_csv(country_detail_csv, index=False, float_format="%.4f")
    LOG.info("Saved iteration history to %s", history_csv)
    LOG.info("Saved last country detail to %s", country_detail_csv)


if __name__ == "__main__":
    main()
