#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iterative country-level nuclear availability scaling loop for baseyear calibration.

The prepare hook applies per-country multipliers to nuclear generator p_max_pu
(and clips to the configured cap, typically 1.0).
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

LOG = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _helpers import three_2_two_digits_country  # noqa: E402


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


def _load_owid_nuclear_reference(owid_csv, year):
    usecols = {"year", "iso_code", "nuclear_electricity"}
    owid = pd.read_csv(owid_csv, usecols=lambda c: c in usecols)
    owid = owid.loc[owid["year"] == int(year)].copy()
    owid["country"] = owid["iso_code"].apply(_safe_iso3_to_iso2)
    owid = owid.loc[owid["country"].notna()].copy()
    owid["nuclear_electricity"] = pd.to_numeric(owid["nuclear_electricity"], errors="coerce").fillna(0.0)
    return owid.groupby("country")["nuclear_electricity"].sum(min_count=1).sort_index()


def _load_model_nuclear_by_country(network_path):
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

    # In standard runs nuclear is a Generator carrier='nuclear'. In hydrogen-colors
    # runs electricity may appear on a Link carrier='nuclear-to-grid'. Count both.
    nuclear = (
        eb.loc[
            ((eb.component == "Generator") & (eb.carrier == "nuclear"))
            | ((eb.component == "Link") & (eb.carrier == "nuclear-to-grid"))
        ]
        .groupby("country")["energy_mwh"]
        .sum()
        / 1e6
    )
    return pd.DataFrame({"nuclear_twh": nuclear}).fillna(0.0).sort_index()


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


def _update_scales(
    current_scales,
    model_df,
    target_df,
    alpha,
    min_adjust,
    max_adjust,
    min_total_scale,
    max_total_scale,
):
    countries = target_df.index
    scales = current_scales.reindex(countries).fillna(1.0).copy()
    model = model_df.reindex(countries).fillna(0.0)

    adj = pd.Series(index=countries, dtype=float)
    for c in countries:
        adj.loc[c] = _compute_adjustment(
            float(target_df.at[c, "target_nuclear_twh"]),
            float(model.at[c, "nuclear_twh"]),
            min_adjust,
            max_adjust,
        )

    scales["nuclear_scale"] = (
        scales["nuclear_scale"] * np.power(adj, alpha)
    ).clip(lower=min_total_scale, upper=max_total_scale)
    return scales


def _compute_metrics(model_df, target_df, min_ref_country_twh):
    comp = target_df.join(model_df[["nuclear_twh"]], how="left").fillna(0.0)
    comp["error_twh"] = comp["nuclear_twh"] - comp["target_nuclear_twh"]
    comp["abs_error_twh"] = comp["error_twh"].abs()
    comp["ape_pct"] = np.where(
        comp["target_nuclear_twh"] > 0,
        100.0 * comp["abs_error_twh"] / comp["target_nuclear_twh"],
        np.nan,
    )

    valid = comp["target_nuclear_twh"] >= float(min_ref_country_twh)
    comp_valid = comp.loc[valid].copy()
    if comp_valid.empty:
        country_wape = np.nan
    else:
        denom = comp_valid["target_nuclear_twh"].sum()
        country_wape = 100.0 * comp_valid["abs_error_twh"].sum() / denom if denom > 0 else np.nan

    target_total = float(comp["target_nuclear_twh"].sum())
    model_total = float(comp["nuclear_twh"].sum())
    return {
        "country_wape_pct": country_wape,
        "nuclear_model_twh": model_total,
        "nuclear_target_twh": target_total,
        "global_abs_pct": (100.0 * abs(model_total - target_total) / target_total) if target_total > 0 else np.nan,
        "countries_eval": int(valid.sum()),
    }, comp


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


def parse_args():
    p = argparse.ArgumentParser(description="Iterative nuclear p_max_pu scaling calibration loop.")
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
    p.add_argument("--owid-csv", default="validation/data/owid-energy-data.csv")
    p.add_argument("--year", type=int, default=2020)
    p.add_argument("--iterations", type=int, default=6)
    p.add_argument("--override-csv", default="validation/data/nuclear_iteration_scaling_overrides.csv")
    p.add_argument("--history-csv", default="validation/results_compare/nuclear_iteration_history.csv")
    p.add_argument(
        "--country-detail-csv",
        default="validation/results_compare/nuclear_iteration_country_detail.csv",
    )
    p.add_argument("--country-min-ref-twh", type=float, default=1.0)
    p.add_argument("--target-country-ape-pct", type=float, default=10.0)
    p.add_argument("--target-country-success-share-pct", type=float, default=85.0)
    p.add_argument("--target-country-wape-pct", type=float, default=None)
    p.add_argument("--target-global-error-pct", type=float, default=3.0)
    p.add_argument("--adjust-alpha", type=float, default=0.5)
    p.add_argument("--min-adjust", type=float, default=0.5)
    p.add_argument("--max-adjust", type=float, default=2.0)
    p.add_argument("--min-total-scale", type=float, default=0.05)
    p.add_argument("--max-total-scale", type=float, default=20.0)
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
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    network_target = _repo_path(args.network_target)
    owid_csv = _repo_path(args.owid_csv)
    override_csv = _repo_path(args.override_csv)
    history_csv = _repo_path(args.history_csv)
    country_detail_csv = _repo_path(args.country_detail_csv)

    if not owid_csv.exists():
        raise FileNotFoundError(f"OWID CSV not found: {owid_csv}")

    ref_nuclear_twh = _load_owid_nuclear_reference(owid_csv, args.year)
    if ref_nuclear_twh.empty:
        raise RuntimeError(f"No OWID nuclear reference found for year {args.year} in {owid_csv}")

    override_csv.parent.mkdir(parents=True, exist_ok=True)
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    country_detail_csv.parent.mkdir(parents=True, exist_ok=True)

    if not network_target.exists():
        LOG.info("Initial solved network missing, running first solve.")
        _run_snakemake(args)

    history_rows = []
    detail_last = None
    scales = pd.DataFrame(index=ref_nuclear_twh.index, data={"nuclear_scale": 1.0})
    if override_csv.exists():
        try:
            prev = pd.read_csv(override_csv)
            prev["country"] = prev["country"].astype(str).str.upper().str.strip()
            prev = prev.set_index("country")
            if "nuclear_scale" in prev.columns:
                scales["nuclear_scale"] = (
                    pd.to_numeric(prev["nuclear_scale"], errors="coerce")
                    .reindex(scales.index)
                    .fillna(scales["nuclear_scale"])
                )
            LOG.info("Loaded existing override scales from %s", override_csv)
        except Exception as exc:
            LOG.warning("Could not load existing overrides from %s: %s", override_csv, exc)

    targets = pd.DataFrame({"target_nuclear_twh": ref_nuclear_twh.clip(lower=0.0)})

    for i in range(1, args.iterations + 1):
        if not network_target.exists():
            raise FileNotFoundError(f"Solved network not found after solve: {network_target}")

        model_nuclear = _load_model_nuclear_by_country(network_target)
        metrics, detail = _compute_metrics(model_nuclear, targets, args.country_min_ref_twh)
        detail = detail.join(scales, how="left")
        eval_detail = detail.loc[detail["target_nuclear_twh"] >= float(args.country_min_ref_twh)].copy()
        within_mask = eval_detail["ape_pct"] <= float(args.target_country_ape_pct)
        countries_within = int(within_mask.sum())
        countries_eval = int(len(eval_detail))
        country_success_share_pct = 100.0 * countries_within / countries_eval if countries_eval > 0 else np.nan
        detail_last = detail.reset_index().rename(columns={"index": "country"})

        row = {"iteration": i}
        row.update(metrics)
        row["countries_within_target"] = countries_within
        row["country_success_share_pct"] = country_success_share_pct
        row["scale_min_current"] = float(detail["nuclear_scale"].min()) if len(detail) else np.nan
        row["scale_max_current"] = float(detail["nuclear_scale"].max()) if len(detail) else np.nan
        history_rows.append(row)

        LOG.info(
            "Iteration %d: country_wape=%.2f%%, within_%s%%=%d/%d (%.1f%%), nuclear=%.1f/%.1f TWh, global_err=%.2f%%, scale_range=[%.3f, %.3f]",
            i,
            metrics["country_wape_pct"],
            int(args.target_country_ape_pct),
            countries_within,
            countries_eval,
            country_success_share_pct,
            metrics["nuclear_model_twh"],
            metrics["nuclear_target_twh"],
            metrics["global_abs_pct"],
            row["scale_min_current"],
            row["scale_max_current"],
        )

        country_share_ok = (
            np.isnan(country_success_share_pct)
            or country_success_share_pct >= float(args.target_country_success_share_pct)
        )
        country_wape_ok = (
            True
            if args.target_country_wape_pct is None
            else (
                np.isnan(metrics["country_wape_pct"])
                or metrics["country_wape_pct"] <= float(args.target_country_wape_pct)
            )
        )
        global_ok = (
            np.isnan(metrics["global_abs_pct"])
            or metrics["global_abs_pct"] <= float(args.target_global_error_pct)
        )
        converged = country_share_ok and country_wape_ok and global_ok
        if converged:
            LOG.info("Converged at iteration %d.", i)
            break

        scales = _update_scales(
            current_scales=scales,
            model_df=model_nuclear,
            target_df=targets,
            alpha=args.adjust_alpha,
            min_adjust=args.min_adjust,
            max_adjust=args.max_adjust,
            min_total_scale=args.min_total_scale,
            max_total_scale=args.max_total_scale,
        )
        scales_out = scales.reset_index().rename(columns={"index": "country"}).sort_values("country")
        scales_out.to_csv(override_csv, index=False, float_format="%.6f")
        LOG.info("Wrote overrides: %s", override_csv)

        _run_snakemake(args)

    history = pd.DataFrame(history_rows)
    history.to_csv(history_csv, index=False, float_format="%.4f")
    if detail_last is not None:
        detail_last.to_csv(country_detail_csv, index=False, float_format="%.4f")
    LOG.info("Saved iteration history to %s", history_csv)
    LOG.info("Saved last country detail to %s", country_detail_csv)


if __name__ == "__main__":
    main()
