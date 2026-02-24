#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iterative country-level hydro scaling loop for baseyear calibration.

Hydro is calibrated jointly (reservoir + run-of-river) because OWID does not
provide a split. The prepare hook applies one per-country factor to:
- StorageUnit carrier='hydro' inflow time series
- Generator carrier='ror' p_max_pu availability
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


def _load_owid_hydro_reference(owid_csv, year):
    usecols = {"year", "iso_code", "hydro_electricity"}
    owid = pd.read_csv(owid_csv, usecols=lambda c: c in usecols)
    owid = owid.loc[owid["year"] == int(year)].copy()
    owid["country"] = owid["iso_code"].apply(_safe_iso3_to_iso2)
    owid = owid.loc[owid["country"].notna()].copy()
    owid["hydro_electricity"] = pd.to_numeric(owid["hydro_electricity"], errors="coerce").fillna(0.0)
    return owid.groupby("country")["hydro_electricity"].sum(min_count=1).sort_index()


def _load_model_hydro_by_country(network_path):
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

    res = (
        eb.loc[(eb.component == "StorageUnit") & (eb.carrier == "hydro")]
        .groupby("country")["energy_mwh"]
        .sum()
        / 1e6
    )
    ror = (
        eb.loc[(eb.component == "Generator") & (eb.carrier == "ror")]
        .groupby("country")["energy_mwh"]
        .sum()
        / 1e6
    )
    out = pd.DataFrame({"reservoir_twh": res, "ror_twh": ror}).fillna(0.0)
    out["hydro_twh"] = out["reservoir_twh"] + out["ror_twh"]
    return out.sort_index()


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
            float(target_df.at[c, "target_hydro_twh"]),
            float(model.at[c, "hydro_twh"]),
            min_adjust,
            max_adjust,
        )

    scales["hydro_scale"] = (
        scales["hydro_scale"] * np.power(adj, alpha)
    ).clip(lower=min_total_scale, upper=max_total_scale)
    return scales


def _compute_metrics(model_df, target_df, min_ref_country_twh):
    comp = target_df.join(model_df[["hydro_twh", "reservoir_twh", "ror_twh"]], how="left").fillna(0.0)
    comp["error_twh"] = comp["hydro_twh"] - comp["target_hydro_twh"]
    comp["abs_error_twh"] = comp["error_twh"].abs()
    comp["ape_pct"] = np.where(
        comp["target_hydro_twh"] > 0,
        100.0 * comp["abs_error_twh"] / comp["target_hydro_twh"],
        np.nan,
    )

    valid = comp["target_hydro_twh"] >= float(min_ref_country_twh)
    comp_valid = comp.loc[valid].copy()
    if comp_valid.empty:
        country_wape = np.nan
    else:
        denom = comp_valid["target_hydro_twh"].sum()
        country_wape = 100.0 * comp_valid["abs_error_twh"].sum() / denom if denom > 0 else np.nan

    target_total = float(comp["target_hydro_twh"].sum())
    model_total = float(comp["hydro_twh"].sum())
    return {
        "country_wape_pct": country_wape,
        "hydro_model_twh": model_total,
        "hydro_target_twh": target_total,
        "global_abs_pct": (100.0 * abs(model_total - target_total) / target_total) if target_total > 0 else np.nan,
        "countries_eval": int(valid.sum()),
        "reservoir_model_twh": float(comp["reservoir_twh"].sum()),
        "ror_model_twh": float(comp["ror_twh"].sum()),
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


def _write_scales_csv(path: Path, scales: pd.DataFrame) -> pd.DataFrame:
    out = scales.reset_index().rename(columns={"index": "country"}).sort_values("country")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, float_format="%.6f")
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Iterative hydro scaling calibration loop (reservoir+ror jointly).")
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
    p.add_argument("--override-csv", default="validation/data/hydro_iteration_scaling_overrides.csv")
    p.add_argument("--history-csv", default="validation/results_compare/hydro_iteration_history.csv")
    p.add_argument(
        "--country-detail-csv",
        default="validation/results_compare/hydro_iteration_country_detail.csv",
    )
    p.add_argument("--country-min-ref-twh", type=float, default=1.0)
    p.add_argument("--target-country-ape-pct", type=float, default=10.0)
    p.add_argument("--target-country-success-share-pct", type=float, default=85.0)
    p.add_argument("--target-country-wape-pct", type=float, default=None)
    p.add_argument("--target-global-error-pct", type=float, default=3.0)
    p.add_argument("--adjust-alpha", type=float, default=0.7)
    p.add_argument("--min-adjust", type=float, default=0.25)
    p.add_argument("--max-adjust", type=float, default=4.0)
    p.add_argument("--min-total-scale", type=float, default=0.05)
    p.add_argument("--max-total-scale", type=float, default=20.0)
    p.add_argument(
        "--max-countries-at-max-scale",
        type=int,
        default=10,
        help="Safety stop: abort update if more than this many countries hit max-total-scale.",
    )
    p.add_argument(
        "--last-good-override-csv",
        default="validation/data/hydro_iteration_scaling_overrides_last_good.csv",
        help="Rollback target written after successful solves; restored if a solve fails.",
    )
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
    override_csv = _repo_path(args.override_csv)
    last_good_override_csv = _repo_path(args.last_good_override_csv)
    history_csv = _repo_path(args.history_csv)
    country_detail_csv = _repo_path(args.country_detail_csv)

    if not owid_csv.exists():
        raise FileNotFoundError(f"OWID CSV not found: {owid_csv}")

    ref_hydro_twh = _load_owid_hydro_reference(owid_csv, args.year)
    if ref_hydro_twh.empty:
        raise RuntimeError(f"No OWID hydro reference found for year {args.year} in {owid_csv}")

    override_csv.parent.mkdir(parents=True, exist_ok=True)
    last_good_override_csv.parent.mkdir(parents=True, exist_ok=True)
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    country_detail_csv.parent.mkdir(parents=True, exist_ok=True)

    if not network_target.exists():
        LOG.info("Initial solved network missing, running first solve.")
        _run_snakemake(args)

    history_rows = []
    detail_last = None
    scales = pd.DataFrame(index=ref_hydro_twh.index, data={"hydro_scale": 1.0})
    if override_csv.exists():
        try:
            prev = pd.read_csv(override_csv)
            prev["country"] = prev["country"].astype(str).str.upper().str.strip()
            prev = prev.set_index("country")
            if "hydro_scale" in prev.columns:
                scales["hydro_scale"] = (
                    pd.to_numeric(prev["hydro_scale"], errors="coerce")
                    .reindex(scales.index)
                    .fillna(scales["hydro_scale"])
                )
            LOG.info("Loaded existing override scales from %s", override_csv)
        except Exception as exc:
            LOG.warning("Could not load existing overrides from %s: %s", override_csv, exc)
    # Seed rollback file from current known-good scales (the solved network we start from).
    if not sync_last_good_from_mutable(override_csv, last_good_override_csv):
        _write_scales_csv(last_good_override_csv, scales)
        LOG.info("Seeded last-good hydro overrides: %s", last_good_override_csv)
    else:
        LOG.info("Seeded last-good hydro overrides from current mutable file: %s", last_good_override_csv)

    targets = pd.DataFrame({"target_hydro_twh": ref_hydro_twh.clip(lower=0.0)})

    for i in range(1, args.iterations + 1):
        if not network_target.exists():
            raise FileNotFoundError(f"Solved network not found after solve: {network_target}")

        model_hydro = _load_model_hydro_by_country(network_target)
        metrics, detail = _compute_metrics(model_hydro, targets, args.country_min_ref_twh)
        eval_detail = detail.loc[detail["target_hydro_twh"] >= float(args.country_min_ref_twh)].copy()
        within_mask = eval_detail["ape_pct"] <= float(args.target_country_ape_pct)
        countries_within = int(within_mask.sum())
        countries_eval = int(len(eval_detail))
        country_success_share_pct = 100.0 * countries_within / countries_eval if countries_eval > 0 else np.nan
        detail_last = detail.reset_index().rename(columns={"index": "country"})

        row = {"iteration": i}
        row.update(metrics)
        row["countries_within_target"] = countries_within
        row["country_success_share_pct"] = country_success_share_pct
        history_rows.append(row)

        LOG.info(
            "Iteration %d: country_wape=%.2f%%, within_%s%%=%d/%d (%.1f%%), hydro=%.1f/%.1f TWh, global_err=%.2f%%, reservoir=%.1f TWh, ror=%.1f TWh",
            i,
            metrics["country_wape_pct"],
            int(args.target_country_ape_pct),
            countries_within,
            countries_eval,
            country_success_share_pct,
            metrics["hydro_model_twh"],
            metrics["hydro_target_twh"],
            metrics["global_abs_pct"],
            metrics["reservoir_model_twh"],
            metrics["ror_model_twh"],
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

        current_good_scales = scales.copy()
        proposed_scales = _update_scales(
            current_scales=scales,
            model_df=model_hydro,
            target_df=targets,
            alpha=args.adjust_alpha,
            min_adjust=args.min_adjust,
            max_adjust=args.max_adjust,
            min_total_scale=args.min_total_scale,
            max_total_scale=args.max_total_scale,
        )
        cap_hits = int(np.isclose(proposed_scales["hydro_scale"], float(args.max_total_scale), atol=1e-12).sum())
        if cap_hits > int(args.max_countries_at_max_scale):
            # Keep the current good overrides on disk and abort before triggering an unstable solve.
            _write_scales_csv(override_csv, current_good_scales)
            _write_scales_csv(last_good_override_csv, current_good_scales)
            msg = (
                f"Hydro safety stop before iteration {i+1 if i < args.iterations else i}: "
                f"{cap_hits} countries hit max_total_scale={args.max_total_scale:.3f} "
                f"(limit={args.max_countries_at_max_scale}). "
                f"Restored current good overrides to {override_csv}."
            )
            LOG.error(msg)
            raise RuntimeError(msg)

        scales = proposed_scales
        _write_scales_csv(override_csv, scales)
        LOG.info(
            "Wrote overrides: %s (countries_at_max_scale=%d, max_total_scale=%.3f)",
            override_csv,
            cap_hits,
            float(args.max_total_scale),
        )

        try:
            raise_if_simulated_failure(args.simulate_post_write_failure, "hydro tuner")
            _run_snakemake(args)
        except Exception as exc:
            restored = restore_from_last_good(override_csv, last_good_override_csv)
            LOG.error(
                "Hydro solve/update failed after writing new overrides. Rolled back to last good hydro overrides at %s (restored=%s, snapshot=%s). Error: %s",
                override_csv,
                restored,
                last_good_override_csv,
                exc,
            )
            raise

        # Solve succeeded; persist this as the new rollback point.
        sync_last_good_from_mutable(override_csv, last_good_override_csv)

    history = pd.DataFrame(history_rows)
    history.to_csv(history_csv, index=False, float_format="%.4f")
    if detail_last is not None:
        detail_last.to_csv(country_detail_csv, index=False, float_format="%.4f")
    LOG.info("Saved iteration history to %s", history_csv)
    LOG.info("Saved last country detail to %s", country_detail_csv)


if __name__ == "__main__":
    main()
