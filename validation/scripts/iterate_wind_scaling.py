#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iterative country-level wind scaling loop for baseyear calibration.

Loop:
1) Read latest solved network and compute country on/off/total wind generation.
2) Build country targets from OWID total wind plus optional global on/off split.
3) Update per-country onwind/offwind p_max_pu scaling multipliers.
4) Write override CSV consumed by prepare_sector_network.
5) Force rerun prepare->solve chain via Snakemake.
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


def _load_owid_wind_reference(owid_csv, year):
    usecols = {"year", "iso_code", "wind_electricity"}
    owid = pd.read_csv(owid_csv, usecols=lambda c: c in usecols)
    owid = owid.loc[owid["year"] == int(year)].copy()
    owid["country"] = owid["iso_code"].apply(_safe_iso3_to_iso2)
    owid = owid.loc[owid["country"].notna()].copy()
    owid["wind_electricity"] = pd.to_numeric(owid["wind_electricity"], errors="coerce").fillna(0.0)
    return owid.groupby("country")["wind_electricity"].sum(min_count=1).sort_index()


def _load_model_wind_by_country(network_path):
    n = pypsa.Network(network_path)
    eb = n.statistics.energy_balance(
        bus_carrier="AC",
        aggregate_time="sum",
        aggregate_groups="sum",
        groupby=n.statistics.groupers.get_country_and_carrier,
        nice_names=False,
    )
    eb = eb.rename("energy_mwh").reset_index()
    eb = eb.loc[(eb.country != "") & (eb.energy_mwh > 0) & (eb.component == "Generator")].copy()

    on = eb.loc[eb.carrier == "onwind"].groupby("country")["energy_mwh"].sum() / 1e6
    off = eb.loc[eb.carrier.isin(["offwind-ac", "offwind-dc"])].groupby("country")["energy_mwh"].sum() / 1e6

    cap = n.generators.copy()
    bus_country = n.buses["country"].replace("", np.nan)

    on_cap = pd.Series(dtype=float)
    on_cap_df = cap.loc[cap.carrier.eq("onwind")]
    if len(on_cap_df) > 0:
        on_country = on_cap_df.bus.map(bus_country).fillna("")
        on_cap_df = on_cap_df.loc[on_country != ""].copy()
        on_country = on_country.loc[on_cap_df.index]
        on_cap = pd.to_numeric(on_cap_df.get("p_nom_opt", on_cap_df["p_nom"]), errors="coerce").fillna(0.0)
        on_cap = on_cap.groupby(on_country).sum()

    cap = cap.loc[cap.carrier.isin(["offwind-ac", "offwind-dc"])]
    if len(cap) > 0:
        cap_country = cap.bus.map(bus_country).fillna("")
        cap = cap.loc[cap_country != ""].copy()
        cap_country = cap_country.loc[cap.index]
        off_cap = pd.to_numeric(cap.get("p_nom_opt", cap["p_nom"]), errors="coerce").fillna(0.0)
        off_cap = off_cap.groupby(cap_country).sum()
    else:
        off_cap = pd.Series(dtype=float)

    out = pd.DataFrame({"on_twh": on, "off_twh": off, "on_cap_mw": on_cap, "off_cap_mw": off_cap}).fillna(0.0)
    out["wind_total_twh"] = out["on_twh"] + out["off_twh"]
    return out.sort_index()


def _allocate_capped_proportional(total, weights, caps):
    caps = caps.clip(lower=0.0).fillna(0.0)
    total = float(np.clip(total, 0.0, caps.sum()))
    alloc = pd.Series(0.0, index=caps.index, dtype=float)
    if total <= 0 or caps.sum() <= 0:
        return alloc

    weights = weights.reindex(caps.index).fillna(0.0).clip(lower=0.0)
    if weights.sum() <= 0:
        weights = caps.gt(0.0).astype(float)

    remaining = total
    active = caps.gt(0.0)

    for _ in range(len(caps) + 3):
        if remaining <= 1e-9 or not active.any():
            break
        w = weights.loc[active]
        if w.sum() <= 0:
            w = pd.Series(1.0, index=w.index)
        proposal = remaining * (w / w.sum())
        cap_left = (caps - alloc).loc[active]
        hit_cap = proposal >= (cap_left - 1e-12)
        if hit_cap.any():
            hit_idx = hit_cap.index[hit_cap.values]
            alloc.loc[hit_idx] += cap_left.loc[hit_idx]
            active.loc[hit_idx] = False
            remaining = total - alloc.sum()
        else:
            alloc.loc[proposal.index] += proposal
            remaining = 0.0
            break

    if remaining > 1e-8 and active.any():
        add = remaining / active.sum()
        alloc.loc[active] += np.minimum(add, (caps - alloc).loc[active])

    return alloc.clip(lower=0.0, upper=caps)


def _build_country_targets(ref_wind_twh, model_df, on_global_target, off_global_target, mode):
    countries = ref_wind_twh.index
    model = model_df.reindex(countries).fillna(0.0)
    ref_total = float(ref_wind_twh.sum())

    if on_global_target is None or off_global_target is None:
        # Fallback: preserve current global split while matching country wind totals.
        on_global_target = float(model["on_twh"].sum())
        off_global_target = float(model["off_twh"].sum())

    split_sum = on_global_target + off_global_target
    if split_sum <= 0:
        on_global_target = ref_total
        off_global_target = 0.0
    elif ref_total > 0 and abs(split_sum - ref_total) > 1e-9:
        scale = ref_total / split_sum
        on_global_target *= scale
        off_global_target *= scale

    caps = ref_wind_twh.clip(lower=0.0)
    if mode == "off_capacity":
        weights = model["off_cap_mw"]
    elif mode == "equal":
        weights = caps.gt(0).astype(float)
    else:
        weights = model["off_twh"] + 1e-6 * model["off_cap_mw"].gt(0).astype(float)

    off_target = _allocate_capped_proportional(off_global_target, weights, caps)
    on_target = (caps - off_target).clip(lower=0.0)
    return pd.DataFrame(
        {
            "ref_wind_twh": caps,
            "target_on_twh": on_target,
            "target_off_twh": off_target,
        }
    )


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
    min_update_capacity_mw,
    min_update_model_twh,
):
    countries = target_df.index
    scales = current_scales.reindex(countries).fillna(1.0).copy()
    model = model_df.reindex(countries).fillna(0.0)

    on_adj = []
    off_adj = []
    for c in countries:
        on_ratio = _compute_adjustment(
            float(target_df.at[c, "target_on_twh"]),
            float(model.at[c, "on_twh"]),
            min_adjust,
            max_adjust,
        )
        off_ratio = _compute_adjustment(
            float(target_df.at[c, "target_off_twh"]),
            float(model.at[c, "off_twh"]),
            min_adjust,
            max_adjust,
        )
        on_adj.append(on_ratio)
        off_adj.append(off_ratio)

    on_adj = pd.Series(on_adj, index=countries, dtype=float)
    off_adj = pd.Series(off_adj, index=countries, dtype=float)

    # If there is no modeled capacity, or the model has effectively zero dispatch,
    # a profile multiplier cannot calibrate the country. Reset to neutral instead
    # of creating unphysical CFs that can distort later runs.
    on_guard = (
        (target_df["target_on_twh"] > 0.0)
        & (
            (model["on_cap_mw"] <= min_update_capacity_mw)
            | (model["on_twh"] <= min_update_model_twh)
        )
    )
    off_guard = (
        (target_df["target_off_twh"] > 0.0)
        & (
            (model["off_cap_mw"] <= min_update_capacity_mw)
            | (model["off_twh"] <= min_update_model_twh)
        )
    )
    scales.loc[on_guard, "onwind_scale"] = 1.0
    scales.loc[off_guard, "offwind_scale"] = 1.0
    on_adj.loc[on_guard] = 1.0
    off_adj.loc[off_guard] = 1.0

    # Damped multiplicative update.
    scales["onwind_scale"] = (
        scales["onwind_scale"].pow(1.0)
        * np.power(on_adj, alpha)
    ).clip(lower=min_total_scale, upper=max_total_scale)
    scales["offwind_scale"] = (
        scales["offwind_scale"].pow(1.0)
        * np.power(off_adj, alpha)
    ).clip(lower=min_total_scale, upper=max_total_scale)
    return scales


def _compute_metrics(model_df, target_df, min_ref_country_twh):
    comp = target_df.join(model_df[["on_twh", "off_twh", "wind_total_twh"]], how="left").fillna(0.0)
    comp["target_total_twh"] = comp["ref_wind_twh"]
    comp["error_total_twh"] = comp["wind_total_twh"] - comp["target_total_twh"]
    comp["abs_error_total_twh"] = comp["error_total_twh"].abs()
    comp["ape_total_pct"] = np.where(
        comp["target_total_twh"] > 0,
        100.0 * comp["abs_error_total_twh"] / comp["target_total_twh"],
        np.nan,
    )

    valid = comp["target_total_twh"] >= float(min_ref_country_twh)
    comp_valid = comp.loc[valid].copy()
    if comp_valid.empty:
        country_wape = np.nan
    else:
        denom = comp_valid["target_total_twh"].sum()
        country_wape = 100.0 * comp_valid["abs_error_total_twh"].sum() / denom if denom > 0 else np.nan

    on_target = float(comp["target_on_twh"].sum())
    off_target = float(comp["target_off_twh"].sum())
    on_model = float(comp["on_twh"].sum())
    off_model = float(comp["off_twh"].sum())

    on_split_err_pct = 100.0 * abs(on_model - on_target) / on_target if on_target > 0 else np.nan
    off_split_err_pct = 100.0 * abs(off_model - off_target) / off_target if off_target > 0 else np.nan

    return {
        "country_wape_pct": country_wape,
        "on_model_twh": on_model,
        "on_target_twh": on_target,
        "off_model_twh": off_model,
        "off_target_twh": off_target,
        "on_split_abs_pct": on_split_err_pct,
        "off_split_abs_pct": off_split_err_pct,
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
    p = argparse.ArgumentParser(description="Iterative wind scaling calibration loop.")
    p.add_argument(
        "--network-target",
        default="results/Global_200/postnetworks/elec_s_200_ec_lcopt_1h_1h_2020_0.071_AB_0.0export_base.nc",
        help="Snakemake output target for solved network.",
    )
    p.add_argument(
        "--configfiles",
        nargs="+",
        default=[
            "config.myopic.yaml",
            "config.learning.yaml",
            "validation/config.iteration_common.yaml",
        ],
        help="Config files passed to Snakemake in order.",
    )
    p.add_argument("--owid-csv", default="validation/data/owid-energy-data.csv")
    p.add_argument("--year", type=int, default=2020)
    p.add_argument("--iterations", type=int, default=6, help="Maximum calibration updates.")
    p.add_argument(
        "--override-csv",
        default="validation/data/wind_iteration_scaling_overrides.csv",
        help="Country scaling override file consumed by prepare_sector_network.",
    )
    p.add_argument(
        "--history-csv",
        default="validation/results_compare/wind_iteration_history.csv",
        help="Per-iteration history output CSV.",
    )
    p.add_argument(
        "--country-detail-csv",
        default="validation/results_compare/wind_iteration_country_detail.csv",
        help="Last iteration country detail output CSV.",
    )
    p.add_argument(
        "--last-good-override-csv",
        default="validation/data/wind_iteration_scaling_overrides_last_good.csv",
        help="Rollback snapshot of the last known-good wind override CSV.",
    )
    p.add_argument("--onshore-global-target-twh", type=float, default=None)
    p.add_argument("--offshore-global-target-twh", type=float, default=None)
    p.add_argument(
        "--offshore-weight-mode",
        choices=["off_generation", "off_capacity", "equal"],
        default="off_generation",
    )
    p.add_argument("--country-min-ref-twh", type=float, default=1.0)
    p.add_argument(
        "--target-country-ape-pct",
        type=float,
        default=10.0,
        help="Per-country APE threshold for wind total matching.",
    )
    p.add_argument(
        "--target-country-success-share-pct",
        type=float,
        default=85.0,
        help="Required share of evaluated countries within target-country-ape-pct.",
    )
    p.add_argument(
        "--target-country-wape-pct",
        type=float,
        default=None,
        help="Optional extra WAPE threshold; unset to disable as stop criterion.",
    )
    p.add_argument("--target-split-error-pct", type=float, default=3.0)
    p.add_argument("--adjust-alpha", type=float, default=0.7)
    p.add_argument("--min-adjust", type=float, default=0.25)
    p.add_argument("--max-adjust", type=float, default=4.0)
    p.add_argument("--min-total-scale", type=float, default=0.05)
    p.add_argument("--max-total-scale", type=float, default=20.0)
    p.add_argument("--min-update-capacity-mw", type=float, default=1.0)
    p.add_argument("--min-update-model-twh", type=float, default=0.001)
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

    ref_wind_twh = _load_owid_wind_reference(owid_csv, args.year)
    if ref_wind_twh.empty:
        raise RuntimeError(f"No OWID wind reference found for year {args.year} in {owid_csv}")

    override_csv.parent.mkdir(parents=True, exist_ok=True)
    last_good_override_csv.parent.mkdir(parents=True, exist_ok=True)
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    country_detail_csv.parent.mkdir(parents=True, exist_ok=True)

    if not network_target.exists():
        LOG.info("Initial solved network missing, running first solve.")
        _run_snakemake(args)

    history_rows = []
    detail_last = None
    scales = pd.DataFrame(index=ref_wind_twh.index, data={"onwind_scale": 1.0, "offwind_scale": 1.0})
    if override_csv.exists():
        try:
            prev = pd.read_csv(override_csv)
            prev["country"] = prev["country"].astype(str).str.upper().str.strip()
            prev = prev.set_index("country")
            for col in ["onwind_scale", "offwind_scale"]:
                if col in prev.columns:
                    scales[col] = (
                        pd.to_numeric(prev[col], errors="coerce")
                        .reindex(scales.index)
                        .fillna(scales[col])
                    )
            LOG.info("Loaded existing override scales from %s", override_csv)
        except Exception as exc:
            LOG.warning("Could not load existing overrides from %s: %s", override_csv, exc)

    if not sync_last_good_from_mutable(override_csv, last_good_override_csv):
        seed_out = scales.reset_index().rename(columns={"index": "country"}).sort_values("country")
        seed_out.to_csv(last_good_override_csv, index=False, float_format="%.6f")
        LOG.info("Seeded last-good wind overrides: %s", last_good_override_csv)
    else:
        LOG.info("Seeded last-good wind overrides from current mutable file: %s", last_good_override_csv)

    for i in range(1, args.iterations + 1):
        if not network_target.exists():
            raise FileNotFoundError(f"Solved network not found after solve: {network_target}")

        model_wind = _load_model_wind_by_country(network_target)
        targets = _build_country_targets(
            ref_wind_twh=ref_wind_twh,
            model_df=model_wind,
            on_global_target=args.onshore_global_target_twh,
            off_global_target=args.offshore_global_target_twh,
            mode=args.offshore_weight_mode,
        )
        metrics, detail = _compute_metrics(model_wind, targets, args.country_min_ref_twh)
        eval_detail = detail.loc[detail["target_total_twh"] >= float(args.country_min_ref_twh)].copy()
        within_mask = eval_detail["ape_total_pct"] <= float(args.target_country_ape_pct)
        countries_within = int(within_mask.sum())
        countries_eval = int(len(eval_detail))
        country_success_share_pct = (
            100.0 * countries_within / countries_eval if countries_eval > 0 else np.nan
        )
        detail_last = detail.reset_index().rename(columns={"index": "country"})
        detail_last.insert(0, "iteration", i)

        row = {"iteration": i}
        row.update(metrics)
        row["countries_within_target"] = countries_within
        row["country_success_share_pct"] = country_success_share_pct
        history_rows.append(row)
        history = pd.DataFrame(history_rows)
        history.to_csv(history_csv, index=False, float_format="%.4f")
        detail_last.to_csv(country_detail_csv, index=False, float_format="%.4f")

        LOG.info(
            "Iteration %d: country_wape=%.2f%%, within_%s%%=%d/%d (%.1f%%), on=%.1f/%.1f TWh, off=%.1f/%.1f TWh, on_split_err=%.2f%%, off_split_err=%.2f%%",
            i,
            metrics["country_wape_pct"],
            int(args.target_country_ape_pct),
            countries_within,
            countries_eval,
            country_success_share_pct,
            metrics["on_model_twh"],
            metrics["on_target_twh"],
            metrics["off_model_twh"],
            metrics["off_target_twh"],
            metrics["on_split_abs_pct"],
            metrics["off_split_abs_pct"],
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
        converged = (
            country_share_ok
            and country_wape_ok
            and (np.isnan(metrics["on_split_abs_pct"]) or metrics["on_split_abs_pct"] <= args.target_split_error_pct)
            and (np.isnan(metrics["off_split_abs_pct"]) or metrics["off_split_abs_pct"] <= args.target_split_error_pct)
        )
        if converged:
            LOG.info("Converged at iteration %d.", i)
            break

        scales = _update_scales(
            current_scales=scales,
            model_df=model_wind,
            target_df=targets,
            alpha=args.adjust_alpha,
            min_adjust=args.min_adjust,
            max_adjust=args.max_adjust,
            min_total_scale=args.min_total_scale,
            max_total_scale=args.max_total_scale,
            min_update_capacity_mw=args.min_update_capacity_mw,
            min_update_model_twh=args.min_update_model_twh,
        )
        scales_out = scales.reset_index().rename(columns={"index": "country"})
        scales_out = scales_out.sort_values("country")
        scales_out.to_csv(override_csv, index=False, float_format="%.6f")
        LOG.info("Wrote overrides: %s", override_csv)
        try:
            raise_if_simulated_failure(args.simulate_post_write_failure, "wind tuner")
            _run_snakemake(args)
        except Exception as exc:
            restored = restore_from_last_good(override_csv, last_good_override_csv)
            LOG.error(
                "Wind solve/update failed after writing overrides. Rolled back to last good overrides at %s (restored=%s, snapshot=%s). Error: %s",
                override_csv,
                restored,
                last_good_override_csv,
                exc,
            )
            raise
        sync_last_good_from_mutable(override_csv, last_good_override_csv)

    history = pd.DataFrame(history_rows)
    history.to_csv(history_csv, index=False, float_format="%.4f")
    if detail_last is not None:
        detail_last.to_csv(country_detail_csv, index=False, float_format="%.4f")

    LOG.info("Saved iteration history to %s", history_csv)
    LOG.info("Saved last country detail to %s", country_detail_csv)


if __name__ == "__main__":
    main()
