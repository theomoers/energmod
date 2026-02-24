#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iteratively tune country-level fossil fuel prices to improve 2020 fossil electricity dispatch validation.

The script writes `validation/data/fossil_price_tuning_overrides.csv`, which is consumed by
`prepare_sector_network.py` via the fossil price override hook.

Targets (OWID):
- gas_electricity
- coal_electricity
- oil_electricity

Model outputs are read from solved AC energy balance (Generators + Links).
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

MODEL_CARRIER_TO_FUEL = {
    "OCGT": "gas",
    "CCGT": "gas",
    "urban central gas CHP": "gas",
    "urban central gas CHP CC": "gas",
    "coal": "coal",
    "lignite": "coal",
    "oil": "oil",
}
OWID_METRIC_TO_FUEL = {
    "gas_electricity": "gas",
    "coal_electricity": "coal",
    "oil_electricity": "oil",
}
FUEL_TO_OWID_METRIC = {v: k for k, v in OWID_METRIC_TO_FUEL.items()}


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
    tmp_cfg = REPO_ROOT / "validation" / ".tmp" / "fossil_tuning_enable.yaml"
    tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
    tmp_cfg.write_text(
        "global_specific:\\n  baseyear_generation:\\n    fossil_price_tuning_enabled: true\\n",
        encoding="utf-8",
    )
    configfiles = [*args.configfiles, str(tmp_cfg.relative_to(REPO_ROOT))]

    if args.unlock_first:
        unlock_cmd = [
            "snakemake",
            "--unlock",
            "--configfile",
            *configfiles,
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
        *configfiles,
        "--rerun-trigger",
        "mtime",
        "--forcerun",
        *args.force_rules,
    ]
    _run_cmd(cmd, REPO_ROOT, env=env, dry_run=args.dry_run)


def _load_owid_fossil_electricity_reference(owid_csv, year):
    usecols = {"year", "iso_code", "gas_electricity", "coal_electricity", "oil_electricity"}
    owid = pd.read_csv(owid_csv, usecols=lambda c: c in usecols)
    owid = owid.loc[owid["year"] == int(year)].copy()
    owid["country"] = owid["iso_code"].apply(_safe_iso3_to_iso2)
    owid = owid.loc[owid["country"].notna()].copy()
    for col in OWID_METRIC_TO_FUEL.keys():
        owid[col] = pd.to_numeric(owid[col], errors="coerce").fillna(0.0)
    grouped = owid.groupby("country", as_index=False)[list(OWID_METRIC_TO_FUEL.keys())].sum(min_count=1)
    long = grouped.melt(id_vars="country", var_name="metric", value_name="target_twh")
    long["fuel_type"] = long["metric"].map(OWID_METRIC_TO_FUEL)
    out = long[["country", "fuel_type", "target_twh"]].copy()
    return out


def _load_model_fossil_electricity(network_path):
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
    eb = eb.loc[eb.component.isin(["Generator", "Link"])].copy()
    eb["fuel_type"] = eb["carrier"].map(MODEL_CARRIER_TO_FUEL)
    eb = eb.loc[eb["fuel_type"].notna()].copy()
    model = (
        eb.groupby(["country", "fuel_type"], as_index=False)["energy_mwh"].sum()
        .rename(columns={"energy_mwh": "model_twh"})
    )
    model["model_twh"] = model["model_twh"] / 1e6
    return model


def _load_base_fuel_prices(price_csv, year):
    df = pd.read_csv(price_csv)
    required = {"country", "fuel_type", "year", "price_eur_mwh"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Fuel price CSV missing columns: {sorted(missing)}")
    df = df.loc[df["year"].astype(int).eq(int(year))].copy()
    df["country"] = df["country"].astype(str).str.upper().str.strip()
    df["fuel_type"] = df["fuel_type"].astype(str).str.lower().str.strip()
    df = df.loc[df["fuel_type"].isin(["gas", "coal", "oil"]) & df["country"].str.len().eq(2)].copy()
    df["price_eur_mwh"] = pd.to_numeric(df["price_eur_mwh"], errors="coerce")
    df = df.dropna(subset=["price_eur_mwh"]).copy()
    # Keep first if duplicates (country,fuel,year,market variants unexpected but possible).
    df = df.drop_duplicates(subset=["country", "fuel_type"], keep="first")
    return df[["country", "fuel_type", "price_eur_mwh"]]


def _load_existing_overrides(override_csv, base_prices):
    idx = base_prices.set_index(["country", "fuel_type"]).index
    mult = pd.Series(1.0, index=idx, dtype=float)
    if not override_csv.exists():
        return mult
    try:
        ov = pd.read_csv(override_csv)
    except Exception as exc:
        LOG.warning("Could not read existing overrides %s: %s", override_csv, exc)
        return mult
    if ov.empty:
        return mult
    cols = {c.lower().strip(): c for c in ov.columns}
    if "country" not in cols or "fuel_type" not in cols:
        return mult
    ov = ov.rename(columns={cols["country"]: "country", cols["fuel_type"]: "fuel_type"})
    if "price_multiplier" in cols:
        ov = ov.rename(columns={cols["price_multiplier"]: "price_multiplier"})
    if "price_eur_mwh" in cols:
        ov = ov.rename(columns={cols["price_eur_mwh"]: "price_eur_mwh"})
    ov["country"] = ov["country"].astype(str).str.upper().str.strip()
    ov["fuel_type"] = ov["fuel_type"].astype(str).str.lower().str.strip()
    if "year" in cols:
        ycol = cols["year"]
        ov[ycol] = pd.to_numeric(ov[ycol], errors="coerce")
    merged = base_prices.merge(ov, on=["country", "fuel_type"], how="left", suffixes=("_base", "_ov"))
    if "price_multiplier" in merged.columns:
        mult = pd.to_numeric(merged["price_multiplier"], errors="coerce")
        mult = pd.Series(mult.values, index=idx).fillna(1.0)
    elif "price_eur_mwh" in merged.columns:
        abs_price = pd.to_numeric(merged["price_eur_mwh"], errors="coerce")
        m = abs_price / pd.to_numeric(merged["price_eur_mwh_base"], errors="coerce").replace(0.0, np.nan)
        mult = pd.Series(m.values, index=idx).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return mult.astype(float)


def _merge_model_ref(model_df, ref_df):
    cmp_df = model_df.merge(ref_df, on=["country", "fuel_type"], how="outer")
    cmp_df["model_twh"] = cmp_df["model_twh"].fillna(0.0)
    cmp_df["target_twh"] = cmp_df["target_twh"].fillna(0.0)
    cmp_df["error_twh"] = cmp_df["model_twh"] - cmp_df["target_twh"]
    cmp_df["abs_error_twh"] = cmp_df["error_twh"].abs()
    cmp_df["ape_pct"] = np.where(
        cmp_df["target_twh"] > 0,
        100.0 * cmp_df["abs_error_twh"] / cmp_df["target_twh"],
        np.nan,
    )
    return cmp_df


def _compute_metrics(cmp_df, min_ref_twh):
    eval_mask = cmp_df["target_twh"] >= float(min_ref_twh)
    eval_df = cmp_df.loc[eval_mask].copy()
    if eval_df.empty:
        wape = np.nan
    else:
        denom = eval_df["target_twh"].sum()
        wape = 100.0 * eval_df["abs_error_twh"].sum() / denom if denom > 0 else np.nan

    by_fuel = (
        cmp_df.groupby("fuel_type", as_index=False)[["model_twh", "target_twh", "abs_error_twh"]]
        .sum()
        .rename(columns={"abs_error_twh": "abs_error_total_twh"})
    )
    by_fuel["wape_pct"] = np.where(
        by_fuel["target_twh"] > 0,
        100.0 * by_fuel["abs_error_total_twh"] / by_fuel["target_twh"],
        np.nan,
    )

    return {
        "entry_wape_pct": wape,
        "eval_entries": int(eval_mask.sum()),
        "global_fossil_model_twh": float(cmp_df["model_twh"].sum()),
        "global_fossil_target_twh": float(cmp_df["target_twh"].sum()),
        "global_fossil_wape_pct": (
            100.0 * cmp_df["abs_error_twh"].sum() / max(cmp_df["target_twh"].sum(), 1e-9)
        ),
        "gas_model_twh": float(by_fuel.loc[by_fuel["fuel_type"] == "gas", "model_twh"].sum()),
        "gas_target_twh": float(by_fuel.loc[by_fuel["fuel_type"] == "gas", "target_twh"].sum()),
        "coal_model_twh": float(by_fuel.loc[by_fuel["fuel_type"] == "coal", "model_twh"].sum()),
        "coal_target_twh": float(by_fuel.loc[by_fuel["fuel_type"] == "coal", "target_twh"].sum()),
        "oil_model_twh": float(by_fuel.loc[by_fuel["fuel_type"] == "oil", "model_twh"].sum()),
        "oil_target_twh": float(by_fuel.loc[by_fuel["fuel_type"] == "oil", "target_twh"].sum()),
    }, eval_df, by_fuel


def _dispatch_price_adjustment(target, model, min_adjust, max_adjust):
    # Price update direction is inverse to generation target ratio:
    # if model > target, increase price (adj > 1); if model < target, decrease price.
    if target > 0 and model > 0:
        ratio = model / target
    elif target > 0 and model <= 0:
        ratio = min_adjust
    elif target <= 0 and model > 0:
        ratio = max_adjust
    else:
        ratio = 1.0
    return float(np.clip(ratio, min_adjust, max_adjust))


def _write_override_csv(path, base_prices, multipliers, year):
    base_idx = base_prices.set_index(["country", "fuel_type"])
    out = base_idx.copy()
    out["price_multiplier"] = multipliers.reindex(base_idx.index).fillna(1.0)
    out["price_eur_mwh_tuned"] = out["price_eur_mwh"] * out["price_multiplier"]
    out["delta_price_eur_mwh"] = out["price_eur_mwh_tuned"] - out["price_eur_mwh"]
    out["delta_pct"] = np.where(out["price_eur_mwh"] > 0, 100.0 * out["delta_price_eur_mwh"] / out["price_eur_mwh"], np.nan)
    out = out.reset_index().rename(columns={"price_eur_mwh": "baseline_price_eur_mwh", "price_eur_mwh_tuned": "price_eur_mwh"})
    out.insert(2, "year", int(year))
    out = out[["country", "fuel_type", "year", "price_multiplier", "price_eur_mwh", "baseline_price_eur_mwh", "delta_price_eur_mwh", "delta_pct"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, float_format="%.6f")
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Iteratively tune fossil dispatch via country fuel price overrides.")
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
    p.add_argument("--fuel-price-csv", default="data/fuels/all_fuels_prices_by_country.csv")
    p.add_argument("--year", type=int, default=2020)
    p.add_argument("--iterations", type=int, default=6)
    p.add_argument("--override-csv", default="validation/data/fossil_price_tuning_overrides.csv")
    p.add_argument(
        "--last-good-override-csv",
        default="validation/data/fossil_price_tuning_overrides_last_good.csv",
        help="Rollback snapshot of the last known-good fossil price override CSV.",
    )
    p.add_argument("--history-csv", default="validation/results_compare/fossil_dispatch_tuning_history.csv")
    p.add_argument("--detail-csv", default="validation/results_compare/fossil_dispatch_tuning_country_fuel_detail.csv")
    p.add_argument("--price-summary-csv", default="validation/results_compare/fossil_dispatch_price_comparison.csv")
    p.add_argument("--min-ref-entry-twh", type=float, default=1.0)
    p.add_argument("--target-entry-ape-pct", type=float, default=10.0)
    p.add_argument("--target-success-share-pct", type=float, default=70.0)
    p.add_argument("--target-entry-wape-pct", type=float, default=None)
    p.add_argument("--target-global-wape-pct", type=float, default=10.0)
    p.add_argument("--adjust-alpha", type=float, default=0.35)
    p.add_argument("--min-adjust", type=float, default=0.7)
    p.add_argument("--max-adjust", type=float, default=1.5)
    p.add_argument("--min-total-multiplier", type=float, default=0.2)
    p.add_argument("--max-total-multiplier", type=float, default=5.0)
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
    fuel_price_csv = _repo_path(args.fuel_price_csv)
    override_csv = _repo_path(args.override_csv)
    last_good_override_csv = _repo_path(args.last_good_override_csv)
    history_csv = _repo_path(args.history_csv)
    detail_csv = _repo_path(args.detail_csv)
    price_summary_csv = _repo_path(args.price_summary_csv)

    ref_fossil = _load_owid_fossil_electricity_reference(owid_csv, args.year)
    if ref_fossil.empty:
        raise RuntimeError(f"No OWID fossil electricity reference found for {args.year} in {owid_csv}")
    base_prices = _load_base_fuel_prices(fuel_price_csv, args.year)
    if base_prices.empty:
        raise RuntimeError(f"No base fuel prices found for year {args.year} in {fuel_price_csv}")

    # Restrict tuning universe to rows with explicit country prices in the model price CSV.
    base_idx = base_prices.set_index(["country", "fuel_type"]).index
    multipliers = _load_existing_overrides(override_csv, base_prices)
    LOG.info("Loaded base price table for %d country-fuel entries (year=%s).", len(base_prices), args.year)

    history_csv.parent.mkdir(parents=True, exist_ok=True)
    detail_csv.parent.mkdir(parents=True, exist_ok=True)
    price_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    override_csv.parent.mkdir(parents=True, exist_ok=True)
    last_good_override_csv.parent.mkdir(parents=True, exist_ok=True)

    if not sync_last_good_from_mutable(override_csv, last_good_override_csv):
        _write_override_csv(last_good_override_csv, base_prices, multipliers, args.year)
        LOG.info("Seeded last-good fossil overrides: %s", last_good_override_csv)
    else:
        LOG.info(
            "Seeded last-good fossil overrides from current mutable file: %s",
            last_good_override_csv,
        )

    if not network_target.exists():
        LOG.info("Initial solved network missing, running first solve.")
        _write_override_csv(override_csv, base_prices, multipliers, args.year)
        _run_snakemake(args)

    history_rows = []
    detail_last = None
    for i in range(1, args.iterations + 1):
        if not network_target.exists():
            raise FileNotFoundError(f"Solved network not found after solve: {network_target}")

        model_fossil = _load_model_fossil_electricity(network_target)
        cmp_df = _merge_model_ref(model_fossil, ref_fossil)
        cmp_df = cmp_df.merge(base_prices, on=["country", "fuel_type"], how="left")
        cmp_df["has_base_price"] = cmp_df["price_eur_mwh"].notna()
        cmp_df["price_multiplier"] = multipliers.reindex(
            pd.MultiIndex.from_frame(cmp_df[["country", "fuel_type"]])
        ).to_numpy()
        cmp_df["price_multiplier"] = pd.to_numeric(cmp_df["price_multiplier"], errors="coerce").fillna(1.0)
        cmp_df["tuned_price_eur_mwh"] = cmp_df["price_eur_mwh"] * cmp_df["price_multiplier"]

        metrics, eval_df, by_fuel = _compute_metrics(cmp_df, args.min_ref_entry_twh)
        eval_entries = eval_df.copy()
        within_mask = eval_entries["ape_pct"] <= float(args.target_entry_ape_pct)
        entries_within = int(within_mask.sum())
        entries_eval = int(len(eval_entries))
        success_share = 100.0 * entries_within / entries_eval if entries_eval > 0 else np.nan

        tunable_eval = eval_entries.loc[eval_entries["has_base_price"].fillna(False)].copy()
        untunable_eval = int(entries_eval - len(tunable_eval))

        detail_last = cmp_df.sort_values(["fuel_type", "country"]).copy()
        row = {"iteration": i}
        row.update(metrics)
        row["entries_within_target"] = entries_within
        row["entry_success_share_pct"] = success_share
        row["tunable_eval_entries"] = int(len(tunable_eval))
        row["untunable_eval_entries"] = untunable_eval
        row["price_multiplier_min"] = float(multipliers.min()) if len(multipliers) else np.nan
        row["price_multiplier_max"] = float(multipliers.max()) if len(multipliers) else np.nan
        row["avg_abs_price_delta_pct"] = float(
            _write_override_csv(override_csv, base_prices, multipliers, args.year)["delta_pct"].abs().mean()
        ) if len(base_prices) else np.nan
        history_rows.append(row)

        LOG.info(
            "Iteration %d: fossil_global_wape=%.2f%%, entry_wape=%.2f%%, within_%s%%=%d/%d (%.1f%%), tunable_eval=%d, untunable_eval=%d, gas=%.1f/%.1f TWh, coal=%.1f/%.1f TWh, oil=%.1f/%.1f TWh",
            i,
            metrics["global_fossil_wape_pct"],
            metrics["entry_wape_pct"],
            int(args.target_entry_ape_pct),
            entries_within,
            entries_eval,
            success_share,
            int(len(tunable_eval)),
            untunable_eval,
            metrics["gas_model_twh"],
            metrics["gas_target_twh"],
            metrics["coal_model_twh"],
            metrics["coal_target_twh"],
            metrics["oil_model_twh"],
            metrics["oil_target_twh"],
        )

        share_ok = np.isnan(success_share) or success_share >= float(args.target_success_share_pct)
        entry_wape_ok = (
            True
            if args.target_entry_wape_pct is None
            else (np.isnan(metrics["entry_wape_pct"]) or metrics["entry_wape_pct"] <= float(args.target_entry_wape_pct))
        )
        global_wape_ok = np.isnan(metrics["global_fossil_wape_pct"]) or metrics["global_fossil_wape_pct"] <= float(args.target_global_wape_pct)
        if share_ok and entry_wape_ok and global_wape_ok:
            LOG.info("Converged at iteration %d.", i)
            break

        # Update multipliers for tunable eval entries only.
        updated = 0
        for _, row_eval in tunable_eval.iterrows():
            idx = (row_eval["country"], row_eval["fuel_type"])
            adj = _dispatch_price_adjustment(
                target=float(row_eval["target_twh"]),
                model=float(row_eval["model_twh"]),
                min_adjust=args.min_adjust,
                max_adjust=args.max_adjust,
            )
            new_mult = float(np.clip(multipliers.loc[idx] * (adj ** float(args.adjust_alpha)), args.min_total_multiplier, args.max_total_multiplier))
            if not np.isclose(new_mult, multipliers.loc[idx], atol=1e-12):
                updated += 1
            multipliers.loc[idx] = new_mult

        if updated == 0:
            LOG.warning("No tunable price multipliers updated in iteration %d; stopping.", i)
            break

        price_comp = _write_override_csv(override_csv, base_prices, multipliers, args.year)
        price_comp.to_csv(price_summary_csv, index=False, float_format="%.4f")
        LOG.info("Wrote fossil price overrides: %s (updated=%d)", override_csv, updated)

        if detail_last is not None:
            detail_last.to_csv(detail_csv, index=False, float_format="%.4f")
        pd.DataFrame(history_rows).to_csv(history_csv, index=False, float_format="%.4f")
        try:
            raise_if_simulated_failure(args.simulate_post_write_failure, "fossil tuner")
            _run_snakemake(args)
        except Exception as exc:
            restored = restore_from_last_good(override_csv, last_good_override_csv)
            LOG.error(
                "Fossil solve/update failed after writing overrides. Rolled back to last good overrides at %s (restored=%s, snapshot=%s). Error: %s",
                override_csv,
                restored,
                last_good_override_csv,
                exc,
            )
            raise
        sync_last_good_from_mutable(override_csv, last_good_override_csv)

    # Final exports
    final_price_comp = _write_override_csv(override_csv, base_prices, multipliers, args.year)
    final_price_comp.to_csv(price_summary_csv, index=False, float_format="%.4f")
    if detail_last is not None:
        detail_last.to_csv(detail_csv, index=False, float_format="%.4f")
    pd.DataFrame(history_rows).to_csv(history_csv, index=False, float_format="%.4f")

    if not final_price_comp.empty:
        summary = (
            final_price_comp.groupby("fuel_type", as_index=False)
            .agg(
                entries=("country", "count"),
                mean_baseline_price_eur_mwh=("baseline_price_eur_mwh", "mean"),
                mean_tuned_price_eur_mwh=("price_eur_mwh", "mean"),
                mean_abs_delta_pct=("delta_pct", lambda s: float(np.nanmean(np.abs(s)))),
                p90_abs_delta_pct=("delta_pct", lambda s: float(np.nanpercentile(np.abs(s.dropna()), 90)) if s.notna().any() else np.nan),
            )
        )
        LOG.info("Tuned-vs-current price summary by fuel (current=model input CSV prices for %s):", args.year)
        for _, r in summary.iterrows():
            LOG.info(
                "  %-4s entries=%4d mean_price %.2f -> %.2f EUR/MWh (mean |delta|=%.1f%%, p90 |delta|=%.1f%%)",
                r["fuel_type"],
                int(r["entries"]),
                r["mean_baseline_price_eur_mwh"],
                r["mean_tuned_price_eur_mwh"],
                r["mean_abs_delta_pct"],
                r["p90_abs_delta_pct"],
            )

    LOG.info("Saved iteration history to %s", history_csv)
    LOG.info("Saved country-fuel detail to %s", detail_csv)
    LOG.info("Saved tuned-vs-current price comparison to %s", price_summary_csv)


if __name__ == "__main__":
    main()
