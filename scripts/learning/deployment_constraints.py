# -*- coding: utf-8 -*-
"""Helpers for stochastic supply-chain deployment constraints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SUPPORTED_DEPLOYMENT_CONSTRAINT_MODES = {"first_order", "second_order", "both"}


def get_deployment_constraint_cfg(learning_cfg):
    cfg = (learning_cfg or {}).get("deployment_constraints", {}) or {}
    if not bool(cfg.get("enabled", False)):
        return None
    mode = str(cfg.get("mode", "first_order")).strip()
    if mode not in SUPPORTED_DEPLOYMENT_CONSTRAINT_MODES:
        raise ValueError(
            f"Unsupported learning.deployment_constraints.mode={mode!r}. "
            f"Supported values: {sorted(SUPPORTED_DEPLOYMENT_CONSTRAINT_MODES)}"
        )
    return cfg


def _load_manifest(learning_cfg, config_file=None):
    manifest = learning_cfg.get("_manifest", None)
    root = learning_cfg.get("_manifest_root", None)
    if manifest is None or root is None:
        manifest_path = Path(learning_cfg.get("artifact_manifest", "data/learning-data/manifest.json"))
        if not manifest_path.is_absolute():
            base = Path(config_file).resolve().parent if config_file else Path.cwd()
            manifest_path = base / manifest_path
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        root = manifest_path.parent
    return manifest, Path(root)


def load_deployment_constraint_support(learning_cfg, config_file=None):
    manifest, root = _load_manifest(learning_cfg, config_file=config_file)
    support = (((manifest.get("deployment_constraints", {}) or {}).get("support", {})) or {})
    calibration_rel = support.get("calibration_csv")
    history_rel = support.get("annual_history_csv")
    if not calibration_rel or not history_rel:
        raise ValueError("Learning manifest is missing deployment constraint support artifacts.")
    calibration = pd.read_csv(root / calibration_rel)
    history = pd.read_csv(root / history_rel)
    for col in (
        "window_start_year",
        "window_end_year",
        "quantile",
        "annual_addition_anchor_2024",
        "first_diff_q90",
        "second_diff_q90",
        "anchor_first_diff_2024",
        "first_order_residual_std",
        "second_order_residual_std",
    ):
        if col in calibration.columns:
            calibration[col] = pd.to_numeric(calibration[col], errors="coerce")
    for col in (
        "year",
        "annual_addition",
        "log_annual_addition",
        "first_diff",
        "second_diff",
        "first_order_residual",
        "second_order_residual",
    ):
        if col in history.columns:
            history[col] = pd.to_numeric(history[col], errors="coerce")
    return calibration, history, support


def _seeded_rng(learning_cfg, learning_seed=None):
    dc_cfg = (learning_cfg.get("deployment_constraints", {}) or {}).get("uncertainty", {}) or {}
    seed_source = str(dc_cfg.get("random_seed_source", "learning_seed")).strip()
    if seed_source == "learning_seed":
        seed_value = learning_seed if learning_seed not in (None, "") else learning_cfg.get("seed", 0)
    else:
        seed_value = learning_cfg.get("seed", 0)
    try:
        seed_int = int(str(seed_value).replace("s", ""))
    except Exception:
        seed_int = 0
    return np.random.default_rng(seed_int)


def _annual_years_for_block(current_year):
    end_year = int(current_year)
    start_year = end_year - 4
    return list(range(start_year, end_year + 1))


def _build_mode_mean_path(row, mode, years):
    first_diff = float(row["first_diff_q90"])
    second_diff = float(row["second_diff_q90"])
    anchor_addition = float(row["annual_addition_anchor_2024"])
    anchor_growth = float(row["anchor_first_diff_2024"])

    def _first_order():
        out = {}
        prev = np.log(anchor_addition)
        for year in years:
            prev = prev + first_diff
            out[year] = float(np.exp(prev))
        return out

    def _second_order():
        out = {}
        prev = np.log(anchor_addition)
        growth = anchor_growth
        for year in years:
            growth = growth + second_diff
            prev = prev + growth
            out[year] = float(np.exp(prev))
        return out

    if mode == "first_order":
        return _first_order()
    if mode == "second_order":
        return _second_order()
    first = _first_order()
    second = _second_order()
    return {year: min(first[year], second[year]) for year in years}


def build_deployment_constraint_block_paths(
    learning_cfg,
    current_year,
    learning_seed=None,
    config_file=None,
):
    cfg = get_deployment_constraint_cfg(learning_cfg)
    if cfg is None:
        return {}

    current_year = int(current_year)
    apply_from_year = int(cfg.get("apply_from_year", 2030))
    if current_year < apply_from_year:
        return {}

    calibration, _, _ = load_deployment_constraint_support(learning_cfg, config_file=config_file)
    techs = list(cfg.get("technologies", [])) or list(calibration["technology"].unique())
    mode = str(cfg.get("mode", "first_order")).strip()
    annual_years = _annual_years_for_block(current_year)
    uncertainty_cfg = (cfg.get("uncertainty", {}) or {})
    uncertainty_enabled = bool(uncertainty_cfg.get("enabled", False)) and str(
        uncertainty_cfg.get("mode", "stochastic_draw")
    ).strip() == "stochastic_draw"
    annual_fraction = float(uncertainty_cfg.get("annual_scale_fraction", 0.25))
    rng = _seeded_rng(learning_cfg, learning_seed=learning_seed)

    payload = {}
    for technology in techs:
        row = calibration.loc[calibration["technology"].eq(technology)]
        if row.empty:
            continue
        row = row.iloc[-1]
        mean_annual = _build_mode_mean_path(row, mode, annual_years)
        sigma_key = "first_order_residual_std" if mode == "first_order" else "second_order_residual_std"
        if mode == "both":
            sigma_persistent = max(
                float(row["first_order_residual_std"]),
                float(row["second_order_residual_std"]),
            )
        else:
            sigma_persistent = float(row[sigma_key])
        persistent_shock = 0.0
        annual_shocks = {year: 0.0 for year in annual_years}
        if uncertainty_enabled and sigma_persistent > 0.0:
            persistent_shock = float(rng.normal(0.0, sigma_persistent))
            sigma_annual = float(max(annual_fraction, 0.0) * sigma_persistent)
            if sigma_annual > 0.0:
                annual_shocks = {
                    year: float(rng.normal(0.0, sigma_annual))
                    for year in annual_years
                }
        annual_capped = {
            year: float(np.exp(np.log(max(mean_annual[year], 1.0e-12)) + persistent_shock + annual_shocks[year]))
            for year in annual_years
        }
        payload[technology] = {
            "technology": technology,
            "mode": mode,
            "uncertainty_enabled": uncertainty_enabled,
            "persistent_shock": persistent_shock,
            "annual_shocks": annual_shocks,
            "annual_mean_caps": mean_annual,
            "annual_capped_additions": annual_capped,
            "allowed_block_addition": float(sum(annual_capped.values())),
            "annual_years": annual_years,
        }
    return payload


def summarize_realized_block_additions(
    current_year,
    solved_capacity_by_tech,
    committed_state_payload,
    learning_cfg,
):
    current_year = int(current_year)
    prev_year = current_year - 5
    modeled_history = (committed_state_payload.get("modeled_capacity_history", {}) or {})
    cumulative_history = (committed_state_payload.get("capacity_history", {}) or {})
    realized = {}
    for technology, solved_capacity in (solved_capacity_by_tech or {}).items():
        tech_modeled = modeled_history.get(technology, {}) or {}
        prev_modeled = float(tech_modeled.get(str(prev_year), 0.0))
        modeled_addition = max(float(solved_capacity) - prev_modeled, 0.0)
        constrained_addition = modeled_addition
        battery_phi_block = np.nan
        if technology == "battery_energy" and current_year > prev_year:
            from learning.apply_learning_costs import get_battery_phi_for_block

            battery_phi_block = float(get_battery_phi_for_block(learning_cfg, prev_year, current_year))
            constrained_addition = modeled_addition * battery_phi_block
        realized[technology] = {
            "technology": technology,
            "modeled_block_addition": modeled_addition,
            "constrained_basis_addition": constrained_addition,
            "battery_phi_block": battery_phi_block,
            "previous_modeled_capacity": prev_modeled,
            "current_modeled_capacity": float(solved_capacity),
            "previous_constrained_capacity": float((cumulative_history.get(technology, {}) or {}).get(str(prev_year), np.nan)),
        }
    return realized
