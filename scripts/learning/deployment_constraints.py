# -*- coding: utf-8 -*-
"""Helpers for stochastic supply-chain deployment constraints."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


SUPPORTED_DEPLOYMENT_CONSTRAINT_MODES = {"first_order", "second_order", "both"}

DEPLOYMENT_CONSTRAINT_SPECS = {
    "solar_power": {
        "component_attr": "generators",
        "extendable_col": "p_nom_extendable",
        "carrier": "solar",
        "build_year_col": "build_year",
        "optimized_col": "p_nom_opt",
        "fallback_col": "p_nom",
        "max_col": "p_nom_max",
        "model_var": "Generator-p_nom",
        "raw_to_basis_divisor": 1e3,
        "constraint_basis_unit": "GW",
        "phi_mapped": False,
    },
    "onwind_power": {
        "component_attr": "generators",
        "extendable_col": "p_nom_extendable",
        "carrier": "onwind",
        "build_year_col": "build_year",
        "optimized_col": "p_nom_opt",
        "fallback_col": "p_nom",
        "max_col": "p_nom_max",
        "model_var": "Generator-p_nom",
        "raw_to_basis_divisor": 1e3,
        "constraint_basis_unit": "GW",
        "phi_mapped": False,
    },
    "battery_energy": {
        "component_attr": "stores",
        "extendable_col": "e_nom_extendable",
        "carrier": "battery",
        "build_year_col": "build_year",
        "optimized_col": "e_nom_opt",
        "fallback_col": "e_nom",
        "max_col": "e_nom_max",
        "model_var": "Store-e_nom",
        "raw_to_basis_divisor": 1e3,
        "constraint_basis_unit": "global_liion_equivalent_GWh",
        "phi_mapped": True,
    },
}


def get_deployment_constraint_spec(technology):
    try:
        return DEPLOYMENT_CONSTRAINT_SPECS[technology]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported deployment-constraint technology: {technology!r}. "
            f"Supported values: {sorted(DEPLOYMENT_CONSTRAINT_SPECS)}"
        ) from exc


def get_constraint_basis_unit(technology):
    return str(get_deployment_constraint_spec(technology)["constraint_basis_unit"])


def normalize_constraint_basis_value(technology, value):
    spec = get_deployment_constraint_spec(technology)
    return float(value) / float(spec["raw_to_basis_divisor"])


def _select_current_vintage_indices(df, planning_year, spec):
    if df is None or df.empty:
        return pd.Index([])
    subset = df.copy()
    extendable_col = spec["extendable_col"]
    if extendable_col in subset.columns:
        subset = subset[subset[extendable_col].fillna(False)]
    subset = subset[subset["carrier"].astype(str).eq(spec["carrier"])]
    build_year_col = spec["build_year_col"]
    if build_year_col in subset.columns:
        build_year = pd.to_numeric(subset[build_year_col], errors="coerce")
        subset = subset[build_year.fillna(planning_year).astype(int).eq(int(planning_year))]
    return subset.index


def get_current_vintage_indices(n, technology, planning_year):
    spec = get_deployment_constraint_spec(technology)
    component_df = getattr(n, spec["component_attr"], None)
    return _select_current_vintage_indices(component_df, planning_year, spec)


def _get_capacity_series(df, spec):
    optimized_col = spec["optimized_col"]
    fallback_col = spec["fallback_col"]
    if optimized_col in df.columns:
        series = pd.to_numeric(df[optimized_col], errors="coerce")
        if fallback_col in df.columns:
            series = series.fillna(pd.to_numeric(df[fallback_col], errors="coerce"))
        return series.fillna(0.0)
    if fallback_col in df.columns:
        return pd.to_numeric(df[fallback_col], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index, dtype=float)


def get_battery_phi_block(learning_cfg, current_year):
    current_year = int(current_year)
    prev_year = current_year - 5
    from learning.apply_learning_costs import get_battery_phi_for_block

    return float(get_battery_phi_for_block(learning_cfg, prev_year, current_year))


def get_current_vintage_constraint_basis_value(n, technology, planning_year, learning_cfg):
    spec = get_deployment_constraint_spec(technology)
    idx = get_current_vintage_indices(n, technology, planning_year)
    result = {
        "technology": technology,
        "constraint_basis_unit": spec["constraint_basis_unit"],
        "selected_assets": int(len(idx)),
        "constrained_basis_value": 0.0,
        "battery_phi_block": np.nan,
    }
    if len(idx) == 0:
        return result

    component_df = getattr(n, spec["component_attr"], None)
    values = _get_capacity_series(component_df.loc[idx], spec)
    constrained_value = normalize_constraint_basis_value(technology, values.sum())
    if spec["phi_mapped"]:
        phi_block = get_battery_phi_block(learning_cfg, planning_year)
        result["battery_phi_block"] = phi_block
        constrained_value *= phi_block
    result["constrained_basis_value"] = float(constrained_value)
    return result


def get_current_vintage_constraint_basis_upper_bound(n, technology, planning_year, learning_cfg):
    spec = get_deployment_constraint_spec(technology)
    idx = get_current_vintage_indices(n, technology, planning_year)
    result = {
        "technology": technology,
        "constraint_basis_unit": spec["constraint_basis_unit"],
        "upper_bound_basis_value": np.nan,
        "upper_bound_is_finite": False,
        "upper_bound_nonfinite_assets": int(len(idx)),
        "battery_phi_block_upper_bound": np.nan,
    }
    if len(idx) == 0:
        result["upper_bound_basis_value"] = 0.0
        result["upper_bound_is_finite"] = True
        result["upper_bound_nonfinite_assets"] = 0
        return result

    component_df = getattr(n, spec["component_attr"], None)
    max_col = spec["max_col"]
    if component_df is None or max_col not in component_df.columns:
        return result

    raw_upper = pd.to_numeric(component_df.loc[idx, max_col], errors="coerce")
    finite_upper = raw_upper.replace([np.inf, -np.inf], np.nan).dropna()
    result["upper_bound_nonfinite_assets"] = int(len(raw_upper) - len(finite_upper))
    if len(finite_upper) != len(raw_upper):
        return result

    upper_value = normalize_constraint_basis_value(technology, finite_upper.sum())
    if spec["phi_mapped"]:
        phi_block = get_battery_phi_block(learning_cfg, planning_year)
        result["battery_phi_block_upper_bound"] = phi_block
        upper_value *= phi_block
    result["upper_bound_basis_value"] = float(upper_value)
    result["upper_bound_is_finite"] = True
    return result


def build_deployment_constraint_lhs(n, technology, planning_year, learning_cfg):
    spec = get_deployment_constraint_spec(technology)
    idx = get_current_vintage_indices(n, technology, planning_year)
    if len(idx) == 0:
        return None, {
            "technology": technology,
            "constraint_basis_unit": spec["constraint_basis_unit"],
            "selected_assets": 0,
            "battery_phi_block": np.nan,
        }

    lhs = n.model[spec["model_var"]].loc[idx].sum()
    lhs = lhs / float(spec["raw_to_basis_divisor"])
    metadata = {
        "technology": technology,
        "constraint_basis_unit": spec["constraint_basis_unit"],
        "selected_assets": int(len(idx)),
        "battery_phi_block": np.nan,
    }
    if spec["phi_mapped"]:
        phi_block = get_battery_phi_block(learning_cfg, planning_year)
        metadata["battery_phi_block"] = phi_block
        lhs = lhs * phi_block
    return lhs, metadata


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


def _deployment_calibration_request(learning_cfg):
    cfg = get_deployment_constraint_cfg(learning_cfg)
    if cfg is None:
        return None
    calibration_cfg = cfg.get("calibration", {}) or {}
    if "quantile" not in calibration_cfg:
        return None
    return {
        "window_start_year": int(calibration_cfg.get("window_start_year", 2015)),
        "window_end_year": int(calibration_cfg.get("window_end_year", 2024)),
        "quantile": float(calibration_cfg["quantile"]),
        "difference_space": str(
            calibration_cfg.get("difference_space", "log_annual_additions")
        ),
    }


def _matching_calibration_rows(calibration, request):
    if calibration.empty:
        return calibration
    mask = np.isclose(
        pd.to_numeric(calibration["quantile"], errors="coerce"),
        float(request["quantile"]),
        rtol=0.0,
        atol=1.0e-12,
    )
    mask &= pd.to_numeric(
        calibration["window_start_year"], errors="coerce"
    ).eq(int(request["window_start_year"]))
    mask &= pd.to_numeric(
        calibration["window_end_year"], errors="coerce"
    ).eq(int(request["window_end_year"]))
    return calibration.loc[mask].copy()


def _derive_requested_calibration_from_history(history, calibration, request):
    if str(request["difference_space"]) != "log_annual_additions":
        raise ValueError(
            "Deployment constraint runtime calibration currently supports "
            "difference_space='log_annual_additions' only."
        )

    start_year = int(request["window_start_year"])
    end_year = int(request["window_end_year"])
    quantile = float(request["quantile"])
    technologies = list(calibration["technology"].dropna().unique())
    if not technologies:
        technologies = list(history["technology"].dropna().unique())

    rows = []
    for technology in technologies:
        tech_history = history.loc[history["technology"].eq(technology)].copy()
        window = tech_history.loc[
            tech_history["year"].between(start_year, end_year, inclusive="both")
        ].copy()
        if window.empty:
            raise ValueError(
                f"Cannot derive deployment calibration for {technology}: "
                f"no history in {start_year}-{end_year}."
            )
        anchor = window.loc[window["year"].eq(end_year)]
        if anchor.empty:
            raise ValueError(
                f"Cannot derive deployment calibration for {technology}: "
                f"missing anchor year {end_year} in history."
            )
        first_diff = pd.to_numeric(window["first_diff"], errors="coerce").dropna()
        second_diff = pd.to_numeric(window["second_diff"], errors="coerce").dropna()
        if first_diff.empty or second_diff.empty:
            raise ValueError(
                f"Cannot derive deployment calibration for {technology}: "
                "missing first/second difference history."
            )

        first_q = float(first_diff.quantile(quantile))
        second_q = float(second_diff.quantile(quantile))
        anchor = anchor.iloc[-1]
        first_residuals = first_diff - first_q
        second_residuals = second_diff - second_q
        rows.append(
            {
                "technology": technology,
                "window_start_year": start_year,
                "window_end_year": end_year,
                "quantile": quantile,
                "annual_addition_anchor_2024": float(anchor["annual_addition"]),
                # Keep the historical column names for compatibility; values are
                # resolved from the configured quantile at load time.
                "first_diff_q90": first_q,
                "second_diff_q90": second_q,
                "anchor_first_diff_2024": float(anchor["first_diff"]),
                "first_order_residual_std": float(first_residuals.std()),
                "second_order_residual_std": float(second_residuals.std()),
            }
        )
    return pd.DataFrame(rows, columns=calibration.columns)


def _resolve_configured_calibration(calibration, history, learning_cfg):
    request = _deployment_calibration_request(learning_cfg)
    if request is None:
        return calibration

    matching = _matching_calibration_rows(calibration, request)
    expected_technologies = set(calibration["technology"].dropna().unique())
    matching_technologies = set(matching["technology"].dropna().unique())
    if expected_technologies and matching_technologies == expected_technologies:
        return matching

    return _derive_requested_calibration_from_history(history, calibration, request)


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
    calibration = _resolve_configured_calibration(calibration, history, learning_cfg)
    return calibration, history, support


def _deployment_constraint_seed_int(learning_cfg, learning_seed=None):
    dc_cfg = (learning_cfg.get("deployment_constraints", {}) or {}).get("uncertainty", {}) or {}
    seed_source = str(dc_cfg.get("random_seed_source", "learning_seed")).strip()
    if seed_source == "learning_seed":
        seed_value = learning_seed if learning_seed not in (None, "") else learning_cfg.get("seed", 0)
    else:
        seed_value = learning_cfg.get("seed", 0)
    try:
        return int(str(seed_value).replace("s", ""))
    except Exception:
        return 0


def _seeded_rng(learning_cfg, learning_seed=None):
    return np.random.default_rng(_deployment_constraint_seed_int(learning_cfg, learning_seed))


def _stable_deployment_rng(learning_cfg, learning_seed, technology, draw_kind, year=None):
    seed_int = _deployment_constraint_seed_int(learning_cfg, learning_seed)
    parts = [str(seed_int), str(technology), str(draw_kind)]
    if year is not None:
        parts.append(str(int(year)))
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little", signed=False))


def _annual_years_for_block(current_year):
    end_year = int(current_year)
    start_year = end_year - 4
    return list(range(start_year, end_year + 1))


def _annual_years_for_trajectory(current_year, anchor_year=2024):
    current_year = int(current_year)
    anchor_year = int(anchor_year)
    if current_year <= anchor_year:
        return []
    return list(range(anchor_year + 1, current_year + 1))


def _anchor_year_from_calibration(row):
    try:
        value = row.get("window_end_year", 2024)
        if pd.isna(value):
            return 2024
        return int(value)
    except Exception:
        return 2024


def _build_mode_mean_path(row, mode, years):
    requested_years = sorted({int(year) for year in years})
    if not requested_years:
        return {}

    first_diff = float(row["first_diff_q90"])
    second_diff = float(row["second_diff_q90"])
    anchor_addition = float(row["annual_addition_anchor_2024"])
    anchor_growth = float(row["anchor_first_diff_2024"])
    anchor_year = _anchor_year_from_calibration(row)
    trajectory_years = range(anchor_year + 1, max(requested_years) + 1)

    def _first_order():
        out = {}
        prev = np.log(anchor_addition)
        for year in trajectory_years:
            prev = prev + first_diff
            out[year] = float(np.exp(prev))
        return out

    def _second_order():
        out = {}
        prev = np.log(anchor_addition)
        growth = anchor_growth
        for year in trajectory_years:
            growth = growth + second_diff
            prev = prev + growth
            out[year] = float(np.exp(prev))
        return out

    if mode == "first_order":
        path = _first_order()
        return {year: path[year] for year in requested_years}
    if mode == "second_order":
        path = _second_order()
        return {year: path[year] for year in requested_years}
    first = _first_order()
    second = _second_order()
    return {year: min(first[year], second[year]) for year in requested_years}


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

    payload = {}
    for technology in techs:
        row = calibration.loc[calibration["technology"].eq(technology)]
        if row.empty:
            continue
        row = row.iloc[-1]
        trajectory_years = _annual_years_for_trajectory(
            current_year,
            anchor_year=_anchor_year_from_calibration(row),
        )
        mean_annual_full = _build_mode_mean_path(row, mode, trajectory_years)
        mean_annual = {
            year: mean_annual_full[year]
            for year in annual_years
            if year in mean_annual_full
        }
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
            persistent_shock = float(
                _stable_deployment_rng(
                    learning_cfg,
                    learning_seed,
                    technology,
                    "persistent",
                ).normal(0.0, sigma_persistent)
            )
            sigma_annual = float(max(annual_fraction, 0.0) * sigma_persistent)
            if sigma_annual > 0.0:
                annual_shocks = {
                    year: float(
                        _stable_deployment_rng(
                            learning_cfg,
                            learning_seed,
                            technology,
                            "annual",
                            year=year,
                        ).normal(0.0, sigma_annual)
                    )
                    for year in annual_years
                }
        annual_capped = {
            year: float(np.exp(np.log(max(mean_annual[year], 1.0e-12)) + persistent_shock + annual_shocks[year]))
            for year in annual_years
        }
        mean_annual_basis = {
            year: normalize_constraint_basis_value(technology, value)
            for year, value in mean_annual.items()
        }
        annual_capped_basis = {
            year: normalize_constraint_basis_value(technology, value)
            for year, value in annual_capped.items()
        }
        payload[technology] = {
            "technology": technology,
            "mode": mode,
            "uncertainty_enabled": uncertainty_enabled,
            "persistent_shock": persistent_shock,
            "annual_shocks": annual_shocks,
            "annual_mean_caps": mean_annual_basis,
            "annual_capped_additions": annual_capped_basis,
            "allowed_block_addition": float(sum(annual_capped_basis.values())),
            "annual_years": annual_years,
            "constraint_basis_unit": get_constraint_basis_unit(technology),
        }
    return payload


def summarize_realized_block_additions(
    current_year,
    solved_capacity_by_tech,
    committed_state_payload,
    learning_cfg,
    network=None,
):
    current_year = int(current_year)
    prev_year = current_year - 5
    modeled_history = (committed_state_payload.get("modeled_capacity_history", {}) or {})
    cumulative_history = (committed_state_payload.get("capacity_history", {}) or {})
    realized = {}
    for technology, solved_capacity in (solved_capacity_by_tech or {}).items():
        if technology not in DEPLOYMENT_CONSTRAINT_SPECS:
            continue
        tech_modeled = modeled_history.get(technology, {}) or {}
        prev_modeled = float(tech_modeled.get(str(prev_year), 0.0))
        modeled_addition = max(float(solved_capacity) - prev_modeled, 0.0)
        constrained_addition = modeled_addition
        basis_unit = get_constraint_basis_unit(technology)
        battery_phi_block = np.nan
        if network is not None:
            constrained_info = get_current_vintage_constraint_basis_value(
                network,
                technology,
                current_year,
                learning_cfg,
            )
            constrained_addition = float(constrained_info["constrained_basis_value"])
            basis_unit = str(constrained_info["constraint_basis_unit"])
            battery_phi_block = constrained_info["battery_phi_block"]
        elif technology == "battery_energy" and current_year > prev_year:
            battery_phi_block = get_battery_phi_block(learning_cfg, current_year)
            constrained_addition = modeled_addition * battery_phi_block
        realized[technology] = {
            "technology": technology,
            "modeled_block_addition": modeled_addition,
            "constrained_basis_addition": constrained_addition,
            "battery_phi_block": battery_phi_block,
            "constraint_basis_unit": basis_unit,
            "previous_modeled_capacity": prev_modeled,
            "current_modeled_capacity": float(solved_capacity),
            "previous_constrained_capacity": float((cumulative_history.get(technology, {}) or {}).get(str(prev_year), np.nan)),
        }
    return realized
