# -*- coding: utf-8 -*-
"""Helpers for stochastic supply-chain deployment constraints."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


SUPPORTED_DEPLOYMENT_CONSTRAINT_MODES = {"first_order", "second_order", "both"}
SUPPORTED_DEPLOYMENT_CONSTRAINT_FORMULATIONS = {"hard_cap", "three_segment_wedge"}
PERCENT_OF_CAPEX_PENALTY_BASIS = "%_of_capex"

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
    formulation = str(cfg.get("formulation", "hard_cap")).strip()
    if formulation not in SUPPORTED_DEPLOYMENT_CONSTRAINT_FORMULATIONS:
        raise ValueError(
            "Unsupported learning.deployment_constraints.formulation="
            f"{formulation!r}. Supported values: "
            f"{sorted(SUPPORTED_DEPLOYMENT_CONSTRAINT_FORMULATIONS)}"
        )
    mode = str(cfg.get("mode", "first_order")).strip()
    if mode not in SUPPORTED_DEPLOYMENT_CONSTRAINT_MODES:
        raise ValueError(
            f"Unsupported learning.deployment_constraints.mode={mode!r}. "
            f"Supported values: {sorted(SUPPORTED_DEPLOYMENT_CONSTRAINT_MODES)}"
        )
    return cfg


def get_deployment_constraint_formulation(learning_cfg):
    cfg = get_deployment_constraint_cfg(learning_cfg)
    if cfg is None:
        return None
    return str(cfg.get("formulation", "hard_cap")).strip()


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
        "anchor_smoothing_years": int(calibration_cfg.get("anchor_smoothing_years", 1)),
        "quantile": float(calibration_cfg["quantile"]),
        "difference_space": str(
            calibration_cfg.get("difference_space", "log_annual_additions")
        ),
    }


def _matching_calibration_rows(calibration, request):
    if int(request.get("anchor_smoothing_years", 1)) != 1:
        # Existing support artifacts store a single endpoint anchor. Smoothed
        # anchors are derived from annual history at load time.
        return calibration.iloc[0:0].copy()
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
    anchor_smoothing_years = max(int(request.get("anchor_smoothing_years", 1)), 1)
    anchor_start_year = end_year - anchor_smoothing_years + 1
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
        anchor_window = window.loc[
            window["year"].between(anchor_start_year, end_year, inclusive="both")
        ].copy()
        if anchor_window.empty:
            raise ValueError(
                f"Cannot derive deployment calibration for {technology}: "
                f"missing anchor history in {anchor_start_year}-{end_year}."
            )
        anchor_additions = pd.to_numeric(
            anchor_window["annual_addition"], errors="coerce"
        ).dropna()
        anchor_growth = pd.to_numeric(
            anchor_window["first_diff"], errors="coerce"
        ).dropna()
        if anchor_additions.empty or anchor_growth.empty:
            raise ValueError(
                f"Cannot derive deployment calibration for {technology}: "
                f"missing smoothed anchor additions/growth in {anchor_start_year}-{end_year}."
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
        first_residuals = first_diff - first_q
        second_residuals = second_diff - second_q
        rows.append(
            {
                "technology": technology,
                "window_start_year": start_year,
                "window_end_year": end_year,
                "quantile": quantile,
                "annual_addition_anchor_2024": float(anchor_additions.mean()),
                # Keep the historical column names for compatibility; values are
                # resolved from the configured quantile at load time.
                "first_diff_q90": first_q,
                "second_diff_q90": second_q,
                "anchor_first_diff_2024": float(anchor_growth.mean()),
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


def _sigma_persistent_for_mode(row, mode):
    if mode == "first_order":
        return float(row["first_order_residual_std"])
    if mode == "second_order":
        return float(row["second_order_residual_std"])
    return max(
        float(row["first_order_residual_std"]),
        float(row["second_order_residual_std"]),
    )


def _historical_residual_series(history, technology, row, mode):
    tech_history = history.loc[history["technology"].eq(technology)].sort_values("year")
    if mode == "first_order":
        return pd.to_numeric(tech_history["first_diff"], errors="coerce") - float(row["first_diff_q90"])
    if mode == "second_order":
        return pd.to_numeric(tech_history["second_diff"], errors="coerce") - float(row["second_diff_q90"])
    raise ValueError(f"Unsupported residual mode for annual scale: {mode!r}")


def _annual_innovation_sigma(history, technology, row, mode):
    residuals = _historical_residual_series(history, technology, row, mode).dropna()
    if len(residuals) < 3:
        return np.nan
    return float(residuals.diff().dropna().std() / np.sqrt(2.0))


def _sigma_annual_for_mode(row, mode, technology, history, uncertainty_cfg):
    source = str(uncertainty_cfg.get("annual_scale_source", "configured_fraction")).strip()
    sigma_persistent = _sigma_persistent_for_mode(row, mode)

    if source in {"configured_fraction", "fraction_of_persistent"}:
        annual_fraction = float(uncertainty_cfg.get("annual_scale_fraction", 0.25))
        return float(max(annual_fraction, 0.0) * sigma_persistent), source, np.nan

    if source == "calibrated_residual_innovation":
        shrinkage = float(uncertainty_cfg.get("annual_scale_shrinkage", 1.0))
        if mode == "both":
            raw_sigma = max(
                _annual_innovation_sigma(history, technology, row, "first_order"),
                _annual_innovation_sigma(history, technology, row, "second_order"),
            )
        else:
            raw_sigma = _annual_innovation_sigma(history, technology, row, mode)
        if not np.isfinite(raw_sigma):
            annual_fraction = float(uncertainty_cfg.get("annual_scale_fraction", 0.25))
            fallback = float(max(annual_fraction, 0.0) * sigma_persistent)
            return fallback, "configured_fraction_fallback", np.nan
        return float(max(shrinkage, 0.0) * raw_sigma), source, float(raw_sigma)

    raise ValueError(
        "Unsupported learning.deployment_constraints.uncertainty."
        f"annual_scale_source={source!r}. Supported values are "
        "'configured_fraction' and 'calibrated_residual_innovation'."
    )


def _resolve_runtime_path(path_value, config_file=None):
    path = Path(path_value)
    if path.is_absolute():
        return path
    base = Path(config_file).resolve().parent if config_file else Path.cwd()
    return (base / path).resolve()


def _normalize_wedge_columns(frame):
    rename_map = {}
    for column in frame.columns:
        key = str(column).strip().lower()
        if key in {"country", "region"}:
            rename_map[column] = "region"
        elif key in {"technology", "tech"}:
            rename_map[column] = "technology"
        elif key in {"year", "period", "planning_year"}:
            rename_map[column] = "year"
        elif key in {"b1", "threshold_1", "normal_threshold"}:
            rename_map[column] = "b1"
        elif key in {"b2", "threshold_2", "stress_threshold"}:
            rename_map[column] = "b2"
        elif key in {"phi2", "penalty_2", "wedge_2"}:
            rename_map[column] = "phi2"
        elif key in {"phi3", "penalty_3", "wedge_3"}:
            rename_map[column] = "phi3"
        elif key in {"basis_unit", "unit"}:
            rename_map[column] = "basis_unit"
        elif key in {"penalty_basis", "penalty_type"}:
            rename_map[column] = "penalty_basis"
    return frame.rename(columns=rename_map)


def _normalize_country_history_columns(frame):
    rename_map = {}
    for column in frame.columns:
        key = str(column).strip().lower()
        if key in {"country", "region"}:
            rename_map[column] = "region"
        elif key in {"technology", "tech"}:
            rename_map[column] = "technology"
        elif key in {"year", "period", "planning_year"}:
            rename_map[column] = "year"
        elif key in {"annual_addition", "addition", "value", "reference_annual_addition"}:
            rename_map[column] = "annual_addition"
        elif key in {"basis_unit", "unit"}:
            rename_map[column] = "basis_unit"
    return frame.rename(columns=rename_map)


def _safe_float(value, default=np.nan):
    try:
        result = float(value)
    except Exception:
        return float(default)
    if np.isnan(result):
        return float(default)
    return result


def _block_year_count(current_year):
    return max(len(_annual_years_for_block(current_year)), 1)


def _get_wedge_cfg(learning_cfg):
    cfg = get_deployment_constraint_cfg(learning_cfg)
    if cfg is None or get_deployment_constraint_formulation(learning_cfg) != "three_segment_wedge":
        return {}
    wedge_cfg = (cfg.get("wedge", {}) or {})
    battery_basis = str(wedge_cfg.get("battery_basis", "local_energy")).strip()
    if battery_basis != "local_energy":
        raise ValueError(
            "three_segment_wedge currently supports only wedge.battery_basis='local_energy'."
        )
    return wedge_cfg


def get_deployment_wedge_basis_unit(technology):
    if str(technology) == "battery_energy":
        return "GWh"
    return str(get_constraint_basis_unit(technology))


def _resolve_history_addition_to_basis(
    technology,
    annual_addition,
    basis_unit,
    current_year,
    learning_cfg,
):
    target_unit = get_deployment_wedge_basis_unit(technology)
    spec = get_deployment_constraint_spec(technology)
    resolved = _safe_float(annual_addition)
    source_unit = str(basis_unit or "").strip()
    if not np.isfinite(resolved):
        return np.nan
    if source_unit == target_unit:
        return float(resolved)
    resolved = normalize_constraint_basis_value(technology, resolved)
    return float(resolved)


def _irena_capacity_history_to_country_annual_additions(irena_csv, technologies, current_year):
    frame = pd.read_csv(irena_csv)
    frame = frame.rename(columns={col: str(col).replace("\ufeff", "").strip() for col in frame.columns})
    wide_years = sorted(int(c) for c in frame.columns if isinstance(c, str) and c.isdigit())
    if not {"Technology", "Country"}.issubset(frame.columns) or len(wide_years) < 2:
        return pd.DataFrame(columns=["region", "technology", "year", "annual_addition", "basis_unit"])

    tech_map = {
        "solar_power": {"PV"},
        "onwind_power": {"Onshore"},
    }
    rows = []
    for technology in technologies:
        irena_techs = tech_map.get(str(technology), set())
        if not irena_techs:
            continue
        subset = frame.loc[frame["Technology"].astype(str).str.strip().isin(irena_techs)].copy()
        if subset.empty:
            continue
        subset["region"] = subset["Country"].fillna("").astype(str).str.strip().str.upper()
        subset = subset.loc[subset["region"].str.match(r"^[A-Z]{2}$", na=False)].copy()
        if subset.empty:
            continue
        numeric = subset.loc[:, [str(year) for year in wide_years]].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        additions = numeric.diff(axis=1)
        additions.columns = wide_years
        additions = additions.loc[:, [year for year in additions.columns if year <= int(current_year)]]
        if additions.empty:
            continue
        melted = additions.clip(lower=0.0).rename_axis(index="row_id", columns="year").stack().rename("annual_addition").reset_index()
        melted["region"] = subset.iloc[melted["row_id"].to_numpy()]["region"].to_numpy()
        rows.append(
            melted.loc[:, ["region", "year", "annual_addition"]].assign(
                technology=str(technology),
                basis_unit="MW",
            )
        )
    if not rows:
        return pd.DataFrame(columns=["region", "technology", "year", "annual_addition", "basis_unit"])
    return pd.concat(rows, ignore_index=True)


def _battery_capacity_history_to_country_annual_additions(battery_csv, current_year):
    frame = pd.read_csv(battery_csv)
    if "country" not in frame.columns or "capa_2020" not in frame.columns or "capa_2025" not in frame.columns:
        return pd.DataFrame(columns=["region", "technology", "year", "annual_addition", "basis_unit"])
    frame["region"] = frame["country"].fillna("").astype(str).str.strip().str.upper()
    frame = frame.loc[frame["region"].str.match(r"^[A-Z]{2}$", na=False)].copy()
    if frame.empty:
        return pd.DataFrame(columns=["region", "technology", "year", "annual_addition", "basis_unit"])
    annual = (
        pd.to_numeric(frame["capa_2025"], errors="coerce").fillna(0.0)
        - pd.to_numeric(frame["capa_2020"], errors="coerce").fillna(0.0)
    ) / 5.0
    rows = []
    for year in range(2021, min(int(current_year), 2025) + 1):
        rows.append(
            frame.loc[:, ["region"]].assign(
                technology="battery_energy",
                year=year,
                annual_addition=annual.clip(lower=0.0).astype(float),
                basis_unit="MWh",
            )
        )
    if not rows:
        return pd.DataFrame(columns=["region", "technology", "year", "annual_addition", "basis_unit"])
    return pd.concat(rows, ignore_index=True)


def load_deployment_country_history(
    learning_cfg,
    current_year,
    technologies=None,
    config_file=None,
):
    wedge_cfg = _get_wedge_cfg(learning_cfg)
    cfg = get_deployment_constraint_cfg(learning_cfg) or {}
    techs = list(technologies or cfg.get("technologies", []) or [])
    history_csv = wedge_cfg.get("country_history_csv")
    if history_csv:
        history = _normalize_country_history_columns(
            pd.read_csv(_resolve_runtime_path(history_csv, config_file=config_file))
        )
    else:
        irena_csv = _resolve_runtime_path(
            wedge_cfg.get("irena_history_csv", "validation/data/irena_capacity_by_technology.csv"),
            config_file=config_file,
        )
        battery_csv = _resolve_runtime_path(
            wedge_cfg.get("battery_history_csv", "data/energy_storage/battery_storage_capa_bycountry.csv"),
            config_file=config_file,
        )
        history = pd.concat(
            [
                _irena_capacity_history_to_country_annual_additions(irena_csv, techs, current_year),
                _battery_capacity_history_to_country_annual_additions(battery_csv, current_year),
            ],
            ignore_index=True,
        )

    if history.empty:
        return pd.DataFrame(columns=["region", "technology", "year", "annual_addition", "basis_unit"])
    history = _normalize_country_history_columns(history)
    history["region"] = history["region"].fillna("").astype(str).str.strip().str.upper()
    history["technology"] = history["technology"].fillna("").astype(str).str.strip()
    history["year"] = pd.to_numeric(history["year"], errors="coerce")
    history["annual_addition"] = pd.to_numeric(history["annual_addition"], errors="coerce")
    if "basis_unit" not in history.columns:
        history["basis_unit"] = history["technology"].map(get_constraint_basis_unit)
    history["basis_unit"] = history["basis_unit"].fillna("").astype(str).str.strip()
    history = history.dropna(subset=["year", "annual_addition"])
    history = history.loc[history["region"].str.match(r"^[A-Z]{2}$", na=False)].copy()
    if technologies is not None:
        tech_set = {str(t).strip() for t in technologies}
        history = history.loc[history["technology"].isin(tech_set)].copy()
    history["year"] = history["year"].astype(int)
    return history.sort_values(["technology", "region", "year"], ignore_index=True)


def build_deployment_wedge_table_from_history(
    learning_cfg,
    current_year,
    technologies=None,
    config_file=None,
):
    history = load_deployment_country_history(
        learning_cfg,
        current_year=current_year,
        technologies=technologies,
        config_file=config_file,
    )
    empty_cols = [
        "region",
        "technology",
        "year",
        "b1",
        "b2",
        "phi2",
        "phi3",
        "width1",
        "width2",
        "basis_unit",
        "penalty_basis",
        "history_year",
        "reference_annual_addition",
    ]
    if history.empty:
        return pd.DataFrame(columns=empty_cols)

    wedge_cfg = _get_wedge_cfg(learning_cfg)
    current_year = int(current_year)
    block_years = _block_year_count(current_year)
    b1_multiplier = float(wedge_cfg.get("b1_multiplier", 0.8))
    b2_multiplier = float(wedge_cfg.get("b2_multiplier", 1.2))
    phi2_pct_capex = float(wedge_cfg.get("phi2_pct_capex", 0.5))
    phi3_pct_capex = float(wedge_cfg.get("phi3_pct_capex", 1.0))
    smoothing_years = int(
        wedge_cfg.get(
            "anchor_smoothing_years",
            ((get_deployment_constraint_cfg(learning_cfg) or {}).get("calibration", {}) or {}).get("anchor_smoothing_years", 3),
        )
    )
    smoothing_years = max(smoothing_years, 1)
    reference_statistic = str(wedge_cfg.get("reference_statistic", "mean")).strip().lower()
    if reference_statistic not in {"mean", "median", "max_mean_median"}:
        raise ValueError(
            "Unsupported learning.deployment_constraints.wedge.reference_statistic="
            f"{reference_statistic!r}. Supported values: ['mean', 'median', 'max_mean_median']"
        )

    rows = []
    for (region, technology), group in history.groupby(["region", "technology"], sort=False):
        smoothed = group.sort_values("year").copy()
        smoothed["annual_basis"] = smoothed.apply(
            lambda row: _resolve_history_addition_to_basis(
                technology=technology,
                annual_addition=row["annual_addition"],
                basis_unit=row.get("basis_unit", ""),
                current_year=current_year,
                learning_cfg=learning_cfg,
            ),
            axis=1,
        )
        recent = smoothed.dropna(subset=["annual_basis"]).tail(smoothing_years).copy()
        if recent.empty:
            continue
        mean_recent = float(recent["annual_basis"].mean())
        median_recent = float(recent["annual_basis"].median())
        if reference_statistic == "max_mean_median":
            annual_basis = max(mean_recent, median_recent)
        elif reference_statistic == "median":
            annual_basis = median_recent
        else:
            annual_basis = mean_recent
        if not np.isfinite(annual_basis) or annual_basis <= 0.0:
            continue
        block_reference = float(annual_basis) * float(block_years)
        rows.append(
            {
                "region": region,
                "technology": technology,
                "year": current_year,
                "b1": max(block_reference * b1_multiplier, 0.0),
                "b2": max(block_reference * b2_multiplier, 0.0),
                "phi2": max(phi2_pct_capex, 0.0),
                "phi3": max(phi3_pct_capex, 0.0),
                "basis_unit": get_deployment_wedge_basis_unit(technology),
                "penalty_basis": PERCENT_OF_CAPEX_PENALTY_BASIS,
                "history_year": int(recent["year"].max()),
                "reference_annual_addition": float(annual_basis),
            }
        )
    if not rows:
        return pd.DataFrame(columns=empty_cols)
    frame = pd.DataFrame(rows)
    frame["width1"] = frame["b1"].astype(float)
    frame["width2"] = (frame["b2"] - frame["b1"]).clip(lower=0.0).astype(float)
    return frame.sort_values(["technology", "region"], ignore_index=True)


def load_deployment_wedge_table(
    learning_cfg,
    current_year=None,
    technologies=None,
    config_file=None,
):
    cfg = get_deployment_constraint_cfg(learning_cfg)
    if cfg is None or get_deployment_constraint_formulation(learning_cfg) != "three_segment_wedge":
        return pd.DataFrame(
            columns=[
                "region",
                "technology",
                "year",
                "b1",
                "b2",
                "phi2",
                "phi3",
                "width1",
                "width2",
                "basis_unit",
                "penalty_basis",
                "history_year",
                "reference_annual_addition",
            ]
        )

    wedge_cfg = cfg.get("wedge", {}) or {}
    table_csv = wedge_cfg.get("table_csv")
    if table_csv:
        frame = _normalize_wedge_columns(
            pd.read_csv(_resolve_runtime_path(table_csv, config_file=config_file))
        )
    else:
        frame = build_deployment_wedge_table_from_history(
            learning_cfg,
            current_year=current_year,
            technologies=technologies,
            config_file=config_file,
        )

    required = {"region", "technology", "year", "b1", "b2", "phi2", "phi3"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Deployment wedge table is missing required columns: "
            f"{sorted(missing)}"
        )

    keep = required | {"basis_unit", "penalty_basis", "history_year", "reference_annual_addition"}
    frame = frame.loc[:, [col for col in frame.columns if col in keep]].copy()
    frame["region"] = frame["region"].fillna("").astype(str).str.strip().str.upper()
    frame["technology"] = frame["technology"].fillna("").astype(str).str.strip()
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    for col in ("b1", "b2", "phi2", "phi3"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "basis_unit" not in frame.columns:
        frame["basis_unit"] = frame["technology"].map(get_deployment_wedge_basis_unit)
    frame["basis_unit"] = frame["basis_unit"].fillna("").astype(str).str.strip()
    if "penalty_basis" not in frame.columns:
        frame["penalty_basis"] = PERCENT_OF_CAPEX_PENALTY_BASIS
    frame["penalty_basis"] = frame["penalty_basis"].fillna(PERCENT_OF_CAPEX_PENALTY_BASIS).astype(str).str.strip()
    frame = frame.dropna(subset=["year", "b1", "b2", "phi2", "phi3"])
    if current_year is not None:
        frame = frame.loc[frame["year"].astype(int).eq(int(current_year))].copy()
    if technologies is not None:
        techs = {str(t).strip() for t in technologies}
        frame = frame.loc[frame["technology"].isin(techs)].copy()
    expected_basis_units = frame["technology"].map(get_deployment_wedge_basis_unit)
    unit_mismatch = expected_basis_units.ne(frame["basis_unit"])
    if unit_mismatch.any():
        mismatch_rows = frame.loc[unit_mismatch, ["region", "technology", "year", "basis_unit"]].head(5)
        raise ValueError(
            "Deployment wedge table basis_unit must match the local wedge basis. "
            f"Examples: {mismatch_rows.to_dict(orient='records')}"
        )
    invalid = frame.loc[
        frame["b1"].lt(0.0)
        | frame["b2"].lt(frame["b1"])
        | frame["phi2"].lt(0.0)
        | frame["phi3"].lt(frame["phi2"])
    ]
    if not invalid.empty:
        raise ValueError(
            "Deployment wedge table must satisfy b2 >= b1 >= 0 and phi3 >= phi2 >= 0. "
            f"First invalid row: {invalid.iloc[0].to_dict()}"
        )
    frame["year"] = frame["year"].astype(int)
    frame["width1"] = frame["b1"].astype(float)
    frame["width2"] = (frame["b2"] - frame["b1"]).clip(lower=0.0).astype(float)
    duplicated = frame.duplicated(subset=["region", "technology", "year"], keep=False)
    if duplicated.any():
        dup_rows = frame.loc[duplicated, ["region", "technology", "year"]].drop_duplicates()
        raise ValueError(
            "Deployment wedge table contains duplicate (region, technology, year) rows: "
            f"{dup_rows.to_dict(orient='records')}"
        )
    return frame.sort_values(["technology", "year", "region"], ignore_index=True)


def _get_bus_country_lookup(n):
    buses = getattr(n, "buses", pd.DataFrame())
    if buses.empty:
        return pd.Series(dtype=object)
    country = buses["country"] if "country" in buses.columns else pd.Series("", index=buses.index)
    location = buses["location"] if "location" in buses.columns else pd.Series("", index=buses.index)
    country = country.fillna("").astype(str).str.strip().str.upper()
    location = location.fillna("").astype(str).str.strip().str.upper()
    bus_name = buses.index.to_series(index=buses.index).astype(str).str.strip().str.upper()
    inferred = location.str.extract(r"^([A-Z]{2})(?:\b|\s|[-_])", expand=False).fillna("")
    country = country.where(country.ne(""), inferred)
    inferred_name = bus_name.str.extract(r"^([A-Z]{2})(?:\b|\s|[-_])", expand=False).fillna("")
    return country.where(country.ne(""), inferred_name).fillna("").astype(str)


def _basis_cost_from_capital_cost(technology, capital_cost, planning_year, learning_cfg):
    spec = get_deployment_constraint_spec(technology)
    basis_cost = float(capital_cost) * float(spec["raw_to_basis_divisor"])
    return float(basis_cost)


def build_current_vintage_country_technology_asset_table(
    n,
    technology,
    planning_year,
    learning_cfg,
):
    spec = get_deployment_constraint_spec(technology)
    idx = get_current_vintage_indices(n, technology, planning_year)
    columns = [
        "asset",
        "region",
        "technology",
        "model_var",
        "raw_to_basis_divisor",
        "capital_cost",
        "basis_objective_cost",
        "basis_unit",
        "battery_phi_block",
    ]
    if len(idx) == 0:
        return pd.DataFrame(columns=columns)
    component_df = getattr(n, spec["component_attr"], None)
    if component_df is None or component_df.empty or "bus" not in component_df.columns:
        return pd.DataFrame(columns=columns)
    subset = component_df.loc[idx].copy()
    subset["region"] = subset["bus"].map(_get_bus_country_lookup(n)).fillna("").astype(str).str.strip().str.upper()
    subset = subset.loc[subset["region"].str.match(r"^[A-Z]{2}$", na=False)].copy()
    if subset.empty:
        return pd.DataFrame(columns=columns)
    if "capital_cost" in subset.columns:
        subset["capital_cost"] = pd.to_numeric(subset["capital_cost"], errors="coerce")
    else:
        subset["capital_cost"] = np.nan
    subset["basis_objective_cost"] = subset["capital_cost"].apply(
        lambda value: _basis_cost_from_capital_cost(technology, value, planning_year, learning_cfg)
    )
    subset["asset"] = subset.index.astype(str)
    subset["technology"] = str(technology)
    subset["model_var"] = str(spec["model_var"])
    subset["raw_to_basis_divisor"] = float(spec["raw_to_basis_divisor"])
    subset["basis_unit"] = get_deployment_wedge_basis_unit(technology)
    subset["battery_phi_block"] = get_battery_phi_block(learning_cfg, planning_year) if spec["phi_mapped"] else np.nan
    return subset.loc[:, columns].reset_index(drop=True)


def _get_current_vintage_basis_build_by_region(n, technology, planning_year, learning_cfg):
    spec = get_deployment_constraint_spec(technology)
    idx = get_current_vintage_indices(n, technology, planning_year)
    if len(idx) == 0:
        return pd.Series(dtype=float)
    component_df = getattr(n, spec["component_attr"], None)
    if component_df is None or component_df.empty or "bus" not in component_df.columns:
        return pd.Series(dtype=float)
    subset = component_df.loc[idx].copy()
    subset["optimized_capacity"] = _get_capacity_series(subset, spec)
    subset["region"] = subset["bus"].map(_get_bus_country_lookup(n)).fillna("").astype(str).str.strip().str.upper()
    subset = subset.loc[subset["region"].str.match(r"^[A-Z]{2}$", na=False)].copy()
    if subset.empty:
        return pd.Series(dtype=float)
    subset["basis_value"] = subset["optimized_capacity"].astype(float) / float(spec["raw_to_basis_divisor"])
    return subset.groupby("region")["basis_value"].sum().sort_index()


def partition_three_segment_build(build_value, width1, width2):
    build_value = max(float(build_value), 0.0)
    width1 = max(float(width1), 0.0)
    width2 = max(float(width2), 0.0)
    seg1 = min(build_value, width1)
    seg2 = min(max(build_value - seg1, 0.0), width2)
    seg3 = max(build_value - seg1 - seg2, 0.0)
    return float(seg1), float(seg2), float(seg3)


def summarize_realized_deployment_wedge_rows(
    n,
    current_year,
    learning_cfg,
    technologies=None,
    config_file=None,
):
    wedge_table = load_deployment_wedge_table(
        learning_cfg,
        current_year=current_year,
        technologies=technologies,
        config_file=config_file,
    )
    if wedge_table.empty:
        return pd.DataFrame()

    rows = []
    for technology in sorted(wedge_table["technology"].unique()):
        realized_by_region = _get_current_vintage_basis_build_by_region(
            n,
            technology,
            planning_year=current_year,
            learning_cfg=learning_cfg,
        )
        asset_table = build_current_vintage_country_technology_asset_table(
            n,
            technology,
            planning_year=current_year,
            learning_cfg=learning_cfg,
        )
        basis_cost_by_region = {}
        if not asset_table.empty:
            basis_cost_by_region = asset_table.groupby("region")["basis_objective_cost"].mean().to_dict()
        tech_wedge = wedge_table.loc[wedge_table["technology"].eq(technology)]
        for _, wedge_row in tech_wedge.iterrows():
            region = wedge_row["region"]
            realized = float(realized_by_region.get(region, 0.0))
            seg1, seg2, seg3 = partition_three_segment_build(
                realized,
                wedge_row["width1"],
                wedge_row["width2"],
            )
            penalty_basis = str(wedge_row.get("penalty_basis", PERCENT_OF_CAPEX_PENALTY_BASIS))
            basis_cost = _safe_float(basis_cost_by_region.get(region, np.nan))
            phi2 = float(wedge_row["phi2"])
            phi3 = float(wedge_row["phi3"])
            if penalty_basis == PERCENT_OF_CAPEX_PENALTY_BASIS and np.isfinite(basis_cost):
                phi2_cost = phi2 * basis_cost
                phi3_cost = phi3 * basis_cost
            else:
                phi2_cost = phi2
                phi3_cost = phi3
            rows.append(
                {
                    "year": int(current_year),
                    "country": region,
                    "technology": technology,
                    "constraint_basis_unit": str(wedge_row.get("basis_unit", get_constraint_basis_unit(technology))),
                    "penalty_basis": penalty_basis,
                    "b1": float(wedge_row["b1"]),
                    "b2": float(wedge_row["b2"]),
                    "width1": float(wedge_row["width1"]),
                    "width2": float(wedge_row["width2"]),
                    "phi2": phi2,
                    "phi3": phi3,
                    "realized_block_addition_constrained_basis": realized,
                    "realized_seg1": seg1,
                    "realized_seg2": seg2,
                    "realized_seg3": seg3,
                    "realized_wedge_cost_eur": float(phi2_cost * seg2 + phi3_cost * seg3),
                    "history_year": _safe_float(wedge_row.get("history_year", np.nan)),
                    "reference_annual_addition": _safe_float(wedge_row.get("reference_annual_addition", np.nan)),
                }
            )
    return pd.DataFrame(rows)


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

    calibration, history, _ = load_deployment_constraint_support(learning_cfg, config_file=config_file)
    techs = list(cfg.get("technologies", [])) or list(calibration["technology"].unique())
    mode = str(cfg.get("mode", "first_order")).strip()
    annual_years = _annual_years_for_block(current_year)
    uncertainty_cfg = (cfg.get("uncertainty", {}) or {})
    uncertainty_enabled = bool(uncertainty_cfg.get("enabled", False)) and str(
        uncertainty_cfg.get("mode", "stochastic_draw")
    ).strip() == "stochastic_draw"

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
        sigma_persistent = _sigma_persistent_for_mode(row, mode)
        sigma_annual, annual_scale_source, annual_innovation_sigma = _sigma_annual_for_mode(
            row,
            mode,
            technology,
            history,
            uncertainty_cfg,
        )
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
            "persistent_shock_sigma": sigma_persistent,
            "persistent_shock": persistent_shock,
            "annual_scale_source": annual_scale_source,
            "annual_innovation_sigma": annual_innovation_sigma,
            "annual_shock_sigma": sigma_annual,
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
