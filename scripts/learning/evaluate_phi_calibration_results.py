#!/usr/bin/env python3
"""Evaluate phi2/phi3 deployment-wedge calibration results."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from submit_learning_phi_calibration_array import (
    DEFAULT_RESULTS_ROOT,
    ROOT_DIR,
    build_tasks,
    phi_calibration_output_paths,
    phi_calibration_solve_target,
    phi_calibration_target,
)


PHI_TASK_RE = re.compile(r"^phi2_(?P<phi2>[0-9]+p[0-9]+)_phi3_(?P<phi3>[0-9]+p[0-9]+)$")
DEFAULT_TECHNOLOGIES = ("solar_power", "onwind_power")
MODEL_ADDITION_METRIC = "realized_block_addition_constrained_basis"
TARGET_ADDITION_KIND = "irena_capacity_delta"
TARGET_BASIS_UNIT = "GW"
NETWORK_CARRIER_BY_TECHNOLOGY = {
    "solar_power": "solar",
    "onwind_power": "onwind",
}
IRENA_PRODUCT_MAP = {
    "solar_power": "Solar photovoltaic",
    "onwind_power": "Onshore wind energy",
}
IRENA_LONG_REQUIRED_COLUMNS = {
    "Region/area (ISO 3)",
    "Year",
    "Product Name",
    "Grid Type",
    "Unit",
}


def _decode_phi_token(token: str) -> float:
    return float(str(token).replace("p", "."))


def _safe_float(value, default=math.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _sum_column(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return math.nan
    return float(pd.to_numeric(frame[column], errors="coerce").sum())


def _parse_numeric_capacity_value(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def _first_number(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return math.nan
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    if series.empty:
        return math.nan
    return float(series.iloc[0])


def _min_column(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return math.nan
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    if series.empty:
        return math.nan
    return float(series.min())


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_tasks_from_manifest(manifest_path: Path) -> list[dict]:
    tasks = _read_json(manifest_path)
    if not isinstance(tasks, list):
        raise ValueError(f"Expected task manifest to contain a list: {manifest_path}")
    return tasks


def parse_technologies(raw_values: list[str] | str | None) -> list[str]:
    if raw_values is None:
        return list(DEFAULT_TECHNOLOGIES)
    if isinstance(raw_values, str):
        raw_values = [raw_values]
    technologies = []
    for value in raw_values:
        for part in str(value).split(","):
            technology = part.strip()
            if technology:
                technologies.append(technology)
    return list(dict.fromkeys(technologies)) or list(DEFAULT_TECHNOLOGIES)


def _normalize_csv_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).strip().lstrip("\ufeff") for column in frame.columns]
    return frame


def _capacity_unit_to_gw_multiplier(unit: str) -> float:
    unit = str(unit).strip().lower()
    if unit in {"gw", "gigawatt", "gigawatts"}:
        return 1.0
    if unit in {"mw", "megawatt", "megawatts"}:
        return 1e-3
    raise ValueError(f"Unsupported IRENA capacity unit for phi target conversion: {unit!r}")


def _load_irena_target_block_additions(frame: pd.DataFrame, planning_year: int) -> dict[str, float]:
    frame = frame.copy()
    frame["region_code"] = frame["Region/area (ISO 3)"].fillna("").astype(str).str.strip().str.upper()
    frame["year"] = pd.to_numeric(frame["Year"], errors="coerce")
    frame["product_name"] = frame["Product Name"].fillna("").astype(str).str.strip()
    frame["capacity_value"] = _parse_numeric_capacity_value(frame["Value"])
    frame = frame.loc[frame["region_code"].eq("GLO")].dropna(subset=["year", "capacity_value"]).copy()
    frame["year"] = frame["year"].astype(int)
    previous_year = int(planning_year) - 5

    targets = {}
    for technology, product_name in IRENA_PRODUCT_MAP.items():
        subset = frame.loc[
            frame["product_name"].eq(product_name)
            & frame["year"].isin([previous_year, int(planning_year)])
        ].copy()
        if subset.empty:
            raise ValueError(f"IRENA target CSV is missing global rows for {product_name!r}")
        unit_names = {str(unit).strip() for unit in subset["Unit"].dropna().unique()}
        multipliers = {_capacity_unit_to_gw_multiplier(unit) for unit in unit_names}
        if len(multipliers) != 1:
            raise ValueError(
                f"IRENA rows for {product_name!r} use mixed capacity units: {sorted(unit_names)}"
            )
        multiplier = multipliers.pop()
        capacity = subset.groupby("year", sort=True)["capacity_value"].sum() * multiplier
        missing_years = {previous_year, int(planning_year)} - set(capacity.index)
        if missing_years:
            raise ValueError(
                f"IRENA target CSV is missing {product_name!r} capacity for years {sorted(missing_years)}"
            )
        targets[technology] = max(
            float(capacity.loc[int(planning_year)] - capacity.loc[previous_year]),
            0.0,
        )
    missing_technologies = set(IRENA_PRODUCT_MAP) - set(targets)
    if missing_technologies:
        raise ValueError(
            f"IRENA target CSV did not produce targets for {sorted(missing_technologies)}"
        )
    return targets


def _validate_target_comparison_basis(metrics: dict, technology: str) -> None:
    unit = str(metrics.get("constraint_basis_unit") or "").strip()
    if unit and unit != TARGET_BASIS_UNIT:
        raise ValueError(
            f"Cannot compare IRENA {TARGET_ADDITION_KIND} target for {technology!r}: "
            f"model metric {MODEL_ADDITION_METRIC!r} is reported in {unit!r}, "
            f"expected {TARGET_BASIS_UNIT!r}"
        )


def load_target_block_additions(path: Path | None, planning_year: int = 2025) -> dict[str, float]:
    if path is None:
        raise ValueError("--target-block-additions-csv is required and must point to an IRENA capacity CSV")
    frame = _normalize_csv_columns(pd.read_csv(path))
    required = set(IRENA_LONG_REQUIRED_COLUMNS) | {"Value"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "Target CSV must be a raw IRENA long-format capacity CSV; "
            f"missing required columns {sorted(missing)}"
        )
    return _load_irena_target_block_additions(frame, planning_year=planning_year)


def _latest_submission_dir(submit_root: Path) -> Path:
    candidates = [
        path
        for path in submit_root.glob("phi_calibration_*")
        if path.is_dir() and (path / "task_manifest.json").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No phi_calibration_* submissions found under {submit_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_manifest(args: argparse.Namespace) -> Path | None:
    if args.manifest:
        return Path(args.manifest).resolve()
    if args.submission_dir:
        return (Path(args.submission_dir).resolve() / "task_manifest.json")
    if args.latest_submission:
        return _latest_submission_dir(Path(args.submit_root).resolve()) / "task_manifest.json"
    return None


def _task_from_result_dir(path: Path, results_root: Path) -> dict | None:
    match = PHI_TASK_RE.match(path.name)
    if not match:
        return None
    phi2 = _decode_phi_token(match.group("phi2"))
    phi3 = _decode_phi_token(match.group("phi3"))
    try:
        rel = path.resolve().relative_to((ROOT_DIR / "results").resolve())
        sector_name = str(rel)
    except ValueError:
        sector_name = str(Path("Global_200") / "phi_calibration" / path.name)
    return {
        "name": path.name,
        "sector_name": sector_name,
        "phi2_pct_capex": phi2,
        "phi3_pct_capex": phi3,
        "b1_multiplier": math.nan,
        "b2_multiplier": math.nan,
        "planning_horizons": [2020, 2025],
    }


def discover_tasks(results_root: Path) -> list[dict]:
    if not results_root.exists():
        return []
    tasks = []
    for path in sorted(results_root.iterdir(), key=lambda item: item.name):
        if not path.is_dir() or path.name.startswith("_"):
            continue
        task = _task_from_result_dir(path, results_root)
        if task is not None:
            tasks.append(task)
    return tasks


def default_tasks_if_requested(args: argparse.Namespace) -> list[dict]:
    return build_tasks(
        phi2_min=args.phi2_min,
        phi2_max=args.phi2_max,
        phi2_count=args.phi2_count,
        phi3_min=args.phi3_min,
        phi3_max=args.phi3_max,
        phi3_count=args.phi3_count,
        model=args.model,
        seed=args.seed,
    )


def deployment_constraint_path(task: dict) -> Path:
    for path in phi_calibration_output_paths(task):
        candidate = Path(path)
        if candidate.name.startswith("deployment_constraints_"):
            return ROOT_DIR / candidate
    raise ValueError(f"Could not resolve deployment constraint path for {task['name']}")


def system_cost_path(task: dict) -> Path:
    for path in phi_calibration_output_paths(task):
        candidate = Path(path)
        if candidate.name.startswith("system_costs_"):
            return ROOT_DIR / candidate
    raise ValueError(f"Could not resolve system-cost path for {task['name']}")


def solved_network_path(task: dict) -> Path:
    return ROOT_DIR / phi_calibration_solve_target(task)


def required_evaluation_output_paths(task: dict) -> list[Path]:
    return [solved_network_path(task)]


def _network_current_vintage_addition_metrics(
    network,
    technology: str,
    planning_year: int,
) -> dict[str, float | str | bool]:
    carrier = NETWORK_CARRIER_BY_TECHNOLOGY.get(str(technology))
    if carrier is None:
        return {
            "deployment_constraints_exists": False,
            "deployment_metrics_source": "none",
            "deployment_rows": 0,
        }
    generators = getattr(network, "generators", pd.DataFrame())
    if generators.empty:
        realized = 0.0
        rows = 0
    else:
        subset = generators.loc[generators["carrier"].astype(str).eq(carrier)].copy()
        if "p_nom_extendable" in subset.columns:
            subset = subset.loc[subset["p_nom_extendable"].fillna(False).astype(bool)].copy()
        if "build_year" in subset.columns:
            build_year = pd.to_numeric(subset["build_year"], errors="coerce")
            subset = subset.loc[build_year.fillna(-1).astype(int).eq(int(planning_year))].copy()
        capacity_col = "p_nom_opt" if "p_nom_opt" in subset.columns else "p_nom"
        realized = float(pd.to_numeric(subset.get(capacity_col, 0.0), errors="coerce").fillna(0.0).sum() / 1e3)
        rows = int(len(subset))
    return {
        "deployment_constraints_exists": False,
        "deployment_metrics_source": "solved_network",
        "deployment_rows": rows,
        "realized_block_addition_constrained_basis": realized,
        "realized_block_addition_modeled": realized,
        "realized_seg1": math.nan,
        "realized_seg2": math.nan,
        "realized_seg3": math.nan,
        "realized_wedge_cost_eur": math.nan,
        "binding_slack_min": math.nan,
        "b1": math.nan,
        "b2": math.nan,
        "allowed_block_addition": math.nan,
        "battery_phi_block": math.nan,
        "constraint_basis_unit": TARGET_BASIS_UNIT,
    }


def _read_network_deployment_metrics(path: Path, technology: str, planning_year: int) -> dict[str, float | str | bool]:
    if not path.is_file():
        return {"deployment_constraints_exists": False, "deployment_metrics_source": "none"}
    import pypsa

    network = pypsa.Network(str(path))
    return _network_current_vintage_addition_metrics(network, technology, planning_year)


def _read_deployment_metrics(
    path: Path,
    technology: str,
    planning_year: int,
    network=None,
    network_path: Path | None = None,
) -> dict[str, float | str | bool]:
    if not path.is_file():
        if network is not None:
            return _network_current_vintage_addition_metrics(network, technology, planning_year)
        if network_path is not None:
            return _read_network_deployment_metrics(network_path, technology, planning_year)
        return {"deployment_constraints_exists": False, "deployment_metrics_source": "none"}
    frame = pd.read_csv(path)
    if "technology" in frame.columns:
        frame = frame.loc[frame["technology"].astype(str).eq(technology)]
    if "year" in frame.columns:
        years = pd.to_numeric(frame["year"], errors="coerce")
        frame = frame.loc[years.eq(int(planning_year))]
    return {
        "deployment_constraints_exists": True,
        "deployment_metrics_source": "deployment_constraints_csv",
        "deployment_rows": int(len(frame)),
        "realized_block_addition_constrained_basis": _sum_column(
            frame,
            "realized_block_addition_constrained_basis",
        ),
        "realized_block_addition_modeled": _sum_column(frame, "realized_block_addition_modeled"),
        "realized_seg1": _sum_column(frame, "realized_seg1"),
        "realized_seg2": _sum_column(frame, "realized_seg2"),
        "realized_seg3": _sum_column(frame, "realized_seg3"),
        "realized_wedge_cost_eur": _sum_column(frame, "realized_wedge_cost_eur"),
        "binding_slack_min": _min_column(frame, "binding_slack"),
        "b1": _first_number(frame, "b1"),
        "b2": _first_number(frame, "b2"),
        "allowed_block_addition": _first_number(frame, "allowed_block_addition"),
        "battery_phi_block": _first_number(frame, "battery_phi_block"),
        "constraint_basis_unit": str(frame["constraint_basis_unit"].dropna().iloc[0])
        if "constraint_basis_unit" in frame.columns and not frame["constraint_basis_unit"].dropna().empty
        else "",
    }


def _read_system_cost_metrics(path: Path) -> dict[str, float | bool]:
    if not path.is_file():
        return {"system_costs_exists": False}
    frame = pd.read_csv(path)
    return {
        "system_costs_exists": True,
        "total_system_cost_eur": _sum_column(frame, "total_system_cost_eur"),
        "annualized_capital_cost_eur": _sum_column(frame, "annualized_capital_cost_eur"),
        "operating_cost_eur": _sum_column(frame, "operating_cost_eur"),
    }


def evaluate_task(task: dict, technology: str, planning_year: int, target_block_addition: float | None) -> dict:
    solved_cost_log = ROOT_DIR / phi_calibration_target(task)
    deploy_path = deployment_constraint_path(task)
    system_path = system_cost_path(task)
    network_path = solved_network_path(task)
    missing_outputs = [
        str(path)
        for path in required_evaluation_output_paths(task)
        if not path.exists()
    ]
    row = {
        "name": task["name"],
        "sector_name": task["sector_name"],
        "phi2_pct_capex": _safe_float(task.get("phi2_pct_capex")),
        "phi3_pct_capex": _safe_float(task.get("phi3_pct_capex")),
        "planning_year": int(planning_year),
        "solved_cost_log_exists": solved_cost_log.is_file(),
        "solved_network_exists": network_path.is_file(),
        "complete": len(missing_outputs) == 0,
        "missing_output_count": len(missing_outputs),
        "first_missing_output": missing_outputs[0] if missing_outputs else "",
    }
    row.update(_read_deployment_metrics(deploy_path, technology, planning_year, network_path=network_path))
    row.update(_read_system_cost_metrics(system_path))
    if target_block_addition is not None:
        _validate_target_comparison_basis(row, technology)
        realized = _safe_float(row.get(MODEL_ADDITION_METRIC))
        row["target_block_addition"] = float(target_block_addition)
        row["target_addition_kind"] = TARGET_ADDITION_KIND
        row["target_basis_unit"] = TARGET_BASIS_UNIT
        row["target_comparison_model_metric"] = MODEL_ADDITION_METRIC
        row["target_block_start_year"] = int(planning_year) - 5
        row["target_block_end_year"] = int(planning_year)
        row["target_error"] = realized - float(target_block_addition)
        row["target_abs_error"] = abs(row["target_error"]) if math.isfinite(realized) else math.nan
    return row


def _prefix_metrics(metrics: dict, technology: str) -> dict:
    return {f"{technology}__{key}": value for key, value in metrics.items()}


def evaluate_task_multi(
    task: dict,
    technologies: list[str],
    planning_year: int,
    target_block_additions: dict[str, float],
) -> dict:
    solved_cost_log = ROOT_DIR / phi_calibration_target(task)
    deploy_path = deployment_constraint_path(task)
    system_path = system_cost_path(task)
    network_path = solved_network_path(task)
    missing_outputs = [
        str(path)
        for path in required_evaluation_output_paths(task)
        if not path.exists()
    ]
    row = {
        "name": task["name"],
        "sector_name": task["sector_name"],
        "phi2_pct_capex": _safe_float(task.get("phi2_pct_capex")),
        "phi3_pct_capex": _safe_float(task.get("phi3_pct_capex")),
        "planning_year": int(planning_year),
        "solved_cost_log_exists": solved_cost_log.is_file(),
        "solved_network_exists": network_path.is_file(),
        "complete": len(missing_outputs) == 0,
        "missing_output_count": len(missing_outputs),
        "first_missing_output": missing_outputs[0] if missing_outputs else "",
    }
    network = None
    if not deploy_path.is_file() and network_path.is_file():
        import pypsa

        network = pypsa.Network(str(network_path))
    aggregate_abs_error = 0.0
    aggregate_sq_pct_error = 0.0
    scored_technologies = 0
    for technology in technologies:
        metrics = _read_deployment_metrics(
            deploy_path,
            technology,
            planning_year,
            network=network,
            network_path=network_path,
        )
        target = target_block_additions.get(technology)
        if target is not None:
            _validate_target_comparison_basis(metrics, technology)
            realized = _safe_float(metrics.get(MODEL_ADDITION_METRIC))
            metrics["target_block_addition"] = float(target)
            metrics["target_addition_kind"] = TARGET_ADDITION_KIND
            metrics["target_basis_unit"] = TARGET_BASIS_UNIT
            metrics["target_comparison_model_metric"] = MODEL_ADDITION_METRIC
            metrics["target_block_start_year"] = int(planning_year) - 5
            metrics["target_block_end_year"] = int(planning_year)
            metrics["target_error"] = realized - float(target)
            metrics["target_abs_error"] = abs(metrics["target_error"]) if math.isfinite(realized) else math.nan
            if math.isfinite(_safe_float(metrics["target_abs_error"])):
                aggregate_abs_error += float(metrics["target_abs_error"])
                if abs(float(target)) > 0.0:
                    aggregate_sq_pct_error += (float(metrics["target_error"]) / float(target)) ** 2
                scored_technologies += 1
        row.update(_prefix_metrics(metrics, technology))
    row.update(_read_system_cost_metrics(system_path))
    if scored_technologies:
        row["aggregate_target_abs_error"] = aggregate_abs_error
        row["aggregate_target_rmse_pct"] = math.sqrt(aggregate_sq_pct_error / scored_technologies)
        row["scored_technology_count"] = scored_technologies
    return row


def evaluate_tasks(
    tasks: list[dict],
    technology: str | list[str],
    planning_year: int,
    target_block_addition: float | None = None,
    target_block_additions: dict[str, float] | None = None,
) -> pd.DataFrame:
    if isinstance(technology, str):
        technologies = [technology]
    else:
        technologies = list(technology)
    target_block_additions = target_block_additions or {}
    if len(technologies) == 1 and not target_block_additions:
        rows = [
            evaluate_task(
                task,
                technology=technologies[0],
                planning_year=planning_year,
                target_block_addition=target_block_addition,
            )
            for task in tasks
        ]
    else:
        if target_block_addition is not None and len(technologies) == 1:
            target_block_additions = {technologies[0]: float(target_block_addition)}
        rows = [
            evaluate_task_multi(
                task,
                technologies=technologies,
                planning_year=planning_year,
                target_block_additions=target_block_additions,
            )
            for task in tasks
        ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    sort_cols = ["complete"]
    ascending = [False]
    if "aggregate_target_rmse_pct" in frame.columns:
        sort_cols.append("aggregate_target_rmse_pct")
        ascending.append(True)
    elif "aggregate_target_abs_error" in frame.columns:
        sort_cols.append("aggregate_target_abs_error")
        ascending.append(True)
    elif target_block_addition is not None and "target_abs_error" in frame.columns:
        sort_cols.append("target_abs_error")
        ascending.append(True)
    sort_cols.extend(["phi2_pct_capex", "phi3_pct_capex"])
    ascending.extend([True, True])
    return frame.sort_values(sort_cols, ascending=ascending, kind="stable", ignore_index=True)


def write_outputs(frame: pd.DataFrame, output_dir: Path, prefix: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{prefix}_summary.csv"
    frame.to_csv(summary_path, index=False)
    outputs = {"summary": summary_path}

    if not frame.empty and {"phi2_pct_capex", "phi3_pct_capex"}.issubset(frame.columns):
        for metric in [
            "realized_block_addition_constrained_basis",
            "realized_wedge_cost_eur",
            "target_abs_error",
            "aggregate_target_abs_error",
            "aggregate_target_rmse_pct",
            "total_system_cost_eur",
        ]:
            if metric not in frame.columns:
                continue
            pivot = frame.pivot_table(
                index="phi2_pct_capex",
                columns="phi3_pct_capex",
                values=metric,
                aggfunc="first",
            )
            path = output_dir / f"{prefix}_{metric}_pivot.csv"
            pivot.to_csv(path)
            outputs[metric] = path
    return outputs


def print_report(frame: pd.DataFrame, outputs: dict[str, Path], has_targets: bool) -> None:
    print("PHI_CALIBRATION_EVALUATION", outputs["summary"])
    print("RUNS_TOTAL", len(frame))
    if frame.empty:
        return
    print("RUNS_COMPLETE", int(frame["complete"].sum()) if "complete" in frame.columns else 0)
    if has_targets and ("aggregate_target_rmse_pct" in frame.columns or "target_abs_error" in frame.columns):
        score_col = "aggregate_target_rmse_pct" if "aggregate_target_rmse_pct" in frame.columns else "target_abs_error"
        ranked = frame.loc[frame[score_col].notna()].head(5)
        print("BEST_BY_TARGET_ERROR")
        cols = [
            "name",
            "phi2_pct_capex",
            "phi3_pct_capex",
            "aggregate_target_abs_error",
            "aggregate_target_rmse_pct",
            "realized_block_addition_constrained_basis",
            "target_block_addition",
            "target_abs_error",
            "realized_wedge_cost_eur",
        ]
        print(ranked.loc[:, [c for c in cols if c in ranked.columns]].to_string(index=False))
    else:
        cols = [
            "name",
            "complete",
            "phi2_pct_capex",
            "phi3_pct_capex",
            "realized_block_addition_constrained_basis",
            "realized_wedge_cost_eur",
            "total_system_cost_eur",
        ]
        print(frame.head(10).loc[:, [c for c in cols if c in frame.columns]].to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", help="Path to a phi calibration task_manifest.json")
    parser.add_argument("--submission-dir", help="Submission directory containing task_manifest.json")
    parser.add_argument("--latest-submission", action="store_true", help="Use latest phi_calibration_* submission")
    parser.add_argument("--submit-root", default=str(ROOT_DIR / "cluster_submissions"))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output-dir", help="Directory for evaluation CSVs")
    parser.add_argument("--output-prefix", default="phi_calibration")
    parser.add_argument(
        "--technology",
        action="append",
        help="Technology to evaluate; comma-separated or repeatable. Default: solar_power,onwind_power",
    )
    parser.add_argument("--planning-year", type=int, default=2025)
    parser.add_argument("--target-block-addition", type=float, help=argparse.SUPPRESS)
    parser.add_argument(
        "--target-block-additions-csv",
        help=(
            "Raw IRENA long-format cumulative capacity CSV used to derive global block additions"
        ),
    )
    parser.add_argument("--default-grid", action="store_true", help="Evaluate the default task grid if no manifest is supplied")
    parser.add_argument("--model", default="shared_state_bayesian_regime_wright")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--phi2-min", type=float, default=0.0)
    parser.add_argument("--phi2-max", type=float, default=0.45)
    parser.add_argument("--phi2-count", type=int, default=10)
    parser.add_argument("--phi3-min", type=float, default=0.10)
    parser.add_argument("--phi3-max", type=float, default=1.00)
    parser.add_argument("--phi3-count", type=int, default=10)
    args = parser.parse_args()

    manifest = resolve_manifest(args)
    results_root = Path(args.results_root).resolve()
    if manifest is not None:
        tasks = _load_tasks_from_manifest(manifest)
    elif args.default_grid:
        tasks = default_tasks_if_requested(args)
    else:
        tasks = discover_tasks(results_root)

    if not tasks:
        raise FileNotFoundError(
            "No phi calibration tasks found. Pass --manifest, --submission-dir, "
            "--latest-submission, --default-grid, or run after result directories exist."
        )

    technologies = parse_technologies(args.technology)
    if args.target_block_addition is not None:
        raise ValueError("Direct target values are disabled; validate phi calibration against IRENA data only")
    target_block_additions = load_target_block_additions(
        Path(args.target_block_additions_csv).resolve()
        if args.target_block_additions_csv
        else None,
        planning_year=args.planning_year,
    )

    frame = evaluate_tasks(
        tasks,
        technology=technologies,
        planning_year=args.planning_year,
        target_block_addition=args.target_block_addition,
        target_block_additions=target_block_additions,
    )
    output_dir = Path(args.output_dir).resolve() if args.output_dir else results_root / "evaluation"
    outputs = write_outputs(frame, output_dir, args.output_prefix)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest) if manifest else None,
        "results_root": str(results_root),
        "technologies": technologies,
        "planning_year": args.planning_year,
        "target_block_addition": args.target_block_addition,
        "target_block_additions": target_block_additions,
        "target_addition_kind": TARGET_ADDITION_KIND,
        "target_basis_unit": TARGET_BASIS_UNIT,
        "target_block_start_year": int(args.planning_year) - 5,
        "target_block_end_year": int(args.planning_year),
        "target_comparison_model_metric": MODEL_ADDITION_METRIC,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    metadata_path = output_dir / f"{args.output_prefix}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print_report(frame, outputs, bool(target_block_additions) or args.target_block_addition is not None)
    print("PHI_CALIBRATION_EVALUATION_METADATA", metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
