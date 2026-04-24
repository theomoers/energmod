#!/usr/bin/env python3
"""Evaluate phi2/phi3 battery deployment-wedge calibration results."""

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
    phi_calibration_target,
)


PHI_TASK_RE = re.compile(r"^phi2_(?P<phi2>[0-9]+p[0-9]+)_phi3_(?P<phi3>[0-9]+p[0-9]+)$")
DEFAULT_TECHNOLOGY = "battery_energy"


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


def _read_deployment_metrics(path: Path, technology: str, planning_year: int) -> dict[str, float | str | bool]:
    if not path.is_file():
        return {"deployment_constraints_exists": False}
    frame = pd.read_csv(path)
    if "technology" in frame.columns:
        frame = frame.loc[frame["technology"].astype(str).eq(technology)]
    if "year" in frame.columns:
        years = pd.to_numeric(frame["year"], errors="coerce")
        frame = frame.loc[years.eq(int(planning_year))]
    return {
        "deployment_constraints_exists": True,
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
    missing_outputs = [
        str(ROOT_DIR / rel_path)
        for rel_path in phi_calibration_output_paths(task)
        if not (ROOT_DIR / rel_path).exists()
    ]
    row = {
        "name": task["name"],
        "sector_name": task["sector_name"],
        "phi2_pct_capex": _safe_float(task.get("phi2_pct_capex")),
        "phi3_pct_capex": _safe_float(task.get("phi3_pct_capex")),
        "planning_year": int(planning_year),
        "solved_cost_log_exists": solved_cost_log.is_file(),
        "complete": len(missing_outputs) == 0,
        "missing_output_count": len(missing_outputs),
        "first_missing_output": missing_outputs[0] if missing_outputs else "",
    }
    row.update(_read_deployment_metrics(deploy_path, technology, planning_year))
    row.update(_read_system_cost_metrics(system_path))
    if target_block_addition is not None:
        realized = _safe_float(row.get("realized_block_addition_constrained_basis"))
        row["target_block_addition"] = float(target_block_addition)
        row["target_error"] = realized - float(target_block_addition)
        row["target_abs_error"] = abs(row["target_error"]) if math.isfinite(realized) else math.nan
    return row


def evaluate_tasks(
    tasks: list[dict],
    technology: str,
    planning_year: int,
    target_block_addition: float | None = None,
) -> pd.DataFrame:
    rows = [
        evaluate_task(
            task,
            technology=technology,
            planning_year=planning_year,
            target_block_addition=target_block_addition,
        )
        for task in tasks
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    sort_cols = ["complete"]
    ascending = [False]
    if target_block_addition is not None and "target_abs_error" in frame.columns:
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


def print_report(frame: pd.DataFrame, outputs: dict[str, Path], target_block_addition: float | None) -> None:
    print("PHI_CALIBRATION_EVALUATION", outputs["summary"])
    print("RUNS_TOTAL", len(frame))
    if frame.empty:
        return
    print("RUNS_COMPLETE", int(frame["complete"].sum()) if "complete" in frame.columns else 0)
    if target_block_addition is not None and "target_abs_error" in frame.columns:
        ranked = frame.loc[frame["target_abs_error"].notna()].head(5)
        print("BEST_BY_TARGET_ABS_ERROR")
        cols = [
            "name",
            "phi2_pct_capex",
            "phi3_pct_capex",
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
    parser.add_argument("--technology", default=DEFAULT_TECHNOLOGY)
    parser.add_argument("--planning-year", type=int, default=2025)
    parser.add_argument("--target-block-addition", type=float)
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

    frame = evaluate_tasks(
        tasks,
        technology=args.technology,
        planning_year=args.planning_year,
        target_block_addition=args.target_block_addition,
    )
    output_dir = Path(args.output_dir).resolve() if args.output_dir else results_root / "evaluation"
    outputs = write_outputs(frame, output_dir, args.output_prefix)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest) if manifest else None,
        "results_root": str(results_root),
        "technology": args.technology,
        "planning_year": args.planning_year,
        "target_block_addition": args.target_block_addition,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    metadata_path = output_dir / f"{args.output_prefix}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print_report(frame, outputs, args.target_block_addition)
    print("PHI_CALIBRATION_EVALUATION_METADATA", metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
