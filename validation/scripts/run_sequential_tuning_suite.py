#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run all 2020 tuning loops sequentially in a sensible order.

Default order:
1) wind (on/off split + country total)
2) solar
3) hydro (reservoir + ror jointly)
4) nuclear
5) biomass/biogas potential inputs
6) fossil dispatch tuning (fuel prices)
7) final validation

Designed for overnight unattended runs. Each tuning step defaults to max 4
iterations and uses the current script defaults for country thresholds
(i.e. no forced `--min-ref-entry-twh 2.5`).
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


LOG = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "sequential_tuning_suite_artifacts_v1"
GUARDRAIL_STATUS_FILENAME = "workflow_guardrail_status.json"

DEFAULT_NETWORK_TARGET = (
    "results/Global_200/postnetworks/"
    "elec_s_200_ec_lcopt_1h_1h_2020_0.071_AB_0.0export_base.nc"
)

ALL_STEPS = ["wind", "solar", "hydro", "nuclear", "biomass", "fossil", "validate"]

STEP_MUTABLE_STATE_FILES: dict[str, list[str]] = {
    "wind": [
        "validation/data/wind_iteration_scaling_overrides.csv",
        "validation/data/wind_iteration_scaling_overrides_last_good.csv",
    ],
    "solar": [
        "validation/data/solar_iteration_scaling_overrides.csv",
        "validation/data/solar_iteration_scaling_overrides_last_good.csv",
    ],
    "hydro": [
        "validation/data/hydro_iteration_scaling_overrides.csv",
        "validation/data/hydro_iteration_scaling_overrides_last_good.csv",
    ],
    "nuclear": [
        "validation/data/nuclear_iteration_scaling_overrides.csv",
        "validation/data/nuclear_iteration_scaling_overrides_last_good.csv",
    ],
    "biomass": [
        "validation/config.iteration_common.yaml",
        "validation/config.iteration_common_biomass_last_good.yaml",
    ],
    "fossil": [
        "validation/data/fossil_price_tuning_overrides.csv",
        "validation/data/fossil_price_tuning_overrides_last_good.csv",
    ],
    "validate": [],
}


def _repo_path(path_like: str | Path) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else REPO_ROOT / p


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_cmd(cmd: list[str], *, env: dict[str, str], cwd: Path, dry_run: bool) -> None:
    LOG.info("RUN: %s", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _step_script(step: str) -> str:
    scripts = {
        "wind": "validation/scripts/iterate_wind_scaling.py",
        "solar": "validation/scripts/iterate_solar_scaling.py",
        "hydro": "validation/scripts/iterate_hydro_scaling.py",
        "nuclear": "validation/scripts/iterate_nuclear_scaling.py",
        "biomass": "validation/scripts/tune_biomass_biogas_potential.py",
        "fossil": "validation/scripts/iterate_fossil_dispatch_tuning.py",
        "validate": "validation/scripts/validate_solved_network.py",
    }
    return scripts[step]


def _step_default_configfiles(step: str) -> list[str]:
    common = [
        "config.myopic.yaml",
        "config.learning.yaml",
        "validation/config.iteration_common.yaml",
    ]
    return common.copy()


def _step_default_artifacts(step: str, args: argparse.Namespace) -> list[str]:
    if step == "wind":
        return [
            "validation/data/wind_iteration_scaling_overrides.csv",
            "validation/data/wind_iteration_scaling_overrides_last_good.csv",
            "validation/results_compare/wind_iteration_history.csv",
            "validation/results_compare/wind_iteration_country_detail.csv",
        ]
    if step == "solar":
        return [
            "validation/data/solar_iteration_scaling_overrides.csv",
            "validation/data/solar_iteration_scaling_overrides_last_good.csv",
            "validation/results_compare/solar_iteration_history.csv",
            "validation/results_compare/solar_iteration_country_detail.csv",
        ]
    if step == "hydro":
        return [
            "validation/data/hydro_iteration_scaling_overrides.csv",
            "validation/data/hydro_iteration_scaling_overrides_last_good.csv",
            "validation/results_compare/hydro_iteration_history.csv",
            "validation/results_compare/hydro_iteration_country_detail.csv",
        ]
    if step == "nuclear":
        return [
            "validation/data/nuclear_iteration_scaling_overrides.csv",
            "validation/data/nuclear_iteration_scaling_overrides_last_good.csv",
            "validation/results_compare/nuclear_iteration_history.csv",
            "validation/results_compare/nuclear_iteration_country_detail.csv",
        ]
    if step == "biomass":
        return [
            "validation/config.iteration_common.yaml",
            "validation/config.iteration_common_biomass_last_good.yaml",
            "validation/results_compare/biomass_tuning_history.csv",
            "validation/results_compare/biomass_tuning_country_detail.csv",
        ]
    if step == "fossil":
        return [
            "validation/data/fossil_price_tuning_overrides.csv",
            "validation/data/fossil_price_tuning_overrides_last_good.csv",
            "validation/results_compare/fossil_dispatch_tuning_history.csv",
            "validation/results_compare/fossil_dispatch_tuning_country_fuel_detail.csv",
            "validation/results_compare/fossil_dispatch_price_comparison.csv",
        ]
    if step == "validate":
        return [args.validation_output_dir]
    raise KeyError(step)


def _build_step_command(args: argparse.Namespace, step: str, *, unlock: bool) -> list[str]:
    py = args.python or sys.executable
    cmd = [py, _step_script(step)]

    if step == "validate":
        cmd += [
            "--network",
            args.network_target,
            "--year",
            str(args.year),
            "--output-dir",
            args.validation_output_dir,
        ]
        return cmd

    cmd += [
        "--iterations",
        str(args.iterations_per_tuning),
        "--network-target",
        args.network_target,
        "--snakemake-cores",
        str(args.snakemake_cores),
    ]
    cfgs = _step_default_configfiles(step)
    cfgs.extend(args.extra_configfiles)
    if args.disable_generation_constraints:
        cfgs.append(args.no_constraint_overlay_config)
    cmd += ["--configfiles", *cfgs]
    if unlock:
        cmd.append("--unlock-first")
    if args.propagate_dry_run:
        cmd.append("--dry-run")

    if step == "wind":
        if args.wind_onshore_global_target_twh is not None:
            cmd += ["--onshore-global-target-twh", str(args.wind_onshore_global_target_twh)]
        if args.wind_offshore_global_target_twh is not None:
            cmd += ["--offshore-global-target-twh", str(args.wind_offshore_global_target_twh)]
    elif step == "biomass":
        cmd += ["--mode", args.biomass_mode]

    return cmd


def _guardrail_cmd(args: argparse.Namespace, output_dir: Path) -> list[str]:
    py = args.python or sys.executable
    return [
        py,
        "validation/scripts/validate_solved_network.py",
        "--network",
        args.network_target,
        "--year",
        str(args.year),
        "--output-dir",
        str(output_dir),
        "--guardrails-only",
    ]


def _guardrail_rank(status: str) -> int:
    return {"pass": 0, "dry_run": 0, "skipped": 0, "warn": 1, "fail": 2}.get(str(status), 2)


def _guardrail_should_fail(status: str, threshold: str) -> bool:
    return _guardrail_rank(status) >= _guardrail_rank(threshold) and _guardrail_rank(status) > 0


def _guardrail_phase_enabled(mode: str, phase: str) -> bool:
    if mode == "none":
        return False
    if mode == "both":
        return True
    return mode == phase


def _run_guardrail_check(
    args: argparse.Namespace,
    *,
    env: dict[str, str],
    phase: str,
    step: str,
    order: int,
    artifact_dir: Path,
    wrapper_dry_run: bool,
) -> dict[str, object]:
    result: dict[str, object] = {
        "phase": phase,
        "step": step,
        "status": "skipped",
        "overall_status": "skipped",
        "checks_fail": 0,
        "checks_warn": 0,
        "checks_pass": 0,
        "detail_rows": 0,
        "output_dir": "",
        "command": "",
        "error": "",
    }

    if not _guardrail_phase_enabled(args.guardrail_checks, phase):
        result["reason"] = f"guardrail_checks={args.guardrail_checks}"
        return result
    if wrapper_dry_run:
        result["status"] = "dry_run"
        result["overall_status"] = "dry_run"
        result["reason"] = "wrapper dry-run"
        return result

    network_target = _repo_path(args.network_target)
    if not network_target.exists():
        result["status"] = "failed(missing-network)"
        result["overall_status"] = "fail"
        result["error"] = f"network missing: {network_target}"
        return result

    out_dir = artifact_dir / "guardrails" / f"{order:02d}_{step}" / phase
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = _guardrail_cmd(args, out_dir)
    result["output_dir"] = str(out_dir.relative_to(REPO_ROOT))
    result["command"] = " ".join(cmd)

    try:
        _run_cmd(cmd, env=env, cwd=REPO_ROOT, dry_run=False)
    except subprocess.CalledProcessError as exc:
        result["status"] = "failed(command)"
        result["overall_status"] = "fail"
        result["error"] = str(exc)
        return result

    status_path = out_dir / GUARDRAIL_STATUS_FILENAME
    if not status_path.exists():
        result["status"] = "failed(missing-status)"
        result["overall_status"] = "fail"
        result["error"] = f"Missing {GUARDRAIL_STATUS_FILENAME}"
        return result

    status = _read_json(status_path)
    result.update(
        {
            "status": "ok",
            "overall_status": str(status.get("overall_status", "fail")),
            "checks_fail": int(status.get("checks_fail", 0)),
            "checks_warn": int(status.get("checks_warn", 0)),
            "checks_pass": int(status.get("checks_pass", 0)),
            "detail_rows": int(status.get("detail_rows", 0)),
        }
    )
    return result


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _baseline_snapshot_path(baseline_dir: Path, rel_path: str) -> Path:
    return baseline_dir / rel_path


def _snapshot_baseline_state(args: argparse.Namespace, *, artifact_dir: Path) -> dict[str, object]:
    baseline_dir = artifact_dir / "baseline_state"
    manifest_path = artifact_dir / "baseline_state_manifest.json"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    all_files: list[str] = []
    for step in args.steps:
        all_files.extend(STEP_MUTABLE_STATE_FILES.get(step, []))
    unique_files = sorted(set(all_files))

    entries: list[dict[str, object]] = []
    for rel in unique_files:
        src = _repo_path(rel)
        dst = _baseline_snapshot_path(baseline_dir, rel)
        exists = _copy_if_exists(src, dst)
        entries.append({"path": rel, "captured": bool(exists)})

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "mode": "baseline_snapshot",
        "entries": entries,
    }
    _write_json(manifest_path, manifest)
    return manifest


def _ensure_baseline_snapshot(args: argparse.Namespace, *, artifact_dir: Path) -> dict[str, object]:
    manifest_path = artifact_dir / "baseline_state_manifest.json"
    if manifest_path.exists():
        return _read_json(manifest_path)
    LOG.info("Creating baseline mutable-state snapshot under %s", artifact_dir / "baseline_state")
    return _snapshot_baseline_state(args, artifact_dir=artifact_dir)


def _restore_frozen_baseline_for_step(step: str, *, artifact_dir: Path) -> list[str]:
    restored: list[str] = []
    baseline_dir = artifact_dir / "baseline_state"
    for rel in STEP_MUTABLE_STATE_FILES.get(step, []):
        src = _baseline_snapshot_path(baseline_dir, rel)
        dst = _repo_path(rel)
        if _copy_if_exists(src, dst):
            restored.append(rel)
    return restored


def _history_fieldnames() -> list[str]:
    return [
        "schema_version",
        "run_name",
        "invocation_started_at_utc",
        "order",
        "step",
        "step_kind",
        "status",
        "started_at_utc",
        "ended_at_utc",
        "duration_s",
        "resume_mode",
        "unlock_used",
        "wrapper_dry_run",
        "subscript_dry_run",
        "config_stack_json",
        "artifacts_json",
        "guardrail_pre_status",
        "guardrail_pre_fail",
        "guardrail_pre_warn",
        "guardrail_post_status",
        "guardrail_post_fail",
        "guardrail_post_warn",
        "restored_frozen_baseline_json",
        "command",
        "error",
    ]


def _write_history_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = _history_fieldnames()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_history_json(path: Path, rows: list[dict[str, object]]) -> None:
    _write_json(path, {"schema_version": SCHEMA_VERSION, "rows": rows})


def _write_suite_schema(path: Path) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifacts": {
            "step_history_csv": {"columns": _history_fieldnames()},
            "step_history_json": {"shape": {"rows": "list[step history row]"}},
            "suite_status_json": {
                "fields": [
                    "schema_version",
                    "run_name",
                    "invocation_started_at_utc",
                    "updated_at_utc",
                    "overall_status",
                    "steps_planned",
                    "steps_recorded",
                    "steps_failed",
                    "history_csv",
                    "history_json",
                    "final_summary_md",
                    "final_summary_json",
                ]
            },
            "workflow_guardrail_status_json": {
                "producer": "validation/scripts/validate_solved_network.py --guardrails-only",
                "files": [
                    "workflow_guardrail_status.json",
                    "workflow_guardrail_summary.csv",
                    "workflow_guardrail_detail.csv",
                ],
            },
        },
    }
    _write_json(path, payload)


def _write_suite_status(
    *,
    path: Path,
    args: argparse.Namespace,
    invocation_started_at_utc: str,
    history_csv: Path,
    history_json: Path,
    final_summary_md: Path,
    final_summary_json: Path,
    rows: list[dict[str, object]],
) -> None:
    failed = [r for r in rows if str(r.get("status", "")).startswith("failed")]
    running = any(str(r.get("status", "")) == "running" for r in rows)
    if running:
        overall = "running"
    elif failed:
        overall = "failed"
    elif rows:
        overall = "ok"
    else:
        overall = "pending"

    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_name": args.run_name,
        "invocation_started_at_utc": invocation_started_at_utc,
        "updated_at_utc": _utc_now_iso(),
        "overall_status": overall,
        "steps_planned": args.steps,
        "steps_recorded": len(rows),
        "steps_failed": len(failed),
        "history_csv": str(history_csv.relative_to(REPO_ROOT)),
        "history_json": str(history_json.relative_to(REPO_ROOT)),
        "final_summary_md": str(final_summary_md.relative_to(REPO_ROOT)),
        "final_summary_json": str(final_summary_json.relative_to(REPO_ROOT)),
    }
    _write_json(path, payload)


def _write_final_summary(
    *,
    args: argparse.Namespace,
    artifact_dir: Path,
    history_rows: list[dict[str, object]],
    invocation_started_at_utc: str,
) -> tuple[Path, Path]:
    failed = [r for r in history_rows if str(r.get("status", "")).startswith("failed")]
    overall_status = "failed" if failed else ("ok" if history_rows else "empty")

    summary_json_path = artifact_dir / "final_qualification_summary.json"
    summary_md_path = artifact_dir / "final_qualification_summary.md"

    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_name": args.run_name,
        "invocation_started_at_utc": invocation_started_at_utc,
        "generated_at_utc": _utc_now_iso(),
        "overall_status": overall_status,
        "steps_planned": args.steps,
        "steps_recorded": len(history_rows),
        "failed_steps": [
            {
                "order": int(r.get("order", 0) or 0),
                "step": r.get("step", ""),
                "status": r.get("status", ""),
                "error": r.get("error", ""),
            }
            for r in failed
        ],
        "validation_output_dir": args.validation_output_dir,
        "history_csv": str((_repo_path(args.history_csv) if args.history_csv else (REPO_ROOT / "validation" / "results_compare" / f"{args.run_name}_step_history.csv")).relative_to(REPO_ROOT)),
        "artifacts_dir": str(artifact_dir.relative_to(REPO_ROOT)),
        "resume_mode": args.resume_mode,
        "guardrail_checks": args.guardrail_checks,
        "guardrail_fail_level": args.guardrail_fail_level,
    }
    _write_json(summary_json_path, payload)

    lines = [
        "# Sequential Tuning Suite Qualification Summary",
        "",
        f"- Run name: `{args.run_name}`",
        f"- Overall status: **{overall_status}**",
        f"- Invocation started (UTC): `{invocation_started_at_utc}`",
        f"- Generated (UTC): `{payload['generated_at_utc']}`",
        f"- Steps planned: `{', '.join(args.steps)}`",
        f"- Steps recorded: `{len(history_rows)}`",
        f"- Resume mode: `{args.resume_mode}`",
        f"- Guardrail checks: `{args.guardrail_checks}` (fail on `{args.guardrail_fail_level}`)",
        f"- Validation output dir: `{args.validation_output_dir}`",
        "",
        "## Step Statuses",
        "",
        "| order | step | status | pre-guardrail | post-guardrail | duration_s |",
        "|---:|---|---|---|---|---:|",
    ]
    for r in history_rows:
        lines.append(
            "| {order} | {step} | {status} | {gpre} | {gpost} | {dur} |".format(
                order=r.get("order", ""),
                step=r.get("step", ""),
                status=r.get("status", ""),
                gpre=r.get("guardrail_pre_status", ""),
                gpost=r.get("guardrail_post_status", ""),
                dur=r.get("duration_s", ""),
            )
        )
    if failed:
        lines.extend(["", "## Failed Steps", ""])
        for r in failed:
            lines.append(
                f"- `{r.get('step')}` ({r.get('status')}): {r.get('error') or 'see step history'}"
            )

    summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_md_path, summary_json_path


def _load_existing_history(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        return rows
    except Exception as exc:
        LOG.warning("Could not read existing history CSV %s: %s", path, exc)
        return []


def _resume_skip_count(steps: list[str], prior_rows: list[dict[str, object]], resume_from_step: str | None) -> int:
    if resume_from_step:
        try:
            return steps.index(resume_from_step)
        except ValueError as exc:
            raise ValueError(f"resume-from-step '{resume_from_step}' is not in requested --steps") from exc
    if not prior_rows:
        return 0

    prefix_ok = 0
    for step in steps:
        matches = [r for r in prior_rows if r.get("step") == step]
        if not matches:
            break
        last = matches[-1]
        if str(last.get("status", "")) == "ok":
            prefix_ok += 1
            continue
        break
    return prefix_ok


def _resume_seed_rows(steps: list[str], prior_rows: list[dict[str, object]], skip_prefix: int) -> list[dict[str, object]]:
    if skip_prefix <= 0:
        return []
    seeded: list[dict[str, object]] = []
    for step in steps[:skip_prefix]:
        matches = [r for r in prior_rows if r.get("step") == step]
        if not matches:
            continue
        seeded.append(dict(matches[-1]))
    return seeded


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run all tuning loops sequentially (wind, solar, hydro, nuclear, biomass, fossil, validation)."
    )
    p.add_argument(
        "--steps",
        nargs="+",
        choices=ALL_STEPS,
        default=ALL_STEPS,
        help="Subset/order of steps to run. Default is the full recommended sequence.",
    )
    p.add_argument(
        "--network-target",
        default=DEFAULT_NETWORK_TARGET,
        help="Solved network target path used by all tuning scripts.",
    )
    p.add_argument("--year", type=int, default=2020, help="Validation/reference year.")
    p.add_argument(
        "--iterations-per-tuning",
        type=int,
        default=4,
        help="Max iterations for each tuning step (wind/solar/hydro/nuclear/biomass/fossil).",
    )
    p.add_argument("--snakemake-cores", type=int, default=1, help="Cores passed to each tuning wrapper.")
    p.add_argument(
        "--wind-onshore-global-target-twh",
        type=float,
        default=1484.0,
        help="Global onshore wind target for wind tuning wrapper.",
    )
    p.add_argument(
        "--wind-offshore-global-target-twh",
        type=float,
        default=104.0,
        help="Global offshore wind target for wind tuning wrapper.",
    )
    p.add_argument(
        "--biomass-mode",
        choices=["solid-only", "joint"],
        default="solid-only",
        help="Biomass/biogas tuner mode.",
    )
    p.add_argument(
        "--validation-output-dir",
        default="validation/results_compare/tuning_suite_final",
        help="Output directory for the final validation step.",
    )
    p.add_argument(
        "--disable-generation-constraints",
        dest="disable_generation_constraints",
        action="store_true",
        default=True,
        help="Disable 2020 baseyear generation constraints during tuning steps (default: enabled).",
    )
    p.add_argument(
        "--keep-generation-constraints",
        dest="disable_generation_constraints",
        action="store_false",
        help="Do not add the no-constraint overlay; tune under current generation constraints.",
    )
    p.add_argument(
        "--no-constraint-overlay-config",
        default="validation/config.no_baseyear_generation_constraint.yaml",
        help="Overlay config appended to tuning-step Snakemake configfiles when generation constraints are disabled.",
    )
    p.add_argument(
        "--extra-configfiles",
        nargs="*",
        default=[],
        help="Additional Snakemake config overlays appended to each tuning step.",
    )
    p.add_argument(
        "--run-name",
        default="sequential_tuning_suite",
        help="Label used for the step history CSV filename.",
    )
    p.add_argument(
        "--history-csv",
        default=None,
        help="Optional path for step execution history CSV (defaults under validation/results_compare).",
    )
    p.add_argument(
        "--artifact-dir",
        default=None,
        help="Directory for wrapper artifacts (schema, guardrails, suite status, final summary).",
    )
    p.add_argument(
        "--python",
        default=None,
        help="Python executable to use for sub-scripts (default: current interpreter).",
    )
    p.add_argument(
        "--unlock-first",
        action="store_true",
        help="Pass --unlock-first to the first tuning step only (or each tuning step if --unlock-each-step).",
    )
    p.add_argument(
        "--unlock-each-step",
        action="store_true",
        help="Pass --unlock-first to every tuning step.",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to later steps if a step fails (status recorded in history CSV).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the existing step history (skip leading steps whose latest status is ok).",
    )
    p.add_argument(
        "--resume-from-step",
        choices=ALL_STEPS,
        default=None,
        help="Start execution from this step (overrides automatic skip-prefix detection when used with --resume).",
    )
    p.add_argument(
        "--guardrail-checks",
        choices=["none", "pre", "post", "both"],
        default="both",
        help="Run workflow guardrail checks before/after each step using validator --guardrails-only.",
    )
    p.add_argument(
        "--guardrail-fail-level",
        choices=["warn", "fail"],
        default="fail",
        help="Treat guardrail overall status at or above this level as a hard wrapper failure.",
    )
    p.add_argument(
        "--simulate-step-failure",
        choices=ALL_STEPS,
        default=None,
        help="Inject a deterministic wrapper failure for the named step (for smoke testing status/rollback handling).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run, but do not execute anything.",
    )
    p.add_argument(
        "--propagate-dry-run",
        action="store_true",
        help="Execute sub-scripts but pass their --dry-run flag (useful for smoke-testing command assembly).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    if args.dry_run and args.propagate_dry_run:
        raise ValueError("Use either --dry-run (wrapper only) or --propagate-dry-run (sub-scripts), not both.")
    if args.resume_from_step and not args.resume:
        raise ValueError("--resume-from-step requires --resume.")
    # Latest mutable overrides are the only supported resume behavior.
    args.resume_mode = "current-mutable"

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp")
    env.setdefault("XDG_CACHE_HOME", "/tmp")

    history_csv = (
        _repo_path(args.history_csv)
        if args.history_csv
        else REPO_ROOT / "validation" / "results_compare" / f"{args.run_name}_step_history.csv"
    )
    args.history_csv = str(history_csv.relative_to(REPO_ROOT))

    artifact_dir = (
        _repo_path(args.artifact_dir)
        if args.artifact_dir
        else REPO_ROOT / "validation" / "results_compare" / f"{args.run_name}_artifacts"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_dir = str(artifact_dir.relative_to(REPO_ROOT))

    history_json = artifact_dir / "step_history.json"
    schema_json = artifact_dir / "suite_artifact_schema.json"
    suite_status_json = artifact_dir / "suite_status.json"
    final_summary_md = artifact_dir / "final_qualification_summary.md"
    final_summary_json = artifact_dir / "final_qualification_summary.json"

    _repo_path(args.validation_output_dir).mkdir(parents=True, exist_ok=True)
    _write_suite_schema(schema_json)

    invocation_started_at_utc = _utc_now_iso()

    prior_rows = _load_existing_history(history_csv) if args.resume else []
    skip_prefix = _resume_skip_count(args.steps, prior_rows, args.resume_from_step) if args.resume else 0
    history_rows: list[dict[str, object]] = (
        _resume_seed_rows(args.steps, prior_rows, skip_prefix) if args.resume else []
    )

    LOG.info(
        "Starting sequential tuning suite: steps=%s, max_iterations_per_tuning=%d, network_target=%s",
        " -> ".join(args.steps),
        args.iterations_per_tuning,
        args.network_target,
    )
    LOG.info("Threshold behavior: wrapper does not override min-ref thresholds (no forced 2.5 TWh).")
    LOG.info(
        "Generation constraints during tuning: %s%s",
        "DISABLED via overlay " if args.disable_generation_constraints else "kept from base config ",
        args.no_constraint_overlay_config if args.disable_generation_constraints else "",
    )
    LOG.info(
        "Wrapper artifacts: history_csv=%s artifact_dir=%s resume=%s resume_mode=%s guardrails=%s",
        history_csv,
        artifact_dir,
        args.resume,
        args.resume_mode,
        args.guardrail_checks,
    )

    _write_history_csv(history_csv, history_rows)
    _write_history_json(history_json, history_rows)
    _write_suite_status(
        path=suite_status_json,
        args=args,
        invocation_started_at_utc=invocation_started_at_utc,
        history_csv=history_csv,
        history_json=history_json,
        final_summary_md=final_summary_md,
        final_summary_json=final_summary_json,
        rows=history_rows,
    )

    first_tuning_seen = False
    for order, step in enumerate(args.steps, start=1):
        if order <= skip_prefix:
            LOG.info("Resume: skipping already-completed step %d/%d '%s'.", order, len(args.steps), step)
            continue

        is_tuning_step = step != "validate"
        unlock = False
        if is_tuning_step:
            if args.unlock_each_step:
                unlock = True
            elif args.unlock_first and not first_tuning_seen:
                unlock = True
            first_tuning_seen = True

        restored_files = []

        cmd = _build_step_command(args, step, unlock=unlock)
        cfg_stack = [] if step == "validate" else _step_default_configfiles(step)
        if step != "validate" and args.disable_generation_constraints:
            cfg_stack = [*cfg_stack, args.no_constraint_overlay_config]

        pre_guardrail = _run_guardrail_check(
            args,
            env=env,
            phase="pre",
            step=step,
            order=order,
            artifact_dir=artifact_dir,
            wrapper_dry_run=args.dry_run,
        )

        status = "ok"
        err = ""
        post_guardrail: dict[str, object] = {
            "overall_status": "skipped",
            "checks_fail": 0,
            "checks_warn": 0,
            "status": "skipped",
        }
        t0 = time.time()
        started_at = _utc_now_iso()

        pre_overall = str(pre_guardrail.get("overall_status", "skipped"))
        if _guardrail_should_fail(pre_overall, args.guardrail_fail_level):
            status = "failed(guardrail-pre)"
            err = f"pre-step guardrail overall_status={pre_overall}"
            LOG.error("Pre-step guardrail failed for step '%s': %s", step, err)
        else:
            try:
                if args.simulate_step_failure == step:
                    raise RuntimeError(f"Simulated wrapper step failure for {step}")
                _run_cmd(cmd, env=env, cwd=REPO_ROOT, dry_run=args.dry_run)
            except subprocess.CalledProcessError as exc:
                status = f"failed({exc.returncode})"
                err = str(exc)
                LOG.error("Step '%s' failed with return code %s.", step, exc.returncode)
            except Exception as exc:
                status = "failed(exception)"
                err = repr(exc)
                LOG.exception("Step '%s' failed.", step)

            post_guardrail = _run_guardrail_check(
                args,
                env=env,
                phase="post",
                step=step,
                order=order,
                artifact_dir=artifact_dir,
                wrapper_dry_run=args.dry_run,
            )
            post_overall = str(post_guardrail.get("overall_status", "skipped"))
            if status == "ok" and _guardrail_should_fail(post_overall, args.guardrail_fail_level):
                status = "failed(guardrail-post)"
                err = f"post-step guardrail overall_status={post_overall}"
                LOG.error("Post-step guardrail failed for step '%s': %s", step, err)

        ended_at = _utc_now_iso()
        step_artifacts = _step_default_artifacts(step, args)
        for g in (pre_guardrail, post_guardrail):
            gdir = str(g.get("output_dir", "") or "")
            if gdir:
                step_artifacts.append(gdir)

        row = {
            "schema_version": SCHEMA_VERSION,
            "run_name": args.run_name,
            "invocation_started_at_utc": invocation_started_at_utc,
            "order": order,
            "step": step,
            "step_kind": "tuning" if step != "validate" else "validation",
            "status": status,
            "started_at_utc": started_at,
            "ended_at_utc": ended_at,
            "duration_s": round(time.time() - t0, 2),
            "resume_mode": args.resume_mode,
            "unlock_used": int(bool(unlock)),
            "wrapper_dry_run": int(bool(args.dry_run)),
            "subscript_dry_run": int(bool(args.propagate_dry_run)),
            "config_stack_json": json.dumps(cfg_stack),
            "artifacts_json": json.dumps(step_artifacts),
            "guardrail_pre_status": str(pre_guardrail.get("overall_status", "skipped")),
            "guardrail_pre_fail": int(pre_guardrail.get("checks_fail", 0) or 0),
            "guardrail_pre_warn": int(pre_guardrail.get("checks_warn", 0) or 0),
            "guardrail_post_status": str(post_guardrail.get("overall_status", "skipped")),
            "guardrail_post_fail": int(post_guardrail.get("checks_fail", 0) or 0),
            "guardrail_post_warn": int(post_guardrail.get("checks_warn", 0) or 0),
            "restored_frozen_baseline_json": json.dumps(restored_files),
            "command": " ".join(cmd),
            "error": err,
        }
        history_rows.append(row)
        _write_history_csv(history_csv, history_rows)
        _write_history_json(history_json, history_rows)

        final_summary_md, final_summary_json = _write_final_summary(
            args=args,
            artifact_dir=artifact_dir,
            history_rows=history_rows,
            invocation_started_at_utc=invocation_started_at_utc,
        )
        _write_suite_status(
            path=suite_status_json,
            args=args,
            invocation_started_at_utc=invocation_started_at_utc,
            history_csv=history_csv,
            history_json=history_json,
            final_summary_md=final_summary_md,
            final_summary_json=final_summary_json,
            rows=history_rows,
        )

        if status != "ok" and not args.continue_on_error:
            break

    final_summary_md, final_summary_json = _write_final_summary(
        args=args,
        artifact_dir=artifact_dir,
        history_rows=history_rows,
        invocation_started_at_utc=invocation_started_at_utc,
    )
    _write_suite_status(
        path=suite_status_json,
        args=args,
        invocation_started_at_utc=invocation_started_at_utc,
        history_csv=history_csv,
        history_json=history_json,
        final_summary_md=final_summary_md,
        final_summary_json=final_summary_json,
        rows=history_rows,
    )

    LOG.info("Saved step history to %s", history_csv)
    LOG.info("Saved step history JSON to %s", history_json)
    LOG.info("Saved wrapper artifact schema to %s", schema_json)
    LOG.info("Saved suite status to %s", suite_status_json)
    LOG.info("Saved final qualification summary to %s and %s", final_summary_md, final_summary_json)

    if history_rows:
        failed = [r for r in history_rows if str(r.get("status", "")).startswith("failed")]
        if failed:
            LOG.warning(
                "Sequential tuning suite finished with failures (%d/%d recorded rows).",
                len(failed),
                len(history_rows),
            )
            raise SystemExit(1)
        LOG.info("Sequential tuning suite completed successfully (%d recorded rows).", len(history_rows))


if __name__ == "__main__":
    main()
