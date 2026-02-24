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
import logging
import os
from pathlib import Path
import subprocess
import sys
import time


LOG = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_NETWORK_TARGET = (
    "results/Global_200/postnetworks/"
    "elec_s_200_ec_lcopt_1h_1h_2020_0.071_AB_0.0export_base.nc"
)

ALL_STEPS = ["wind", "solar", "hydro", "nuclear", "biomass", "fossil", "validate"]


def _repo_path(path_like: str | Path) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else REPO_ROOT / p


def _run_cmd(cmd: list[str], *, env: dict[str, str], cwd: Path, dry_run: bool) -> None:
    LOG.info("RUN: %s", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


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

    # Common tuning script args
    cmd += [
        "--iterations",
        str(args.iterations_per_tuning),
        "--network-target",
        args.network_target,
        "--snakemake-cores",
        str(args.snakemake_cores),
    ]
    cfgs = _step_default_configfiles(step)
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

    # Important: we intentionally do NOT pass `--country-min-ref-twh 2.5` or
    # `--min-ref-entry-twh 2.5`; each script keeps its default threshold.
    return cmd


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

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp")
    env.setdefault("XDG_CACHE_HOME", "/tmp")

    history_csv = (
        _repo_path(args.history_csv)
        if args.history_csv
        else REPO_ROOT / "validation" / "results_compare" / f"{args.run_name}_step_history.csv"
    )
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    _repo_path(args.validation_output_dir).mkdir(parents=True, exist_ok=True)

    LOG.info(
        "Starting sequential tuning suite: steps=%s, max_iterations_per_tuning=%d, network_target=%s",
        " -> ".join(args.steps),
        args.iterations_per_tuning,
        args.network_target,
    )
    LOG.info(
        "Threshold behavior: wrapper does not override min-ref thresholds (no forced 2.5 TWh)."
    )
    LOG.info(
        "Generation constraints during tuning: %s%s",
        "DISABLED via overlay " if args.disable_generation_constraints else "kept from base config ",
        args.no_constraint_overlay_config if args.disable_generation_constraints else "",
    )

    history_rows: list[dict[str, object]] = []
    first_tuning_seen = False
    for order, step in enumerate(args.steps, start=1):
        is_tuning_step = step != "validate"
        unlock = False
        if is_tuning_step:
            if args.unlock_each_step:
                unlock = True
            elif args.unlock_first and not first_tuning_seen:
                unlock = True
            first_tuning_seen = True

        cmd = _build_step_command(args, step, unlock=unlock)
        t0 = time.time()
        status = "ok"
        err = ""
        try:
            _run_cmd(cmd, env=env, cwd=REPO_ROOT, dry_run=args.dry_run)
        except subprocess.CalledProcessError as exc:
            status = f"failed({exc.returncode})"
            err = str(exc)
            LOG.error("Step '%s' failed with return code %s.", step, exc.returncode)
            if not args.continue_on_error:
                history_rows.append(
                    {
                        "order": order,
                        "step": step,
                        "status": status,
                        "duration_s": round(time.time() - t0, 2),
                        "command": " ".join(cmd),
                        "error": err,
                    }
                )
                break
        except Exception as exc:  # pragma: no cover (defensive)
            status = "failed(exception)"
            err = repr(exc)
            LOG.exception("Step '%s' failed.", step)
            if not args.continue_on_error:
                history_rows.append(
                    {
                        "order": order,
                        "step": step,
                        "status": status,
                        "duration_s": round(time.time() - t0, 2),
                        "command": " ".join(cmd),
                        "error": err,
                    }
                )
                break

        history_rows.append(
            {
                "order": order,
                "step": step,
                "status": status,
                "duration_s": round(time.time() - t0, 2),
                "command": " ".join(cmd),
                "error": err,
            }
        )
        _write_history_csv(history_csv, history_rows)

        if status != "ok" and not args.continue_on_error:
            break

    _write_history_csv(history_csv, history_rows)
    LOG.info("Saved step history to %s", history_csv)
    if history_rows:
        failed = [r for r in history_rows if str(r["status"]).startswith("failed")]
        if failed:
            LOG.warning(
                "Sequential tuning suite finished with failures (%d/%d steps).",
                len(failed),
                len(history_rows),
            )
            raise SystemExit(1)
        LOG.info("Sequential tuning suite completed successfully (%d steps).", len(history_rows))


def _write_history_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["order", "step", "status", "duration_s", "command", "error"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
