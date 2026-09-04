#!/usr/bin/env python3
"""Iteratively tune renewable profiles in place for the configured Global sector.

The runner removes only that sector's exact results tree between solved passes.
``--resume-from-results`` consumes a retained solved pass before continuing.
"""

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pypsa
import yaml


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "Global_200_v7"
SECTOR_NAME_PATTERN = re.compile(r"Global_200_v\d+")
BUILDER = ROOT / "scripts/non_workflow/build_renewable_profile_tuning_overrides.py"
BOOTSTRAP = ROOT / "scripts/learning/run_learning_bootstrap_job.sh"
PROFILE_TECHNOLOGIES = {
    "solar": ("generators", {"solar"}),
    "onwind": ("generators", {"onwind"}),
    "offwind": ("generators", {"offwind-ac", "offwind-dc"}),
    "nuclear": ("generators", {"nuclear"}),
    "hydro": ("generators", {"ror"}),
}


def load_config():
    with (ROOT / "config.myopic.yaml").open() as stream:
        return yaml.safe_load(stream)


def bus_countries(network):
    country = network.buses.get("country", pd.Series("", index=network.buses.index))
    country = country.fillna("").astype(str).str.strip().str.upper()
    missing = country.eq("")
    country.loc[missing] = country.index[missing].astype(str).str[:2].str.upper()
    return country


def renewable_capacity_by_bus(network_path, year):
    """Installed profile-technology capacity by bus, country, and technology."""
    network = pypsa.Network(network_path)
    countries = bus_countries(network)
    rows = []
    for technology, (component, carriers) in PROFILE_TECHNOLOGIES.items():
        assets = getattr(network, component)
        selected = assets.loc[
            assets.carrier.astype(str).isin(carriers), ["bus", "p_nom"]
        ].copy()
        if selected.empty:
            continue
        selected["country"] = selected["bus"].map(countries).fillna(
            selected["bus"].astype(str).str[:2].str.upper()
        )
        selected["technology"] = technology
        selected["capacity_mw"] = pd.to_numeric(
            selected["p_nom"], errors="coerce"
        ).fillna(0.0)
        rows.append(selected[["country", "bus", "technology", "capacity_mw"]])

    hydro_storage = network.storage_units.loc[
        network.storage_units.carrier.astype(str).eq("hydro"), ["bus", "p_nom"]
    ].copy()
    if not hydro_storage.empty:
        hydro_storage["country"] = hydro_storage["bus"].map(countries).fillna(
            hydro_storage["bus"].astype(str).str[:2].str.upper()
        )
        hydro_storage["technology"] = "hydro"
        hydro_storage["capacity_mw"] = pd.to_numeric(
            hydro_storage["p_nom"], errors="coerce"
        ).fillna(0.0)
        rows.append(hydro_storage[["country", "bus", "technology", "capacity_mw"]])

    if not rows:
        return pd.DataFrame(
            columns=["year", "country", "bus", "technology", "capacity_mw"]
        )
    capacity = pd.concat(rows, ignore_index=True).groupby(
        ["country", "bus", "technology"], as_index=False
    )["capacity_mw"].sum()
    capacity.insert(0, "year", int(year))
    return capacity.sort_values(["year", "country", "bus", "technology"])


def capacity_audit(networks, output_path):
    capacity = pd.concat(
        [renewable_capacity_by_bus(path, year) for year, path in sorted(networks.items())],
        ignore_index=True,
    )
    capacity.to_csv(output_path, index=False)
    return capacity


def compare_capacity(reference, current, output_path, tolerance):
    """Write the nodal allocation comparison; differences are diagnostic only."""
    keys = ["year", "country", "bus", "technology"]
    diff = reference.merge(
        current, on=keys, how="outer", suffixes=("_reference", "_current")
    )
    columns = ["capacity_mw_reference", "capacity_mw_current"]
    diff[columns] = diff[columns].fillna(0.0)
    diff["difference_mw"] = diff["capacity_mw_current"] - diff["capacity_mw_reference"]
    diff["within_tolerance"] = diff["difference_mw"].abs().le(tolerance)
    diff.sort_values(keys).to_csv(output_path, index=False)
    changed = diff.loc[~diff["within_tolerance"]]
    if not changed.empty:
        logger = (
            f"Nodal renewable allocation changed at {len(changed)} rows; "
            f"audit retained at {output_path}."
        )
        print(logger, flush=True)
    return changed


def expected_networks():
    networks = {}
    for year in (2020, 2025):
        candidates = sorted((RESULTS_DIR / "postnetworks").glob(f"*_{year}_*.nc"))
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected exactly one {year} postnetwork in "
                f"{RESULTS_DIR / postnetworks}, found {len(candidates)}: {candidates}"
            )
        networks[year] = candidates[0]
    return networks


def tuning_args(tuning):
    arguments = [
        "--scale-min", str(tuning.get("scale_min", 0.5)),
        "--scale-max", str(tuning.get("scale_max", 2.0)),
        "--min-reference-twh",
        str(tuning.get("capacity_alignment", {}).get("min_reference_twh", 0.1)),
    ]
    for prefix in ("solar", "wind", "nuclear", "hydro"):
        for bound in ("min", "max"):
            value = tuning.get(f"{prefix}_scale_{bound}")
            if value is not None:
                arguments.extend([f"--{prefix}-scale-{bound}", str(value)])
    return arguments


def write_overlay(path, enable, overrides=None):
    tuning = {"enable": bool(enable)}
    if overrides is not None:
        tuning["override_csv"] = str(overrides.resolve())
    with path.open("w") as stream:
        yaml.safe_dump(
            {"global_specific": {"renewable_profile_tuning": tuning}},
            stream,
            sort_keys=False,
        )


def remove_sector_results(dry_run):
    """Remove only the configured Global sector's exact results directory."""
    if (
        RESULTS_DIR.parent != ROOT / "results"
        or not SECTOR_NAME_PATTERN.fullmatch(RESULTS_DIR.name)
    ):
        raise RuntimeError(f"Refusing unexpected result directory: {RESULTS_DIR}")
    if not RESULTS_DIR.exists():
        print(f"No results directory to remove: {RESULTS_DIR}", flush=True)
        return
    print(f"Removing only: {RESULTS_DIR}", flush=True)
    if not dry_run:
        shutil.rmtree(RESULTS_DIR)


def run(command, dry_run):
    print("+", shlex.join([str(part) for part in command]), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def factor_file(workspace, iteration):
    return workspace / f"overrides_iter{iteration:02d}.csv"


def latest_factor_iteration(workspace):
    found = []
    for path in workspace.glob("overrides_iter*.csv"):
        match = re.fullmatch(r"overrides_iter(\d+)\.csv", path.name)
        if match:
            found.append((int(match.group(1)), path))
    if not found:
        raise RuntimeError(f"No completed factor file found in {workspace}")
    return max(found)


def read_manifest(path):
    if not path.exists():
        return {"passes": []}
    return json.loads(path.read_text())


def write_manifest(path, manifest):
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def update_factors(networks, existing, output, damping, tuning, reset_year, dry_run):
    # Each annual update must build on the file written by the preceding one.
    # Otherwise the 2025 write replaces the just-updated 2020 rows with their
    # values from ``existing``.
    source = existing
    for year in (2020, 2025):
        command = [
            sys.executable, str(BUILDER),
            "--network", str(networks[year]),
            "--year", str(year),
            "--existing", str(source),
            "--output", str(output),
            "--damping", str(damping),
            *tuning_args(tuning),
        ]
        if reset_year:
            command.append("--reset-year")
        run(command, dry_run)
        source = output

def record_solved_pass(
    iteration, workspace, manifest, reference_capacity, current_overrides,
    damping, tuning, tolerance, reset_year, dry_run,
):
    """Audit retained results, derive the next factors, and record the pass."""
    iteration_dir = workspace / f"iteration_{iteration:02d}"
    iteration_dir.mkdir(exist_ok=True)
    networks = expected_networks()
    capacity = capacity_audit(networks, iteration_dir / "renewable_capacity_by_bus.csv")
    if reference_capacity is None:
        reference_capacity = capacity
    else:
        compare_capacity(
            reference_capacity,
            capacity,
            iteration_dir / "renewable_capacity_difference_from_untuned.csv",
            tolerance,
        )
    output = factor_file(workspace, iteration)
    update_factors(
        networks, current_overrides, output, damping, tuning, reset_year, dry_run
    )
    manifest.setdefault("passes", []).append(
        {
            "iteration": iteration,
            "mode": "untuned" if reset_year else "tuned",
            "input_overrides": None if reset_year else str(current_overrides),
            "output_overrides": str(output),
            "networks_before_removal": {
                str(year): str(path) for year, path in networks.items()
            },
        }
    )
    write_manifest(workspace / "manifest.json", manifest)
    return capacity if reference_capacity is None else reference_capacity, output


def main():
    parser = argparse.ArgumentParser(
        description="Iteratively tune the configured Global sector without recreating its resources."
    )
    parser.add_argument(
        "--iterations", type=int, default=4,
        help="Factor-update passes including the initial untuned pass (default: 4).",
    )
    parser.add_argument(
        "--damping", type=float, default=0.5,
        help="Solved-generation feedback exponent (default: 0.5).",
    )
    parser.add_argument("--run-id", default=None, help="Existing or new audit token.")
    parser.add_argument(
        "--initial-overrides", default="validation/data/renewable_profile_tuning_overrides.csv",
        help="CSV schema used for the first reset-to-1.0 multiplier pass.",
    )
    parser.add_argument(
        "--workspace", default="validation/results/renewable_profile_tuning_iterations",
        help="Directory for factors, audits, and manifest; never deleted by this script.",
    )
    parser.add_argument(
        "--overlay-config", action="append", default=[],
        help="Additional overlay forwarded to every solve; tuning overlay remains last.",
    )
    parser.add_argument("--capacity-tolerance-mw", type=float, default=1e-6)
    parser.add_argument(
        "--resume-from-results", action="store_true",
        help="Treat the retained configured-sector results as the next solved pass of --run-id.",
    )
    parser.add_argument(
        "--resume-from-factors", action="store_true",
        help="Resume at the next pass after the latest factor file; any partial configured-sector results are removed first.",
    )
    parser.add_argument(
        "--start-tuned", action="store_true",
        help="Start a new workspace with --initial-overrides enabled in pass one, then apply feedback in later passes.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be at least one")
    if not 0.0 < args.damping <= 1.0:
        parser.error("--damping must lie in (0, 1]")
    if args.capacity_tolerance_mw < 0.0:
        parser.error("--capacity-tolerance-mw cannot be negative")
    if (args.resume_from_results or args.resume_from_factors) and not args.run_id:
        parser.error("--resume-from-results and --resume-from-factors require --run-id")
    if args.resume_from_results and args.resume_from_factors:
        parser.error("Choose only one of --resume-from-results and --resume-from-factors")
    if args.start_tuned and (args.resume_from_results or args.resume_from_factors):
        parser.error("--start-tuned cannot be combined with a resume option")

    config = load_config()
    sector_name = str(config.get("run", {}).get("sector_name", ""))
    if not SECTOR_NAME_PATTERN.fullmatch(sector_name):
        raise RuntimeError(
            f"This workflow only accepts run.sector_name matching Global_200_vN, got: {sector_name!r}"
        )
    global RESULTS_DIR
    RESULTS_DIR = ROOT / "results" / sector_name
    tuning = config.get("global_specific", {}).get("renewable_profile_tuning", {})
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    workspace = (ROOT / args.workspace / run_id).resolve()
    seed = (ROOT / args.initial_overrides).resolve()
    manifest_path = workspace / "manifest.json"

    if args.dry_run:
        if args.resume_from_results:
            completed, _ = latest_factor_iteration(workspace)
            print(
                f"Resume retained pass {completed + 1}: derive its factors, remove only "
                f"{RESULTS_DIR}, then solve passes {completed + 2}..{args.iterations}.",
                flush=True,
            )
        elif args.resume_from_factors:
            completed, _ = latest_factor_iteration(workspace)
            print(
                f"Resume after factors through iteration {completed}: remove only {RESULTS_DIR}, "
                f"then solve passes {completed + 1}..{args.iterations}.",
                flush=True,
            )
        elif args.start_tuned:
            print(
                f"Start {args.iterations} passes with {seed} enabled in pass one, then "
                f"{args.iterations - 1} feedback passes.",
                flush=True,
            )
        else:
            print(
                f"Start {args.iterations} passes: untuned baseline plus "
                f"{args.iterations - 1} tuned feedback passes.", flush=True
            )
        return

    if args.resume_from_results:
        if not workspace.is_dir():
            raise FileNotFoundError(f"Resume workspace does not exist: {workspace}")
        if not RESULTS_DIR.is_dir():
            raise FileNotFoundError(f"Retained solved results do not exist: {RESULTS_DIR}")
        completed, current_overrides = latest_factor_iteration(workspace)
        iteration = completed + 1
        if iteration > args.iterations:
            raise RuntimeError(
                f"Workspace already has factors through iteration {completed}; "
                f"--iterations={args.iterations} leaves nothing to resume."
            )
        reference_path = workspace / "iteration_01" / "renewable_capacity_by_bus.csv"
        if not reference_path.exists():
            raise FileNotFoundError(f"Missing baseline capacity audit: {reference_path}")
        reference_capacity = pd.read_csv(reference_path)
        manifest = read_manifest(manifest_path)
        print(f"Resuming from retained solved iteration {iteration}.", flush=True)
        reference_capacity, current_overrides = record_solved_pass(
            iteration, workspace, manifest, reference_capacity, current_overrides,
            args.damping, tuning, args.capacity_tolerance_mw, False, False,
        )
        remove_sector_results(dry_run=False)
        start_iteration = iteration + 1
    elif args.resume_from_factors:
        if not workspace.is_dir():
            raise FileNotFoundError(f"Resume workspace does not exist: {workspace}")
        completed, current_overrides = latest_factor_iteration(workspace)
        start_iteration = completed + 1
        if start_iteration > args.iterations:
            raise RuntimeError(
                f"Workspace already has factors through iteration {completed}; "
                f"--iterations={args.iterations} leaves nothing to resume."
            )
        reference_path = workspace / "iteration_01" / "renewable_capacity_by_bus.csv"
        if not reference_path.exists():
            raise FileNotFoundError(f"Missing baseline capacity audit: {reference_path}")
        reference_capacity = pd.read_csv(reference_path)
        manifest = read_manifest(manifest_path)
        print(
            f"Resuming after factor iteration {completed}; partial results will be removed before iteration {start_iteration}.",
            flush=True,
        )
    else:
        if workspace.exists():
            raise FileExistsError(f"Workspace already exists: {workspace}; choose --run-id or resume it")
        if not seed.exists():
            raise FileNotFoundError(f"Initial factor file does not exist: {seed}")
        workspace.mkdir(parents=True)
        shutil.copy2(seed, workspace / "overrides_seed.csv")
        manifest = {
            "run_id": run_id,
            "iterations": args.iterations,
            "damping": args.damping,
            "result_directory_removed_between_passes": str(RESULTS_DIR),
            "seed_overrides": str(seed),
            "passes": [],
        }
        reference_capacity = None
        current_overrides = workspace / "overrides_seed.csv"
        start_iteration = 1

    for iteration in range(start_iteration, args.iterations + 1):
        untuned = iteration == 1 and not args.start_tuned
        iteration_dir = workspace / f"iteration_{iteration:02d}"
        iteration_dir.mkdir(exist_ok=True)
        overlay = iteration_dir / "profile_tuning_overlay.yaml"
        write_overlay(overlay, enable=not untuned, overrides=None if untuned else current_overrides)
        remove_sector_results(dry_run=False)
        command = ["bash", str(BOOTSTRAP)]
        for extra_overlay in args.overlay_config:
            command.extend(["--overlay-config", str((ROOT / extra_overlay).resolve())])
        command.extend(["--overlay-config", str(overlay)])
        run(command, dry_run=False)
        reference_capacity, current_overrides = record_solved_pass(
            iteration, workspace, manifest, reference_capacity, current_overrides,
            args.damping, tuning, args.capacity_tolerance_mw, untuned, False,
        )
        remove_sector_results(dry_run=False)

    print(f"Completed. Final factors: {current_overrides}")
    print(f"No result network is retained; {RESULTS_DIR} was removed after the final update.")


if __name__ == "__main__":
    main()
