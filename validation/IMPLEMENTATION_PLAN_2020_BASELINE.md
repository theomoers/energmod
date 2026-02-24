# Implementation Plan: Validated 2020 Baseline + Robust 2025 Handoff (Global_200)

## Purpose

This document defines the implementation plan for a **validated and fully functioning 2020 `Global_200` baseline** of the sector-coupled PyPSA-Earth workflow, plus a **robust 2020->2025 myopic handoff**.

It is designed to be:
- Handed to AI agents for implementation and testing
- Used as the stable reference plan for future work
- Decision-complete enough to avoid ad hoc choices during execution

## Project Goal (Locked)

Build a validated baseline energy model that can later be extended with political and institutional frictions in the objective function.

For now, the priority is:
- A model that runs reliably
- A 2020 baseline that is validated against historical data
- A workflow that preserves intended capacities across myopic years

## Scope (Locked)

- Official acceptance scope for first milestone: **`Global_200`, year `2020`**
- Plan horizon: **2020 baseline + 2025 handoff + workflow guardrails across the model workflow**
- Primary validation focus: **renewable/electricity capacity and electricity generation/demand**
- Fossil and hard-to-map categories: **secondary gate (diagnostic, non-blocking for first milestone)**

## Definition of Done (First Milestone)

The model is "done" for the first milestone when all blocking items below pass on the solved 2020 `Global_200` network.

### Blocking Validation Gate

1. Electricity demand (`electricity_demand`, defined as `load + link`, not storage charging)
- Global WAPE <= 5%
- Country pass rates reported with materiality thresholds (metric-specific; default electricity threshold `>= 2.5 TWh`)

2. Electricity balance (country-tech / country-metric)
- Blocking metrics:
  - `wind_electricity`
  - `solar_electricity`
  - `hydro_electricity`
  - `nuclear_electricity`
  - `electricity_generation`
- Global WAPE <= 10% for each blocking metric
- Country pass rate (material countries only): >= 85% within 10% APE

3. Renewable capacity validation (country-tech)
- Blocking technologies:
  - `onshore_wind`
  - `solar`
  - `hydro`
- Global WAPE <= 10% for each
- Country pass rate (material countries only): >= 85% within 20% divergence

4. Workflow integrity checks (hard-fail)
- No severe zero-profile/pathology cases (unless explicitly excluded)
- No NaN load time series in required electricity demand components
- No unintended capacity loss across brownfield handoff (unless classified and documented)
- Tuner rollback/resume integrity passes

### Secondary Gate (Non-Blocking, Mandatory Reporting)

- Fossil capacity diagnostics (gas / coal / oil aggregated sanity)
- Fossil electricity dispatch diagnostics
- Biofuel and `other_renewable_electricity` mapping diagnostics
- Offshore wind split diagnostics and reference limitations

## Core Model Semantics (Locked)

### 2020 Baseline Semantics

For the 2020 calibrated baseline:
- Physical capacities should be fixed / non-extendable
- Applies to physical `Generator`, `Link`, `StorageUnit`, `Store`, and network capacity classes unless explicitly exempted
- Exemptions must be defined in an **explicit allowlist** (accounting/tracking/slack-like components only)

### Tuning Policy

- Tuning runs should be performed **without baseyear generation constraints**
- Final confirmation run may re-enable constraints if desired
- Tuning is cumulative:
  - wind -> solar -> hydro -> nuclear -> biomass -> fossil
- Solves on shared outputs must be serialized

## Data Source Hierarchy (Locked)

Use a documented source hierarchy by domain:

- OWID: electricity generation and electricity demand
- IRENA: renewable capacities (including hydro aggregation)
- GEM: fossil electricity capacity (coal/gas primary, with explicit operating/start/retirement filters)
- Nuclear: IAEA primary, IRENA fallback for missing capacities
- `validation/ppl/custom_powerplants.csv`: model-input audit target, not validation ground truth

## Known Current Gaps (As of This Plan)

1. Hydro profile/inflow coverage issues
- Some reservoir units have missing inflow columns
- Some `ror` units have zero availability profiles
- This prevents hydro tuning from reaching reference targets cleanly

2. Fossil capacity/dispatch mismatch
- Large capacity and generation mismatches remain for gas/coal/oil
- Fossil tuning can be numerically unstable / expensive
- Fossil remains secondary gate for first milestone

3. Workflow hardening incomplete
- Some tuners already have rollback/safeguards (hydro)
- Need standardized rollback/resume and guardrails across all tuners

## Workstreams and Tasks

## WS1. Validation Gate Formalization (Blocking Foundation)

### Goal
Turn current validation outputs into a machine-checkable acceptance gate.

### Target Files
- `validation/scripts/validate_solved_network.py`
- `validation/scripts/run_sequential_tuning_suite.py`
- `validation/config.validation_gate.yaml` (new)

### Tasks
1. Add validation gate config with thresholds, materiality, and blocking/secondary metrics.
2. Add gate evaluation summary output (pass/fail by metric + overall).
3. Make demand metric definitions explicit in dashboard/output:
   - `electricity_demand` (`load + link`)
   - `electricity_demand_total_ac_withdrawal` (`load + link + storage + store`)
4. Add materiality-aware country pass-rate reporting.
5. Add guardrail statuses (workflow checks) into validation outputs.

### Acceptance
- Validator emits a single machine-readable gate summary and clear blocking/secondary status.

## WS2. 2020 Non-Extendability Enforcement (Blocking Foundation)

### Goal
Ensure 2020 baseline is a real-capacity operational solve, not hidden expansion.

### Target Files
- `scripts/add_existing_baseyear.py`
- `scripts/prepare_sector_network.py`
- `scripts/solve_network.py`
- `validation/scripts/validate_solved_network.py`
- `validation/config.baseyear_extendability_allowlist.yaml` (new)

### Tasks
1. Define allowlist for extendable 2020 components (accounting/tracking/slack only).
2. Add validator hard-fail for unexpected extendable physical assets in 2020.
3. Emit diagnostic listing all 2020 extendable components and classification.
4. Add minimal enforcement hook if validator-only approach proves too late.

### Acceptance
- No unexpected extendable physical capacities in solved 2020 baseline.

## WS3. Workflow Guardrails (Hard-Fail) (Blocking Foundation)

### Goal
Stop bad networks from passing through tuning/validation silently.

### Target Files
- `validation/scripts/validate_solved_network.py`
- `validation/scripts/run_sequential_tuning_suite.py`
- All tuner scripts in `validation/scripts/`

### Tasks
1. Promote zero-profile/pathology diagnostics to severity-based checks (warn/fail).
2. Add NaN load checks with severity classification.
3. Standardize rollback/resume behavior across all tuners (hydro pattern -> all tuners).
4. Add wrapper pre-step and post-step guardrail checks.
5. Standardize artifacts:
   - step history
   - last-good overrides
   - detail/history CSVs
   - final qualification summary

### Acceptance
- Wrapper fails fast on severe pathologies and restores valid tuner state after failures.

## WS4. Hydro Profile Coverage + Hydro Tuning Stabilization (High Priority)

### Goal
Fix structural hydro data/profile issues before over-tuning hydro scaling.

### Target Files
- `scripts/add_electricity.py`
- `validation/scripts/iterate_hydro_scaling.py`
- `validation/scripts/validate_solved_network.py`
- `validation/config.iteration_common.yaml`

### Tasks
1. Finalize and test hydro fallback in `add_electricity.py`:
   - `ror`: nearest non-zero profile fallback
   - reservoir: nearest non-zero inflow fallback with capacity-scaled transfer
2. Add hydro-specific diagnostics:
   - missing inflow columns for nonzero reservoir capacity
   - zero `ror` availability for nonzero `ror` capacity
3. Keep hydro tuning stable:
   - no double hydro scaling during hydro tuner runs
   - rollback + max-cap safeguards
   - conservative default tuning settings
4. Re-run hydro tuning after upstream hydro profile fix and compare outcomes.

### Acceptance
- Hydro profile coverage issues materially reduced and hydro tuning runs stably.

## WS5. Wind/Solar Tuning Finalization (Close-Out)

### Goal
Harden and freeze the already-strong wind/solar tuning.

### Target Files
- `validation/scripts/iterate_wind_scaling.py`
- `validation/scripts/iterate_solar_scaling.py`
- `scripts/prepare_sector_network.py`
- `config.myopic.yaml`

### Tasks
1. Add rollback/cap safeguards to wind and solar tuners.
2. Add final-evaluation row to iteration histories.
3. Freeze solar overrides to scenario-specific immutable CSV after acceptance.
4. Decide and implement default solar override enablement in `config.myopic.yaml` after milestone pass.

### Acceptance
- Wind/solar remain within gate thresholds and tuners are restart-safe.

## WS6. Nuclear Tuning + Reference Reconciliation (Blocking Metric)

### Goal
Keep nuclear capacity/generation validation source-consistent and stable.

### Target Files
- `validation/scripts/iterate_nuclear_scaling.py`
- `validation/scripts/validate_solved_network.py`
- Nuclear reference files in `validation/data/`

### Tasks
1. Make nuclear source arbitration explicit (IAEA primary + IRENA fallback).
2. Emit source attribution diagnostics for nuclear capacity reference.
3. Add rollback/resume safeguards to nuclear tuner.
4. Add diagnostics for nuclear availability saturation / ceiling cases.

### Acceptance
- Nuclear generation WAPE <= 10% and source usage is explicit in outputs.

## WS7. Fossil Capacity/Dispatch Diagnostics + Tuning (Secondary Gate)

### Goal
Make fossil validation credible, diagnosable, and stable (non-blocking for first milestone).

### Target Files
- `validation/scripts/prepare_gem_capacity_reference.py`
- `validation/scripts/reconcile_custom_powerplants_to_reference.py`
- `validation/scripts/iterate_fossil_dispatch_tuning.py`
- `validation/scripts/validate_solved_network.py`
- `validation/ppl/custom_powerplants.csv`

### Tasks
1. Finalize fossil source hierarchy and make it explicit in outputs.
2. Reconcile `custom_powerplants.csv` totals vs GEM/IRENA references and flag large discrepancies.
3. Harden fossil tuner state handling (rollback/resume) and numeric safeguards.
4. Keep fossil metrics in secondary gate for first milestone, but require complete diagnostic reporting.

### Acceptance
- Fossil reference builder and diagnostics are reproducible and source filters are explicit.

## WS8. Biomass/Biogas Potential Tuning + Mapping Diagnostics (Secondary Gate)

### Goal
Tune biomass inputs enough to support electricity validation while clarifying OWID mapping limits.

### Target Files
- `validation/scripts/tune_biomass_biogas_potential.py`
- `validation/config.iteration_common.yaml` (contains biomass tuning overrides alongside other tuning overrides)
- `validation/scripts/validate_solved_network.py`
- `config.myopic.yaml`

### Tasks
1. Formalize `solid-only` as default biomass tuning mode for baseline milestone.
2. Add biomass/biogas contribution diagnostics in validation outputs.
3. Add rollback/resume safeguards to biomass tuner.

### Acceptance
- Biomass tuning outputs are interpretable and mapping limitations are explicit.

## WS9. 2020->2025 Brownfield Handoff Integrity (Blocking for Workflow Quality)

### Goal
Detect unintended capacity loss/gain across the myopic handoff.

### Target Files
- `scripts/add_brownfield.py`
- `scripts/add_existing_baseyear.py`
- `validation/scripts/check_brownfield_capacity_retention.py` (new recommended)
- or `validation/scripts/validate_solved_network.py` (if integrated there)

### Tasks
1. Implement a brownfield retention diff checker:
   - compare solved 2020 capacities to 2025 brownfield imported baseline
2. Classify differences:
   - expected retirements
   - intentional resets/new-year candidate assets
   - expected filtering due data/pathology cleanup
   - unexpected losses/gains
3. Produce a machine-readable report and make it part of milestone qualification.

### Acceptance
- Handoff report exists and flags unexpected losses reliably.

## WS10. Suite Hardening + Agent Execution Framework (Operational)

### Goal
Make tuning/validation runs resumable and safe for multi-agent collaboration.

### Target Files
- `validation/scripts/run_sequential_tuning_suite.py`
- All tuner scripts

### Tasks
1. Standardize step history schema across wrapper runs.
2. Add explicit resume modes:
   - resume with mutable current overrides
   - resume from frozen baseline overrides
3. Add final qualification summary artifact.
4. Document and enforce serial-solve policy for shared scenario outputs.

### Acceptance
- Aborted suite can resume cleanly with reproducible state and logs.

## Multi-Agent Implementation Strategy (How To Execute This)

## High-Level Rule

- **Parallelize prep and code/data/diagnostic work**
- **Serialize all solves/tuning runs that touch the same scenario outputs**

This is the critical rule to avoid corrupted overrides, Snakemake locks, and conflicting postnetwork outputs.

## Recommended Team Topology (Agents)

### Coordinator Agent (1)
Responsibilities:
- Owns the main branch integration and merge sequencing
- Owns serial solve execution queue on `Global_200`
- Runs final validation gate and milestone qualification
- Maintains this plan document and status board

### Implementation Agents (2-4 in parallel)
Recommended split:
- Agent A: Validation gate + dashboards + guardrails (`WS1`, part of `WS3`)
- Agent B: 2020 semantics + brownfield retention checker (`WS2`, `WS9`)
- Agent C: Hydro/fallback/data-pathology fixes (`WS4`)
- Agent D: Tuner hardening + wrapper/suite operational improvements (`WS3`, `WS10`)

Optional later agents:
- Agent E: Fossil data reconciliation and fossil diagnostics (`WS7`)
- Agent F: Biomass mapping/tuning diagnostics (`WS8`)

## Repository / Branch / Worktree Strategy

Use one integration branch and separate worktrees for agents.

### Branching
- Integration branch (coordinator): `codex/validation-baseline-2020`
- Agent branches:
  - `codex/ws1-validation-gate`
  - `codex/ws2-baseyear-semantics`
  - `codex/ws4-hydro-fallback`
  - `codex/ws3-ws10-suite-hardening`
  - etc.

### Worktrees (recommended)
- One worktree per agent to avoid local branch conflicts.
- Solves should only run from the coordinator worktree unless an agent uses an isolated scenario/output path.

## Task Packaging for AI Agents (Required Format)

For each agent task, provide:
1. Goal
2. In-scope files
3. Out-of-scope files
4. Exact acceptance tests
5. Expected artifacts (CSV/MD/log outputs)
6. Non-goals / constraints (for example: "do not run full solves")

### Example Task Ticket Skeleton

```md
Task: WS1-01 Validation Gate Config + Summary Output

Goal:
Add a machine-readable validation gate config and summary output to the validator.

In-scope:
- validation/scripts/validate_solved_network.py
- validation/config.validation_gate.yaml (new)

Out-of-scope:
- tuning scripts
- Snakemake rules

Acceptance:
- Validator emits validation_gate_summary.csv
- Blocking vs secondary metrics are labeled
- Country pass rates respect materiality thresholds from config

Do not:
- Run full optimization solves
```

## Execution Order (Multiple Agents)

## Phase A (Parallel, mostly no heavy solves)

Run these in parallel:
- WS1 (validation gate formalization)
- WS3 (guardrail framework + rollback standardization design)
- WS9 (brownfield retention checker)
- WS7 non-solve part (fossil reference/data reconciliation)
- WS8 non-solve part (biomass mapping diagnostics)

Coordinator responsibilities in Phase A:
- Merge compatible PRs/patches incrementally
- Resolve interface/schema conflicts early (especially output CSV schemas)

## Phase B (Serial solves on main scenario outputs)

Run in this order on the coordinator branch:
1. WS2 (2020 semantics validation/enforcement)
2. WS4 (hydro fallback + hydro tuning)
3. WS6 (nuclear tuning)
4. WS8 solve-side biomass tuning (if needed)
5. WS7 fossil tuning (optional for first milestone)

Important:
- Only one tuning/suite run at a time against `results/Global_200/...`
- If parallel solving is needed, use isolated scenario/output directories and separate mutable override files

## Phase C (Qualification)

Coordinator runs:
1. Final 2020 validation gate
2. 2020->2025 brownfield retention check
3. Final milestone qualification summary

## Operational Guardrails for Multi-Agent Work

1. Mutable vs frozen overrides must never be confused.
- Mutable:
  - `validation/data/*_iteration_scaling_overrides.csv`
- Frozen:
  - scenario-specific validated files (for example wind)

2. Every tuner must preserve a `last_good` override file.

3. Every suite run must write:
- step history CSV
- per-step history/detail CSVs
- validation output directory
- final qualification summary

4. If an agent changes solver/tuner behavior:
- they must document rollback/resume behavior
- they must provide a dry-run or smoke-test path

5. Avoid parallel solves on shared outputs.
- If unavoidable, isolate:
  - clone/worktree
  - result path
  - mutable override files
  - logs

## Commands (Coordinator Patterns)

### Validation-only (after an agent patch)
```bash
cd "/Users/tlm2160/Library/Mobile Documents/com~apple~CloudDocs/Documents/Research 2/CKI/energmod"
HOME=/tmp XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp conda run --no-capture-output -n pypsa-earth \
  python validation/scripts/validate_solved_network.py \
  --network results/Global_200/postnetworks/elec_s_200_ec_lcopt_1h_1h_2020_0.071_AB_0.0export_base.nc \
  --output-dir validation/results_compare/coord_validation_check \
  --year 2020
```

### Resume tuning suite after a completed hydro step
```bash
cd "/Users/tlm2160/Library/Mobile Documents/com~apple~CloudDocs/Documents/Research 2/CKI/energmod"
STAMP=$(date +%Y%m%d_%H%M%S)
HOME=/tmp XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp GRB_LICENSE_FILE=/Users/tlm2160/gurobi.lic \
conda run --live-stream -n pypsa-earth \
  python validation/scripts/run_sequential_tuning_suite.py \
  --steps nuclear biomass fossil validate \
  --iterations-per-tuning 4 \
  --unlock-first \
  --run-name "sequential_tuning_suite_noconstr_resume_after_hydro_${STAMP}" \
  --validation-output-dir "validation/results_compare/tuning_suite_final_noconstr_resume_after_hydro_${STAMP}"
```

## Milestone Qualification Deliverables

1. Final solved 2020 `Global_200` network (qualified run)
2. Validation gate summary (machine-readable)
3. Validation dashboard and CSVs
4. Brownfield 2020->2025 retention report
5. Tuning suite run history and qualification summary
6. Frozen override files promoted for accepted tunings
7. Residual issues log (secondary gate items: fossil, mapping limits, remaining known data issues)

## Status Tracking (Recommended)

Keep a short status section at the top of this file (or a separate `validation/IMPLEMENTATION_PLAN_STATUS.md`) with:
- Current milestone phase (A/B/C)
- Current blocking failures
- Last qualified run path
- Last known good override files
- Next coordinator action

## Notes for Future Plans

This plan intentionally stops at:
- Validated 2020 baseline
- Robust 2020->2025 handoff
- Workflow hardening for tuning/validation

A future plan can build on this for:
- political/institutional friction objective integration
- calibration of friction terms
- policy evaluation experiments
