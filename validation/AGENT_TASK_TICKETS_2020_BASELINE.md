# Agent Task Tickets: 2020 Baseline Validation + 2025 Handoff

This file converts the implementation plan into concrete tickets for AI agents.

Reference plan:
- `validation/IMPLEMENTATION_PLAN_2020_BASELINE.md`

## Coordination Rules (Read First)

1. Do not run heavy solves/tuning wrappers in parallel on shared `results/Global_200/...` outputs.
2. Prefer separate git worktrees/branches per agent.
3. If a task needs a solve, mark it clearly and coordinate with the main runner.
4. Every task must report:
   - files changed
   - smoke tests run
   - artifacts produced
   - unresolved risks

## Suggested Agent Allocation

- Agent A: `WS1-*` (Validation gate + reporting)
- Agent B: `WS2-*`, `WS9-*` (2020 semantics + brownfield retention)
- Agent C: `WS4-*` (Hydro data/profile coverage)
- Agent D: `WS3-*`, `WS10-*` (Guardrails + suite hardening)
- Agent E (optional): `WS7-*` (Fossil reconciliation/diagnostics)
- Agent F (optional): `WS8-*` (Biomass diagnostics/tuning hardening)

## Ticket Format

Each ticket is decision-complete and can be pasted into a separate Codex session.

---

## WS1-01: Add Validation Gate Config and Summary Output

### Goal
Add a machine-readable validation gate configuration and a validator-generated gate summary (blocking vs secondary metrics, thresholds, pass/fail).

### In Scope
- `validation/scripts/validate_solved_network.py`
- `validation/config.validation_gate.yaml` (new)

### Out of Scope
- Tuning scripts
- Snakemake rules
- Core model scripts

### Requirements
- Config defines:
  - blocking metrics
  - secondary metrics
  - thresholds (WAPE, country pass-rate, APE)
  - materiality thresholds
  - workflow guardrail severity defaults
- Validator emits a machine-readable summary file:
  - recommended: `validation_gate_summary.csv`
  - optional JSON mirror: `validation_gate_summary.json`
- Summary includes:
  - metric
  - blocking/secondary
  - threshold values
  - actual values
  - pass/fail
  - overall gate result

### Acceptance
- Running validator on an existing solved network produces the gate summary artifact.
- Gate summary distinguishes blocking vs secondary metrics.
- Thresholds are read from config, not hard-coded in multiple places.

### Suggested Smoke Test
- Run validator on an existing `Global_200` solved network and confirm `validation_gate_summary.csv` exists.

---

## WS1-02: Materiality-Aware Country Pass Rates and Demand Metric Clarity

### Goal
Make pass-rate reporting robust to tiny-country denominator artifacts and clarify the two electricity demand definitions.

### In Scope
- `validation/scripts/validate_solved_network.py`
- `validation/results_compare/...` output schema updates

### Out of Scope
- Tuning logic
- Core model demand scaling logic

### Requirements
- Add materiality-aware pass-rate outputs for blocking metrics.
- Keep reporting all countries, but compute official pass-rates on material-country subsets.
- Clearly label:
  - `electricity_demand` = `load + link`
  - `electricity_demand_total_ac_withdrawal` = `load + link + storage + store`
- Add a global decomposition check artifact (or explicit line in summary/dashboard).

### Acceptance
- Validator outputs both all-country and material-country pass-rate columns.
- Demand summary/dashboard text makes the metric definitions explicit.

---

## WS2-01: 2020 Extendability Allowlist Spec + Validator Check

### Goal
Define and enforce (validator-side) the 2020 rule that physical capacities are non-extendable except allowlisted accounting/tracking components.

### In Scope
- `validation/scripts/validate_solved_network.py`
- `validation/config.baseyear_extendability_allowlist.yaml` (new)

### Out of Scope
- Core enforcement in `prepare_sector_network.py` / `add_existing_baseyear.py` (unless needed later)

### Requirements
- Add allowlist config with matching rules by component/carrier/name and rationale.
- Validator scans solved 2020 network for extendable components.
- Classify each extendable component as:
  - `allowed`
  - `unexpected`
- Emit artifact:
  - `baseyear_extendability_audit.csv`
- Fail validation gate if unexpected extendable physical assets exist.

### Acceptance
- Audit artifact produced on validator run.
- Validation gate fails when a non-allowlisted physical asset is extendable in 2020.

---

## WS2-02: Optional Core Enforcement Hook for 2020 Non-Extendability (If Needed)

### Goal
If validator-only checks prove too late, add a minimal core-script enforcement hook to prevent accidental 2020 expansion semantics.

### In Scope
- `scripts/add_existing_baseyear.py`
- `scripts/prepare_sector_network.py` (only if necessary)

### Out of Scope
- Broad refactoring of asset creation logic

### Requirements
- Enforce non-extendability for physical assets in baseyear using allowlist.
- Preserve accounting/tracking exceptions.
- Emit informative logs for any components changed.

### Acceptance
- 2020 pre-solve network has no unexpected extendable physical capacities.
- Existing intended accounting components continue to work.

### Note
- This ticket should only be started if `WS2-01` reveals repeated violations that are hard to catch late.

---

## WS3-01: Standardize Tuner Rollback/Resume Across All Tuners

### Goal
Port hydro-style rollback and `last_good` safeguards to all iterative tuners and biomass/fossil tuners.

### In Scope
- `validation/scripts/iterate_wind_scaling.py`
- `validation/scripts/iterate_solar_scaling.py`
- `validation/scripts/iterate_nuclear_scaling.py`
- `validation/scripts/iterate_fossil_dispatch_tuning.py`
- `validation/scripts/tune_biomass_biogas_potential.py`

### Out of Scope
- New tuning logic/objectives
- Validation metrics

### Requirements
- All tuners should:
  - maintain `*_last_good` state
  - restore on failed solve/update
  - avoid leaving corrupted mutable override state after abort/failure
- Standardize CLI args and logging patterns where possible.

### Acceptance
- Simulated failure test restores last-good state for each tuner.
- Tuner logs clearly state restore action and restored file path.

---

## WS3-02: Promote Zero-Profile and NaN Diagnostics to Guardrail Checks

### Goal
Convert current diagnostics into severity-based workflow guardrails that can fail validation/suite runs.

### In Scope
- `validation/scripts/validate_solved_network.py`
- `validation/scripts/run_sequential_tuning_suite.py`

### Out of Scope
- Fixing upstream causes (hydro/wind profiles) except classification

### Requirements
- Add severity classification (`pass`, `warn`, `fail`) for:
  - zero-profile assets with nonzero capacity
  - missing hydro inflow / zero `ror` profiles
  - NaN load components
- Emit standardized guardrail summary/detail artifacts.
- Wrapper should fail fast when guardrails indicate `fail`.

### Acceptance
- Guardrail artifacts are generated on validator run.
- Wrapper can parse/consume guardrail status and stop the suite.

---

## WS4-01: Validate and Finalize Hydro Fallback in `add_electricity.py`

### Goal
Ensure hydro fallback logic (for missing/zero `ror` and reservoir inflow profiles) works in the actual upstream build path.

### In Scope
- `scripts/add_electricity.py`
- minimal test/debug helpers (if needed)

### Out of Scope
- Hydro tuning algorithm changes
- Validation dashboards

### Requirements
- Confirm fallback is applied during hydro attachment in the upstream path (pre-sector network build).
- `ror`: nearest non-zero profile fallback
- reservoir: nearest non-zero inflow fallback with capacity-scaled transfer
- Log patched and dropped assets with counts and examples.

### Acceptance
- Rebuilt pre-electric/pre-sector network shows reduced missing hydro inflow / zero `ror` profile cases.
- No syntax/runtime errors in hydro attachment path.

### Solve Coordination
- Requires coordinator for integrated run if full rebuild is needed.

---

## WS4-02: Hydro Pathology Diagnostics in Validator

### Goal
Add explicit hydro-specific diagnostics to quantify remaining hydro profile coverage issues and likely impact on generation gaps.

### In Scope
- `validation/scripts/validate_solved_network.py`

### Out of Scope
- Upstream data fixing
- Hydro tuning loop changes

### Requirements
- Emit hydro pathology artifacts such as:
  - missing reservoir inflow for nonzero capacity
  - zero `ror` profiles for nonzero capacity
  - country-level affected capacity
  - rough gap contribution estimate (if feasible)

### Acceptance
- Validator output includes hydro pathology CSV(s) and dashboard references.

---

## WS4-03: Hydro Tuning Stability Preset + Safeguard Integration

### Goal
Make hydro tuning defaults safer for unconstrained tuning runs.

### In Scope
- `validation/scripts/iterate_hydro_scaling.py`
- `validation/scripts/run_sequential_tuning_suite.py`
- `validation/config.hydro_iteration.yaml`

### Out of Scope
- Hydro profile fallback logic

### Requirements
- Keep no-double-scaling behavior (`hydro_inflow_alignment: false` during hydro tuning)
- Expose/document a conservative hydro tuning preset
- Optionally allow wrapper to pass hydro-specific safe knobs when `hydro` is included

### Acceptance
- Hydro tuner can run 4 iterations under the preset without frequent numerical failures on known baseline.

---

## WS5-01: Wind/Solar Tuner Hardening and Freeze Workflow

### Goal
Harden wind and solar tuners and finalize a repeatable freeze/promotion workflow for validated overrides.

### In Scope
- `validation/scripts/iterate_wind_scaling.py`
- `validation/scripts/iterate_solar_scaling.py`
- `config.myopic.yaml`

### Out of Scope
- Retuning targets
- Core renewable profile generation

### Requirements
- Add rollback/last-good support if missing.
- Add final-evaluation row in iteration history outputs.
- Define and document promotion process:
  - mutable -> frozen scenario-specific file
- Optional:
  - enable solar frozen overrides by default once accepted

### Acceptance
- Wind and solar tuners recover from failures cleanly.
- Promotion process is documented and reproducible.

---

## WS6-01: Nuclear Reference Source Attribution + Fallback Transparency

### Goal
Make nuclear capacity reference sourcing explicit and reproducible (IAEA primary, IRENA fallback).

### In Scope
- `validation/scripts/validate_solved_network.py`
- nuclear reference data handling in `validation/data/`

### Out of Scope
- Nuclear tuning mechanics (separate ticket)

### Requirements
- Output a source attribution artifact for nuclear capacities by country.
- Explicitly show when IRENA fallback is used due missing IAEA entries.

### Acceptance
- Nuclear validation outputs include source attribution file and no silent fallback behavior.

---

## WS6-02: Nuclear Tuner Rollback + Saturation Diagnostics

### Goal
Harden nuclear tuner and explain residual nuclear generation gaps.

### In Scope
- `validation/scripts/iterate_nuclear_scaling.py`
- `validation/scripts/validate_solved_network.py` (for diagnostics if needed)

### Out of Scope
- Nuclear reference source logic (WS6-01)

### Requirements
- Add rollback/last-good behavior to nuclear tuner.
- Add diagnostics for countries hitting availability/profile ceilings.

### Acceptance
- Failed nuclear tuning iteration restores previous state.
- Diagnostics identify whether remaining gaps are due scaling limits vs model constraints.

---

## WS7-01: Fossil Reference Builder and Source Documentation Cleanup

### Goal
Make fossil capacity references (especially GEM coal + gas) reproducible and explicitly filtered.

### In Scope
- `validation/scripts/prepare_gem_capacity_reference.py`
- `validation/scripts/validate_solved_network.py`
- fossil reference CSV outputs in `validation/data/`

### Out of Scope
- Fossil price tuning loop stability

### Requirements
- Ensure GEM simplified capacity reference includes both gas and coal.
- Explicitly apply and document:
  - status = operating
  - start year <= 2020
  - retirement year >= 2020 (or missing retirement)
- Emit source/filter metadata artifact used by validator.

### Acceptance
- GEM reference outputs include gas and coal with clear filters.
- Validator labels source used for fossil capacity comparisons.

---

## WS7-02: Reconcile `custom_powerplants.csv` vs GEM/IRENA (Diagnostics)

### Goal
Quantify where model-input plant capacities differ from external references (especially gas in key countries).

### In Scope
- `validation/scripts/reconcile_custom_powerplants_to_reference.py`
- `validation/ppl/custom_powerplants.csv`

### Out of Scope
- Editing the actual model-input powerplant dataset (diagnostic first)

### Requirements
- Compare capacities by country/carrier vs:
  - GEM (fossil)
  - IRENA (renewables/aggregates where relevant)
- Produce a ranked discrepancy report and suggested remediation categories.

### Acceptance
- Output clearly highlights major discrepancies (e.g. China gas) and likely source of mismatch.

---

## WS7-03: Fossil Tuner Hardening (Secondary)

### Goal
Stabilize fossil dispatch tuning state handling and improve abort/restart safety.

### In Scope
- `validation/scripts/iterate_fossil_dispatch_tuning.py`
- `validation/scripts/run_sequential_tuning_suite.py`

### Out of Scope
- Reworking fossil objective/price formulation

### Requirements
- Add rollback/last-good behavior.
- Improve handling of aborted fossil runs (avoid partial override reuse unless valid).
- Preserve and summarize tuned-vs-current fuel price comparisons.

### Acceptance
- Aborted fossil tuner run does not leave ambiguous state.
- Resuming fossil tuning is deterministic.

---

## WS8-01: Biomass/Biogas Mapping Diagnostics and Tuner Hardening

### Goal
Clarify biomass/biogas contributions to OWID `biofuel_electricity` and harden biomass tuner state handling.

### In Scope
- `validation/scripts/tune_biomass_biogas_potential.py`
- `validation/scripts/validate_solved_network.py`
- `validation/config.iteration_common.yaml` (contains biomass tuning overrides alongside other tuning overrides)

### Out of Scope
- Broader biomass system redesign

### Requirements
- Make default `solid-only` mode explicit in script/help/output.
- Add rollback/last-good behavior if missing.
- Add validation diagnostics separating biomass/biogas-related model contributions where feasible.

### Acceptance
- Biomass tuner can fail/restart safely.
- Validator outputs clearer biomass mapping interpretation.

---

## WS9-01: Brownfield Capacity Retention Checker (2020->2025)

### Goal
Add a dedicated checker that compares solved 2020 capacities to 2025 brownfield imported baselines and classifies losses/gains.

### In Scope
- New file recommended: `validation/scripts/check_brownfield_capacity_retention.py`
- `scripts/add_brownfield.py` and `scripts/add_existing_baseyear.py` (read-only understanding; modify only if needed for metadata)

### Out of Scope
- Changing brownfield merge behavior (diagnostics first)

### Requirements
- Compare by component/country/carrier and build-year/lifetime context.
- Classification labels:
  - `expected_retirement`
  - `expected_reset_new_year`
  - `expected_filter`
  - `unexpected_loss`
  - `unexpected_gain`
- Emit a machine-readable CSV diff report.

### Acceptance
- Checker runs on a known 2020->2025 pair and produces actionable classification output.

---

## WS10-01: Sequential Tuning Suite Artifact and Resume Hardening

### Goal
Make the sequential tuning suite fully resumable and easier for multiple agents to operate around.

### In Scope
- `validation/scripts/run_sequential_tuning_suite.py`

### Out of Scope
- Tuning algorithms themselves (except interface integration)

### Requirements
- Standardize step history schema (timestamps, config stack, status, artifacts).
- Add explicit resume modes:
  - current mutable overrides
  - frozen baseline overrides
- Emit final qualification summary artifact with links to validation outputs and tuner states.
- Log exact config stack for each step (already partly present; standardize).

### Acceptance
- Aborted suite can resume cleanly with no manual state surgery.
- Final summary artifact exists and is human-readable.

---

## Coordinator-Only Tickets (Serial Solve Execution)

These should not be delegated to parallel agents unless outputs are isolated.

## COORD-01: Integrated Hydro Rebuild + Hydro Tuning Qualification

### Goal
After hydro profile fallback and hydro diagnostics patches are merged, rebuild upstream path and rerun hydro tuning to confirm improvement.

### Requires
- `WS4-01`, `WS4-02`, `WS4-03` merged

### Output
- Updated hydro tuning history
- Validation run showing hydro impact

---

## COORD-02: Final 2020 Baseline Qualification Run

### Goal
Run the accepted tuning/validation pipeline and produce milestone artifacts.

### Requires
- WS1, WS2, WS3 merged
- WS4 and WS6 merged (blocking metrics)
- Optional WS5 freeze updates

### Output
- Final validator gate summary
- Dashboard/CSVs
- Tuning suite history
- Residual issues summary

---

## COORD-03: 2020->2025 Handoff Qualification

### Goal
Run brownfield retention checks and confirm no unexpected capacity losses in the myopic handoff.

### Requires
- `WS9-01` merged

### Output
- Brownfield capacity retention diff report
- Pass/fail classification summary

---

## Suggested Execution Sequence (Coordinator)

### Phase A (Parallel Agent Implementation)
1. Merge `WS1-01`, `WS1-02`
2. Merge `WS3-01`, `WS3-02`
3. Merge `WS2-01`
4. Merge `WS9-01`
5. Merge `WS4-02`
6. Merge `WS7-01`, `WS7-02` (secondary)
7. Merge `WS8-01` (secondary)

### Phase B (Hydro + Nuclear + Core Qualification)
1. Merge `WS4-01`, `WS4-03`
2. Run `COORD-01`
3. Merge `WS6-01`, `WS6-02`
4. Run nuclear tuning + validation check
5. Merge `WS10-01`

### Phase C (Milestone Qualification)
1. Run `COORD-02`
2. Run `COORD-03`
3. Freeze accepted overrides and record qualified run paths

## Minimum Metadata Each Agent Should Return

Use this exact structure in agent handoff notes:

1. `Task ID`
2. `Files changed`
3. `What changed`
4. `Smoke tests run`
5. `Artifacts generated (paths)`
6. `Known limitations / follow-up`
7. `Does this require coordinator solve? (yes/no)`
