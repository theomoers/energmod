Here’s a Codex-ready note you can paste into your repo/task tracker.

````markdown
# Codex task: Build adaptable sensitivity-analysis ensemble runner

## Goal

Build a configurable ensemble system for running Monte Carlo sensitivity analyses around the main stochastic energy-system specification.

The system should let us choose:

1. Which sensitivity runs to execute.
2. How many Monte Carlo draws to use per run.
3. Whether to use a preset budget tier, e.g. screening/core/headline.
4. Whether to reuse common random numbers across runs.
5. Whether to resume incomplete runs.

The runner should be adaptable enough that we can start with low-draw screening runs and later scale selected runs to 250, 500, 1000, or 2000 draws without changing the model code.

---

## Current implementation status

Use:

```bash
python scripts/learning/submit_learning_sensitivity_ensemble_array.py \
  --config configs/sensitivity_ensemble.yaml \
  --dry-run
```

The current runnable supplementary suite is 18 sensitivity runs and 6200 total draw tasks. `BASE`, `L1_CGRW`, and `L2_WAY_RHO035` are treated as already completed 1000-draw headline runs, so the full study total after the new suite is 9200 draws. The submission script emits one SGE array for the selected tasks, currently:

```bash
grid_run ... --grid_array=1-6200/200 scripts/learning/run_learning_sensitivity_ensemble_array_task.sh ...
```

Learning seeds are draw-matched across runs by default (`seed_mode: sequential`, common random numbers enabled), so draw `i` uses learning seed `i` in every selected run.

Branch-mode runs must use bootstrap/prerequisite state that matches their demand and opts wildcards. The default AB bootstrap is not valid for DF, NZ, or the CO2-cap case. For the full suite, provide grouped bootstrap sources:

```bash
python scripts/learning/submit_learning_sensitivity_ensemble_array.py \
  --config configs/sensitivity_ensemble.yaml \
  --bootstrap-state-source AB=/path/to/bootstrap_AB \
  --bootstrap-state-source DF=/path/to/bootstrap_DF \
  --bootstrap-state-source NZ=/path/to/bootstrap_NZ \
  --bootstrap-state-source CO2CAP=/path/to/bootstrap_CO2CAP
```

If those sector prerequisites are already staged in `results/Global_200/sens_*`, pass `--assume-bootstrap-state-present` explicitly. Shared branch prerequisites can be generated with `scripts/learning/run_learning_branch_prereqs_job.sh`; use `--overlay-config configs/sensitivity_bootstrap_DF.yaml`, `--overlay-config configs/sensitivity_bootstrap_NZ.yaml`, or `--overlay-config configs/sensitivity_bootstrap_CO2CAP.yaml` before saving those bootstrap states.

Cluster order:

1. Submit the four bootstrap jobs first. They can run in parallel because each uses a different `--scenario-name` and a different saved-state directory.
2. After all four complete, submit the sensitivity array with all four `--bootstrap-state-source GROUP=/path` arguments.
3. For a smoke test, add `--test-draws-per-run 2`; this produces 36 tasks, or two draw-matched seeds for each of the 18 runnable sensitivities.

`P1_BINDING_CO2_CAP` now uses `electricity.myopic_co2_cap_trajectory` rather than a static `Co2L` wildcard. The hook leaves 2020 and 2025 unconstrained by this policy cap, then adds a year-specific PyPSA `CO2Limit` on `carrier_attribute="co2_emissions"` from 2030 onward. The percentages are relative to the model's historically constrained 2025 Power emissions in `../learning_estimation/pypsa-results/endogenous-complete-v1`: 13.864 GtCO2 across the first 10 checked seeds. The cap path is 7.63 GtCO2 in 2030, 2.77 GtCO2 in 2035, 1.11 GtCO2 in 2040, 0.28 GtCO2 in 2045, and exact zero in 2050.

Potential additions if runtime budget allows: a no-explicit-BEV battery-experience sensitivity, a legacy battery-experience accounting sensitivity, and a softer CO2 trajectory to separate "binding policy" effects from near infeasibility or extreme shadow-price behavior.

---

## Baseline specification

Use the SI baseline unless explicitly overridden:

- Learning model: `SSBR`
- Demand: central demand scenario
- Fossil prices: stochastic baseline process
- Deployment wedge: active, global
- Wedge premia: `phi2 = 0.35`, `phi3 = 0.50`
- Wedge thresholds: `B1 = 0.8 * A_tilde`, `B2 = 1.2 * A_tilde`
- Threshold shocks: stochastic, baseline shrinkage
- Cost expectation kernel: 5-year uniform kernel
- Procurement/cost lags:
  - solar: 1 years
  - onshore wind: 4 years
  - batteries: 1 years
- Learning training window: full sample
- Recursive cost updating: active
- Emissions cap: none

---

## Required interface

Add a command-line runner, for example:

```bash
python run_ensemble.py \
  --config configs/sensitivity_ensemble.yaml \
  --budget core \
  --n-draws 500 \
  --run-ids BASE,L1,L2,W0,JW1,JW2 \
  --resume \
  --common-random-numbers
````

The runner should support:

```bash
--config PATH
--budget {screening,secondary,core,headline,custom}
--n-draws INT
--run-ids comma,separated,list
--max-runs INT
--resume
--overwrite
--common-random-numbers / --no-common-random-numbers
--seed INT
--dry-run
```

Behavior:

* `--n-draws` overrides draw counts from the config.
* `--budget` selects default draw counts if `--n-draws` is not provided.
* `--run-ids` limits the ensemble to a selected subset.
* `--max-runs` executes only the first `N` selected runs, useful for testing.
* `--dry-run` prints the resolved run list and draw counts without solving.
* `--resume` skips completed draw outputs and continues incomplete runs.
* `--overwrite` deletes/replaces existing outputs for selected runs.
* `--common-random-numbers` ensures draw `i` uses the same `learning_seed` across baseline and sensitivity runs wherever feasible.

---

## Draw-count presets

Implement these defaults:

```yaml
draw_budgets:
  screening: 100
  secondary: 250
  core: 500
  headline: 1000
  baseline_high: 2000
```

Suggested usage:

* `BASE`: 1000 or 2000 draws
* Headline learning-model comparisons: 1000 draws
* Main sensitivity runs: 500 draws
* Secondary appendix sensitivities: 250 draws
* Exploratory screening: 100 draws
* Fast-transition-definition sensitivities: no new solves; reuse existing draws

Minimum headline set to promote after screening:

```text
BASE
L1_CGRW
L2_WAY_RHO035
L4_DETERMINISTIC_LEARNING
W0_NO_WEDGE
JW1_TIGHT_WEDGE
JW2_LOOSE_WEDGE
D1_LOW_DEMAND
D2_HIGH_DEMAND
F0_DETERMINISTIC_FOSSIL
R1_NO_DEPLOYMENT_FEEDBACK
TW1_ORIGIN_2020
WU0_DETERMINISTIC_THRESHOLDS
K0_POINT_COST
P1_BINDING_CO2_CAP
```

---

## Ensemble config structure

Create a YAML file like:

```yaml
ensemble:
  name: stochastic_learning_sensitivity_ensemble
  base_seed: 20260504
  use_common_random_numbers: true
  default_budget: core

draw_budgets:
  screening: 100
  secondary: 250
  core: 500
  headline: 1000
  baseline_high: 2000

baseline:
  run_id: BASE
  description: Main SSBR stochastic learning specification
  budget: headline
  parameters:
    learning_model: SSBR
    demand_scenario: central
    fossil_price_mode: stochastic
    deployment_wedge: true
    phi2: 0.35
    phi3: 0.50
    B1_multiplier: 0.8
    B2_multiplier: 1.2
    threshold_shock_mode: baseline_shrunk
    cost_kernel: uniform_5yr
    lag_solar: 1
    lag_onwind: 4
    lag_battery: 1
    training_window: full_sample
    recursive_cost_update: true
    emissions_cap: none

runs:
  - run_id: L1_CGRW
    group: learning_model
    description: CGRW instead of SSBR
    budget: headline
    overrides:
      learning_model: CGRW

  - run_id: L2_WAY_RHO035
    group: learning_model
    description: Way fixed-rho benchmark, rho = 0.35
    budget: headline
    overrides:
      learning_model: WAY
      way_rho: 0.35

  - run_id: L4_DETERMINISTIC_LEARNING
    group: learning_model
    description: Deterministic learning curve (in log levels, NOT differences)
    budget: core
    overrides:
      learning_draw_mode: deterministic_median

  - run_id: W0_NO_WEDGE
    group: deployment_wedge
    description: Remove soft deployment wedge
    budget: core
    overrides:
      deployment_wedge: false

  - run_id: W6_HIGH_PHI3
    group: deployment_wedge
    description: Higher second-segment wedge premium
    budget: core
    overrides:
      phi2: 0.35
      phi3: 0.60

  - run_id: W9_ABS_GW_CALIBRATION
    group: deployment_wedge
    description: Alternative wedge calibration near absolute-GW optimum
    budget: core
    overrides:
      phi2: 0.00
      phi3: 0.40

  - run_id: B1_LOW
    group: deployment_thresholds
    description: Lower first threshold
    budget: secondary
    overrides:
      B1_multiplier: 0.6
      B2_multiplier: 1.5

  - run_id: B1_HIGH
    group: deployment_thresholds
    description: Higher first threshold
    budget: secondary
    overrides:
      B1_multiplier: 1.0
      B2_multiplier: 1.5

  - run_id: B2_LOW
    group: deployment_thresholds
    description: Lower second threshold
    budget: secondary
    overrides:
      B1_multiplier: 0.8
      B2_multiplier: 1.2

  - run_id: B2_HIGH
    group: deployment_thresholds
    description: Higher second threshold
    budget: secondary
    overrides:
      B1_multiplier: 0.8
      B2_multiplier: 2.0

  - run_id: JW1_TIGHT_WEDGE
    group: joint_wedge
    description: Tight deployment-friction case
    budget: core
    overrides:
      B1_multiplier: 0.6
      B2_multiplier: 1.2
      phi2: 0.35
      phi3: 0.75

  - run_id: JW2_LOOSE_WEDGE
    group: joint_wedge
    description: Loose deployment-friction case
    budget: core
    overrides:
      B1_multiplier: 1.0
      B2_multiplier: 2.0
      phi2: 0.15
      phi3: 0.40

  - run_id: JW3_CONTEXT_TEX
    group: joint_wedge
    description: Config-consistency check matching current context.tex values; do not promote unless context.tex differs from baseline
    budget: screening
    overrides:
      B1_multiplier: 0.8
      B2_multiplier: 1.2
      phi2: 0.35
      phi3: 0.50

  - run_id: D1_LOW_DEMAND
    group: demand
    description: Low demand scenario
    budget: core
    overrides:
      demand_scenario: low

  - run_id: D2_HIGH_DEMAND
    group: demand
    description: High demand scenario
    budget: core
    overrides:
      demand_scenario: high

  - run_id: F0_DETERMINISTIC_FOSSIL
    group: fossil_prices
    description: Deterministic fossil fuel prices
    budget: core
    overrides:
      fossil_price_mode: deterministic

  - run_id: F1_LOW_FOSSIL
    group: fossil_prices
    description: Low fossil price path
    budget: core
    overrides:
      fossil_price_scenario: low

  - run_id: F2_HIGH_FOSSIL
    group: fossil_prices
    description: High fossil price path
    budget: core
    overrides:
      fossil_price_scenario: high

  - run_id: F3_LOW_FOSSIL_VOL
    group: fossil_prices
    description: Lower fossil price shock volatility
    budget: secondary
    overrides:
      fossil_price_volatility_scale: 0.5

  - run_id: F4_HIGH_FOSSIL_VOL
    group: fossil_prices
    description: Higher fossil price shock volatility
    budget: secondary
    overrides:
      fossil_price_volatility_scale: 1.5

  - run_id: TW1_ORIGIN_2020
    group: learning_artifacts
    description: Use stochastic learning artifacts trained only to the 2020 origin cutoff
    budget: core
    overrides:
      training_window: origin_cutoff

  - run_id: WU0_DETERMINISTIC_THRESHOLDS
    group: deployment_threshold_uncertainty
    description: Disable stochastic deployment-threshold shocks while keeping the mean wedge calibration
    budget: core
    overrides:
      threshold_shock_mode: deterministic

  - run_id: K0_POINT_COST
    group: cost_kernel_lags
    description: Use point costs instead of 5-year block-average expected costs
    budget: core
    overrides:
      cost_expectation_mode: point_cost

  - run_id: K1_GLOBAL_CURRENT_WINDOW
    group: cost_kernel_lags
    description: Use the same current 5-year expectation window for all learning technologies
    budget: secondary
    overrides:
      cost_kernel_mode: global_current_window

  - run_id: K_SOL_SHORT_LAG
    group: cost_kernel_lags
    description: Solar lag shortened by 2 years
    budget: secondary
    overrides:
      lag_solar: 0

  - run_id: K_SOL_LONG_LAG
    group: cost_kernel_lags
    description: Solar lag lengthened by 2 years
    budget: secondary
    overrides:
      lag_solar: 4

  - run_id: K_WIND_SHORT_LAG
    group: cost_kernel_lags
    description: Wind lag shortened by 2 years
    budget: secondary
    overrides:
      lag_onwind: 2

  - run_id: K_WIND_LONG_LAG
    group: cost_kernel_lags
    description: Wind lag lengthened by 2 years
    budget: secondary
    overrides:
      lag_onwind: 6

  - run_id: K_BAT_SHORT_LAG
    group: cost_kernel_lags
    description: Battery lag shortened by 2 years
    budget: secondary
    overrides:
      lag_battery: 0

  - run_id: K_BAT_LONG_LAG
    group: cost_kernel_lags
    description: Battery lag lengthened by 2 years
    budget: secondary
    overrides:
      lag_battery: 4

  - run_id: JK1_ALL_SHORT_LAGS
    group: joint_kernel_lags
    description: All procurement lags shortened
    budget: core
    overrides:
      lag_solar: 0
      lag_onwind: 2
      lag_battery: 0

  - run_id: JK2_ALL_LONG_LAGS
    group: joint_kernel_lags
    description: All procurement lags lengthened
    budget: core
    overrides:
      lag_solar: 4
      lag_onwind: 6
      lag_battery: 4

  - run_id: BAT1_LOW_PSI
    group: battery_mapping
    description: Low lithium-ion-equivalent mapping
    budget: core
    overrides:
      battery_psi_scenario: low

  - run_id: BAT2_HIGH_PSI
    group: battery_mapping
    description: High lithium-ion-equivalent mapping
    budget: core
    overrides:
      battery_psi_scenario: high

  - run_id: BAT3_CONSTANT_PSI
    group: battery_mapping
    description: Constant psi fixed at recent historical value
    budget: secondary
    overrides:
      battery_psi_scenario: constant_recent

  - run_id: BAT4_STATIONARY_ONLY
    group: battery_mapping
    description: Stationary grid batteries only for battery learning experience
    budget: core
    overrides:
      battery_learning_boundary: stationary_only

  - run_id: CF1_WIND_FLOOR
    group: cost_floors
    description: Onshore wind cost floor
    budget: core
    overrides:
      cost_floor_onwind: 450

  - run_id: CF4_ALL_FLOORS_50
    group: cost_floors
    description: Solar, wind, and battery floors at 50 percent of 2025 cost
    budget: core
    overrides:
      cost_floor_mode: all_clean_tech_50pct_2025

  - run_id: CF5_ALL_FLOORS_70
    group: cost_floors
    description: Solar, wind, and battery floors at 70 percent of 2025 cost
    budget: core
    overrides:
      cost_floor_mode: all_clean_tech_70pct_2025

  - run_id: R1_NO_DEPLOYMENT_FEEDBACK
    group: recursive_learning
    description: Modeled deployment does not update cumulative experience
    budget: core
    overrides:
      recursive_cost_update: false

  - run_id: R4_LONG_UPDATE_DELAY
    group: recursive_learning
    description: Deployment affects costs only after two planning blocks
    budget: core
    overrides:
      cost_update_delay_blocks: 2

  - run_id: P1_BINDING_CO2_CAP
    group: policy_boundary
    description: Add a binding cumulative or net-zero-compatible CO2 policy boundary
    budget: core
    overrides:
      emissions_cap: binding_policy
```

---

## Run expansion logic

For each selected run:

1. Start from `baseline.parameters`.
2. Apply the run-specific `overrides`.
3. Resolve `n_draws`:

   * If CLI `--n-draws` is provided, use it.
   * Else use the run’s `budget`.
   * Else use `ensemble.default_budget`.
4. Generate draw-level jobs:

   * one job per `run_id × draw_id`.
5. Save one output directory per run:

```text
outputs/ensemble/{ensemble_name}/{run_id}/
  resolved_config.yaml
  manifest.csv
  draws/
    draw_000001/
    draw_000002/
    ...
  summaries/
    outcome_summary.csv
    quantiles.csv
    fast_transition_metrics.csv
```

---

## Manifest

Write a manifest before launching jobs:

```csv
ensemble_name,run_id,group,draw_id,learning_seed,status,output_path
```

Status values:

```text
pending
running
complete
failed
skipped
```

The runner should update the manifest after each draw.

---

## Common random numbers

Phase 1 should use draw-matched learning seeds across runs. This matches the current learning runtime contract, where stochastic learning, fossil-price uncertainty, and stochastic deployment thresholds are keyed primarily from `learning_seed`.

Implement deterministic draw seeds as:

```python
learning_seed = draw_id
```

When `--common-random-numbers` is active, do not include `run_id` in the learning seed. For example, draw `17` should use `learning_seed = 17` or the equivalent token `s0017` in `BASE`, `L1_CGRW`, `W0_NO_WEDGE`, `D1_LOW_DEMAND`, etc.

If `learning.monte_carlo.seed_mode=random` is used, generate one reproducible seed list from `base_seed` and reuse that same ordered list for every run. If `learning.monte_carlo.seed_mode=sequential` is used, reuse the same sequential seeds for every run.

Future phase-2 stream separation may use deterministic seeds like:

```python
stream_seed = hash_function(base_seed, draw_id, stochastic_stream_name)
```

with separate streams such as:

```text
technology_cost_shocks
posterior_parameter_draws
latent_state_draws
fossil_price_shocks
demand_shocks
deployment_threshold_shocks
solver_tiebreak_seed
```

For now, the acceptance requirement is draw-matched `learning_seed`s. If a sensitivity disables a stochastic channel, keep the other enabled channels aligned through the same draw seed wherever the current model allows.

---

## Resume behavior

When `--resume` is active:

* Skip draw outputs that have a valid completion marker.
* Re-run failed or incomplete draws.
* Do not overwrite completed summaries unless explicitly requested.
* After all draws finish, regenerate run-level summaries.

Each draw should write:

```text
draw_complete.json
```

containing:

```json
{
  "run_id": "BASE",
  "draw_id": 1,
  "learning_seed": 1,
  "completed": true,
  "timestamp": "...",
  "model_status": "optimal"
}
```

---

## Summary outputs

For each run, compute and save:

```csv
outcome_summary.csv
```

with at least:

```text
run_id
group
n_draws
renewable_share_2035_mean
renewable_share_2040_mean
renewable_share_2050_mean
fossil_share_2035_mean
fossil_share_2040_mean
fossil_share_2050_mean
solar_capacity_2050_mean
onwind_capacity_2050_mean
battery_energy_capacity_2050_mean
annual_emissions_2050_mean
cumulative_emissions_2025_2050_mean
discounted_system_cost_mean
fast_transition_probability
fast_transition_probability_se
```

Also save quantiles:

```text
p05, p10, p25, p50, p75, p90, p95
```

for all key outcomes.

For paired comparisons to baseline, compute:

```csv
paired_differences_vs_BASE.csv
```

using draw-matched differences:

```text
delta_cumulative_emissions
delta_system_cost
delta_solar_capacity
delta_battery_capacity
delta_fast_transition_indicator
```

This is important because common random numbers reduce Monte Carlo noise.

Also save Monte Carlo precision diagnostics:

```csv
monte_carlo_precision.csv
```

with at least:

```text
run_id
metric
n_draws
mean
standard_error
ci95_low
ci95_high
prefix_100_mean
prefix_250_mean
prefix_500_mean
prefix_1000_mean
```

For binary outcomes such as fast-transition indicators, report binomial standard errors and confidence intervals. For paired differences versus `BASE`, report standard errors on the draw-matched differences, not only unpaired run-level uncertainty.

---

## Adaptive scaling workflow

The system should support this workflow:

### Step 1: Screening

```bash
python run_ensemble.py \
  --config configs/sensitivity_ensemble.yaml \
  --budget screening \
  --max-runs 40 \
  --common-random-numbers \
  --resume
```

This runs about 100 draws per selected run.

### Step 2: Promote important runs

After screening, manually select runs that materially affect conclusions.

```bash
python run_ensemble.py \
  --config configs/sensitivity_ensemble.yaml \
  --run-ids BASE,L1_CGRW,L2_WAY_RHO035,L4_DETERMINISTIC_LEARNING,W0_NO_WEDGE,JW1_TIGHT_WEDGE,JW2_LOOSE_WEDGE,D1_LOW_DEMAND,D2_HIGH_DEMAND,F0_DETERMINISTIC_FOSSIL,R1_NO_DEPLOYMENT_FEEDBACK,TW1_ORIGIN_2020,WU0_DETERMINISTIC_THRESHOLDS,K0_POINT_COST,P1_BINDING_CO2_CAP \
  --n-draws 500 \
  --common-random-numbers \
  --resume
```

### Step 3: Headline runs

```bash
python run_ensemble.py \
  --config configs/sensitivity_ensemble.yaml \
  --run-ids BASE,L1_CGRW,L2_WAY_RHO035,L4_DETERMINISTIC_LEARNING,W0_NO_WEDGE,JW1_TIGHT_WEDGE,JW2_LOOSE_WEDGE \
  --n-draws 1000 \
  --common-random-numbers \
  --resume
```

If affordable, run BASE at 2000 draws:

```bash
python run_ensemble.py \
  --config configs/sensitivity_ensemble.yaml \
  --run-ids BASE \
  --n-draws 2000 \
  --resume
```

---

## Acceptance criteria

The implementation is complete when:

1. I can select arbitrary run IDs from the YAML.
2. I can override the number of MC draws from the command line.
3. I can run only the first `N` selected runs using `--max-runs`.
4. I can resume failed or incomplete ensembles.
5. Outputs are stored in a predictable directory structure.
6. Each run writes a resolved config and manifest.
7. Draw-level seeds are reproducible.
8. Common random numbers are supported through draw-matched `learning_seed`s across all selected runs.
9. Run-level summary tables are generated automatically.
10. Paired differences versus BASE are computed where draw IDs overlap.
11. Monte Carlo precision diagnostics are generated for headline metrics, including standard errors and 95 percent confidence intervals.
12. Convergence-by-prefix diagnostics are generated for promoted runs where enough draws exist.

---

## Practical default

For now, set the default to:

```yaml
ensemble:
  default_budget: screening
```

Then promote runs by explicitly passing `--n-draws 250`, `--n-draws 500`, or `--n-draws 1000`.

This avoids accidentally launching a very expensive full ensemble.

```
```
