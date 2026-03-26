# Learning Cluster Handoff

This note is the practical handoff for the next AI agent or operator running the stochastic learning workflow on the cluster.

## Current Status

The learning integration is implemented in `energymod`.

The runtime now supports:

- `legacy_curve`
- `shared_state_bayesian_regime_wright`
- `way_fixed_rho_benchmark_035`
- `correlated_geometric_random_walk`

The production stochastic cluster shape is:

- one cluster job = one model x one seed x one scenario

Monte Carlo configuration now lives in:

- `config.learning.yaml`

and not in `config.myopic.yaml`.

The learning runtime is self-contained inside `energymod`:

- artifacts: `data/learning-data`
- manifest: `data/learning-data/manifest.json`

No runtime dependency on `learning_estimation` remains.

## What Is Left Before Cluster Testing

There are no major implementation blockers left in the learning integration itself.

The main remaining operational steps are:

1. commit the current `energymod` changes
2. push them to GitHub
3. pull the updated `energymod` repo on the cluster
4. run one small pilot array first
5. inspect logs and outputs
6. only then scale up to the full Monte Carlo run

One caveat from local testing:

- the staged cluster-shaped dry-run from the iCloud-backed local checkout was slowed by repo copying
- no learning-runtime bug was identified in that step
- the cluster should use a normal pulled repo in a standard filesystem, so that local iCloud copy issue should not be treated as a learning blocker

## Files To Know

Core config:

- `config.myopic.yaml`
- `config.learning.yaml`

Core runtime:

- `scripts/learning/apply_learning_costs.py`
- `scripts/learning/export_postsolve_learning_costs.py`
- `scripts/learning/learning_data_io.py`

Cluster submission:

- `scripts/learning/submit_learning_cluster_array.py`
- `scripts/learning/run_learning_cluster_array_task.sh`

Internal cluster execution helpers:

- `scripts/learning/run_learning_cluster_job.sh`
- `scripts/learning/run_learning_stochastic_job.sh`

Docs:

- `doc/learning_stochastic_cluster_run.md`
- `doc/learning_runtime_operator.md`
- `doc/learning_runtime_failures.md`

## Required Config Semantics

For stochastic runs:

- `learning.engine` must be `stochastic_forecast`
- `learning.sample_mode` should be `single_draw`
- `scenario.learning_rate` must be `base`

The Monte Carlo block is:

```yaml
learning:
  monte_carlo:
    enable: true
    include_legacy_curve: false
    stochastic_models:
      - shared_state_bayesian_regime_wright
      - way_fixed_rho_benchmark_035
      - correlated_geometric_random_walk
    draws: 500
    seed_mode: random
    random_seed: 12345
    seed_upper_bound: 1000000000
```

Interpretation:

- `draws` = number of runs per stochastic model
- `seed_mode: random` = auto-generate distinct seeds reproducibly
- `random_seed` = master seed controlling that generation

With the above settings:

- `3` stochastic models
- `500` draws each
- `1500` cluster jobs per scenario

## Recommended Cluster Workflow

### Step 1. Pull the latest repo

On the cluster:

```bash
git pull
```

Then verify:

- `data/learning-data/manifest.json` exists
- `scripts/learning/submit_learning_cluster_array.py` exists
- `scripts/learning/run_learning_cluster_array_task.sh` exists
- `config.learning.yaml` contains the `learning.monte_carlo` block

### Step 2. Check the config

Open:

- `config.learning.yaml`
- `config.myopic.yaml`

Confirm:

- the scenario you want uses `learning_rate: [base]`
- the stochastic model list is correct
- `draws` is the intended run count

### Step 3. Run a print-only submission first

From the `energymod` repo:

```bash
conda run -n pypsa-earth python scripts/learning/submit_learning_cluster_array.py --scenario-name pilot --print-only
```

What to check:

- `ARRAY_SIZE` is what you expect
- the printed `QSUB_CMD` looks sane
- the `TASK_MANIFEST` exists

For example, if:

- `2` models
- `3` draws

then `ARRAY_SIZE` should be `6`.

### Step 4. Inspect the generated task manifest

Open the printed `TASK_MANIFEST` JSON and confirm:

- each row has:
  - `scenario_name`
  - `model`
  - `seed`
- the number of tasks matches:
  - `len(stochastic_models) * draws`

### Step 5. Submit a small pilot

For a pilot, use a very small Monte Carlo setting first, for example:

- one model
- `draws: 2`

Then submit:

```bash
conda run -n pypsa-earth python scripts/learning/submit_learning_cluster_array.py --scenario-name pilot --job-root /scratch/$USER
```

This will submit one SGE array where each task runs:

- one model
- one seed
- one scenario

The array job itself runs the short shell wrapper:

- `scripts/learning/run_learning_cluster_array_task.sh`

That shell wrapper activates conda and calls the Python worker mode in:

- `scripts/learning/submit_learning_cluster_array.py --worker`

The worker then uses the staging and execution helpers internally. Those helpers are not separate submission interfaces and should not be submitted directly.

### Step 6. Inspect early outputs

Look first at:

- cluster stdout/stderr from the array submission
- `logs/apply_learning_costs_*.log`
- `logs/export_postsolve_learning_costs_*.log`
- `logs/*_solver.log`
- `results/.../learning/cost_log_*.csv`
- `results/.../learning/state_proposed_*.json`
- `results/.../learning/state_committed_*.json`

## Expected Runtime Behavior

Deployment-conditioned models:

- `shared_state_bayesian_regime_wright`
- `way_fixed_rho_benchmark_035`

Time-conditioned model:

- `correlated_geometric_random_walk`

Battery treatment in v1:

- `battery_energy` learns
- `battery_power` remains on deterministic default cost files

Timing convention:

- deployment realized in horizon `t` affects costs in `t+5`
- not in the same solve

## Common Failure Modes

### Non-base learning rate

Symptom:

- stochastic run fails before solve

Cause:

- `scenario.learning_rate` is not `base`

### Missing artifact or malformed manifest

Symptom:

- `apply_learning_costs.py` fails before writing the prenetwork

Cause:

- broken or incomplete `data/learning-data`

### Missing previous committed state

Symptom:

- later-horizon stochastic run fails before solve

Cause:

- previous `state_committed_*.json` missing

### Missing realized block history

Symptom:

- Way or shared-state run fails during capacity-path construction

Cause:

- required cumulative deployment is missing at the start or end of a five-year block

### Shared-state state corruption

Symptom:

- shared-state run fails while loading regime state

Cause:

- malformed `state_proposed` or `state_committed` JSON

## Recommended Rollout Order

1. one-model, two-draw pilot
2. one-model, ten-draw pilot
3. all three models, small draw count
4. full production run

## Full Production Example

If you want full production:

- `draws: 500`
- `stochastic_models` = all three shortlisted models

Then run:

```bash
conda run -n pypsa-earth python scripts/learning/submit_learning_cluster_array.py --scenario-name prod500 --job-root /scratch/$USER
```

That will create:

- `1500` tasks per scenario

## Bottom Line

The learning integration is ready for a cluster pilot.

The next agent should:

1. pull the latest `energymod`
2. verify configs and learning artifacts
3. run `--print-only`
4. submit a small pilot array
5. inspect logs and state files
6. then scale up
