# Stochastic Learning Runs and Cluster Shape

This note records the production-facing run pattern for the stochastic learning integration.

There is now exactly one supported cluster submission entrypoint:

- `python scripts/learning/submit_learning_cluster_array.py ...`

Everything else in `scripts/learning/` is either a local debug helper or an internal execution helper used by that submitter.

## What Is Self-Contained

The learning integration is now self-contained inside `energymod`:

- learning artifacts live in `data/learning-data`
- the runtime manifest is `data/learning-data/manifest.json`
- deterministic Wright support data for `legacy_curve` is inside `data/learning-data/artifacts/support/legacy_curve`
- repo resource symlinks now point to the repo-local `permstorage/` tree rather than a machine-specific absolute path

As a result, `energymod` no longer needs the `learning_estimation` checkout at runtime.

## Supported Runtime Models

Stochastic production runs support:

- `shared_state_bayesian_regime_wright`
- `way_fixed_rho_benchmark_035`
- `correlated_geometric_random_walk`

The deterministic baseline remains:

- `legacy_curve`

## Recommended Job Granularity

For cluster production, the recommended unit of work is:

- one model
- one seed
- one scenario

In other words, Monte Carlo should be implemented on the cluster as many independent Snakemake jobs, not as one giant multi-seed Snakemake invocation.

This is the preferred shape because it:

- isolates failures cleanly
- makes reruns simple
- avoids coupling many seeds inside one scheduler job
- matches the GRID@CBS “many independent jobs” model well

The aggregate target `solve_sector_networks_myopic_stochastic_mc` is still useful locally for validation, but it should not be the default production submission mode on the cluster.

## Local Debug Only

Single stochastic job:

```bash
JOBS=4 bash scripts/learning/run_learning_stochastic_job.sh shared_state_bayesian_regime_wright 7
```

Dry-run:

```bash
JOBS=4 bash scripts/learning/run_learning_stochastic_job.sh shared_state_bayesian_regime_wright 7 --dry-run
```

The helper script generates a temporary overlay config that enforces:

- a unique `run.sector_name` per model-seed job
- `scenario.learning_rate = ["base"]`
- `scenario.learning_model = [<selected stochastic model>]`
- `learning.engine = stochastic_forecast`
- `learning.selected_model = <selected stochastic model>`
- `learning.seed = <selected seed>`
- `learning.monte_carlo.enable = false`
- `permstore.path = permstorage`

Internally it runs:

```bash
snakemake -j4 solve_sector_networks_myopic \
  --configfile config.myopic.yaml config.learning.yaml <temp_overlay>.yaml \
  --rerun-trigger mtime
```

The unique `run.sector_name` is important. It prevents unrelated non-learning outputs from colliding across different model-seed jobs.

These commands are useful for local debugging and smoke tests. They are not the cluster submission interface.

## Cluster Submission Shape

The cluster-facing pattern should be:

1. upload the full `energymod` repo, including:
   - `data/learning-data`
   - `permstorage`
   - the normal config and script tree
2. submit one job per model x seed x scenario
3. let Snakemake execute the myopic chain inside that one job

### Single Supported Submission Entry Point

The preferred submission entrypoint is now:

```bash
python scripts/learning/submit_learning_cluster_array.py --scenario-name myscenario --print-only
```

The Python submitter reads:

- `config.myopic.yaml`
- `config.learning.yaml`
- any extra YAML overlays you pass on the command line

and constructs one array task per:

- model
- seed
- scenario invocation

In other words, you call the submitter once per scenario configuration, and the array expands over `model x seed`.

The actual SGE array target is a short shell wrapper:

```bash
scripts/learning/run_learning_cluster_array_task.sh
```

That wrapper:

- activates the target conda environment
- reads `SGE_TASK_ID`
- calls the Python worker mode in:
  - `scripts/learning/submit_learning_cluster_array.py`

So the cluster pattern now matches the preferred example shape:

- local Python submitter builds the manifest and calls `qsub`
- the array executes a short shell script
- the shell script invokes a Python worker that uses `SGE_TASK_ID`

The submitter uses:

- `learning.monte_carlo.stochastic_models`
- `learning.monte_carlo.draws`
- `learning.monte_carlo.seed_mode`
- `learning.monte_carlo.random_seed`

to generate the task manifest automatically.

Example print-only submission:

```bash
python scripts/learning/submit_learning_cluster_array.py \
  --scenario-name myscenario \
  --print-only
```

Example real submission:

```bash
python scripts/learning/submit_learning_cluster_array.py \
  --scenario-name myscenario \
  --job-root /scratch/$USER
```

### Internal Execution Path

The submitter is the only supported public cluster interface.

Internally, each array task then:

1. runs `scripts/learning/run_learning_cluster_array_task.sh`
2. calls the Python worker mode in `scripts/learning/submit_learning_cluster_array.py --worker`
3. stages a per-task working copy via `scripts/learning/run_learning_cluster_job.sh`
4. executes the actual Snakemake solve through `scripts/learning/run_learning_stochastic_job.sh`

Those helper scripts are internal implementation details of the one supported submission system. They should not be used as separate cluster submission paths.

## Pre-Upload Checklist

Before uploading to the cluster, verify:

- `data/learning-data/manifest.json` exists
- `data/learning-data/artifacts/origin_2020/...` exists for the selected training window
- `data/learning-data/artifacts/full_sample/...` exists if you plan to use full-sample training
- `permstorage/` exists in the repo and contains the resources needed by the selected scenario
- `config.myopic.yaml` and `config.default.yaml` point to repo-local `permstorage`
- no repo resource symlinks are broken

## First Files to Inspect When a Run Fails

Check these in order:

- `logs/apply_learning_costs_*.log`
- `logs/export_postsolve_learning_costs_*.log`
- `logs/*_solver.log`
- `results/.../learning/cost_log_*.csv`
- `results/.../learning/state_proposed_*.json`
- `results/.../learning/state_committed_*.json`

If the failure happens before optimization, the learning logs and state JSONs are usually the fastest place to look.

## Current Production Guidance

Use this pattern for production stochastic learning runs:

- one cluster job = one model x one seed x one scenario
- `learning_rate = base`
- `battery_power` stays on deterministic default cost files
- use `scripts/learning/submit_learning_cluster_array.py` for submission
- do not submit `run_learning_cluster_job.sh` or `run_learning_stochastic_job.sh` directly
- let the submitter manage the shell wrapper, worker dispatch, and staged execution internally

That is the least coupled and easiest-to-rerun production shape for the current integration.
