# Learning Runtime Operator Note

This note records the practical operator-facing commands for the integrated learning runtime.

## Runtime Modes

The runtime switch is in `config.learning.yaml`:

- `learning.engine: legacy_curve`
- `learning.engine: stochastic_forecast`

Use:

- `legacy_curve` for the deterministic Wright benchmark
- `stochastic_forecast` for the shortlisted stochastic models

The selected runtime model is:

- `learning.selected_model`

Supported stochastic values:

- `shared_state_bayesian_regime_wright`
- `way_fixed_rho_benchmark_035`
- `correlated_geometric_random_walk`

## Single-Path Runs

For one stochastic job:

```bash
JOBS=4 bash scripts/learning/run_learning_stochastic_job.sh shared_state_bayesian_regime_wright 7
```

Dry-run:

```bash
JOBS=4 bash scripts/learning/run_learning_stochastic_job.sh shared_state_bayesian_regime_wright 7 --dry-run
```

## Monte Carlo Runs

Monte Carlo is configured in `config.learning.yaml` under:

- `learning.monte_carlo.enable`
- `learning.monte_carlo.stochastic_models`
- `learning.monte_carlo.draws`
- `learning.monte_carlo.seed_mode`
- `learning.monte_carlo.random_seed`

Example:

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
```

For local aggregate validation:

```bash
MPLCONFIGDIR=/tmp GRB_LICENSE_FILE=/path/to/gurobi.lic conda run -n pypsa-earth snakemake -j4 solve_sector_networks_myopic_stochastic_mc --configfile config.myopic.yaml config.learning.yaml --rerun-trigger mtime
```

For cluster production, prefer the Python array submitter:

```bash
python scripts/learning/submit_learning_cluster_array.py --scenario-name myscenario
```

## Runtime Scope

In v1:

- `solar_power` learns
- `onwind_power` learns
- `battery_energy` learns
- `battery_power` remains on deterministic default cost files

## Important Guard

Stochastic runs require:

- `scenario.learning_rate = base`

Non-base `learning_rate` scenarios are only valid for `legacy_curve`.
