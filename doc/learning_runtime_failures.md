# Learning Runtime Failure Note

This note records the common hard-fail modes for the integrated learning runtime.

## Common Failures

### Non-base learning rate with stochastic model

Symptom:

- stochastic run aborts before solve

Cause:

- `scenario.learning_rate` is not `base`

Fix:

- use `learning_rate: [base]` for all stochastic runs

### Missing stochastic artifact field

Symptom:

- `scripts/learning/apply_learning_costs.py` fails while loading the manifest or model bundle

Cause:

- missing required field in `data/learning-data/manifest.json`
- missing `initial_state`
- missing model artifact directory

Fix:

- refresh the learning-data export inside `energymod`

### Missing previous committed state

Symptom:

- later-horizon stochastic run fails before writing the prenetwork

Cause:

- previous committed state JSON is missing

Fix:

- check the previous horizon output under `results/.../learning/state_committed_*.json`

### Missing realized block history

Symptom:

- Way or shared-state run hard-fails while constructing realized five-year growth

Cause:

- required cumulative capacity is missing at the block start or block end year

Fix:

- inspect historical deployment data and the relevant committed previous-horizon state

### Malformed shared-state runtime state

Symptom:

- shared-state model fails while loading regime probabilities or transition state

Cause:

- malformed or incomplete shared-state JSON

Fix:

- inspect `state_proposed_*.json` and `state_committed_*.json`

## First Files To Inspect

- `logs/apply_learning_costs_*.log`
- `logs/export_postsolve_learning_costs_*.log`
- `logs/*_solver.log`
- `results/.../learning/cost_log_*.csv`
- `results/.../learning/state_proposed_*.json`
- `results/.../learning/state_committed_*.json`
