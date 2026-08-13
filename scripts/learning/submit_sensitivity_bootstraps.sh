#!/usr/bin/env bash
set -euo pipefail

BOOT="${BOOT:-/shared/share_cki25/energymod/bootstrap-states/sensitivity_ne_si_6000}"
GRID_MEM="${GRID_MEM:-400G}"
GRID_NCPUS="${GRID_NCPUS:-40}"
GRID_SUBMIT="${GRID_SUBMIT:-batch}"

submit_bootstrap() {
  local group="$1"
  local scenario_name="$2"
  shift 2

  grid_run "--grid_mem=${GRID_MEM}" "--grid_ncpus=${GRID_NCPUS}" "--grid_submit=${GRID_SUBMIT}" \
    bash scripts/learning/run_learning_bootstrap_job.sh \
    --scenario-name "$scenario_name" \
    "$@" \
    --save-bootstrap-state "$BOOT/$group"
}

submit_bootstrap AB bootstrap_AB
submit_bootstrap DF bootstrap_DF --overlay-config configs/sensitivity_bootstrap_DF.yaml
submit_bootstrap NZ bootstrap_NZ --overlay-config configs/sensitivity_bootstrap_NZ.yaml
submit_bootstrap TX0 bootstrap_TX0 --overlay-config configs/sensitivity_bootstrap_TX0.yaml
submit_bootstrap TX1 bootstrap_TX1 --overlay-config configs/sensitivity_bootstrap_TX1.yaml
submit_bootstrap TD0 bootstrap_TD0 --overlay-config configs/sensitivity_bootstrap_TD0.yaml
submit_bootstrap TD1 bootstrap_TD1 --overlay-config configs/sensitivity_bootstrap_TD1.yaml
