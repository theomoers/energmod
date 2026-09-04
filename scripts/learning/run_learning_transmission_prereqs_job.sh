#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage:
  bash scripts/learning/run_learning_transmission_prereqs_job.sh TX0|TX1 [--dry-run]

Rebuilds only the future (2030--2050) transmission-sensitive shared inputs:
prepare_sector_network and add_export. It restores the current AB state into a
separate results scenario, unlinks the outputs that will be replaced (so the
AB source cannot be modified through hardlinks), and saves the resulting
prerequisite state for the sensitivity ensemble.
USAGE
}

CASE="${1:-}"
if [[ -z "$CASE" || "$CASE" == "-h" || "$CASE" == "--help" ]]; then
  usage
  exit 0
fi
shift

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
if [[ $# -ne 0 ]]; then
  usage >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
BOOT_ROOT="$ROOT_DIR/bootstrap-states/sensitivity_ne_si_5500_20260811"
SOURCE_DIR="$ROOT_DIR/results/Global_200_v8"
PREREQ_TAG="${PREREQ_TAG:-}"

case "$CASE" in
  TX0|TX1)
    OVERLAY_CONFIG="configs/sensitivity_bootstrap_${CASE}.yaml"
    SCENARIO_NAME="prereqs_${CASE}_20260813"
    SAVE_STATE="$BOOT_ROOT/$CASE"
    ;;
  *)
    echo "Unsupported transmission case: $CASE (expected TX0 or TX1)" >&2
    exit 1
    ;;
esac

if [[ -n "$PREREQ_TAG" ]]; then
  SCENARIO_NAME="${SCENARIO_NAME}_${PREREQ_TAG}"
  SAVE_STATE="${SAVE_STATE}_${PREREQ_TAG}"
fi

if [[ ! -f "$ROOT_DIR/$OVERLAY_CONFIG" ]]; then
  echo "Missing overlay config: $ROOT_DIR/$OVERLAY_CONFIG" >&2
  exit 1
fi
if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Missing AB source directory: $SOURCE_DIR" >&2
  exit 1
fi

# A dry-run validates against the existing AB state: the helper does not
# restore a state while dry-running, so the isolated scenario does not exist.
if [[ "$DRY_RUN" -eq 1 ]]; then
  SCENARIO_ARGS=()
else
  SCENARIO_ARGS=(
    --scenario-name "$SCENARIO_NAME"
    --bootstrap-state-source "$SOURCE_DIR"
    --save-bootstrap-state "$SAVE_STATE"
  )
fi

ARGS=(
  "${SCENARIO_ARGS[@]}"
  --overlay-config "$OVERLAY_CONFIG"
  --allowed-rule solve_sector_networks_myopic_stochastic_shared_inputs
  --allowed-rule prepare_sector_network
  --allowed-rule add_export
)

for YEAR in 2030 2035 2040 2045 2050; do
  BASE="elec_s_200_ec_lcopt_1h_1h_${YEAR}_0.071_AB"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    TARGET_PREFIX="results/Global_200_v8"
  else
    TARGET_PREFIX="results/Global_200_v8/$SCENARIO_NAME"
  fi
  ARGS+=(
    --force-target "$TARGET_PREFIX/prenetworks/${BASE}.nc"
    --force-target "$TARGET_PREFIX/prenetworks/${BASE}_0.0export.nc"
  )
  if [[ "$DRY_RUN" -eq 0 ]]; then
    ARGS+=(
      --unlink-path "prenetworks/${BASE}.nc"
      --unlink-path "prenetworks/${BASE}_0.0export.nc"
      --unlink-path "tsam_clustering/clustering_s_200_lcopt_1h_1h_${YEAR}_0.071_AB.pkl"
      --unlink-path "benchmarks/prepare_network/${BASE}"
    )
  fi
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  ARGS+=(--dry-run)
fi

exec bash "$ROOT_DIR/scripts/learning/run_learning_branch_prereqs_job.sh" "${ARGS[@]}"
