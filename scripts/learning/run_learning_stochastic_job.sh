#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/learning/run_learning_stochastic_job.sh <model> <seed> [--dry-run]

Supported models:
  - shared_state_bayesian_regime_wright
  - way_fixed_rho_benchmark_035
  - correlated_geometric_random_walk

Environment variables:
  JOBS   Snakemake parallelism for the solve chain (default: 4)

This is a local/manual execution helper. For cluster submission, use
scripts/learning/submit_learning_cluster_array.py instead.
EOF
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage >&2
  exit 1
fi

MODEL="$1"
SEED="$2"
DRY_RUN_FLAG="${3:-}"

case "$MODEL" in
  shared_state_bayesian_regime_wright|way_fixed_rho_benchmark_035|correlated_geometric_random_walk)
    ;;
  *)
    echo "Unsupported stochastic learning model: $MODEL" >&2
    usage >&2
    exit 1
    ;;
esac

if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
  echo "Seed must be a non-negative integer, got: $SEED" >&2
  exit 1
fi

if [[ -n "$DRY_RUN_FLAG" && "$DRY_RUN_FLAG" != "--dry-run" ]]; then
  echo "Only optional flag supported is --dry-run" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
TMPDIR_ROOT="${TMPDIR:-/tmp}"
OVERLAY_FILE="$(mktemp "$TMPDIR_ROOT/energymod_learning_job.XXXXXX.yaml")"
trap 'rm -f "$OVERLAY_FILE"' EXIT
JOB_SECTOR_NAME="Global_200_${MODEL}_seed_s$(printf '%04d' "$SEED")"

cat >"$OVERLAY_FILE" <<EOF
run:
  allow_scenario_failure: false
  sector_name: "$JOB_SECTOR_NAME"

scenario:
  learning_rate:
    - "base"
  learning_model:
    - "$MODEL"

permstore:
  path: permstorage

learning:
  enabled: true
  engine: stochastic_forecast
  selected_model: "$MODEL"
  seed: $SEED
  monte_carlo:
    enable: false
EOF

CMD=(
  snakemake
  "-j${JOBS:-4}"
  solve_sector_networks_myopic
  --configfile
  config.myopic.yaml
  config.learning.yaml
  "$OVERLAY_FILE"
  --rerun-trigger
  mtime
)

if [[ "$DRY_RUN_FLAG" == "--dry-run" ]]; then
  CMD+=(-n)
fi

echo "Running stochastic learning job:"
echo "  model=$MODEL"
echo "  seed=$SEED"
echo "  sector_name=$JOB_SECTOR_NAME"
echo "  overlay=$OVERLAY_FILE"

cd "$ROOT_DIR"
"${CMD[@]}"
