#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/learning/run_learning_bootstrap_job.sh [--dry-run]

Environment variables:
  Bootstrap always runs with Snakemake parallelism fixed to 100 jobs
  LEARNING_SECTOR_NAME Override the shared sector_name (default: Global_200)

This runs the shared deterministic learning bootstrap through the first two
planning horizons for the stochastic model list configured in config.learning.yaml.
USAGE
}

DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
TMPDIR_ROOT="${TMPDIR:-/tmp}"
OVERLAY_FILE="$(mktemp "$TMPDIR_ROOT/energymod_learning_bootstrap.XXXXXX.yaml")"
trap 'rm -f "$OVERLAY_FILE"' EXIT
JOB_SECTOR_NAME="${LEARNING_SECTOR_NAME:-Global_200}"
SNAKEMAKE_JOBS="100"
CONDA_ENV_NAME="${LEARNING_CONDA_ENV:-/shared/share_cki25/envs/sh-pypsa-earth-main}"

if ! command -v snakemake >/dev/null 2>&1; then
  source /apps/anaconda3/etc/profile.d/conda.sh
  conda activate "$CONDA_ENV_NAME"
fi

cat >"$OVERLAY_FILE" <<EOF2
run:
  allow_scenario_failure: false
  sector_name: "$JOB_SECTOR_NAME"

scenario:
  learning_rate:
    - "base"

permstore:
  path: permstorage

learning:
  enabled: true
  execution_mode: "bootstrap"
EOF2

CMD=(
  snakemake
  "-j${SNAKEMAKE_JOBS}"
  solve_sector_networks_myopic_learning_bootstrap
  --configfile
  config.myopic.yaml
  config.learning.yaml
  "$OVERLAY_FILE"
  --rerun-trigger
  mtime
)

if [[ "$DRY_RUN" -eq 1 ]]; then
  CMD+=(-n)
fi

echo "Running shared learning bootstrap:"
echo "  sector_name=$JOB_SECTOR_NAME"
echo "  jobs=$SNAKEMAKE_JOBS"
echo "  overlay=$OVERLAY_FILE"

cd "$ROOT_DIR"
if command -v stdbuf >/dev/null 2>&1; then
  stdbuf -oL -eL "${CMD[@]}"
else
  "${CMD[@]}"
fi
