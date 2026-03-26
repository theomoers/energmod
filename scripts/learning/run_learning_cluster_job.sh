#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/learning/run_learning_cluster_job.sh <model> <seed> <job_root> [--dry-run]

This stages a fresh per-job working copy of energymod under <job_root> and then
executes scripts/learning/run_learning_stochastic_job.sh inside that staged copy.
This is an internal helper for the array submission system, not a separate
cluster submission interface.
EOF
}

if [[ $# -lt 3 || $# -gt 4 ]]; then
  usage >&2
  exit 1
fi

MODEL="$1"
SEED="$2"
JOB_ROOT="$3"
DRY_RUN_FLAG="${4:-}"

if [[ -n "$DRY_RUN_FLAG" && "$DRY_RUN_FLAG" != "--dry-run" ]]; then
  echo "Only optional flag supported is --dry-run" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
JOB_TAG="${MODEL}_s$(printf '%04d' "$SEED")"
mkdir -p "$JOB_ROOT"
JOB_DIR="$(cd "$JOB_ROOT" && pwd)/energymod_${JOB_TAG}"

if [[ -e "$JOB_DIR" ]]; then
  echo "Refusing to overwrite existing job directory: $JOB_DIR" >&2
  exit 1
fi

if command -v rsync >/dev/null 2>&1; then
  rsync -a \
    --exclude '.git' \
    --exclude '.snakemake' \
    --exclude 'results' \
    --exclude 'logs' \
    --exclude 'benchmarks' \
    "$ROOT_DIR/" "$JOB_DIR/"
else
  cp -a "$ROOT_DIR" "$JOB_DIR"
  rm -rf "$JOB_DIR/.git" "$JOB_DIR/.snakemake" "$JOB_DIR/results" "$JOB_DIR/logs" "$JOB_DIR/benchmarks"
fi

echo "Staged job working copy at: $JOB_DIR"
cd "$JOB_DIR"
bash scripts/learning/run_learning_stochastic_job.sh "$MODEL" "$SEED" ${DRY_RUN_FLAG:+"$DRY_RUN_FLAG"}
