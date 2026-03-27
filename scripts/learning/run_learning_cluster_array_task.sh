#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DEFAULT_JOB_ROOT="${ROOT_DIR}/cluster_workdirs"
DEFAULT_CONDA_ENV="/shared/share_cki25/envs/sh-pypsa-earth-main"

if [[ $# -gt 0 && -z "${LEARNING_ARRAY_MANIFEST:-}" && "${1:-}" != *=* ]]; then
  LEARNING_ARRAY_MANIFEST="$1"
  shift
fi

for kv in "$@"; do
  case "$kv" in
    *=*) export "$kv" ;;
  esac
done

LEARNING_ARRAY_MANIFEST="${LEARNING_ARRAY_MANIFEST:-}"
LEARNING_JOB_ROOT="${LEARNING_JOB_ROOT:-${DEFAULT_JOB_ROOT}}"
CONDA_ENV_NAME="${LEARNING_CONDA_ENV:-${DEFAULT_CONDA_ENV}}"
LEARNING_RUN_MODE="${LEARNING_RUN_MODE:-branch}"
export JOBS="${JOBS:-${NSLOTS:-4}}"
export LEARNING_RUN_MODE

if [[ -z "${LEARNING_ARRAY_MANIFEST}" ]]; then
  echo "Usage: $0 <task_manifest.json> [KEY=VALUE ...]" >&2
  echo "LEARNING_ARRAY_MANIFEST is required" >&2
  exit 1
fi

echo "===== energymod stochastic learning array task ====="
echo "Host: ${HOSTNAME:-unknown}"
echo "Job: ${JOB_ID:-?}  Task: ${SGE_TASK_ID:-?}"
echo "Manifest: ${LEARNING_ARRAY_MANIFEST}"
echo "Job root: ${LEARNING_JOB_ROOT}"
echo "Conda env: ${CONDA_ENV_NAME}"
echo "Run mode: ${LEARNING_RUN_MODE}"
echo "Snakemake jobs: ${JOBS}"
echo "===================================================="

source /apps/anaconda3/etc/profile.d/conda.sh && conda activate "${CONDA_ENV_NAME}"
cd "${ROOT_DIR}"

python scripts/learning/submit_learning_cluster_array.py \
  --worker \
  --manifest "${LEARNING_ARRAY_MANIFEST}" \
  --job-root "${LEARNING_JOB_ROOT}"
