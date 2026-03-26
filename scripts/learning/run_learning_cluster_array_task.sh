#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

if [[ -z "${LEARNING_ARRAY_MANIFEST:-}" ]]; then
  echo "LEARNING_ARRAY_MANIFEST is required" >&2
  exit 1
fi

if [[ -z "${LEARNING_JOB_ROOT:-}" ]]; then
  echo "LEARNING_JOB_ROOT is required" >&2
  exit 1
fi

CONDA_ENV_NAME="${LEARNING_CONDA_ENV:-pypsa-earth}"

echo "===== energymod stochastic learning array task ====="
echo "Host: ${HOSTNAME:-unknown}"
echo "Job: ${JOB_ID:-?}  Task: ${SGE_TASK_ID:-?}"
echo "Manifest: ${LEARNING_ARRAY_MANIFEST}"
echo "Job root: ${LEARNING_JOB_ROOT}"
echo "Conda env: ${CONDA_ENV_NAME}"
echo "===================================================="

source /apps/anaconda3/etc/profile.d/conda.sh && conda activate "${CONDA_ENV_NAME}"
cd "${ROOT_DIR}"

python scripts/learning/submit_learning_cluster_array.py \
  --worker \
  --manifest "${LEARNING_ARRAY_MANIFEST}" \
  --job-root "${LEARNING_JOB_ROOT}"
