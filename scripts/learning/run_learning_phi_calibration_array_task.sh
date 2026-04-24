#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DEFAULT_JOB_ROOT="${ROOT_DIR}/cluster_workdirs"
DEFAULT_CONDA_ENV="/shared/share_cki25/envs/sh-pypsa-earth-main"

if [[ $# -gt 0 && -z "${PHI_CALIBRATION_ARRAY_MANIFEST:-}" && "${1:-}" != *=* ]]; then
  PHI_CALIBRATION_ARRAY_MANIFEST="$1"
  shift
fi

for kv in "$@"; do
  case "$kv" in
    *=*) export "$kv" ;;
  esac
done

PHI_CALIBRATION_ARRAY_MANIFEST="${PHI_CALIBRATION_ARRAY_MANIFEST:-}"
PHI_CALIBRATION_JOB_ROOT="${PHI_CALIBRATION_JOB_ROOT:-${DEFAULT_JOB_ROOT}}"
CONDA_ENV_NAME="${PHI_CALIBRATION_CONDA_ENV:-${DEFAULT_CONDA_ENV}}"
export JOBS="${JOBS:-${NSLOTS:-4}}"

if [[ -z "${PHI_CALIBRATION_ARRAY_MANIFEST}" ]]; then
  echo "Usage: $0 <task_manifest.json> [KEY=VALUE ...]" >&2
  echo "PHI_CALIBRATION_ARRAY_MANIFEST is required" >&2
  exit 1
fi

echo "===== energymod phi calibration array task ====="
echo "Host: ${HOSTNAME:-unknown}"
echo "Job: ${JOB_ID:-?}  Task: ${SGE_TASK_ID:-?}"
echo "Manifest: ${PHI_CALIBRATION_ARRAY_MANIFEST}"
echo "Job root: ${PHI_CALIBRATION_JOB_ROOT}"
echo "Conda env: ${CONDA_ENV_NAME}"
echo "Snakemake jobs: ${JOBS}"
echo "================================================"

source /apps/anaconda3/etc/profile.d/conda.sh && conda activate "${CONDA_ENV_NAME}"
cd "${ROOT_DIR}"

python scripts/learning/submit_learning_phi_calibration_array.py \
  --worker \
  --manifest "${PHI_CALIBRATION_ARRAY_MANIFEST}" \
  --job-root "${PHI_CALIBRATION_JOB_ROOT}"
