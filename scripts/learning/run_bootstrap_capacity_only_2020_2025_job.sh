#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_ENV_NAME="${LEARNING_CONDA_ENV:-/shared/share_cki25/envs/sh-pypsa-earth-main}"
SECTOR_TOKEN="bootstrap_capacity_only_2020_2025"
SECTOR_NAME="Global_200/${SECTOR_TOKEN}"
RESULT_SECTOR_DIR="${ROOT_DIR}/results/${SECTOR_NAME}"
RESOURCE_SECTOR_DIR="${ROOT_DIR}/resources/${SECTOR_NAME}"
SAVE_BOOTSTRAP_STATE="${SAVE_BOOTSTRAP_STATE:-${ROOT_DIR}/bootstrap-states/bootstrap_capacity_only_2020_2025}"
SNAKEMAKE_JOBS="${SNAKEMAKE_JOBS:-4}"
DRY_RUN="${DRY_RUN:-0}"

ensure_link() {
  local link_path="$1"
  local target_path="$2"

  if [[ -L "$link_path" ]]; then
    local current
    current="$(readlink "$link_path")"
    if [[ "$current" == "$target_path" ]]; then
      return 0
    fi
    echo "Refusing to replace existing symlink with different target: $link_path -> $current" >&2
    exit 1
  fi

  if [[ -e "$link_path" ]]; then
    echo "Refusing to replace existing non-symlink path: $link_path" >&2
    exit 1
  fi

  ln -s "$target_path" "$link_path"
}

prepare_global200_inputs() {
  mkdir -p "$RESULT_SECTOR_DIR"
  mkdir -p "$RESOURCE_SECTOR_DIR"

  ensure_link "${RESULT_SECTOR_DIR}/prenetworks" "../prenetworks"
  ensure_link "${RESULT_SECTOR_DIR}/tsam_clustering" "../tsam_clustering"

  local path name
  for path in "${ROOT_DIR}/resources/Global_200"/*; do
    name="$(basename "$path")"
    if [[ "$name" == "$SECTOR_TOKEN" ]]; then
      continue
    fi
    ensure_link "${RESOURCE_SECTOR_DIR}/${name}" "../${name}"
  done
}

run_snakemake() {
  local overlay="$1"
  local cmd=(
    snakemake
    "-j${SNAKEMAKE_JOBS}"
    solve_sector_networks_myopic_learning_bootstrap
    --configfile
    config.myopic.yaml
    config.learning.yaml
    validation/config.iteration_common.yaml
    "$overlay"
    --nolock
    --rerun-incomplete
    --rerun-triggers
    code
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    cmd+=(-n)
  fi

  echo "Running bootstrap capacity-only stage:"
  echo "  overlay=$overlay"
  echo "  sector_name=$SECTOR_NAME"
  echo "  snakemake_jobs=$SNAKEMAKE_JOBS"
  echo "  dry_run=$DRY_RUN"
  "${cmd[@]}"
}

cd "$ROOT_DIR"

if ! command -v snakemake >/dev/null 2>&1; then
  source /apps/anaconda3/etc/profile.d/conda.sh
  conda activate "$CONDA_ENV_NAME"
fi

prepare_global200_inputs

run_snakemake configs/bootstrap_capacity_only_2020.yaml
run_snakemake configs/bootstrap_capacity_only_2025.yaml

if [[ "$DRY_RUN" != "1" ]]; then
  python scripts/learning/bootstrap_state_store.py save \
    --source-sector-dir "$RESULT_SECTOR_DIR" \
    --target-dir "$SAVE_BOOTSTRAP_STATE"
fi

