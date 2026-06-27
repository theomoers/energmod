#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_ENV_NAME="${LEARNING_CONDA_ENV:-/shared/share_cki25/envs/sh-pypsa-earth-main}"
SECTOR_NAME="Global_200/bootstrap_capacity_only_2020_2025"
RESULT_SECTOR_REL="results/${SECTOR_NAME}"
RESULT_SECTOR_DIR="${ROOT_DIR}/results/${SECTOR_NAME}"
DRY_RUN="${DRY_RUN:-1}"
SNAKEMAKE_JOBS="${SNAKEMAKE_JOBS:-4}"

target_2025_postnetwork="${RESULT_SECTOR_REL}/postnetworks/elec_s_200_ec_lcopt_1h_1h_2025_0.071_AB_0.0export_base.nc"

forced_outputs=(
  "${RESULT_SECTOR_REL}/prenetworks-brownfield/elec_s_200_lcopt_1h_1h_2020_0.071_AB_0.0export_base.nc"
  "${RESULT_SECTOR_REL}/prenetworks-learning/elec_s_200_lcopt_1h_1h_2020_0.071_AB_0.0export_base.nc"
  "${RESULT_SECTOR_REL}/postnetworks/elec_s_200_ec_lcopt_1h_1h_2020_0.071_AB_0.0export_base.nc"
  "${RESULT_SECTOR_REL}/learning/state_committed_elec_s_200_lcopt_1h_1h_2020_0.071_AB_0.0export_base.json"
  "${RESULT_SECTOR_REL}/prenetworks-brownfield/elec_s_200_lcopt_1h_1h_2025_0.071_AB_0.0export_base.nc"
  "${RESULT_SECTOR_REL}/prenetworks-learning/elec_s_200_lcopt_1h_1h_2025_0.071_AB_0.0export_base.nc"
  "${RESULT_SECTOR_REL}/postnetworks/elec_s_200_ec_lcopt_1h_1h_2025_0.071_AB_0.0export_base.nc"
)

ensure_link() {
  local link_path="$1"
  local target_path="$2"

  mkdir -p "$(dirname "$link_path")"
  if [[ -L "$link_path" ]]; then
    ln -sfn "$target_path" "$link_path"
    return 0
  fi
  if [[ -e "$link_path" ]]; then
    echo "Refusing to replace existing non-symlink path: $link_path" >&2
    exit 1
  fi
  ln -s "$target_path" "$link_path"
}

prepare_existing_inputs() {
  mkdir -p "${RESULT_SECTOR_DIR}/learning"
  ensure_link \
    "${RESULT_SECTOR_DIR}/prenetworks" \
    "../prenetworks"
  ensure_link \
    "${RESULT_SECTOR_DIR}/tsam_clustering" \
    "../tsam_clustering"
  ensure_link \
    "${RESULT_SECTOR_DIR}/configs/config.yaml" \
    "../../bootstrap_AB/configs/config.yaml"
  local baseyear_brownfield="${RESULT_SECTOR_DIR}/prenetworks-brownfield/elec_s_200_lcopt_1h_1h_2020_0.071_AB_0.0export_base.nc"
  if [[ -L "$baseyear_brownfield" ]]; then
    unlink "$baseyear_brownfield"
  fi
}

cd "$ROOT_DIR"

if ! command -v snakemake >/dev/null 2>&1; then
  source /apps/anaconda3/etc/profile.d/conda.sh
  conda activate "$CONDA_ENV_NAME"
fi

prepare_existing_inputs

cmd=(
  snakemake
  "-j${SNAKEMAKE_JOBS}"
  "$target_2025_postnetwork"
  --configfile
  config.myopic.yaml
  config.learning.yaml
  validation/config.iteration_common.yaml
  configs/bootstrap_capacity_only_2025.yaml
  --nolock
  --forcerun
  "${forced_outputs[@]}"
  --allowed-rules
  add_existing_baseyear_learning_bootstrap
  apply_learning_costs_learning_bootstrap
  solve_network_myopic_learning_bootstrap
  export_postsolve_learning_costs_learning_bootstrap
  add_brownfield_learning_bootstrap
  copy_config
  --rerun-triggers
  code
)

if [[ "$DRY_RUN" == "1" ]]; then
  cmd+=(-n)
fi

echo "Running capacity-only 2020/2025 constrained bootstrap chain"
echo "  sector_name=${SECTOR_NAME}"
echo "  dry_run=${DRY_RUN}"
echo "  snakemake_jobs=${SNAKEMAKE_JOBS}"
echo "  target=${target_2025_postnetwork}"

"${cmd[@]}"

