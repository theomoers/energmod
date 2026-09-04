#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage:
  bash scripts/learning/run_learning_capacity_factor_prereqs_job.sh CF0|CF1|CF2 [--dry-run]

Rebuilds only the capacity-factor-sensitive outputs for a capacity-factor sensitivity:
2020/2025 prepare-sector, bootstrap existing/brownfield/learning/solve/export
chain, and 2030--2050 shared prenetwork exports. It restores the AB state into
an isolated scenario and unlinks every output that will be overwritten, so the
AB source remains untouched.
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
CONDA_ENV_NAME="${LEARNING_CONDA_ENV:-/shared/share_cki25/envs/sh-pypsa-earth-main}"
BOOT_ROOT="$ROOT_DIR/bootstrap-states/sensitivity_ne_si_5500_20260811"
SOURCE_DIR="$ROOT_DIR/results/Global_200_v8"
SNAKEMAKE_JOBS="${JOBS:-${NSLOTS:-4}}"

case "$CASE" in
  CF0|CF1)
    OVERLAY_CONFIG="configs/sensitivity_bootstrap_${CASE}.yaml"
    SCENARIO_NAME="prereqs_${CASE}_20260814"
    SAVE_STATE="$BOOT_ROOT/$CASE"
    ;;
  CF2)
    OVERLAY_CONFIG="configs/sensitivity_bootstrap_${CASE}.yaml"
    SCENARIO_NAME="prereqs_CF2_SOLAR_WIND_PROFILE_CAP_150_20260901"
    SAVE_STATE="$BOOT_ROOT/$CASE"
    ;;
  *)
    echo "Unsupported capacity-factor case: $CASE (expected CF0, CF1, or CF2)" >&2
    exit 1
    ;;
esac

if [[ ! -f "$ROOT_DIR/$OVERLAY_CONFIG" ]]; then
  echo "Missing overlay config: $ROOT_DIR/$OVERLAY_CONFIG" >&2
  exit 1
fi
if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Missing AB source directory: $SOURCE_DIR" >&2
  exit 1
fi

if ! command -v snakemake >/dev/null 2>&1; then
  source /apps/anaconda3/etc/profile.d/conda.sh
  conda activate "$CONDA_ENV_NAME"
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  SECTOR_NAME="Global_200_v8"
else
  SECTOR_NAME="Global_200_v8/$SCENARIO_NAME"
fi
TARGET_DIR="$ROOT_DIR/results/$SECTOR_NAME"
TMPDIR_ROOT="${TMPDIR:-/tmp}"
RUNTIME_OVERLAY="$(mktemp "$TMPDIR_ROOT/energymod_learning_capacity_factor.XXXXXX.yaml")"
trap "rm -f \"$RUNTIME_OVERLAY\"" EXIT

printf "%s\n" \
  "run:" \
  "  allow_scenario_failure: false" \
  "  sector_name: \"$SECTOR_NAME\"" \
  "" \
  "scenario:" \
  "  learning_rate:" \
  "    - \"base\"" \
  "" \
  "permstore:" \
  "  path: permstorage" \
  "" \
  "learning:" \
  "  enabled: true" \
  "  execution_mode: \"bootstrap\"" > "$RUNTIME_OVERLAY"

TARGETS=()
for YEAR in 2020 2025; do
  PRE="results/$SECTOR_NAME/prenetworks/elec_s_200_ec_lcopt_1h_1h_${YEAR}_0.071_AB"
  BROWN="results/$SECTOR_NAME/prenetworks-brownfield/elec_s_200_lcopt_1h_1h_${YEAR}_0.071_AB_0.0export_base.nc"
  LEARN="results/$SECTOR_NAME/prenetworks-learning/elec_s_200_lcopt_1h_1h_${YEAR}_0.071_AB_0.0export_base.nc"
  POST="results/$SECTOR_NAME/postnetworks/elec_s_200_ec_lcopt_1h_1h_${YEAR}_0.071_AB_0.0export_base.nc"
  STATE="results/$SECTOR_NAME/learning/state_committed_elec_s_200_lcopt_1h_1h_${YEAR}_0.071_AB_0.0export_base.json"
  TARGETS+=("${PRE}.nc" "${PRE}_0.0export.nc" "$BROWN" "$LEARN" "$POST" "$STATE")
done
for YEAR in 2030 2035 2040 2045 2050; do
  PRE="results/$SECTOR_NAME/prenetworks/elec_s_200_ec_lcopt_1h_1h_${YEAR}_0.071_AB"
  TARGETS+=("${PRE}.nc" "${PRE}_0.0export.nc")
done
TARGETS+=("results/$SECTOR_NAME/learning/bootstrap_complete_elec_s_200_lcopt_1h_1h_0.071_AB_0.0export_base.txt")

if [[ "$DRY_RUN" -eq 0 ]]; then
  echo "Restoring AB state into $TARGET_DIR"
  python "$ROOT_DIR/scripts/learning/bootstrap_state_store.py" restore \
    --source-dir "$SOURCE_DIR" \
    --target-sector-dir "$TARGET_DIR"

  rm -f "$TARGET_DIR/configs/config.yaml" "$TARGET_DIR/configs/copy_config.log"
  rm -f "$TARGET_DIR/learning/bootstrap_complete_elec_s_200_lcopt_1h_1h_0.071_AB_0.0export_base.txt"
  for YEAR in 2020 2025; do
    BASE="elec_s_200_ec_lcopt_1h_1h_${YEAR}_0.071_AB"
    BROWN="elec_s_200_lcopt_1h_1h_${YEAR}_0.071_AB_0.0export_base.nc"
    rm -f \
      "$TARGET_DIR/prenetworks/${BASE}.nc" \
      "$TARGET_DIR/prenetworks/${BASE}_0.0export.nc" \
      "$TARGET_DIR/tsam_clustering/clustering_s_200_lcopt_1h_1h_${YEAR}_0.071_AB.pkl" \
      "$TARGET_DIR/prenetworks-brownfield/$BROWN" \
      "$TARGET_DIR/prenetworks-learning/$BROWN" \
      "$TARGET_DIR/postnetworks/${BASE}_0.0export_base.nc"
    rm -f "$TARGET_DIR/learning/"*"_${YEAR}_0.071_AB_0.0export_base."{csv,json}
    rm -f "$TARGET_DIR/logs/"*"_${YEAR}_0.071_AB_0.0export_base"*
    find "$TARGET_DIR/benchmarks" -type f -name "*_${YEAR}_0.071_AB_0.0export_base*" -delete
  done
  for YEAR in 2030 2035 2040 2045 2050; do
    BASE="elec_s_200_ec_lcopt_1h_1h_${YEAR}_0.071_AB"
    rm -f \
      "$TARGET_DIR/prenetworks/${BASE}.nc" \
      "$TARGET_DIR/prenetworks/${BASE}_0.0export.nc" \
      "$TARGET_DIR/tsam_clustering/clustering_s_200_lcopt_1h_1h_${YEAR}_0.071_AB.pkl" \
      "$TARGET_DIR/benchmarks/prepare_network/${BASE}"
  done
fi

CMD=(
  snakemake
  "-j${SNAKEMAKE_JOBS}"
  "${TARGETS[@]}"
  --configfile
  config.myopic.yaml
  config.learning.yaml
  validation/config.iteration_common.yaml
  "$RUNTIME_OVERLAY"
  "$OVERLAY_CONFIG"
  --nolock
  --rerun-incomplete
  --rerun-trigger mtime
  --forcerun
  "${TARGETS[@]}"
  --allowed-rules
  prepare_sector_network
  add_export
  add_existing_baseyear_learning_bootstrap
  add_brownfield_learning_bootstrap
  apply_learning_costs_learning_bootstrap
  solve_network_myopic_learning_bootstrap
  export_postsolve_learning_costs_learning_bootstrap
  mark_learning_bootstrap_complete
  copy_config
)

if [[ "$DRY_RUN" -eq 1 ]]; then
  CMD+=(-n)
fi

echo "Running capacity-factor prerequisites: case=$CASE sector_name=$SECTOR_NAME dry_run=$DRY_RUN"
cd "$ROOT_DIR"
if command -v stdbuf >/dev/null 2>&1; then
  stdbuf -oL -eL "${CMD[@]}"
else
  "${CMD[@]}"
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
  python scripts/learning/bootstrap_state_store.py save \
    --source-sector-dir "$TARGET_DIR" \
    --target-dir "$SAVE_STATE" \
    --prereq-only
fi
