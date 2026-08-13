#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/learning/run_learning_branch_prereqs_job.sh [--scenario-name NAME] [--overlay-config PATH] [--bootstrap-state-source PATH] [--reuse-resources-from PATH] [--save-bootstrap-state PATH] [--dry-run]

Environment variables:
  JOBS                 Snakemake parallelism override
  NSLOTS               Cluster slot count fallback when JOBS is unset
  LEARNING_SECTOR_NAME Override the shared sector_name (default: run.sector_name from config.myopic.yaml)
  LEARNING_CONDA_ENV   Conda env to activate when snakemake is not already on PATH
  LEARNING_SNAKEMAKE_LOCK
                       auto|on|off; auto disables the lock only when
                       --scenario-name creates a separate result dir

This builds the shared 2030+ prenetwork export inputs once before launching the
seed-specific stochastic branch array. A bootstrap state source can be restored
into a fresh scenario before building the future shared inputs.
USAGE
}

DRY_RUN=0
SCENARIO_NAME=""
SAVE_BOOTSTRAP_STATE=""
BOOTSTRAP_STATE_SOURCE=""
EXTRA_CONFIGFILES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --scenario-name)
      if [[ $# -lt 2 ]]; then
        echo "--scenario-name requires an argument" >&2
        exit 1
      fi
      SCENARIO_NAME="$2"
      shift 2
      ;;
    --overlay-config)
      if [[ $# -lt 2 ]]; then
        echo "--overlay-config requires an argument" >&2
        exit 1
      fi
      EXTRA_CONFIGFILES+=("$2")
      shift 2
      ;;
    --bootstrap-state-source)
      if [[ $# -lt 2 ]]; then
        echo "--bootstrap-state-source requires an argument" >&2
        exit 1
      fi
      BOOTSTRAP_STATE_SOURCE="$2"
      shift 2
      ;;
    --save-bootstrap-state)
      if [[ $# -lt 2 ]]; then
        echo "--save-bootstrap-state requires an argument" >&2
        exit 1
      fi
      SAVE_BOOTSTRAP_STATE="$2"
      shift 2
      ;;
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
STAGING_LOCK_FILE="${LEARNING_STAGING_LOCK_FILE:-$ROOT_DIR/.snakemake/learning_staging.lock}"
TMPDIR_ROOT="${TMPDIR:-/tmp}"
OVERLAY_FILE="$(mktemp "$TMPDIR_ROOT/energymod_learning_branch_prereqs.XXXXXX.yaml")"
trap 'rm -f "$OVERLAY_FILE"' EXIT


read_config_sector_name() {
  awk '
    in_run && /^  sector_name:/ {
      value = $0
      sub(/^  sector_name: */, "", value)
      sub(/ *#.*$/, "", value)
      sub(/^"/, "", value)
      sub(/"$/, "", value)
      sub(/^\047/, "", value)
      sub(/\047$/, "", value)
      print value
      exit
    }
    /^run:$/ { in_run = 1; next }
    in_run && /^[A-Za-z0-9_-]+:/ { exit }
  ' "$ROOT_DIR/config.myopic.yaml"
}

sanitize_token() {
  local raw="$1"
  local token
  token="$(printf '%s' "$raw" | tr -cs '[:alnum:]_.-' '_')"
  token="$(printf '%s' "$token" | sed -e 's/^[._-]\+//' -e 's/[._-]\+$//')"
  if [[ -z "$token" ]]; then
    token="default"
  fi
  printf '%s' "$token"
}

CONFIG_SECTOR_NAME="$(read_config_sector_name)"
if [[ -z "$CONFIG_SECTOR_NAME" ]]; then
  CONFIG_SECTOR_NAME="Global_200"
fi

if [[ -n "${LEARNING_SECTOR_NAME:-}" ]]; then
  JOB_SECTOR_NAME="${LEARNING_SECTOR_NAME}"
elif [[ -n "$SCENARIO_NAME" ]]; then
  JOB_SECTOR_NAME="${CONFIG_SECTOR_NAME}/$(sanitize_token "$SCENARIO_NAME")"
else
  JOB_SECTOR_NAME="$CONFIG_SECTOR_NAME"
fi

SNAKEMAKE_JOBS="${JOBS:-${NSLOTS:-4}}"
CONDA_ENV_NAME="${LEARNING_CONDA_ENV:-/shared/share_cki25/envs/sh-pypsa-earth-main}"
SNAKEMAKE_LOCK_MODE="${LEARNING_SNAKEMAKE_LOCK:-auto}"
SNAKEMAKE_LOCK_ARGS=()

case "$SNAKEMAKE_LOCK_MODE" in
  auto)
    if [[ -n "$SCENARIO_NAME" && -z "${LEARNING_SECTOR_NAME:-}" ]]; then
      SNAKEMAKE_LOCK_ARGS+=(--nolock)
    fi
    ;;
  on)
    ;;
  off)
    SNAKEMAKE_LOCK_ARGS+=(--nolock)
    ;;
  *)
    echo "LEARNING_SNAKEMAKE_LOCK must be one of: auto, on, off" >&2
    exit 1
    ;;
esac

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
  execution_mode: "branch"
EOF2

CMD=(
  snakemake
  "-j${SNAKEMAKE_JOBS}"
  solve_sector_networks_myopic_stochastic_shared_inputs
  --configfile
  config.myopic.yaml
  config.learning.yaml
  validation/config.iteration_common.yaml
  "$OVERLAY_FILE"
  "${EXTRA_CONFIGFILES[@]}"
  "${SNAKEMAKE_LOCK_ARGS[@]}"
  --rerun-incomplete
  --rerun-trigger
  mtime
)

if [[ "$DRY_RUN" -eq 1 ]]; then
  CMD+=(-n)
fi

if [[ "$DRY_RUN" -eq 0 && -n "$BOOTSTRAP_STATE_SOURCE" ]]; then
  python "$ROOT_DIR/scripts/learning/bootstrap_state_store.py" restore \
    --source-dir "$BOOTSTRAP_STATE_SOURCE" \
    --target-sector-dir "$ROOT_DIR/results/$JOB_SECTOR_NAME"
fi

echo "Running shared stochastic branch prerequisites:"
echo "  sector_name=$JOB_SECTOR_NAME"
echo "  jobs=$SNAKEMAKE_JOBS"
if [[ "${#SNAKEMAKE_LOCK_ARGS[@]}" -gt 0 ]]; then
  echo "  snakemake_lock=$SNAKEMAKE_LOCK_MODE (--nolock)"
else
  echo "  snakemake_lock=$SNAKEMAKE_LOCK_MODE"
fi
echo "  overlay=$OVERLAY_FILE"
if [[ "${#EXTRA_CONFIGFILES[@]}" -gt 0 ]]; then
  echo "  extra_configfiles=${EXTRA_CONFIGFILES[*]}"
fi
if [[ -n "$BOOTSTRAP_STATE_SOURCE" ]]; then
  echo "  bootstrap_state_source=$BOOTSTRAP_STATE_SOURCE"
fi
if [[ -n "$SAVE_BOOTSTRAP_STATE" ]]; then
  echo "  save_bootstrap_state=$SAVE_BOOTSTRAP_STATE"
fi

if command -v flock >/dev/null 2>&1; then
  mkdir -p "$(dirname "$STAGING_LOCK_FILE")"
  exec 9>"$STAGING_LOCK_FILE"
  flock 9
  echo "  staging_lock=$STAGING_LOCK_FILE"
fi

cd "$ROOT_DIR"
if command -v stdbuf >/dev/null 2>&1; then
  stdbuf -oL -eL "${CMD[@]}"
else
  "${CMD[@]}"
fi

if [[ "$DRY_RUN" -eq 0 && -n "$SAVE_BOOTSTRAP_STATE" ]]; then
  python scripts/learning/bootstrap_state_store.py save \
    --source-sector-dir "$ROOT_DIR/results/$JOB_SECTOR_NAME" \
    --target-dir "$SAVE_BOOTSTRAP_STATE" \
    --prereq-only
fi
