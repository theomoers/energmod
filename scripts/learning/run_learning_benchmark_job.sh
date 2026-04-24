#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/learning/run_learning_benchmark_job.sh <model> <job_root> [--scenario-name NAME] [--run-mode full|branch] [--configfile PATH]... [--dry-run]

Supported models:
  - legacy_curve
  - iea_weo_exogenous_path
USAGE
}

if [[ $# -lt 2 ]]; then
  usage >&2
  exit 1
fi

MODEL="$1"
JOB_ROOT="$2"
shift 2

RUN_MODE="full"
DRY_RUN=0
SCENARIO_NAME=""
CONFIGFILES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenario-name)
      if [[ $# -lt 2 ]]; then
        echo "--scenario-name requires an argument" >&2
        exit 1
      fi
      SCENARIO_NAME="$2"
      shift 2
      ;;
    --run-mode)
      if [[ $# -lt 2 ]]; then
        echo "--run-mode requires an argument" >&2
        exit 1
      fi
      RUN_MODE="$2"
      shift 2
      ;;
    --configfile)
      if [[ $# -lt 2 ]]; then
        echo "--configfile requires an argument" >&2
        exit 1
      fi
      CONFIGFILES+=("$2")
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

case "$MODEL" in
  legacy_curve)
    LEARNING_ENGINE="legacy_curve"
    ;;
  iea_weo_exogenous_path)
    LEARNING_ENGINE="exogenous_path"
    ;;
  *)
    echo "Unsupported benchmark learning model: $MODEL" >&2
    usage >&2
    exit 1
    ;;
esac

case "$RUN_MODE" in
  full)
    TARGET_RULE="solve_sector_networks_myopic"
    ;;
  branch)
    TARGET_RULE="solve_sector_networks_myopic_stochastic_branch"
    ;;
  *)
    echo "Unsupported run mode: $RUN_MODE" >&2
    usage >&2
    exit 1
    ;;
esac

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
TMPDIR_ROOT="${TMPDIR:-/tmp}"
OVERLAY_FILE="$(mktemp "$TMPDIR_ROOT/energymod_benchmark_job.XXXXXX.yaml")"
trap 'rm -f "$OVERLAY_FILE"' EXIT

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

if [[ -n "$SCENARIO_NAME" ]]; then
  JOB_SECTOR_NAME="Global_200/$(sanitize_token "$SCENARIO_NAME")"
else
  JOB_SECTOR_NAME="Global_200/$(sanitize_token "$MODEL")"
fi

SCENARIO_TAG_RAW="${SCENARIO_NAME:-$MODEL}"
SCENARIO_TAG="$(printf '%s' "$SCENARIO_TAG_RAW" | tr -cs '[:alnum:]_.-' '_')"
JOB_TAG="${SCENARIO_TAG}_${MODEL}"
mkdir -p "$JOB_ROOT"
JOB_DIR="$(cd "$JOB_ROOT" && pwd)/energymod_${JOB_TAG}"
SNAKEMAKE_JOBS="${JOBS:-${NSLOTS:-4}}"
CONDA_ENV_NAME="${LEARNING_CONDA_ENV:-/shared/share_cki25/envs/sh-pypsa-earth-main}"

mkdir -p "$JOB_DIR"
while IFS= read -r name; do
  case "$name" in
    .git|.snakemake|cluster_workdirs)
      continue
      ;;
  esac
  if [[ -L "$JOB_DIR/$name" ]]; then
    continue
  fi
  if [[ -e "$JOB_DIR/$name" ]]; then
    echo "Existing non-symlink path blocks staged control directory entry: $JOB_DIR/$name" >&2
    exit 1
  fi
  ln -s "$ROOT_DIR/$name" "$JOB_DIR/$name"
done < <(find "$ROOT_DIR" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort)

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
  learning_model:
    - "$MODEL"

permstore:
  path: permstorage

learning:
  enabled: true
  execution_mode: "$RUN_MODE"
  engine: "$LEARNING_ENGINE"
  selected_model: "$MODEL"
  monte_carlo:
    enable: false
EOF2

CMD=(
  snakemake
  "-j${SNAKEMAKE_JOBS}"
  "$TARGET_RULE"
  --configfile
  config.myopic.yaml
  config.learning.yaml
)

for cfg in "${CONFIGFILES[@]}"; do
  CMD+=("$cfg")
done

CMD+=(
  "$OVERLAY_FILE"
  --rerun-trigger
  mtime
)

if [[ "$DRY_RUN" -eq 1 ]]; then
  CMD+=(-n)
fi

echo "Running deterministic benchmark learning job:"
echo "  model=$MODEL"
echo "  engine=$LEARNING_ENGINE"
echo "  mode=$RUN_MODE"
echo "  target=$TARGET_RULE"
echo "  sector_name=$JOB_SECTOR_NAME"
echo "  jobs=$SNAKEMAKE_JOBS"
echo "  overlay=$OVERLAY_FILE"
if [[ "${#CONFIGFILES[@]}" -gt 0 ]]; then
  printf '  configfiles=%s\n' "${CONFIGFILES[*]}"
fi

cd "$JOB_DIR"
if command -v stdbuf >/dev/null 2>&1; then
  stdbuf -oL -eL "${CMD[@]}"
else
  "${CMD[@]}"
fi
