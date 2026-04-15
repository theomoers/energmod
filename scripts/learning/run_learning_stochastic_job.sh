#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/learning/run_learning_stochastic_job.sh <model> <seed> [--mode full|branch] [--scenario-name NAME] [--dry-run]

Supported models:
  - shared_state_bayesian_regime_wright
  - way_fixed_rho_benchmark_035
  - correlated_geometric_random_walk

Environment variables:
  JOBS                 Snakemake parallelism override
  NSLOTS               Cluster slot count fallback when JOBS is unset
  LEARNING_SECTOR_NAME Override the shared sector_name (default: Global_200)

Modes:
  full    Run the full seeded learning chain
  branch  Run only the stochastic branch after the shared deterministic bootstrap
USAGE
}

if [[ $# -lt 2 ]]; then
  usage >&2
  exit 1
fi

MODEL="$1"
SEED="$2"
shift 2

RUN_MODE="${LEARNING_RUN_MODE:-full}"
DRY_RUN=0
SCENARIO_NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      if [[ $# -lt 2 ]]; then
        echo "--mode requires an argument" >&2
        exit 1
      fi
      RUN_MODE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --scenario-name)
      if [[ $# -lt 2 ]]; then
        echo "--scenario-name requires an argument" >&2
        exit 1
      fi
      SCENARIO_NAME="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "$MODEL" in
  shared_state_bayesian_regime_wright|way_fixed_rho_benchmark_035|correlated_geometric_random_walk)
    ;;
  *)
    echo "Unsupported stochastic learning model: $MODEL" >&2
    usage >&2
    exit 1
    ;;
esac

if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
  echo "Seed must be a non-negative integer, got: $SEED" >&2
  exit 1
fi

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
OVERLAY_FILE="$(mktemp "$TMPDIR_ROOT/energymod_learning_job.XXXXXX.yaml")"
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

if [[ -n "${LEARNING_SECTOR_NAME:-}" ]]; then
  JOB_SECTOR_NAME="${LEARNING_SECTOR_NAME}"
elif [[ -n "$SCENARIO_NAME" ]]; then
  JOB_SECTOR_NAME="Global_200/$(sanitize_token "$SCENARIO_NAME")"
elif [[ -n "${LEARNING_SCENARIO_NAME:-}" ]]; then
  JOB_SECTOR_NAME="Global_200/$(sanitize_token "$LEARNING_SCENARIO_NAME")"
else
  JOB_SECTOR_NAME="Global_200"
fi
SNAKEMAKE_JOBS="${JOBS:-${NSLOTS:-4}}"
CONDA_ENV_NAME="${LEARNING_CONDA_ENV:-/shared/share_cki25/envs/sh-pypsa-earth-main}"
COST_EXPECTATION_MODE="${LEARNING_COST_EXPECTATION_MODE:-}"
COST_EXPECTATION_WEIGHTS="${LEARNING_COST_EXPECTATION_WEIGHTS:-}"

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
  engine: stochastic_forecast
  selected_model: "$MODEL"
  seed: $SEED
  monte_carlo:
    enable: false
EOF2

if [[ -n "$COST_EXPECTATION_MODE" || -n "$COST_EXPECTATION_WEIGHTS" ]]; then
  {
    echo "  cost_expectations:"
    echo "    mode: \"${COST_EXPECTATION_MODE:-block_average_expected}\""
    if [[ -n "$COST_EXPECTATION_WEIGHTS" ]]; then
      echo "    annual_weights: ${COST_EXPECTATION_WEIGHTS}"
    fi
  } >>"$OVERLAY_FILE"
fi

CMD=(
  snakemake
  "-j${SNAKEMAKE_JOBS}"
  "$TARGET_RULE"
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

echo "Running stochastic learning job:"
echo "  model=$MODEL"
echo "  seed=$SEED"
echo "  mode=$RUN_MODE"
echo "  target=$TARGET_RULE"
echo "  sector_name=$JOB_SECTOR_NAME"
echo "  jobs=$SNAKEMAKE_JOBS"
echo "  overlay=$OVERLAY_FILE"

cd "$ROOT_DIR"
if command -v stdbuf >/dev/null 2>&1; then
  stdbuf -oL -eL "${CMD[@]}"
else
  "${CMD[@]}"
fi
