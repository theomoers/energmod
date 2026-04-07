#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/learning/run_learning_cluster_job.sh <model> <seed> <job_root> [--dry-run]

This creates a lightweight per-job control directory under <job_root> with
symlinks back to the shared checkout, then executes
scripts/learning/run_learning_stochastic_job.sh in branch mode there.
USAGE
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
SCENARIO_TAG_RAW="${LEARNING_SCENARIO_NAME:-}"
SCENARIO_TAG="$(printf '%s' "$SCENARIO_TAG_RAW" | tr -cs '[:alnum:]_.-' '_')"
JOB_TAG="${SCENARIO_TAG:+${SCENARIO_TAG}_}${MODEL}_s$(printf '%04d' "$SEED")"
mkdir -p "$JOB_ROOT"
JOB_DIR="$(cd "$JOB_ROOT" && pwd)/energymod_${JOB_TAG}"
RUN_MODE="${LEARNING_RUN_MODE:-branch}"

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

echo "Staged lightweight control directory at: $JOB_DIR"
if [[ -n "$SCENARIO_TAG_RAW" ]]; then
  echo "Scenario: $SCENARIO_TAG_RAW"
fi
echo "Run mode: $RUN_MODE"
cd "$JOB_DIR"
RUNNER="${LEARNING_CLUSTER_JOB_RUNNER:-bash scripts/learning/run_learning_stochastic_job.sh}"
TIMEOUT_BIN="${LEARNING_TIMEOUT_BIN:-/usr/bin/timeout}"
export LEARNING_TASK_START_EPOCH="${LEARNING_TASK_START_EPOCH:-$(date +%s)}"
export LEARNING_TASK_MAX_SECONDS="${LEARNING_TASK_MAX_SECONDS:-28800}"
export LEARNING_TASK_RESERVE_SECONDS="${LEARNING_TASK_RESERVE_SECONDS:-900}"
export LEARNING_RETRY_MIN_SECONDS="${LEARNING_RETRY_MIN_SECONDS:-3600}"
export LEARNING_TIMEOUT_GRACE_SECONDS="${LEARNING_TIMEOUT_GRACE_SECONDS:-300}"

echo "Task runtime budget: max=${LEARNING_TASK_MAX_SECONDS}s reserve=${LEARNING_TASK_RESERVE_SECONDS}s retry_min=${LEARNING_RETRY_MIN_SECONDS}s grace=${LEARNING_TIMEOUT_GRACE_SECONDS}s"

set +e
if [[ -x "$TIMEOUT_BIN" ]]; then
  # shellcheck disable=SC2086
  "$TIMEOUT_BIN" --signal=TERM --kill-after="${LEARNING_TIMEOUT_GRACE_SECONDS}s" "${LEARNING_TASK_MAX_SECONDS}s"     $RUNNER "$MODEL" "$SEED" --mode "$RUN_MODE" ${DRY_RUN_FLAG:+"$DRY_RUN_FLAG"}
  STATUS=$?
else
  # shellcheck disable=SC2086
  $RUNNER "$MODEL" "$SEED" --mode "$RUN_MODE" ${DRY_RUN_FLAG:+"$DRY_RUN_FLAG"}
  STATUS=$?
fi
set -e

if [[ "$STATUS" -eq 124 ]]; then
  echo "Task exceeded hard wallclock limit of ${LEARNING_TASK_MAX_SECONDS}s and was terminated by timeout" >&2
fi

if [[ "$STATUS" -eq 0 ]]; then
  cd "$JOB_ROOT"
  rm -rf "$JOB_DIR"
fi

exit "$STATUS"
