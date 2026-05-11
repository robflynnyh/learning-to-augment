#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/../../../../.."

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-60}"
SCREEN_NAME="${SCREEN_NAME:-l2a_oracle_lr1e5_searchlr9e2}"
RUNNER_LABEL="${RUNNER_LABEL:-screen:${SCREEN_NAME}}"
LOG_PATH="${LOG_PATH:-exp/results/repro/oracle/logs/lr1e-5_searchlr9e-2_gpu.log}"
RESULTS_PATH="${RESULTS_PATH:-exp/results/repro/oracle}"
QUEUED_COMMAND="${QUEUED_COMMAND:-screen -L -Logfile exp/results/repro/oracle/logs/lr1e-5_searchlr9e-2_queue.log -dmS l2a_oracle_lr1e5_searchlr9e2 bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-60 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash exp/results/repro/oracle/jobs/lr1e-5_searchlr9e-2_gpu.sh'}"
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"

callback_args=()
if [ "${L2A_CALLBACK_DRY_RUN:-0}" = "1" ]; then
  callback_args+=(--dry-run)
fi

on_exit() {
  status=$?
  set +e
  if [ -z "${LINEAR_API_KEY:-}" ] && [ "${L2A_CALLBACK_DRY_RUN:-0}" != "1" ]; then
    echo "LINEAR_API_KEY is not set; cannot post Linear completion callback" >&2
    exit "${status}"
  fi
  python3 scripts/callbacks/linear_experiment_callback.py \
    --issue "${LINEAR_ISSUE}" \
    --status-code "${status}" \
    --log "${LOG_PATH}" \
    --results "${RESULTS_PATH}" \
    --screen-name "${SCREEN_NAME}" \
    --runner-label "${RUNNER_LABEL}" \
    --queued-command "${QUEUED_COMMAND}" \
    --branch "${GIT_BRANCH}" \
    --commit "${GIT_COMMIT}" \
    --note "ROB-60 lr=1e-5/search_lr=9e-2 oracle sweep finished. Inspect RMM/RFM text files and update OUTCOME/plot before final handoff." \
    "${callback_args[@]}"
  callback_status=$?
  if [ "${callback_status}" -ne 0 ]; then
    echo "Linear completion callback failed with status ${callback_status}" >&2
  fi
  exit "${status}"
}
trap on_exit EXIT

set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/exp/exp4/acp21rjf/.scratch/matplotlib-cache}"
export L2A_TEDLIUM3_LEGACY_DIR="${L2A_TEDLIUM3_LEGACY_DIR:-/store/store4/data/TEDLIUM_release-3/legacy}"
mkdir -p "${MPLCONFIGDIR}" "$(dirname "${LOG_PATH}")" \
  exp/results/repro/oracle/RMM exp/results/repro/oracle/RFM

exec > >(tee -a "${LOG_PATH}") 2>&1

echo "[$(date -Iseconds)] ROB-60 oracle sweep lr=1e-5 search_lr=9e-2"
echo "branch=${GIT_BRANCH} commit=${GIT_COMMIT} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

if [ "${L2A_CALLBACK_SMOKE_TEST:-0}" = "1" ]; then
  echo "callback smoke test requested; exiting before experiment"
  exit 0
fi

run_combo() {
  local method="$1"
  local result="exp/results/repro/oracle/${method}/tedlium_lr1e-5_searchlr9e-2.txt"
  : > "${result}"
  for repeats in 1 2 3 4 5 10 20 50; do
    echo "[$(date -Iseconds)] running ${method} repeats=${repeats}"
    PYTHONDONTWRITEBYTECODE=1 python3 exp/oracle_eval.py \
      --config "exp/results/repro/oracle/${method}/configs/tedlium_lr1e-5_searchlr9e-2_repeats${repeats}.yaml"
  done
}

run_combo RMM
run_combo RFM

echo "[$(date -Iseconds)] completed ROB-60 oracle sweep lr=1e-5 search_lr=9e-2"
