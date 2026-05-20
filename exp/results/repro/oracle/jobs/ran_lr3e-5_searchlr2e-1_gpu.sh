#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/../../../../.."

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-60}"
SCREEN_NAME="${SCREEN_NAME:-l2a_oracle_ran_lr3e5_searchlr2e1}"
RUNNER_LABEL="${RUNNER_LABEL:-screen:${SCREEN_NAME}}"
LOG_PATH="${LOG_PATH:-exp/results/repro/oracle/logs/ran_lr3e-5_searchlr2e-1_gpu.log}"
RESULTS_PATH="${RESULTS_PATH:-exp/results/repro/oracle/RAN}"
QUEUED_COMMAND="${QUEUED_COMMAND:-screen -L -Logfile exp/results/repro/oracle/logs/ran_lr3e-5_searchlr2e-1_queue.log -dmS l2a_oracle_ran_lr3e5_searchlr2e1 bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-60 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash exp/results/repro/oracle/jobs/ran_lr3e-5_searchlr2e-1_gpu.sh'}"
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
    --note "ROB-60 random additive-noise oracle sweep lr=3e-5/search_lr=2e-1 finished. Inspect RAN text file and update OUTCOME/plot before final handoff." \
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
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p "${MPLCONFIGDIR}" "$(dirname "${LOG_PATH}")" exp/results/repro/oracle/RAN

exec > >(tee -a "${LOG_PATH}") 2>&1

echo "[$(date -Iseconds)] ROB-60 random additive-noise oracle sweep lr=3e-5 search_lr=2e-1"
echo "branch=${GIT_BRANCH} commit=${GIT_COMMIT} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

if [ "${L2A_CALLBACK_SMOKE_TEST:-0}" = "1" ]; then
  echo "callback smoke test requested; exiting before experiment"
  exit 0
fi

result="exp/results/repro/oracle/RAN/tedlium_lr3e-5_searchlr2e-1.txt"
config_dir="exp/results/repro/oracle/RAN/.generated/tedlium_grid"
: > "${result}"
PYTHONDONTWRITEBYTECODE=1 python3 exp/run_config_grid.py \
  --grid-config "exp/results/repro/oracle/RAN/tedlium_grid.yaml" \
  --materialize-only

for repeats in 1 2 3 4 5 10 20 50; do
  echo "[$(date -Iseconds)] running RAN repeats=${repeats}"
  PYTHONDONTWRITEBYTECODE=1 python3 exp/oracle_eval.py \
    --config "${config_dir}/tedlium_lr3e-5_searchlr2e-1_repeats${repeats}.yaml"
done

echo "[$(date -Iseconds)] completed ROB-60 random additive-noise oracle sweep lr=3e-5 search_lr=2e-1"
