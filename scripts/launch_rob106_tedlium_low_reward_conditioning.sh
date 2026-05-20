#!/usr/bin/env bash
# Queue-safe ROB-106 TED-LIUM dev low-reward CMultiStepVQLM comparison.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-106}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/exp/results/repro/sweeps/no_audio_cmultistep_vqlm/rob106_low_reward_conditioning}"
LOG_PATH="${LOG_PATH:-${RESULT_ROOT}/logs/rob106_tedlium_low_reward_conditioning.log}"
SCREEN_NAME="${SCREEN_NAME:-rob106_tedlium_low_reward_conditioning}"
RUNNER_LABEL="${RUNNER_LABEL:-screen:${SCREEN_NAME}}"
QUEUED_COMMAND="${QUEUED_COMMAND:-/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob106_tedlium_low_reward_conditioning.sh}"
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"

on_exit() {
  status=$?
  set +e
  if [ "${ROB106_DISABLE_CALLBACK:-0}" = "1" ]; then
    exit "${status}"
  fi
  if [ -z "${LINEAR_API_KEY:-}" ]; then
    echo "LINEAR_API_KEY is not set; cannot post Linear completion callback" >&2
    exit "${status}"
  fi
  callback_args=(
    --issue "${LINEAR_ISSUE}"
    --status-code "${status}"
    --log "${LOG_PATH}"
    --results "${RESULT_ROOT}"
    --screen-name "${SCREEN_NAME}"
    --runner-label "${RUNNER_LABEL}"
    --queued-command "${QUEUED_COMMAND}"
    --branch "${GIT_BRANCH}"
    --commit "${GIT_COMMIT}"
    --target-state "${CALLBACK_TARGET_STATE:-Todo}"
    --note "${CALLBACK_NOTE:-ROB-106 TED-LIUM dev low-reward CMultiStepVQLM comparison wrapper exited. See exp/results/repro/sweeps/no_audio_cmultistep_vqlm/rob106_low_reward_conditioning/OUTCOME.md when complete.}"
    --tail-lines "${CALLBACK_TAIL_LINES:-80}"
    --max-log-chars "${CALLBACK_MAX_LOG_CHARS:-6000}"
    --max-comment-chars "${CALLBACK_MAX_COMMENT_CHARS:-10000}"
  )
  if [ "${ROB106_CALLBACK_CHECK_ONLY:-0}" = "1" ]; then
    callback_args+=(--check-only)
  fi
  python3 "${REPO_DIR}/scripts/callbacks/linear_experiment_callback.py" "${callback_args[@]}"
  callback_status=$?
  if [ "${callback_status}" -ne 0 ]; then
    echo "Linear completion callback failed with status ${callback_status}" >&2
  fi
  exit "${status}"
}
trap on_exit EXIT

set -euo pipefail

mkdir -p "$(dirname "${LOG_PATH}")" "${RESULT_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

export LINEAR_ISSUE RESULT_ROOT LOG_PATH SCREEN_NAME RUNNER_LABEL QUEUED_COMMAND GIT_BRANCH GIT_COMMIT
export ROB80_DISABLE_CALLBACK=1
export ROB80_DATASET="${ROB80_DATASET:-tedlium}"
export ROB80_SPLIT="${ROB80_SPLIT:-dev}"
export ROB80_TAG_PREFIX="${ROB80_TAG_PREFIX:-tedlium_dev}"
export ROB80_EPOCHS="${ROB80_EPOCHS:-5}"
export ROB80_LRS="${ROB80_LRS:-5e-6}"
export ROB80_REPEATS="${ROB80_REPEATS:-1 2}"
export ROB80_SUMMARY_METHODS="${ROB80_SUMMARY_METHODS:-CMultiStepVQLMReward1 CMultiStepVQLMReward0 CMultiStepVQLMRandomReward0to1}"
export ROB80_CSV_NAME="${ROB80_CSV_NAME:-rob106_tedlium_noaudio_low_reward_conditioning.csv}"
export ROB80_OUTCOME_NAME="${ROB80_OUTCOME_NAME:-OUTCOME.md}"
export ROB80_TITLE="${ROB80_TITLE:-ROB-106 TED-LIUM Dev No-Audio CMultiStepVQLM Low-Reward Conditioning}"
export ROB80_NOTE="${ROB80_NOTE:-Compares the ROB-80 best comparable no-audio CMultiStepVQLM setting, TED-LIUM dev with 5 adaptation epochs and lr=5e-6, under fixed reward 1.0, fixed reward 0.0, and random reward sampled uniformly from [0.0, 1.0]. Rewards are in the trained MultiStepDataset min-max normalized range.}"

echo "[rob106] branch=${GIT_BRANCH}"
echo "[rob106] commit=${GIT_COMMIT}"
echo "[rob106] result_root=${RESULT_ROOT}"
echo "[rob106] dataset=${ROB80_DATASET}"
echo "[rob106] split=${ROB80_SPLIT}"
echo "[rob106] epochs=${ROB80_EPOCHS}"
echo "[rob106] lrs=${ROB80_LRS}"
echo "[rob106] repeats=${ROB80_REPEATS}"
echo "[rob106] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

if [ "${ROB106_CALLBACK_ONLY:-0}" = "1" ]; then
  echo "[rob106] callback-only smoke path requested; exiting before config generation."
  exit 0
fi

run_condition() {
  local method="$1"
  local default_reward="$2"
  local reward_range="${3:-}"

  export ROB80_METHOD="${method}"
  export ROB80_DEFAULT_CONDITIONING_REWARD="${default_reward}"
  if [ -n "${reward_range}" ]; then
    export ROB80_CONDITIONING_REWARD_RANGE="${reward_range}"
  else
    unset ROB80_CONDITIONING_REWARD_RANGE
  fi

  echo "[rob106] running method=${ROB80_METHOD} default_reward=${ROB80_DEFAULT_CONDITIONING_REWARD} range=${reward_range:-fixed}"
  scripts/launch_rob80_tedlium_noaudio_cmultistep_sweep.sh
}

run_condition CMultiStepVQLMReward1 1.0
run_condition CMultiStepVQLMReward0 0.0
run_condition CMultiStepVQLMRandomReward0to1 1.0 "0.0 1.0"

echo "[rob106] finished"
