#!/usr/bin/env bash
# Queue ROB-80 repeat-2 no-audio CMultiStepVQLM fixed/random reward sweep.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-80}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/exp/results/repro/sweeps/no_audio_cmultistep_vqlm}"
LOG_PATH="${LOG_PATH:-${RESULT_ROOT}/logs/rob80_tedlium_noaudio_repeat2_sweep.log}"
SCREEN_NAME="${SCREEN_NAME:-rob80_tedlium_noaudio_repeat2_sweep}"
RUNNER_LABEL="${RUNNER_LABEL:-screen:${SCREEN_NAME}}"
QUEUED_COMMAND="${QUEUED_COMMAND:-/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob80_tedlium_noaudio_repeat2_sweep.sh}"
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"

on_exit() {
  status=$?
  set +e
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
    --note "${CALLBACK_NOTE:-ROB-80 TED-LIUM dev no-audio CMultiStepVQLM repeat-2 fixed/random reward sweep exited. See exp/results/repro/sweeps/no_audio_cmultistep_vqlm/ROB-80_NOAUDIO_REWARD_CONDITIONING_REPEAT_COMPARISON.md for the averaged table when complete.}"
    --tail-lines "${CALLBACK_TAIL_LINES:-80}"
    --max-log-chars "${CALLBACK_MAX_LOG_CHARS:-6000}"
    --max-comment-chars "${CALLBACK_MAX_COMMENT_CHARS:-10000}"
  )
  if [ "${ROB80_CALLBACK_CHECK_ONLY:-0}" = "1" ]; then
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

echo "[rob80-noaudio-repeat2] branch=${GIT_BRANCH}"
echo "[rob80-noaudio-repeat2] commit=${GIT_COMMIT}"
echo "[rob80-noaudio-repeat2] result_root=${RESULT_ROOT}"
echo "[rob80-noaudio-repeat2] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

if [ "${ROB80_CALLBACK_ONLY:-0}" = "1" ]; then
  echo "[rob80-noaudio-repeat2] callback-only smoke path requested; exiting before eval."
  exit 0
fi

COMMON_ENV=(
  ROB80_DISABLE_CALLBACK=1
  RESULT_ROOT="${RESULT_ROOT}"
  ROB80_REPEATS="1 2"
  ROB80_RUN_REPEATS="${ROB80_RUN_REPEATS:-2}"
  ROB80_SUMMARY_METHODS="CMultiStepVQLM CMultiStepVQLMRandomReward"
  ROB80_CSV_NAME="rob80_tedlium_noaudio_reward_conditioning_repeat_comparison.csv"
  ROB80_OUTCOME_NAME="ROB-80_NOAUDIO_REWARD_CONDITIONING_REPEAT_COMPARISON.md"
  ROB80_TITLE="ROB-80 TED-LIUM Dev No-Audio CMultiStepVQLM Reward Conditioning Repeat Comparison"
  ROB80_NOTE="Compares fixed reward 1.0 against uniform [0.5, 1.0] random reward conditioning across repeat 1 and repeat 2. Repeat 2 uses rollout seed 123457."
)

env "${COMMON_ENV[@]}" \
  ROB80_METHOD="CMultiStepVQLM" \
  ROB80_CONDITIONING_REWARD_RANGE="" \
  scripts/launch_rob80_tedlium_noaudio_cmultistep_sweep.sh

env "${COMMON_ENV[@]}" \
  ROB80_METHOD="CMultiStepVQLMRandomReward" \
  ROB80_CONDITIONING_REWARD_RANGE="0.5 1.0" \
  scripts/launch_rob80_tedlium_noaudio_cmultistep_sweep.sh

echo "[rob80-noaudio-repeat2] finished"
