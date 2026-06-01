#!/usr/bin/env bash
# Queue the ROB-158 RFM smoke, then queue the full eval only if the smoke passes.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-158}"
AGGREGATE_DIR="${AGGREGATE_DIR:-${REPO_DIR}/exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048}"
LOG_PATH="${LOG_PATH:-${AGGREGATE_DIR}/logs/rob158_rfm_smoke_then_eval.log}"
SCREEN_NAME="${SCREEN_NAME:-rob158_rfm_smoke_then_eval}"
FULL_SCREEN_NAME="${FULL_SCREEN_NAME:-rob158_rfm_large_asr_eval}"
FULL_SCREEN_LOG="${FULL_SCREEN_LOG:-${AGGREGATE_DIR}/logs/rob158_rfm_large_asr_screen.log}"
FULL_LOG_PATH="${FULL_LOG_PATH:-${AGGREGATE_DIR}/logs/rob158_rfm_large_asr_eval.log}"
RESULT_ROOT="${RESULT_ROOT:-${AGGREGATE_DIR}/results}"
GPU_POOL="${ROB158_GPU_POOL:-1,2}"
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"
SMOKE_COMMAND="${SMOKE_COMMAND:-/store/store5/software/simple-gpu-schedule/with-gpu ${GPU_POOL} -- env ROB158_DISABLE_CALLBACK=1 ROB158_SMOKE=1 ROB158_DONT_SAVE=1 scripts/launch_rob158_rfm_large_asr_eval.sh}"
FULL_QUEUED_COMMAND="${FULL_QUEUED_COMMAND:-/store/store5/software/simple-gpu-schedule/with-gpu ${GPU_POOL} -- scripts/launch_rob158_rfm_large_asr_eval.sh}"
QUEUED_COMMAND="${QUEUED_COMMAND:-${SMOKE_COMMAND} && screen -dmS ${FULL_SCREEN_NAME} ... ${FULL_QUEUED_COMMAND}}"

launched_full=0

on_exit() {
  status=$?
  set +e
  if [ "${ROB158_DISPATCH_DISABLE_CALLBACK:-0}" = "1" ]; then
    exit "${status}"
  fi
  if [ -z "${LINEAR_API_KEY:-}" ] && [ "${ROB158_DISPATCH_CALLBACK_DRY_RUN:-0}" != "1" ]; then
    echo "LINEAR_API_KEY is not set; cannot post Linear dispatch callback" >&2
    exit "${status}"
  fi
  if [ "${status}" -eq 0 ] && [ "${launched_full}" = "1" ]; then
    target_state="${ROB158_DISPATCH_SUCCESS_STATE:-Backlog}"
    note="ROB-158 RFM large-ASR smoke passed and the full callback-backed eval was queued in screen '${FULL_SCREEN_NAME}'. Full eval callback should move the issue back to Todo when it exits. Full eval log: ${FULL_LOG_PATH}. Results: ${AGGREGATE_DIR}."
  else
    target_state="${ROB158_DISPATCH_FAILURE_STATE:-Todo}"
    note="ROB-158 RFM large-ASR smoke/dispatch wrapper exited before queueing the full eval. Inspect ${LOG_PATH}; the full eval was not launched."
  fi
  callback_args=(
    --issue "${LINEAR_ISSUE}"
    --status-code "${status}"
    --log "${LOG_PATH}"
    --results "${AGGREGATE_DIR}"
    --screen-name "${SCREEN_NAME}"
    --runner-label "screen:${SCREEN_NAME}"
    --queued-command "${QUEUED_COMMAND}"
    --branch "${GIT_BRANCH}"
    --commit "${GIT_COMMIT}"
    --target-state "${target_state}"
    --note "${note}"
    --tail-lines "${CALLBACK_TAIL_LINES:-80}"
    --max-log-chars "${CALLBACK_MAX_LOG_CHARS:-6000}"
    --max-comment-chars "${CALLBACK_MAX_COMMENT_CHARS:-10000}"
  )
  if [ "${ROB158_DISPATCH_CALLBACK_CHECK_ONLY:-0}" = "1" ]; then
    callback_args+=(--check-only)
  fi
  if [ "${ROB158_DISPATCH_CALLBACK_DRY_RUN:-0}" = "1" ]; then
    callback_args+=(--dry-run)
  fi
  python3 "${REPO_DIR}/scripts/callbacks/linear_experiment_callback.py" "${callback_args[@]}"
  callback_status=$?
  if [ "${callback_status}" -ne 0 ]; then
    echo "Linear dispatch callback failed with status ${callback_status}" >&2
  fi
  exit "${status}"
}
trap on_exit EXIT

set -euo pipefail

mkdir -p "$(dirname "${LOG_PATH}")" "${AGGREGATE_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "[rob158-dispatch] branch=${GIT_BRANCH}"
echo "[rob158-dispatch] commit=${GIT_COMMIT}"
echo "[rob158-dispatch] smoke_command=${SMOKE_COMMAND}"
echo "[rob158-dispatch] full_queued_command=${FULL_QUEUED_COMMAND}"
echo "[rob158-dispatch] full_screen=${FULL_SCREEN_NAME}"
echo "[rob158-dispatch] full_screen_log=${FULL_SCREEN_LOG}"
echo "[rob158-dispatch] full_log=${FULL_LOG_PATH}"
echo "[rob158-dispatch] results=${AGGREGATE_DIR}"

if [ "${ROB158_DISPATCH_CALLBACK_ONLY:-0}" = "1" ]; then
  echo "[rob158-dispatch] callback-only path requested; exiting before smoke."
  exit 0
fi

if screen -ls | grep -q "[.]${FULL_SCREEN_NAME}[[:space:]]"; then
  echo "Full eval screen '${FULL_SCREEN_NAME}' already exists; refusing duplicate launch." >&2
  exit 1
fi

echo "[rob158-dispatch] starting smoke"
eval "${SMOKE_COMMAND}"
echo "[rob158-dispatch] smoke passed"

if [ "${ROB158_DISPATCH_SMOKE_ONLY:-0}" = "1" ]; then
  echo "[rob158-dispatch] smoke-only path requested; exiting before full eval launch."
  exit 0
fi

echo "[rob158-dispatch] launching full eval screen"
screen -L -Logfile "${FULL_SCREEN_LOG}" -dmS "${FULL_SCREEN_NAME}" bash -lc \
  "cd '${REPO_DIR}' && ${FULL_QUEUED_COMMAND}"
launched_full=1
echo "[rob158-dispatch] launched full eval screen ${FULL_SCREEN_NAME}"
