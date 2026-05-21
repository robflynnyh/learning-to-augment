#!/usr/bin/env bash
# Callback-backed ROB-106 sync of the Stanage UVQLM rollout folder to Mimas store4.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-106}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/exp/results/repro/sweeps/no_audio_cmultistep_vqlm/rob106_low_reward_conditioning/uvqlm_rollout_store4_sync}"
LOG_PATH="${LOG_PATH:-${RESULT_ROOT}/rob106_uvqlm_store4_rsync.log}"
SCREEN_LOG_PATH="${SCREEN_LOG_PATH:-${RESULT_ROOT}/rob106_uvqlm_store4_rsync.screen.log}"
SCREEN_NAME="${SCREEN_NAME:-rob106-uvqlm-store4-sync}"
RUNNER_LABEL="${RUNNER_LABEL:-screen:${SCREEN_NAME}}"
SOURCE_ROOT="${SOURCE_ROOT:-stanage.shef.ac.uk:/mnt/parscratch/users/acp21rjf/l2augment_rollout_uvqmlm/}"
DEST_ROOT="${DEST_ROOT:-/store/store4/data/l2augment_rollout_uvqmlm/}"
QUEUED_COMMAND="${QUEUED_COMMAND:-screen -L -Logfile ${SCREEN_LOG_PATH} -dmS ${SCREEN_NAME} bash -lc 'cd ${REPO_DIR} && ./scripts/launch_rob106_uvqlm_store4_sync.sh'}"
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
    --note "${CALLBACK_NOTE:-ROB-106 UVQLM rollout rsync to /store/store4 exited. Expected destination: ${DEST_ROOT}. Rerun the same wrapper to resume if interrupted.}"
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

export LINEAR_ISSUE RESULT_ROOT LOG_PATH SCREEN_LOG_PATH SCREEN_NAME RUNNER_LABEL
export SOURCE_ROOT DEST_ROOT QUEUED_COMMAND GIT_BRANCH GIT_COMMIT

echo "[rob106-store4-sync] branch=${GIT_BRANCH}"
echo "[rob106-store4-sync] commit=${GIT_COMMIT}"
echo "[rob106-store4-sync] source=${SOURCE_ROOT}"
echo "[rob106-store4-sync] destination=${DEST_ROOT}"
echo "[rob106-store4-sync] result_root=${RESULT_ROOT}"
echo "[rob106-store4-sync] log_path=${LOG_PATH}"
echo "[rob106-store4-sync] host=$(hostname)"

df -h /store/store4
df -B1 /store/store4
df -ih /store/store4

if [ "${ROB106_CALLBACK_ONLY:-0}" = "1" ]; then
  echo "[rob106-store4-sync] callback-only smoke path requested; exiting before mkdir/rsync."
  exit 0
fi

mkdir -p "${DEST_ROOT}"

rsync -ah --info=progress2 --partial --append-verify --mkpath \
  "${SOURCE_ROOT}" \
  "${DEST_ROOT}"

echo "[rob106-store4-sync] rsync complete"
echo "[rob106-store4-sync] destination summary"
find "${DEST_ROOT}" -maxdepth 2 -type f -name '*.pt' | wc -l
du -sh "${DEST_ROOT}"
