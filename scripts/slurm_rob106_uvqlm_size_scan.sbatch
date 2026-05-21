#!/usr/bin/env bash
# ROB-106 CPU-only Stanage size scan for the UVQLM rollout folder.

#SBATCH --job-name=rob106-uvqlm-size
#SBATCH --partition=sheffield
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/%x-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/%x-%j.err

set -uo pipefail

LINEAR_ENV_FILE="${LINEAR_ENV_FILE:-${HOME}/.config/learning-to-augment/linear.env}"
if [ -f "${LINEAR_ENV_FILE}" ]; then
  set -a
  . "${LINEAR_ENV_FILE}"
  set +a
fi

REPO_DIR="${REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-learning-to-augment/ROB-106}"
LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-106}"
ROLLOUT_ROOT="${ROLLOUT_ROOT:-/mnt/parscratch/users/acp21rjf/l2augment_rollout_uvqmlm}"
RESULTS_PATH="${RESULTS_PATH:-/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-106/uvqlm_rollout_size}"
LOG_PATH="${LOG_PATH:-${RESULTS_PATH}/rob106_uvqlm_size_scan_${SLURM_JOB_ID:-manual}.log}"
RUNNER_LABEL="${RUNNER_LABEL:-slurm:${SLURM_JOB_ID:-unknown}}"
QUEUED_COMMAND="${QUEUED_COMMAND:-sbatch scripts/slurm_rob106_uvqlm_size_scan.sbatch}"

cd "${REPO_DIR}"

GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"

on_exit() {
  status=$?
  set +e
  if [ "${ROB106_SIZE_DISABLE_CALLBACK:-0}" = "1" ]; then
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
    --results "${RESULTS_PATH}"
    --runner-label "${RUNNER_LABEL}"
    --queued-command "${QUEUED_COMMAND}"
    --branch "${GIT_BRANCH}"
    --commit "${GIT_COMMIT}"
    --target-state "${CALLBACK_TARGET_STATE:-Todo}"
    --note "${CALLBACK_NOTE:-ROB-106 CPU-node exact size scan for /mnt/parscratch/users/acp21rjf/l2augment_rollout_uvqmlm exited. Primary output: ${RESULTS_PATH}/OUTCOME.md}"
    --tail-lines "${CALLBACK_TAIL_LINES:-80}"
    --max-log-chars "${CALLBACK_MAX_LOG_CHARS:-6000}"
    --max-comment-chars "${CALLBACK_MAX_COMMENT_CHARS:-10000}"
  )
  if [ "${ROB106_SIZE_CALLBACK_CHECK_ONLY:-0}" = "1" ]; then
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

mkdir -p "$(dirname "${LOG_PATH}")" "${RESULTS_PATH}"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "[rob106-size] host=$(hostname)"
echo "[rob106-size] branch=${GIT_BRANCH}"
echo "[rob106-size] commit=${GIT_COMMIT}"
echo "[rob106-size] rollout_root=${ROLLOUT_ROOT}"
echo "[rob106-size] results=${RESULTS_PATH}"

if [ "${ROB106_SIZE_CALLBACK_ONLY:-0}" = "1" ]; then
  echo "[rob106-size] callback-only smoke path requested; exiting before filesystem scan."
  exit 0
fi

python3 scripts/measure_rollout_tree_size.py \
  --root "${ROLLOUT_ROOT}" \
  --output-dir "${RESULTS_PATH}" \
  --progress-every "${ROB106_SIZE_PROGRESS_EVERY:-50000}"
