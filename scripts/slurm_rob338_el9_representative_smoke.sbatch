#!/usr/bin/env bash
# Run one short, non-saving ROB-338 evaluation on EL9 before the recovery sweep.

#SBATCH --job-name=rob338-el9-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=gpu
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/%x-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/%x-%j.err

set -uo pipefail

LINEAR_ENV_FILE="${LINEAR_ENV_FILE:-${HOME}/.config/learning-to-augment/linear.env}"
if [ -f "${LINEAR_ENV_FILE}" ]; then
  set -a
  . "${LINEAR_ENV_FILE}"
  set +a
fi

REPO_DIR="${REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-learning-to-augment/ROB-338}"
LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-338}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1}"
ROB338_PYTHON_BIN="${ROB338_PYTHON_BIN:-/mnt/parscratch/users/acp21rjf/conda/rob338-el9/bin/python}"
LOG_PATH="${LOG_PATH:-${RESULT_ROOT}/logs/stanage/rob338/rob338_el9_representative_smoke-${SLURM_JOB_ID:-manual}.log}"
RUNNER_LABEL="${RUNNER_LABEL:-slurm-smoke:${SLURM_JOB_ID:-manual}}"
QUEUED_COMMAND="${QUEUED_COMMAND:-sbatch scripts/slurm_rob338_el9_representative_smoke.sbatch}"

mkdir -p "$(dirname "${LOG_PATH}")"
exec > >(tee -a "${LOG_PATH}") 2>&1
cd "${REPO_DIR}"

GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"

on_exit() {
  status=$?
  set +e
  callback_python="${ROB338_CALLBACK_PYTHON:-/usr/bin/python3}"
  if [ ! -x "${callback_python}" ]; then
    callback_python="${ROB338_PYTHON_BIN}"
  fi
  if [ -z "${LINEAR_API_KEY:-}" ]; then
    echo "LINEAR_API_KEY is not set; cannot post ROB-338 EL9 smoke callback" >&2
    exit "${status}"
  fi
  "${callback_python}" scripts/callbacks/linear_experiment_callback.py \
    --issue "${LINEAR_ISSUE}" \
    --status-code "${status}" \
    --log "${LOG_PATH}" \
    --results "${ROB338_PYTHON_BIN}" \
    --runner-label "${RUNNER_LABEL}" \
    --queued-command "${QUEUED_COMMAND}" \
    --branch "${GIT_BRANCH}" \
    --commit "${GIT_COMMIT}" \
    --target-state Todo \
    --note "ROB-338 EL9 representative H100 smoke finished. Inspect this log before submitting the 34-cell recovery sweep. This smoke uses TED-LIUM test index 0, reward 1.0, epoch 1, repeat 2, max_steps 1, and does not save a result row."
  callback_status=$?
  if [ "${callback_status}" -ne 0 ]; then
    echo "ROB-338 EL9 smoke callback failed with status ${callback_status}" >&2
  fi
  exit "${status}"
}
trap on_exit EXIT

set -euo pipefail

if [ ! -r /etc/os-release ]; then
  echo "Cannot verify the compute-node operating system" >&2
  exit 1
fi
# shellcheck disable=SC1091
. /etc/os-release
if [[ "${VERSION_ID:-}" != 9* ]]; then
  echo "Representative smoke requires EL9; found VERSION_ID=${VERSION_ID:-unknown}" >&2
  exit 1
fi

export ROB338_DATASET="${ROB338_DATASET:-tedlium}"
export ROB338_REWARD="${ROB338_REWARD:-1.0}"
export ROB338_EPOCH="${ROB338_EPOCH:-1}"
export ROB338_REPEAT="${ROB338_REPEAT:-2}"
export ROB338_GPU_TYPE="${ROB338_GPU_TYPE:-h100}"
export ROB338_INDEXES="${ROB338_INDEXES:-0}"
export ROB338_MAX_STEPS="${ROB338_MAX_STEPS:-1}"
export ROB338_DONT_SAVE=1
export FORCE_RERUN=0
export LOG_PATH

echo "[rob338-el9-smoke] os=${PRETTY_NAME:-unknown}"
echo "[rob338-el9-smoke] branch=${GIT_BRANCH}"
echo "[rob338-el9-smoke] commit=${GIT_COMMIT}"
echo "[rob338-el9-smoke] python=${ROB338_PYTHON_BIN}"
echo "[rob338-el9-smoke] job_id=${SLURM_JOB_ID:-manual}"

bash scripts/slurm_rob338_rc_mlm_fixed_reward_cell.sbatch
echo "[rob338-el9-smoke] representative_eval=passed"
