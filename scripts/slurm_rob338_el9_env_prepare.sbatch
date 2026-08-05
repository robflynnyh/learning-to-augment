#!/usr/bin/env bash
# Build the ROB-338 Python environment on an EL9 H100 compute node.

#SBATCH --job-name=rob338-el9-env
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=gpu
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
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
ROB338_EL9_ENV_DIR="${ROB338_EL9_ENV_DIR:-/mnt/parscratch/users/acp21rjf/conda/rob338-el9}"
ROB338_EL9_PYTHON_MODULE="${ROB338_EL9_PYTHON_MODULE:-Python/3.10.8-GCCcore-12.2.0}"
LOG_PATH="${LOG_PATH:-${RESULT_ROOT}/logs/stanage/rob338/rob338_el9_env_prepare-${SLURM_JOB_ID:-manual}.log}"
RUNNER_LABEL="${RUNNER_LABEL:-slurm-env-prep:${SLURM_JOB_ID:-manual}}"
QUEUED_COMMAND="${QUEUED_COMMAND:-sbatch scripts/slurm_rob338_el9_env_prepare.sbatch}"

mkdir -p "$(dirname "${LOG_PATH}")"
exec > >(tee -a "${LOG_PATH}") 2>&1
cd "${REPO_DIR}"

GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"

on_exit() {
  status=$?
  set +e
  callback_python="${ROB338_CALLBACK_PYTHON:-/usr/bin/python3}"
  if [ -z "${LINEAR_API_KEY:-}" ]; then
    echo "LINEAR_API_KEY is not set; cannot post ROB-338 EL9 environment callback" >&2
    exit "${status}"
  fi
  callback_args=(
    --issue "${LINEAR_ISSUE}"
    --status-code "${status}"
    --log "${LOG_PATH}"
    --results "${ROB338_EL9_ENV_DIR}"
    --runner-label "${RUNNER_LABEL}"
    --queued-command "${QUEUED_COMMAND}"
    --branch "${GIT_BRANCH}"
    --commit "${GIT_COMMIT}"
    --target-state Todo
    --note "${CALLBACK_NOTE:-ROB-338 EL9 environment preparation finished. Inspect this log and the environment import validation before submitting the representative H100 evaluation smoke.}"
  )
  if [ "${ROB338_ENV_PREP_CALLBACK_CHECK_ONLY:-0}" = "1" ]; then
    callback_args+=(--check-only)
  fi
  if [ "${ROB338_ENV_PREP_CALLBACK_DRY_RUN:-0}" = "1" ]; then
    callback_args+=(--dry-run)
  fi
  "${callback_python}" scripts/callbacks/linear_experiment_callback.py "${callback_args[@]}"
  callback_status=$?
  if [ "${callback_status}" -ne 0 ]; then
    echo "ROB-338 EL9 environment callback failed with status ${callback_status}" >&2
    if [ "${status}" -eq 0 ]; then
      status="${callback_status}"
    fi
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
  echo "ROB-338 environment must be built on EL9; found VERSION_ID=${VERSION_ID:-unknown}" >&2
  exit 1
fi

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
elif [ -f /usr/share/lmod/lmod/init/bash ]; then
  source /usr/share/lmod/lmod/init/bash
fi
module purge
module load "${ROB338_EL9_PYTHON_MODULE}"

ROB338_EL9_BASE_PYTHON="$(command -v python3)"
export ROB338_EL9_BASE_PYTHON ROB338_EL9_ENV_DIR

echo "[rob338-el9-env-wrapper] job_id=${SLURM_JOB_ID:-manual}"
echo "[rob338-el9-env-wrapper] host=$(hostname)"
echo "[rob338-el9-env-wrapper] os=${PRETTY_NAME:-unknown}"
echo "[rob338-el9-env-wrapper] python_module=${ROB338_EL9_PYTHON_MODULE}"
echo "[rob338-el9-env-wrapper] base_python=${ROB338_EL9_BASE_PYTHON}"
echo "[rob338-el9-env-wrapper] environment=${ROB338_EL9_ENV_DIR}"
echo "[rob338-el9-env-wrapper] branch=${GIT_BRANCH}"
echo "[rob338-el9-env-wrapper] commit=${GIT_COMMIT}"

"${ROB338_EL9_BASE_PYTHON}" --version

if [ "${ROB338_ENV_PREP_SKIP_BUILD:-0}" = "1" ]; then
  echo "[rob338-el9-env-wrapper] build skipped for wrapper/callback smoke"
  exit 0
fi

bash scripts/prepare_rob338_el9_env.sh
echo "[rob338-el9-env-wrapper] preparation=passed"
