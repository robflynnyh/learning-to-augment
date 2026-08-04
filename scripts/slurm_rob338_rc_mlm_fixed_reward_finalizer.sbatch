#!/usr/bin/env bash
# Aggregate ROB-338 RC-MLM fixed-reward repeat cells and wake Symphony through Linear.

#SBATCH --job-name=rob338-final
#SBATCH --partition=sheffield
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
LOG_PATH="${LOG_PATH:-${RESULT_ROOT}/logs/stanage/rob338/rob338_stanage_finalizer-${SLURM_JOB_ID:-manual}.log}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/parscratch/users/acp21rjf/l2augment_model/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt}"
ROB338_FIXED_REWARDS="${ROB338_FIXED_REWARDS:-1.0 0.0}"
ROB338_DATASETS="${ROB338_DATASETS:-tedlium earnings22 rev16 TAL chime6}"
ROB338_EPOCHS="${ROB338_EPOCHS:-1 5}"
ROB338_REPEATS="${ROB338_REPEATS:-1 2 3}"
ROB338_LR="${ROB338_LR:-1e-5}"
ROB338_PROVENANCE_NOT_BEFORE="${ROB338_PROVENANCE_NOT_BEFORE:-2026-07-29T19:04:00Z}"
CSV_NAME="${CSV_NAME:-rob124_384_dropout_all_dataset_fixed_rewards_0_and_1.csv}"
RUNNER_LABEL="${RUNNER_LABEL:-slurm-finalizer:${SLURM_JOB_ID:-manual}}"
QUEUED_COMMAND="${QUEUED_COMMAND:-sbatch ${SLURM_SUBMIT_SCRIPT:-scripts/slurm_rob338_rc_mlm_fixed_reward_finalizer.sbatch}}"

mkdir -p "$(dirname "${LOG_PATH}")" "${RESULT_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

cd "${REPO_DIR}"
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"
summary_status=0

echo "[rob338-stanage-finalizer] job_id=${SLURM_JOB_ID:-manual}"
echo "[rob338-stanage-finalizer] host=$(hostname)"
echo "[rob338-stanage-finalizer] result_root=${RESULT_ROOT}"
echo "[rob338-stanage-finalizer] branch=${GIT_BRANCH}"
echo "[rob338-stanage-finalizer] commit=${GIT_COMMIT}"
echo "[rob338-stanage-finalizer] queued_command=${QUEUED_COMMAND}"
echo "[rob338-stanage-finalizer] repeats=${ROB338_REPEATS}"

if [ "${ROB338_FINALIZER_SKIP_SUMMARY:-0}" != "1" ]; then
  if command -v module >/dev/null 2>&1; then
    module load Anaconda3/2022.10
  fi
  source activate /mnt/parscratch/users/acp21rjf/conda/main
  python3 scripts/summarize_rob124_384_dropout_all_dataset_fixed_rewards.py \
    --result-root "${RESULT_ROOT}" \
    --fixed-rewards "${ROB338_FIXED_REWARDS}" \
    --datasets "${ROB338_DATASETS}" \
    --epochs "${ROB338_EPOCHS}" \
    --lr "${ROB338_LR}" \
    --repeats "${ROB338_REPEATS}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --command "${QUEUED_COMMAND}" \
    --branch "${GIT_BRANCH}" \
    --commit "${GIT_COMMIT}" \
    --log-path "${LOG_PATH}" \
    --screen-log-path "slurm" \
    --csv-name "${CSV_NAME}" \
    --title "ROB-338 RC-MLM Fixed-Reward WER Repeats" || summary_status=$?

  python3 scripts/validate_rob338_seed_fix_provenance.py \
    --csv "${RESULT_ROOT}/${CSV_NAME}" \
    --result-root "${RESULT_ROOT}" \
    --not-before "${ROB338_PROVENANCE_NOT_BEFORE}" || summary_status=$?
else
  echo "[rob338-stanage-finalizer] summary skipped for callback smoke"
fi

callback_status_code="${summary_status}"
if [ -z "${LINEAR_API_KEY:-}" ] && [ "${ROB338_FINALIZER_CALLBACK_DRY_RUN:-0}" != "1" ]; then
  echo "LINEAR_API_KEY is not set; cannot post Linear completion callback" >&2
  exit "${summary_status}"
fi

callback_args=(
  --issue "${LINEAR_ISSUE}"
  --status-code "${callback_status_code}"
  --log "${LOG_PATH}"
  --results "${RESULT_ROOT}"
  --runner-label "${RUNNER_LABEL}"
  --queued-command "${QUEUED_COMMAND}"
  --branch "${GIT_BRANCH}"
  --commit "${GIT_COMMIT}"
  --target-state "${CALLBACK_TARGET_STATE:-Todo}"
  --note "${CALLBACK_NOTE:-ROB-338 Stanage RC-MLM fixed-reward repeat finalizer completed. Inspect OUTCOME.md, the CSV, and per-cell logs under ${RESULT_ROOT}/logs/stanage/rob338 before final Linear/PR handoff.}"
  --tail-lines "${CALLBACK_TAIL_LINES:-80}"
  --max-log-chars "${CALLBACK_MAX_LOG_CHARS:-6000}"
  --max-comment-chars "${CALLBACK_MAX_COMMENT_CHARS:-10000}"
)
if [ "${ROB338_FINALIZER_CALLBACK_CHECK_ONLY:-0}" = "1" ]; then
  callback_args+=(--check-only)
fi
if [ "${ROB338_FINALIZER_CALLBACK_DRY_RUN:-0}" = "1" ]; then
  callback_args+=(--dry-run)
fi

if command -v module >/dev/null 2>&1; then
  module load Anaconda3/2022.10
fi
if [ -f /mnt/parscratch/users/acp21rjf/conda/main/bin/activate ]; then
  source activate /mnt/parscratch/users/acp21rjf/conda/main
fi
python3 scripts/callbacks/linear_experiment_callback.py "${callback_args[@]}"
callback_status=$?
if [ "${callback_status}" -ne 0 ]; then
  echo "Linear completion callback failed with status ${callback_status}" >&2
fi

exit "${summary_status}"
