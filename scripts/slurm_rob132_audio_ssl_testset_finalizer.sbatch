#!/usr/bin/env bash
# Aggregate ROB-132 Stanage test-set cells and wake Symphony through Linear.

#SBATCH --job-name=rob132-final
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

REPO_DIR="${REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-learning-to-augment/ROB-132}"
LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-132}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384/eval/test_fixed_rewards_0_and_1}"
LOG_PATH="${LOG_PATH:-${RESULT_ROOT}/logs/stanage/rob132_stanage_finalizer-${SLURM_JOB_ID:-manual}.log}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/parscratch/users/acp21rjf/l2augment_model/reward_conditioned_mask_lm/audio_ssl_hubert_base_tedlium_per_utterance_transformer384_dropout0p1_500ep_lr1e3.pt}"
ROB132_FIXED_REWARDS="${ROB132_FIXED_REWARDS:-1.0 0.0}"
ROB132_DATASETS="${ROB132_DATASETS:-tedlium earnings22 rev16 TAL chime6}"
ROB132_EPOCHS="${ROB132_EPOCHS:-1 5}"
ROB132_LR="${ROB132_LR:-1e-5}"
CSV_NAME="${CSV_NAME:-rob132_audio_ssl_test_fixed_rewards.csv}"
RUNNER_LABEL="${RUNNER_LABEL:-slurm-finalizer:${SLURM_JOB_ID:-manual}}"
QUEUED_COMMAND="${QUEUED_COMMAND:-sbatch ${SLURM_SUBMIT_SCRIPT:-scripts/slurm_rob132_audio_ssl_testset_finalizer.sbatch}}"

mkdir -p "$(dirname "${LOG_PATH}")" "${RESULT_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

cd "${REPO_DIR}"
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"
summary_status=0

echo "[rob132-stanage-finalizer] job_id=${SLURM_JOB_ID:-manual}"
echo "[rob132-stanage-finalizer] host=$(hostname)"
echo "[rob132-stanage-finalizer] result_root=${RESULT_ROOT}"
echo "[rob132-stanage-finalizer] branch=${GIT_BRANCH}"
echo "[rob132-stanage-finalizer] commit=${GIT_COMMIT}"
echo "[rob132-stanage-finalizer] queued_command=${QUEUED_COMMAND}"

if [ "${ROB132_FINALIZER_SKIP_SUMMARY:-0}" != "1" ]; then
  if command -v module >/dev/null 2>&1; then
    module load Anaconda3/2022.10
  fi
  source activate /mnt/parscratch/users/acp21rjf/conda/main
  python3 scripts/summarize_rob132_audio_ssl_self_train_test_sets.py \
    --result-root "${RESULT_ROOT}" \
    --fixed-rewards "${ROB132_FIXED_REWARDS}" \
    --datasets "${ROB132_DATASETS}" \
    --epochs "${ROB132_EPOCHS}" \
    --lr "${ROB132_LR}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --command "${QUEUED_COMMAND}" \
    --branch "${GIT_BRANCH}" \
    --commit "${GIT_COMMIT}" \
    --log-path "${LOG_PATH}" \
    --screen-log-path "slurm" \
    --csv-name "${CSV_NAME}" || summary_status=$?

  python3 - "${RESULT_ROOT}/${CSV_NAME}" <<'PY' || summary_status=$?
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
rows = list(csv.DictReader(csv_path.open()))
complete = [row for row in rows if row["status"] == "complete"]
print(f"[rob132-stanage-finalizer] complete={len(complete)}/{len(rows)}")
if len(complete) != len(rows):
    for row in rows:
        if row["status"] != "complete":
            print(
                "[rob132-stanage-finalizer] missing "
                f"reward={row['reward']} dataset={row['dataset_tag']} epoch={row['epochs']}"
            )
    raise SystemExit(1)
PY
else
  echo "[rob132-stanage-finalizer] summary skipped for callback smoke"
fi

callback_status_code="${summary_status}"
if [ -z "${LINEAR_API_KEY:-}" ] && [ "${ROB132_FINALIZER_CALLBACK_DRY_RUN:-0}" != "1" ]; then
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
  --note "${CALLBACK_NOTE:-ROB-132 Stanage split test-set eval finalizer completed. Inspect OUTCOME.md and per-cell logs under ${RESULT_ROOT}/logs/stanage, then finish PR/Linear handoff.}"
  --tail-lines "${CALLBACK_TAIL_LINES:-80}"
  --max-log-chars "${CALLBACK_MAX_LOG_CHARS:-6000}"
  --max-comment-chars "${CALLBACK_MAX_COMMENT_CHARS:-10000}"
)
if [ "${ROB132_FINALIZER_CALLBACK_CHECK_ONLY:-0}" = "1" ]; then
  callback_args+=(--check-only)
fi
if [ "${ROB132_FINALIZER_CALLBACK_DRY_RUN:-0}" = "1" ]; then
  callback_args+=(--dry-run)
fi

if command -v module >/dev/null 2>&1; then
  module load Anaconda3/2022.10
fi
source activate /mnt/parscratch/users/acp21rjf/conda/main
python3 scripts/callbacks/linear_experiment_callback.py "${callback_args[@]}"
callback_status=$?
if [ "${callback_status}" -ne 0 ]; then
  echo "Linear completion callback failed with status ${callback_status}" >&2
fi

exit "${summary_status}"
