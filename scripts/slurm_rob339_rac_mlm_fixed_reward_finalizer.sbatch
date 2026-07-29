#!/usr/bin/env bash
# Aggregate ROB-339 RAC-MLM repeat cells and wake Symphony through Linear.

#SBATCH --job-name=rob339-final
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

REPO_DIR="${REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-learning-to-augment/ROB-339}"
LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-339}"
HISTORICAL_ROOT="${HISTORICAL_ROOT:-${REPO_DIR}/exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384/eval/test_fixed_rewards_0_and_1}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384/eval/test_fixed_rewards_0_and_1_rob339_repeats}"
LOG_PATH="${LOG_PATH:-${RESULT_ROOT}/logs/stanage/rob339_stanage_finalizer-${SLURM_JOB_ID:-manual}.log}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/parscratch/users/acp21rjf/l2augment_model/reward_conditioned_mask_lm/audio_ssl_hubert_base_tedlium_per_utterance_transformer384_dropout0p1_500ep_lr1e3.pt}"
ROB339_FIXED_REWARDS="${ROB339_FIXED_REWARDS:-1.0 0.0}"
ROB339_DATASETS="${ROB339_DATASETS:-tedlium earnings22 rev16 TAL chime6}"
ROB339_EPOCHS="${ROB339_EPOCHS:-1 5}"
ROB339_REPEATS="${ROB339_REPEATS:-1 2 3}"
ROB339_LR="${ROB339_LR:-1e-5}"
CSV_NAME="${CSV_NAME:-rob339_rac_mlm_fixed_reward_repeats.csv}"
AGGREGATE_CSV_NAME="${AGGREGATE_CSV_NAME:-rob339_rac_mlm_fixed_reward_repeats_aggregate.csv}"
RUNNER_LABEL="${RUNNER_LABEL:-slurm-finalizer:${SLURM_JOB_ID:-manual}}"
QUEUED_COMMAND="${QUEUED_COMMAND:-sbatch ${SLURM_SUBMIT_SCRIPT:-scripts/slurm_rob339_rac_mlm_fixed_reward_finalizer.sbatch}}"

mkdir -p "$(dirname "${LOG_PATH}")" "${RESULT_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

cd "${REPO_DIR}"
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"
summary_status=0

echo "[rob339-stanage-finalizer] job_id=${SLURM_JOB_ID:-manual}"
echo "[rob339-stanage-finalizer] host=$(hostname)"
echo "[rob339-stanage-finalizer] historical_root=${HISTORICAL_ROOT}"
echo "[rob339-stanage-finalizer] result_root=${RESULT_ROOT}"
echo "[rob339-stanage-finalizer] branch=${GIT_BRANCH}"
echo "[rob339-stanage-finalizer] commit=${GIT_COMMIT}"
echo "[rob339-stanage-finalizer] queued_command=${QUEUED_COMMAND}"

if [ "${ROB339_FINALIZER_SKIP_SUMMARY:-0}" != "1" ]; then
  if command -v module >/dev/null 2>&1; then
    module load Anaconda3/2022.10
  fi
  source activate /mnt/parscratch/users/acp21rjf/conda/main
  python3 scripts/summarize_rob339_rac_mlm_fixed_reward_repeats.py \
    --historical-root "${HISTORICAL_ROOT}" \
    --result-root "${RESULT_ROOT}" \
    --fixed-rewards "${ROB339_FIXED_REWARDS}" \
    --datasets "${ROB339_DATASETS}" \
    --epochs "${ROB339_EPOCHS}" \
    --lr "${ROB339_LR}" \
    --repeats "${ROB339_REPEATS}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --command "${QUEUED_COMMAND}" \
    --branch "${GIT_BRANCH}" \
    --commit "${GIT_COMMIT}" \
    --log-path "${LOG_PATH}" \
    --screen-log-path "slurm" \
    --csv-name "${CSV_NAME}" \
    --aggregate-csv-name "${AGGREGATE_CSV_NAME}" || summary_status=$?

  python3 - "${RESULT_ROOT}/${CSV_NAME}" "${RESULT_ROOT}/${AGGREGATE_CSV_NAME}" <<'PY' || summary_status=$?
import csv
import sys
from pathlib import Path

rows = list(csv.DictReader(Path(sys.argv[1]).open()))
complete = [row for row in rows if row["status"] == "complete"]
print(f"[rob339-stanage-finalizer] complete={len(complete)}/{len(rows)}")
if len(complete) != len(rows):
    for row in rows:
        if row["status"] != "complete":
            print(
                "[rob339-stanage-finalizer] missing "
                f"reward={row['reward']} dataset={row['dataset_tag']} "
                f"epoch={row['epochs']} repeat={row['repeat']}"
            )
    raise SystemExit(1)

aggregate = list(csv.DictReader(Path(sys.argv[2]).open()))
bad_n = [row for row in aggregate if row["n"] != "3"]
if bad_n:
    for row in bad_n:
        print(
            "[rob339-stanage-finalizer] bad aggregate n "
            f"reward={row['reward']} dataset={row['dataset_tag']} "
            f"epoch={row['epochs']} n={row['n']}"
        )
    raise SystemExit(1)
PY
else
  echo "[rob339-stanage-finalizer] summary skipped for callback smoke"
fi

callback_status_code="${summary_status}"
if [ -z "${LINEAR_API_KEY:-}" ] && [ "${ROB339_FINALIZER_CALLBACK_DRY_RUN:-0}" != "1" ]; then
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
  --note "${CALLBACK_NOTE:-ROB-339 Stanage RAC-MLM fixed-reward repeat finalizer completed. Inspect OUTCOME.md, aggregate CSV, and per-cell logs under ${RESULT_ROOT}/logs/stanage, then finish PR/Linear handoff.}"
  --tail-lines "${CALLBACK_TAIL_LINES:-80}"
  --max-log-chars "${CALLBACK_MAX_LOG_CHARS:-6000}"
  --max-comment-chars "${CALLBACK_MAX_COMMENT_CHARS:-10000}"
)
if [ "${ROB339_FINALIZER_CALLBACK_CHECK_ONLY:-0}" = "1" ]; then
  callback_args+=(--check-only)
fi
if [ "${ROB339_FINALIZER_CALLBACK_DRY_RUN:-0}" = "1" ]; then
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
