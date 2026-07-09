#!/usr/bin/env bash
# Aggregate ROB-337 learnt-augmentation repeat cells and wake Symphony through Linear.

#SBATCH --job-name=rob337-final
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

REPO_DIR="${REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-learning-to-augment/ROB-337}"
LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-337}"
BASE_REPRO_DIR="${BASE_REPRO_DIR:-${REPO_DIR}/exp/results/repro}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/exp/results/repro/learnt_augmentation_repeats/rob337}"
LOG_PATH="${LOG_PATH:-${RESULT_ROOT}/logs/stanage/rob337_stanage_finalizer-${SLURM_JOB_ID:-manual}.log}"
QUEUED_JOBS_PATH="${QUEUED_JOBS_PATH:-${RESULT_ROOT}/queued_jobs.tsv}"
ROB337_METHODS="${ROB337_METHODS:-UFMR RFM RMM UVQLM}"
ROB337_DATASETS="${ROB337_DATASETS:-tedlium earnings22 rev16 TAL chime6}"
ROB337_EPOCHS="${ROB337_EPOCHS:-1 5}"
ROB337_REPEATS="${ROB337_REPEATS:-1 2 3}"
ROB337_LR="${ROB337_LR:-1e-5}"
RUNNER_LABEL="${RUNNER_LABEL:-slurm-finalizer:${SLURM_JOB_ID:-manual}}"
QUEUED_COMMAND="${QUEUED_COMMAND:-sbatch ${SLURM_SUBMIT_SCRIPT:-scripts/slurm_rob337_learnt_augmentation_finalizer.sbatch}}"

mkdir -p "$(dirname "${LOG_PATH}")" "${RESULT_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

cd "${REPO_DIR}"
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"
summary_status=0

echo "[rob337-finalizer] job_id=${SLURM_JOB_ID:-manual}"
echo "[rob337-finalizer] host=$(hostname)"
echo "[rob337-finalizer] result_root=${RESULT_ROOT}"
echo "[rob337-finalizer] base_repro_dir=${BASE_REPRO_DIR}"
echo "[rob337-finalizer] branch=${GIT_BRANCH}"
echo "[rob337-finalizer] commit=${GIT_COMMIT}"
echo "[rob337-finalizer] queued_command=${QUEUED_COMMAND}"

if [ "${ROB337_FINALIZER_SKIP_SUMMARY:-0}" != "1" ]; then
  if command -v module >/dev/null 2>&1; then
    module load Anaconda3/2022.10
  fi
  source activate /mnt/parscratch/users/acp21rjf/conda/main
  python3 scripts/summarize_rob337_learnt_augmentation_repeats.py \
    --base-repro-dir "${BASE_REPRO_DIR}" \
    --result-root "${RESULT_ROOT}" \
    --methods "${ROB337_METHODS}" \
    --datasets "${ROB337_DATASETS}" \
    --epochs "${ROB337_EPOCHS}" \
    --repeats "${ROB337_REPEATS}" \
    --lr "${ROB337_LR}" \
    --command "${QUEUED_COMMAND}" \
    --branch "${GIT_BRANCH}" \
    --commit "${GIT_COMMIT}" \
    --log-path "${LOG_PATH}" \
    --queued-jobs-path "${QUEUED_JOBS_PATH}" || summary_status=$?

  python3 - "${RESULT_ROOT}/rob337_learnt_augmentation_repeats.csv" "${RESULT_ROOT}/rob337_learnt_augmentation_repeats_aggregate.csv" <<'PY' || summary_status=$?
import csv
import sys
from pathlib import Path

per_repeat_path = Path(sys.argv[1])
aggregate_path = Path(sys.argv[2])
rows = list(csv.DictReader(per_repeat_path.open()))
aggregates = list(csv.DictReader(aggregate_path.open()))
complete = [row for row in rows if row["status"] == "complete"]
n3 = [row for row in aggregates if row["n"] == "3"]
print(f"[rob337-finalizer] complete_repeat_rows={len(complete)}/{len(rows)}")
print(f"[rob337-finalizer] aggregate_n3={len(n3)}/{len(aggregates)}")
if len(complete) != len(rows) or len(n3) != 40:
    for row in rows:
        if row["status"] != "complete":
            print(
                "[rob337-finalizer] missing "
                f"method={row['method']} dataset={row['dataset_tag']} epoch={row['epochs']} repeat={row['repeat']}"
            )
    for row in aggregates:
        if row["n"] != "3":
            print(
                "[rob337-finalizer] incomplete aggregate "
                f"method={row['method']} dataset={row['dataset_tag']} epoch={row['epochs']} n={row['n']}"
            )
    raise SystemExit(1)
PY
else
  echo "[rob337-finalizer] summary skipped for callback smoke"
fi

callback_status_code="${summary_status}"
if [ -z "${LINEAR_API_KEY:-}" ] && [ "${ROB337_FINALIZER_CALLBACK_DRY_RUN:-0}" != "1" ]; then
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
  --note "${CALLBACK_NOTE:-ROB-337 Stanage learnt-augmentation repeat finalizer completed. Inspect OUTCOME.md, per-repeat CSV, aggregate CSV, queued_jobs.tsv, and per-cell logs under ${RESULT_ROOT}.}"
  --tail-lines "${CALLBACK_TAIL_LINES:-80}"
  --max-log-chars "${CALLBACK_MAX_LOG_CHARS:-6000}"
  --max-comment-chars "${CALLBACK_MAX_COMMENT_CHARS:-10000}"
)
if [ "${ROB337_FINALIZER_CALLBACK_CHECK_ONLY:-0}" = "1" ]; then
  callback_args+=(--check-only)
fi
if [ "${ROB337_FINALIZER_CALLBACK_DRY_RUN:-0}" = "1" ]; then
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
