#!/usr/bin/env bash
# Aggregate ROB-340 large-ASR repeat cells and wake Symphony through Linear.

#SBATCH --job-name=rob340-final
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

REPO_DIR="${REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-learning-to-augment/ROB-340}"
LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-340}"
AGGREGATE_DIR="${AGGREGATE_DIR:-${REPO_DIR}/exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048}"
RESULT_ROOT="${RESULT_ROOT:-${AGGREGATE_DIR}/results}"
LOG_PATH="${LOG_PATH:-${AGGREGATE_DIR}/logs/stanage/rob340_stanage_finalizer-${SLURM_JOB_ID:-manual}.log}"
ROB340_METHODS="${ROB340_METHODS:-UFMR RFM}"
ROB340_DATASETS="${ROB340_DATASETS:-tedlium earnings22 chime6 rev16 TAL}"
ROB340_REPEATS="${ROB340_REPEATS:-1 2 3}"
ROB340_LR="${ROB340_LR:-1e-5}"
ROB340_EPOCH="${ROB340_EPOCH:-1}"
CSV_NAME="${CSV_NAME:-rob340_large_asr_rfm_ufmr_repeats.csv}"
OUTCOME_NAME="${OUTCOME_NAME:-ROB-340_OUTCOME.md}"
RUNNER_LABEL="${RUNNER_LABEL:-slurm-finalizer:${SLURM_JOB_ID:-manual}}"
QUEUED_COMMAND="${QUEUED_COMMAND:-sbatch ${SLURM_SUBMIT_SCRIPT:-scripts/slurm_rob340_large_asr_finalizer.sbatch}}"

mkdir -p "$(dirname "${LOG_PATH}")" "${AGGREGATE_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

cd "${REPO_DIR}"
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"
summary_status=0

activate_l2a_env() {
  if command -v module >/dev/null 2>&1; then
    module load Anaconda3/2022.10
  fi
  if [ -d /mnt/parscratch/users/acp21rjf/conda/main ]; then
    source activate /mnt/parscratch/users/acp21rjf/conda/main
  else
    echo "[rob340-stanage-finalizer] Stanage conda env not found; using current Python environment"
  fi
}

echo "[rob340-stanage-finalizer] job_id=${SLURM_JOB_ID:-manual}"
echo "[rob340-stanage-finalizer] host=$(hostname)"
echo "[rob340-stanage-finalizer] aggregate_dir=${AGGREGATE_DIR}"
echo "[rob340-stanage-finalizer] result_root=${RESULT_ROOT}"
echo "[rob340-stanage-finalizer] branch=${GIT_BRANCH}"
echo "[rob340-stanage-finalizer] commit=${GIT_COMMIT}"
echo "[rob340-stanage-finalizer] methods=${ROB340_METHODS}"
echo "[rob340-stanage-finalizer] datasets=${ROB340_DATASETS}"
echo "[rob340-stanage-finalizer] repeats=${ROB340_REPEATS}"
echo "[rob340-stanage-finalizer] queued_command=${QUEUED_COMMAND}"

if [ "${ROB340_FINALIZER_SKIP_SUMMARY:-0}" != "1" ]; then
  activate_l2a_env
  python3 scripts/summarize_rob108_test_policy_evals.py \
    --result-root "${RESULT_ROOT}" \
    --output-dir "${AGGREGATE_DIR}" \
    --datasets "${ROB340_DATASETS}" \
    --methods "${ROB340_METHODS}" \
    --repeats "${ROB340_REPEATS}" \
    --epoch1-lrs "${ROB340_LR}" \
    --epoch5-lrs "" \
    --csv-name "${CSV_NAME}" \
    --outcome-name "${OUTCOME_NAME}" \
    --title "ROB-340 UFMR/RFM Large-ASR Repeat Completion" \
    --note "Thesis-table large-ASR UFMR/RFM cells only: 90M 2048-context ASR, test split, one adaptation epoch, lr=1e-5. Repeat 1 is the existing ROB-158 seed 123456 artifact; repeats 2 and 3 are new ROB-340 Stanage jobs with seeds 123457 and 123458." || summary_status=$?

  python3 - "${AGGREGATE_DIR}/${CSV_NAME}" "${ROB340_METHODS}" "${ROB340_DATASETS}" "${ROB340_REPEATS}" <<'PY' || summary_status=$?
import csv
import sys
from collections import defaultdict
from pathlib import Path

csv_path = Path(sys.argv[1])
methods = tuple(sys.argv[2].split())
datasets = tuple(sys.argv[3].split())
repeats = tuple(sys.argv[4].split())
rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
complete = [row for row in rows if row["status"] == "complete"]
expected_count = len(methods) * len(datasets) * len(repeats)
print(f"[rob340-stanage-finalizer] complete={len(complete)}/{len(rows)} expected={expected_count}")
if len(rows) != expected_count or len(complete) != expected_count:
    for row in rows:
        if row["status"] != "complete":
            print(
                "[rob340-stanage-finalizer] missing "
                f"method={row['method']} dataset={row['dataset_tag']} repeat={row['repeat']} "
                f"epoch={row['epochs']} lr={row['lr']} result={row['result_path']}"
            )
    raise SystemExit(1)

baseline_tolerance = 1e-3
by_cell: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
baselines: dict[str, list[float]] = defaultdict(list)
for row in complete:
    by_cell[(row["dataset_tag"], row["method"])].append(row)
    baselines[row["dataset_tag"]].append(float(row["original_wer"]))

bad_n = [(dataset, method, len(group)) for (dataset, method), group in sorted(by_cell.items()) if len(group) != len(repeats)]
if bad_n:
    for dataset, method, n in bad_n:
        print(f"[rob340-stanage-finalizer] bad_n dataset={dataset} method={method} n={n}")
    raise SystemExit(1)

for dataset in datasets:
    values = sorted(baselines[dataset])
    spread = max(values) - min(values)
    formatted = ",".join(f"{value:.6f}" for value in values)
    print(
        f"[rob340-stanage-finalizer] baseline_original_wer dataset={dataset} "
        f"values={formatted} spread={spread:.6f} tolerance={baseline_tolerance:.6f}"
    )
    if spread > baseline_tolerance:
        raise SystemExit(f"baseline original WER mismatch for {dataset}: {formatted}")
PY
else
  echo "[rob340-stanage-finalizer] summary skipped for callback smoke"
fi

callback_status_code="${summary_status}"
if [ -z "${LINEAR_API_KEY:-}" ] && [ "${ROB340_FINALIZER_CALLBACK_DRY_RUN:-0}" != "1" ]; then
  echo "LINEAR_API_KEY is not set; cannot post Linear completion callback" >&2
  exit "${summary_status}"
fi

callback_args=(
  --issue "${LINEAR_ISSUE}"
  --status-code "${callback_status_code}"
  --log "${LOG_PATH}"
  --results "${AGGREGATE_DIR}"
  --runner-label "${RUNNER_LABEL}"
  --queued-command "${QUEUED_COMMAND}"
  --branch "${GIT_BRANCH}"
  --commit "${GIT_COMMIT}"
  --target-state "${CALLBACK_TARGET_STATE:-Todo}"
  --note "${CALLBACK_NOTE:-ROB-340 Stanage large-ASR repeat finalizer completed. Inspect ${AGGREGATE_DIR}/${OUTCOME_NAME}, ${AGGREGATE_DIR}/${CSV_NAME}, and per-cell logs under ${AGGREGATE_DIR}/logs/stanage before final handoff.}"
  --tail-lines "${CALLBACK_TAIL_LINES:-80}"
  --max-log-chars "${CALLBACK_MAX_LOG_CHARS:-6000}"
  --max-comment-chars "${CALLBACK_MAX_COMMENT_CHARS:-10000}"
)
if [ "${ROB340_FINALIZER_CALLBACK_CHECK_ONLY:-0}" = "1" ]; then
  callback_args+=(--check-only)
fi
if [ "${ROB340_FINALIZER_CALLBACK_DRY_RUN:-0}" = "1" ]; then
  callback_args+=(--dry-run)
fi

activate_l2a_env
python3 scripts/callbacks/linear_experiment_callback.py "${callback_args[@]}"
callback_status=$?
if [ "${callback_status}" -ne 0 ]; then
  echo "Linear completion callback failed with status ${callback_status}" >&2
fi

exit "${summary_status}"
