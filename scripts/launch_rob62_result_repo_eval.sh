#!/usr/bin/env bash
# Queue-safe ROB-62 result-repo 2048 single-epoch RMM/RFM eval wrapper.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-62}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/exp/results/repro/policy/ROB-62_result_repo_2048_1epoch}"
LOG_PATH="${LOG_PATH:-${RESULT_ROOT}/logs/run.log}"
SCREEN_NAME="${SCREEN_NAME:-rob62_rmm_result_repo_2048_1epoch}"
RUNNER_LABEL="${RUNNER_LABEL:-screen:${SCREEN_NAME}}"
QUEUED_COMMAND="${QUEUED_COMMAND:-/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob62_result_repo_eval.sh}"
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"
ASR_CKPT="${ASR_CKPT:-/store/store5/data/acp21rjf_checkpoints/spotify/rotary_pos_6l_256d_seq_sched/n_seq_sched_2048_rp_1/step_105360.pt}"

on_exit() {
  status=$?
  set +e
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
    --note "${CALLBACK_NOTE:-ROB-62 result-repo 2048 single-epoch RMM/RFM eval wrapper exited.}"
    --tail-lines "${CALLBACK_TAIL_LINES:-80}"
    --max-log-chars "${CALLBACK_MAX_LOG_CHARS:-6000}"
    --max-comment-chars "${CALLBACK_MAX_COMMENT_CHARS:-10000}"
  )
  if [ "${ROB62_CALLBACK_CHECK_ONLY:-0}" = "1" ]; then
    callback_args+=(--check-only)
  fi
  python3 scripts/callbacks/linear_experiment_callback.py "${callback_args[@]}"
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

echo "[rob62] branch=${GIT_BRANCH}"
echo "[rob62] commit=${GIT_COMMIT}"
echo "[rob62] result_root=${RESULT_ROOT}"
echo "[rob62] asr_ckpt=${ASR_CKPT}"
echo "[rob62] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

if [ "${ROB62_CALLBACK_ONLY:-0}" = "1" ]; then
  echo "[rob62] callback-only smoke path requested; exiting before eval."
  exit 0
fi

if [ ! -f "${ASR_CKPT}" ]; then
  echo "Missing ASR checkpoint: ${ASR_CKPT}" >&2
  exit 1
fi

export L2A_EARNINGS22_DIR="${L2A_EARNINGS22_DIR:-/store/store4/data/earnings-22}"
export L2A_TEDLIUM3_LEGACY_DIR="${L2A_TEDLIUM3_LEGACY_DIR:-/store/store4/data/TEDLIUM_release-3/legacy/}"
export L2A_REV16_DIR="${L2A_REV16_DIR:-/store/store4/data/rev_benchmark}"
export L2A_CHIME6_DIR="${L2A_CHIME6_DIR:-/store/store4/data/chime6/}"
export L2A_TAL_DIR="${L2A_TAL_DIR:-/store/store5/data/this_american_life}"

if [ "${INCLUDE_TAL:-0}" = "1" ]; then
  if [ -z "${L2A_TAL_DIR:-}" ] || [ ! -d "${L2A_TAL_DIR}" ]; then
    echo "INCLUDE_TAL=1 requires L2A_TAL_DIR to point at an existing TAL directory" >&2
    exit 1
  fi
fi

python3 - "${RESULT_ROOT}" "${ASR_CKPT}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
asr_ckpt = sys.argv[2]
datasets = {
    "tedlium": ("tedlium", "test"),
    "e22": ("earnings22", "test"),
    "rev16": ("rev16", "test"),
    "chime6": ("chime6", "test"),
    "TAL": ("this_american_life", "test"),
}
methods = {
    "RFM": "FrequencyMaskingRanker",
    "RMM": "MixedMaskingRanker",
}
for method, policy_class in methods.items():
    (root / "configs" / method).mkdir(parents=True, exist_ok=True)
    (root / method).mkdir(parents=True, exist_ok=True)
    for tag, (dataset, split) in datasets.items():
        save_path = root / method / f"{tag}.txt"
        config_path = root / "configs" / method / f"{tag}.yaml"
        config_path.write_text(
            f"""checkpointing:
  asr_model: {asr_ckpt}

training:
  device: 'cuda'
  random_seed: 1234
  batch_size: 84
  epochs: 100

evaluation:
  id: 'ROB-62-result-repo-2048-1epoch-5e-6-{method}-{tag}'
  dataset: '{dataset}'
  split: '{split}'
  use_cer: false
  epochs: 1
  augmentation_config:
    repeats: 1
    use_random: true
  optim_args:
    lr: 5e-6
  save_path: {save_path}

policy:
  lr: 1e-4
  class: {policy_class}
"""
        )
        print(f"[rob62] wrote config {config_path}")
PY

source /store/store4/software/bin/anaconda3/etc/profile.d/conda.sh
conda activate /store/store4/software/bin/anaconda3/envs/flash_attn_pytorch2

export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

METHODS="${ROB62_METHODS:-RFM RMM}"
DATASETS="${ROB62_DATASETS:-tedlium e22 rev16 chime6}"
if [ "${INCLUDE_TAL:-0}" = "1" ]; then
  DATASETS="${DATASETS} TAL"
fi

if [ "${ROB62_SMOKE:-0}" = "1" ]; then
  METHODS="${ROB62_SMOKE_METHODS:-RMM}"
  DATASETS="${ROB62_SMOKE_DATASETS:-tedlium}"
  ROB62_INDEXES="${ROB62_INDEXES:-0}"
  ROB62_DONT_SAVE="${ROB62_DONT_SAVE:-1}"
  echo "[rob62] smoke mode: methods=${METHODS}; datasets=${DATASETS}; indexes=${ROB62_INDEXES}; dont_save=${ROB62_DONT_SAVE}"
fi

cd "${REPO_DIR}/exp"
for method in ${METHODS}; do
  for dataset in ${DATASETS}; do
    config="${RESULT_ROOT}/configs/${method}/${dataset}.yaml"
    save_path="${RESULT_ROOT}/${method}/${dataset}.txt"
    if [ ! -f "${config}" ]; then
      echo "Missing generated config: ${config}" >&2
      exit 1
    fi
    if [ "${FORCE_RERUN:-0}" != "1" ] && [ -f "${save_path}" ] && grep -q "Updated_WER:" "${save_path}"; then
      echo "[rob62] skipping completed ${method}/${dataset}: ${save_path}"
      continue
    fi
    if [ "${FORCE_RERUN:-0}" = "1" ]; then
      rm -f "${save_path}"
    fi
    args=(python eval.py --config "${config}")
    if [ -n "${ROB62_INDEXES:-}" ]; then
      args+=(--indexes ${ROB62_INDEXES})
    fi
    if [ "${ROB62_DONT_SAVE:-0}" = "1" ]; then
      args+=(--dont_save)
    fi
    echo "[rob62] running ${method}/${dataset}: ${args[*]}"
    "${args[@]}"
  done
done

cd "${REPO_DIR}"
if [ "${ROB62_SMOKE:-0}" = "1" ]; then
  echo "[rob62] smoke mode: skipping final comparison table generation"
else
  python3 "${RESULT_ROOT}/summarize_results.py"
fi
echo "[rob62] finished"
