#!/usr/bin/env bash
# Queue-safe ROB-80 TED-LIUM dev LR sweep for RFM, RMM, and UFMR.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-80}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/exp/results/repro/sweeps}"
LOG_PATH="${LOG_PATH:-${RESULT_ROOT}/logs/rob80_tedlium_policy_sweep.log}"
SCREEN_NAME="${SCREEN_NAME:-rob80_tedlium_policy_sweep}"
RUNNER_LABEL="${RUNNER_LABEL:-screen:${SCREEN_NAME}}"
QUEUED_COMMAND="${QUEUED_COMMAND:-/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob80_tedlium_policy_sweep.sh}"
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"
ASR_CKPT="${ASR_CKPT:-/store/store5/data/acp21rjf_checkpoints/spotify/rotary_pos_6l_256d_seq_sched/n_seq_sched_2048_rp_1/step_105360.pt}"
UFMR_VARIANT="${UFMR_VARIANT:-test_wer}"
UFMR_CKPT="${UFMR_CKPT:-/store/store5/data/acp21rjf_checkpoints/l2augment/ufmr/${UFMR_VARIANT}/model.pt}"
BASE_LRS="${ROB80_BASE_LRS:-5e-6 1e-5 2e-5}"
UFMR_EXTRA_LRS="${ROB80_UFMR_EXTRA_LRS:-4e-5 8e-5 1.6e-4}"
DATASET="${ROB80_DATASET:-tedlium}"
SPLIT="${ROB80_SPLIT:-dev}"
TAG_PREFIX="${ROB80_TAG_PREFIX:-tedlium_dev}"
TITLE="${ROB80_TITLE:-ROB-80 TED-LIUM Dev Policy LR Sweep}"
NOTE="${ROB80_NOTE:-UFMR includes higher-LR follow-up cells requested after the initial sweep: \`4e-5\`, \`8e-5\`, and \`1.6e-4\`.}"
CSV_NAME="${ROB80_CSV_NAME:-rob80_tedlium_policy_sweep.csv}"
OUTCOME_NAME="${ROB80_OUTCOME_NAME:-ROB-80_OUTCOME.md}"
INCLUDE_UFMR_EXTRA="${ROB80_INCLUDE_UFMR_EXTRA:-1}"

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
    --note "${CALLBACK_NOTE:-ROB-80 TED-LIUM dev RFM/RMM/UFMR LR sweep wrapper exited. See exp/results/repro/sweeps/ROB-80_OUTCOME.md for the result table when complete.}"
    --tail-lines "${CALLBACK_TAIL_LINES:-80}"
    --max-log-chars "${CALLBACK_MAX_LOG_CHARS:-6000}"
    --max-comment-chars "${CALLBACK_MAX_COMMENT_CHARS:-10000}"
  )
  if [ "${ROB80_CALLBACK_CHECK_ONLY:-0}" = "1" ]; then
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

echo "[rob80] branch=${GIT_BRANCH}"
echo "[rob80] commit=${GIT_COMMIT}"
echo "[rob80] result_root=${RESULT_ROOT}"
echo "[rob80] asr_ckpt=${ASR_CKPT}"
echo "[rob80] ufmr_variant=${UFMR_VARIANT}"
echo "[rob80] ufmr_ckpt=${UFMR_CKPT}"
echo "[rob80] dataset=${DATASET}"
echo "[rob80] split=${SPLIT}"
echo "[rob80] tag_prefix=${TAG_PREFIX}"
echo "[rob80] base_lrs=${BASE_LRS}"
echo "[rob80] ufmr_extra_lrs=${UFMR_EXTRA_LRS}"
echo "[rob80] include_ufmr_extra=${INCLUDE_UFMR_EXTRA}"
echo "[rob80] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

if [ "${ROB80_CALLBACK_ONLY:-0}" = "1" ]; then
  echo "[rob80] callback-only smoke path requested; exiting before eval."
  exit 0
fi

if [ ! -f "${ASR_CKPT}" ]; then
  echo "Missing ASR checkpoint: ${ASR_CKPT}" >&2
  exit 1
fi

if [ ! -f "${UFMR_CKPT}" ]; then
  fallback="/store/store5/data/acp21rjf_checkpoints/l2augment/ufmr/${UFMR_VARIANT}/tmp_model.pt"
  if [ -f "${fallback}" ]; then
    echo "[rob80] no model.pt for UFMR variant '${UFMR_VARIANT}'; using tmp_model.pt" >&2
    UFMR_CKPT="${fallback}"
  else
    echo "Missing UFMR checkpoint for variant '${UFMR_VARIANT}': ${UFMR_CKPT}" >&2
    exit 1
  fi
fi

export L2A_EARNINGS22_DIR="${L2A_EARNINGS22_DIR:-/store/store4/data/earnings-22}"
export L2A_TEDLIUM3_LEGACY_DIR="${L2A_TEDLIUM3_LEGACY_DIR:-/store/store4/data/TEDLIUM_release-3/legacy/}"
export L2A_REV16_DIR="${L2A_REV16_DIR:-/store/store4/data/rev_benchmark}"
export L2A_CHIME6_DIR="${L2A_CHIME6_DIR:-/store/store4/data/chime6/}"

python3 - "${RESULT_ROOT}" "${ASR_CKPT}" "${UFMR_CKPT}" "${BASE_LRS}" "${UFMR_EXTRA_LRS}" "${DATASET}" "${SPLIT}" "${TAG_PREFIX}" "${INCLUDE_UFMR_EXTRA}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
asr_ckpt = sys.argv[2]
ufmr_ckpt = sys.argv[3]
base_lrs = tuple(sys.argv[4].split())
ufmr_extra_lrs = tuple(sys.argv[5].split())
dataset = sys.argv[6]
split = sys.argv[7]
tag_prefix = sys.argv[8]
include_ufmr_extra = sys.argv[9] == "1"
epochs = (1, 5)
ufmr_lrs = base_lrs + (ufmr_extra_lrs if include_ufmr_extra else ())
methods = {
    "RFM": ("FrequencyMaskingRanker", None, 1, True, base_lrs),
    "RMM": ("MixedMaskingRanker", None, 1, True, base_lrs),
    "UFMR": (
        "UnconditionalFrequencyMaskingRanker",
        ufmr_ckpt,
        15,
        False,
        ufmr_lrs,
    ),
}
for method, (policy_class, policy_ckpt, repeats, use_random, method_lrs) in methods.items():
    (root / method / "configs").mkdir(parents=True, exist_ok=True)
    for epoch_count in epochs:
        for lr in method_lrs:
            tag = f"{tag_prefix}_epoch{epoch_count}_lr{lr}"
            save_path = root / method / f"{tag}.txt"
            config_path = root / method / "configs" / f"{tag}.yaml"
            training_extra = ""
            if policy_ckpt is not None:
                training_extra = (
                    f"  model_save_path: {policy_ckpt}\n"
                    f"  tmp_model_save_path: {policy_ckpt}\n"
                )
            config_path.write_text(
                f"""checkpointing:
  asr_model: {asr_ckpt}

training:
  device: 'cuda'
  random_seed: 1234
  batch_size: 84
  epochs: 100
{training_extra}
evaluation:
  id: 'ROB-80-{dataset}-{split}-{method}-epoch{epoch_count}-lr{lr}'
  dataset: '{dataset}'
  split: '{split}'
  rollout_setting: policy
  use_cer: false
  epochs: {epoch_count}
  augmentation_config:
    repeats: {repeats}
    use_random: {str(use_random).lower()}
  optim_args:
    lr: {lr}
  save_path: {save_path}

policy:
  lr: 1e-4
  class: {policy_class}
"""
            )
            print(f"[rob80] wrote config {config_path}")
PY

if [ "${ROB80_CONFIG_ONLY:-0}" = "1" ]; then
  echo "[rob80] config-only smoke path requested; exiting before eval."
  exit 0
fi

source /store/store4/software/bin/anaconda3/etc/profile.d/conda.sh
conda activate /store/store4/software/bin/anaconda3/envs/flash_attn_pytorch2

export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

METHODS="${ROB80_METHODS:-RFM RMM UFMR}"
EPOCHS="${ROB80_EPOCHS:-1 5}"
OVERRIDE_LRS="${ROB80_LRS:-}"
EVAL_SCRIPT="${ROB80_EVAL_SCRIPT:-eval.py}"

if [ "${ROB80_SMOKE:-0}" = "1" ]; then
  METHODS="${ROB80_SMOKE_METHODS:-RFM}"
  EPOCHS="${ROB80_SMOKE_EPOCHS:-1}"
  OVERRIDE_LRS="${ROB80_SMOKE_LRS:-5e-6}"
  ROB80_INDEXES="${ROB80_INDEXES:-0}"
  ROB80_DONT_SAVE="${ROB80_DONT_SAVE:-1}"
  echo "[rob80] smoke mode: methods=${METHODS}; epochs=${EPOCHS}; lrs=${OVERRIDE_LRS}; indexes=${ROB80_INDEXES}; dont_save=${ROB80_DONT_SAVE}"
fi

cd "${REPO_DIR}/exp"
for method in ${METHODS}; do
  method_lrs="${OVERRIDE_LRS}"
  if [ -z "${method_lrs}" ]; then
    method_lrs="${BASE_LRS}"
    if [ "${method}" = "UFMR" ] && [ "${INCLUDE_UFMR_EXTRA}" = "1" ]; then
      method_lrs="${BASE_LRS} ${UFMR_EXTRA_LRS}"
    fi
  fi
  for epoch_count in ${EPOCHS}; do
    for lr in ${method_lrs}; do
      tag="${TAG_PREFIX}_epoch${epoch_count}_lr${lr}"
      config="${RESULT_ROOT}/${method}/configs/${tag}.yaml"
      save_path="${RESULT_ROOT}/${method}/${tag}.txt"
      if [ ! -f "${config}" ]; then
        echo "Missing generated config: ${config}" >&2
        exit 1
      fi
      if [ "${FORCE_RERUN:-0}" != "1" ] && [ -f "${save_path}" ] && grep -q "Updated_WER:" "${save_path}"; then
        echo "[rob80] skipping completed ${method}/${tag}: ${save_path}"
        continue
      fi
      if [ "${FORCE_RERUN:-0}" = "1" ]; then
        rm -f "${save_path}"
      fi
      args=(python "${EVAL_SCRIPT}" --config "${config}")
      if [ -n "${ROB80_INDEXES:-}" ]; then
        args+=(--indexes ${ROB80_INDEXES})
      fi
      if [ "${ROB80_DONT_SAVE:-0}" = "1" ]; then
        args+=(--dont_save)
      fi
      echo "[rob80] running ${method}/${tag}: ${args[*]}"
      "${args[@]}"
    done
  done
done

cd "${REPO_DIR}"
summary_args=(
  --result-root "${RESULT_ROOT}"
  --dataset "${DATASET}"
  --split "${SPLIT}"
  --tag-prefix "${TAG_PREFIX}"
  --csv-name "${CSV_NAME}"
  --outcome-name "${OUTCOME_NAME}"
  --title "${TITLE}"
  --note "${NOTE}"
)
if [ "${INCLUDE_UFMR_EXTRA}" = "1" ]; then
  summary_args+=(--include-ufmr-extra)
else
  summary_args+=(--no-include-ufmr-extra)
fi
python3 scripts/summarize_rob80_tedlium_sweep.py "${summary_args[@]}"
echo "[rob80] finished"
