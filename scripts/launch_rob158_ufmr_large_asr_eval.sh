#!/usr/bin/env bash
# Callback-backed ROB-158 UFMR eval with the larger 2048-seq-len 90M ASR model.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-158}"
AGGREGATE_DIR="${AGGREGATE_DIR:-${REPO_DIR}/exp/results/repro/symphony/rob-158/large_asr_2048_90m}"
RESULT_ROOT="${RESULT_ROOT:-${AGGREGATE_DIR}/results}"
LOG_PATH="${LOG_PATH:-${AGGREGATE_DIR}/logs/rob158_ufmr_large_asr_eval.log}"
SCREEN_NAME="${SCREEN_NAME:-rob158_ufmr_large_asr_eval}"
RUNNER_LABEL="${RUNNER_LABEL:-screen:${SCREEN_NAME}}"
QUEUED_COMMAND="${QUEUED_COMMAND:-/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob158_ufmr_large_asr_eval.sh}"
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"

ASR_CKPT="${ASR_CKPT:-/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_2048_rp_1/step_105360.pt}"
UFMR_VARIANT="${UFMR_VARIANT:-test_wer}"
UFMR_CKPT="${UFMR_CKPT:-/store/store5/data/acp21rjf_checkpoints/l2augment/ufmr/${UFMR_VARIANT}/model.pt}"

DATASETS="${ROB158_DATASETS:-tedlium earnings22 chime6 rev16 TAL}"
METHODS="UFMR"
REPEATS="${ROB158_REPEATS:-1}"
EPOCH1_LRS="${ROB158_EPOCH1_LRS:-1e-5 3e-5}"
EPOCH5_LRS="${ROB158_EPOCH5_LRS:-1e-5}"
UFMR_SEARCH_REPEATS="${ROB158_UFMR_SEARCH_REPEATS:-15}"

on_exit() {
  status=$?
  set +e
  if [ "${ROB158_DISABLE_CALLBACK:-0}" = "1" ]; then
    exit "${status}"
  fi
  if [ -z "${LINEAR_API_KEY:-}" ] && [ "${ROB158_CALLBACK_DRY_RUN:-0}" != "1" ]; then
    echo "LINEAR_API_KEY is not set; cannot post Linear completion callback" >&2
    exit "${status}"
  fi
  callback_args=(
    --issue "${LINEAR_ISSUE}"
    --status-code "${status}"
    --log "${LOG_PATH}"
    --results "${AGGREGATE_DIR}"
    --screen-name "${SCREEN_NAME}"
    --runner-label "${RUNNER_LABEL}"
    --queued-command "${QUEUED_COMMAND}"
    --branch "${GIT_BRANCH}"
    --commit "${GIT_COMMIT}"
    --target-state "${CALLBACK_TARGET_STATE:-Todo}"
    --note "${CALLBACK_NOTE:-ROB-158 UFMR large-ASR eval wrapper exited. Inspect ${AGGREGATE_DIR}/ROB-158_OUTCOME.md and compare with the ROB-108 small-ASR UFMR rows before final handoff.}"
    --tail-lines "${CALLBACK_TAIL_LINES:-80}"
    --max-log-chars "${CALLBACK_MAX_LOG_CHARS:-6000}"
    --max-comment-chars "${CALLBACK_MAX_COMMENT_CHARS:-10000}"
  )
  if [ "${ROB158_CALLBACK_CHECK_ONLY:-0}" = "1" ]; then
    callback_args+=(--check-only)
  fi
  if [ "${ROB158_CALLBACK_DRY_RUN:-0}" = "1" ]; then
    callback_args+=(--dry-run)
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

mkdir -p "$(dirname "${LOG_PATH}")" "${RESULT_ROOT}" "${AGGREGATE_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "[rob158] branch=${GIT_BRANCH}"
echo "[rob158] commit=${GIT_COMMIT}"
echo "[rob158] result_root=${RESULT_ROOT}"
echo "[rob158] aggregate_dir=${AGGREGATE_DIR}"
echo "[rob158] asr_ckpt=${ASR_CKPT}"
echo "[rob158] ufmr_ckpt=${UFMR_CKPT}"
echo "[rob158] datasets=${DATASETS}"
echo "[rob158] repeats=${REPEATS}"
echo "[rob158] epoch1_lrs=${EPOCH1_LRS}"
echo "[rob158] epoch5_lrs=${EPOCH5_LRS}"
echo "[rob158] ufmr_search_repeats=${UFMR_SEARCH_REPEATS}"
echo "[rob158] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

if [ "${ROB158_CALLBACK_ONLY:-0}" = "1" ]; then
  echo "[rob158] callback-only smoke path requested; exiting before eval."
  exit 0
fi

if [ ! -f "${ASR_CKPT}" ]; then
  echo "Missing ASR checkpoint: ${ASR_CKPT}" >&2
  exit 1
fi
if [ ! -f "${UFMR_CKPT}" ]; then
  fallback="/store/store5/data/acp21rjf_checkpoints/l2augment/ufmr/${UFMR_VARIANT}/tmp_model.pt"
  if [ -f "${fallback}" ]; then
    echo "[rob158] no model.pt for UFMR variant '${UFMR_VARIANT}'; using tmp_model.pt" >&2
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
export L2A_TAL_DIR="${L2A_TAL_DIR:-/store/store5/data/this_american_life}"

for required_dir in "${L2A_EARNINGS22_DIR}" "${L2A_TEDLIUM3_LEGACY_DIR}" "${L2A_REV16_DIR}" "${L2A_CHIME6_DIR}" "${L2A_TAL_DIR}"; do
  if [ ! -d "${required_dir}" ]; then
    echo "Missing dataset directory: ${required_dir}" >&2
    exit 1
  fi
done

python3 - "${RESULT_ROOT}" "${ASR_CKPT}" "${UFMR_CKPT}" "${DATASETS}" "${REPEATS}" "${EPOCH1_LRS}" "${EPOCH5_LRS}" "${UFMR_SEARCH_REPEATS}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
asr_ckpt = sys.argv[2]
ufmr_ckpt = sys.argv[3]
dataset_tags = tuple(sys.argv[4].split())
repeats = tuple(int(item) for item in sys.argv[5].split())
epoch1_lrs = tuple(sys.argv[6].split())
epoch5_lrs = tuple(sys.argv[7].split())
ufmr_search_repeats = int(sys.argv[8])

datasets = {
    "tedlium": ("tedlium", "test"),
    "earnings22": ("earnings22", "test"),
    "chime6": ("chime6", "test"),
    "rev16": ("rev16", "test"),
    "TAL": ("this_american_life", "test"),
}

(root / "UFMR" / "configs").mkdir(parents=True, exist_ok=True)

for dataset_tag in dataset_tags:
    dataset, split = datasets[dataset_tag]
    for repeat in repeats:
        repeat_suffix = "" if repeat == 1 else f"_repeat{repeat}"
        seed = 123456 + repeat - 1
        cells = tuple((1, lr) for lr in epoch1_lrs) + tuple((5, lr) for lr in epoch5_lrs)
        for epoch_count, lr in cells:
            tag = f"{dataset_tag}_epoch{epoch_count}_lr{lr}{repeat_suffix}"
            save_path = root / "UFMR" / f"{tag}.txt"
            config_path = root / "UFMR" / "configs" / f"{tag}.yaml"
            config_path.write_text(
                f"""checkpointing:
  asr_model: {asr_ckpt}

training:
  device: 'cuda'
  random_seed: {seed}
  batch_size: 84
  epochs: 100
  model_save_path: {ufmr_ckpt}
  tmp_model_save_path: {ufmr_ckpt}

evaluation:
  id: 'ROB-158-{dataset}-{split}-UFMR-{tag}-repeat{repeat}'
  dataset: '{dataset}'
  split: '{split}'
  rollout_setting: policy
  use_cer: false
  epochs: {epoch_count}
  augmentation_config:
    repeats: {ufmr_search_repeats}
    seed: {seed}
    use_random: false
  optim_args:
    lr: {lr}
  save_path: {save_path}

policy:
  lr: 1e-4
  class: UnconditionalFrequencyMaskingRanker
""",
                encoding="utf-8",
            )
            print(f"[rob158] wrote config {config_path}")
PY

python3 scripts/summarize_rob108_test_policy_evals.py \
  --result-root "${RESULT_ROOT}" \
  --output-dir "${AGGREGATE_DIR}" \
  --datasets "${DATASETS}" \
  --methods "${METHODS}" \
  --repeats "${REPEATS}" \
  --epoch1-lrs "${EPOCH1_LRS}" \
  --epoch5-lrs "${EPOCH5_LRS}" \
  --csv-name rob158_ufmr_large_asr_eval.csv \
  --outcome-name ROB-158_OUTCOME.md \
  --title "ROB-158 UFMR Large-ASR Evaluation" \
  --note "UFMR test-split evals using the SAP-style 2048-seq-len 90M ASR checkpoint. The UFMR policy, datasets, repeats, candidate count, and LR/epoch cells match the ROB-108 UFMR repro setup so the intended variable is ASR model size."

if [ "${ROB158_CONFIG_ONLY:-0}" = "1" ]; then
  echo "[rob158] config-only smoke path requested; exiting before eval."
  exit 0
fi

source /store/store4/software/bin/anaconda3/etc/profile.d/conda.sh
conda activate /store/store4/software/bin/anaconda3/envs/flash_attn_pytorch2

export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

RUN_DATASETS="${ROB158_RUN_DATASETS:-${DATASETS}}"
RUN_REPEATS="${ROB158_RUN_REPEATS:-${REPEATS}}"
RUN_EPOCHS="${ROB158_RUN_EPOCHS:-1 5}"
RUN_EPOCH1_LRS="${ROB158_RUN_EPOCH1_LRS:-${EPOCH1_LRS}}"
RUN_EPOCH5_LRS="${ROB158_RUN_EPOCH5_LRS:-${EPOCH5_LRS}}"
EVAL_SCRIPT="${ROB158_EVAL_SCRIPT:-eval.py}"

if [ "${ROB158_SMOKE:-0}" = "1" ]; then
  RUN_DATASETS="${ROB158_SMOKE_DATASETS:-tedlium}"
  RUN_REPEATS="${ROB158_SMOKE_REPEATS:-1}"
  RUN_EPOCHS="${ROB158_SMOKE_EPOCHS:-1}"
  RUN_EPOCH1_LRS="${ROB158_SMOKE_LRS:-1e-5}"
  RUN_EPOCH5_LRS="${ROB158_SMOKE_LRS:-1e-5}"
  ROB158_INDEXES="${ROB158_INDEXES:-0}"
  ROB158_DONT_SAVE="${ROB158_DONT_SAVE:-1}"
  echo "[rob158] smoke mode: datasets=${RUN_DATASETS}; repeats=${RUN_REPEATS}; epochs=${RUN_EPOCHS}; lrs=${ROB158_SMOKE_LRS:-1e-5}; indexes=${ROB158_INDEXES}; dont_save=${ROB158_DONT_SAVE}"
fi

cd "${REPO_DIR}/exp"
for dataset_tag in ${RUN_DATASETS}; do
  for repeat in ${RUN_REPEATS}; do
    repeat_suffix=""
    if [ "${repeat}" != "1" ]; then
      repeat_suffix="_repeat${repeat}"
    fi
    for epoch_count in ${RUN_EPOCHS}; do
      if [ "${epoch_count}" = "1" ]; then
        run_lrs="${ROB158_RUN_LRS:-${RUN_EPOCH1_LRS}}"
      elif [ "${epoch_count}" = "5" ]; then
        run_lrs="${ROB158_RUN_LRS:-${RUN_EPOCH5_LRS}}"
      else
        echo "Unsupported epoch count for ROB-158: ${epoch_count}" >&2
        exit 1
      fi
      for lr in ${run_lrs}; do
        tag="${dataset_tag}_epoch${epoch_count}_lr${lr}${repeat_suffix}"
        config="${RESULT_ROOT}/UFMR/configs/${tag}.yaml"
        save_path="${RESULT_ROOT}/UFMR/${tag}.txt"
        if [ ! -f "${config}" ]; then
          echo "Missing generated config: ${config}" >&2
          exit 1
        fi
        if [ "${FORCE_RERUN:-0}" != "1" ] && [ -f "${save_path}" ] && grep -q "Updated_WER:" "${save_path}"; then
          echo "[rob158] skipping completed UFMR/${tag}: ${save_path}"
          continue
        fi
        if [ "${FORCE_RERUN:-0}" = "1" ]; then
          rm -f "${save_path}"
        fi
        args=(python "${EVAL_SCRIPT}" --config "${config}")
        if [ -n "${ROB158_INDEXES:-}" ]; then
          args+=(--indexes ${ROB158_INDEXES})
        fi
        if [ "${ROB158_DONT_SAVE:-0}" = "1" ]; then
          args+=(--dont_save)
        fi
        echo "[rob158] running UFMR/${tag}: ${args[*]}"
        "${args[@]}"
      done
    done
  done
done

cd "${REPO_DIR}"
python3 scripts/summarize_rob108_test_policy_evals.py \
  --result-root "${RESULT_ROOT}" \
  --output-dir "${AGGREGATE_DIR}" \
  --datasets "${DATASETS}" \
  --methods "${METHODS}" \
  --repeats "${REPEATS}" \
  --epoch1-lrs "${EPOCH1_LRS}" \
  --epoch5-lrs "${EPOCH5_LRS}" \
  --csv-name rob158_ufmr_large_asr_eval.csv \
  --outcome-name ROB-158_OUTCOME.md \
  --title "ROB-158 UFMR Large-ASR Evaluation" \
  --note "UFMR test-split evals using the SAP-style 2048-seq-len 90M ASR checkpoint. The UFMR policy, datasets, repeats, candidate count, and LR/epoch cells match the ROB-108 UFMR repro setup so the intended variable is ASR model size."
echo "[rob158] finished"
