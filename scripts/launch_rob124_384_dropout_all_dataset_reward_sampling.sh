#!/usr/bin/env bash
# Callback-backed ROB-124 all-dataset eval for 384/dropout reward sampling.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-124}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_reward_sampling}"
LOG_PATH="${LOG_PATH:-${RESULT_ROOT}/logs/rob124_384_dropout_all_dataset_reward_sampling.log}"
SCREEN_LOG_PATH="${SCREEN_LOG_PATH:-${RESULT_ROOT}/logs/rob124_384_dropout_all_dataset_reward_sampling.screen.log}"
SCREEN_NAME="${SCREEN_NAME:-rob124-384-dropout-all-dataset-sampling}"
RUNNER_LABEL="${RUNNER_LABEL:-screen:${SCREEN_NAME}}"
REWARD_RANGE_LOW="${ROB124_ALLDATA_REWARD_RANGE_LOW:-0.5}"
REWARD_RANGE_HIGH="${ROB124_ALLDATA_REWARD_RANGE_HIGH:-1.0}"
REWARD_RANGE_ID="${ROB124_ALLDATA_REWARD_RANGE_ID:-0p5to1}"
REWARD_RANGE_LABEL="${ROB124_ALLDATA_REWARD_RANGE_LABEL:-[0.5, 1.0]}"
METHOD="${ROB124_ALLDATA_METHOD:-RewardConditionedMaskLMUniform${REWARD_RANGE_ID}}"
CONDITION="${ROB124_ALLDATA_CONDITION:-uniform_0.5_1.0}"
LABEL="${ROB124_ALLDATA_LABEL:-uniform sampled reward [0.5, 1.0]}"
CSV_NAME="${ROB124_ALLDATA_CSV_NAME:-rob124_384_dropout_all_dataset_reward_sampling.csv}"
OUTCOME_TITLE="${ROB124_ALLDATA_OUTCOME_TITLE:-ROB-124 384-Dropout All-Dataset Reward Sampling Eval}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt}"
ASR_CKPT="${ASR_CKPT:-/store/store5/data/acp21rjf_checkpoints/spotify/rotary_pos_6l_256d_seq_sched/n_seq_sched_2048_rp_1/step_105360.pt}"
MASK_VAE_CKPT="${MASK_VAE_CKPT:-/store/store5/data/acp21rjf_checkpoints/l2augment/models/bvae/bvae_USINGTHISFORNOW_2048gpu.pt}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/exp/exp4/acp21rjf/rob124-all-dataset-sampling-scratch}"
DATASETS="${ROB124_ALLDATA_DATASETS:-tedlium earnings22 chime6 rev16 TAL}"
EPOCHS="${ROB124_ALLDATA_EPOCHS:-1 5}"
LR="${ROB124_ALLDATA_LR:-1e-5}"
REPEATS="${ROB124_ALLDATA_REPEATS:-1}"
QUEUED_COMMAND="${QUEUED_COMMAND:-screen -L -Logfile ${SCREEN_LOG_PATH} -dmS ${SCREEN_NAME} bash -lc 'cd ${REPO_DIR} && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob124_384_dropout_all_dataset_reward_sampling.sh'}"
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"

on_exit() {
  status=$?
  set +e
  if [ "${ROB124_ALLDATA_DISABLE_CALLBACK:-0}" = "1" ]; then
    exit "${status}"
  fi
  if [ -z "${LINEAR_API_KEY:-}" ] && [ "${ROB124_ALLDATA_CALLBACK_DRY_RUN:-0}" != "1" ]; then
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
    --note "${CALLBACK_NOTE:-ROB-124 384/dropout all-dataset sampled reward ${REWARD_RANGE_LABEL} eval wrapper exited. Inspect OUTCOME.md under ${RESULT_ROOT}.}"
    --tail-lines "${CALLBACK_TAIL_LINES:-80}"
    --max-log-chars "${CALLBACK_MAX_LOG_CHARS:-6000}"
    --max-comment-chars "${CALLBACK_MAX_COMMENT_CHARS:-10000}"
  )
  if [ "${ROB124_ALLDATA_CALLBACK_CHECK_ONLY:-0}" = "1" ]; then
    callback_args+=(--check-only)
  fi
  if [ "${ROB124_ALLDATA_CALLBACK_DRY_RUN:-0}" = "1" ]; then
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

mkdir -p "$(dirname "${LOG_PATH}")" "${RESULT_ROOT}" "${SCRATCH_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

export REPO_DIR RESULT_ROOT LOG_PATH SCREEN_LOG_PATH SCREEN_NAME RUNNER_LABEL
export CHECKPOINT_PATH ASR_CKPT MASK_VAE_CKPT SCRATCH_ROOT QUEUED_COMMAND GIT_BRANCH GIT_COMMIT
export TMPDIR="${SCRATCH_ROOT}/tmp"
export WANDB_DIR="${RESULT_ROOT}/wandb"
export WANDB_CACHE_DIR="${SCRATCH_ROOT}/wandb-cache"
export WANDB_CONFIG_DIR="${SCRATCH_ROOT}/wandb-config"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/xdg-cache"
export MPLCONFIGDIR="${SCRATCH_ROOT}/matplotlib"
mkdir -p "${TMPDIR}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"

echo "[rob124-alldata] branch=${GIT_BRANCH}"
echo "[rob124-alldata] commit=${GIT_COMMIT}"
echo "[rob124-alldata] host=$(hostname)"
echo "[rob124-alldata] result_root=${RESULT_ROOT}"
echo "[rob124-alldata] checkpoint=${CHECKPOINT_PATH}"
echo "[rob124-alldata] asr_ckpt=${ASR_CKPT}"
echo "[rob124-alldata] mask_vae_ckpt=${MASK_VAE_CKPT}"
echo "[rob124-alldata] reward_range=${REWARD_RANGE_LABEL}"
echo "[rob124-alldata] method=${METHOD}"
echo "[rob124-alldata] datasets=${DATASETS}"
echo "[rob124-alldata] epochs=${EPOCHS}"
echo "[rob124-alldata] lr=${LR}"
echo "[rob124-alldata] repeats=${REPEATS}"
echo "[rob124-alldata] log_path=${LOG_PATH}"
echo "[rob124-alldata] screen_log_path=${SCREEN_LOG_PATH}"
echo "[rob124-alldata] queued_command=${QUEUED_COMMAND}"
echo "[rob124-alldata] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[rob124-alldata] tmpdir=${TMPDIR}"

bash -ic 'python - <<'"'"'PY'"'"'
import sys
import torch

print("[rob124-alldata] python_executable=" + sys.executable)
print("[rob124-alldata] python_version=" + sys.version.split()[0])
print("[rob124-alldata] torch_version=" + torch.__version__)
print("[rob124-alldata] cuda_available=" + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("[rob124-alldata] cuda_device=" + torch.cuda.get_device_name(0))
PY'

if [ "${ROB124_ALLDATA_CALLBACK_ONLY:-0}" = "1" ]; then
  echo "[rob124-alldata] callback-only smoke path requested; exiting before config generation."
  exit 0
fi

for required_path in "${CHECKPOINT_PATH}" "${ASR_CKPT}" "${MASK_VAE_CKPT}"; do
  if [ ! -f "${required_path}" ]; then
    echo "Missing required checkpoint: ${required_path}" >&2
    exit 1
  fi
done

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

python3 - "${RESULT_ROOT}" "${ASR_CKPT}" "${CHECKPOINT_PATH}" "${MASK_VAE_CKPT}" "${LINEAR_ISSUE}" "${DATASETS}" "${EPOCHS}" "${LR}" "${REPEATS}" "${REWARD_RANGE_LOW}" "${REWARD_RANGE_HIGH}" "${METHOD}" "${REWARD_RANGE_ID}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
asr_ckpt = sys.argv[2]
policy_ckpt = sys.argv[3]
mask_vae_ckpt = sys.argv[4]
linear_issue = sys.argv[5]
dataset_tags = tuple(sys.argv[6].split())
epochs = tuple(int(item) for item in sys.argv[7].split())
lr = sys.argv[8]
repeats = tuple(int(item) for item in sys.argv[9].split())
reward_low = sys.argv[10]
reward_high = sys.argv[11]
method = sys.argv[12]
reward_range_id = sys.argv[13]

datasets = {
    "tedlium": ("tedlium", "test"),
    "earnings22": ("earnings22", "test"),
    "chime6": ("chime6", "test"),
    "rev16": ("rev16", "test"),
    "TAL": ("this_american_life", "test"),
}

(root / method / "configs").mkdir(parents=True, exist_ok=True)
for dataset_tag in dataset_tags:
    if dataset_tag not in datasets:
        raise ValueError(f"Unknown dataset tag: {dataset_tag}")
    dataset, split = datasets[dataset_tag]
    for repeat in repeats:
        seed = 123456 + repeat - 1
        repeat_suffix = "" if repeat == 1 else f"_repeat{repeat}"
        for epoch_count in epochs:
            tag = f"{dataset_tag}_{split}_epoch{epoch_count}_lr{lr}{repeat_suffix}"
            save_path = root / method / f"{tag}.txt"
            config_path = root / method / "configs" / f"{tag}.yaml"
            config_path.write_text(
                f"""checkpointing:
  asr_model: {asr_ckpt}

training:
  device: cuda
  random_seed: {seed}
  batch_size: 1
  epochs: 500
  model_save_path: {policy_ckpt}
  tmp_model_save_path: {policy_ckpt}
  prefetch_factor: null
  num_workers: 0

evaluation:
  id: {linear_issue}-{dataset_tag}-{split}-384d-dropout0p1-uniform{reward_range_id}-epoch{epoch_count}-lr{lr}-repeat{repeat}
  dataset: {dataset}
  split: {split}
  rollout_fn: multistep
  use_cer: false
  epochs: {epoch_count}
  augmentation_config:
    sample: true
    seed: {seed}
  optim_args:
    lr: {lr}
  save_path: {save_path}

policy:
  lr: 1e-3
  class: RewardConditionedMaskLM
  config:
    hidden_dim: 384
    dropout: 0.1
    default_conditioning_reward: 1.0
    conditioning_reward_range: [{reward_low}, {reward_high}]
    reward_encoder: timestep
    sample_generation: true
    mask_vae_state_dict_path: {mask_vae_ckpt}
    mask_vae_config:
      latent_dim: 128
      codebook_size: 2048
      use_vq: true
""",
                encoding="utf-8",
            )
            print(f"[rob124-alldata] wrote config {config_path}")
PY

python3 scripts/summarize_rob124_384_dropout_all_dataset_reward_sampling.py \
  --result-root "${RESULT_ROOT}" \
  --datasets "${DATASETS}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --repeats "${REPEATS}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --command "${QUEUED_COMMAND}" \
  --branch "${GIT_BRANCH}" \
  --commit "${GIT_COMMIT}" \
  --log-path "${LOG_PATH}" \
  --screen-log-path "${SCREEN_LOG_PATH}" \
  --method "${METHOD}" \
  --condition "${CONDITION}" \
  --label "${LABEL}" \
  --reward-range "${REWARD_RANGE_LOW}:${REWARD_RANGE_HIGH}" \
  --reward-control "${REWARD_RANGE_LABEL}" \
  --csv-name "${CSV_NAME}" \
  --title "${OUTCOME_TITLE}"

if [ "${ROB124_ALLDATA_CONFIG_ONLY:-0}" = "1" ]; then
  echo "[rob124-alldata] config-only smoke path requested; exiting before eval."
  exit 0
fi

RUN_DATASETS="${ROB124_ALLDATA_RUN_DATASETS:-${DATASETS}}"
RUN_EPOCHS="${ROB124_ALLDATA_RUN_EPOCHS:-${EPOCHS}}"
RUN_REPEATS="${ROB124_ALLDATA_RUN_REPEATS:-${REPEATS}}"
EVAL_SCRIPT="${ROB124_ALLDATA_EVAL_SCRIPT:-eval.py}"

if [ "${ROB124_ALLDATA_SMOKE:-0}" = "1" ]; then
  RUN_DATASETS="${ROB124_ALLDATA_SMOKE_DATASETS:-tedlium}"
  RUN_EPOCHS="${ROB124_ALLDATA_SMOKE_EPOCHS:-1}"
  RUN_REPEATS="${ROB124_ALLDATA_SMOKE_REPEATS:-1}"
  ROB124_ALLDATA_INDEXES="${ROB124_ALLDATA_INDEXES:-0}"
  ROB124_ALLDATA_DONT_SAVE="${ROB124_ALLDATA_DONT_SAVE:-1}"
  echo "[rob124-alldata] smoke mode: datasets=${RUN_DATASETS}; epochs=${RUN_EPOCHS}; repeats=${RUN_REPEATS}; indexes=${ROB124_ALLDATA_INDEXES}; dont_save=${ROB124_ALLDATA_DONT_SAVE}"
fi

cd "${REPO_DIR}/exp"
for dataset_tag in ${RUN_DATASETS}; do
  for repeat in ${RUN_REPEATS}; do
    repeat_suffix=""
    if [ "${repeat}" != "1" ]; then
      repeat_suffix="_repeat${repeat}"
    fi
    for epoch_count in ${RUN_EPOCHS}; do
      tag="${dataset_tag}_test_epoch${epoch_count}_lr${LR}${repeat_suffix}"
      config="${RESULT_ROOT}/${METHOD}/configs/${tag}.yaml"
      save_path="${RESULT_ROOT}/${METHOD}/${tag}.txt"
      if [ ! -f "${config}" ]; then
        echo "Missing generated config: ${config}" >&2
        exit 1
      fi
      if [ "${FORCE_RERUN:-0}" != "1" ] && [ -f "${save_path}" ] && grep -q "Updated_WER:" "${save_path}"; then
        echo "[rob124-alldata] skipping completed ${dataset_tag}/epoch${epoch_count}: ${save_path}"
        continue
      fi
      if [ "${FORCE_RERUN:-0}" = "1" ]; then
        rm -f "${save_path}"
      fi
      CONFIG_TO_RUN="${config}" EVAL_SCRIPT="${EVAL_SCRIPT}" bash -ic '
        cd "$REPO_DIR/exp"
        export PYTHONPATH="$REPO_DIR:$REPO_DIR/exp:/exp/exp4/acp21rjf/long-context-asr:/exp/exp4/acp21rjf/language_modelling${PYTHONPATH:+:$PYTHONPATH}"
        args=(python "$EVAL_SCRIPT" --config "$CONFIG_TO_RUN")
        if [ -n "${ROB124_ALLDATA_INDEXES:-}" ]; then
          read -r -a indexes <<< "$ROB124_ALLDATA_INDEXES"
          args+=(--indexes "${indexes[@]}")
        fi
        if [ "${ROB124_ALLDATA_DONT_SAVE:-0}" = "1" ]; then
          args+=(--dont_save)
        fi
        echo "[rob124-alldata] eval command: ${args[*]}"
        "${args[@]}"
      '
    done
  done
done

cd "${REPO_DIR}"
python3 scripts/summarize_rob124_384_dropout_all_dataset_reward_sampling.py \
  --result-root "${RESULT_ROOT}" \
  --datasets "${DATASETS}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --repeats "${REPEATS}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --command "${QUEUED_COMMAND}" \
  --branch "${GIT_BRANCH}" \
  --commit "${GIT_COMMIT}" \
  --log-path "${LOG_PATH}" \
  --screen-log-path "${SCREEN_LOG_PATH}" \
  --method "${METHOD}" \
  --condition "${CONDITION}" \
  --label "${LABEL}" \
  --reward-range "${REWARD_RANGE_LOW}:${REWARD_RANGE_HIGH}" \
  --reward-control "${REWARD_RANGE_LABEL}" \
  --csv-name "${CSV_NAME}" \
  --title "${OUTCOME_TITLE}"

echo "[rob124-alldata] finished"
