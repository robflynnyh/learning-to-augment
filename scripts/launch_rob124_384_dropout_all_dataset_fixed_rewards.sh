#!/usr/bin/env bash
# Callback-backed ROB-124 all-dataset eval for fixed reward 1.0 and 0.0.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-124}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1}"
LOG_PATH="${LOG_PATH:-${RESULT_ROOT}/logs/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1.log}"
SCREEN_LOG_PATH="${SCREEN_LOG_PATH:-${RESULT_ROOT}/logs/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1.screen.log}"
SCREEN_NAME="${SCREEN_NAME:-rob124-384-dropout-fixed-rewards-0-and-1}"
RUNNER_LABEL="${RUNNER_LABEL:-screen:${SCREEN_NAME}}"
FIXED_REWARDS="${ROB124_ALLDATA_FIXED_REWARDS:-1.0 0.0}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt}"
ASR_CKPT="${ASR_CKPT:-/store/store5/data/acp21rjf_checkpoints/spotify/rotary_pos_6l_256d_seq_sched/n_seq_sched_2048_rp_1/step_105360.pt}"
MASK_VAE_CKPT="${MASK_VAE_CKPT:-/store/store5/data/acp21rjf_checkpoints/l2augment/models/bvae/bvae_USINGTHISFORNOW_2048gpu.pt}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/exp/exp4/acp21rjf/rob124-fixed-rewards-0-and-1-scratch}"
DATASETS="${ROB124_ALLDATA_DATASETS:-tedlium earnings22 chime6 rev16 TAL}"
EPOCHS="${ROB124_ALLDATA_EPOCHS:-1 5}"
LR="${ROB124_ALLDATA_LR:-1e-5}"
REPEATS="${ROB124_ALLDATA_REPEATS:-1}"
CSV_NAME="${ROB124_ALLDATA_CSV_NAME:-rob124_384_dropout_all_dataset_fixed_rewards_0_and_1.csv}"
OUTCOME_TITLE="${ROB124_ALLDATA_OUTCOME_TITLE:-ROB-124 384-Dropout All-Dataset Fixed Reward 0 And 1 Eval}"
QUEUED_COMMAND="${QUEUED_COMMAND:-screen -L -Logfile ${SCREEN_LOG_PATH} -dmS ${SCREEN_NAME} bash -lc 'cd ${REPO_DIR} && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob124_384_dropout_all_dataset_fixed_rewards.sh'}"
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
    --note "${CALLBACK_NOTE:-ROB-124 384/dropout all-dataset fixed reward 1.0 and 0.0 eval wrapper exited. Inspect OUTCOME.md under ${RESULT_ROOT}.}"
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

echo "[rob124-fixed] branch=${GIT_BRANCH}"
echo "[rob124-fixed] commit=${GIT_COMMIT}"
echo "[rob124-fixed] host=$(hostname)"
echo "[rob124-fixed] result_root=${RESULT_ROOT}"
echo "[rob124-fixed] checkpoint=${CHECKPOINT_PATH}"
echo "[rob124-fixed] asr_ckpt=${ASR_CKPT}"
echo "[rob124-fixed] mask_vae_ckpt=${MASK_VAE_CKPT}"
echo "[rob124-fixed] fixed_rewards=${FIXED_REWARDS}"
echo "[rob124-fixed] datasets=${DATASETS}"
echo "[rob124-fixed] epochs=${EPOCHS}"
echo "[rob124-fixed] lr=${LR}"
echo "[rob124-fixed] repeats=${REPEATS}"
echo "[rob124-fixed] log_path=${LOG_PATH}"
echo "[rob124-fixed] screen_log_path=${SCREEN_LOG_PATH}"
echo "[rob124-fixed] queued_command=${QUEUED_COMMAND}"
echo "[rob124-fixed] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[rob124-fixed] tmpdir=${TMPDIR}"

bash -ic 'python - <<'"'"'PY'"'"'
import sys
import torch

print("[rob124-fixed] python_executable=" + sys.executable)
print("[rob124-fixed] python_version=" + sys.version.split()[0])
print("[rob124-fixed] torch_version=" + torch.__version__)
print("[rob124-fixed] cuda_available=" + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("[rob124-fixed] cuda_device=" + torch.cuda.get_device_name(0))
PY'

if [ "${ROB124_ALLDATA_CALLBACK_ONLY:-0}" = "1" ]; then
  echo "[rob124-fixed] callback-only smoke path requested; exiting before config generation."
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

python3 - "${RESULT_ROOT}" "${ASR_CKPT}" "${CHECKPOINT_PATH}" "${MASK_VAE_CKPT}" "${LINEAR_ISSUE}" "${DATASETS}" "${EPOCHS}" "${LR}" "${REPEATS}" "${FIXED_REWARDS}" <<'PY'
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
fixed_rewards = tuple(sys.argv[10].split())

datasets = {
    "tedlium": ("tedlium", "test"),
    "earnings22": ("earnings22", "test"),
    "chime6": ("chime6", "test"),
    "rev16": ("rev16", "test"),
    "TAL": ("this_american_life", "test"),
}


def reward_token(reward: str) -> str:
    value = float(reward)
    if value == 0.0:
        return "0"
    if value == 1.0:
        return "1"
    return reward.replace(".", "p").replace("-", "m")


for reward in fixed_rewards:
    token = reward_token(reward)
    method = f"RewardConditionedMaskLMReward{token}"
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
  id: {linear_issue}-{dataset_tag}-{split}-384d-dropout0p1-reward{token}-epoch{epoch_count}-lr{lr}-repeat{repeat}
  dataset: {dataset}
  split: {split}
  rollout_fn: multistep
  use_cer: false
  epochs: {epoch_count}
  augmentation_config:
    conditioning_reward: {reward}
    sample: true
  optim_args:
    lr: {lr}
  save_path: {save_path}

policy:
  lr: 1e-3
  class: RewardConditionedMaskLM
  config:
    hidden_dim: 384
    dropout: 0.1
    default_conditioning_reward: {reward}
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
                print(f"[rob124-fixed] wrote config {config_path}")
PY

python3 scripts/summarize_rob124_384_dropout_all_dataset_fixed_rewards.py \
  --result-root "${RESULT_ROOT}" \
  --fixed-rewards "${FIXED_REWARDS}" \
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
  --csv-name "${CSV_NAME}" \
  --title "${OUTCOME_TITLE}"

if [ "${ROB124_ALLDATA_CONFIG_ONLY:-0}" = "1" ]; then
  echo "[rob124-fixed] config-only smoke path requested; exiting before eval."
  exit 0
fi

RUN_REWARDS="${ROB124_ALLDATA_RUN_FIXED_REWARDS:-${FIXED_REWARDS}}"
RUN_DATASETS="${ROB124_ALLDATA_RUN_DATASETS:-${DATASETS}}"
RUN_EPOCHS="${ROB124_ALLDATA_RUN_EPOCHS:-${EPOCHS}}"
RUN_REPEATS="${ROB124_ALLDATA_RUN_REPEATS:-${REPEATS}}"
EVAL_SCRIPT="${ROB124_ALLDATA_EVAL_SCRIPT:-eval.py}"

if [ "${ROB124_ALLDATA_SMOKE:-0}" = "1" ]; then
  RUN_REWARDS="${ROB124_ALLDATA_SMOKE_FIXED_REWARDS:-1.0}"
  RUN_DATASETS="${ROB124_ALLDATA_SMOKE_DATASETS:-tedlium}"
  RUN_EPOCHS="${ROB124_ALLDATA_SMOKE_EPOCHS:-1}"
  RUN_REPEATS="${ROB124_ALLDATA_SMOKE_REPEATS:-1}"
  ROB124_ALLDATA_INDEXES="${ROB124_ALLDATA_INDEXES:-0}"
  ROB124_ALLDATA_DONT_SAVE="${ROB124_ALLDATA_DONT_SAVE:-1}"
  echo "[rob124-fixed] smoke mode: rewards=${RUN_REWARDS}; datasets=${RUN_DATASETS}; epochs=${RUN_EPOCHS}; repeats=${RUN_REPEATS}; indexes=${ROB124_ALLDATA_INDEXES}; dont_save=${ROB124_ALLDATA_DONT_SAVE}"
fi

cd "${REPO_DIR}/exp"
for reward in ${RUN_REWARDS}; do
  token="${reward//./p}"
  token="${token//-/m}"
  if [ "${reward}" = "0.0" ]; then
    token="0"
  elif [ "${reward}" = "1.0" ]; then
    token="1"
  fi
  method="RewardConditionedMaskLMReward${token}"
  for dataset_tag in ${RUN_DATASETS}; do
    for repeat in ${RUN_REPEATS}; do
      repeat_suffix=""
      if [ "${repeat}" != "1" ]; then
        repeat_suffix="_repeat${repeat}"
      fi
      for epoch_count in ${RUN_EPOCHS}; do
        tag="${dataset_tag}_test_epoch${epoch_count}_lr${LR}${repeat_suffix}"
        config="${RESULT_ROOT}/${method}/configs/${tag}.yaml"
        save_path="${RESULT_ROOT}/${method}/${tag}.txt"
        if [ ! -f "${config}" ]; then
          echo "Missing generated config: ${config}" >&2
          exit 1
        fi
        if [ "${FORCE_RERUN:-0}" != "1" ] && [ -f "${save_path}" ] && grep -q "Updated_WER:" "${save_path}"; then
          echo "[rob124-fixed] skipping completed reward${reward}/${dataset_tag}/epoch${epoch_count}: ${save_path}"
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
          echo "[rob124-fixed] eval command: ${args[*]}"
          "${args[@]}"
        '
      done
    done
  done
done

cd "${REPO_DIR}"
python3 scripts/summarize_rob124_384_dropout_all_dataset_fixed_rewards.py \
  --result-root "${RESULT_ROOT}" \
  --fixed-rewards "${FIXED_REWARDS}" \
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
  --csv-name "${CSV_NAME}" \
  --title "${OUTCOME_TITLE}"

echo "[rob124-fixed] finished"
