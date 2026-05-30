#!/usr/bin/env bash
# Callback-backed ROB-132 audio SSL fixed-reward eval on selected test sets.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-132}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_tedlium_earnings22}"
LOG_PATH="${LOG_PATH:-${RESULT_ROOT}/logs/rob132_audio_ssl_self_train_test_sets_fixed_rewards.log}"
SCREEN_LOG_PATH="${SCREEN_LOG_PATH:-${RESULT_ROOT}/logs/rob132_audio_ssl_self_train_test_sets_fixed_rewards.screen.log}"
SCREEN_NAME="${SCREEN_NAME:-rob132-audio-ssl-testsets-fixed-rewards}"
RUNNER_LABEL="${RUNNER_LABEL:-screen:${SCREEN_NAME}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/audio_ssl_hubert_base_tedlium_per_utterance_transformer384_dropout0p1_500ep_lr1e3.pt}"
ASR_CKPT="${ASR_CKPT:-/store/store5/data/acp21rjf_checkpoints/spotify/rotary_pos_6l_256d_seq_sched/n_seq_sched_2048_rp_1/step_105360.pt}"
MASK_VAE_CKPT="${MASK_VAE_CKPT:-/store/store5/data/acp21rjf_checkpoints/l2augment/models/bvae/bvae_USINGTHISFORNOW_2048gpu.pt}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/exp/exp4/acp21rjf/rob132-audio-ssl-scratch}"
DATASETS="${ROB132_TESTSETS_DATASETS:-tedlium earnings22}"
FIXED_REWARDS="${ROB132_TESTSETS_FIXED_REWARDS:-1.0 0.0}"
EPOCHS="${ROB132_TESTSETS_EPOCHS:-1 5}"
LR="${ROB132_TESTSETS_LR:-1e-5}"
CSV_NAME="${ROB132_TESTSETS_CSV_NAME:-rob132_audio_ssl_self_train_test_sets_fixed_rewards.csv}"
QUEUED_COMMAND="${QUEUED_COMMAND:-screen -L -Logfile ${SCREEN_LOG_PATH} -dmS ${SCREEN_NAME} bash -lc 'cd ${REPO_DIR} && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob132_audio_ssl_self_train_test_sets.sh'}"
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"

on_exit() {
  status=$?
  set +e
  if [ "${ROB132_TESTSETS_DISABLE_CALLBACK:-0}" = "1" ]; then
    exit "${status}"
  fi
  if [ -z "${LINEAR_API_KEY:-}" ] && [ "${ROB132_TESTSETS_CALLBACK_DRY_RUN:-0}" != "1" ]; then
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
    --note "${CALLBACK_NOTE:-ROB-132 audio SSL test fixed-reward self-training eval wrapper exited. Inspect OUTCOME.md under ${RESULT_ROOT}, then post the result summary.}"
    --tail-lines "${CALLBACK_TAIL_LINES:-80}"
    --max-log-chars "${CALLBACK_MAX_LOG_CHARS:-6000}"
    --max-comment-chars "${CALLBACK_MAX_COMMENT_CHARS:-10000}"
  )
  if [ "${ROB132_TESTSETS_CALLBACK_CHECK_ONLY:-0}" = "1" ]; then
    callback_args+=(--check-only)
  fi
  if [ "${ROB132_TESTSETS_CALLBACK_DRY_RUN:-0}" = "1" ]; then
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
export TORCH_HOME="${SCRATCH_ROOT}/torch"
export HF_HOME="${SCRATCH_ROOT}/huggingface"
export WANDB_DIR="${RESULT_ROOT}/wandb"
export WANDB_CACHE_DIR="${SCRATCH_ROOT}/wandb-cache"
export WANDB_CONFIG_DIR="${SCRATCH_ROOT}/wandb-config"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/xdg-cache"
export MPLCONFIGDIR="${SCRATCH_ROOT}/matplotlib"
mkdir -p "${TMPDIR}" "${TORCH_HOME}" "${HF_HOME}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"

echo "[rob132-testsets] branch=${GIT_BRANCH}"
echo "[rob132-testsets] commit=${GIT_COMMIT}"
echo "[rob132-testsets] host=$(hostname)"
echo "[rob132-testsets] result_root=${RESULT_ROOT}"
echo "[rob132-testsets] checkpoint=${CHECKPOINT_PATH}"
echo "[rob132-testsets] asr_ckpt=${ASR_CKPT}"
echo "[rob132-testsets] mask_vae_ckpt=${MASK_VAE_CKPT}"
echo "[rob132-testsets] datasets=${DATASETS}"
echo "[rob132-testsets] fixed_rewards=${FIXED_REWARDS}"
echo "[rob132-testsets] epochs=${EPOCHS}"
echo "[rob132-testsets] lr=${LR}"
echo "[rob132-testsets] log_path=${LOG_PATH}"
echo "[rob132-testsets] screen_log_path=${SCREEN_LOG_PATH}"
echo "[rob132-testsets] queued_command=${QUEUED_COMMAND}"
echo "[rob132-testsets] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[rob132-testsets] tmpdir=${TMPDIR}"
echo "[rob132-testsets] torch_home=${TORCH_HOME}"

bash -ic 'python - <<'"'"'PY'"'"'
import sys
import torch
import torchaudio

print("[rob132-testsets] python_executable=" + sys.executable)
print("[rob132-testsets] python_version=" + sys.version.split()[0])
print("[rob132-testsets] torch_version=" + torch.__version__)
print("[rob132-testsets] torchaudio_version=" + torchaudio.__version__)
print("[rob132-testsets] cuda_available=" + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("[rob132-testsets] cuda_device=" + torch.cuda.get_device_name(0))
PY'

if [ "${ROB132_TESTSETS_CALLBACK_ONLY:-0}" = "1" ]; then
  echo "[rob132-testsets] callback-only smoke path requested; exiting before config generation."
  exit 0
fi

for required_path in "${CHECKPOINT_PATH}" "${ASR_CKPT}" "${MASK_VAE_CKPT}"; do
  if [ ! -f "${required_path}" ]; then
    echo "Missing required checkpoint: ${required_path}" >&2
    exit 1
  fi
done

export L2A_TEDLIUM3_LEGACY_DIR="${L2A_TEDLIUM3_LEGACY_DIR:-/store/store4/data/TEDLIUM_release-3/legacy/}"
export L2A_EARNINGS22_DIR="${L2A_EARNINGS22_DIR:-/store/store4/data/earnings-22}"
export L2A_REV16_DIR="${L2A_REV16_DIR:-/mnt/parscratch/users/acp21rjf/rev_benchmark}"
export L2A_TAL_DIR="${L2A_TAL_DIR:-/store/store5/data/this_american_life}"
export L2A_CHIME6_DIR="${L2A_CHIME6_DIR:-/mnt/parscratch/users/acp21rjf/chime6}"

for dataset_tag in ${DATASETS}; do
  case "${dataset_tag}" in
    tedlium)
      required_dir="${L2A_TEDLIUM3_LEGACY_DIR}"
      ;;
    earnings22)
      required_dir="${L2A_EARNINGS22_DIR}"
      ;;
    rev16)
      required_dir="${L2A_REV16_DIR}"
      ;;
    TAL|tal|this_american_life)
      required_dir="${L2A_TAL_DIR}"
      ;;
    chime6)
      required_dir="${L2A_CHIME6_DIR}"
      ;;
    *)
      echo "Unknown ROB-132 test-set dataset tag: ${dataset_tag}" >&2
      exit 1
      ;;
  esac
  if [ ! -d "${required_dir}" ]; then
    echo "Missing dataset directory for ${dataset_tag}: ${required_dir}" >&2
    exit 1
  fi
done

python3 - "${RESULT_ROOT}" "${ASR_CKPT}" "${CHECKPOINT_PATH}" "${MASK_VAE_CKPT}" "${LINEAR_ISSUE}" "${DATASETS}" "${FIXED_REWARDS}" "${EPOCHS}" "${LR}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
asr_ckpt = sys.argv[2]
policy_ckpt = sys.argv[3]
mask_vae_ckpt = sys.argv[4]
linear_issue = sys.argv[5]
dataset_tags = tuple(sys.argv[6].split())
fixed_rewards = tuple(sys.argv[7].split())
epochs = tuple(int(item) for item in sys.argv[8].split())
lr = sys.argv[9]

datasets = {
    "tedlium": ("tedlium", "test"),
    "earnings22": ("earnings22", "test"),
    "rev16": ("rev16", "test"),
    "TAL": ("this_american_life", "test"),
    "tal": ("this_american_life", "test"),
    "this_american_life": ("this_american_life", "test"),
    "chime6": ("chime6", "test"),
}


def reward_tag(value: str) -> str:
    return value.replace(".", "p").replace("-", "m")


for reward in fixed_rewards:
    method = f"AudioRewardConditionedMaskLMReward{reward_tag(reward)}"
    (root / method / "configs").mkdir(parents=True, exist_ok=True)
    for dataset_tag in dataset_tags:
        if dataset_tag not in datasets:
            raise ValueError(f"Unknown dataset tag: {dataset_tag}")
        dataset, split = datasets[dataset_tag]
        for epoch_count in epochs:
            tag = f"{dataset_tag}_{split}_reward{reward_tag(reward)}_epoch{epoch_count}_lr{lr}"
            save_path = root / method / f"{tag}.txt"
            config_path = root / method / "configs" / f"{tag}.yaml"
            config_path.write_text(
                f"""checkpointing:
  asr_model: {asr_ckpt}

training:
  device: cuda
  random_seed: 123456
  batch_size: 1
  epochs: 500
  model_save_path: {policy_ckpt}
  tmp_model_save_path: {policy_ckpt}
  prefetch_factor: null
  num_workers: 0

evaluation:
  id: {linear_issue}-{dataset_tag}-{split}-audio-ssl-transformer384-reward{reward_tag(reward)}-epoch{epoch_count}-lr{lr}
  dataset: {dataset}
  split: {split}
  rollout_fn: multistep
  use_cer: false
  epochs: {epoch_count}
  augmentation_config:
    conditioning_reward: {reward}
    sample: true
    seed: 123456
  optim_args:
    lr: {lr}
  save_path: {save_path}

dataset:
  ssl_bundle: HUBERT_BASE
  ssl_device: cuda
  tedlium_base: /store/store4/data/TEDLIUM_release-3/legacy

policy:
  lr: 1e-3
  class: AudioRewardConditionedMaskLM
  config:
    hidden_dim: 384
    ssl_dim: 768
    num_heads: 8
    decoder_layers: 4
    candidate_microbatch_size: 120
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
            print(f"[rob132-testsets] wrote config {config_path}")
PY

python3 scripts/summarize_rob132_audio_ssl_self_train_test_sets.py \
  --result-root "${RESULT_ROOT}" \
  --fixed-rewards "${FIXED_REWARDS}" \
  --datasets "${DATASETS}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --command "${QUEUED_COMMAND}" \
  --branch "${GIT_BRANCH}" \
  --commit "${GIT_COMMIT}" \
  --log-path "${LOG_PATH}" \
  --screen-log-path "${SCREEN_LOG_PATH}" \
  --csv-name "${CSV_NAME}"

if [ "${ROB132_TESTSETS_CONFIG_ONLY:-0}" = "1" ]; then
  echo "[rob132-testsets] config-only smoke path requested; exiting before eval."
  exit 0
fi

RUN_REWARDS="${ROB132_TESTSETS_RUN_FIXED_REWARDS:-${FIXED_REWARDS}}"
RUN_DATASETS="${ROB132_TESTSETS_RUN_DATASETS:-${DATASETS}}"
RUN_EPOCHS="${ROB132_TESTSETS_RUN_EPOCHS:-${EPOCHS}}"
EVAL_SCRIPT="${ROB132_TESTSETS_EVAL_SCRIPT:-eval.py}"

if [ "${ROB132_TESTSETS_SMOKE:-0}" = "1" ]; then
  RUN_REWARDS="${ROB132_TESTSETS_SMOKE_FIXED_REWARDS:-1.0}"
  RUN_DATASETS="${ROB132_TESTSETS_SMOKE_DATASETS:-tedlium}"
  RUN_EPOCHS="${ROB132_TESTSETS_SMOKE_EPOCHS:-1}"
  ROB132_TESTSETS_INDEXES="${ROB132_TESTSETS_INDEXES:-0}"
  ROB132_TESTSETS_DONT_SAVE="${ROB132_TESTSETS_DONT_SAVE:-1}"
  echo "[rob132-testsets] smoke mode: rewards=${RUN_REWARDS}; datasets=${RUN_DATASETS}; epochs=${RUN_EPOCHS}; indexes=${ROB132_TESTSETS_INDEXES}; dont_save=${ROB132_TESTSETS_DONT_SAVE}"
fi

cd "${REPO_DIR}/exp"
for reward in ${RUN_REWARDS}; do
  reward_tag="${reward//./p}"
  reward_tag="${reward_tag//-/m}"
  method="AudioRewardConditionedMaskLMReward${reward_tag}"
  for dataset_tag in ${RUN_DATASETS}; do
    for epoch_count in ${RUN_EPOCHS}; do
      tag="${dataset_tag}_test_reward${reward_tag}_epoch${epoch_count}_lr${LR}"
      config="${RESULT_ROOT}/${method}/configs/${tag}.yaml"
      save_path="${RESULT_ROOT}/${method}/${tag}.txt"
      if [ ! -f "${config}" ]; then
        echo "Missing generated config: ${config}" >&2
        exit 1
      fi
      if [ "${FORCE_RERUN:-0}" != "1" ] && [ -f "${save_path}" ] && grep -q "Updated_WER:" "${save_path}"; then
        echo "[rob132-testsets] skipping completed reward=${reward}/${dataset_tag}/epoch${epoch_count}: ${save_path}"
        continue
      fi
      if [ "${FORCE_RERUN:-0}" = "1" ]; then
        rm -f "${save_path}"
      fi
      CONFIG_TO_RUN="${config}" EVAL_SCRIPT="${EVAL_SCRIPT}" bash -ic '
        cd "$REPO_DIR/exp"
        export PYTHONPATH="$REPO_DIR:$REPO_DIR/exp:/exp/exp4/acp21rjf/long-context-asr:/exp/exp4/acp21rjf/language_modelling${PYTHONPATH:+:$PYTHONPATH}"
        args=(python "$EVAL_SCRIPT" --config "$CONFIG_TO_RUN")
        if [ -n "${ROB132_TESTSETS_INDEXES:-}" ]; then
          read -r -a indexes <<< "$ROB132_TESTSETS_INDEXES"
          args+=(--indexes "${indexes[@]}")
        fi
        if [ -n "${ROB132_TESTSETS_MAX_STEPS:-}" ]; then
          args+=(--max_steps "$ROB132_TESTSETS_MAX_STEPS")
        fi
        if [ "${ROB132_TESTSETS_DONT_SAVE:-0}" = "1" ]; then
          args+=(--dont_save)
        fi
        echo "[rob132-testsets] eval command: ${args[*]}"
        "${args[@]}"
      '
    done
  done
done

cd "${REPO_DIR}"
python3 scripts/summarize_rob132_audio_ssl_self_train_test_sets.py \
  --result-root "${RESULT_ROOT}" \
  --fixed-rewards "${FIXED_REWARDS}" \
  --datasets "${DATASETS}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --command "${QUEUED_COMMAND}" \
  --branch "${GIT_BRANCH}" \
  --commit "${GIT_COMMIT}" \
  --log-path "${LOG_PATH}" \
  --screen-log-path "${SCREEN_LOG_PATH}" \
  --csv-name "${CSV_NAME}"

echo "[rob132-testsets] finished"
