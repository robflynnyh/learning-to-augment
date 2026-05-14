#!/usr/bin/env bash
# Queue-safe ROB-80 TED-LIUM dev LR sweep for no-audio CMultiStepVQLM.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-80}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/exp/results/repro/sweeps/no_audio_cmultistep_vqlm}"
LOG_PATH="${LOG_PATH:-${RESULT_ROOT}/logs/rob80_tedlium_noaudio_cmultistep_sweep.log}"
SCREEN_NAME="${SCREEN_NAME:-rob80_tedlium_noaudio_cmultistep_sweep}"
RUNNER_LABEL="${RUNNER_LABEL:-screen:${SCREEN_NAME}}"
QUEUED_COMMAND="${QUEUED_COMMAND:-/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob80_tedlium_noaudio_cmultistep_sweep.sh}"
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"
ASR_CKPT="${ASR_CKPT:-/store/store5/data/acp21rjf_checkpoints/l2augment/asr/step_105360.pt}"
POLICY_CKPT="${ROB80_NOAUDIO_CMULTISTEP_CKPT:-/store/store5/data/acp21rjf_checkpoints/l2augment/models/CMultiStepMLM/no_audio_modelsignals.pt}"
MASK_VAE_CKPT="${ROB80_MASK_VAE_CKPT:-/store/store5/data/acp21rjf_checkpoints/l2augment/models/bvae/bvae_USINGTHISFORNOW_2048gpu.pt}"
DECODER_LAYERS="${ROB80_NOAUDIO_CMULTISTEP_DECODER_LAYERS:-4}"
DATASET="${ROB80_DATASET:-tedlium}"
SPLIT="${ROB80_SPLIT:-dev}"
TAG_PREFIX="${ROB80_TAG_PREFIX:-tedlium_dev}"
METHOD="${ROB80_METHOD:-CMultiStepVQLM}"
LRS="${ROB80_LRS:-5e-6 1e-5 2e-5}"
EPOCHS="${ROB80_EPOCHS:-1 5}"
TITLE="${ROB80_TITLE:-ROB-80 TED-LIUM Dev No-Audio CMultiStepVQLM LR Sweep}"
NOTE="${ROB80_NOTE:-No-audio CMultiStepVQLM uses \`ConditionalMultiStepMaskGenerator\` with \`condition_on_audio: false\`, so generation is conditioned on reward/signal inputs and recurrent mask history, not raw audio.}"
CSV_NAME="${ROB80_CSV_NAME:-rob80_tedlium_noaudio_cmultistep_sweep.csv}"
OUTCOME_NAME="${ROB80_OUTCOME_NAME:-ROB-80_NOAUDIO_CMULTISTEP_OUTCOME.md}"

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
    --note "${CALLBACK_NOTE:-ROB-80 TED-LIUM dev no-audio CMultiStepVQLM LR sweep wrapper exited. See exp/results/repro/sweeps/no_audio_cmultistep_vqlm/ROB-80_NOAUDIO_CMULTISTEP_OUTCOME.md for the result table when complete.}"
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

echo "[rob80-noaudio-cmultistep] branch=${GIT_BRANCH}"
echo "[rob80-noaudio-cmultistep] commit=${GIT_COMMIT}"
echo "[rob80-noaudio-cmultistep] result_root=${RESULT_ROOT}"
echo "[rob80-noaudio-cmultistep] asr_ckpt=${ASR_CKPT}"
echo "[rob80-noaudio-cmultistep] policy_ckpt=${POLICY_CKPT}"
echo "[rob80-noaudio-cmultistep] mask_vae_ckpt=${MASK_VAE_CKPT}"
echo "[rob80-noaudio-cmultistep] decoder_layers=${DECODER_LAYERS}"
echo "[rob80-noaudio-cmultistep] dataset=${DATASET}"
echo "[rob80-noaudio-cmultistep] split=${SPLIT}"
echo "[rob80-noaudio-cmultistep] tag_prefix=${TAG_PREFIX}"
echo "[rob80-noaudio-cmultistep] lrs=${LRS}"
echo "[rob80-noaudio-cmultistep] epochs=${EPOCHS}"
echo "[rob80-noaudio-cmultistep] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

if [ "${ROB80_CALLBACK_ONLY:-0}" = "1" ]; then
  echo "[rob80-noaudio-cmultistep] callback-only smoke path requested; exiting before eval."
  exit 0
fi

for required_path in "${ASR_CKPT}" "${POLICY_CKPT}" "${MASK_VAE_CKPT}"; do
  if [ ! -f "${required_path}" ]; then
    echo "Missing required checkpoint: ${required_path}" >&2
    exit 1
  fi
done

export L2A_EARNINGS22_DIR="${L2A_EARNINGS22_DIR:-/store/store4/data/earnings-22}"
export L2A_TEDLIUM3_LEGACY_DIR="${L2A_TEDLIUM3_LEGACY_DIR:-/store/store4/data/TEDLIUM_release-3/legacy/}"
export L2A_REV16_DIR="${L2A_REV16_DIR:-/store/store4/data/rev_benchmark}"
export L2A_CHIME6_DIR="${L2A_CHIME6_DIR:-/store/store4/data/chime6/}"

python3 - "${RESULT_ROOT}" "${ASR_CKPT}" "${POLICY_CKPT}" "${MASK_VAE_CKPT}" "${DECODER_LAYERS}" "${DATASET}" "${SPLIT}" "${TAG_PREFIX}" "${METHOD}" "${LRS}" "${EPOCHS}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
asr_ckpt = sys.argv[2]
policy_ckpt = sys.argv[3]
mask_vae_ckpt = sys.argv[4]
decoder_layers = int(sys.argv[5])
dataset = sys.argv[6]
split = sys.argv[7]
tag_prefix = sys.argv[8]
method = sys.argv[9]
lrs = tuple(sys.argv[10].split())
epochs = tuple(int(item) for item in sys.argv[11].split())

(root / method / "configs").mkdir(parents=True, exist_ok=True)
for epoch_count in epochs:
    for lr in lrs:
        tag = f"{tag_prefix}_epoch{epoch_count}_lr{lr}"
        save_path = root / method / f"{tag}.txt"
        config_path = root / method / "configs" / f"{tag}.yaml"
        config_path.write_text(
            f"""checkpointing:
  asr_model: {asr_ckpt}

training:
  device: 'cuda'
  random_seed: 1234
  batch_size: 1
  epochs: 100
  model_save_path: {policy_ckpt}
  tmp_model_save_path: {policy_ckpt}
  max_steps: 10000
  prefetch_factor: null
  num_workers: 0

evaluation:
  id: 'ROB-80-{dataset}-{split}-{method}-epoch{epoch_count}-lr{lr}'
  dataset: '{dataset}'
  split: '{split}'
  use_cer: false
  epochs: {epoch_count}
  optim_args:
    lr: {lr}
  save_path: {save_path}

policy:
  lr: 1e-3
  class: ConditionalMultiStepMaskGenerator
  config:
    codebook_size: 2048
    hidden_dim: 512
    embedding_dim: 512
    decoder_layers: {decoder_layers}
    default_conditioning_reward: 1.0
    condition_on_audio: false
    mask_vae_state_dict_path: {mask_vae_ckpt}
    mask_vae_config:
      latent_dim: 128
      codebook_size: 2048
      use_vq: true
"""
        )
        print(f"[rob80-noaudio-cmultistep] wrote config {config_path}")
PY

if [ "${ROB80_CONFIG_ONLY:-0}" = "1" ]; then
  echo "[rob80-noaudio-cmultistep] config-only smoke path requested; exiting before eval."
  exit 0
fi

source /store/store4/software/bin/anaconda3/etc/profile.d/conda.sh
conda activate /store/store4/software/bin/anaconda3/envs/flash_attn_pytorch2

export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

RUN_METHODS="${ROB80_METHODS:-${METHOD}}"
RUN_EPOCHS="${ROB80_RUN_EPOCHS:-${EPOCHS}}"
RUN_LRS="${ROB80_RUN_LRS:-${LRS}}"

if [ "${ROB80_SMOKE:-0}" = "1" ]; then
  RUN_METHODS="${ROB80_SMOKE_METHODS:-${METHOD}}"
  RUN_EPOCHS="${ROB80_SMOKE_EPOCHS:-1}"
  RUN_LRS="${ROB80_SMOKE_LRS:-5e-6}"
  ROB80_INDEXES="${ROB80_INDEXES:-0}"
  ROB80_DONT_SAVE="${ROB80_DONT_SAVE:-1}"
  echo "[rob80-noaudio-cmultistep] smoke mode: methods=${RUN_METHODS}; epochs=${RUN_EPOCHS}; lrs=${RUN_LRS}; indexes=${ROB80_INDEXES}; dont_save=${ROB80_DONT_SAVE}"
fi

cd "${REPO_DIR}/exp"
for method in ${RUN_METHODS}; do
  for epoch_count in ${RUN_EPOCHS}; do
    for lr in ${RUN_LRS}; do
      tag="${TAG_PREFIX}_epoch${epoch_count}_lr${lr}"
      config="${RESULT_ROOT}/${method}/configs/${tag}.yaml"
      save_path="${RESULT_ROOT}/${method}/${tag}.txt"
      if [ ! -f "${config}" ]; then
        echo "Missing generated config: ${config}" >&2
        exit 1
      fi
      if [ "${FORCE_RERUN:-0}" != "1" ] && [ -f "${save_path}" ] && grep -q "Updated_WER:" "${save_path}"; then
        echo "[rob80-noaudio-cmultistep] skipping completed ${method}/${tag}: ${save_path}"
        continue
      fi
      if [ "${FORCE_RERUN:-0}" = "1" ]; then
        rm -f "${save_path}"
      fi
      args=(python eval.py --config "${config}")
      if [ -n "${ROB80_INDEXES:-}" ]; then
        args+=(--indexes ${ROB80_INDEXES})
      fi
      if [ "${ROB80_DONT_SAVE:-0}" = "1" ]; then
        args+=(--dont_save)
      fi
      echo "[rob80-noaudio-cmultistep] running ${method}/${tag}: ${args[*]}"
      "${args[@]}"
    done
  done
done

cd "${REPO_DIR}"
python3 scripts/summarize_rob80_noaudio_cmultistep_sweep.py \
  --result-root "${RESULT_ROOT}" \
  --method "${METHOD}" \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --tag-prefix "${TAG_PREFIX}" \
  --epochs "${EPOCHS}" \
  --lrs "${LRS}" \
  --csv-name "${CSV_NAME}" \
  --outcome-name "${OUTCOME_NAME}" \
  --title "${TITLE}" \
  --note "${NOTE}"
echo "[rob80-noaudio-cmultistep] finished"
