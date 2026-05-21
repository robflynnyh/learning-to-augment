#!/usr/bin/env bash
# Queue-safe ROB-82 TED-LIUM dev LR sweep for UVQLM.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-82}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/exp/results/repro/sweeps/uvqlm}"
LOG_PATH="${LOG_PATH:-${RESULT_ROOT}/logs/rob82_tedlium_uvqlm_sweep.log}"
SCREEN_NAME="${SCREEN_NAME:-rob82_tedlium_uvqlm_sweep}"
RUNNER_LABEL="${RUNNER_LABEL:-screen:${SCREEN_NAME}}"
QUEUED_COMMAND="${QUEUED_COMMAND:-/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob82_tedlium_uvqlm_sweep.sh}"
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"
ASR_CKPT="${ASR_CKPT:-/store/store5/data/acp21rjf_checkpoints/spotify/rotary_pos_6l_256d_seq_sched/n_seq_sched_2048_rp_1/step_105360.pt}"
UVQLM_CKPT="${ROB82_UVQLM_CKPT:-/store/store5/data/acp21rjf_checkpoints/l2augment/models/UMLM/modelgpu.pt}"
MASK_VAE_CKPT="${ROB82_MASK_VAE_CKPT:-/store/store5/data/acp21rjf_checkpoints/l2augment/models/bvae/bvae_USINGTHISFORNOW_2048gpu.pt}"
METHOD="${ROB82_METHOD:-UVQLM}"
LRS="${ROB82_LRS:-5e-6 1e-5 2e-5}"
EPOCHS="${ROB82_EPOCHS:-1 5}"
REPEATS="${ROB82_REPEATS:-1 2}"

on_exit() {
  status=$?
  set +e
  if [ "${ROB82_DISABLE_CALLBACK:-0}" = "1" ]; then
    exit "${status}"
  fi
  if [ -z "${LINEAR_API_KEY:-}" ] && [ "${ROB82_CALLBACK_DRY_RUN:-0}" != "1" ]; then
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
    --note "${CALLBACK_NOTE:-ROB-82 TED-LIUM dev UVQLM LR sweep wrapper exited. Inspect exp/results/repro/sweeps/uvqlm/tedlium_dev/ for the result table when complete.}"
    --tail-lines "${CALLBACK_TAIL_LINES:-80}"
    --max-log-chars "${CALLBACK_MAX_LOG_CHARS:-6000}"
    --max-comment-chars "${CALLBACK_MAX_COMMENT_CHARS:-10000}"
  )
  if [ "${ROB82_CALLBACK_CHECK_ONLY:-0}" = "1" ]; then
    callback_args+=(--check-only)
  fi
  if [ "${ROB82_CALLBACK_DRY_RUN:-0}" = "1" ]; then
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

mkdir -p "$(dirname "${LOG_PATH}")" "${RESULT_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "[rob82-uvqlm] branch=${GIT_BRANCH}"
echo "[rob82-uvqlm] commit=${GIT_COMMIT}"
echo "[rob82-uvqlm] result_root=${RESULT_ROOT}"
echo "[rob82-uvqlm] asr_ckpt=${ASR_CKPT}"
echo "[rob82-uvqlm] uvqlm_ckpt=${UVQLM_CKPT}"
echo "[rob82-uvqlm] mask_vae_ckpt=${MASK_VAE_CKPT}"
echo "[rob82-uvqlm] method=${METHOD}"
echo "[rob82-uvqlm] lrs=${LRS}"
echo "[rob82-uvqlm] epochs=${EPOCHS}"
echo "[rob82-uvqlm] repeats=${REPEATS}"
echo "[rob82-uvqlm] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

if [ "${ROB82_CALLBACK_ONLY:-0}" = "1" ]; then
  echo "[rob82-uvqlm] callback-only smoke path requested; exiting before eval."
  exit 0
fi

for required_path in "${ASR_CKPT}" "${UVQLM_CKPT}" "${MASK_VAE_CKPT}"; do
  if [ ! -f "${required_path}" ]; then
    echo "Missing required checkpoint: ${required_path}" >&2
    exit 1
  fi
done

export L2A_EARNINGS22_DIR="${L2A_EARNINGS22_DIR:-/store/store4/data/earnings-22}"
export L2A_TEDLIUM3_LEGACY_DIR="${L2A_TEDLIUM3_LEGACY_DIR:-/store/store4/data/TEDLIUM_release-3/legacy/}"
export L2A_REV16_DIR="${L2A_REV16_DIR:-/store/store4/data/rev_benchmark}"
export L2A_CHIME6_DIR="${L2A_CHIME6_DIR:-/store/store4/data/chime6/}"

python3 - "${RESULT_ROOT}" "${ASR_CKPT}" "${UVQLM_CKPT}" "${MASK_VAE_CKPT}" "${METHOD}" "${LRS}" "${EPOCHS}" "${REPEATS}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
asr_ckpt = sys.argv[2]
uvqlm_ckpt = sys.argv[3]
mask_vae_ckpt = sys.argv[4]
method = sys.argv[5]
lrs = tuple(sys.argv[6].split())
epochs = tuple(int(item) for item in sys.argv[7].split())
repeats = tuple(int(item) for item in sys.argv[8].split())

specs = (("tedlium_dev", "tedlium", "dev", "eval.py"),)
for split_dir, dataset, split, _eval_script in specs:
    result_root = root / split_dir
    (result_root / method / "configs").mkdir(parents=True, exist_ok=True)
    for repeat in repeats:
        repeat_suffix = "" if repeat == 1 else f"_repeat{repeat}"
        seed = 123456 + repeat - 1
        for epoch_count in epochs:
            for lr in lrs:
                tag = f"{split_dir}_epoch{epoch_count}_lr{lr}{repeat_suffix}"
                save_path = result_root / method / f"{tag}.txt"
                config_path = result_root / method / "configs" / f"{tag}.yaml"
                config_path.write_text(
                    f"""checkpointing:
  asr_model: {asr_ckpt}

training:
  device: 'cuda'
  random_seed: {seed}
  batch_size: 84
  epochs: 100
  model_save_path: {uvqlm_ckpt}
  tmp_model_save_path: {uvqlm_ckpt}

evaluation:
  id: 'ROB-82-{dataset}-{split}-{method}-epoch{epoch_count}-lr{lr}-repeat{repeat}'
  dataset: '{dataset}'
  split: '{split}'
  rollout_setting: policy
  use_cer: false
  epochs: {epoch_count}
  augmentation_config:
    seed: {seed}
  optim_args:
    lr: {lr}
  save_path: {save_path}

policy:
  lr: 6e-4
  class: UnconditionalMaskGenerator
  config:
    mask_vae_state_dict_path: {mask_vae_ckpt}
    mask_vae_config:
      latent_dim: 128
      codebook_size: 2048
      use_vq: true
"""
                )
                print(f"[rob82-uvqlm] wrote config {config_path}")
PY

if [ "${ROB82_CONFIG_ONLY:-0}" = "1" ]; then
  echo "[rob82-uvqlm] config-only smoke path requested; exiting before eval."
  exit 0
fi

source /store/store4/software/bin/anaconda3/etc/profile.d/conda.sh
conda activate /store/store4/software/bin/anaconda3/envs/flash_attn_pytorch2

export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

run_split() {
  local split_dir="$1"
  local dataset="$2"
  local split="$3"
  local eval_script="$4"
  local title="$5"
  local note="$6"
  local csv_name="$7"
  local outcome_name="$8"
  local run_epochs="${ROB82_RUN_EPOCHS:-${EPOCHS}}"
  local run_lrs="${ROB82_RUN_LRS:-${LRS}}"
  local run_repeats="${ROB82_RUN_REPEATS:-${REPEATS}}"
  local result_dir="${RESULT_ROOT}/${split_dir}"
  local tag_prefix="${split_dir}"

  if [ "${ROB82_SMOKE:-0}" = "1" ]; then
    run_epochs="${ROB82_SMOKE_EPOCHS:-1}"
    run_lrs="${ROB82_SMOKE_LRS:-5e-6}"
    run_repeats="${ROB82_SMOKE_REPEATS:-1}"
    ROB82_INDEXES="${ROB82_INDEXES:-0}"
    ROB82_DONT_SAVE="${ROB82_DONT_SAVE:-1}"
    echo "[rob82-uvqlm] smoke mode for ${split_dir}: epochs=${run_epochs}; lrs=${run_lrs}; repeats=${run_repeats}; indexes=${ROB82_INDEXES}; dont_save=${ROB82_DONT_SAVE}"
  fi

  cd "${REPO_DIR}/exp"
  for repeat in ${run_repeats}; do
    repeat_suffix=""
    if [ "${repeat}" != "1" ]; then
      repeat_suffix="_repeat${repeat}"
    fi
    for epoch_count in ${run_epochs}; do
      for lr in ${run_lrs}; do
        tag="${tag_prefix}_epoch${epoch_count}_lr${lr}${repeat_suffix}"
        config="${result_dir}/${METHOD}/configs/${tag}.yaml"
        save_path="${result_dir}/${METHOD}/${tag}.txt"
        if [ ! -f "${config}" ]; then
          echo "Missing generated config: ${config}" >&2
          exit 1
        fi
        if [ "${FORCE_RERUN:-0}" != "1" ] && [ -f "${save_path}" ] && grep -q "Updated_WER:" "${save_path}"; then
          echo "[rob82-uvqlm] skipping completed ${split_dir}/${METHOD}/${tag}: ${save_path}"
          continue
        fi
        if [ "${FORCE_RERUN:-0}" = "1" ]; then
          rm -f "${save_path}"
        fi
        args=(python "${eval_script}" --config "${config}")
        if [ -n "${ROB82_INDEXES:-}" ]; then
          args+=(--indexes ${ROB82_INDEXES})
        fi
        if [ "${ROB82_DONT_SAVE:-0}" = "1" ]; then
          args+=(--dont_save)
        fi
        echo "[rob82-uvqlm] running ${split_dir}/${METHOD}/${tag}: ${args[*]}"
        "${args[@]}"
      done
    done
  done

  cd "${REPO_DIR}"
  python3 scripts/summarize_rob82_tedlium_uvqlm_sweep.py \
    --result-root "${result_dir}" \
    --method "${METHOD}" \
    --dataset "${dataset}" \
    --split "${split}" \
    --tag-prefix "${tag_prefix}" \
    --epochs "${EPOCHS}" \
    --lrs "${LRS}" \
    --repeats "${REPEATS}" \
    --csv-name "${csv_name}" \
    --outcome-name "${outcome_name}" \
    --title "${title}" \
    --note "${note}"
}

run_split \
  "tedlium_dev" \
  "tedlium" \
  "dev" \
  "eval.py" \
  "ROB-82 TED-LIUM Dev UVQLM LR Sweep" \
  "UVQLM is evaluated as a separate method family from UFMR, using the ROB-80 centered LR grid and two repeats for consistency with the final ROB-80 sweep contract." \
  "rob82_tedlium_dev_uvqlm_sweep.csv" \
  "ROB-82_TEDLIUM_DEV_UVQLM_OUTCOME.md"

echo "[rob82-uvqlm] finished"
