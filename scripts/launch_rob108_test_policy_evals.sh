#!/usr/bin/env bash
# Queue-safe ROB-108 test-set policy evals for RMM, RFM, UFMR, UVQLM, plus NoAug baselines.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-108}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/exp/results/repro/rob108_test_policy_evals}"
LOG_PATH="${LOG_PATH:-${RESULT_ROOT}/logs/rob108_test_policy_evals.log}"
SCREEN_NAME="${SCREEN_NAME:-rob108_test_policy_evals}"
RUNNER_LABEL="${RUNNER_LABEL:-screen:${SCREEN_NAME}}"
QUEUED_COMMAND="${QUEUED_COMMAND:-/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob108_test_policy_evals.sh}"
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"

ASR_CKPT="${ASR_CKPT:-/store/store5/data/acp21rjf_checkpoints/spotify/rotary_pos_6l_256d_seq_sched/n_seq_sched_2048_rp_1/step_105360.pt}"
UFMR_VARIANT="${UFMR_VARIANT:-test_wer}"
UFMR_CKPT="${UFMR_CKPT:-/store/store5/data/acp21rjf_checkpoints/l2augment/ufmr/${UFMR_VARIANT}/model.pt}"
UVQLM_CKPT="${ROB108_UVQLM_CKPT:-/store/store5/data/acp21rjf_checkpoints/l2augment/models/UMLM/modelgpu.pt}"
MASK_VAE_CKPT="${ROB108_MASK_VAE_CKPT:-/store/store5/data/acp21rjf_checkpoints/l2augment/models/bvae/bvae_USINGTHISFORNOW_2048gpu.pt}"

DATASETS="${ROB108_DATASETS:-tedlium earnings22 chime6 rev16 TAL}"
METHODS="${ROB108_METHODS:-NoAug RFM RMM UFMR UVQLM}"
REPEATS="${ROB108_REPEATS:-1}"
EPOCH1_LRS="${ROB108_EPOCH1_LRS:-1e-5 3e-5}"
EPOCH5_LRS="${ROB108_EPOCH5_LRS:-1e-5}"
UFMR_SEARCH_REPEATS="${ROB108_UFMR_SEARCH_REPEATS:-15}"

on_exit() {
  status=$?
  set +e
  if [ "${ROB108_DISABLE_CALLBACK:-0}" = "1" ]; then
    exit "${status}"
  fi
  if [ -z "${LINEAR_API_KEY:-}" ] && [ "${ROB108_CALLBACK_DRY_RUN:-0}" != "1" ]; then
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
    --note "${CALLBACK_NOTE:-ROB-108 test-set policy eval wrapper exited. Inspect exp/results/repro/rob108_test_policy_evals/OUTCOME.md after completion.}"
    --tail-lines "${CALLBACK_TAIL_LINES:-80}"
    --max-log-chars "${CALLBACK_MAX_LOG_CHARS:-6000}"
    --max-comment-chars "${CALLBACK_MAX_COMMENT_CHARS:-10000}"
  )
  if [ "${ROB108_CALLBACK_CHECK_ONLY:-0}" = "1" ]; then
    callback_args+=(--check-only)
  fi
  if [ "${ROB108_CALLBACK_DRY_RUN:-0}" = "1" ]; then
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

echo "[rob108] branch=${GIT_BRANCH}"
echo "[rob108] commit=${GIT_COMMIT}"
echo "[rob108] result_root=${RESULT_ROOT}"
echo "[rob108] asr_ckpt=${ASR_CKPT}"
echo "[rob108] ufmr_ckpt=${UFMR_CKPT}"
echo "[rob108] uvqlm_ckpt=${UVQLM_CKPT}"
echo "[rob108] mask_vae_ckpt=${MASK_VAE_CKPT}"
echo "[rob108] datasets=${DATASETS}"
echo "[rob108] methods=${METHODS}"
echo "[rob108] repeats=${REPEATS}"
echo "[rob108] epoch1_lrs=${EPOCH1_LRS}"
echo "[rob108] epoch5_lrs=${EPOCH5_LRS}"
echo "[rob108] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

if [ "${ROB108_CALLBACK_ONLY:-0}" = "1" ]; then
  echo "[rob108] callback-only smoke path requested; exiting before eval."
  exit 0
fi

if [ ! -f "${ASR_CKPT}" ]; then
  echo "Missing ASR checkpoint: ${ASR_CKPT}" >&2
  exit 1
fi
if [ ! -f "${UFMR_CKPT}" ]; then
  fallback="/store/store5/data/acp21rjf_checkpoints/l2augment/ufmr/${UFMR_VARIANT}/tmp_model.pt"
  if [ -f "${fallback}" ]; then
    echo "[rob108] no model.pt for UFMR variant '${UFMR_VARIANT}'; using tmp_model.pt" >&2
    UFMR_CKPT="${fallback}"
  else
    echo "Missing UFMR checkpoint for variant '${UFMR_VARIANT}': ${UFMR_CKPT}" >&2
    exit 1
  fi
fi
for required_path in "${UVQLM_CKPT}" "${MASK_VAE_CKPT}"; do
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

python3 - "${RESULT_ROOT}" "${ASR_CKPT}" "${UFMR_CKPT}" "${UVQLM_CKPT}" "${MASK_VAE_CKPT}" "${DATASETS}" "${METHODS}" "${REPEATS}" "${EPOCH1_LRS}" "${EPOCH5_LRS}" "${UFMR_SEARCH_REPEATS}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
asr_ckpt = sys.argv[2]
ufmr_ckpt = sys.argv[3]
uvqlm_ckpt = sys.argv[4]
mask_vae_ckpt = sys.argv[5]
dataset_tags = tuple(sys.argv[6].split())
methods = tuple(sys.argv[7].split())
repeats = tuple(int(item) for item in sys.argv[8].split())
epoch1_lrs = tuple(sys.argv[9].split())
epoch5_lrs = tuple(sys.argv[10].split())
ufmr_search_repeats = int(sys.argv[11])

datasets = {
    "tedlium": ("tedlium", "test"),
    "earnings22": ("earnings22", "test"),
    "chime6": ("chime6", "test"),
    "rev16": ("rev16", "test"),
    "TAL": ("this_american_life", "test"),
}
method_specs = {
    "NoAug": {
        "policy_class": "NoAugmentation",
        "policy_lr": "1e-4",
        "training_extra": "",
        "policy_config": "",
        "augmentation": "    seed: {seed}\n",
    },
    "RFM": {
        "policy_class": "FrequencyMaskingRanker",
        "policy_lr": "1e-4",
        "training_extra": "",
        "policy_config": "",
        "augmentation": "    repeats: 1\n    seed: {seed}\n    use_random: true\n",
    },
    "RMM": {
        "policy_class": "MixedMaskingRanker",
        "policy_lr": "1e-4",
        "training_extra": "",
        "policy_config": "",
        "augmentation": "    repeats: 1\n    seed: {seed}\n    use_random: true\n",
    },
    "UFMR": {
        "policy_class": "UnconditionalFrequencyMaskingRanker",
        "policy_lr": "1e-4",
        "training_extra": f"  model_save_path: {ufmr_ckpt}\n  tmp_model_save_path: {ufmr_ckpt}\n",
        "policy_config": "",
        "augmentation": f"    repeats: {ufmr_search_repeats}\n    seed: {{seed}}\n    use_random: false\n",
    },
    "UVQLM": {
        "policy_class": "UnconditionalMaskGenerator",
        "policy_lr": "6e-4",
        "training_extra": f"  model_save_path: {uvqlm_ckpt}\n  tmp_model_save_path: {uvqlm_ckpt}\n",
        "policy_config": (
            "  config:\n"
            f"    mask_vae_state_dict_path: {mask_vae_ckpt}\n"
            "    mask_vae_config:\n"
            "      latent_dim: 128\n"
            "      codebook_size: 2048\n"
            "      use_vq: true\n"
        ),
        "augmentation": "    seed: {seed}\n",
    },
}

for method in methods:
    if method not in method_specs:
        raise ValueError(f"Unknown method: {method}")
    (root / method / "configs").mkdir(parents=True, exist_ok=True)

for dataset_tag in dataset_tags:
    dataset, split = datasets[dataset_tag]
    for method in methods:
        spec = method_specs[method]
        for repeat in repeats:
            repeat_suffix = "" if repeat == 1 else f"_repeat{repeat}"
            seed = 123456 + repeat - 1
            if method == "NoAug":
                cells = ((0, "baseline"),)
                tag_template = f"{dataset_tag}_baseline{repeat_suffix}"
            else:
                cells = tuple((1, lr) for lr in epoch1_lrs) + tuple((5, lr) for lr in epoch5_lrs)
                tag_template = None
            for epoch_count, lr in cells:
                tag = tag_template or f"{dataset_tag}_epoch{epoch_count}_lr{lr}{repeat_suffix}"
                save_path = root / method / f"{tag}.txt"
                config_path = root / method / "configs" / f"{tag}.yaml"
                augmentation = spec["augmentation"].format(seed=seed)
                config_path.write_text(
                    f"""checkpointing:
  asr_model: {asr_ckpt}

training:
  device: 'cuda'
  random_seed: {seed}
  batch_size: 84
  epochs: 100
{spec["training_extra"]}
evaluation:
  id: 'ROB-108-{dataset}-{split}-{method}-{tag}-repeat{repeat}'
  dataset: '{dataset}'
  split: '{split}'
  rollout_setting: policy
  use_cer: false
  epochs: {epoch_count}
  augmentation_config:
{augmentation}  optim_args:
    lr: {0.0 if method == "NoAug" else lr}
  save_path: {save_path}

policy:
  lr: {spec["policy_lr"]}
  class: {spec["policy_class"]}
{spec["policy_config"]}"""
                )
                print(f"[rob108] wrote config {config_path}")
PY

python3 scripts/summarize_rob108_test_policy_evals.py \
  --result-root "${RESULT_ROOT}" \
  --datasets "${DATASETS}" \
  --methods "${METHODS}" \
  --repeats "${REPEATS}" \
  --epoch1-lrs "${EPOCH1_LRS}" \
  --epoch5-lrs "${EPOCH5_LRS}"

if [ "${ROB108_CONFIG_ONLY:-0}" = "1" ]; then
  echo "[rob108] config-only smoke path requested; exiting before eval."
  exit 0
fi

source /store/store4/software/bin/anaconda3/etc/profile.d/conda.sh
conda activate /store/store4/software/bin/anaconda3/envs/flash_attn_pytorch2

export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

RUN_METHODS="${ROB108_RUN_METHODS:-${METHODS}}"
RUN_DATASETS="${ROB108_RUN_DATASETS:-${DATASETS}}"
RUN_REPEATS="${ROB108_RUN_REPEATS:-${REPEATS}}"
RUN_EPOCHS="${ROB108_RUN_EPOCHS:-1 5}"
RUN_EPOCH1_LRS="${ROB108_RUN_EPOCH1_LRS:-${EPOCH1_LRS}}"
RUN_EPOCH5_LRS="${ROB108_RUN_EPOCH5_LRS:-${EPOCH5_LRS}}"
EVAL_SCRIPT="${ROB108_EVAL_SCRIPT:-eval.py}"

if [ "${ROB108_SMOKE:-0}" = "1" ]; then
  RUN_METHODS="${ROB108_SMOKE_METHODS:-RFM}"
  RUN_DATASETS="${ROB108_SMOKE_DATASETS:-tedlium}"
  RUN_REPEATS="${ROB108_SMOKE_REPEATS:-1}"
  RUN_EPOCHS="${ROB108_SMOKE_EPOCHS:-1}"
  RUN_EPOCH1_LRS="${ROB108_SMOKE_LRS:-1e-5}"
  RUN_EPOCH5_LRS="${ROB108_SMOKE_LRS:-1e-5}"
  ROB108_INDEXES="${ROB108_INDEXES:-0}"
  ROB108_DONT_SAVE="${ROB108_DONT_SAVE:-1}"
  echo "[rob108] smoke mode: methods=${RUN_METHODS}; datasets=${RUN_DATASETS}; repeats=${RUN_REPEATS}; epochs=${RUN_EPOCHS}; lrs=${ROB108_SMOKE_LRS:-1e-5}; indexes=${ROB108_INDEXES}; dont_save=${ROB108_DONT_SAVE}"
fi

cd "${REPO_DIR}/exp"
for method in ${RUN_METHODS}; do
  for dataset_tag in ${RUN_DATASETS}; do
    for repeat in ${RUN_REPEATS}; do
      repeat_suffix=""
      if [ "${repeat}" != "1" ]; then
        repeat_suffix="_repeat${repeat}"
      fi
      if [ "${method}" = "NoAug" ]; then
        tag="${dataset_tag}_baseline${repeat_suffix}"
        config="${RESULT_ROOT}/${method}/configs/${tag}.yaml"
        save_path="${RESULT_ROOT}/${method}/${tag}.txt"
        if [ "${FORCE_RERUN:-0}" != "1" ] && [ -f "${save_path}" ] && grep -q "Updated_WER:" "${save_path}"; then
          echo "[rob108] skipping completed ${method}/${tag}: ${save_path}"
          continue
        fi
        if [ "${FORCE_RERUN:-0}" = "1" ]; then
          rm -f "${save_path}"
        fi
        args=(python "${EVAL_SCRIPT}" --config "${config}")
        if [ -n "${ROB108_INDEXES:-}" ]; then
          args+=(--indexes ${ROB108_INDEXES})
        fi
        if [ "${ROB108_DONT_SAVE:-0}" = "1" ]; then
          args+=(--dont_save)
        fi
        echo "[rob108] running ${method}/${tag}: ${args[*]}"
        "${args[@]}"
        continue
      fi
      for epoch_count in ${RUN_EPOCHS}; do
        if [ "${epoch_count}" = "1" ]; then
          run_lrs="${ROB108_RUN_LRS:-${RUN_EPOCH1_LRS}}"
        elif [ "${epoch_count}" = "5" ]; then
          run_lrs="${ROB108_RUN_LRS:-${RUN_EPOCH5_LRS}}"
        else
          echo "Unsupported epoch count for ROB-108: ${epoch_count}" >&2
          exit 1
        fi
        for lr in ${run_lrs}; do
          tag="${dataset_tag}_epoch${epoch_count}_lr${lr}${repeat_suffix}"
          config="${RESULT_ROOT}/${method}/configs/${tag}.yaml"
          save_path="${RESULT_ROOT}/${method}/${tag}.txt"
          if [ ! -f "${config}" ]; then
            echo "Missing generated config: ${config}" >&2
            exit 1
          fi
          if [ "${FORCE_RERUN:-0}" != "1" ] && [ -f "${save_path}" ] && grep -q "Updated_WER:" "${save_path}"; then
            echo "[rob108] skipping completed ${method}/${tag}: ${save_path}"
            continue
          fi
          if [ "${FORCE_RERUN:-0}" = "1" ]; then
            rm -f "${save_path}"
          fi
          args=(python "${EVAL_SCRIPT}" --config "${config}")
          if [ -n "${ROB108_INDEXES:-}" ]; then
            args+=(--indexes ${ROB108_INDEXES})
          fi
          if [ "${ROB108_DONT_SAVE:-0}" = "1" ]; then
            args+=(--dont_save)
          fi
          echo "[rob108] running ${method}/${tag}: ${args[*]}"
          "${args[@]}"
        done
      done
    done
  done
done

cd "${REPO_DIR}"
python3 scripts/summarize_rob108_test_policy_evals.py \
  --result-root "${RESULT_ROOT}" \
  --datasets "${DATASETS}" \
  --methods "${METHODS}" \
  --repeats "${REPEATS}" \
  --epoch1-lrs "${EPOCH1_LRS}" \
  --epoch5-lrs "${EPOCH5_LRS}"
echo "[rob108] finished"
