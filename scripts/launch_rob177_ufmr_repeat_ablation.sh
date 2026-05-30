#!/usr/bin/env bash
# Queue-safe ROB-177 UFMR candidate-repeat ablation.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-177}"
AGGREGATE_DIR="${AGGREGATE_DIR:-${REPO_DIR}/exp/results/repro/symphony/rob-177}"
RESULT_ROOT="${RESULT_ROOT:-${AGGREGATE_DIR}/results}"
LOG_PATH="${LOG_PATH:-${AGGREGATE_DIR}/logs/rob177_ufmr_repeat_ablation.log}"
SCREEN_NAME="${SCREEN_NAME:-rob177_ufmr_repeat_ablation}"
RUNNER_LABEL="${RUNNER_LABEL:-screen:${SCREEN_NAME}}"
QUEUED_COMMAND="${QUEUED_COMMAND:-/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob177_ufmr_repeat_ablation.sh}"
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"

ASR_CKPT="${ASR_CKPT:-/store/store5/data/acp21rjf_checkpoints/spotify/rotary_pos_6l_256d_seq_sched/n_seq_sched_2048_rp_1/step_105360.pt}"
UFMR_VARIANT="${UFMR_VARIANT:-test_wer}"
UFMR_CKPT="${UFMR_CKPT:-/store/store5/data/acp21rjf_checkpoints/l2augment/ufmr/${UFMR_VARIANT}/model.pt}"

SPLIT="${ROB177_SPLIT:-test}"
EPOCHS="${ROB177_EPOCHS:-1}"
ADAPT_LR="${ROB177_LR:-1e-5}"
DATASETS="${ROB177_DATASETS:-earnings22 tedlium}"
CANDIDATE_REPEATS="${ROB177_CANDIDATE_REPEATS:-1 2 5 10 15 20 40 100 200 1000}"
SEEDS="${ROB177_SEEDS:-123456 123457 123458}"
REFERENCE_RESULTS="${ROB177_REFERENCE_RESULTS:-earnings22=${REPO_DIR}/exp/results/repro/UFMR/earnings22_epoch1_lr1e-5.txt tedlium=${REPO_DIR}/exp/results/repro/UFMR/tedlium_epoch1_lr1e-5.txt}"
LEGACY_FIRST_SEED_REPEATS="${ROB177_LEGACY_FIRST_SEED_REPEATS:-2 5 10 20 40 100 200}"

on_exit() {
  status=$?
  set +e
  if [ "${ROB177_DISABLE_CALLBACK:-0}" = "1" ]; then
    exit "${status}"
  fi
  if [ -z "${LINEAR_API_KEY:-}" ] && [ "${ROB177_CALLBACK_DRY_RUN:-0}" != "1" ]; then
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
    --note "${CALLBACK_NOTE:-ROB-177 UFMR candidate-repeat ablation wrapper exited. Inspect exp/results/repro/symphony/rob-177/ROB-177_OUTCOME.md after completion.}"
    --tail-lines "${CALLBACK_TAIL_LINES:-80}"
    --max-log-chars "${CALLBACK_MAX_LOG_CHARS:-6000}"
    --max-comment-chars "${CALLBACK_MAX_COMMENT_CHARS:-10000}"
  )
  if [ "${ROB177_CALLBACK_CHECK_ONLY:-0}" = "1" ]; then
    callback_args+=(--check-only)
  fi
  if [ "${ROB177_CALLBACK_DRY_RUN:-0}" = "1" ]; then
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

echo "[rob177] branch=${GIT_BRANCH}"
echo "[rob177] commit=${GIT_COMMIT}"
echo "[rob177] aggregate_dir=${AGGREGATE_DIR}"
echo "[rob177] result_root=${RESULT_ROOT}"
echo "[rob177] asr_ckpt=${ASR_CKPT}"
echo "[rob177] ufmr_ckpt=${UFMR_CKPT}"
echo "[rob177] datasets=${DATASETS}"
echo "[rob177] split=${SPLIT}"
echo "[rob177] epochs=${EPOCHS}"
echo "[rob177] lr=${ADAPT_LR}"
echo "[rob177] candidate_repeats=${CANDIDATE_REPEATS}"
echo "[rob177] seeds=${SEEDS}"
echo "[rob177] reference_results=${REFERENCE_RESULTS}"
echo "[rob177] legacy_first_seed_repeats=${LEGACY_FIRST_SEED_REPEATS}"
echo "[rob177] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

if [ "${ROB177_CALLBACK_ONLY:-0}" = "1" ]; then
  echo "[rob177] callback-only smoke path requested; exiting before eval."
  exit 0
fi

if [ ! -f "${ASR_CKPT}" ]; then
  echo "Missing ASR checkpoint: ${ASR_CKPT}" >&2
  exit 1
fi
if [ ! -f "${UFMR_CKPT}" ]; then
  fallback="/store/store5/data/acp21rjf_checkpoints/l2augment/ufmr/${UFMR_VARIANT}/tmp_model.pt"
  if [ -f "${fallback}" ]; then
    echo "[rob177] no model.pt for UFMR variant '${UFMR_VARIANT}'; using tmp_model.pt" >&2
    UFMR_CKPT="${fallback}"
  else
    echo "Missing UFMR checkpoint for variant '${UFMR_VARIANT}': ${UFMR_CKPT}" >&2
    exit 1
  fi
fi

export L2A_EARNINGS22_DIR="${L2A_EARNINGS22_DIR:-/store/store4/data/earnings-22}"
export L2A_TEDLIUM3_LEGACY_DIR="${L2A_TEDLIUM3_LEGACY_DIR:-/store/store4/data/TEDLIUM_release-3/legacy/}"
for required_dir in "${L2A_EARNINGS22_DIR}" "${L2A_TEDLIUM3_LEGACY_DIR}"; do
  if [ ! -d "${required_dir}" ]; then
    echo "Missing required data directory: ${required_dir}" >&2
    exit 1
  fi
done

python3 - "${RESULT_ROOT}" "${ASR_CKPT}" "${UFMR_CKPT}" "${DATASETS}" "${SPLIT}" "${EPOCHS}" "${ADAPT_LR}" "${CANDIDATE_REPEATS}" "${SEEDS}" "${LEGACY_FIRST_SEED_REPEATS}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
asr_ckpt = sys.argv[2]
ufmr_ckpt = sys.argv[3]
dataset_tags = tuple(sys.argv[4].split())
split = sys.argv[5]
epochs = int(sys.argv[6])
lr = sys.argv[7]
candidate_repeats = tuple(int(item) for item in sys.argv[8].split())
seeds = tuple(int(item) for item in sys.argv[9].split())
legacy_first_seed_repeats = {int(item) for item in sys.argv[10].split()}
first_seed = seeds[0] if seeds else None

datasets = {
    "earnings22": ("earnings22", "test"),
    "tedlium": ("tedlium", "test"),
}

method_root = root / "UFMR"
(method_root / "configs").mkdir(parents=True, exist_ok=True)

for dataset_tag in dataset_tags:
    if dataset_tag not in datasets:
        raise ValueError(f"Unknown ROB-177 dataset tag: {dataset_tag}")
    dataset, default_split = datasets[dataset_tag]
    active_split = split or default_split
    for repeat_count in candidate_repeats:
        for trial_index, seed in enumerate(seeds, start=1):
            legacy_name = (
                dataset_tag == "earnings22"
                and seed == first_seed
                and repeat_count in legacy_first_seed_repeats
            )
            if legacy_name:
                tag = f"{dataset_tag}_{active_split}_candidate_repeats{repeat_count}_epoch{epochs}_lr{lr}"
            else:
                tag = f"{dataset_tag}_{active_split}_candidate_repeats{repeat_count}_seed{seed}_epoch{epochs}_lr{lr}"
            save_path = method_root / f"{tag}.txt"
            config_path = method_root / "configs" / f"{tag}.yaml"
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
  id: 'ROB-177-{dataset}-{active_split}-UFMR-candidate-repeats{repeat_count}-trial{trial_index}-seed{seed}-epoch{epochs}-lr{lr}'
  dataset: '{dataset}'
  split: '{active_split}'
  rollout_setting: policy
  use_cer: false
  epochs: {epochs}
  augmentation_config:
    repeats: {repeat_count}
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
            print(f"[rob177] wrote config {config_path}")
PY

python3 scripts/summarize_rob177_ufmr_repeat_ablation.py \
  --result-root "${RESULT_ROOT}" \
  --output-dir "${AGGREGATE_DIR}" \
  --candidate-repeats "${CANDIDATE_REPEATS}" \
  --seeds "${SEEDS}" \
  --datasets "${DATASETS}" \
  --split "${SPLIT}" \
  --epochs "${EPOCHS}" \
  --lr "${ADAPT_LR}" \
  --reference-results "${REFERENCE_RESULTS}" \
  --legacy-first-seed-repeats "${LEGACY_FIRST_SEED_REPEATS}"

if [ "${ROB177_CONFIG_ONLY:-0}" = "1" ]; then
  echo "[rob177] config-only smoke path requested; exiting before eval."
  exit 0
fi

source /store/store4/software/bin/anaconda3/etc/profile.d/conda.sh
conda activate /store/store4/software/bin/anaconda3/envs/flash_attn_pytorch2

export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

RUN_DATASETS="${ROB177_RUN_DATASETS:-${DATASETS}}"
RUN_CANDIDATE_REPEATS="${ROB177_RUN_CANDIDATE_REPEATS:-${CANDIDATE_REPEATS}}"
RUN_SEEDS="${ROB177_RUN_SEEDS:-${SEEDS}}"
FIRST_SEED="${SEEDS%% *}"
EVAL_SCRIPT="${ROB177_EVAL_SCRIPT:-eval.py}"

if [ "${ROB177_SMOKE:-0}" = "1" ]; then
  RUN_DATASETS="${ROB177_SMOKE_DATASETS:-tedlium}"
  RUN_CANDIDATE_REPEATS="${ROB177_SMOKE_CANDIDATE_REPEATS:-2}"
  RUN_SEEDS="${ROB177_SMOKE_SEEDS:-123457}"
  ROB177_INDEXES="${ROB177_INDEXES:-0}"
  ROB177_DONT_SAVE="${ROB177_DONT_SAVE:-1}"
  echo "[rob177] smoke mode: datasets=${RUN_DATASETS}; candidate_repeats=${RUN_CANDIDATE_REPEATS}; seeds=${RUN_SEEDS}; indexes=${ROB177_INDEXES}; dont_save=${ROB177_DONT_SAVE}"
fi

cd "${REPO_DIR}/exp"
for dataset_tag in ${RUN_DATASETS}; do
  for repeat_count in ${RUN_CANDIDATE_REPEATS}; do
    for seed in ${RUN_SEEDS}; do
      legacy_name=0
      if [ "${dataset_tag}" = "earnings22" ] && [ "${seed}" = "${FIRST_SEED}" ]; then
        for legacy_repeat in ${LEGACY_FIRST_SEED_REPEATS}; do
          if [ "${repeat_count}" = "${legacy_repeat}" ]; then
            legacy_name=1
            break
          fi
        done
      fi
      if [ "${legacy_name}" = "1" ]; then
        tag="${dataset_tag}_${SPLIT}_candidate_repeats${repeat_count}_epoch${EPOCHS}_lr${ADAPT_LR}"
      else
        tag="${dataset_tag}_${SPLIT}_candidate_repeats${repeat_count}_seed${seed}_epoch${EPOCHS}_lr${ADAPT_LR}"
      fi
      config="${RESULT_ROOT}/UFMR/configs/${tag}.yaml"
      save_path="${RESULT_ROOT}/UFMR/${tag}.txt"
      if [ ! -f "${config}" ]; then
        echo "Missing generated config: ${config}" >&2
        exit 1
      fi
      if [ "${FORCE_RERUN:-0}" != "1" ] && [ -f "${save_path}" ] && grep -q "Updated_WER:" "${save_path}"; then
        echo "[rob177] skipping completed UFMR/${tag}: ${save_path}"
        continue
      fi
      if [ "${FORCE_RERUN:-0}" = "1" ]; then
        rm -f "${save_path}"
      fi
      args=(python "${EVAL_SCRIPT}" --config "${config}")
      if [ -n "${ROB177_INDEXES:-}" ]; then
        args+=(--indexes ${ROB177_INDEXES})
      fi
      if [ "${ROB177_DONT_SAVE:-0}" = "1" ]; then
        args+=(--dont_save)
      fi
      echo "[rob177] running UFMR/${tag}: ${args[*]}"
      "${args[@]}"
    done
  done
done

cd "${REPO_DIR}"
python3 scripts/summarize_rob177_ufmr_repeat_ablation.py \
  --result-root "${RESULT_ROOT}" \
  --output-dir "${AGGREGATE_DIR}" \
  --candidate-repeats "${CANDIDATE_REPEATS}" \
  --seeds "${SEEDS}" \
  --datasets "${DATASETS}" \
  --split "${SPLIT}" \
  --epochs "${EPOCHS}" \
  --lr "${ADAPT_LR}" \
  --reference-results "${REFERENCE_RESULTS}" \
  --legacy-first-seed-repeats "${LEGACY_FIRST_SEED_REPEATS}"
echo "[rob177] finished"
