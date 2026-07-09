#!/usr/bin/env bash
# Submit ROB-340 large-ASR UFMR/RFM repeat cells to Stanage.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-learning-to-augment/ROB-340}"
LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-340}"
AGGREGATE_DIR="${AGGREGATE_DIR:-${REPO_DIR}/exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048}"
RESULT_ROOT="${RESULT_ROOT:-${AGGREGATE_DIR}/results}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/mnt/parscratch/users/acp21rjf/rob340-large-asr-repeats-scratch}"
ASR_CKPT="${ASR_CKPT:-/mnt/parscratch/users/acp21rjf/spotify/rotary_pos_6l_256d_seq_sched/n_seq_sched_2048_rp_1/step_105360.pt}"
UFMR_VARIANT="${UFMR_VARIANT:-test_wer}"
UFMR_CKPT="${UFMR_CKPT:-/mnt/parscratch/users/acp21rjf/l2augment_model/ufm/${UFMR_VARIANT}/model.pt}"
CELL_SCRIPT="${CELL_SCRIPT:-scripts/slurm_rob340_large_asr_repeat_cell.sbatch}"
FINALIZER_SCRIPT="${FINALIZER_SCRIPT:-scripts/slurm_rob340_large_asr_finalizer.sbatch}"
FINALIZER_PARTITION="${FINALIZER_PARTITION:-sheffield}"
DATASETS="${ROB340_DATASETS:-tedlium earnings22 chime6 rev16 TAL}"
METHODS="${ROB340_METHODS:-UFMR RFM}"
REPEATS="${ROB340_REPEATS:-2 3}"
SUMMARY_REPEATS="${ROB340_SUMMARY_REPEATS:-1 2 3}"
LR="${ROB340_LR:-1e-5}"
EPOCH="${ROB340_EPOCH:-1}"
UFMR_SEARCH_REPEATS="${ROB340_UFMR_SEARCH_REPEATS:-15}"
CSV_NAME="${ROB340_CSV_NAME:-rob340_large_asr_rfm_ufmr_repeats.csv}"
OUTCOME_NAME="${ROB340_OUTCOME_NAME:-ROB-340_OUTCOME.md}"
PARTITIONS="${ROB340_PARTITIONS:-gpu-h100-nvl gpu-h100 gpu}"

cd "${REPO_DIR}"

mkdir -p "${RESULT_ROOT}"

python3 - "${RESULT_ROOT}" "${ASR_CKPT}" "${UFMR_CKPT}" "${DATASETS}" "${METHODS}" "${REPEATS}" "${LR}" "${EPOCH}" "${UFMR_SEARCH_REPEATS}" "${LINEAR_ISSUE}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
asr_ckpt = sys.argv[2]
ufmr_ckpt = sys.argv[3]
dataset_tags = tuple(sys.argv[4].split())
methods = tuple(sys.argv[5].split())
repeats = tuple(int(item) for item in sys.argv[6].split())
lr = sys.argv[7]
epoch_count = int(sys.argv[8])
ufmr_search_repeats = int(sys.argv[9])
linear_issue = sys.argv[10]

datasets = {
    "tedlium": ("tedlium", "test"),
    "earnings22": ("earnings22", "test"),
    "chime6": ("chime6", "test"),
    "rev16": ("rev16", "test"),
    "TAL": ("this_american_life", "test"),
}

for method in methods:
    if method not in {"UFMR", "RFM"}:
        raise ValueError(f"Unsupported ROB-340 method: {method}")
    (root / method / "configs").mkdir(parents=True, exist_ok=True)
    for dataset_tag in dataset_tags:
        if dataset_tag not in datasets:
            raise ValueError(f"Unsupported ROB-340 dataset: {dataset_tag}")
        dataset, split = datasets[dataset_tag]
        for repeat in repeats:
            if repeat == 1:
                raise ValueError("ROB-340 submitter must not regenerate repeat 1")
            seed = 123456 + repeat - 1
            repeat_suffix = f"_repeat{repeat}"
            tag = f"{dataset_tag}_epoch{epoch_count}_lr{lr}{repeat_suffix}"
            save_path = root / method / f"{tag}.txt"
            config_path = root / method / "configs" / f"{tag}.yaml"
            if method == "UFMR":
                body = f"""checkpointing:
  asr_model: {asr_ckpt}

training:
  device: 'cuda'
  random_seed: {seed}
  batch_size: 84
  epochs: 100
  model_save_path: {ufmr_ckpt}
  tmp_model_save_path: {ufmr_ckpt}

evaluation:
  id: '{linear_issue}-{dataset}-{split}-UFMR-{tag}-repeat{repeat}'
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
"""
            else:
                body = f"""checkpointing:
  asr_model: {asr_ckpt}

training:
  device: 'cuda'
  random_seed: {seed}
  batch_size: 84
  epochs: 100

evaluation:
  id: '{linear_issue}-{dataset}-{split}-RFM-{tag}-repeat{repeat}'
  dataset: '{dataset}'
  split: '{split}'
  rollout_setting: policy
  use_cer: false
  epochs: {epoch_count}
  augmentation_config:
    repeats: 1
    seed: {seed}
    use_random: true
  optim_args:
    lr: {lr}
  save_path: {save_path}

policy:
  lr: 1e-4
  class: FrequencyMaskingRanker
"""
            config_path.write_text(body, encoding="utf-8")
            print(f"[rob340-submit] wrote config {config_path}")
PY

if [ "${ROB340_SUBMIT_CONFIG_ONLY:-0}" = "1" ]; then
  echo "[rob340-submit] config-only mode requested; exiting before sbatch submission."
  exit 0
fi

read -r -a partition_list <<< "${PARTITIONS}"

job_ids=()
cell_index=0
for method in ${METHODS}; do
  for dataset in ${DATASETS}; do
    for repeat in ${REPEATS}; do
      partition="${partition_list[$((cell_index % ${#partition_list[@]}))]}"
      cell_index=$((cell_index + 1))
      job_name="r340-${method,,}-${dataset:0:3}-r${repeat}"
      job_id="$(
        sbatch --parsable \
          --job-name="${job_name}" \
          --partition="${partition}" \
          --export=ALL,REPO_DIR="${REPO_DIR}",RESULT_ROOT="${RESULT_ROOT}",SCRATCH_ROOT="${SCRATCH_ROOT}",ASR_CKPT="${ASR_CKPT}",UFMR_CKPT="${UFMR_CKPT}",ROB340_METHOD="${method}",ROB340_DATASET="${dataset}",ROB340_REPEAT="${repeat}",ROB340_LR="${LR}",ROB340_EPOCH="${EPOCH}" \
          "${CELL_SCRIPT}"
      )"
      job_ids+=("${job_id}")
      echo "${job_id}|${partition}|${method}|${dataset}|repeat${repeat}"
    done
  done
done

dependency="$(IFS=:; echo "${job_ids[*]}")"
finalizer_id="$(
  sbatch --parsable \
    --partition="${FINALIZER_PARTITION}" \
    --dependency="afterany:${dependency}" \
    --export=ALL,REPO_DIR="${REPO_DIR}",LINEAR_ISSUE="${LINEAR_ISSUE}",AGGREGATE_DIR="${AGGREGATE_DIR}",RESULT_ROOT="${RESULT_ROOT}",ROB340_METHODS="${METHODS}",ROB340_DATASETS="${DATASETS}",ROB340_REPEATS="${SUMMARY_REPEATS}",ROB340_LR="${LR}",ROB340_EPOCH="${EPOCH}",CSV_NAME="${CSV_NAME}",OUTCOME_NAME="${OUTCOME_NAME}",QUEUED_COMMAND="scripts/submit_rob340_large_asr_repeats_stanage.sh" \
    "${FINALIZER_SCRIPT}"
)"
echo "finalizer|${finalizer_id}|afterany:${dependency}"
