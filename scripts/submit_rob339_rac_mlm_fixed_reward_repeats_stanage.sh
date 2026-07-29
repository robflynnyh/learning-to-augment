#!/usr/bin/env bash
# Submit ROB-339 RAC-MLM fixed-reward repeat cells to Stanage.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-learning-to-augment/ROB-339}"
LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-339}"
HISTORICAL_ROOT="${HISTORICAL_ROOT:-${REPO_DIR}/exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384/eval/test_fixed_rewards_0_and_1}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384/eval/test_fixed_rewards_0_and_1_rob339_repeats}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/mnt/parscratch/users/acp21rjf/rob339-rac-mlm-fixed-reward-scratch}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/parscratch/users/acp21rjf/l2augment_model/reward_conditioned_mask_lm/audio_ssl_hubert_base_tedlium_per_utterance_transformer384_dropout0p1_500ep_lr1e3.pt}"
ASR_CKPT="${ASR_CKPT:-/mnt/parscratch/users/acp21rjf/spotify/rotary_pos_6l_256d_seq_sched/n_seq_sched_2048_rp_1/step_105360.pt}"
MASK_VAE_CKPT="${MASK_VAE_CKPT:-/mnt/parscratch/users/acp21rjf/l2augment_model/bvae/bvae_USINGTHISFORNOW_2048gpu.pt}"
CELL_SCRIPT="${CELL_SCRIPT:-scripts/slurm_rob339_rac_mlm_fixed_reward_cell.sbatch}"
FINALIZER_SCRIPT="${FINALIZER_SCRIPT:-scripts/slurm_rob339_rac_mlm_fixed_reward_finalizer.sbatch}"
FINALIZER_PARTITION="${FINALIZER_PARTITION:-sheffield}"
DATASETS="${ROB339_DATASETS:-tedlium earnings22 rev16 TAL chime6}"
FIXED_REWARDS="${ROB339_FIXED_REWARDS:-1.0 0.0}"
EPOCHS="${ROB339_EPOCHS:-1 5}"
NEW_REPEATS="${ROB339_NEW_REPEATS:-2 3}"
SUMMARY_REPEATS="${ROB339_SUMMARY_REPEATS:-1 2 3}"
LR="${ROB339_LR:-1e-5}"
CSV_NAME="${ROB339_CSV_NAME:-rob339_rac_mlm_fixed_reward_repeats.csv}"
AGGREGATE_CSV_NAME="${ROB339_AGGREGATE_CSV_NAME:-rob339_rac_mlm_fixed_reward_repeats_aggregate.csv}"
PARTITIONS="${ROB339_PARTITIONS:-gpu-h100-nvl gpu-h100 gpu}"
SSL_EXTRACTION_BATCH_SIZE="${ROB339_SSL_EXTRACTION_BATCH_SIZE:-32}"

cd "${REPO_DIR}"

mkdir -p "${RESULT_ROOT}"

python3 - "${RESULT_ROOT}" "${ASR_CKPT}" "${CHECKPOINT_PATH}" "${MASK_VAE_CKPT}" "${DATASETS}" "${FIXED_REWARDS}" "${EPOCHS}" "${NEW_REPEATS}" "${LR}" "${LINEAR_ISSUE}" "${SSL_EXTRACTION_BATCH_SIZE}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
asr_ckpt = sys.argv[2]
policy_ckpt = sys.argv[3]
mask_vae_ckpt = sys.argv[4]
dataset_tags = tuple(sys.argv[5].split())
fixed_rewards = tuple(sys.argv[6].split())
epochs = tuple(int(item) for item in sys.argv[7].split())
repeats = tuple(int(item) for item in sys.argv[8].split())
lr = sys.argv[9]
linear_issue = sys.argv[10]
ssl_extraction_batch_size = int(sys.argv[11])

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
            for repeat in repeats:
                seed = 123456 + repeat - 1
                tag = f"{dataset_tag}_{split}_reward{reward_tag(reward)}_epoch{epoch_count}_lr{lr}_repeat{repeat}"
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
  id: {linear_issue}-{dataset_tag}-{split}-audio-ssl-transformer384-reward{reward_tag(reward)}-epoch{epoch_count}-lr{lr}-repeat{repeat}
  dataset: {dataset}
  split: {split}
  rollout_fn: multistep
  use_cer: false
  epochs: {epoch_count}
  augmentation_config:
    conditioning_reward: {reward}
    sample: true
    seed: {seed}
  optim_args:
    lr: {lr}
  save_path: {save_path}

dataset:
  ssl_bundle: HUBERT_BASE
  ssl_device: cuda
  ssl_extraction_batch_size: {ssl_extraction_batch_size}
  tedlium_base: /mnt/parscratch/users/acp21rjf/TEDLIUM_release-3/legacy

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
                print(f"[rob339-submit] wrote config {config_path}")
PY

python3 scripts/summarize_rob339_rac_mlm_fixed_reward_repeats.py \
  --historical-root "${HISTORICAL_ROOT}" \
  --result-root "${RESULT_ROOT}" \
  --fixed-rewards "${FIXED_REWARDS}" \
  --datasets "${DATASETS}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --repeats "${SUMMARY_REPEATS}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --command "scripts/submit_rob339_rac_mlm_fixed_reward_repeats_stanage.sh" \
  --branch "$(git rev-parse --abbrev-ref HEAD)" \
  --commit "$(git rev-parse HEAD)" \
  --log-path "${RESULT_ROOT}/logs/stanage/prelaunch-summary.log" \
  --screen-log-path "slurm" \
  --csv-name "${CSV_NAME}" \
  --aggregate-csv-name "${AGGREGATE_CSV_NAME}"

if [ "${ROB339_SUBMIT_CONFIG_ONLY:-0}" = "1" ]; then
  echo "[rob339-submit] config-only mode requested; exiting before sbatch submission."
  exit 0
fi

read -r -a partition_list <<< "${PARTITIONS}"

job_ids=()
cell_index=0
for reward in ${FIXED_REWARDS}; do
  for dataset in ${DATASETS}; do
    for epoch in ${EPOCHS}; do
      for repeat in ${NEW_REPEATS}; do
        partition="${partition_list[$((cell_index % ${#partition_list[@]}))]}"
        cell_index=$((cell_index + 1))
        seed=$((123456 + repeat - 1))
        reward_tag="${reward//./p}"
        reward_tag="${reward_tag//-/m}"
        job_name="r339-${dataset:0:3}-r${reward_tag}-e${epoch}-p${repeat}"
        job_id="$(
          sbatch --parsable \
            --job-name="${job_name}" \
            --partition="${partition}" \
            --export=ALL,REPO_DIR="${REPO_DIR}",RESULT_ROOT="${RESULT_ROOT}",SCRATCH_ROOT="${SCRATCH_ROOT}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",ASR_CKPT="${ASR_CKPT}",MASK_VAE_CKPT="${MASK_VAE_CKPT}",ROB339_DATASET="${dataset}",ROB339_REWARD="${reward}",ROB339_EPOCH="${epoch}",ROB339_REPEAT="${repeat}",ROB339_SEED="${seed}",ROB339_LR="${LR}" \
            "${CELL_SCRIPT}"
        )"
        job_ids+=("${job_id}")
        echo "${job_id}|${partition}|${dataset}|${reward}|${epoch}|repeat${repeat}|seed${seed}"
      done
    done
  done
done

dependency="$(IFS=:; echo "${job_ids[*]}")"
finalizer_id="$(
  sbatch --parsable \
    --partition="${FINALIZER_PARTITION}" \
    --dependency="afterany:${dependency}" \
    --export=ALL,REPO_DIR="${REPO_DIR}",LINEAR_ISSUE="${LINEAR_ISSUE}",HISTORICAL_ROOT="${HISTORICAL_ROOT}",RESULT_ROOT="${RESULT_ROOT}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",ROB339_FIXED_REWARDS="${FIXED_REWARDS}",ROB339_DATASETS="${DATASETS}",ROB339_EPOCHS="${EPOCHS}",ROB339_REPEATS="${SUMMARY_REPEATS}",ROB339_LR="${LR}",CSV_NAME="${CSV_NAME}",AGGREGATE_CSV_NAME="${AGGREGATE_CSV_NAME}",QUEUED_COMMAND="scripts/submit_rob339_rac_mlm_fixed_reward_repeats_stanage.sh" \
    "${FINALIZER_SCRIPT}"
)"
echo "finalizer|${finalizer_id}|afterany:${dependency}"
