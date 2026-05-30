#!/usr/bin/env bash
# Submit ROB-132 audio SSL fixed-reward test-set eval cells to Stanage.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-learning-to-augment/ROB-132}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_remaining_datasets}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/mnt/parscratch/users/acp21rjf/rob132-audio-ssl-scratch}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/parscratch/users/acp21rjf/l2augment_model/reward_conditioned_mask_lm/audio_ssl_hubert_base_tedlium_per_utterance_transformer384_dropout0p1_500ep_lr1e3.pt}"
ASR_CKPT="${ASR_CKPT:-/mnt/parscratch/users/acp21rjf/spotify/rotary_pos_6l_256d_seq_sched/n_seq_sched_2048_rp_1/step_105360.pt}"
MASK_VAE_CKPT="${MASK_VAE_CKPT:-/mnt/parscratch/users/acp21rjf/l2augment_model/bvae/bvae_USINGTHISFORNOW_2048gpu.pt}"
CELL_SCRIPT="${CELL_SCRIPT:-scripts/slurm_rob132_audio_ssl_testset_cell.sbatch}"
FINALIZER_SCRIPT="${FINALIZER_SCRIPT:-scripts/slurm_rob132_audio_ssl_testset_finalizer.sbatch}"
FINALIZER_PARTITION="${FINALIZER_PARTITION:-sheffield}"
DATASETS="${ROB132_TESTSETS_DATASETS:-rev16 TAL chime6}"
FIXED_REWARDS="${ROB132_TESTSETS_FIXED_REWARDS:-1.0 0.0}"
EPOCHS="${ROB132_TESTSETS_EPOCHS:-1 5}"
LR="${ROB132_TESTSETS_LR:-1e-5}"
CSV_NAME="${ROB132_TESTSETS_CSV_NAME:-rob132_audio_ssl_self_train_remaining_datasets_fixed_rewards.csv}"
PARTITIONS="${ROB132_TESTSETS_PARTITIONS:-gpu-h100-nvl gpu-h100 gpu}"

cd "${REPO_DIR}"

mkdir -p "${RESULT_ROOT}"

python3 - "${RESULT_ROOT}" "${ASR_CKPT}" "${CHECKPOINT_PATH}" "${MASK_VAE_CKPT}" "${DATASETS}" "${FIXED_REWARDS}" "${EPOCHS}" "${LR}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
asr_ckpt = sys.argv[2]
policy_ckpt = sys.argv[3]
mask_vae_ckpt = sys.argv[4]
dataset_tags = tuple(sys.argv[5].split())
fixed_rewards = tuple(sys.argv[6].split())
epochs = tuple(int(item) for item in sys.argv[7].split())
lr = sys.argv[8]

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
  id: ROB-132-{dataset_tag}-{split}-audio-ssl-transformer384-reward{reward_tag(reward)}-epoch{epoch_count}-lr{lr}
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
            print(f"[rob132-submit] wrote config {config_path}")
PY

if [ "${ROB132_SUBMIT_CONFIG_ONLY:-0}" = "1" ]; then
  echo "[rob132-submit] config-only mode requested; exiting before sbatch submission."
  exit 0
fi

read -r -a partition_list <<< "${PARTITIONS}"

job_ids=()
cell_index=0
for reward in ${FIXED_REWARDS}; do
  for dataset in ${DATASETS}; do
    for epoch in ${EPOCHS}; do
      partition="${partition_list[$((cell_index % ${#partition_list[@]}))]}"
      cell_index=$((cell_index + 1))
      reward_tag="${reward//./p}"
      reward_tag="${reward_tag//-/m}"
      job_name="r132-${dataset:0:3}-r${reward_tag}-e${epoch}"
      job_id="$(
        sbatch --parsable \
          --job-name="${job_name}" \
          --partition="${partition}" \
          --export=ALL,REPO_DIR="${REPO_DIR}",RESULT_ROOT="${RESULT_ROOT}",SCRATCH_ROOT="${SCRATCH_ROOT}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",ASR_CKPT="${ASR_CKPT}",MASK_VAE_CKPT="${MASK_VAE_CKPT}",ROB132_DATASET="${dataset}",ROB132_REWARD="${reward}",ROB132_EPOCH="${epoch}",ROB132_LR="${LR}" \
          "${CELL_SCRIPT}"
      )"
      job_ids+=("${job_id}")
      echo "${job_id}|${partition}|${dataset}|${reward}|${epoch}"
    done
  done
done

dependency="$(IFS=:; echo "${job_ids[*]}")"
finalizer_id="$(
  sbatch --parsable \
    --partition="${FINALIZER_PARTITION}" \
    --dependency="afterany:${dependency}" \
    --export=ALL,REPO_DIR="${REPO_DIR}",RESULT_ROOT="${RESULT_ROOT}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",ROB132_FIXED_REWARDS="${FIXED_REWARDS}",ROB132_DATASETS="${DATASETS}",ROB132_EPOCHS="${EPOCHS}",ROB132_LR="${LR}",CSV_NAME="${CSV_NAME}",QUEUED_COMMAND="scripts/submit_rob132_audio_ssl_testsets_stanage.sh" \
    "${FINALIZER_SCRIPT}"
)"
echo "finalizer|${finalizer_id}|afterany:${dependency}"
