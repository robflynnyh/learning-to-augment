#!/usr/bin/env bash
# Submit ROB-338 RC-MLM fixed-reward repeat cells to Stanage.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-learning-to-augment/ROB-338}"
LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-338}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/mnt/parscratch/users/acp21rjf/rob338-rc-mlm-fixed-reward-repeats-scratch}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/parscratch/users/acp21rjf/l2augment_model/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt}"
ASR_CKPT="${ASR_CKPT:-/mnt/parscratch/users/acp21rjf/spotify/rotary_pos_6l_256d_seq_sched/n_seq_sched_2048_rp_1/step_105360.pt}"
MASK_VAE_CKPT="${MASK_VAE_CKPT:-/mnt/parscratch/users/acp21rjf/l2augment_model/bvae/bvae_USINGTHISFORNOW_2048gpu.pt}"
CELL_SCRIPT="${CELL_SCRIPT:-scripts/slurm_rob338_rc_mlm_fixed_reward_cell.sbatch}"
FINALIZER_SCRIPT="${FINALIZER_SCRIPT:-scripts/slurm_rob338_rc_mlm_fixed_reward_finalizer.sbatch}"
FINALIZER_PARTITION="${FINALIZER_PARTITION:-sheffield}"
DATASETS="${ROB338_DATASETS:-tedlium earnings22 rev16 TAL chime6}"
FIXED_REWARDS="${ROB338_FIXED_REWARDS:-1.0 0.0}"
EPOCHS="${ROB338_EPOCHS:-1 5}"
LR="${ROB338_LR:-1e-5}"
ALL_REPEATS="${ROB338_ALL_REPEATS:-1 2 3}"
RUN_REPEATS="${ROB338_RUN_REPEATS:-2 3}"
CSV_NAME="${ROB338_CSV_NAME:-rob124_384_dropout_all_dataset_fixed_rewards_0_and_1.csv}"
PARTITIONS="${ROB338_PARTITIONS:-gpu-h100-nvl gpu-h100 gpu}"
SUMMARY_DATASETS="${ROB338_SUMMARY_DATASETS:-${DATASETS}}"
SUMMARY_EPOCHS="${ROB338_SUMMARY_EPOCHS:-${EPOCHS}}"
SUMMARY_FIXED_REWARDS="${ROB338_SUMMARY_FIXED_REWARDS:-${FIXED_REWARDS}}"
SUMMARY_REPEATS="${ROB338_SUMMARY_REPEATS:-${ALL_REPEATS}}"
QUEUED_COMMAND="${QUEUED_COMMAND:-scripts/submit_rob338_rc_mlm_repeats_stanage.sh}"

cd "${REPO_DIR}"

mkdir -p "${RESULT_ROOT}"

python3 - "${RESULT_ROOT}" "${ASR_CKPT}" "${CHECKPOINT_PATH}" "${MASK_VAE_CKPT}" "${DATASETS}" "${FIXED_REWARDS}" "${EPOCHS}" "${LR}" "${ALL_REPEATS}" "${LINEAR_ISSUE}" <<'PY'
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
repeats = tuple(int(item) for item in sys.argv[9].split())
linear_issue = sys.argv[10]

datasets = {
    "tedlium": ("tedlium", "test"),
    "earnings22": ("earnings22", "test"),
    "rev16": ("rev16", "test"),
    "TAL": ("this_american_life", "test"),
    "tal": ("this_american_life", "test"),
    "this_american_life": ("this_american_life", "test"),
    "chime6": ("chime6", "test"),
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
                print(f"[rob338-submit] wrote config {config_path}")
PY

if [ "${ROB338_SUBMIT_CONFIG_ONLY:-0}" = "1" ]; then
  echo "[rob338-submit] config-only mode requested; exiting before sbatch submission."
  exit 0
fi

read -r -a partition_list <<< "${PARTITIONS}"

job_ids=()
cell_index=0
for repeat in ${RUN_REPEATS}; do
  for reward in ${FIXED_REWARDS}; do
    for dataset in ${DATASETS}; do
      for epoch in ${EPOCHS}; do
        partition="${partition_list[$((cell_index % ${#partition_list[@]}))]}"
        cell_index=$((cell_index + 1))
        reward_tag="${reward//./p}"
        reward_tag="${reward_tag//-/m}"
        dataset_tag="${dataset,,}"
        job_name="r338-${dataset_tag:0:3}-r${reward_tag}-e${epoch}-p${repeat}"
        job_id="$(
          sbatch --parsable \
            --job-name="${job_name}" \
            --partition="${partition}" \
            --export=ALL,REPO_DIR="${REPO_DIR}",RESULT_ROOT="${RESULT_ROOT}",SCRATCH_ROOT="${SCRATCH_ROOT}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",ASR_CKPT="${ASR_CKPT}",MASK_VAE_CKPT="${MASK_VAE_CKPT}",ROB338_DATASET="${dataset}",ROB338_REWARD="${reward}",ROB338_EPOCH="${epoch}",ROB338_REPEAT="${repeat}",ROB338_LR="${LR}" \
            "${CELL_SCRIPT}"
        )"
        job_ids+=("${job_id}")
        echo "${job_id}|${partition}|${dataset}|${reward}|${epoch}|repeat${repeat}"
      done
    done
  done
done

dependency="$(IFS=:; echo "${job_ids[*]}")"
finalizer_id="$(
  sbatch --parsable \
    --partition="${FINALIZER_PARTITION}" \
    --dependency="afterany:${dependency}" \
    --export=ALL,REPO_DIR="${REPO_DIR}",LINEAR_ISSUE="${LINEAR_ISSUE}",RESULT_ROOT="${RESULT_ROOT}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",ROB338_FIXED_REWARDS="${SUMMARY_FIXED_REWARDS}",ROB338_DATASETS="${SUMMARY_DATASETS}",ROB338_EPOCHS="${SUMMARY_EPOCHS}",ROB338_REPEATS="${SUMMARY_REPEATS}",ROB338_LR="${LR}",CSV_NAME="${CSV_NAME}",QUEUED_COMMAND="${QUEUED_COMMAND}" \
    "${FINALIZER_SCRIPT}"
)"
echo "finalizer|${finalizer_id}|afterany:${dependency}"
