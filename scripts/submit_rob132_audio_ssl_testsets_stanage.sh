#!/usr/bin/env bash
# Submit the remaining ROB-132 fixed-reward test-set eval cells to Stanage.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-learning-to-augment/ROB-132}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_tedlium_earnings22}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/mnt/parscratch/users/acp21rjf/rob132-audio-ssl-scratch}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/parscratch/users/acp21rjf/l2augment_model/reward_conditioned_mask_lm/audio_ssl_hubert_base_tedlium_per_utterance_transformer384_dropout0p1_500ep_lr1e3.pt}"
ASR_CKPT="${ASR_CKPT:-/mnt/parscratch/users/acp21rjf/spotify/rotary_pos_6l_256d_seq_sched/n_seq_sched_2048_rp_1/step_105360.pt}"
MASK_VAE_CKPT="${MASK_VAE_CKPT:-/mnt/parscratch/users/acp21rjf/l2augment_model/bvae/bvae_USINGTHISFORNOW_2048gpu.pt}"
CELL_SCRIPT="${CELL_SCRIPT:-scripts/slurm_rob132_audio_ssl_testset_cell.sbatch}"
FINALIZER_SCRIPT="${FINALIZER_SCRIPT:-scripts/slurm_rob132_audio_ssl_testset_finalizer.sbatch}"

cd "${REPO_DIR}"

cells=(
  "earnings22 1.0 5 gpu-h100-nvl"
  "tedlium 0.0 1 hp-a100"
  "tedlium 0.0 5 gpu-h100"
  "earnings22 0.0 1 hp-h100"
  "earnings22 0.0 5 gpu"
)

job_ids=()
for cell in "${cells[@]}"; do
  read -r dataset reward epoch partition <<< "${cell}"
  reward_tag="${reward//./p}"
  reward_tag="${reward_tag//-/m}"
  job_name="r132-${dataset:0:3}-r${reward_tag}-e${epoch}"
  job_id="$(
    sbatch --parsable \
      --job-name="${job_name}" \
      --partition="${partition}" \
      --export=ALL,REPO_DIR="${REPO_DIR}",RESULT_ROOT="${RESULT_ROOT}",SCRATCH_ROOT="${SCRATCH_ROOT}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",ASR_CKPT="${ASR_CKPT}",MASK_VAE_CKPT="${MASK_VAE_CKPT}",ROB132_DATASET="${dataset}",ROB132_REWARD="${reward}",ROB132_EPOCH="${epoch}" \
      "${CELL_SCRIPT}"
  )"
  job_ids+=("${job_id}")
  echo "${job_id}|${partition}|${dataset}|${reward}|${epoch}"
done

dependency="$(IFS=:; echo "${job_ids[*]}")"
finalizer_id="$(
  sbatch --parsable \
    --dependency="afterany:${dependency}" \
    --export=ALL,REPO_DIR="${REPO_DIR}",RESULT_ROOT="${RESULT_ROOT}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",QUEUED_COMMAND="scripts/submit_rob132_audio_ssl_testsets_stanage.sh" \
    "${FINALIZER_SCRIPT}"
)"
echo "finalizer|${finalizer_id}|afterany:${dependency}"
