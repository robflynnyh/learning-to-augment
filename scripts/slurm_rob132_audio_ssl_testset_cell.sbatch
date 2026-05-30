#!/usr/bin/env bash
# Run one ROB-132 audio SSL fixed-reward test-set self-training eval cell on Stanage.

#SBATCH --job-name=rob132-cell
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=4-00:00:00
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/%x-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-learning-to-augment/ROB-132}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_tedlium_earnings22}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/mnt/parscratch/users/acp21rjf/rob132-audio-ssl-scratch}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/parscratch/users/acp21rjf/l2augment_model/reward_conditioned_mask_lm/audio_ssl_hubert_base_tedlium_per_utterance_transformer384_dropout0p1_500ep_lr1e3.pt}"
ASR_CKPT="${ASR_CKPT:-/mnt/parscratch/users/acp21rjf/spotify/rotary_pos_6l_256d_seq_sched/n_seq_sched_2048_rp_1/step_105360.pt}"
MASK_VAE_CKPT="${MASK_VAE_CKPT:-/mnt/parscratch/users/acp21rjf/l2augment_model/bvae/bvae_USINGTHISFORNOW_2048gpu.pt}"
ROB132_DATASET="${ROB132_DATASET:?ROB132_DATASET is required}"
ROB132_REWARD="${ROB132_REWARD:?ROB132_REWARD is required}"
ROB132_EPOCH="${ROB132_EPOCH:?ROB132_EPOCH is required}"
ROB132_LR="${ROB132_LR:-1e-5}"
LOG_DIR="${LOG_DIR:-${RESULT_ROOT}/logs/stanage}"

reward_tag="${ROB132_REWARD//./p}"
reward_tag="${reward_tag//-/m}"
method="AudioRewardConditionedMaskLMReward${reward_tag}"
tag="${ROB132_DATASET}_test_reward${reward_tag}_epoch${ROB132_EPOCH}_lr${ROB132_LR}"
config="${RESULT_ROOT}/${method}/configs/${tag}.yaml"
save_path="${RESULT_ROOT}/${method}/${tag}.txt"
log_path="${LOG_PATH:-${LOG_DIR}/${tag}-${SLURM_JOB_ID:-manual}.log}"

mkdir -p "${LOG_DIR}" "${SCRATCH_ROOT}" "$(dirname "${save_path}")"
exec > >(tee -a "${log_path}") 2>&1

echo "[rob132-stanage-cell] job_id=${SLURM_JOB_ID:-manual}"
echo "[rob132-stanage-cell] host=$(hostname)"
echo "[rob132-stanage-cell] repo=${REPO_DIR}"
echo "[rob132-stanage-cell] result_root=${RESULT_ROOT}"
echo "[rob132-stanage-cell] dataset=${ROB132_DATASET}"
echo "[rob132-stanage-cell] reward=${ROB132_REWARD}"
echo "[rob132-stanage-cell] epoch=${ROB132_EPOCH}"
echo "[rob132-stanage-cell] config=${config}"
echo "[rob132-stanage-cell] save_path=${save_path}"
echo "[rob132-stanage-cell] log_path=${log_path}"
echo "[rob132-stanage-cell] slurm_partition=${SLURM_JOB_PARTITION:-unknown}"
echo "[rob132-stanage-cell] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

for required_path in "${CHECKPOINT_PATH}" "${ASR_CKPT}" "${MASK_VAE_CKPT}" "${config}"; do
  if [ ! -e "${required_path}" ]; then
    echo "Missing required path: ${required_path}" >&2
    exit 1
  fi
done

if [ "${FORCE_RERUN:-0}" != "1" ] && [ -f "${save_path}" ] && grep -q "Updated_WER:" "${save_path}"; then
  echo "[rob132-stanage-cell] skipping completed cell: ${save_path}"
  exit 0
fi
if [ "${FORCE_RERUN:-0}" = "1" ]; then
  rm -f "${save_path}"
fi

export TMPDIR="${SCRATCH_ROOT}/tmp"
export TORCH_HOME="${SCRATCH_ROOT}/torch"
export HF_HOME="${SCRATCH_ROOT}/huggingface"
export WANDB_DIR="${RESULT_ROOT}/wandb"
export WANDB_CACHE_DIR="${SCRATCH_ROOT}/wandb-cache"
export WANDB_CONFIG_DIR="${SCRATCH_ROOT}/wandb-config"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/xdg-cache"
export MPLCONFIGDIR="${SCRATCH_ROOT}/matplotlib"
export L2A_TEDLIUM3_LEGACY_DIR="${L2A_TEDLIUM3_LEGACY_DIR:-/mnt/parscratch/users/acp21rjf/TEDLIUM_release-3/legacy}"
export L2A_EARNINGS22_DIR="${L2A_EARNINGS22_DIR:-/mnt/parscratch/users/acp21rjf/earnings22}"
export L2A_REV16_DIR="${L2A_REV16_DIR:-/mnt/parscratch/users/acp21rjf/rev_benchmark}"
export L2A_TAL_DIR="${L2A_TAL_DIR:-/mnt/parscratch/users/acp21rjf/this_american_life}"
export L2A_CHIME6_DIR="${L2A_CHIME6_DIR:-/mnt/parscratch/users/acp21rjf/chime6}"
mkdir -p "${TMPDIR}" "${TORCH_HOME}" "${HF_HOME}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"

cd "${REPO_DIR}/exp"
if command -v module >/dev/null 2>&1; then
  module load Anaconda3/2022.10
fi
source activate /mnt/parscratch/users/acp21rjf/conda/main
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/exp:/mnt/parscratch/users/acp21rjf/lcasr:/mnt/parscratch/users/acp21rjf/language_modelling${PYTHONPATH:+:${PYTHONPATH}}"

python - <<'PY'
import sys
import torch
import torchaudio

print("[rob132-stanage-cell] python_executable=" + sys.executable)
print("[rob132-stanage-cell] python_version=" + sys.version.split()[0])
print("[rob132-stanage-cell] torch_version=" + torch.__version__)
print("[rob132-stanage-cell] torchaudio_version=" + torchaudio.__version__)
print("[rob132-stanage-cell] cuda_available=" + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("[rob132-stanage-cell] cuda_device=" + torch.cuda.get_device_name(0))
PY

args=(python eval.py --config "${config}")
if [ -n "${ROB132_INDEXES:-}" ]; then
  read -r -a indexes <<< "${ROB132_INDEXES}"
  args+=(--indexes "${indexes[@]}")
fi
if [ "${ROB132_DONT_SAVE:-0}" = "1" ]; then
  args+=(--dont_save)
fi
echo "[rob132-stanage-cell] eval command: ${args[*]}"
"${args[@]}"

echo "[rob132-stanage-cell] finished"
