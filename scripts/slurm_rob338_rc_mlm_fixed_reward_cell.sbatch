#!/usr/bin/env bash
# Run one ROB-338 RC-MLM fixed-reward repeat cell on Stanage.

#SBATCH --job-name=rob338-cell
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=gpu
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=4-00:00:00
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/%x-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-learning-to-augment/ROB-338}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/mnt/parscratch/users/acp21rjf/rob338-rc-mlm-fixed-reward-repeats-scratch}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/parscratch/users/acp21rjf/l2augment_model/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt}"
ASR_CKPT="${ASR_CKPT:-/mnt/parscratch/users/acp21rjf/spotify/rotary_pos_6l_256d_seq_sched/n_seq_sched_2048_rp_1/step_105360.pt}"
MASK_VAE_CKPT="${MASK_VAE_CKPT:-/mnt/parscratch/users/acp21rjf/l2augment_model/bvae/bvae_USINGTHISFORNOW_2048gpu.pt}"
ROB338_DATASET="${ROB338_DATASET:?ROB338_DATASET is required}"
ROB338_REWARD="${ROB338_REWARD:?ROB338_REWARD is required}"
ROB338_EPOCH="${ROB338_EPOCH:?ROB338_EPOCH is required}"
ROB338_REPEAT="${ROB338_REPEAT:?ROB338_REPEAT is required}"
ROB338_LR="${ROB338_LR:-1e-5}"
ROB338_GPU_TYPE="${ROB338_GPU_TYPE:-h100}"
ROB338_PYTHON_BIN="${ROB338_PYTHON_BIN:-/mnt/parscratch/users/acp21rjf/conda/rob338-el9/bin/python}"
ROB338_LCASR_ROOT="${ROB338_LCASR_ROOT:-/mnt/parscratch/users/acp21rjf/long-context-asr}"
ROB338_WHISPER_ROOT="${ROB338_WHISPER_ROOT:-/mnt/parscratch/users/acp21rjf/openai-whisper}"
LOG_DIR="${LOG_DIR:-${RESULT_ROOT}/logs/stanage/rob338}"

reward_token="${ROB338_REWARD//./p}"
reward_token="${reward_token//-/m}"
if [ "${ROB338_REWARD}" = "0.0" ]; then
  reward_token="0"
elif [ "${ROB338_REWARD}" = "1.0" ]; then
  reward_token="1"
fi

method="RewardConditionedMaskLMReward${reward_token}"
repeat_suffix=""
if [ "${ROB338_REPEAT}" != "1" ]; then
  repeat_suffix="_repeat${ROB338_REPEAT}"
fi
tag="${ROB338_DATASET}_test_epoch${ROB338_EPOCH}_lr${ROB338_LR}${repeat_suffix}"
config="${RESULT_ROOT}/${method}/configs/${tag}.yaml"
save_path="${RESULT_ROOT}/${method}/${tag}.txt"
log_path="${LOG_PATH:-${LOG_DIR}/${tag}-reward${reward_token}-${SLURM_JOB_ID:-manual}.log}"

mkdir -p "${LOG_DIR}" "${SCRATCH_ROOT}" "$(dirname "${save_path}")"
exec > >(tee -a "${log_path}") 2>&1

echo "[rob338-stanage-cell] job_id=${SLURM_JOB_ID:-manual}"
echo "[rob338-stanage-cell] host=$(hostname)"
echo "[rob338-stanage-cell] repo=${REPO_DIR}"
echo "[rob338-stanage-cell] result_root=${RESULT_ROOT}"
echo "[rob338-stanage-cell] dataset=${ROB338_DATASET}"
echo "[rob338-stanage-cell] reward=${ROB338_REWARD}"
echo "[rob338-stanage-cell] epoch=${ROB338_EPOCH}"
echo "[rob338-stanage-cell] repeat=${ROB338_REPEAT}"
echo "[rob338-stanage-cell] seed=$((123456 + ROB338_REPEAT - 1))"
echo "[rob338-stanage-cell] config=${config}"
echo "[rob338-stanage-cell] save_path=${save_path}"
echo "[rob338-stanage-cell] log_path=${log_path}"
echo "[rob338-stanage-cell] slurm_partition=${SLURM_JOB_PARTITION:-unknown}"
echo "[rob338-stanage-cell] gpu_type=${ROB338_GPU_TYPE}"
echo "[rob338-stanage-cell] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
if [ -f /etc/os-release ]; then
  echo "[rob338-stanage-cell] os_release=$(grep '^PRETTY_NAME=' /etc/os-release | cut -d= -f2- | tr -d '\"')"
fi

for required_path in "${CHECKPOINT_PATH}" "${ASR_CKPT}" "${MASK_VAE_CKPT}" "${config}" "${ROB338_PYTHON_BIN}" "${ROB338_LCASR_ROOT}/lcasr/__init__.py" "${ROB338_WHISPER_ROOT}/whisper/normalizers/__init__.py"; do
  if [ ! -e "${required_path}" ]; then
    echo "Missing required path: ${required_path}" >&2
    exit 1
  fi
done

if [ "${ROB338_DONT_SAVE:-0}" != "1" ] && [ "${FORCE_RERUN:-0}" != "1" ] && [ -f "${save_path}" ] && grep -q "Updated_WER:" "${save_path}"; then
  echo "[rob338-stanage-cell] skipping completed cell: ${save_path}"
  exit 0
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
if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
elif [ -f /usr/share/lmod/lmod/init/bash ]; then
  source /usr/share/lmod/lmod/init/bash
fi
if command -v module >/dev/null 2>&1; then
  module purge
fi
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/exp:${ROB338_LCASR_ROOT}:${ROB338_WHISPER_ROOT}:/mnt/parscratch/users/acp21rjf/language_modelling${PYTHONPATH:+:${PYTHONPATH}}"

"${ROB338_PYTHON_BIN}" - <<'PY'
import sys
import lcasr
import torch
import torchaudio
import whisper
from whisper.normalizers import EnglishTextNormalizer

EnglishTextNormalizer()

print("[rob338-stanage-cell] python_executable=" + sys.executable)
print("[rob338-stanage-cell] python_version=" + sys.version.split()[0])
print("[rob338-stanage-cell] torch_version=" + torch.__version__)
print("[rob338-stanage-cell] torchaudio_version=" + torchaudio.__version__)
print("[rob338-stanage-cell] lcasr_source=" + lcasr.__file__)
print("[rob338-stanage-cell] whisper_version=" + whisper.__version__)
print("[rob338-stanage-cell] whisper_source=" + whisper.__file__)
cuda_available = torch.cuda.is_available()
print("[rob338-stanage-cell] cuda_available=" + str(cuda_available))
if not cuda_available:
    raise SystemExit(
        "ROB-338 requires an allocated CUDA GPU; refusing CPU fallback in a GPU job"
    )
print("[rob338-stanage-cell] cuda_device=" + torch.cuda.get_device_name(0))
PY

if [ "${ROB338_DONT_SAVE:-0}" != "1" ] && [ "${FORCE_RERUN:-0}" = "1" ]; then
  rm -f "${save_path}"
fi

args=("${ROB338_PYTHON_BIN}" eval.py --config "${config}")
if [ -n "${ROB338_INDEXES:-}" ]; then
  read -r -a indexes <<< "${ROB338_INDEXES}"
  args+=(--indexes "${indexes[@]}")
fi
if [ -n "${ROB338_MAX_STEPS:-}" ]; then
  args+=(--max_steps "${ROB338_MAX_STEPS}")
fi
if [ "${ROB338_DONT_SAVE:-0}" = "1" ]; then
  args+=(--dont_save)
fi
echo "[rob338-stanage-cell] eval command: ${args[*]}"
"${args[@]}"

echo "[rob338-stanage-cell] finished"
