#!/usr/bin/env bash
# Run one ROB-337 learnt-augmentation repeat eval cell on Stanage.

#SBATCH --job-name=rob337-cell
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=4-00:00:00
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/%x-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-learning-to-augment/ROB-337}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/exp/results/repro/learnt_augmentation_repeats/rob337}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/mnt/parscratch/users/acp21rjf/rob337-learnt-augmentation-scratch}"
ASR_CKPT="${ASR_CKPT:-/mnt/parscratch/users/acp21rjf/spotify/rotary_pos_6l_256d_seq_sched/n_seq_sched_2048_rp_1/step_105360.pt}"
UFMR_CKPT="${UFMR_CKPT:-/mnt/parscratch/users/acp21rjf/l2augment_model/ufm/test_wer/model.pt}"
UMLM_CKPT="${UMLM_CKPT:-/mnt/parscratch/users/acp21rjf/l2augment_model/UMLM/modelgpu.pt}"
MASK_VAE_CKPT="${MASK_VAE_CKPT:-/mnt/parscratch/users/acp21rjf/l2augment_model/bvae/bvae_USINGTHISFORNOW_2048gpu.pt}"
ROB337_METHOD="${ROB337_METHOD:?ROB337_METHOD is required}"
ROB337_DATASET="${ROB337_DATASET:?ROB337_DATASET is required}"
ROB337_EPOCH="${ROB337_EPOCH:?ROB337_EPOCH is required}"
ROB337_REPEAT="${ROB337_REPEAT:?ROB337_REPEAT is required}"
ROB337_LR="${ROB337_LR:-1e-5}"
LOG_DIR="${LOG_DIR:-${RESULT_ROOT}/logs/stanage}"

tag="${ROB337_DATASET}_epoch${ROB337_EPOCH}_lr${ROB337_LR}_repeat${ROB337_REPEAT}"
config="${RESULT_ROOT}/${ROB337_METHOD}/configs/${tag}.yaml"
save_path="${RESULT_ROOT}/${ROB337_METHOD}/${tag}.txt"
log_path="${LOG_PATH:-${LOG_DIR}/${ROB337_METHOD}_${tag}-${SLURM_JOB_ID:-manual}.log}"

mkdir -p "${LOG_DIR}" "${SCRATCH_ROOT}" "$(dirname "${save_path}")"
exec > >(tee -a "${log_path}") 2>&1

echo "[rob337-cell] job_id=${SLURM_JOB_ID:-manual}"
echo "[rob337-cell] host=$(hostname)"
echo "[rob337-cell] repo=${REPO_DIR}"
echo "[rob337-cell] result_root=${RESULT_ROOT}"
echo "[rob337-cell] method=${ROB337_METHOD}"
echo "[rob337-cell] dataset=${ROB337_DATASET}"
echo "[rob337-cell] epoch=${ROB337_EPOCH}"
echo "[rob337-cell] repeat=${ROB337_REPEAT}"
echo "[rob337-cell] lr=${ROB337_LR}"
echo "[rob337-cell] config=${config}"
echo "[rob337-cell] save_path=${save_path}"
echo "[rob337-cell] log_path=${log_path}"
echo "[rob337-cell] slurm_partition=${SLURM_JOB_PARTITION:-unknown}"
echo "[rob337-cell] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

for required_path in "${ASR_CKPT}" "${config}"; do
  if [ ! -e "${required_path}" ]; then
    echo "Missing required path: ${required_path}" >&2
    exit 1
  fi
done
if [ "${ROB337_METHOD}" = "UFMR" ] && [ ! -e "${UFMR_CKPT}" ]; then
  echo "Missing UFMR checkpoint: ${UFMR_CKPT}" >&2
  exit 1
fi
if [ "${ROB337_METHOD}" = "UVQLM" ]; then
  for required_path in "${UMLM_CKPT}" "${MASK_VAE_CKPT}"; do
    if [ ! -e "${required_path}" ]; then
      echo "Missing UVQLM checkpoint dependency: ${required_path}" >&2
      exit 1
    fi
  done
fi

if [ "${FORCE_RERUN:-0}" != "1" ] && [ -f "${save_path}" ] && grep -q "Updated_WER:" "${save_path}"; then
  echo "[rob337-cell] skipping completed cell: ${save_path}"
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
PYTHON_BIN="${PYTHON_BIN:-/mnt/parscratch/users/acp21rjf/conda/main/bin/python}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/exp:/mnt/parscratch/users/acp21rjf/lcasr:/mnt/parscratch/users/acp21rjf/language_modelling${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" - <<'PY'
import sys
import torch
import torchaudio

print("[rob337-cell] python_executable=" + sys.executable)
print("[rob337-cell] python_version=" + sys.version.split()[0])
print("[rob337-cell] torch_version=" + torch.__version__)
print("[rob337-cell] torchaudio_version=" + torchaudio.__version__)
print("[rob337-cell] cuda_available=" + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("[rob337-cell] cuda_device=" + torch.cuda.get_device_name(0))
PY

args=("${PYTHON_BIN}" eval.py --config "${config}")
if [ -n "${ROB337_INDEXES:-}" ]; then
  read -r -a indexes <<< "${ROB337_INDEXES}"
  args+=(--indexes "${indexes[@]}")
fi
if [ -n "${ROB337_MAX_STEPS:-}" ]; then
  args+=(--max_steps "${ROB337_MAX_STEPS}")
fi
if [ "${ROB337_DONT_SAVE:-0}" = "1" ]; then
  args+=(--dont_save)
fi
echo "[rob337-cell] eval command: ${args[*]}"
"${args[@]}"

if ! grep -q "Updated_WER:" "${save_path}"; then
  echo "[rob337-cell] expected result was not written: ${save_path}" >&2
  exit 1
fi
echo "[rob337-cell] finished"
