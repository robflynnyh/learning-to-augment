#!/usr/bin/env bash
# Run one ROB-340 large-ASR UFMR/RFM repeat cell on Stanage.

#SBATCH --job-name=rob340-cell
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=130GB
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/%x-%j.out
#SBATCH --error=/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-learning-to-augment/ROB-340}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/mnt/parscratch/users/acp21rjf/rob340-large-asr-repeats-scratch}"
ASR_CKPT="${ASR_CKPT:-/mnt/parscratch/users/acp21rjf/spotify/rotary_pos_6l_256d_seq_sched/n_seq_sched_2048_rp_1/step_105360.pt}"
UFMR_CKPT="${UFMR_CKPT:-/mnt/parscratch/users/acp21rjf/l2augment_model/ufm/test_wer/model.pt}"
ROB340_METHOD="${ROB340_METHOD:?ROB340_METHOD is required}"
ROB340_DATASET="${ROB340_DATASET:?ROB340_DATASET is required}"
ROB340_REPEAT="${ROB340_REPEAT:?ROB340_REPEAT is required}"
ROB340_LR="${ROB340_LR:-1e-5}"
ROB340_EPOCH="${ROB340_EPOCH:-1}"
LOG_DIR="${LOG_DIR:-${RESULT_ROOT}/../logs/stanage}"

repeat_suffix=""
if [ "${ROB340_REPEAT}" != "1" ]; then
  repeat_suffix="_repeat${ROB340_REPEAT}"
fi
tag="${ROB340_DATASET}_epoch${ROB340_EPOCH}_lr${ROB340_LR}${repeat_suffix}"
config="${RESULT_ROOT}/${ROB340_METHOD}/configs/${tag}.yaml"
save_path="${RESULT_ROOT}/${ROB340_METHOD}/${tag}.txt"
log_path="${LOG_PATH:-${LOG_DIR}/${ROB340_METHOD}_${tag}-${SLURM_JOB_ID:-manual}.log}"

mkdir -p "${LOG_DIR}" "${SCRATCH_ROOT}" "$(dirname "${save_path}")"
exec > >(tee -a "${log_path}") 2>&1

echo "[rob340-stanage-cell] job_id=${SLURM_JOB_ID:-manual}"
echo "[rob340-stanage-cell] host=$(hostname)"
echo "[rob340-stanage-cell] repo=${REPO_DIR}"
echo "[rob340-stanage-cell] result_root=${RESULT_ROOT}"
echo "[rob340-stanage-cell] method=${ROB340_METHOD}"
echo "[rob340-stanage-cell] dataset=${ROB340_DATASET}"
echo "[rob340-stanage-cell] repeat=${ROB340_REPEAT}"
echo "[rob340-stanage-cell] epoch=${ROB340_EPOCH}"
echo "[rob340-stanage-cell] lr=${ROB340_LR}"
echo "[rob340-stanage-cell] config=${config}"
echo "[rob340-stanage-cell] save_path=${save_path}"
echo "[rob340-stanage-cell] log_path=${log_path}"
echo "[rob340-stanage-cell] slurm_partition=${SLURM_JOB_PARTITION:-unknown}"
echo "[rob340-stanage-cell] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

for required_path in "${ASR_CKPT}" "${config}"; do
  if [ ! -e "${required_path}" ]; then
    echo "Missing required path: ${required_path}" >&2
    exit 1
  fi
done
if [ "${ROB340_METHOD}" = "UFMR" ] && [ ! -e "${UFMR_CKPT}" ]; then
  echo "Missing UFMR checkpoint: ${UFMR_CKPT}" >&2
  exit 1
fi

if [ "${FORCE_RERUN:-0}" != "1" ] && [ -f "${save_path}" ] && grep -q "Updated_WER:" "${save_path}"; then
  echo "[rob340-stanage-cell] skipping completed cell: ${save_path}"
  exit 0
fi
if [ "${FORCE_RERUN:-0}" = "1" ]; then
  rm -f "${save_path}"
fi

export TMPDIR="${SCRATCH_ROOT}/tmp"
export TORCH_HOME="${SCRATCH_ROOT}/torch"
export HF_HOME="${SCRATCH_ROOT}/huggingface"
export WANDB_DIR="${RESULT_ROOT}/../wandb"
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

print("[rob340-stanage-cell] python_executable=" + sys.executable)
print("[rob340-stanage-cell] python_version=" + sys.version.split()[0])
print("[rob340-stanage-cell] torch_version=" + torch.__version__)
print("[rob340-stanage-cell] torchaudio_version=" + torchaudio.__version__)
print("[rob340-stanage-cell] cuda_available=" + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("[rob340-stanage-cell] cuda_device=" + torch.cuda.get_device_name(0))
PY

args=(python eval.py --config "${config}")
if [ -n "${ROB340_INDEXES:-}" ]; then
  read -r -a indexes <<< "${ROB340_INDEXES}"
  args+=(--indexes "${indexes[@]}")
fi
if [ -n "${ROB340_MAX_STEPS:-}" ]; then
  args+=(--max_steps "${ROB340_MAX_STEPS}")
fi
if [ "${ROB340_DONT_SAVE:-0}" = "1" ]; then
  args+=(--dont_save)
fi
echo "[rob340-stanage-cell] eval command: ${args[*]}"
"${args[@]}"

echo "[rob340-stanage-cell] finished"
