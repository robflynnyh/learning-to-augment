#!/usr/bin/env bash
# Submit ROB-337 learnt-augmentation repeat eval cells to Stanage.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/parscratch/users/acp21rjf/symphony-workspaces-learning-to-augment/ROB-337}"
LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-337}"
BASE_REPRO_DIR="${BASE_REPRO_DIR:-${REPO_DIR}/exp/results/repro}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/exp/results/repro/learnt_augmentation_repeats/rob337}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/mnt/parscratch/users/acp21rjf/rob337-learnt-augmentation-scratch}"
ASR_CKPT="${ASR_CKPT:-/mnt/parscratch/users/acp21rjf/spotify/rotary_pos_6l_256d_seq_sched/n_seq_sched_2048_rp_1/step_105360.pt}"
UFMR_CKPT="${UFMR_CKPT:-/mnt/parscratch/users/acp21rjf/l2augment_model/ufm/test_wer/model.pt}"
UMLM_CKPT="${UMLM_CKPT:-/mnt/parscratch/users/acp21rjf/l2augment_model/UMLM/modelgpu.pt}"
MASK_VAE_CKPT="${MASK_VAE_CKPT:-/mnt/parscratch/users/acp21rjf/l2augment_model/bvae/bvae_USINGTHISFORNOW_2048gpu.pt}"
CELL_SCRIPT="${CELL_SCRIPT:-scripts/slurm_rob337_learnt_augmentation_cell.sbatch}"
FINALIZER_SCRIPT="${FINALIZER_SCRIPT:-scripts/slurm_rob337_learnt_augmentation_finalizer.sbatch}"
FINALIZER_PARTITION="${FINALIZER_PARTITION:-sheffield}"
METHODS="${ROB337_METHODS:-UFMR RFM RMM UVQLM}"
DATASETS="${ROB337_DATASETS:-tedlium earnings22 rev16 TAL chime6}"
EPOCHS="${ROB337_EPOCHS:-1 5}"
NEW_REPEATS="${ROB337_NEW_REPEATS:-2 3}"
SUMMARY_REPEATS="${ROB337_SUMMARY_REPEATS:-1 2 3}"
LR="${ROB337_LR:-1e-5}"
PARTITIONS="${ROB337_PARTITIONS:-gpu-h100-nvl gpu-h100 gpu}"
QUEUED_JOBS_PATH="${QUEUED_JOBS_PATH:-${RESULT_ROOT}/queued_jobs.tsv}"

cd "${REPO_DIR}"
mkdir -p "${RESULT_ROOT}"

if command -v module >/dev/null 2>&1; then
  module load Anaconda3/2022.10
fi
if [ -d /mnt/parscratch/users/acp21rjf/conda/main ]; then
  source activate /mnt/parscratch/users/acp21rjf/conda/main
fi

python3 - "${BASE_REPRO_DIR}" "${RESULT_ROOT}" "${ASR_CKPT}" "${UFMR_CKPT}" "${UMLM_CKPT}" "${MASK_VAE_CKPT}" "${METHODS}" "${DATASETS}" "${EPOCHS}" "${NEW_REPEATS}" "${LR}" "${LINEAR_ISSUE}" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf

base_repro = Path(sys.argv[1])
result_root = Path(sys.argv[2])
asr_ckpt = sys.argv[3]
ufmr_ckpt = sys.argv[4]
umlm_ckpt = sys.argv[5]
mask_vae_ckpt = sys.argv[6]
methods = tuple(sys.argv[7].split())
datasets = tuple(sys.argv[8].split())
epochs = tuple(int(item) for item in sys.argv[9].split())
repeats = tuple(int(item) for item in sys.argv[10].split())
lr = sys.argv[11]
linear_issue = sys.argv[12]

def seed_for_repeat(repeat: int) -> int:
    if repeat == 1:
        return 123456
    return repeat * 100000 + 23456

for method in methods:
    (result_root / method / "configs").mkdir(parents=True, exist_ok=True)
    for dataset_tag in datasets:
        for epoch_count in epochs:
            template = base_repro / method / "configs" / f"{dataset_tag}_epoch{epoch_count}_lr{lr}.yaml"
            if not template.exists():
                raise FileNotFoundError(template)
            for repeat in repeats:
                seed = seed_for_repeat(repeat)
                tag = f"{dataset_tag}_epoch{epoch_count}_lr{lr}_repeat{repeat}"
                save_path = result_root / method / f"{tag}.txt"
                config_path = result_root / method / "configs" / f"{tag}.yaml"
                cfg = OmegaConf.load(template)
                cfg.checkpointing.asr_model = asr_ckpt
                cfg.training.random_seed = seed
                if method == "UFMR":
                    cfg.training.model_save_path = ufmr_ckpt
                    cfg.training.tmp_model_save_path = ufmr_ckpt
                if method == "UVQLM":
                    cfg.training.model_save_path = umlm_ckpt
                    cfg.training.tmp_model_save_path = umlm_ckpt
                    cfg.policy.config.mask_vae_state_dict_path = mask_vae_ckpt
                if "augmentation_config" not in cfg.evaluation:
                    cfg.evaluation.augmentation_config = {}
                cfg.evaluation.augmentation_config.seed = seed
                cfg.evaluation.id = (
                    f"{linear_issue}-{dataset_tag}-test-{method}-epoch{epoch_count}-"
                    f"lr{lr}-repeat{repeat}-seed{seed}"
                )
                cfg.evaluation.save_path = str(save_path)
                OmegaConf.save(cfg, config_path)
                print(f"[rob337-submit] wrote config {config_path}")
PY

if [ "${ROB337_SUBMIT_CONFIG_ONLY:-0}" = "1" ]; then
  echo "[rob337-submit] config-only mode requested; exiting before sbatch submission."
  exit 0
fi

read -r -a partition_list <<< "${PARTITIONS}"
job_ids=()
cell_index=0
{
  printf 'job_id\tpartition\tmethod\tdataset\tepoch\trepeat\tresult_path\tconfig_path\n'
  for method in ${METHODS}; do
    for dataset in ${DATASETS}; do
      for epoch in ${EPOCHS}; do
        for repeat in ${NEW_REPEATS}; do
          partition="${partition_list[$((cell_index % ${#partition_list[@]}))]}"
          cell_index=$((cell_index + 1))
          job_name="r337-${method:0:3}-${dataset:0:3}-e${epoch}-r${repeat}"
          tag="${dataset}_epoch${epoch}_lr${LR}_repeat${repeat}"
          result_path="${RESULT_ROOT}/${method}/${tag}.txt"
          config_path="${RESULT_ROOT}/${method}/configs/${tag}.yaml"
          job_id="$(
            sbatch --parsable \
              --job-name="${job_name}" \
              --partition="${partition}" \
              --export=ALL,REPO_DIR="${REPO_DIR}",RESULT_ROOT="${RESULT_ROOT}",SCRATCH_ROOT="${SCRATCH_ROOT}",ASR_CKPT="${ASR_CKPT}",UFMR_CKPT="${UFMR_CKPT}",UMLM_CKPT="${UMLM_CKPT}",MASK_VAE_CKPT="${MASK_VAE_CKPT}",ROB337_METHOD="${method}",ROB337_DATASET="${dataset}",ROB337_EPOCH="${epoch}",ROB337_REPEAT="${repeat}",ROB337_LR="${LR}" \
              "${CELL_SCRIPT}"
          )"
          job_ids+=("${job_id}")
          printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${job_id}" "${partition}" "${method}" "${dataset}" "${epoch}" "${repeat}" "${result_path}" "${config_path}"
          echo "${job_id}|${partition}|${method}|${dataset}|${epoch}|repeat${repeat}" >&2
        done
      done
    done
  done
} > "${QUEUED_JOBS_PATH}"

dependency="$(IFS=:; echo "${job_ids[*]}")"
queued_command="REPO_DIR=${REPO_DIR} scripts/submit_rob337_learnt_augmentation_repeats_stanage.sh"
finalizer_id="$(
  sbatch --parsable \
    --partition="${FINALIZER_PARTITION}" \
    --dependency="afterany:${dependency}" \
    --export=ALL,REPO_DIR="${REPO_DIR}",LINEAR_ISSUE="${LINEAR_ISSUE}",BASE_REPRO_DIR="${BASE_REPRO_DIR}",RESULT_ROOT="${RESULT_ROOT}",ROB337_METHODS="${METHODS}",ROB337_DATASETS="${DATASETS}",ROB337_EPOCHS="${EPOCHS}",ROB337_REPEATS="${SUMMARY_REPEATS}",ROB337_LR="${LR}",QUEUED_JOBS_PATH="${QUEUED_JOBS_PATH}",QUEUED_COMMAND="${queued_command}" \
    "${FINALIZER_SCRIPT}"
)"
{
  printf '\nfinalizer_id\tpartition\tdependency\n'
  printf '%s\t%s\t%s\n' "${finalizer_id}" "${FINALIZER_PARTITION}" "afterany:${dependency}"
} >> "${QUEUED_JOBS_PATH}"
echo "finalizer|${finalizer_id}|afterany:${dependency}"
