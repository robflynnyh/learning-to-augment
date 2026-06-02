#!/usr/bin/env bash
# Queue-safe ROB-186 plasticity EGGROLL GPU training launcher.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-186}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/exp/results/plasticity_eggroll}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RESULT_ROOT}/checkpoints}"
LOG_PATH="${LOG_PATH:-${RESULT_ROOT}/logs/rob186_plasticity_eggroll_gpu.log}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_DIR}/exp/configs/plasticity_eggroll.yaml}"
SCREEN_NAME="${SCREEN_NAME:-rob186_plasticity_eggroll_gpu}"
RUNNER_LABEL="${RUNNER_LABEL:-screen:${SCREEN_NAME}}"
QUEUED_COMMAND="${QUEUED_COMMAND:-/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/launch_rob186_plasticity_eggroll_gpu.sh}"
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"
CONDA_ENV="${CONDA_ENV:-/store/store4/software/bin/anaconda3/envs/flash_attn_pytorch2}"

on_exit() {
  status=$?
  set +e
  if [ "${ROB186_DISABLE_CALLBACK:-0}" = "1" ]; then
    exit "${status}"
  fi
  if [ -z "${LINEAR_API_KEY:-}" ] && [ "${ROB186_CALLBACK_DRY_RUN:-0}" != "1" ]; then
    echo "LINEAR_API_KEY is not set; cannot post Linear completion callback" >&2
    exit "${status}"
  fi
  callback_args=(
    --issue "${LINEAR_ISSUE}"
    --status-code "${status}"
    --log "${LOG_PATH}"
    --results "${RESULT_ROOT}"
    --screen-name "${SCREEN_NAME}"
    --runner-label "${RUNNER_LABEL}"
    --queued-command "${QUEUED_COMMAND}"
    --branch "${GIT_BRANCH}"
    --commit "${GIT_COMMIT}"
    --target-state "${CALLBACK_TARGET_STATE:-Todo}"
    --note "${CALLBACK_NOTE:-ROB-186 plasticity EGGROLL GPU launcher exited. Inspect ${LOG_PATH}, ${CHECKPOINT_DIR}, and ${RESULT_ROOT}/OUTCOME.md before final handoff.}"
    --tail-lines "${CALLBACK_TAIL_LINES:-80}"
    --max-log-chars "${CALLBACK_MAX_LOG_CHARS:-6000}"
    --max-comment-chars "${CALLBACK_MAX_COMMENT_CHARS:-10000}"
  )
  if [ "${ROB186_CALLBACK_CHECK_ONLY:-0}" = "1" ]; then
    callback_args+=(--check-only)
  fi
  if [ "${ROB186_CALLBACK_DRY_RUN:-0}" = "1" ]; then
    callback_args+=(--dry-run)
  fi
  python3 "${REPO_DIR}/scripts/callbacks/linear_experiment_callback.py" "${callback_args[@]}"
  callback_status=$?
  if [ "${callback_status}" -ne 0 ]; then
    echo "Linear completion callback failed with status ${callback_status}" >&2
  fi
  exit "${status}"
}
trap on_exit EXIT

set -euo pipefail

mkdir -p "$(dirname "${LOG_PATH}")" "${RESULT_ROOT}" "${CHECKPOINT_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "[rob186] branch=${GIT_BRANCH}"
echo "[rob186] commit=${GIT_COMMIT}"
echo "[rob186] config=${CONFIG_PATH}"
echo "[rob186] result_root=${RESULT_ROOT}"
echo "[rob186] checkpoint_dir=${CHECKPOINT_DIR}"
echo "[rob186] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

if [ "${ROB186_CALLBACK_ONLY:-0}" = "1" ]; then
  echo "[rob186] callback-only smoke path requested; exiting before training."
  exit 0
fi

if [ ! -f "${CONFIG_PATH}" ]; then
  echo "Missing config: ${CONFIG_PATH}" >&2
  exit 1
fi

SCRATCH_ROOT="${SCRATCH_ROOT:-${RESULT_ROOT}/scratch}"
mkdir -p "${SCRATCH_ROOT}" "${SCRATCH_ROOT}/wandb-cache" "${SCRATCH_ROOT}/wandb-config" "${RESULT_ROOT}/wandb"
export WANDB_DIR="${WANDB_DIR:-${RESULT_ROOT}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${SCRATCH_ROOT}/wandb-cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${SCRATCH_ROOT}/wandb-config}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export L2A_TEDLIUM3_LEGACY_DIR="${L2A_TEDLIUM3_LEGACY_DIR:-/store/store4/data/TEDLIUM_release-3/legacy}"
export L2A_EARNINGS22_DIR="${L2A_EARNINGS22_DIR:-/store/store4/data/earnings-22}"

if [ "${ROB186_SKIP_CONDA:-0}" != "1" ]; then
  source /store/store4/software/bin/anaconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV}"
fi

train_overrides=(
  "training.checkpoint_dir=${CHECKPOINT_DIR}"
  "training.model_save_path=${CHECKPOINT_DIR}/latest.pt"
  "training.summary_path=${RESULT_ROOT}/OUTCOME.md"
)

if [ "${ROB186_SMOKE:-0}" = "1" ]; then
  echo "[rob186] one-step smoke mode enabled."
  train_overrides+=(
    "training.split=dev"
    "training.num_steps=1"
    "training.checkpoint_every=1"
    "training.keep_last_checkpoints=1"
    "training.wandb_mode=offline"
    "training.wandb_name=rob186-plasticity-eggroll-smoke"
    "eggroll.num_candidates=2"
    "rollout.batch_size_recordings=1"
  )
fi

if [ -n "${ROB186_EXTRA_OVERRIDES:-}" ]; then
  read -r -a extra_overrides <<< "${ROB186_EXTRA_OVERRIDES}"
  train_overrides+=("${extra_overrides[@]}")
fi

python3 - "${CONFIG_PATH}" "${train_overrides[@]}" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf

config = OmegaConf.merge(OmegaConf.load(sys.argv[1]), OmegaConf.from_dotlist(sys.argv[2:]))
asr_path = Path(config.checkpointing.asr_model)
print(f"[rob186] resolved_asr_checkpoint={asr_path}")
if not asr_path.is_file():
    raise SystemExit(f"Missing ASR checkpoint: {asr_path}")
print(f"[rob186] wandb_enabled={config.training.get('wandb_enabled', False)}")
print(f"[rob186] wandb_mode={config.training.get('wandb_mode', 'unset')}")
print(f"[rob186] num_steps={config.training.num_steps}")
print(f"[rob186] keep_last_checkpoints={config.training.keep_last_checkpoints}")
PY

if [ "${ROB186_CONFIG_ONLY:-0}" = "1" ]; then
  echo "[rob186] config-only smoke path requested; exiting before training."
  exit 0
fi

command=(python3 exp/train_plasticity_eggroll.py --config "${CONFIG_PATH}")
for override in "${train_overrides[@]}"; do
  command+=(--set "${override}")
done

echo "[rob186] command=${command[*]}"
"${command[@]}"
