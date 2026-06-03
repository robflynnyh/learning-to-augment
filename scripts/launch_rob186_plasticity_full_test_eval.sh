#!/usr/bin/env bash
# Queue-safe ROB-186 full test-set plasticity evaluation launcher.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-186}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_DIR}/exp/configs/plasticity_eggroll.yaml}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/exp/results/plasticity_eggroll/full_test_seed_step0_vs_latest_$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_PATH="${LOG_PATH:-${RESULT_ROOT}/logs/rob186_plasticity_full_test_eval.log}"
SCREEN_NAME="${SCREEN_NAME:-rob186_plasticity_full_test_eval}"
RUNNER_LABEL="${RUNNER_LABEL:-screen:${SCREEN_NAME}}"
QUEUED_COMMAND="${QUEUED_COMMAND:-/store/store5/software/simple-gpu-schedule/with-gpu 1,2 --num 1 -- bash scripts/launch_rob186_plasticity_full_test_eval.sh}"
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"
CONDA_ENV="${CONDA_ENV:-/store/store4/software/bin/anaconda3/envs/flash_attn_pytorch2}"
LATEST_CHECKPOINT="${LATEST_CHECKPOINT:-${REPO_DIR}/exp/results/plasticity_eggroll/maxeta01_dense_n32_b8_bf16_multigpu_20260602T211050Z/checkpoints/latest.pt}"

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
    --note "${CALLBACK_NOTE:-ROB-186 full TED-LIUM test evaluation comparing seed ASR, step-0 random-init updater, and latest plasticity checkpoint exited. Inspect ${LOG_PATH}, ${RESULT_ROOT}/summary.json, ${RESULT_ROOT}/per_recording.csv, and ${RESULT_ROOT}/OUTCOME.md before final handoff.}"
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

mkdir -p "$(dirname "${LOG_PATH}")" "${RESULT_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "[rob186-eval] branch=${GIT_BRANCH}"
echo "[rob186-eval] commit=${GIT_COMMIT}"
echo "[rob186-eval] config=${CONFIG_PATH}"
echo "[rob186-eval] result_root=${RESULT_ROOT}"
echo "[rob186-eval] latest_checkpoint=${LATEST_CHECKPOINT}"
echo "[rob186-eval] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

if [ "${ROB186_CALLBACK_ONLY:-0}" = "1" ]; then
  echo "[rob186-eval] callback-only smoke path requested; exiting before evaluation."
  exit 0
fi

if [ ! -f "${CONFIG_PATH}" ]; then
  echo "Missing config: ${CONFIG_PATH}" >&2
  exit 1
fi
if [ ! -f "${LATEST_CHECKPOINT}" ]; then
  echo "Missing latest updater checkpoint: ${LATEST_CHECKPOINT}" >&2
  exit 1
fi

SCRATCH_ROOT="${SCRATCH_ROOT:-${RESULT_ROOT}/scratch}"
mkdir -p "${SCRATCH_ROOT}" "${RESULT_ROOT}/wandb"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TMPDIR="${TMPDIR:-/exp/exp4/acp21rjf/.tmp}"
export L2A_TEDLIUM3_LEGACY_DIR="${L2A_TEDLIUM3_LEGACY_DIR:-/store/store4/data/TEDLIUM_release-3/legacy}"
export L2A_EARNINGS22_DIR="${L2A_EARNINGS22_DIR:-/store/store4/data/earnings-22}"

if [ "${ROB186_SKIP_CONDA:-0}" != "1" ]; then
  source /store/store4/software/bin/anaconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV}"
fi

eval_overrides=(
  "evaluation.dataset=tedlium"
  "evaluation.split=test"
  "evaluation.num_recordings=all"
  "evaluation.batch_size_recordings=${ROB186_EVAL_BATCH_SIZE:-1}"
  "evaluation.device=cuda"
  "evaluation.variants=[seed_asr,step0_random_init,latest_checkpoint]"
  "evaluation.latest_checkpoint=${LATEST_CHECKPOINT}"
  "evaluation.result_dir=${RESULT_ROOT}"
  "training.wandb_enabled=false"
  "rollout.devices=[cuda:0]"
)

if [ "${ROB186_EVAL_SMOKE:-0}" = "1" ]; then
  echo "[rob186-eval] smoke mode enabled."
  eval_overrides=(
    "evaluation.dataset=tedlium"
    "evaluation.split=test"
    "evaluation.num_recordings=1"
    "evaluation.batch_size_recordings=1"
    "evaluation.device=cuda"
    "evaluation.variants=[seed_asr,step0_random_init,latest_checkpoint]"
    "evaluation.latest_checkpoint=${LATEST_CHECKPOINT}"
    "evaluation.result_dir=${RESULT_ROOT}"
    "training.wandb_enabled=false"
    "rollout.devices=[cuda:0]"
  )
fi

if [ -n "${ROB186_EVAL_EXTRA_OVERRIDES:-}" ]; then
  read -r -a extra_overrides <<< "${ROB186_EVAL_EXTRA_OVERRIDES}"
  eval_overrides+=("${extra_overrides[@]}")
fi

python3 - "${CONFIG_PATH}" "${eval_overrides[@]}" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf

config = OmegaConf.merge(OmegaConf.load(sys.argv[1]), OmegaConf.from_dotlist(sys.argv[2:]))
asr_path = Path(config.checkpointing.asr_model)
latest_path = Path(config.evaluation.latest_checkpoint)
print(f"[rob186-eval] resolved_asr_checkpoint={asr_path}")
print(f"[rob186-eval] resolved_latest_checkpoint={latest_path}")
print(f"[rob186-eval] result_dir={config.evaluation.result_dir}")
if not asr_path.is_file():
    raise SystemExit(f"Missing ASR checkpoint: {asr_path}")
if not latest_path.is_file():
    raise SystemExit(f"Missing latest checkpoint: {latest_path}")
PY

if [ "${ROB186_CONFIG_ONLY:-0}" = "1" ]; then
  echo "[rob186-eval] config-only smoke path requested; exiting before evaluation."
  exit 0
fi

command=(python3 exp/eval_plasticity_eggroll.py --config "${CONFIG_PATH}")
for override in "${eval_overrides[@]}"; do
  command+=(--set "${override}")
done

echo "[rob186-eval] command=${command[*]}"
"${command[@]}"
