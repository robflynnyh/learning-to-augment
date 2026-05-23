#!/usr/bin/env bash
# Callback-backed ROB-124 512-dim dropout no-audio reward-conditioned mask LM training.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-124}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_512_dropout}"
LOG_PATH="${LOG_PATH:-${RESULT_ROOT}/logs/rob124_no_audio_reward_conditioned_mask_lm_512d_dropout0p1_500ep_lr1e3.log}"
SCREEN_LOG_PATH="${SCREEN_LOG_PATH:-${RESULT_ROOT}/logs/rob124_no_audio_reward_conditioned_mask_lm_512d_dropout0p1_500ep_lr1e3.screen.log}"
SCREEN_NAME="${SCREEN_NAME:-rob124-reward-conditioned-mask-lm-512d-dropout0p1}"
RUNNER_LABEL="${RUNNER_LABEL:-screen:${SCREEN_NAME}}"
CONFIG_PATH="${CONFIG_PATH:-exp/configs/reward_conditioned_lm/no_audio_conditioning/tedlium_per_utterance_512d_dropout0p1_500ep_lr1e3.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_512d_dropout0p1_500ep_lr1e3.pt}"
TMP_CHECKPOINT_PATH="${TMP_CHECKPOINT_PATH:-/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_512d_dropout0p1_500ep_lr1e3_tmp.pt}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/exp/exp4/acp21rjf/rob124-512-scratch}"
QUEUED_COMMAND="${QUEUED_COMMAND:-screen -L -Logfile ${SCREEN_LOG_PATH} -dmS ${SCREEN_NAME} bash -lc 'cd ${REPO_DIR} && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob124_reward_conditioned_mask_lm_training_512d_dropout0p1.sh'}"
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"

on_exit() {
  status=$?
  set +e
  if [ "${ROB124_512_DISABLE_CALLBACK:-0}" = "1" ]; then
    exit "${status}"
  fi
  if [ -z "${LINEAR_API_KEY:-}" ]; then
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
    --note "${CALLBACK_NOTE:-ROB-124 512-dim dropout no-audio reward-conditioned mask LM training wrapper exited. Check checkpoint path ${CHECKPOINT_PATH}, run post-training reward-control sanity checks, and compare against the ROB-117 baseline plus ROB-124 384/dropout checkpoint before final handoff.}"
    --tail-lines "${CALLBACK_TAIL_LINES:-80}"
    --max-log-chars "${CALLBACK_MAX_LOG_CHARS:-6000}"
    --max-comment-chars "${CALLBACK_MAX_COMMENT_CHARS:-10000}"
  )
  if [ "${ROB124_512_CALLBACK_CHECK_ONLY:-0}" = "1" ]; then
    callback_args+=(--check-only)
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

mkdir -p "$(dirname "${LOG_PATH}")" "${RESULT_ROOT}" "${SCRATCH_ROOT}" "$(dirname "${CHECKPOINT_PATH}")"
exec > >(tee -a "${LOG_PATH}") 2>&1

export REPO_DIR RESULT_ROOT LOG_PATH SCREEN_LOG_PATH SCREEN_NAME RUNNER_LABEL
export CONFIG_PATH CHECKPOINT_PATH TMP_CHECKPOINT_PATH QUEUED_COMMAND GIT_BRANCH GIT_COMMIT
export TMPDIR="${SCRATCH_ROOT}/tmp"
export WANDB_DIR="${RESULT_ROOT}/wandb"
export WANDB_CACHE_DIR="${SCRATCH_ROOT}/wandb-cache"
export WANDB_CONFIG_DIR="${SCRATCH_ROOT}/wandb-config"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/xdg-cache"
export MPLCONFIGDIR="${SCRATCH_ROOT}/matplotlib"
mkdir -p "${TMPDIR}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"

echo "[rob124-512] branch=${GIT_BRANCH}"
echo "[rob124-512] commit=${GIT_COMMIT}"
echo "[rob124-512] host=$(hostname)"
echo "[rob124-512] result_root=${RESULT_ROOT}"
echo "[rob124-512] config=${CONFIG_PATH}"
echo "[rob124-512] checkpoint=${CHECKPOINT_PATH}"
echo "[rob124-512] tmp_checkpoint=${TMP_CHECKPOINT_PATH}"
echo "[rob124-512] log_path=${LOG_PATH}"
echo "[rob124-512] screen_log_path=${SCREEN_LOG_PATH}"
echo "[rob124-512] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[rob124-512] tmpdir=${TMPDIR}"

bash -ic 'python - <<'"'"'PY'"'"'
import sys
import torch

print("[rob124-512] python_executable=" + sys.executable)
print("[rob124-512] python_version=" + sys.version.split()[0])
print("[rob124-512] torch_version=" + torch.__version__)
print("[rob124-512] cuda_available=" + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("[rob124-512] cuda_device=" + torch.cuda.get_device_name(0))
PY'

if [ "${ROB124_512_CALLBACK_ONLY:-0}" = "1" ]; then
  echo "[rob124-512] callback-only smoke path requested; exiting before training."
  exit 0
fi

unset WANDB_MODE
bash -ic 'cd "$REPO_DIR"; export PYTHONPATH="$REPO_DIR:$REPO_DIR/exp:/exp/exp4/acp21rjf/long-context-asr:/exp/exp4/acp21rjf/language_modelling${PYTHONPATH:+:$PYTHONPATH}"; python exp/train_freq_mask.py --config "$CONFIG_PATH"'

echo "[rob124-512] training command finished"
if [ -f "${CHECKPOINT_PATH}" ]; then
  ls -lh "${CHECKPOINT_PATH}"
else
  echo "[rob124-512] expected checkpoint missing: ${CHECKPOINT_PATH}" >&2
  exit 2
fi
