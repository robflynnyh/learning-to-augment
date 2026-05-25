#!/usr/bin/env bash
# Callback-backed ROB-132 audio+reward-conditioned mask LM training.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-132}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384}"
LOG_PATH="${LOG_PATH:-${RESULT_ROOT}/logs/rob132_audio_ssl_hubert_base_transformer384_dropout0p1_500ep_lr1e3.log}"
SCREEN_LOG_PATH="${SCREEN_LOG_PATH:-${RESULT_ROOT}/logs/rob132_audio_ssl_hubert_base_transformer384_dropout0p1_500ep_lr1e3.screen.log}"
SCREEN_NAME="${SCREEN_NAME:-rob132-audio-ssl-mask-lm-transformer384}"
RUNNER_LABEL="${RUNNER_LABEL:-screen:${SCREEN_NAME}}"
CONFIG_PATH="${CONFIG_PATH:-exp/configs/reward_conditioned_lm/audio_ssl_conditioning/tedlium_per_utterance_hubert_base_transformer384_dropout0p1_500ep_lr1e3.yaml}"
SMOKE_CONFIG_PATH="${SMOKE_CONFIG_PATH:-exp/configs/reward_conditioned_lm/audio_ssl_conditioning/tedlium_per_utterance_hubert_base_transformer384_dropout0p1_smoke.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/audio_ssl_hubert_base_tedlium_per_utterance_transformer384_dropout0p1_500ep_lr1e3.pt}"
SSL_CACHE_ROOT="${SSL_CACHE_ROOT:-/store/store5/data/acp21rjf_checkpoints/l2augment/ssl_feature_cache/rob132_hubert_base_tedlium_per_utterance}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/exp/exp4/acp21rjf/rob132-audio-ssl-scratch}"
QUEUED_COMMAND="${QUEUED_COMMAND:-screen -L -Logfile ${SCREEN_LOG_PATH} -dmS ${SCREEN_NAME} bash -lc 'cd ${REPO_DIR} && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob132_audio_ssl_mask_lm_training.sh'}"
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"

on_exit() {
  status=$?
  set +e
  if [ "${ROB132_DISABLE_CALLBACK:-0}" = "1" ]; then
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
    --note "${CALLBACK_NOTE:-ROB-132 audio SSL-conditioned transformer mask LM wrapper exited. Inspect the SSL cache summary, checkpoint path ${CHECKPOINT_PATH}, and logs before running post-training evals.}"
    --tail-lines "${CALLBACK_TAIL_LINES:-80}"
    --max-log-chars "${CALLBACK_MAX_LOG_CHARS:-6000}"
    --max-comment-chars "${CALLBACK_MAX_COMMENT_CHARS:-10000}"
  )
  if [ "${ROB132_CALLBACK_CHECK_ONLY:-0}" = "1" ]; then
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
export CONFIG_PATH CHECKPOINT_PATH SSL_CACHE_ROOT QUEUED_COMMAND GIT_BRANCH GIT_COMMIT
export TMPDIR="${SCRATCH_ROOT}/tmp"
export TORCH_HOME="${SCRATCH_ROOT}/torch"
export HF_HOME="${SCRATCH_ROOT}/huggingface"
export WANDB_DIR="${RESULT_ROOT}/wandb"
export WANDB_CACHE_DIR="${SCRATCH_ROOT}/wandb-cache"
export WANDB_CONFIG_DIR="${SCRATCH_ROOT}/wandb-config"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/xdg-cache"
export MPLCONFIGDIR="${SCRATCH_ROOT}/matplotlib"
mkdir -p "${TMPDIR}" "${TORCH_HOME}" "${HF_HOME}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"

echo "[rob132] branch=${GIT_BRANCH}"
echo "[rob132] commit=${GIT_COMMIT}"
echo "[rob132] host=$(hostname)"
echo "[rob132] result_root=${RESULT_ROOT}"
echo "[rob132] config=${CONFIG_PATH}"
echo "[rob132] ssl_cache_root=${SSL_CACHE_ROOT}"
echo "[rob132] checkpoint=${CHECKPOINT_PATH}"
echo "[rob132] log_path=${LOG_PATH}"
echo "[rob132] screen_log_path=${SCREEN_LOG_PATH}"
echo "[rob132] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[rob132] tmpdir=${TMPDIR}"

bash -ic 'python - <<'"'"'PY'"'"'
import sys
import torch
import torchaudio

print("[rob132] python_executable=" + sys.executable)
print("[rob132] python_version=" + sys.version.split()[0])
print("[rob132] torch_version=" + torch.__version__)
print("[rob132] torchaudio_version=" + torchaudio.__version__)
print("[rob132] cuda_available=" + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("[rob132] cuda_device=" + torch.cuda.get_device_name(0))
PY'

if [ "${ROB132_CALLBACK_ONLY:-0}" = "1" ]; then
  echo "[rob132] callback-only smoke path requested; exiting before cache/training."
  exit 0
fi

if [ "${ROB132_SMOKE:-0}" = "1" ]; then
  echo "[rob132] running smoke cache/training path"
  export CONFIG_PATH="${SMOKE_CONFIG_PATH}"
  export SSL_CACHE_ROOT="${RESULT_ROOT}/smoke/ssl_feature_cache"
  CACHE_MAX_FILES="${ROB132_CACHE_MAX_FILES:-2}"
else
  CACHE_MAX_FILES="${ROB132_CACHE_MAX_FILES:-}"
fi

cache_args=(
  --rollout-root /store/store4/data/l2augment_rollout_uvqmlm
  --cache-root "${SSL_CACHE_ROOT}"
  --tedlium-base /store/store4/data/TEDLIUM_release-3/legacy
  --split train
  --split dev
  --ssl-bundle HUBERT_BASE
  --summary "${RESULT_ROOT}/ssl_cache_summary.json"
)
if [ -n "${CACHE_MAX_FILES:-}" ]; then
  cache_args+=(--max-files "${CACHE_MAX_FILES}")
fi

bash -ic 'cd "$REPO_DIR"; export PYTHONPATH="$REPO_DIR:$REPO_DIR/exp:/exp/exp4/acp21rjf/long-context-asr:/exp/exp4/acp21rjf/language_modelling${PYTHONPATH:+:$PYTHONPATH}"; python scripts/build_rob132_ssl_feature_cache.py "$@"' bash "${cache_args[@]}"

unset WANDB_MODE
bash -ic 'cd "$REPO_DIR"; export PYTHONPATH="$REPO_DIR:$REPO_DIR/exp:/exp/exp4/acp21rjf/long-context-asr:/exp/exp4/acp21rjf/language_modelling${PYTHONPATH:+:$PYTHONPATH}"; python exp/train_freq_mask.py --config "$CONFIG_PATH"'

echo "[rob132] training command finished"
if [ -f "${CHECKPOINT_PATH}" ] || [ "${ROB132_SMOKE:-0}" = "1" ]; then
  ls -lh "${CHECKPOINT_PATH}" 2>/dev/null || true
else
  echo "[rob132] expected checkpoint missing: ${CHECKPOINT_PATH}" >&2
  exit 2
fi
