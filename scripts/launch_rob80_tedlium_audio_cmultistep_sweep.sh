#!/usr/bin/env bash
# Queue ROB-80 audio-conditioned CMultiStepVQLM fixed/random reward sweep.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-80}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/exp/results/repro/sweeps/audio_cmultistep_vqlm}"
LOG_PATH="${LOG_PATH:-${RESULT_ROOT}/logs/rob80_tedlium_audio_cmultistep_sweep.log}"
SCREEN_NAME="${SCREEN_NAME:-rob80_tedlium_audio_cmultistep_sweep}"
RUNNER_LABEL="${RUNNER_LABEL:-screen:${SCREEN_NAME}}"
QUEUED_COMMAND="${QUEUED_COMMAND:-/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob80_tedlium_audio_cmultistep_sweep.sh}"
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"

on_exit() {
  status=$?
  set +e
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
    --note "${CALLBACK_NOTE:-ROB-80 TED-LIUM dev audio-conditioned CMultiStepVQLM fixed/random reward sweep exited. See exp/results/repro/sweeps/audio_cmultistep_vqlm/ROB-80_AUDIO_REWARD_CONDITIONING_COMPARISON.md for the table when complete.}"
    --tail-lines "${CALLBACK_TAIL_LINES:-80}"
    --max-log-chars "${CALLBACK_MAX_LOG_CHARS:-6000}"
    --max-comment-chars "${CALLBACK_MAX_COMMENT_CHARS:-10000}"
  )
  if [ "${ROB80_CALLBACK_CHECK_ONLY:-0}" = "1" ]; then
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

mkdir -p "$(dirname "${LOG_PATH}")" "${RESULT_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "[rob80-audio-cmultistep] branch=${GIT_BRANCH}"
echo "[rob80-audio-cmultistep] commit=${GIT_COMMIT}"
echo "[rob80-audio-cmultistep] result_root=${RESULT_ROOT}"
echo "[rob80-audio-cmultistep] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

if [ "${ROB80_CALLBACK_ONLY:-0}" = "1" ]; then
  echo "[rob80-audio-cmultistep] callback-only smoke path requested; exiting before eval."
  exit 0
fi

COMMON_ENV=(
  ROB80_DISABLE_CALLBACK=1
  RESULT_ROOT="${RESULT_ROOT}"
  ROB80_CMULTISTEP_CKPT="${ROB80_AUDIO_CMULTISTEP_CKPT:-/store/store5/data/acp21rjf_checkpoints/l2augment/models/CMultiStepMLM/curbest.pt}"
  ROB80_AUDIO_VAE_CKPT="${ROB80_AUDIO_VAE_CKPT:-/store/store5/data/acp21rjf_checkpoints/l2augment/models/autoenc_audio/model_gpu.pt}"
  ROB80_MASK_VAE_CKPT="${ROB80_MASK_VAE_CKPT:-/store/store5/data/acp21rjf_checkpoints/l2augment/models/bvae/bvae_USINGTHISFORNOW_2048gpu.pt}"
  ROB80_CONDITION_ON_AUDIO=true
  ROB80_USE_SIGNAL_INPUTS=false
  ROB80_EMBEDDING_DIM=256
  ROB80_REPEATS="${ROB80_REPEATS:-1}"
  ROB80_RUN_REPEATS="${ROB80_RUN_REPEATS:-${ROB80_REPEATS:-1}}"
  ROB80_SUMMARY_METHODS="CMultiStepVQLMAudio CMultiStepVQLMAudioRandomReward"
  ROB80_CSV_NAME="rob80_tedlium_audio_reward_conditioning_comparison.csv"
  ROB80_OUTCOME_NAME="ROB-80_AUDIO_REWARD_CONDITIONING_COMPARISON.md"
  ROB80_TITLE="ROB-80 TED-LIUM Dev Audio-Conditioned CMultiStepVQLM Reward Conditioning Comparison"
  ROB80_NOTE="Audio-conditioned CMultiStepVQLM uses the legacy score-conditioned audio checkpoint with condition_on_audio true and use_signal_inputs false. This compares fixed reward 1.0 against uniform [0.5, 1.0] random reward conditioning."
)

env "${COMMON_ENV[@]}" \
  ROB80_METHOD="CMultiStepVQLMAudio" \
  ROB80_CONDITIONING_REWARD_RANGE="" \
  scripts/launch_rob80_tedlium_noaudio_cmultistep_sweep.sh

env "${COMMON_ENV[@]}" \
  ROB80_METHOD="CMultiStepVQLMAudioRandomReward" \
  ROB80_CONDITIONING_REWARD_RANGE="0.5 1.0" \
  scripts/launch_rob80_tedlium_noaudio_cmultistep_sweep.sh

echo "[rob80-audio-cmultistep] finished"
