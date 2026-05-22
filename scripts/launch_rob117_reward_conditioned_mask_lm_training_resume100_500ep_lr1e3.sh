#!/usr/bin/env bash
# ROB-117 follow-up: resume the 100-epoch checkpoint to epoch 500 at LR 1e-3.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/exp/results/repro/reward_conditioned_lm/no_audio_conditioning}"
LOG_PATH="${LOG_PATH:-${RESULT_ROOT}/logs/rob117_no_audio_reward_conditioned_mask_lm_training_resume100_500ep_lr1e3_20260522.log}"
SCREEN_LOG_PATH="${SCREEN_LOG_PATH:-${RESULT_ROOT}/logs/rob117_no_audio_reward_conditioned_mask_lm_training_resume100_500ep_lr1e3_20260522.screen.log}"
SCREEN_NAME="${SCREEN_NAME:-rob117-reward-conditioned-mask-lm-resume100-500ep-lr1e3}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_resume100_500ep_lr1e3.pt}"

export RESULT_ROOT
export LOG_PATH
export SCREEN_LOG_PATH
export SCREEN_NAME
export CONFIG_PATH="${CONFIG_PATH:-exp/configs/reward_conditioned_lm/no_audio_conditioning/tedlium_per_utterance_resume100_500ep_lr1e3.yaml}"
export CHECKPOINT_PATH
export TMP_CHECKPOINT_PATH="${TMP_CHECKPOINT_PATH:-/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_resume100_500ep_lr1e3_tmp.pt}"
export CALLBACK_NOTE="${CALLBACK_NOTE:-ROB-117 resumed 100-epoch checkpoint to 500 epochs at LR 1e-3. Check checkpoint path ${CHECKPOINT_PATH} and update OUTCOME.md before final handoff.}"
export QUEUED_COMMAND="${QUEUED_COMMAND:-screen -L -Logfile ${SCREEN_LOG_PATH} -dmS ${SCREEN_NAME} bash -lc 'cd ${REPO_DIR} && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob117_reward_conditioned_mask_lm_training_resume100_500ep_lr1e3.sh'}"

exec "${REPO_DIR}/scripts/launch_rob117_reward_conditioned_mask_lm_training.sh"
