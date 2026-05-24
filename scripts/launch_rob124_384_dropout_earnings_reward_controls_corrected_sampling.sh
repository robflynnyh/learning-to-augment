#!/usr/bin/env bash
# Corrected ROB-124 Earnings reward-control rerun after fixing reward-range sampling.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_ROOT="${REPO_DIR}/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_earnings_reward_controls_corrected_sampling"
SCREEN_NAME="rob124-384-dropout-earnings-corrected-sampling"
SCREEN_LOG_PATH="${RESULT_ROOT}/logs/rob124_384_dropout_earnings_reward_controls_corrected_sampling.screen.log"

export RESULT_ROOT
export LOG_PATH="${RESULT_ROOT}/logs/rob124_384_dropout_earnings_reward_controls_corrected_sampling.log"
export SCREEN_LOG_PATH
export SCREEN_NAME
export RUNNER_LABEL="screen:${SCREEN_NAME}"
export SCRATCH_ROOT="/exp/exp4/acp21rjf/rob124-eval-corrected-sampling-scratch"
export FORCE_RERUN="${FORCE_RERUN:-1}"
export QUEUED_COMMAND="screen -L -Logfile ${SCREEN_LOG_PATH} -dmS ${SCREEN_NAME} bash -lc 'cd ${REPO_DIR} && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob124_384_dropout_earnings_reward_controls_corrected_sampling.sh'"
export CALLBACK_NOTE="ROB-124 corrected 384/dropout Earnings reward-control rerun exited. This rerun uses the fixed RewardConditionedMaskLM.augment reward-range sampling path; inspect OUTCOME.md under ${RESULT_ROOT}."

exec "${REPO_DIR}/scripts/launch_rob124_384_dropout_earnings_reward_controls.sh"
