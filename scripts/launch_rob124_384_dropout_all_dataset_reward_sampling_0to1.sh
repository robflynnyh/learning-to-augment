#!/usr/bin/env bash
# ROB-124 follow-up: all-dataset eval with reward sampled from [0.0, 1.0].

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_reward_sampling_0to1}"
LOG_PATH="${LOG_PATH:-${RESULT_ROOT}/logs/rob124_384_dropout_all_dataset_reward_sampling_0to1.log}"
SCREEN_LOG_PATH="${SCREEN_LOG_PATH:-${RESULT_ROOT}/logs/rob124_384_dropout_all_dataset_reward_sampling_0to1.screen.log}"
SCREEN_NAME="${SCREEN_NAME:-rob124-384-dropout-all-dataset-sampling-0to1}"

export RESULT_ROOT
export LOG_PATH
export SCREEN_LOG_PATH
export SCREEN_NAME
export ROB124_ALLDATA_REWARD_RANGE_LOW="${ROB124_ALLDATA_REWARD_RANGE_LOW:-0.0}"
export ROB124_ALLDATA_REWARD_RANGE_HIGH="${ROB124_ALLDATA_REWARD_RANGE_HIGH:-1.0}"
export ROB124_ALLDATA_REWARD_RANGE_ID="${ROB124_ALLDATA_REWARD_RANGE_ID:-0to1}"
export ROB124_ALLDATA_REWARD_RANGE_LABEL="${ROB124_ALLDATA_REWARD_RANGE_LABEL:-[0.0, 1.0]}"
export ROB124_ALLDATA_METHOD="${ROB124_ALLDATA_METHOD:-RewardConditionedMaskLMUniform0to1}"
export ROB124_ALLDATA_CONDITION="${ROB124_ALLDATA_CONDITION:-uniform_0.0_1.0}"
export ROB124_ALLDATA_LABEL="${ROB124_ALLDATA_LABEL:-uniform sampled reward [0.0, 1.0]}"
export ROB124_ALLDATA_CSV_NAME="${ROB124_ALLDATA_CSV_NAME:-rob124_384_dropout_all_dataset_reward_sampling_0to1.csv}"
export ROB124_ALLDATA_OUTCOME_TITLE="${ROB124_ALLDATA_OUTCOME_TITLE:-ROB-124 384-Dropout All-Dataset Reward Sampling [0.0, 1.0] Eval}"
export SCRATCH_ROOT="${SCRATCH_ROOT:-/exp/exp4/acp21rjf/rob124-all-dataset-sampling-0to1-scratch}"
export CALLBACK_NOTE="${CALLBACK_NOTE:-ROB-124 384/dropout all-dataset sampled reward [0.0, 1.0] eval wrapper exited. Inspect OUTCOME.md under ${RESULT_ROOT}.}"
export QUEUED_COMMAND="${QUEUED_COMMAND:-screen -L -Logfile ${SCREEN_LOG_PATH} -dmS ${SCREEN_NAME} bash -lc 'cd ${REPO_DIR} && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob124_384_dropout_all_dataset_reward_sampling_0to1.sh'}"

exec "${REPO_DIR}/scripts/launch_rob124_384_dropout_all_dataset_reward_sampling.sh"
