#!/usr/bin/env bash
# Queue ROB-80 no-audio CMultiStepVQLM with random reward conditioning.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

export ROB80_METHOD="${ROB80_METHOD:-CMultiStepVQLMRandomReward}"
export ROB80_SUMMARY_METHODS="${ROB80_SUMMARY_METHODS:-CMultiStepVQLM CMultiStepVQLMRandomReward}"
export ROB80_CONDITIONING_REWARD_RANGE="${ROB80_CONDITIONING_REWARD_RANGE:-0.5 1.0}"
export ROB80_CSV_NAME="${ROB80_CSV_NAME:-rob80_tedlium_noaudio_reward_conditioning_comparison.csv}"
export ROB80_OUTCOME_NAME="${ROB80_OUTCOME_NAME:-ROB-80_NOAUDIO_REWARD_CONDITIONING_COMPARISON.md}"
export ROB80_TITLE="${ROB80_TITLE:-ROB-80 TED-LIUM Dev No-Audio CMultiStepVQLM Reward Conditioning Comparison}"
export ROB80_NOTE="${ROB80_NOTE:-Compares the committed fixed conditioning reward 1.0 baseline against the follow-up randomized conditioning reward sampled uniformly from [0.5, 1.0].}"
export SCREEN_NAME="${SCREEN_NAME:-rob80_tedlium_noaudio_random_reward_sweep}"
export LOG_PATH="${LOG_PATH:-${REPO_DIR}/exp/results/repro/sweeps/no_audio_cmultistep_vqlm/logs/rob80_tedlium_noaudio_random_reward_sweep.log}"
export RUNNER_LABEL="${RUNNER_LABEL:-screen:${SCREEN_NAME}}"
export QUEUED_COMMAND="${QUEUED_COMMAND:-/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob80_tedlium_noaudio_random_reward_sweep.sh}"
export CALLBACK_NOTE="${CALLBACK_NOTE:-ROB-80 TED-LIUM dev no-audio CMultiStepVQLM random reward-conditioning sweep wrapper exited. See exp/results/repro/sweeps/no_audio_cmultistep_vqlm/ROB-80_NOAUDIO_REWARD_CONDITIONING_COMPARISON.md for the fixed-vs-random comparison table when complete.}"

exec scripts/launch_rob80_tedlium_noaudio_cmultistep_sweep.sh
