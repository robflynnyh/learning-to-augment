#!/usr/bin/env bash
# Queue-safe ROB-80 TED-LIUM segmented dev LR sweep for RFM, RMM, and UFMR.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

export RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/exp/results/repro/sweeps/segmented_dev}"
export LOG_PATH="${LOG_PATH:-${RESULT_ROOT}/logs/rob80_tedlium_segmented_policy_sweep.log}"
export SCREEN_NAME="${SCREEN_NAME:-rob80_tedlium_segmented_policy_sweep}"
export RUNNER_LABEL="${RUNNER_LABEL:-screen:${SCREEN_NAME}}"
export ROB80_DATASET="${ROB80_DATASET:-tedlium3_segmented_data}"
export ROB80_SPLIT="${ROB80_SPLIT:-dev}"
export ROB80_TAG_PREFIX="${ROB80_TAG_PREFIX:-tedlium_segmented_dev}"
export ROB80_BASE_LRS="${ROB80_BASE_LRS:-5e-6 1e-5 2e-5}"
export ROB80_UFMR_EXTRA_LRS="${ROB80_UFMR_EXTRA_LRS:-}"
export ROB80_INCLUDE_UFMR_EXTRA="${ROB80_INCLUDE_UFMR_EXTRA:-0}"
export ROB80_CSV_NAME="${ROB80_CSV_NAME:-rob80_tedlium_segmented_policy_sweep.csv}"
export ROB80_OUTCOME_NAME="${ROB80_OUTCOME_NAME:-ROB-80_SEGMENTED_OUTCOME.md}"
export ROB80_TITLE="${ROB80_TITLE:-ROB-80 TED-LIUM Segmented Dev Policy LR Sweep}"
export ROB80_NOTE="${ROB80_NOTE:-Segmented follow-up requested after the non-segmented dev sweep; LR grid is centered on the non-segmented sweep scale.}"
export CALLBACK_NOTE="${CALLBACK_NOTE:-ROB-80 TED-LIUM segmented dev RFM/RMM/UFMR LR sweep wrapper exited. See exp/results/repro/sweeps/segmented_dev/ROB-80_SEGMENTED_OUTCOME.md for the result table when complete.}"
export QUEUED_COMMAND="${QUEUED_COMMAND:-/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob80_tedlium_segmented_policy_sweep.sh}"

exec scripts/launch_rob80_tedlium_policy_sweep.sh
