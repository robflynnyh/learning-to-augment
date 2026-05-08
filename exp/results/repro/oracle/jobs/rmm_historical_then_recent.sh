#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../../.."
export CUDA_VISIBLE_DEVICES=""
export MPLCONFIGDIR="${MPLCONFIGDIR:-/exp/exp4/acp21rjf/.scratch/matplotlib-cache}"
export L2A_TEDLIUM3_LEGACY_DIR="${L2A_TEDLIUM3_LEGACY_DIR:-/store/store4/data/TEDLIUM_release-3/legacy}"
mkdir -p "${MPLCONFIGDIR}" "exp/results/repro/oracle/RMM/logs"

run_combo() {
  local lr="$1"
  local search_lr="$2"
  local result="exp/results/repro/oracle/RMM/tedlium_lr${lr}_searchlr${search_lr}.txt"
  : > "${result}"
  for repeats in 1 2 3 4 5 10 20 50; do
    PYTHONDONTWRITEBYTECODE=1 python3 exp/oracle_eval.py \
      --config "exp/results/repro/oracle/RMM/configs/tedlium_lr${lr}_searchlr${search_lr}_repeats${repeats}.yaml"
  done
}

# Historical matched setup, followed by the newer/default setup.
run_combo 1e-6 4e-2
run_combo 8e-6 9e-2
