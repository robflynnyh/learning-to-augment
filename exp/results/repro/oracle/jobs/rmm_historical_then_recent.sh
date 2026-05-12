#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../../.."
export CUDA_VISIBLE_DEVICES=""
export MPLCONFIGDIR="${MPLCONFIGDIR:-/exp/exp4/acp21rjf/.scratch/matplotlib-cache}"
export L2A_TEDLIUM3_LEGACY_DIR="${L2A_TEDLIUM3_LEGACY_DIR:-/store/store4/data/TEDLIUM_release-3/legacy}"
mkdir -p "${MPLCONFIGDIR}" "exp/results/repro/oracle/RMM/logs"

: > "exp/results/repro/oracle/RMM/tedlium_lr1e-6_searchlr4e-2.txt"
: > "exp/results/repro/oracle/RMM/tedlium_lr8e-6_searchlr9e-2.txt"

PYTHONDONTWRITEBYTECODE=1 python3 exp/run_config_grid.py \
  --grid-config "exp/results/repro/oracle/RMM/tedlium_grid.yaml" \
  --materialize-only

config_dir="exp/results/repro/oracle/RMM/.generated/tedlium_grid"

run_combo() {
  local lr="$1"
  local search_lr="$2"
  for repeats in 1 2 3 4 5 10 20 50; do
    PYTHONDONTWRITEBYTECODE=1 python3 exp/oracle_eval.py \
      --config "${config_dir}/tedlium_lr${lr}_searchlr${search_lr}_repeats${repeats}.yaml"
  done
}

run_combo 1e-6 4e-2
run_combo 8e-6 9e-2
