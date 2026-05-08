#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../../.."
export CUDA_VISIBLE_DEVICES=""
export MPLCONFIGDIR="${MPLCONFIGDIR:-/exp/exp4/acp21rjf/.scratch/matplotlib-cache}"
export L2A_TEDLIUM3_LEGACY_DIR="${L2A_TEDLIUM3_LEGACY_DIR:-/store/store4/data/TEDLIUM_release-3/legacy}"
mkdir -p "${MPLCONFIGDIR}" "exp/results/repro/oracle/RFM/logs"

result="exp/results/repro/oracle/RFM/tedlium_lr1e-6_searchlr4e-2.txt"
: > "${result}"

for repeats in 1 2 3 4 5 10 20 50; do
  PYTHONDONTWRITEBYTECODE=1 python3 exp/oracle_eval.py \
    --config "exp/results/repro/oracle/RFM/configs/tedlium_lr1e-6_searchlr4e-2_repeats${repeats}.yaml"
done
