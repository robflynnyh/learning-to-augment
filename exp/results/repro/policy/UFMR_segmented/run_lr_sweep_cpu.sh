#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../../.."
export CUDA_VISIBLE_DEVICES=""
export MPLCONFIGDIR="${MPLCONFIGDIR:-/exp/exp4/acp21rjf/.scratch/matplotlib-cache}"
export L2A_TEDLIUM3_LEGACY_DIR="${L2A_TEDLIUM3_LEGACY_DIR:-/store/store4/data/TEDLIUM_release-3/legacy}"
mkdir -p "${MPLCONFIGDIR}"

for lr in 1e-6 8e-6; do
  result="exp/results/repro/policy/UFMR_segmented/tedlium_lr${lr}.txt"
  : > "${result}"
  PYTHONDONTWRITEBYTECODE=1 python3 exp/oracle_eval.py \
    --config "exp/results/repro/policy/UFMR_segmented/configs/tedlium_lr${lr}.yaml"
done
