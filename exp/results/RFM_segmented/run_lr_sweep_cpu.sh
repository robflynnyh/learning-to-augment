#!/usr/bin/env bash
set -euo pipefail

cd /exp/exp4/acp21rjf/learning-to-augment

export CUDA_VISIBLE_DEVICES=""
export L2A_TEDLIUM3_LEGACY_DIR=/store/store4/data/TEDLIUM_release-3/legacy
export MPLCONFIGDIR=/exp/exp4/acp21rjf/.scratch/matplotlib-cache
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

mkdir -p "$MPLCONFIGDIR"

configs=(
  exp/results/RFM_segmented/tedlium_lr1e-5.yaml
  exp/results/RFM_segmented/tedlium_lr5e-5.yaml
  exp/results/RFM_segmented/tedlium_lr9e-5.yaml
)

for config in "${configs[@]}"; do
  echo "[$(date -Is)] running ${config}"
  python3 exp/oracle_eval.py --config "${config}"
  echo "[$(date -Is)] finished ${config}"
done
