#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../../.."
export CUDA_VISIBLE_DEVICES=""
export MPLCONFIGDIR="${MPLCONFIGDIR:-/exp/exp4/acp21rjf/.scratch/matplotlib-cache}"
export L2A_TEDLIUM3_LEGACY_DIR="${L2A_TEDLIUM3_LEGACY_DIR:-/store/store4/data/TEDLIUM_release-3/legacy}"
mkdir -p "${MPLCONFIGDIR}" "exp/results/repro/oracle/RFM/logs"

: > "exp/results/repro/oracle/RFM/tedlium_lr1e-6_searchlr4e-2.txt"
: > "exp/results/repro/oracle/RFM/tedlium_lr8e-6_searchlr9e-2.txt"

PYTHONDONTWRITEBYTECODE=1 python3 exp/run_config_grid.py \
  --grid-config "exp/results/repro/oracle/RFM/tedlium_grid.yaml" \
  --entrypoint "PYTHONDONTWRITEBYTECODE=1 python3 exp/oracle_eval.py --config {config}" \
  --workdir . \
  --stop-on-error
