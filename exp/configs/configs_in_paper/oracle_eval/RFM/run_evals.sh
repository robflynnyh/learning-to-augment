#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

python "$EXP_DIR/run_config_grid.py" \
  --grid-config "$SCRIPT_DIR/tedlium_grid.yaml" \
  --mode slurm \
  --sbatch-script "$EXP_DIR/launch_scripts/run_eval_oracle_cpu.sh" \
  --slurm-chdir "$EXP_DIR/launch_scripts" \
  --workdir "$EXP_DIR" \
  --entrypoint "python oracle_eval.py --config {config}"
