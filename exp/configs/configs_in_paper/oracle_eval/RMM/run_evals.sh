#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
LAUNCH_SCRIPT="$EXP_DIR/launch_scripts/run_eval_oracle_cpu.sh"

echo "launching legacy one-off evaluation for $SCRIPT_DIR/tedlium_1.yaml"
sbatch --chdir="$EXP_DIR/launch_scripts" --export=CONFIG="$SCRIPT_DIR/tedlium_1.yaml" "$LAUNCH_SCRIPT"

python "$EXP_DIR/run_config_grid.py" \
  --grid-config "$SCRIPT_DIR/tedlium_grid.yaml" \
  --mode slurm \
  --sbatch-script "$LAUNCH_SCRIPT" \
  --slurm-chdir "$EXP_DIR/launch_scripts" \
  --workdir "$EXP_DIR" \
  --entrypoint "python oracle_eval.py --config {config}"
