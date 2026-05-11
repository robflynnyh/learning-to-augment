#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
LAUNCH_SCRIPT="$EXP_DIR/launch_scripts/run_eval_oracle_cpu.sh"

run_grid() {
  local grid_config="$1"
  python "$EXP_DIR/run_config_grid.py" \
    --grid-config "$grid_config" \
    --mode slurm \
    --sbatch-script "$LAUNCH_SCRIPT" \
    --slurm-chdir "$EXP_DIR/launch_scripts" \
    --workdir "$EXP_DIR" \
    --entrypoint "python oracle_eval.py --config {config}"
}

run_grid "$SCRIPT_DIR/tedlium_grid.yaml"
run_grid "$SCRIPT_DIR/4e-2/tedlium_grid.yaml"

echo "launching legacy one-off evaluation for $SCRIPT_DIR/8e-2/tedlium_1.yaml"
sbatch --chdir="$EXP_DIR/launch_scripts" --export=CONFIG="$SCRIPT_DIR/8e-2/tedlium_1.yaml" "$LAUNCH_SCRIPT"
run_grid "$SCRIPT_DIR/8e-2/tedlium_grid.yaml"
