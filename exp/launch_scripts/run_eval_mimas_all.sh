#!/usr/bin/env bash
# Sequential UFMR eval launcher for mimas.
# Usage:
#   ./run_eval_mimas_all.sh <gpu_id> [variant ...]
#
# Defaults to all checkpoint variants under /store/store5/.../ufmr.
# Singlestep configs are intentionally excluded.
# Set INCLUDE_TAL=1 if you also want the TAL configs (requires L2A_TAL_DIR).

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <gpu_id> [variant ...]" >&2
    exit 2
fi

GPU="$1"
shift || true
export CUDA_VISIBLE_DEVICES="$GPU"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_WRAPPER="$SCRIPT_DIR/run_eval_mimas.sh"
CONFIG_ROOT="$SCRIPT_DIR/../configs/configs_in_paper/UFRM/UFRM_eval"
CKPT_ROOT=/store/store5/data/acp21rjf_checkpoints/l2augment
VARIANT_ROOT="$CKPT_ROOT/ufmr"

if [[ ! -x "$EVAL_WRAPPER" ]]; then
    echo "Missing eval wrapper: $EVAL_WRAPPER" >&2
    exit 1
fi

if [[ ! -d "$CONFIG_ROOT" ]]; then
    echo "Missing config root: $CONFIG_ROOT" >&2
    exit 1
fi

if [[ $# -gt 0 ]]; then
    VARIANTS=("$@")
else
    mapfile -t VARIANTS < <(find "$VARIANT_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
fi

mapfile -t CONFIGS < <(find "$CONFIG_ROOT" -type f -name '*.yaml' ! -name 'run_evals.sh' ! -path '*/singlestep/*' | sort)

if [[ ${#VARIANTS[@]} -eq 0 ]]; then
    echo "No variants found under $VARIANT_ROOT" >&2
    exit 1
fi

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    echo "No configs found under $CONFIG_ROOT" >&2
    exit 1
fi

echo "[run_eval_mimas_all] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[run_eval_mimas_all] variants: ${VARIANTS[*]}"
echo "[run_eval_mimas_all] configs: ${#CONFIGS[@]}"

for variant in "${VARIANTS[@]}"; do
    if [[ ! -d "$VARIANT_ROOT/$variant" ]]; then
        echo "[run_eval_mimas_all] skipping missing variant dir: $variant" >&2
        continue
    fi

    for cfg in "${CONFIGS[@]}"; do
        dataset="$(basename "$cfg" .yaml)"
        if [[ "$dataset" == "TAL" && "${INCLUDE_TAL:-0}" != "1" ]]; then
            echo "[run_eval_mimas_all] skipping TAL (set INCLUDE_TAL=1 to include it)"
            continue
        fi

        echo "[run_eval_mimas_all] running variant=$variant config=$cfg"
        "$EVAL_WRAPPER" "$cfg" "$variant"
    done
done
