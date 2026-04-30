#!/bin/bash
# Run UFMR eval on mimas (no slurm).
#
# Usage:
#   ./run_eval_mimas.sh <path/to/config.yaml> [test_wer|test_cer]
#
# - Rewrites the YAML to point at the local /store/store5 checkpoints and
#   a results path under exp/results/UFMR_mimas/.
# - Sets L2A_*_DIR env vars so l2augment.utils.data resolves datasets
#   under /store/store4/data on this box.
# - Activates the flash_attn_pytorch2 conda env and runs exp/eval.py.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config.yaml> [test_wer|test_cer]" >&2
    exit 2
fi

CONFIG="$1"
UFMR_VARIANT="${2:-test_wer}"

if [[ ! -f "$CONFIG" ]]; then
    echo "Config not found: $CONFIG" >&2
    exit 1
fi

CKPT_ROOT=/store/store5/data/acp21rjf_checkpoints/l2augment
ASR_CKPT="$CKPT_ROOT/asr/step_105360.pt"
UFMR_CKPT="$CKPT_ROOT/ufmr/$UFMR_VARIANT/model.pt"
# Fall back to tmp_model.pt if best-checkpoint was never written (e.g. variant=test).
if [[ ! -f "$UFMR_CKPT" ]]; then
    if [[ -f "$CKPT_ROOT/ufmr/$UFMR_VARIANT/tmp_model.pt" ]]; then
        echo "[run_eval_mimas] no model.pt for '$UFMR_VARIANT' — using tmp_model.pt" >&2
        UFMR_CKPT="$CKPT_ROOT/ufmr/$UFMR_VARIANT/tmp_model.pt"
    else
        echo "UFMR variant '$UFMR_VARIANT' not found under $CKPT_ROOT/ufmr/" >&2
        echo "Available: $(ls $CKPT_ROOT/ufmr/ 2>/dev/null | tr '\n' ' ')" >&2
        exit 1
    fi
fi

# Dataset roots on mimas (datasets live in /store/store4/data).
# TAL is intentionally unset — it isn't mirrored on this box; leaving it
# unset means the parscratch default kicks in and a TAL run will fail loudly.
export L2A_EARNINGS22_DIR=/store/store4/data/earnings-22
export L2A_TEDLIUM3_LEGACY_DIR=/store/store4/data/TEDLIUM_release-3/legacy/
export L2A_REV16_DIR=/store/store4/data/rev_benchmark
export L2A_CHIME6_DIR=/store/store4/data/chime6/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$EXP_DIR/.." && pwd)"

CONFIG_ABS="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
TAG="$(basename "$CONFIG_ABS" .yaml)"
PARENT="$(basename "$(dirname "$CONFIG_ABS")")"
OUT_DIR="$EXP_DIR/results/UFMR_mimas/$UFMR_VARIANT/$PARENT"
mkdir -p "$OUT_DIR"
PATCHED="$OUT_DIR/${TAG}.yaml"
SAVE_PATH="$OUT_DIR/${TAG}.txt"

if [[ "${FORCE_RERUN:-0}" != "1" && -f "$SAVE_PATH" ]] && grep -q 'Updated_WER:' "$SAVE_PATH"; then
    line_count="$(wc -l < "$SAVE_PATH" | tr -d '[:space:]')"
    if [[ "$line_count" != "1" ]]; then
        echo "[run_eval_mimas] warning: completed result has $line_count lines: $SAVE_PATH" >&2
    fi
    echo "[run_eval_mimas] skipping completed eval: $SAVE_PATH"
    echo "[run_eval_mimas] set FORCE_RERUN=1 to run it again"
    exit 0
fi

source /store/store4/software/bin/anaconda3/etc/profile.d/conda.sh
conda activate /store/store4/software/bin/anaconda3/envs/flash_attn_pytorch2

python - "$CONFIG_ABS" "$PATCHED" "$ASR_CKPT" "$UFMR_CKPT" "$SAVE_PATH" <<'PY'
import sys
from omegaconf import OmegaConf
src, dst, asr_ckpt, ufmr_ckpt, save_path = sys.argv[1:]
cfg = OmegaConf.load(src)
cfg.checkpointing.asr_model = asr_ckpt
cfg.training.model_save_path = ufmr_ckpt
if "tmp_model_save_path" in cfg.training:
    cfg.training.tmp_model_save_path = ufmr_ckpt
cfg.evaluation.save_path = save_path
OmegaConf.save(cfg, dst)
print(f"[run_eval_mimas] patched config -> {dst}")
print(f"[run_eval_mimas] save_path     -> {save_path}")
PY

cd "$EXP_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
echo "[run_eval_mimas] running: PYTHONPATH=$PYTHONPATH python eval.py --config $PATCHED"
exec python eval.py --config "$PATCHED"
