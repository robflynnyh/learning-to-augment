#!/usr/bin/env bash
# Callback-backed ROB-124 384-dim/dropout RMM proposal + reward-1 LM CE rerank evaluation.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-124}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/earnings_rmm_lm_rerank}"
LOG_PATH="${LOG_PATH:-${RESULT_ROOT}/logs/rob124_384_dropout_rmm_lm_rerank.log}"
SCREEN_LOG_PATH="${SCREEN_LOG_PATH:-${RESULT_ROOT}/logs/rob124_384_dropout_rmm_lm_rerank.screen.log}"
SCREEN_NAME="${SCREEN_NAME:-rob124-384-dropout-rmm-lm-rerank}"
RUNNER_LABEL="${RUNNER_LABEL:-screen:${SCREEN_NAME}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt}"
ASR_CKPT="${ASR_CKPT:-/store/store5/data/acp21rjf_checkpoints/spotify/rotary_pos_6l_256d_seq_sched/n_seq_sched_2048_rp_1/step_105360.pt}"
MASK_VAE_CKPT="${MASK_VAE_CKPT:-/store/store5/data/acp21rjf_checkpoints/l2augment/models/bvae/bvae_USINGTHISFORNOW_2048gpu.pt}"
PREVIOUS_ROB124_CSV="${PREVIOUS_ROB124_CSV:-${REPO_DIR}/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/earnings_reward_controls/rob124_384_dropout_earnings_reward_controls.csv}"
ROB120_CSV="${ROB120_CSV:-${REPO_DIR}/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/old_ablations/rob120_earnings_reward_controls/rob120_earnings_reward_controls.csv}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/exp/exp4/acp21rjf/rob124-rerank-scratch}"
QUEUED_COMMAND="${QUEUED_COMMAND:-screen -L -Logfile ${SCREEN_LOG_PATH} -dmS ${SCREEN_NAME} bash -lc 'cd ${REPO_DIR} && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob124_384_dropout_rmm_lm_rerank.sh'}"
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"

on_exit() {
  status=$?
  set +e
  if [ "${ROB124_RERANK_DISABLE_CALLBACK:-0}" = "1" ]; then
    exit "${status}"
  fi
  if [ -z "${LINEAR_API_KEY:-}" ]; then
    echo "LINEAR_API_KEY is not set; cannot post Linear completion callback" >&2
    exit "${status}"
  fi
  callback_args=(
    --issue "${LINEAR_ISSUE}"
    --status-code "${status}"
    --log "${LOG_PATH}"
    --results "${RESULT_ROOT}"
    --screen-name "${SCREEN_NAME}"
    --runner-label "${RUNNER_LABEL}"
    --queued-command "${QUEUED_COMMAND}"
    --branch "${GIT_BRANCH}"
    --commit "${GIT_COMMIT}"
    --target-state "${CALLBACK_TARGET_STATE:-Todo}"
    --note "${CALLBACK_NOTE:-ROB-124 384/dropout RMM proposal plus reward-1 LM CE rerank evaluation wrapper exited. Inspect OUTCOME.md under ${RESULT_ROOT}.}"
    --tail-lines "${CALLBACK_TAIL_LINES:-80}"
    --max-log-chars "${CALLBACK_MAX_LOG_CHARS:-6000}"
    --max-comment-chars "${CALLBACK_MAX_COMMENT_CHARS:-10000}"
  )
  if [ "${ROB124_RERANK_CALLBACK_CHECK_ONLY:-0}" = "1" ]; then
    callback_args+=(--check-only)
  fi
  python3 "${REPO_DIR}/scripts/callbacks/linear_experiment_callback.py" "${callback_args[@]}"
  callback_status=$?
  if [ "${callback_status}" -ne 0 ]; then
    echo "Linear completion callback failed with status ${callback_status}" >&2
  fi
  exit "${status}"
}
trap on_exit EXIT

set -euo pipefail

mkdir -p "$(dirname "${LOG_PATH}")" "${RESULT_ROOT}" "${SCRATCH_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

export REPO_DIR RESULT_ROOT LOG_PATH SCREEN_LOG_PATH SCREEN_NAME RUNNER_LABEL
export CHECKPOINT_PATH ASR_CKPT MASK_VAE_CKPT PREVIOUS_ROB124_CSV ROB120_CSV QUEUED_COMMAND GIT_BRANCH GIT_COMMIT
export TMPDIR="${SCRATCH_ROOT}/tmp"
export WANDB_DIR="${RESULT_ROOT}/wandb"
export WANDB_CACHE_DIR="${SCRATCH_ROOT}/wandb-cache"
export WANDB_CONFIG_DIR="${SCRATCH_ROOT}/wandb-config"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/xdg-cache"
export MPLCONFIGDIR="${SCRATCH_ROOT}/matplotlib"
mkdir -p "${TMPDIR}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"

echo "[rob124-rerank] branch=${GIT_BRANCH}"
echo "[rob124-rerank] commit=${GIT_COMMIT}"
echo "[rob124-rerank] host=$(hostname)"
echo "[rob124-rerank] result_root=${RESULT_ROOT}"
echo "[rob124-rerank] checkpoint=${CHECKPOINT_PATH}"
echo "[rob124-rerank] asr_ckpt=${ASR_CKPT}"
echo "[rob124-rerank] mask_vae_ckpt=${MASK_VAE_CKPT}"
echo "[rob124-rerank] previous_rob124_csv=${PREVIOUS_ROB124_CSV}"
echo "[rob124-rerank] rob120_csv=${ROB120_CSV}"
echo "[rob124-rerank] log_path=${LOG_PATH}"
echo "[rob124-rerank] screen_log_path=${SCREEN_LOG_PATH}"
echo "[rob124-rerank] queued_command=${QUEUED_COMMAND}"
echo "[rob124-rerank] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[rob124-rerank] tmpdir=${TMPDIR}"

bash -ic 'python - <<'"'"'PY'"'"'
import sys
import torch

print("[rob124-rerank] python_executable=" + sys.executable)
print("[rob124-rerank] python_version=" + sys.version.split()[0])
print("[rob124-rerank] torch_version=" + torch.__version__)
print("[rob124-rerank] cuda_available=" + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("[rob124-rerank] cuda_device=" + torch.cuda.get_device_name(0))
PY'

if [ "${ROB124_RERANK_CALLBACK_ONLY:-0}" = "1" ]; then
  echo "[rob124-rerank] callback-only smoke path requested; exiting before config generation."
  exit 0
fi

for required_path in "${CHECKPOINT_PATH}" "${ASR_CKPT}" "${MASK_VAE_CKPT}" "${PREVIOUS_ROB124_CSV}" "${ROB120_CSV}"; do
  if [ ! -f "${required_path}" ]; then
    echo "Missing required path: ${required_path}" >&2
    exit 1
  fi
done

export L2A_EARNINGS22_DIR="${L2A_EARNINGS22_DIR:-/store/store4/data/earnings-22}"

python3 - "${RESULT_ROOT}" "${ASR_CKPT}" "${CHECKPOINT_PATH}" "${MASK_VAE_CKPT}" "${LINEAR_ISSUE}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
asr_ckpt = sys.argv[2]
policy_ckpt = sys.argv[3]
mask_vae_ckpt = sys.argv[4]
linear_issue = sys.argv[5]

method = "RMMReward1LMRerank15"
(root / method / "configs").mkdir(parents=True, exist_ok=True)
save_path = root / method / "earnings22_test_epoch1_lr1e-5.txt"
config_path = root / method / "configs" / "earnings22_test_epoch1_lr1e-5.yaml"
config_path.write_text(
    f"""checkpointing:
  asr_model: {asr_ckpt}

training:
  device: cuda
  random_seed: 123456
  batch_size: 1
  epochs: 500
  model_save_path: null
  tmp_model_save_path: null
  prefetch_factor: null
  num_workers: 0

evaluation:
  id: {linear_issue}-earnings22-test-384d-dropout0p1-rmm15-reward1-lm-rerank-epoch1-lr1e-5
  dataset: earnings22
  split: test
  rollout_fn: multistep
  use_cer: false
  epochs: 1
  augmentation_config: {{}}
  optim_args:
    lr: 1e-5
  save_path: {save_path}

policy:
  lr: 1e-3
  class: RMMRewardConditionedMaskLMReranker
  config:
    candidate_repeats: 15
    scorer_conditioning_reward: 1.0
    rmm_config:
      time_masks_min: 3
      time_masks_max: 16
      freq_masks_min: 5
      freq_masks_max: 7
      freq_mask_param_min: 34
      freq_mask_param_max: 34
    reward_lm_state_dict_path: {policy_ckpt}
    reward_lm_config:
      hidden_dim: 384
      dropout: 0.1
      default_conditioning_reward: 1.0
      reward_encoder: timestep
      sample_generation: true
      mask_vae_state_dict_path: {mask_vae_ckpt}
      mask_vae_config:
        latent_dim: 128
        codebook_size: 2048
        use_vq: true
""",
    encoding="utf-8",
)
print(f"[rob124-rerank] wrote config {config_path}")
PY

if [ "${ROB124_RERANK_CONFIG_ONLY:-0}" = "1" ]; then
  echo "[rob124-rerank] config-only path requested; exiting before eval."
  exit 0
fi

METHOD="RMMReward1LMRerank15"
CONFIG="${RESULT_ROOT}/${METHOD}/configs/earnings22_test_epoch1_lr1e-5.yaml"
SAVE_PATH="${RESULT_ROOT}/${METHOD}/earnings22_test_epoch1_lr1e-5.txt"

if [ "${ROB124_RERANK_SMOKE:-0}" = "1" ]; then
  ROB124_RERANK_INDEXES="${ROB124_RERANK_INDEXES:-0}"
  ROB124_RERANK_DONT_SAVE="${ROB124_RERANK_DONT_SAVE:-1}"
  export ROB124_RERANK_INDEXES ROB124_RERANK_DONT_SAVE
  echo "[rob124-rerank] smoke mode: indexes=${ROB124_RERANK_INDEXES}; dont_save=${ROB124_RERANK_DONT_SAVE}"
fi

if [ "${FORCE_RERUN:-0}" = "1" ]; then
  rm -f "${SAVE_PATH}"
elif [ -f "${SAVE_PATH}" ] && grep -q "Updated_WER:" "${SAVE_PATH}"; then
  echo "[rob124-rerank] skipping completed ${METHOD}: ${SAVE_PATH}"
else
  echo "[rob124-rerank] running ${METHOD}: ${CONFIG}"
  CONFIG_TO_RUN="${CONFIG}" bash -ic '
    cd "$REPO_DIR/exp"
    export PYTHONPATH="$REPO_DIR:$REPO_DIR/exp:/exp/exp4/acp21rjf/long-context-asr:/exp/exp4/acp21rjf/language_modelling${PYTHONPATH:+:$PYTHONPATH}"
    args=(python eval.py --config "$CONFIG_TO_RUN")
    if [ -n "${ROB124_RERANK_INDEXES:-}" ]; then
      read -r -a indexes <<< "$ROB124_RERANK_INDEXES"
      args+=(--indexes "${indexes[@]}")
    fi
    if [ "${ROB124_RERANK_DONT_SAVE:-0}" = "1" ]; then
      args+=(--dont_save)
    fi
    echo "[rob124-rerank] eval command: ${args[*]}"
    "${args[@]}"
  '
fi

if [ "${ROB124_RERANK_SMOKE:-0}" = "1" ]; then
  echo "[rob124-rerank] smoke completed; exiting before summary generation."
  exit 0
fi

python3 scripts/summarize_rob124_384_dropout_rmm_lm_rerank.py \
  --result-root "${RESULT_ROOT}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --previous-rob124-csv "${PREVIOUS_ROB124_CSV}" \
  --rob120-csv "${ROB120_CSV}" \
  --command "${QUEUED_COMMAND}" \
  --branch "${GIT_BRANCH}" \
  --commit "${GIT_COMMIT}" \
  --log-path "${LOG_PATH}" \
  --screen-log-path "${SCREEN_LOG_PATH}"

echo "[rob124-rerank] finished"
