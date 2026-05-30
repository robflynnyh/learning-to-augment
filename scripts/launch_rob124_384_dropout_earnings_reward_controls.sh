#!/usr/bin/env bash
# Callback-backed ROB-124 384-dim dropout Earnings reward-control evaluation.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-124}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/earnings_reward_controls}"
LOG_PATH="${LOG_PATH:-${RESULT_ROOT}/logs/rob124_384_dropout_earnings_reward_controls.log}"
SCREEN_LOG_PATH="${SCREEN_LOG_PATH:-${RESULT_ROOT}/logs/rob124_384_dropout_earnings_reward_controls.screen.log}"
SCREEN_NAME="${SCREEN_NAME:-rob124-384-dropout-earnings-reward-controls}"
RUNNER_LABEL="${RUNNER_LABEL:-screen:${SCREEN_NAME}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt}"
ASR_CKPT="${ASR_CKPT:-/store/store5/data/acp21rjf_checkpoints/spotify/rotary_pos_6l_256d_seq_sched/n_seq_sched_2048_rp_1/step_105360.pt}"
MASK_VAE_CKPT="${MASK_VAE_CKPT:-/store/store5/data/acp21rjf_checkpoints/l2augment/models/bvae/bvae_USINGTHISFORNOW_2048gpu.pt}"
BASELINE_CSV="${BASELINE_CSV:-${REPO_DIR}/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/old_ablations/rob120_earnings_reward_controls/rob120_earnings_reward_controls.csv}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/exp/exp4/acp21rjf/rob124-eval-scratch}"
QUEUED_COMMAND="${QUEUED_COMMAND:-screen -L -Logfile ${SCREEN_LOG_PATH} -dmS ${SCREEN_NAME} bash -lc 'cd ${REPO_DIR} && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob124_384_dropout_earnings_reward_controls.sh'}"
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"

on_exit() {
  status=$?
  set +e
  if [ "${ROB124_EVAL_DISABLE_CALLBACK:-0}" = "1" ]; then
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
    --note "${CALLBACK_NOTE:-ROB-124 384/dropout Earnings reward-control evaluation wrapper exited. Inspect OUTCOME.md under ${RESULT_ROOT}.}"
    --tail-lines "${CALLBACK_TAIL_LINES:-80}"
    --max-log-chars "${CALLBACK_MAX_LOG_CHARS:-6000}"
    --max-comment-chars "${CALLBACK_MAX_COMMENT_CHARS:-10000}"
  )
  if [ "${ROB124_EVAL_CALLBACK_CHECK_ONLY:-0}" = "1" ]; then
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
export CHECKPOINT_PATH ASR_CKPT MASK_VAE_CKPT BASELINE_CSV QUEUED_COMMAND GIT_BRANCH GIT_COMMIT
export TMPDIR="${SCRATCH_ROOT}/tmp"
export WANDB_DIR="${RESULT_ROOT}/wandb"
export WANDB_CACHE_DIR="${SCRATCH_ROOT}/wandb-cache"
export WANDB_CONFIG_DIR="${SCRATCH_ROOT}/wandb-config"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/xdg-cache"
export MPLCONFIGDIR="${SCRATCH_ROOT}/matplotlib"
mkdir -p "${TMPDIR}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"

echo "[rob124-eval] branch=${GIT_BRANCH}"
echo "[rob124-eval] commit=${GIT_COMMIT}"
echo "[rob124-eval] host=$(hostname)"
echo "[rob124-eval] result_root=${RESULT_ROOT}"
echo "[rob124-eval] checkpoint=${CHECKPOINT_PATH}"
echo "[rob124-eval] asr_ckpt=${ASR_CKPT}"
echo "[rob124-eval] mask_vae_ckpt=${MASK_VAE_CKPT}"
echo "[rob124-eval] baseline_csv=${BASELINE_CSV}"
echo "[rob124-eval] log_path=${LOG_PATH}"
echo "[rob124-eval] screen_log_path=${SCREEN_LOG_PATH}"
echo "[rob124-eval] queued_command=${QUEUED_COMMAND}"
echo "[rob124-eval] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[rob124-eval] tmpdir=${TMPDIR}"

bash -ic 'python - <<'"'"'PY'"'"'
import sys
import torch

print("[rob124-eval] python_executable=" + sys.executable)
print("[rob124-eval] python_version=" + sys.version.split()[0])
print("[rob124-eval] torch_version=" + torch.__version__)
print("[rob124-eval] cuda_available=" + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("[rob124-eval] cuda_device=" + torch.cuda.get_device_name(0))
PY'

if [ "${ROB124_EVAL_CALLBACK_ONLY:-0}" = "1" ]; then
  echo "[rob124-eval] callback-only smoke path requested; exiting before config generation."
  exit 0
fi

for required_path in "${CHECKPOINT_PATH}" "${ASR_CKPT}" "${MASK_VAE_CKPT}"; do
  if [ ! -f "${required_path}" ]; then
    echo "Missing required checkpoint: ${required_path}" >&2
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

conditions = [
    {
        "method": "RewardConditionedMaskLMReward0",
        "label": "fixed reward 0.0",
        "default_reward": 0.0,
        "augmentation": "    conditioning_reward: 0.0\n    sample: true\n",
        "range_yaml": "",
    },
    {
        "method": "RewardConditionedMaskLMReward1",
        "label": "fixed reward 1.0",
        "default_reward": 1.0,
        "augmentation": "    conditioning_reward: 1.0\n    sample: true\n",
        "range_yaml": "",
    },
    {
        "method": "RewardConditionedMaskLMUniform0to1",
        "label": "uniform reward [0.0, 1.0]",
        "default_reward": 1.0,
        "augmentation": "    sample: true\n",
        "range_yaml": "    conditioning_reward_range: [0.0, 1.0]\n",
    },
    {
        "method": "RewardConditionedMaskLMUniform0p5to1",
        "label": "uniform reward [0.5, 1.0]",
        "default_reward": 1.0,
        "augmentation": "    sample: true\n",
        "range_yaml": "    conditioning_reward_range: [0.5, 1.0]\n",
    },
]

for condition in conditions:
    method = condition["method"]
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
  model_save_path: {policy_ckpt}
  tmp_model_save_path: {policy_ckpt}
  prefetch_factor: null
  num_workers: 0

evaluation:
  id: {linear_issue}-earnings22-test-384d-dropout0p1-{method}-epoch1-lr1e-5
  dataset: earnings22
  split: test
  rollout_fn: multistep
  use_cer: false
  epochs: 1
  augmentation_config:
{condition["augmentation"]}  optim_args:
    lr: 1e-5
  save_path: {save_path}

policy:
  lr: 1e-3
  class: RewardConditionedMaskLM
  config:
    hidden_dim: 384
    dropout: 0.1
    default_conditioning_reward: {condition["default_reward"]}
{condition["range_yaml"]}    reward_encoder: timestep
    sample_generation: true
    mask_vae_state_dict_path: {mask_vae_ckpt}
    mask_vae_config:
      latent_dim: 128
      codebook_size: 2048
      use_vq: true
""",
        encoding="utf-8",
    )
    print(f"[rob124-eval] wrote config {config_path} ({condition['label']})")
PY

if [ "${ROB124_EVAL_CONFIG_ONLY:-0}" = "1" ]; then
  echo "[rob124-eval] config-only path requested; exiting before eval."
  exit 0
fi

CONDITIONS="${ROB124_EVAL_CONDITIONS:-RewardConditionedMaskLMReward0 RewardConditionedMaskLMReward1 RewardConditionedMaskLMUniform0to1 RewardConditionedMaskLMUniform0p5to1}"

if [ "${ROB124_EVAL_SMOKE:-0}" = "1" ]; then
  CONDITIONS="${ROB124_EVAL_SMOKE_CONDITIONS:-RewardConditionedMaskLMReward1}"
  ROB124_EVAL_INDEXES="${ROB124_EVAL_INDEXES:-0}"
  ROB124_EVAL_DONT_SAVE="${ROB124_EVAL_DONT_SAVE:-1}"
  export ROB124_EVAL_INDEXES ROB124_EVAL_DONT_SAVE
  echo "[rob124-eval] smoke mode: conditions=${CONDITIONS}; indexes=${ROB124_EVAL_INDEXES}; dont_save=${ROB124_EVAL_DONT_SAVE}"
fi

for method in ${CONDITIONS}; do
  config="${RESULT_ROOT}/${method}/configs/earnings22_test_epoch1_lr1e-5.yaml"
  save_path="${RESULT_ROOT}/${method}/earnings22_test_epoch1_lr1e-5.txt"
  if [ ! -f "${config}" ]; then
    echo "Missing generated config: ${config}" >&2
    exit 1
  fi
  if [ "${FORCE_RERUN:-0}" = "1" ]; then
    rm -f "${save_path}"
  elif [ -f "${save_path}" ] && grep -q "Updated_WER:" "${save_path}"; then
    echo "[rob124-eval] skipping completed ${method}: ${save_path}"
    continue
  fi
  echo "[rob124-eval] running ${method}: ${config}"
  CONFIG_TO_RUN="${config}" bash -ic '
    cd "$REPO_DIR/exp"
    export PYTHONPATH="$REPO_DIR:$REPO_DIR/exp:/exp/exp4/acp21rjf/long-context-asr:/exp/exp4/acp21rjf/language_modelling${PYTHONPATH:+:$PYTHONPATH}"
    args=(python eval.py --config "$CONFIG_TO_RUN")
    if [ -n "${ROB124_EVAL_INDEXES:-}" ]; then
      read -r -a indexes <<< "$ROB124_EVAL_INDEXES"
      args+=(--indexes "${indexes[@]}")
    fi
    if [ "${ROB124_EVAL_DONT_SAVE:-0}" = "1" ]; then
      args+=(--dont_save)
    fi
    echo "[rob124-eval] eval command: ${args[*]}"
    "${args[@]}"
  '
done

python3 scripts/summarize_rob124_384_dropout_earnings_reward_controls.py \
  --result-root "${RESULT_ROOT}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --baseline-csv "${BASELINE_CSV}" \
  --command "${QUEUED_COMMAND}" \
  --branch "${GIT_BRANCH}" \
  --commit "${GIT_COMMIT}" \
  --log-path "${LOG_PATH}" \
  --screen-log-path "${SCREEN_LOG_PATH}"

echo "[rob124-eval] finished"
