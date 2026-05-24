#!/usr/bin/env bash
# Callback-backed ROB-120 Earnings reward-control evaluation.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${LINEAR_ISSUE:-ROB-120}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_DIR}/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob120_earnings_reward_controls}"
LOG_PATH="${LOG_PATH:-${RESULT_ROOT}/logs/rob120_earnings_reward_controls.log}"
SCREEN_LOG_PATH="${SCREEN_LOG_PATH:-${RESULT_ROOT}/logs/rob120_earnings_reward_controls.screen.log}"
SCREEN_NAME="${SCREEN_NAME:-rob120-earnings-reward-controls}"
RUNNER_LABEL="${RUNNER_LABEL:-screen:${SCREEN_NAME}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_resume100_500ep_lr1e3.pt}"
ASR_CKPT="${ASR_CKPT:-/store/store5/data/acp21rjf_checkpoints/spotify/rotary_pos_6l_256d_seq_sched/n_seq_sched_2048_rp_1/step_105360.pt}"
MASK_VAE_CKPT="${MASK_VAE_CKPT:-/store/store5/data/acp21rjf_checkpoints/l2augment/models/bvae/bvae_USINGTHISFORNOW_2048gpu.pt}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/exp/exp4/acp21rjf/rob117-scratch}"
QUEUED_COMMAND="${QUEUED_COMMAND:-screen -L -Logfile ${SCREEN_LOG_PATH} -dmS ${SCREEN_NAME} bash -lc 'cd ${REPO_DIR} && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob120_earnings_reward_controls.sh'}"
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}"

on_exit() {
  status=$?
  set +e
  if [ "${ROB120_DISABLE_CALLBACK:-0}" = "1" ]; then
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
    --note "${CALLBACK_NOTE:-ROB-120 Earnings reward-control evaluation wrapper exited. Inspect OUTCOME.md under ${RESULT_ROOT}.}"
    --tail-lines "${CALLBACK_TAIL_LINES:-80}"
    --max-log-chars "${CALLBACK_MAX_LOG_CHARS:-6000}"
    --max-comment-chars "${CALLBACK_MAX_COMMENT_CHARS:-10000}"
  )
  if [ "${ROB120_CALLBACK_CHECK_ONLY:-0}" = "1" ]; then
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
export CHECKPOINT_PATH ASR_CKPT MASK_VAE_CKPT QUEUED_COMMAND GIT_BRANCH GIT_COMMIT
export TMPDIR="${SCRATCH_ROOT}/tmp"
export WANDB_DIR="${RESULT_ROOT}/wandb"
export WANDB_CACHE_DIR="${SCRATCH_ROOT}/wandb-cache"
export WANDB_CONFIG_DIR="${SCRATCH_ROOT}/wandb-config"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/xdg-cache"
export MPLCONFIGDIR="${SCRATCH_ROOT}/matplotlib"
mkdir -p "${TMPDIR}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"

echo "[rob120] branch=${GIT_BRANCH}"
echo "[rob120] commit=${GIT_COMMIT}"
echo "[rob120] host=$(hostname)"
echo "[rob120] result_root=${RESULT_ROOT}"
echo "[rob120] checkpoint=${CHECKPOINT_PATH}"
echo "[rob120] asr_ckpt=${ASR_CKPT}"
echo "[rob120] mask_vae_ckpt=${MASK_VAE_CKPT}"
echo "[rob120] log_path=${LOG_PATH}"
echo "[rob120] screen_log_path=${SCREEN_LOG_PATH}"
echo "[rob120] queued_command=${QUEUED_COMMAND}"
echo "[rob120] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[rob120] tmpdir=${TMPDIR}"

bash -ic 'python - <<'"'"'PY'"'"'
import sys
import torch

print("[rob120] python_executable=" + sys.executable)
print("[rob120] python_version=" + sys.version.split()[0])
print("[rob120] torch_version=" + torch.__version__)
print("[rob120] cuda_available=" + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("[rob120] cuda_device=" + torch.cuda.get_device_name(0))
PY'

if [ "${ROB120_CALLBACK_ONLY:-0}" = "1" ]; then
  echo "[rob120] callback-only smoke path requested; exiting before config generation."
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
  id: {linear_issue}-earnings22-test-{method}-epoch1-lr1e-5
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
    hidden_dim: 256
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
    print(f"[rob120] wrote config {config_path} ({condition['label']})")
PY

if [ "${ROB120_CONFIG_ONLY:-0}" = "1" ]; then
  echo "[rob120] config-only path requested; exiting before eval."
  exit 0
fi

CONDITIONS="${ROB120_CONDITIONS:-RewardConditionedMaskLMReward0 RewardConditionedMaskLMReward1 RewardConditionedMaskLMUniform0to1 RewardConditionedMaskLMUniform0p5to1}"

if [ "${ROB120_SMOKE:-0}" = "1" ]; then
  CONDITIONS="${ROB120_SMOKE_CONDITIONS:-RewardConditionedMaskLMReward1}"
  ROB120_INDEXES="${ROB120_INDEXES:-0}"
  ROB120_DONT_SAVE="${ROB120_DONT_SAVE:-1}"
  echo "[rob120] smoke mode: conditions=${CONDITIONS}; indexes=${ROB120_INDEXES}; dont_save=${ROB120_DONT_SAVE}"
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
    echo "[rob120] skipping completed ${method}: ${save_path}"
    continue
  fi
  echo "[rob120] running ${method}: ${config}"
  CONFIG_TO_RUN="${config}" bash -ic '
    cd "$REPO_DIR/exp"
    export PYTHONPATH="$REPO_DIR:$REPO_DIR/exp:/exp/exp4/acp21rjf/long-context-asr:/exp/exp4/acp21rjf/language_modelling${PYTHONPATH:+:$PYTHONPATH}"
    args=(python eval.py --config "$CONFIG_TO_RUN")
    if [ -n "${ROB120_INDEXES:-}" ]; then
      read -r -a indexes <<< "$ROB120_INDEXES"
      args+=(--indexes "${indexes[@]}")
    fi
    if [ "${ROB120_DONT_SAVE:-0}" = "1" ]; then
      args+=(--dont_save)
    fi
    echo "[rob120] eval command: ${args[*]}"
    "${args[@]}"
  '
done

python3 scripts/summarize_rob120_earnings_reward_controls.py \
  --result-root "${RESULT_ROOT}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --command "${QUEUED_COMMAND}" \
  --branch "${GIT_BRANCH}" \
  --commit "${GIT_COMMIT}" \
  --log-path "${LOG_PATH}" \
  --screen-log-path "${SCREEN_LOG_PATH}"

echo "[rob120] finished"
