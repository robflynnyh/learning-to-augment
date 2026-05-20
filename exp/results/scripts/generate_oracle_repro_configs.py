#!/usr/bin/env python3
"""Generate reproducible grid configs for oracle and UFMR repro experiments."""

from __future__ import annotations

import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "exp/results"
REPRO_ROOT = RESULTS_ROOT / "repro"
ORACLE_REPRO_ROOT = REPRO_ROOT / "oracle"
POLICY_REPRO_ROOT = REPRO_ROOT / "policy"
REPRO_OUTPUT_ROOT = "exp/results/repro"
ASR_MODEL = "/store/store5/data/acp21rjf_checkpoints/l2augment/asr/step_105360.pt"
UFMR_MODEL = "/store/store5/data/acp21rjf_checkpoints/l2augment/ufmr/mseloss/model.pt"
UMLM_MODEL = "/store/store5/data/acp21rjf_checkpoints/l2augment/models/UMLM/modelgpu.pt"
BVAE_MODEL = "/store/store5/data/acp21rjf_checkpoints/l2augment/models/bvae/bvae_USINGTHISFORNOW_2048gpu.pt"
REPEATS = "1, 2, 3, 4, 5, 10, 20, 50"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if path.suffix == ".sh":
        path.chmod(0o755)


def remove_old_config_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def oracle_grid_config(method: str) -> str:
    class_block = {
        "RFM": "  class: FrequencyMaskingRanker\n",
        "RMM": """  class: MixedMaskingRanker
  config:
    time_masks_min: 3
    time_masks_max: 16
    freq_masks_min: 5
    freq_masks_max: 7
    freq_mask_param_min: 34
    freq_mask_param_max: 34
""",
    }[method]
    return f"""checkpointing:
  asr_model: {ASR_MODEL}

training:
  device: cpu
  random_seed: 1234
  batch_size: 84
  epochs: 100

evaluation:
  rollout_setting: search
  search_repeats: 1
  split: test
  use_cer: false
  epochs: 1
  augmentation_config:
    repeats: 1
    use_random: true
  optim_args:
    lr: 1e-6
    single_step_lr: 4e-2
  save_path: {REPRO_OUTPUT_ROOT}/oracle/{method}/tedlium_lr{{grid_label:evaluation.optim_args.lr}}_searchlr{{grid_label:evaluation.optim_args.single_step_lr}}.txt

policy:
  lr: 1e-4
{class_block}
grid:
  name: {method.lower()}_repro_oracle_tedlium_test
  combine: product
  id_template: "tedlium_lr{{grid_label:evaluation.optim_args.lr}}_searchlr{{grid_label:evaluation.optim_args.single_step_lr}}_repeats{{evaluation.search_repeats}}"
  axes:
    evaluation.search_repeats: [{REPEATS}]
  cases:
    - id: historical
      values:
        evaluation.optim_args.lr:
          value: 1e-6
          label: "1e-6"
        evaluation.optim_args.single_step_lr:
          value: 4e-2
          label: "4e-2"
    - id: recent
      values:
        evaluation.optim_args.lr:
          value: 8e-6
          label: "8e-6"
        evaluation.optim_args.single_step_lr:
          value: 9e-2
          label: "9e-2"
    - id: lr1e-5_searchlr2e-1
      values:
        evaluation.optim_args.lr:
          value: 1e-5
          label: "1e-5"
        evaluation.optim_args.single_step_lr:
          value: 2e-1
          label: "2e-1"
    - id: lr8e-6_searchlr2e-1
      values:
        evaluation.optim_args.lr:
          value: 8e-6
          label: "8e-6"
        evaluation.optim_args.single_step_lr:
          value: 2e-1
          label: "2e-1"
    - id: lr1e-5_searchlr9e-2
      values:
        evaluation.optim_args.lr:
          value: 1e-5
          label: "1e-5"
        evaluation.optim_args.single_step_lr:
          value: 9e-2
          label: "9e-2"
    - id: lr3e-5_searchlr2e-1
      values:
        evaluation.optim_args.lr:
          value: 3e-5
          label: "3e-5"
        evaluation.optim_args.single_step_lr:
          value: 2e-1
          label: "2e-1"
    - id: lr1e-4_searchlr2e-1
      values:
        evaluation.optim_args.lr:
          value: 1e-4
          label: "1e-4"
        evaluation.optim_args.single_step_lr:
          value: 2e-1
          label: "2e-1"
    - id: lr6e-5_searchlr2e-1
      values:
        evaluation.optim_args.lr:
          value: 6e-5
          label: "6e-5"
        evaluation.optim_args.single_step_lr:
          value: 2e-1
          label: "2e-1"
"""


def uvqlm_grid_config() -> str:
    return f"""checkpointing:
  asr_model: {ASR_MODEL}

training:
  device: cuda
  random_seed: 1234
  batch_size: 84
  epochs: 100
  model_save_path: {UMLM_MODEL}
  tmp_model_save_path: {UMLM_MODEL}

evaluation:
  rollout_setting: search
  search_repeats: 1
  split: test
  use_cer: false
  epochs: 1
  augmentation_config:
    repeats: 1
    use_random: true
  optim_args:
    lr: 1e-5
    single_step_lr: 2e-1
  save_path: {REPRO_OUTPUT_ROOT}/oracle/UVQLM/tedlium_lr{{grid_label:evaluation.optim_args.lr}}_searchlr{{grid_label:evaluation.optim_args.single_step_lr}}.txt

policy:
  lr: 6e-4
  class: UnconditionalMaskGenerator
  config:
    mask_vae_state_dict_path: {BVAE_MODEL}
    mask_vae_config:
      latent_dim: 128
      codebook_size: 2048
      use_vq: true

grid:
  name: uvqlm_repro_oracle_tedlium_test
  combine: product
  id_template: "tedlium_lr{{grid_label:evaluation.optim_args.lr}}_searchlr{{grid_label:evaluation.optim_args.single_step_lr}}_repeats{{evaluation.search_repeats}}"
  axes:
    evaluation.search_repeats: [{REPEATS}]
  cases:
    - id: lr1e-5_searchlr2e-1
      values:
        evaluation.optim_args.lr:
          value: 1e-5
          label: "1e-5"
        evaluation.optim_args.single_step_lr:
          value: 2e-1
          label: "2e-1"
    - id: lr3e-5_searchlr2e-1
      values:
        evaluation.optim_args.lr:
          value: 3e-5
          label: "3e-5"
        evaluation.optim_args.single_step_lr:
          value: 2e-1
          label: "2e-1"
    - id: lr1e-4_searchlr2e-1
      values:
        evaluation.optim_args.lr:
          value: 1e-4
          label: "1e-4"
        evaluation.optim_args.single_step_lr:
          value: 2e-1
          label: "2e-1"
    - id: lr6e-5_searchlr2e-1
      values:
        evaluation.optim_args.lr:
          value: 6e-5
          label: "6e-5"
        evaluation.optim_args.single_step_lr:
          value: 2e-1
          label: "2e-1"
"""


def random_additive_grid_config() -> str:
    return f"""checkpointing:
  asr_model: {ASR_MODEL}

training:
  device: cpu
  random_seed: 1234
  batch_size: 84
  epochs: 100

evaluation:
  rollout_setting: search
  search_repeats: 1
  split: test
  use_cer: false
  epochs: 1
  augmentation_config:
    repeats: 1
    use_random: true
  optim_args:
    lr: 3e-5
    single_step_lr: 2e-1
  save_path: {REPRO_OUTPUT_ROOT}/oracle/RAN/tedlium_lr{{grid_label:evaluation.optim_args.lr}}_searchlr{{grid_label:evaluation.optim_args.single_step_lr}}.txt

policy:
  lr: 1e-4
  class: AdditivePolicy

grid:
  name: random_additive_repro_oracle_tedlium_test
  combine: product
  id_template: "tedlium_lr{{grid_label:evaluation.optim_args.lr}}_searchlr{{grid_label:evaluation.optim_args.single_step_lr}}_repeats{{evaluation.search_repeats}}"
  axes:
    evaluation.search_repeats: [{REPEATS}]
  cases:
    - id: lr3e-5_searchlr2e-1
      values:
        evaluation.optim_args.lr:
          value: 3e-5
          label: "3e-5"
        evaluation.optim_args.single_step_lr:
          value: 2e-1
          label: "2e-1"
"""


def ufmr_grid_config() -> str:
    return f"""checkpointing:
  asr_model: {ASR_MODEL}

training:
  device: cpu
  random_seed: 1234
  batch_size: 84
  epochs: 100
  model_save_path: {UFMR_MODEL}
  tmp_model_save_path: {UFMR_MODEL}

evaluation:
  rollout_setting: policy
  split: test
  use_cer: false
  epochs: 1
  augmentation_config:
    repeats: 15
    use_random: false
  optim_args:
    lr: 1e-6
  save_path: {REPRO_OUTPUT_ROOT}/policy/UFMR_segmented/tedlium_lr{{grid_label:evaluation.optim_args.lr}}.txt

policy:
  lr: 1e-4
  class: UnconditionalFrequencyMaskingRanker

grid:
  name: ufmr_segmented_repro_tedlium_test
  id_template: "tedlium_lr{{grid_label:evaluation.optim_args.lr}}"
  axes:
    evaluation.optim_args.lr:
      - value: 1e-6
        label: "1e-6"
      - value: 8e-6
        label: "8e-6"
"""


def oracle_stream_script(method: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../../.."
export CUDA_VISIBLE_DEVICES=""
export MPLCONFIGDIR="${{MPLCONFIGDIR:-/exp/exp4/acp21rjf/.scratch/matplotlib-cache}}"
export L2A_TEDLIUM3_LEGACY_DIR="${{L2A_TEDLIUM3_LEGACY_DIR:-/store/store4/data/TEDLIUM_release-3/legacy}}"
mkdir -p "${{MPLCONFIGDIR}}" "exp/results/repro/oracle/{method}/logs"

: > "exp/results/repro/oracle/{method}/tedlium_lr1e-6_searchlr4e-2.txt"
: > "exp/results/repro/oracle/{method}/tedlium_lr8e-6_searchlr9e-2.txt"

PYTHONDONTWRITEBYTECODE=1 python3 exp/run_config_grid.py \\
  --grid-config "exp/results/repro/oracle/{method}/tedlium_grid.yaml" \\
  --materialize-only

config_dir="exp/results/repro/oracle/{method}/.generated/tedlium_grid"

run_combo() {{
  local lr="$1"
  local search_lr="$2"
  for repeats in 1 2 3 4 5 10 20 50; do
    PYTHONDONTWRITEBYTECODE=1 python3 exp/oracle_eval.py \\
      --config "${{config_dir}}/tedlium_lr${{lr}}_searchlr${{search_lr}}_repeats${{repeats}}.yaml"
  done
}}

run_combo 1e-6 4e-2
run_combo 8e-6 9e-2
"""


def screen_safe_lr_name(lr_name: str) -> str:
    return lr_name.replace("-", "")


def gpu_wrapper(lr_name: str, search_lr_name: str) -> str:
    screen_lr_name = screen_safe_lr_name(lr_name)
    screen_search_lr_name = screen_safe_lr_name(search_lr_name)
    screen_name = f"l2a_oracle_lr{screen_lr_name}_searchlr{screen_search_lr_name}"
    log_stem = f"lr{lr_name}_searchlr{search_lr_name}"
    script_name = f"{log_stem}_gpu.sh"
    return f"""#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/../../../../.."

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${{LINEAR_ISSUE:-ROB-60}}"
SCREEN_NAME="${{SCREEN_NAME:-{screen_name}}}"
RUNNER_LABEL="${{RUNNER_LABEL:-screen:${{SCREEN_NAME}}}}"
LOG_PATH="${{LOG_PATH:-exp/results/repro/oracle/logs/{log_stem}_gpu.log}}"
RESULTS_PATH="${{RESULTS_PATH:-exp/results/repro/oracle}}"
QUEUED_COMMAND="${{QUEUED_COMMAND:-screen -L -Logfile exp/results/repro/oracle/logs/{log_stem}_queue.log -dmS {screen_name} bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-60 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash exp/results/repro/oracle/jobs/{script_name}'}}"
GIT_BRANCH="${{GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}}"
GIT_COMMIT="${{GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}}"

callback_args=()
if [ "${{L2A_CALLBACK_DRY_RUN:-0}}" = "1" ]; then
  callback_args+=(--dry-run)
fi

on_exit() {{
  status=$?
  set +e
  if [ -z "${{LINEAR_API_KEY:-}}" ] && [ "${{L2A_CALLBACK_DRY_RUN:-0}}" != "1" ]; then
    echo "LINEAR_API_KEY is not set; cannot post Linear completion callback" >&2
    exit "${{status}}"
  fi
  python3 scripts/callbacks/linear_experiment_callback.py \\
    --issue "${{LINEAR_ISSUE}}" \\
    --status-code "${{status}}" \\
    --log "${{LOG_PATH}}" \\
    --results "${{RESULTS_PATH}}" \\
    --screen-name "${{SCREEN_NAME}}" \\
    --runner-label "${{RUNNER_LABEL}}" \\
    --queued-command "${{QUEUED_COMMAND}}" \\
    --branch "${{GIT_BRANCH}}" \\
    --commit "${{GIT_COMMIT}}" \\
    --note "ROB-60 lr={lr_name}/search_lr={search_lr_name} oracle sweep finished. Inspect RMM/RFM text files and update OUTCOME/plot before final handoff." \\
    "${{callback_args[@]}}"
  callback_status=$?
  if [ "${{callback_status}}" -ne 0 ]; then
    echo "Linear completion callback failed with status ${{callback_status}}" >&2
  fi
  exit "${{status}}"
}}
trap on_exit EXIT

set -euo pipefail

export MPLCONFIGDIR="${{MPLCONFIGDIR:-/exp/exp4/acp21rjf/.scratch/matplotlib-cache}}"
export L2A_TEDLIUM3_LEGACY_DIR="${{L2A_TEDLIUM3_LEGACY_DIR:-/store/store4/data/TEDLIUM_release-3/legacy}}"
export PYTHONPATH="$PWD:${{PYTHONPATH:-}}"
mkdir -p "${{MPLCONFIGDIR}}" "$(dirname "${{LOG_PATH}}")" \\
  exp/results/repro/oracle/RMM exp/results/repro/oracle/RFM

exec > >(tee -a "${{LOG_PATH}}") 2>&1

echo "[$(date -Iseconds)] ROB-60 oracle sweep lr={lr_name} search_lr={search_lr_name}"
echo "branch=${{GIT_BRANCH}} commit=${{GIT_COMMIT}} CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-unset}}"

if [ "${{L2A_CALLBACK_SMOKE_TEST:-0}}" = "1" ]; then
  echo "callback smoke test requested; exiting before experiment"
  exit 0
fi

run_combo() {{
  local method="$1"
  local result="exp/results/repro/oracle/${{method}}/tedlium_lr{lr_name}_searchlr{search_lr_name}.txt"
  local config_dir="exp/results/repro/oracle/${{method}}/.generated/tedlium_grid"
  : > "${{result}}"
  PYTHONDONTWRITEBYTECODE=1 python3 exp/run_config_grid.py \\
    --grid-config "exp/results/repro/oracle/${{method}}/tedlium_grid.yaml" \\
    --materialize-only
  for repeats in 1 2 3 4 5 10 20 50; do
    echo "[$(date -Iseconds)] running ${{method}} repeats=${{repeats}}"
    PYTHONDONTWRITEBYTECODE=1 python3 exp/oracle_eval.py \\
      --config "${{config_dir}}/tedlium_lr{lr_name}_searchlr{search_lr_name}_repeats${{repeats}}.yaml"
  done
}}

run_combo RMM
run_combo RFM

echo "[$(date -Iseconds)] completed ROB-60 oracle sweep lr={lr_name} search_lr={search_lr_name}"
"""


def uvqlm_gpu_wrapper(lr_name: str, search_lr_name: str) -> str:
    screen_lr_name = screen_safe_lr_name(lr_name)
    screen_search_lr_name = screen_safe_lr_name(search_lr_name)
    screen_name = f"l2a_oracle_uvqlm_lr{screen_lr_name}_searchlr{screen_search_lr_name}"
    log_stem = f"uvqlm_lr{lr_name}_searchlr{search_lr_name}"
    script_name = f"{log_stem}_gpu.sh"
    return f"""#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/../../../../.."

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${{LINEAR_ISSUE:-ROB-60}}"
SCREEN_NAME="${{SCREEN_NAME:-{screen_name}}}"
RUNNER_LABEL="${{RUNNER_LABEL:-screen:${{SCREEN_NAME}}}}"
LOG_PATH="${{LOG_PATH:-exp/results/repro/oracle/logs/{log_stem}_gpu.log}}"
RESULTS_PATH="${{RESULTS_PATH:-exp/results/repro/oracle/UVQLM}}"
QUEUED_COMMAND="${{QUEUED_COMMAND:-screen -L -Logfile exp/results/repro/oracle/logs/{log_stem}_queue.log -dmS {screen_name} bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-60 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash exp/results/repro/oracle/jobs/{script_name}'}}"
GIT_BRANCH="${{GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}}"
GIT_COMMIT="${{GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}}"

callback_args=()
if [ "${{L2A_CALLBACK_DRY_RUN:-0}}" = "1" ]; then
  callback_args+=(--dry-run)
fi

on_exit() {{
  status=$?
  set +e
  if [ -z "${{LINEAR_API_KEY:-}}" ] && [ "${{L2A_CALLBACK_DRY_RUN:-0}}" != "1" ]; then
    echo "LINEAR_API_KEY is not set; cannot post Linear completion callback" >&2
    exit "${{status}}"
  fi
  python3 scripts/callbacks/linear_experiment_callback.py \\
    --issue "${{LINEAR_ISSUE}}" \\
    --status-code "${{status}}" \\
    --log "${{LOG_PATH}}" \\
    --results "${{RESULTS_PATH}}" \\
    --screen-name "${{SCREEN_NAME}}" \\
    --runner-label "${{RUNNER_LABEL}}" \\
    --queued-command "${{QUEUED_COMMAND}}" \\
    --branch "${{GIT_BRANCH}}" \\
    --commit "${{GIT_COMMIT}}" \\
    --note "ROB-60 UVQLM oracle sweep lr={lr_name}/search_lr={search_lr_name} finished. Inspect UVQLM text file and update OUTCOME/plot before final handoff." \\
    "${{callback_args[@]}}"
  callback_status=$?
  if [ "${{callback_status}}" -ne 0 ]; then
    echo "Linear completion callback failed with status ${{callback_status}}" >&2
  fi
  exit "${{status}}"
}}
trap on_exit EXIT

set -euo pipefail

export MPLCONFIGDIR="${{MPLCONFIGDIR:-/exp/exp4/acp21rjf/.scratch/matplotlib-cache}}"
export L2A_TEDLIUM3_LEGACY_DIR="${{L2A_TEDLIUM3_LEGACY_DIR:-/store/store4/data/TEDLIUM_release-3/legacy}}"
export PYTHONPATH="$PWD:${{PYTHONPATH:-}}"
mkdir -p "${{MPLCONFIGDIR}}" "$(dirname "${{LOG_PATH}}")" exp/results/repro/oracle/UVQLM

exec > >(tee -a "${{LOG_PATH}}") 2>&1

echo "[$(date -Iseconds)] ROB-60 UVQLM oracle sweep lr={lr_name} search_lr={search_lr_name}"
echo "branch=${{GIT_BRANCH}} commit=${{GIT_COMMIT}} CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-unset}}"

if [ "${{L2A_CALLBACK_SMOKE_TEST:-0}}" = "1" ]; then
  echo "callback smoke test requested; exiting before experiment"
  exit 0
fi

result="exp/results/repro/oracle/UVQLM/tedlium_lr{lr_name}_searchlr{search_lr_name}.txt"
config_dir="exp/results/repro/oracle/UVQLM/.generated/tedlium_grid"
: > "${{result}}"
PYTHONDONTWRITEBYTECODE=1 python3 exp/run_config_grid.py \\
  --grid-config "exp/results/repro/oracle/UVQLM/tedlium_grid.yaml" \\
  --materialize-only

for repeats in 1 2 3 4 5 10 20 50; do
  echo "[$(date -Iseconds)] running UVQLM repeats=${{repeats}}"
  PYTHONDONTWRITEBYTECODE=1 python3 exp/oracle_eval.py \\
    --config "${{config_dir}}/tedlium_lr{lr_name}_searchlr{search_lr_name}_repeats${{repeats}}.yaml"
done

echo "[$(date -Iseconds)] completed ROB-60 UVQLM oracle sweep lr={lr_name} search_lr={search_lr_name}"
"""


def all_policy_gpu_wrapper(lr_name: str, search_lr_name: str) -> str:
    screen_lr_name = screen_safe_lr_name(lr_name)
    screen_search_lr_name = screen_safe_lr_name(search_lr_name)
    screen_name = f"l2a_oracle_all_lr{screen_lr_name}_searchlr{screen_search_lr_name}"
    log_stem = f"lr{lr_name}_searchlr{search_lr_name}_all_policies"
    script_name = f"{log_stem}_gpu.sh"
    return f"""#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/../../../../.."

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${{LINEAR_ISSUE:-ROB-60}}"
SCREEN_NAME="${{SCREEN_NAME:-{screen_name}}}"
RUNNER_LABEL="${{RUNNER_LABEL:-screen:${{SCREEN_NAME}}}}"
LOG_PATH="${{LOG_PATH:-exp/results/repro/oracle/logs/{log_stem}_gpu.log}}"
RESULTS_PATH="${{RESULTS_PATH:-exp/results/repro/oracle}}"
QUEUED_COMMAND="${{QUEUED_COMMAND:-screen -L -Logfile exp/results/repro/oracle/logs/{log_stem}_queue.log -dmS {screen_name} bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-60 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash exp/results/repro/oracle/jobs/{script_name}'}}"
GIT_BRANCH="${{GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}}"
GIT_COMMIT="${{GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}}"

callback_args=()
if [ "${{L2A_CALLBACK_DRY_RUN:-0}}" = "1" ]; then
  callback_args+=(--dry-run)
fi

on_exit() {{
  status=$?
  set +e
  if [ -z "${{LINEAR_API_KEY:-}}" ] && [ "${{L2A_CALLBACK_DRY_RUN:-0}}" != "1" ]; then
    echo "LINEAR_API_KEY is not set; cannot post Linear completion callback" >&2
    exit "${{status}}"
  fi
  python3 scripts/callbacks/linear_experiment_callback.py \\
    --issue "${{LINEAR_ISSUE}}" \\
    --status-code "${{status}}" \\
    --log "${{LOG_PATH}}" \\
    --results "${{RESULTS_PATH}}" \\
    --screen-name "${{SCREEN_NAME}}" \\
    --runner-label "${{RUNNER_LABEL}}" \\
    --queued-command "${{QUEUED_COMMAND}}" \\
    --branch "${{GIT_BRANCH}}" \\
    --commit "${{GIT_COMMIT}}" \\
    --note "ROB-60 RMM/RFM/UVQLM oracle sweep lr={lr_name}/search_lr={search_lr_name} finished. Inspect text files and update OUTCOME/plot before final handoff." \\
    "${{callback_args[@]}}"
  callback_status=$?
  if [ "${{callback_status}}" -ne 0 ]; then
    echo "Linear completion callback failed with status ${{callback_status}}" >&2
  fi
  exit "${{status}}"
}}
trap on_exit EXIT

set -euo pipefail

export MPLCONFIGDIR="${{MPLCONFIGDIR:-/exp/exp4/acp21rjf/.scratch/matplotlib-cache}}"
export L2A_TEDLIUM3_LEGACY_DIR="${{L2A_TEDLIUM3_LEGACY_DIR:-/store/store4/data/TEDLIUM_release-3/legacy}}"
export PYTHONPATH="$PWD:${{PYTHONPATH:-}}"
mkdir -p "${{MPLCONFIGDIR}}" "$(dirname "${{LOG_PATH}}")" \\
  exp/results/repro/oracle/RMM exp/results/repro/oracle/RFM exp/results/repro/oracle/UVQLM

exec > >(tee -a "${{LOG_PATH}}") 2>&1

echo "[$(date -Iseconds)] ROB-60 RMM/RFM/UVQLM oracle sweep lr={lr_name} search_lr={search_lr_name}"
echo "branch=${{GIT_BRANCH}} commit=${{GIT_COMMIT}} CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-unset}}"

if [ "${{L2A_CALLBACK_SMOKE_TEST:-0}}" = "1" ]; then
  echo "callback smoke test requested; exiting before experiment"
  exit 0
fi

run_method() {{
  local method="$1"
  local result="exp/results/repro/oracle/${{method}}/tedlium_lr{lr_name}_searchlr{search_lr_name}.txt"
  local config_dir="exp/results/repro/oracle/${{method}}/.generated/tedlium_grid"
  : > "${{result}}"
  PYTHONDONTWRITEBYTECODE=1 python3 exp/run_config_grid.py \\
    --grid-config "exp/results/repro/oracle/${{method}}/tedlium_grid.yaml" \\
    --materialize-only
  for repeats in 1 2 3 4 5 10 20 50; do
    echo "[$(date -Iseconds)] running ${{method}} repeats=${{repeats}}"
    PYTHONDONTWRITEBYTECODE=1 python3 exp/oracle_eval.py \\
      --config "${{config_dir}}/tedlium_lr{lr_name}_searchlr{search_lr_name}_repeats${{repeats}}.yaml"
  done
}}

run_method RMM
run_method RFM
run_method UVQLM

echo "[$(date -Iseconds)] completed ROB-60 RMM/RFM/UVQLM oracle sweep lr={lr_name} search_lr={search_lr_name}"
"""


def random_additive_gpu_wrapper(lr_name: str, search_lr_name: str) -> str:
    screen_lr_name = screen_safe_lr_name(lr_name)
    screen_search_lr_name = screen_safe_lr_name(search_lr_name)
    screen_name = f"l2a_oracle_ran_lr{screen_lr_name}_searchlr{screen_search_lr_name}"
    log_stem = f"ran_lr{lr_name}_searchlr{search_lr_name}"
    script_name = f"{log_stem}_gpu.sh"
    return f"""#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/../../../../.."

if [ -f /exp/exp4/acp21rjf/symphony-config/.env ]; then
  set -a
  . /exp/exp4/acp21rjf/symphony-config/.env
  set +a
fi

LINEAR_ISSUE="${{LINEAR_ISSUE:-ROB-60}}"
SCREEN_NAME="${{SCREEN_NAME:-{screen_name}}}"
RUNNER_LABEL="${{RUNNER_LABEL:-screen:${{SCREEN_NAME}}}}"
LOG_PATH="${{LOG_PATH:-exp/results/repro/oracle/logs/{log_stem}_gpu.log}}"
RESULTS_PATH="${{RESULTS_PATH:-exp/results/repro/oracle/RAN}}"
QUEUED_COMMAND="${{QUEUED_COMMAND:-screen -L -Logfile exp/results/repro/oracle/logs/{log_stem}_queue.log -dmS {screen_name} bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-60 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash exp/results/repro/oracle/jobs/{script_name}'}}"
GIT_BRANCH="${{GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'unknown')}}"
GIT_COMMIT="${{GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || printf 'unknown')}}"

callback_args=()
if [ "${{L2A_CALLBACK_DRY_RUN:-0}}" = "1" ]; then
  callback_args+=(--dry-run)
fi

on_exit() {{
  status=$?
  set +e
  if [ -z "${{LINEAR_API_KEY:-}}" ] && [ "${{L2A_CALLBACK_DRY_RUN:-0}}" != "1" ]; then
    echo "LINEAR_API_KEY is not set; cannot post Linear completion callback" >&2
    exit "${{status}}"
  fi
  python3 scripts/callbacks/linear_experiment_callback.py \\
    --issue "${{LINEAR_ISSUE}}" \\
    --status-code "${{status}}" \\
    --log "${{LOG_PATH}}" \\
    --results "${{RESULTS_PATH}}" \\
    --screen-name "${{SCREEN_NAME}}" \\
    --runner-label "${{RUNNER_LABEL}}" \\
    --queued-command "${{QUEUED_COMMAND}}" \\
    --branch "${{GIT_BRANCH}}" \\
    --commit "${{GIT_COMMIT}}" \\
    --note "ROB-60 random additive-noise oracle sweep lr={lr_name}/search_lr={search_lr_name} finished. Inspect RAN text file and update OUTCOME/plot before final handoff." \\
    "${{callback_args[@]}}"
  callback_status=$?
  if [ "${{callback_status}}" -ne 0 ]; then
    echo "Linear completion callback failed with status ${{callback_status}}" >&2
  fi
  exit "${{status}}"
}}
trap on_exit EXIT

set -euo pipefail

export MPLCONFIGDIR="${{MPLCONFIGDIR:-/exp/exp4/acp21rjf/.scratch/matplotlib-cache}}"
export L2A_TEDLIUM3_LEGACY_DIR="${{L2A_TEDLIUM3_LEGACY_DIR:-/store/store4/data/TEDLIUM_release-3/legacy}}"
export PYTHONPATH="$PWD:${{PYTHONPATH:-}}"
mkdir -p "${{MPLCONFIGDIR}}" "$(dirname "${{LOG_PATH}}")" exp/results/repro/oracle/RAN

exec > >(tee -a "${{LOG_PATH}}") 2>&1

echo "[$(date -Iseconds)] ROB-60 random additive-noise oracle sweep lr={lr_name} search_lr={search_lr_name}"
echo "branch=${{GIT_BRANCH}} commit=${{GIT_COMMIT}} CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-unset}}"

if [ "${{L2A_CALLBACK_SMOKE_TEST:-0}}" = "1" ]; then
  echo "callback smoke test requested; exiting before experiment"
  exit 0
fi

result="exp/results/repro/oracle/RAN/tedlium_lr{lr_name}_searchlr{search_lr_name}.txt"
config_dir="exp/results/repro/oracle/RAN/.generated/tedlium_grid"
: > "${{result}}"
PYTHONDONTWRITEBYTECODE=1 python3 exp/run_config_grid.py \\
  --grid-config "exp/results/repro/oracle/RAN/tedlium_grid.yaml" \\
  --materialize-only

for repeats in 1 2 3 4 5 10 20 50; do
  echo "[$(date -Iseconds)] running RAN repeats=${{repeats}}"
  PYTHONDONTWRITEBYTECODE=1 python3 exp/oracle_eval.py \\
    --config "${{config_dir}}/tedlium_lr{lr_name}_searchlr{search_lr_name}_repeats${{repeats}}.yaml"
done

echo "[$(date -Iseconds)] completed ROB-60 random additive-noise oracle sweep lr={lr_name} search_lr={search_lr_name}"
"""


def main() -> None:
    job_dir = ORACLE_REPRO_ROOT / "jobs"
    job_dir.mkdir(parents=True, exist_ok=True)
    for config_dir in [
        ORACLE_REPRO_ROOT / "RMM/configs",
        ORACLE_REPRO_ROOT / "RFM/configs",
        POLICY_REPRO_ROOT / "UFMR_segmented/configs",
    ]:
        remove_old_config_dir(config_dir)
    for script_path in job_dir.glob("*.sh"):
        script_path.unlink()
    for script_path in ORACLE_REPRO_ROOT.glob("launch_*.sh"):
        script_path.unlink()

    for method in ["RMM", "RFM"]:
        write(ORACLE_REPRO_ROOT / method / "tedlium_grid.yaml", oracle_grid_config(method))
        write(job_dir / f"{method.lower()}_historical_then_recent.sh", oracle_stream_script(method))

    write(job_dir / "lr1e-5_searchlr2e-1_gpu.sh", gpu_wrapper("1e-5", "2e-1"))
    write(job_dir / "lr8e-6_searchlr2e-1_gpu.sh", gpu_wrapper("8e-6", "2e-1"))
    write(job_dir / "lr1e-5_searchlr9e-2_gpu.sh", gpu_wrapper("1e-5", "9e-2"))
    write(job_dir / "lr3e-5_searchlr2e-1_all_policies_gpu.sh", all_policy_gpu_wrapper("3e-5", "2e-1"))
    write(job_dir / "lr1e-4_searchlr2e-1_all_policies_gpu.sh", all_policy_gpu_wrapper("1e-4", "2e-1"))
    write(job_dir / "lr6e-5_searchlr2e-1_all_policies_gpu.sh", all_policy_gpu_wrapper("6e-5", "2e-1"))
    write(job_dir / "uvqlm_lr1e-5_searchlr2e-1_gpu.sh", uvqlm_gpu_wrapper("1e-5", "2e-1"))
    write(job_dir / "ran_lr3e-5_searchlr2e-1_gpu.sh", random_additive_gpu_wrapper("3e-5", "2e-1"))

    write(ORACLE_REPRO_ROOT / "UVQLM/tedlium_grid.yaml", uvqlm_grid_config())
    write(ORACLE_REPRO_ROOT / "RAN/tedlium_grid.yaml", random_additive_grid_config())
    write(
        ORACLE_REPRO_ROOT / "RAN/README.md",
        """# Random Additive Noise Oracle

`RAN` evaluates `AdditivePolicy` with `use_random: true`, so oracle search
chooses among randomly sampled additive spectrogram perturbations rather than
masking proposals.
""",
    )

    write(POLICY_REPRO_ROOT / "UFMR_segmented/tedlium_grid.yaml", ufmr_grid_config())

    write(
        ORACLE_REPRO_ROOT / "launch_single_sequential.sh",
        """#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."
mkdir -p exp/results/repro/oracle/logs

screen -L -Logfile "exp/results/repro/oracle/logs/single_sequential.log" \\
  -dmS "l2a_oracle_single_sequential" \\
  bash -lc './exp/results/repro/oracle/jobs/rmm_historical_then_recent.sh && ./exp/results/repro/oracle/jobs/rfm_historical_then_recent.sh'

screen -ls | grep 'l2a_'
""",
    )

    write(
        POLICY_REPRO_ROOT / "UFMR_segmented/run_lr_sweep_cpu.sh",
        """#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../../.."
export CUDA_VISIBLE_DEVICES=""
export MPLCONFIGDIR="${MPLCONFIGDIR:-/exp/exp4/acp21rjf/.scratch/matplotlib-cache}"
export L2A_TEDLIUM3_LEGACY_DIR="${L2A_TEDLIUM3_LEGACY_DIR:-/store/store4/data/TEDLIUM_release-3/legacy}"
mkdir -p "${MPLCONFIGDIR}"

: > "exp/results/repro/policy/UFMR_segmented/tedlium_lr1e-6.txt"
: > "exp/results/repro/policy/UFMR_segmented/tedlium_lr8e-6.txt"

PYTHONDONTWRITEBYTECODE=1 python3 exp/run_config_grid.py \\
  --grid-config "exp/results/repro/policy/UFMR_segmented/tedlium_grid.yaml" \\
  --entrypoint "PYTHONDONTWRITEBYTECODE=1 python3 exp/oracle_eval.py --config {config}" \\
  --workdir . \\
  --stop-on-error
""",
    )


if __name__ == "__main__":
    main()
