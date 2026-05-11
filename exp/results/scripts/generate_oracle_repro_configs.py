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
REPRO_OUTPUT_ROOT = "/exp/exp4/acp21rjf/learning-to-augment/exp/results/repro"
ASR_MODEL = "/store/store5/data/acp21rjf_checkpoints/l2augment/asr/step_105360.pt"
UFMR_MODEL = "/store/store5/data/acp21rjf_checkpoints/l2augment/ufmr/mseloss/model.pt"
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
  --entrypoint "PYTHONDONTWRITEBYTECODE=1 python3 exp/oracle_eval.py --config {{config}}" \\
  --workdir . \\
  --stop-on-error
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
