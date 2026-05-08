#!/usr/bin/env python3
"""Generate reproducible configs for RFM and RMM oracle experiments."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "exp/results"
REPRO_ROOT = RESULTS_ROOT / "repro"
ORACLE_REPRO_ROOT = REPRO_ROOT / "oracle"
POLICY_REPRO_ROOT = REPRO_ROOT / "policy"
ASR_MODEL = "/store/store5/data/acp21rjf_checkpoints/l2augment/asr/step_105360.pt"
UFMR_MODEL = "/store/store5/data/acp21rjf_checkpoints/l2augment/ufmr/mseloss/model.pt"
PYTHON = "PYTHONDONTWRITEBYTECODE=1 python3"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if path.suffix == ".sh":
        path.chmod(0o755)


def clean_yaml_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for yaml_path in path.glob("*.yaml"):
        yaml_path.unlink()


def fmt_lr(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")


def rmm_config(repeats: int, lr: float, single_step_lr: float) -> str:
    lr_name = fmt_lr(lr)
    search_lr_name = fmt_lr(single_step_lr)
    return f"""checkpointing:
  asr_model: {ASR_MODEL}

training:
  device: cpu
  random_seed: 1234
  batch_size: 84
  epochs: 100

evaluation:
  rollout_setting: search
  search_repeats: {repeats}
  split: test
  use_cer: false
  epochs: 1
  augmentation_config:
    repeats: 1
    use_random: true
  optim_args:
    lr: {lr:.1e}
    single_step_lr: {single_step_lr:.1e}
  save_path: {ORACLE_REPRO_ROOT}/RMM/tedlium_lr{lr_name}_searchlr{search_lr_name}.txt

policy:
  lr: 0.0001
  class: MixedMaskingRanker
  config:
    time_masks_min: 3
    time_masks_max: 16
    freq_masks_min: 5
    freq_masks_max: 7
    freq_mask_param_min: 34
    freq_mask_param_max: 34
"""


def rfm_config(repeats: int, lr: float, single_step_lr: float) -> str:
    lr_name = fmt_lr(lr)
    search_lr_name = fmt_lr(single_step_lr)
    return f"""checkpointing:
  asr_model: {ASR_MODEL}

training:
  device: cpu
  random_seed: 1234
  batch_size: 84
  epochs: 100

evaluation:
  rollout_setting: search
  search_repeats: {repeats}
  split: test
  use_cer: false
  epochs: 1
  augmentation_config:
    repeats: 1
    use_random: true
  optim_args:
    lr: {lr:.1e}
    single_step_lr: {single_step_lr:.1e}
  save_path: {ORACLE_REPRO_ROOT}/RFM/tedlium_lr{lr_name}_searchlr{search_lr_name}.txt

policy:
  lr: 0.0001
  class: FrequencyMaskingRanker
"""


def ufmr_policy_config(lr: float) -> str:
    lr_name = fmt_lr(lr)
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
    lr: {lr:.1e}
  save_path: {POLICY_REPRO_ROOT}/UFMR_segmented/tedlium_lr{lr_name}.txt

policy:
  lr: 0.0001
  class: UnconditionalFrequencyMaskingRanker
"""


def oracle_stream_script(method: str, repeats_arg: str) -> str:
    result_dir = method
    return f"""#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../../.."
export CUDA_VISIBLE_DEVICES=""
export MPLCONFIGDIR="${{MPLCONFIGDIR:-/exp/exp4/acp21rjf/.scratch/matplotlib-cache}}"
export L2A_TEDLIUM3_LEGACY_DIR="${{L2A_TEDLIUM3_LEGACY_DIR:-/store/store4/data/TEDLIUM_release-3/legacy}}"
mkdir -p "${{MPLCONFIGDIR}}" "exp/results/repro/oracle/{result_dir}/logs"

run_combo() {{
  local lr="$1"
  local search_lr="$2"
  local result="exp/results/repro/oracle/{result_dir}/tedlium_lr${{lr}}_searchlr${{search_lr}}.txt"
  : > "${{result}}"
  for repeats in {repeats_arg}; do
    PYTHONDONTWRITEBYTECODE=1 python3 exp/oracle_eval.py \\
      --config "exp/results/repro/oracle/{result_dir}/configs/tedlium_lr${{lr}}_searchlr${{search_lr}}_repeats${{repeats}}.yaml"
  done
}}

# Historical matched setup, followed by the newer/default setup.
run_combo 1e-6 4e-2
run_combo 8e-6 9e-2
"""


def main() -> None:
    search_repeats = [1, 2, 3, 4, 5, 10, 20, 50]
    search_repeats_arg = " ".join(str(repeats) for repeats in search_repeats)
    adaptation_lrs = [1e-6, 8e-6]
    single_step_lrs = [4e-2, 9e-2]

    rmm_config_dir = ORACLE_REPRO_ROOT / "RMM/configs"
    rfm_config_dir = ORACLE_REPRO_ROOT / "RFM/configs"
    ufmr_config_dir = POLICY_REPRO_ROOT / "UFMR_segmented/configs"
    job_dir = ORACLE_REPRO_ROOT / "jobs"
    clean_yaml_dir(rmm_config_dir)
    clean_yaml_dir(rfm_config_dir)
    clean_yaml_dir(ufmr_config_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    for script_path in job_dir.glob("*.sh"):
        script_path.unlink()

    for lr in adaptation_lrs:
        for single_step_lr in single_step_lrs:
            for repeats in search_repeats:
                write(
                    rmm_config_dir / f"tedlium_lr{fmt_lr(lr)}_searchlr{fmt_lr(single_step_lr)}_repeats{repeats}.yaml",
                    rmm_config(repeats, lr, single_step_lr),
                )
                write(
                    rfm_config_dir / f"tedlium_lr{fmt_lr(lr)}_searchlr{fmt_lr(single_step_lr)}_repeats{repeats}.yaml",
                    rfm_config(repeats, lr, single_step_lr),
                )

    for method in ["RMM", "RFM"]:
        write(
            job_dir / f"{method.lower()}_historical_then_recent.sh",
            oracle_stream_script(method, search_repeats_arg),
        )

    for lr in adaptation_lrs:
        write(ufmr_config_dir / f"tedlium_lr{fmt_lr(lr)}.yaml", ufmr_policy_config(lr))

    write(
        ORACLE_REPRO_ROOT / "launch_two_streams.sh",
        """#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."
mkdir -p exp/results/repro/oracle/logs

for method in rmm rfm; do
  job="exp/results/repro/oracle/jobs/${method}_historical_then_recent.sh"
  screen -L -Logfile "exp/results/repro/oracle/logs/${method}_historical_then_recent.log" \\
    -dmS "l2a_${method}_historical_then_recent" bash "${job}"
done

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

for lr in 1e-6 8e-6; do
  result="exp/results/repro/policy/UFMR_segmented/tedlium_lr${lr}.txt"
  : > "${result}"
  PYTHONDONTWRITEBYTECODE=1 python3 exp/oracle_eval.py \\
    --config "exp/results/repro/policy/UFMR_segmented/configs/tedlium_lr${lr}.yaml"
done
""",
    )


if __name__ == "__main__":
    main()
