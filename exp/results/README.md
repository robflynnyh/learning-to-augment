# Learn-to-Augment Results

This directory contains generated result files, plotting inputs, and reproduction scripts for the
learnt-augmentation experiments.

## Structure

Historical result directories remain at the top level, for example `RMM/`, `RFM/`, `UFMR/`, and
`UVQLM/`. New reproduction-only artifacts are grouped under `repro/`:

```text
repro/
  oracle/
    RMM/
      configs/
      run_cpu.sh
      tedlium_lr*_searchlr*.txt
    RFM/
      configs/
      run_cpu.sh
      tedlium_lr*_searchlr*.txt
    jobs/
      rmm_lr*_searchlr*.sh
      rfm_lr*_searchlr*.sh
    logs/
    launch_screens.sh
    run_cpu.sh
  policy/
    UFMR_segmented/
      configs/
      run_lr_sweep_cpu.sh
      tedlium_lr*.txt
```

`repro/oracle/` is for oracle mask-selection experiments where candidates are scored by a local
single-step reward before the actual adaptation step is applied. `repro/policy/` is for direct policy
rollouts without oracle candidate search, currently the UFMR segmented policy baseline.

## Oracle Reproduction Notes

The historical file `RMM/oracle/tedlium.txt` should not be interpreted as a clean random frequency
masking (RFM) oracle. Git history shows that it was added in the same commit as a hardcoded mixed-mask
candidate generator in `l2augment/rollout/cpu_multistep_oracle.py`. That generator sampled one of
three mask types for each candidate: time masking, frequency masking, or time+frequency masking.

The old hardcoded behaviour has now been moved into the normal policy abstraction. Reproduction configs
use:

```yaml
policy:
  class: MixedMaskingRanker
  config:
    time_masks_min: 3
    time_masks_max: 16
    freq_masks_min: 5
    freq_masks_max: 7
    freq_mask_param_min: 34
    freq_mask_param_max: 34
```

This matches the verified historical mixed-mask distribution: a random number of time masks in
`[3, 16]`, random `min_p = random() / 2`, random number of frequency masks in `[5, 7]`, fixed
frequency-mask width `34`, and a uniform random choice between time-only, frequency-only, and
time+frequency masking.

## Configured Matrix

The current oracle reproduction matrix is configured but has not been launched. It contains eight
top-level runs:

- policies: `RMM` (`MixedMaskingRanker`) and `RFM` (`FrequencyMaskingRanker` with random frequency masks)
- adaptation learning rates: `1e-6` and `8e-6`
- oracle-search learning rates: `4e-2` and `9e-2`

Each top-level run evaluates candidate counts:

```text
1, 2, 3, 4, 5, 10, 20, 50
```

Generated configs live under:

- `repro/oracle/RMM/configs/`
- `repro/oracle/RFM/configs/`

Per-combination job scripts live under:

- `repro/oracle/jobs/`

The aggregate launch script is:

```bash
./exp/results/repro/oracle/launch_screens.sh
```

It starts one detached `screen` session per top-level run and writes logs to
`exp/results/repro/oracle/logs/`. This script has not been run yet.

For sequential CPU execution, use:

```bash
./exp/results/repro/oracle/run_cpu.sh
```

## Paths And Environment

The TEDLIUM loader reads `L2A_TEDLIUM3_LEGACY_DIR`. The generated scripts default this to:

```bash
/store/store4/data/TEDLIUM_release-3/legacy
```

The scripts also set `MPLCONFIGDIR` to:

```bash
/exp/exp4/acp21rjf/.scratch/matplotlib-cache
```

This avoids using `/tmp` for matplotlib cache files.

## Verified 2026-05-08

- The old RMM oracle result came from a mixed-mask search path, not clean RFM.
- `MixedMaskingRanker` is registered as a policy and can represent the old mixed-mask behaviour.
- A one-recording smoke test completed with `MixedMaskingRanker` on TEDLIUM from `/store/store4`.
- The full oracle matrix is configured, but not launched.
