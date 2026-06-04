# RAC-MLM Mask Visualization: TED-LIUM Test Samples 3-5

This directory contains the ROB-196 visualization for the ROB-132
audio+reward-conditioned mask LM on TED-LIUM test audio.

The figures use TED-LIUM test source samples 3-5, removing the earlier samples
1-2. For each audio segment, the script samples RAC-MLM masks at conditioning
rewards `1.0` and `0.0` and plots the two rewards as columns.

The masks are not applied to audio in this artifact. The PDFs plot
`1.0 - keep_mask`, so brighter regions are masked more often.

## Artifacts

- `rac_mlm_masks_tedlium_test_samples3to5_reward1p0_reward0p0_average1000.pdf`:
  two-column PDF where each panel is the average of `1000` sampled masks.
- `rac_mlm_masks_tedlium_test_samples3to5_reward1p0_reward0p0_single.pdf`:
  matching two-column PDF using one generated mask for the same audio/reward
  panels.
- `rac_mlm_masks_tedlium_test_samples3to5_reward1p0_reward0p0_average1000.png`:
  PNG rendering of the averaged-mask figure.
- `rac_mlm_masks_tedlium_test_samples3to5_reward1p0_reward0p0_single.png`:
  PNG rendering of the single-generation figure.
- `rac_mlm_masks_tedlium_test_samples3to5_reward1p0_reward0p0.npz`:
  compressed average and single keep-mask arrays.
- `metadata.json`: source test paths, checkpoint/config paths, generation
  settings, per-panel mask statistics, and example token sequences.

## Generation Command

```bash
MPLCONFIGDIR=/exp/exp4/acp21rjf/.scratch/matplotlib \
PYTHONPATH=$PWD \
/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- \
/store/store4/software/bin/anaconda3/envs/dasr/bin/python \
  exp/results/scripts/plot_rac_mlm_average_masks.py --device cuda
```
