# RAC-MLM Average Mask Visualization: TED-LIUM Test

This directory contains the ROB-196 visualization for the ROB-132
audio+reward-conditioned mask LM on TED-LIUM test audio.

The figure averages sampled RAC-MLM masks for five distinct TED-LIUM test
recordings. For each audio segment, the script streams `1000` sampled masks at
conditioning reward `1.0` and stores only the average keep mask.

The masks are not applied to audio in this artifact. The PDF plots
`1.0 - average_keep_mask`, so brighter regions are masked more often.

## Artifacts

- `rac_mlm_average_masks_tedlium_test_5x1000.pdf`: requested PDF with samples
  1-5.
- `rac_mlm_average_masks_tedlium_test_5x1000.png`: PNG rendering of the same
  figure.
- `rac_mlm_average_masks_tedlium_test_5x1000.npz`: compressed average
  keep-mask arrays.
- `metadata.json`: source test paths, checkpoint/config paths, generation
  settings, per-sample mask statistics, and example token sequences.

## Generation Command

```bash
MPLCONFIGDIR=/exp/exp4/acp21rjf/.scratch/matplotlib \
PYTHONPATH=$PWD \
/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- \
/store/store4/software/bin/anaconda3/envs/dasr/bin/python \
  exp/results/scripts/plot_rac_mlm_average_masks.py --device cuda
```
