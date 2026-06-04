# RAC-MLM Average Mask Visualization

This directory contains the ROB-196 visualization for the ROB-132
audio+reward-conditioned mask LM.

The figure averages sampled RAC-MLM masks for five distinct TED-LIUM train
recordings. For each audio segment, the script streams `1000` sampled masks at
conditioning reward `1.0` and stores only the average keep mask.

## Artifacts

- `rac_mlm_average_masks_5x1000.pdf`: requested PDF with samples 1-5.
- `rac_mlm_average_masks_5x1000.png`: PNG rendering of the same figure.
- `rac_mlm_average_masks_5x1000.npz`: compressed average keep-mask arrays.
- `metadata.json`: source rollout paths, checkpoint/config paths, generation
  settings, per-sample mask statistics, and example token sequences.

The model output is a keep mask where `1.0` keeps a mel/time bin. The plotted
figure shows `1.0 - average_keep_mask`, so brighter regions are masked more
often.

## Generation Command

```bash
/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- \
  env PYTHONPATH=$PWD \
/store/store4/software/bin/anaconda3/envs/dasr/bin/python \
  exp/results/scripts/plot_rac_mlm_average_masks.py --device cuda
```
