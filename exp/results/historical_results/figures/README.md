# Figure Reproduction

Run commands from the repository root.

## Learnt Augmentation WERR Plots

Single epoch:

```bash
python3 exp/results/scripts/plot_single_epoch_werr.py
```

Five epochs:

```bash
python3 exp/results/scripts/plot_single_epoch_werr.py --epoch 5
```

Outputs:

- `single_epoch_werr.pdf` and `single_epoch_werr.csv`
- `epoch5_werr.pdf` and `epoch5_werr.csv`

## Learnt Self-Training WER Table

```bash
python3 exp/results/scripts/generate_wer_table.py
```

Output:

- `learnt_self_training_wer_table.txt`

## Oracle WER Plot

Run the segmented UFMR comparison line first, if `exp/results/historical_results/UFMR_segmented/tedlium.txt` is missing:

```bash
L2A_TEDLIUM3_LEGACY_DIR=/store/store4/data/TEDLIUM_release-3/legacy \
MPLCONFIGDIR=/tmp/matplotlib-cache \
python3 exp/oracle_eval.py --config exp/results/historical_results/UFMR_segmented/tedlium.yaml
```

```bash
python3 exp/results/scripts/plot_oracle_wer.py
```

Source:

- `exp/results/historical_results/RMM/oracle/tedlium.txt`
- `exp/results/historical_results/UFMR_segmented/tedlium.txt`

Outputs:

- `oracle_wer.pdf`
- `oracle_wer.csv`

## UFMR And Random Mask Distributions

All mask-distribution plots below use 100,000 selected masks. UFMR uses `repeats=15`, matching the evaluation configs. Random FM samples directly from the same SpecAugment frequency-mask family.

UFMR, `test_wer` checkpoint:

```bash
python3 exp/results/scripts/plot_ufmr_mask_distribution.py \
  --mode ufmr \
  --checkpoint /store/store5/data/acp21rjf_checkpoints/l2augment/ufmr/test_wer/model.pt \
  --repeats 15 \
  --num-samples 100000 \
  --output exp/results/historical_results/figures/UFMR_mask_n100000.pdf \
  --csv exp/results/historical_results/figures/UFMR_mask_n100000.csv
```

UFMR, `mseloss` checkpoint:

```bash
python3 exp/results/scripts/plot_ufmr_mask_distribution.py \
  --mode ufmr \
  --checkpoint /store/store5/data/acp21rjf_checkpoints/l2augment/ufmr/mseloss/model.pt \
  --repeats 15 \
  --num-samples 100000 \
  --output exp/results/historical_results/figures/UFMR_mask_mseloss_n100000.pdf \
  --csv exp/results/historical_results/figures/UFMR_mask_mseloss_n100000.csv
```

UFMR, `mseloss2e1` checkpoint:

```bash
python3 exp/results/scripts/plot_ufmr_mask_distribution.py \
  --mode ufmr \
  --checkpoint /store/store5/data/acp21rjf_checkpoints/l2augment/ufmr/mseloss2e1/model.pt \
  --repeats 15 \
  --num-samples 100000 \
  --output exp/results/historical_results/figures/UFMR_mask_mseloss2e1_n100000.pdf \
  --csv exp/results/historical_results/figures/UFMR_mask_mseloss2e1_n100000.csv
```

Random FM:

```bash
python3 exp/results/scripts/plot_ufmr_mask_distribution.py \
  --mode random \
  --num-samples 100000 \
  --output exp/results/historical_results/figures/RFM_mask_n100000.pdf \
  --csv exp/results/historical_results/figures/RFM_mask_n100000.csv
```

Combined random-vs-UFMR comparison:

```bash
python3 exp/results/scripts/plot_mask_distribution_comparison.py \
  --series UFMR exp/results/historical_results/figures/UFMR_mask_mseloss_n100000.csv \
  --series 'Random FM' exp/results/historical_results/figures/RFM_mask_n100000.csv \
  --output exp/results/historical_results/figures/mask_distribution_random_vs_mseloss.pdf
```

Output:

- `mask_distribution_random_vs_mseloss.pdf`
