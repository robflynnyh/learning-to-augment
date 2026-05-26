# ROB-144 Layer-Drop Figure Regeneration

This directory regenerates the `fig:selftrain:layer-drop` handoff artifact with
a wider y-axis scale. The source data is the existing layer-drop self-training
summary from the dynamic ASR evaluation result tree:

```text
/exp/exp4/acp21rjf/dynamic-asr-eval/lcasr/results/ctc_self_training_extra_ablation_sweeps/summary.csv
```

The committed CSV copies only the `layer_drop_lr_sweep`, `lr=9e-5` rows used by
the figure. The regenerated plot keeps the existing bar order, labels, WER
values, and unadapted-WER text annotation, but uses zero-based WER axes:

- Earnings22: `0%` to `20%`
- TED-LIUM: `0%` to `7%`

These wider axes avoid visually overstating sub-percent WER differences while
preserving the figure intent.

Regeneration command:

```bash
MPLCONFIGDIR=/exp/exp4/acp21rjf/.scratch/matplotlib-cache \
python3 exp/results/repro/selftrain_layer_drop_axis/plot_layer_drop_larger_axis.py \
  --source-summary /exp/exp4/acp21rjf/dynamic-asr-eval/lcasr/results/ctc_self_training_extra_ablation_sweeps/summary.csv
```

Thesis handoff artifact:

```text
exp/results/repro/selftrain_layer_drop_axis/layer_drop_lr_sweep_ablation_bars_larger_axis.pdf
```
