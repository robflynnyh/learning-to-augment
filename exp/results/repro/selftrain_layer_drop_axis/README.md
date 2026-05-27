# ROB-144 Layer-Drop Figure Regeneration

This directory regenerates the `fig:selftrain:layer-drop` handoff artifact with
a wider y-axis scale. The source data is the existing layer-drop self-training
summary from the dynamic ASR evaluation result tree:

```text
/exp/exp4/acp21rjf/dynamic-asr-eval/lcasr/results/ctc_self_training_extra_ablation_sweeps/summary.csv
```

The committed CSV copies only the `layer_drop_lr_sweep`, `lr=9e-5` rows used by
the figure. The regenerated plot keeps the existing bar order, labels, WER
values, and unadapted-WER text annotation, but uses y-axes set to plus/minus
20% relative to the average plotted WER in each dataset panel:

- Earnings22: `12.8%` to `19.2%`
- TED-LIUM: `4.7%` to `7.1%`

These wider axes reduce the visual amplification of sub-percent WER differences
without flattening the comparison as much as a zero-based axis.

Regeneration command:

```bash
MPLCONFIGDIR=/exp/exp4/acp21rjf/.scratch/matplotlib-cache \
python3 exp/results/repro/selftrain_layer_drop_axis/plot_layer_drop_larger_axis.py \
  --source-summary /exp/exp4/acp21rjf/dynamic-asr-eval/lcasr/results/ctc_self_training_extra_ablation_sweeps/summary.csv
```

Thesis handoff artifact:

```text
exp/results/repro/selftrain_layer_drop_axis/layer_drop_lr_sweep_ablation_bars_larger_axis.pdf
exp/results/repro/selftrain_layer_drop_axis/layer_drop_lr_sweep_ablation_bars_larger_axis.png
```
