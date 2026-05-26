# ROB-144 Outcome

Regenerated `fig:selftrain:layer-drop` with the same source data and figure
content, but with wider WER axes set to plus/minus 20% relative to the average
plotted WER in each dataset panel.

Artifacts:

- `layer_drop_lr_sweep_ablation_bars_larger_axis.pdf`: thesis handoff figure.
- `layer_drop_lr_sweep_ablation_bars_larger_axis.png`: thesis handoff PNG and quick visual preview.
- `layer_drop_lr_sweep_9e-5_source.csv`: copied source rows used by the figure.
- `plot_layer_drop_larger_axis.py`: reproducible plotting script.

The source data came from:

```text
/exp/exp4/acp21rjf/dynamic-asr-eval/lcasr/results/ctc_self_training_extra_ablation_sweeps/summary.csv
```

The figure uses:

- Earnings22 y-axis: `12.8%` to `19.2%`.
- TED-LIUM y-axis: `4.7%` to `7.1%`.

Validation:

```bash
MPLCONFIGDIR=/exp/exp4/acp21rjf/.scratch/matplotlib-cache \
python3 exp/results/repro/selftrain_layer_drop_axis/plot_layer_drop_larger_axis.py \
  --source-summary /exp/exp4/acp21rjf/dynamic-asr-eval/lcasr/results/ctc_self_training_extra_ablation_sweeps/summary.csv
python3 -m py_compile exp/results/repro/selftrain_layer_drop_axis/plot_layer_drop_larger_axis.py
git diff --check
```
