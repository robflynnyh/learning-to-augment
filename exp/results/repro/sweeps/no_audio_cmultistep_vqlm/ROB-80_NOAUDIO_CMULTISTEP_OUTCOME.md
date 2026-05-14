# ROB-80 TED-LIUM Dev No-Audio CMultiStepVQLM LR Sweep

No-audio CMultiStepVQLM uses `ConditionalMultiStepMaskGenerator` with `condition_on_audio: false`, so generation is conditioned on reward/signal inputs and recurrent mask history, not raw audio.

Completed cells: 6/6

| Method | Epochs | LR | Original WER | Updated WER | Abs Delta | Rel Delta % | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CMultiStepVQLM | 1 | `5e-6` | 0.100088 | 0.090140 | -0.009948 | -9.94 | complete |
| CMultiStepVQLM | 1 | `1e-5` | 0.100088 | 0.090030 | -0.010059 | -10.05 | complete |
| CMultiStepVQLM | 1 | `2e-5` | 0.100088 | 0.089256 | -0.010832 | -10.82 | complete |
| CMultiStepVQLM | 5 | `5e-6` | 0.100088 | 0.087488 | -0.012601 | -12.59 | complete |
| CMultiStepVQLM | 5 | `1e-5` | 0.100088 | 0.087322 | -0.012767 | -12.76 | complete |
| CMultiStepVQLM | 5 | `2e-5` | 0.100088 | 0.090251 | -0.009838 | -9.83 | complete |
