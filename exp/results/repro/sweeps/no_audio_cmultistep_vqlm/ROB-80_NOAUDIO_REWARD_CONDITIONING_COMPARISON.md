# ROB-80 TED-LIUM Dev No-Audio CMultiStepVQLM Reward Conditioning Comparison

Compares the committed fixed conditioning reward 1.0 baseline against the follow-up randomized conditioning reward sampled uniformly from [0.5, 1.0].

Completed cells: 12/12

| Method | Conditioning | Epochs | LR | Original WER | Updated WER | Abs Delta | Rel Delta % | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CMultiStepVQLM | fixed_1.0 | 1 | `5e-6` | 0.100088 | 0.090140 | -0.009948 | -9.94 | complete |
| CMultiStepVQLM | fixed_1.0 | 1 | `1e-5` | 0.100088 | 0.090030 | -0.010059 | -10.05 | complete |
| CMultiStepVQLM | fixed_1.0 | 1 | `2e-5` | 0.100088 | 0.089256 | -0.010832 | -10.82 | complete |
| CMultiStepVQLM | fixed_1.0 | 5 | `5e-6` | 0.100088 | 0.087488 | -0.012601 | -12.59 | complete |
| CMultiStepVQLM | fixed_1.0 | 5 | `1e-5` | 0.100088 | 0.087322 | -0.012767 | -12.76 | complete |
| CMultiStepVQLM | fixed_1.0 | 5 | `2e-5` | 0.100088 | 0.090251 | -0.009838 | -9.83 | complete |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 1 | `5e-6` | 0.100088 | 0.089975 | -0.010114 | -10.10 | complete |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 1 | `1e-5` | 0.100088 | 0.089809 | -0.010280 | -10.27 | complete |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 1 | `2e-5` | 0.100088 | 0.087930 | -0.012159 | -12.15 | complete |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 5 | `5e-6` | 0.100088 | 0.085774 | -0.014314 | -14.30 | complete |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 5 | `1e-5` | 0.100088 | 0.086603 | -0.013485 | -13.47 | complete |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 5 | `2e-5` | 0.100088 | 0.088096 | -0.011993 | -11.98 | complete |
