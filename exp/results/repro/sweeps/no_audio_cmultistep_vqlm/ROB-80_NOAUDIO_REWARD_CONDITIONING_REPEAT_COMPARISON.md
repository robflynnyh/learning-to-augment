# ROB-80 TED-LIUM Dev No-Audio CMultiStepVQLM Reward Conditioning Repeat Comparison

Compares fixed reward 1.0 against uniform [0.5, 1.0] random reward conditioning across repeat 1 and repeat 2. Repeat 2 uses rollout seed 123457.

Completed cells: 12/24

## Aggregate

| Method | Conditioning | Epochs | LR | N | Mean Original WER | Mean Updated WER | Updated WER Std | Mean Abs Delta | Mean Rel Delta % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CMultiStepVQLM | fixed_1.0 | 1 | `5e-6` | 1 | 0.100088 | 0.090140 | 0.000000 | -0.009948 | -9.94 |
| CMultiStepVQLM | fixed_1.0 | 1 | `1e-5` | 1 | 0.100088 | 0.090030 | 0.000000 | -0.010058 | -10.05 |
| CMultiStepVQLM | fixed_1.0 | 1 | `2e-5` | 1 | 0.100088 | 0.089256 | 0.000000 | -0.010832 | -10.82 |
| CMultiStepVQLM | fixed_1.0 | 5 | `5e-6` | 1 | 0.100088 | 0.087488 | 0.000000 | -0.012600 | -12.59 |
| CMultiStepVQLM | fixed_1.0 | 5 | `1e-5` | 1 | 0.100088 | 0.087322 | 0.000000 | -0.012766 | -12.75 |
| CMultiStepVQLM | fixed_1.0 | 5 | `2e-5` | 1 | 0.100088 | 0.090251 | 0.000000 | -0.009837 | -9.83 |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 1 | `5e-6` | 1 | 0.100088 | 0.089975 | 0.000000 | -0.010113 | -10.10 |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 1 | `1e-5` | 1 | 0.100088 | 0.089809 | 0.000000 | -0.010279 | -10.27 |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 1 | `2e-5` | 1 | 0.100088 | 0.087930 | 0.000000 | -0.012158 | -12.15 |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 5 | `5e-6` | 1 | 0.100088 | 0.085774 | 0.000000 | -0.014314 | -14.30 |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 5 | `1e-5` | 1 | 0.100088 | 0.086603 | 0.000000 | -0.013485 | -13.47 |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 5 | `2e-5` | 1 | 0.100088 | 0.088096 | 0.000000 | -0.011992 | -11.98 |

## Per Repeat

| Method | Conditioning | Repeat | Seed | Epochs | LR | Original WER | Updated WER | Abs Delta | Rel Delta % | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CMultiStepVQLM | fixed_1.0 | 1 | 123456 | 1 | `5e-6` | 0.100088 | 0.090140 | -0.009948 | -9.94 | complete |
| CMultiStepVQLM | fixed_1.0 | 1 | 123456 | 1 | `1e-5` | 0.100088 | 0.090030 | -0.010059 | -10.05 | complete |
| CMultiStepVQLM | fixed_1.0 | 1 | 123456 | 1 | `2e-5` | 0.100088 | 0.089256 | -0.010832 | -10.82 | complete |
| CMultiStepVQLM | fixed_1.0 | 1 | 123456 | 5 | `5e-6` | 0.100088 | 0.087488 | -0.012601 | -12.59 | complete |
| CMultiStepVQLM | fixed_1.0 | 1 | 123456 | 5 | `1e-5` | 0.100088 | 0.087322 | -0.012767 | -12.76 | complete |
| CMultiStepVQLM | fixed_1.0 | 1 | 123456 | 5 | `2e-5` | 0.100088 | 0.090251 | -0.009838 | -9.83 | complete |
| CMultiStepVQLM | fixed_1.0 | 2 | 123457 | 1 | `5e-6` |  |  |  |  | missing |
| CMultiStepVQLM | fixed_1.0 | 2 | 123457 | 1 | `1e-5` |  |  |  |  | missing |
| CMultiStepVQLM | fixed_1.0 | 2 | 123457 | 1 | `2e-5` |  |  |  |  | missing |
| CMultiStepVQLM | fixed_1.0 | 2 | 123457 | 5 | `5e-6` |  |  |  |  | missing |
| CMultiStepVQLM | fixed_1.0 | 2 | 123457 | 5 | `1e-5` |  |  |  |  | missing |
| CMultiStepVQLM | fixed_1.0 | 2 | 123457 | 5 | `2e-5` |  |  |  |  | missing |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 1 | 123456 | 1 | `5e-6` | 0.100088 | 0.089975 | -0.010114 | -10.10 | complete |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 1 | 123456 | 1 | `1e-5` | 0.100088 | 0.089809 | -0.010280 | -10.27 | complete |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 1 | 123456 | 1 | `2e-5` | 0.100088 | 0.087930 | -0.012159 | -12.15 | complete |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 1 | 123456 | 5 | `5e-6` | 0.100088 | 0.085774 | -0.014314 | -14.30 | complete |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 1 | 123456 | 5 | `1e-5` | 0.100088 | 0.086603 | -0.013485 | -13.47 | complete |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 1 | 123456 | 5 | `2e-5` | 0.100088 | 0.088096 | -0.011993 | -11.98 | complete |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 2 | 123457 | 1 | `5e-6` |  |  |  |  | missing |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 2 | 123457 | 1 | `1e-5` |  |  |  |  | missing |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 2 | 123457 | 1 | `2e-5` |  |  |  |  | missing |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 2 | 123457 | 5 | `5e-6` |  |  |  |  | missing |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 2 | 123457 | 5 | `1e-5` |  |  |  |  | missing |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 2 | 123457 | 5 | `2e-5` |  |  |  |  | missing |
