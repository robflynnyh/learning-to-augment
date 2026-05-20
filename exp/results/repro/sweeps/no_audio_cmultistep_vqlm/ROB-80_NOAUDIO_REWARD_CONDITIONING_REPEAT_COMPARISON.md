# ROB-80 TED-LIUM Dev No-Audio CMultiStepVQLM Reward Conditioning Repeat Comparison

Compares fixed reward 1.0 against uniform [0.5, 1.0] random reward conditioning across repeat 1 and repeat 2. Repeat 2 uses rollout seed 123457.

Completed cells: 24/24

## Aggregate

| Method | Conditioning | Epochs | LR | N | Mean Original WER | Mean Updated WER | Updated WER Std | Mean Abs Delta | Mean Rel Delta % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CMultiStepVQLM | fixed_1.0 | 1 | `5e-6` | 2 | 0.100088 | 0.089588 | 0.000781 | -0.010500 | -10.49 |
| CMultiStepVQLM | fixed_1.0 | 1 | `1e-5` | 2 | 0.100088 | 0.090002 | 0.000039 | -0.010085 | -10.08 |
| CMultiStepVQLM | fixed_1.0 | 1 | `2e-5` | 2 | 0.100088 | 0.089394 | 0.000195 | -0.010694 | -10.68 |
| CMultiStepVQLM | fixed_1.0 | 5 | `5e-6` | 2 | 0.100088 | 0.087266 | 0.000313 | -0.012821 | -12.81 |
| CMultiStepVQLM | fixed_1.0 | 5 | `1e-5` | 2 | 0.100088 | 0.087405 | 0.000117 | -0.012683 | -12.67 |
| CMultiStepVQLM | fixed_1.0 | 5 | `2e-5` | 2 | 0.100088 | 0.089284 | 0.001368 | -0.010804 | -10.79 |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 1 | `5e-6` | 2 | 0.100088 | 0.089836 | 0.000196 | -0.010251 | -10.24 |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 1 | `1e-5` | 2 | 0.100088 | 0.089643 | 0.000235 | -0.010445 | -10.44 |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 1 | `2e-5` | 2 | 0.100088 | 0.089753 | 0.002579 | -0.010334 | -10.33 |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 5 | `5e-6` | 2 | 0.100088 | 0.087238 | 0.002071 | -0.012849 | -12.84 |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 5 | `1e-5` | 2 | 0.100088 | 0.087432 | 0.001172 | -0.012656 | -12.64 |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 5 | `2e-5` | 2 | 0.100088 | 0.088731 | 0.000899 | -0.011357 | -11.35 |

## Per Repeat

| Method | Conditioning | Repeat | Seed | Epochs | LR | Original WER | Updated WER | Abs Delta | Rel Delta % | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CMultiStepVQLM | fixed_1.0 | 1 | 123456 | 1 | `5e-6` | 0.100088 | 0.090140 | -0.009948 | -9.94 | complete |
| CMultiStepVQLM | fixed_1.0 | 1 | 123456 | 1 | `1e-5` | 0.100088 | 0.090030 | -0.010059 | -10.05 | complete |
| CMultiStepVQLM | fixed_1.0 | 1 | 123456 | 1 | `2e-5` | 0.100088 | 0.089256 | -0.010832 | -10.82 | complete |
| CMultiStepVQLM | fixed_1.0 | 1 | 123456 | 5 | `5e-6` | 0.100088 | 0.087488 | -0.012601 | -12.59 | complete |
| CMultiStepVQLM | fixed_1.0 | 1 | 123456 | 5 | `1e-5` | 0.100088 | 0.087322 | -0.012767 | -12.76 | complete |
| CMultiStepVQLM | fixed_1.0 | 1 | 123456 | 5 | `2e-5` | 0.100088 | 0.090251 | -0.009838 | -9.83 | complete |
| CMultiStepVQLM | fixed_1.0 | 2 | 123457 | 1 | `5e-6` | 0.100088 | 0.089035 | -0.011053 | -11.04 | complete |
| CMultiStepVQLM | fixed_1.0 | 2 | 123457 | 1 | `1e-5` | 0.100088 | 0.089975 | -0.010114 | -10.10 | complete |
| CMultiStepVQLM | fixed_1.0 | 2 | 123457 | 1 | `2e-5` | 0.100088 | 0.089532 | -0.010556 | -10.55 | complete |
| CMultiStepVQLM | fixed_1.0 | 2 | 123457 | 5 | `5e-6` | 0.100088 | 0.087045 | -0.013043 | -13.03 | complete |
| CMultiStepVQLM | fixed_1.0 | 2 | 123457 | 5 | `1e-5` | 0.100088 | 0.087488 | -0.012601 | -12.59 | complete |
| CMultiStepVQLM | fixed_1.0 | 2 | 123457 | 5 | `2e-5` | 0.100088 | 0.088317 | -0.011772 | -11.76 | complete |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 1 | 123456 | 1 | `5e-6` | 0.100088 | 0.089975 | -0.010114 | -10.10 | complete |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 1 | 123456 | 1 | `1e-5` | 0.100088 | 0.089809 | -0.010280 | -10.27 | complete |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 1 | 123456 | 1 | `2e-5` | 0.100088 | 0.087930 | -0.012159 | -12.15 | complete |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 1 | 123456 | 5 | `5e-6` | 0.100088 | 0.085774 | -0.014314 | -14.30 | complete |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 1 | 123456 | 5 | `1e-5` | 0.100088 | 0.086603 | -0.013485 | -13.47 | complete |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 1 | 123456 | 5 | `2e-5` | 0.100088 | 0.088096 | -0.011993 | -11.98 | complete |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 2 | 123457 | 1 | `5e-6` | 0.100088 | 0.089698 | -0.010390 | -10.38 | complete |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 2 | 123457 | 1 | `1e-5` | 0.100088 | 0.089477 | -0.010611 | -10.60 | complete |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 2 | 123457 | 1 | `2e-5` | 0.100088 | 0.091577 | -0.008511 | -8.50 | complete |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 2 | 123457 | 5 | `5e-6` | 0.100088 | 0.088703 | -0.011385 | -11.37 | complete |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 2 | 123457 | 5 | `1e-5` | 0.100088 | 0.088261 | -0.011827 | -11.82 | complete |
| CMultiStepVQLMRandomReward | uniform_0.5_1.0 | 2 | 123457 | 5 | `2e-5` | 0.100088 | 0.089367 | -0.010722 | -10.71 | complete |
