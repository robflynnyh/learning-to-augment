# ROB-82 TED-LIUM Dev UVQLM LR Sweep

UVQLM is evaluated as a separate method family from UFMR, using the ROB-80 centered LR grid and two repeats for consistency with the final ROB-80 sweep contract.

Completed cells: 12/12

## Aggregate

| Method | Epochs | LR | N | Mean Original WER | Mean Updated WER | Updated WER Std | Mean Abs Delta | Mean Rel Delta % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| UVQLM | 1 | `5e-6` | 2 | 0.100088 | 0.090030 | 0.000938 | -0.010058 | -10.05 |
| UVQLM | 1 | `1e-5` | 2 | 0.100088 | 0.089090 | 0.001172 | -0.010998 | -10.99 |
| UVQLM | 1 | `2e-5` | 2 | 0.100088 | 0.089974 | 0.000704 | -0.010113 | -10.10 |
| UVQLM | 5 | `5e-6` | 2 | 0.100088 | 0.086824 | 0.000313 | -0.013264 | -13.25 |
| UVQLM | 5 | `1e-5` | 2 | 0.100088 | 0.087211 | 0.001172 | -0.012877 | -12.87 |
| UVQLM | 5 | `2e-5` | 2 | 0.100088 | 0.088068 | 0.002150 | -0.012020 | -12.01 |

## Per Repeat

| Method | Repeat | Seed | Epochs | LR | Original WER | Updated WER | Abs Delta | Rel Delta % | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| UVQLM | 1 | 123456 | 1 | `5e-6` | 0.100088 | 0.090693 | -0.009395 | -9.39 | complete |
| UVQLM | 1 | 123456 | 1 | `1e-5` | 0.100088 | 0.089919 | -0.010169 | -10.16 | complete |
| UVQLM | 1 | 123456 | 1 | `2e-5` | 0.100088 | 0.090472 | -0.009616 | -9.61 | complete |
| UVQLM | 1 | 123456 | 5 | `5e-6` | 0.100088 | 0.086603 | -0.013485 | -13.47 | complete |
| UVQLM | 1 | 123456 | 5 | `1e-5` | 0.100088 | 0.086382 | -0.013706 | -13.69 | complete |
| UVQLM | 1 | 123456 | 5 | `2e-5` | 0.100088 | 0.086548 | -0.013540 | -13.53 | complete |
| UVQLM | 2 | 123457 | 1 | `5e-6` | 0.100088 | 0.089367 | -0.010722 | -10.71 | complete |
| UVQLM | 2 | 123457 | 1 | `1e-5` | 0.100088 | 0.088261 | -0.011827 | -11.82 | complete |
| UVQLM | 2 | 123457 | 1 | `2e-5` | 0.100088 | 0.089477 | -0.010611 | -10.60 | complete |
| UVQLM | 2 | 123457 | 5 | `5e-6` | 0.100088 | 0.087045 | -0.013043 | -13.03 | complete |
| UVQLM | 2 | 123457 | 5 | `1e-5` | 0.100088 | 0.088040 | -0.012048 | -12.04 | complete |
| UVQLM | 2 | 123457 | 5 | `2e-5` | 0.100088 | 0.089588 | -0.010501 | -10.49 | complete |
