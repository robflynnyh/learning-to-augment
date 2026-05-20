# ROB-82 TED-LIUM Dev UVQLM LR Sweep

UVQLM is evaluated as a separate method family from UFMR, using the ROB-80 centered LR grid and two repeats for consistency with the final ROB-80 sweep contract. The originally requested segmented-dev split was dropped by a later Linear comment on 2026-05-20 before completion.

Completed cells: 3/12

## Aggregate

| Method | Epochs | LR | N | Mean Original WER | Mean Updated WER | Updated WER Std | Mean Abs Delta | Mean Rel Delta % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| UVQLM | 1 | `5e-6` | 1 | 0.100088 | 0.090693 | 0.000000 | -0.009395 | -9.39 |
| UVQLM | 1 | `1e-5` | 1 | 0.100088 | 0.089919 | 0.000000 | -0.010169 | -10.16 |
| UVQLM | 1 | `2e-5` | 1 | 0.100088 | 0.090472 | 0.000000 | -0.009616 | -9.61 |

## Per Repeat

| Method | Repeat | Seed | Epochs | LR | Original WER | Updated WER | Abs Delta | Rel Delta % | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| UVQLM | 1 | 123456 | 1 | `5e-6` | 0.100088 | 0.090693 | -0.009395 | -9.39 | complete |
| UVQLM | 1 | 123456 | 1 | `1e-5` | 0.100088 | 0.089919 | -0.010169 | -10.16 | complete |
| UVQLM | 1 | 123456 | 1 | `2e-5` | 0.100088 | 0.090472 | -0.009616 | -9.61 | complete |
| UVQLM | 1 | 123456 | 5 | `5e-6` |  |  |  |  | missing |
| UVQLM | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing |
| UVQLM | 1 | 123456 | 5 | `2e-5` |  |  |  |  | missing |
| UVQLM | 2 | 123457 | 1 | `5e-6` |  |  |  |  | missing |
| UVQLM | 2 | 123457 | 1 | `1e-5` |  |  |  |  | missing |
| UVQLM | 2 | 123457 | 1 | `2e-5` |  |  |  |  | missing |
| UVQLM | 2 | 123457 | 5 | `5e-6` |  |  |  |  | missing |
| UVQLM | 2 | 123457 | 5 | `1e-5` |  |  |  |  | missing |
| UVQLM | 2 | 123457 | 5 | `2e-5` |  |  |  |  | missing |
