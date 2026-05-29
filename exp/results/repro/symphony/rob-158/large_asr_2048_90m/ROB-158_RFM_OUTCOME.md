# ROB-158 RFM Large-ASR Evaluation

RFM test-split evals using the same SAP-style 2048-seq-len 90M ASR checkpoint as the ROB-158 UFMR run. Per the follow-up request, RFM is evaluated only at LR 1e-5 for 1 and 5 adaptation epochs.

Completed cells: 0/10

The `repeat` column is retained even though ROB-108 starts with one repeat, so the table can be extended without changing schema.

## Aggregate

| Dataset | Method | Epochs | LR | N | Mean Original WER | Mean Updated WER | Updated WER Std | Mean Abs Delta | Mean Rel Delta % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## Missing Cells

- tedlium / RFM / repeat 1 / epoch 1 / lr `1e-5`
- tedlium / RFM / repeat 1 / epoch 5 / lr `1e-5`
- earnings22 / RFM / repeat 1 / epoch 1 / lr `1e-5`
- earnings22 / RFM / repeat 1 / epoch 5 / lr `1e-5`
- chime6 / RFM / repeat 1 / epoch 1 / lr `1e-5`
- chime6 / RFM / repeat 1 / epoch 5 / lr `1e-5`
- rev16 / RFM / repeat 1 / epoch 1 / lr `1e-5`
- rev16 / RFM / repeat 1 / epoch 5 / lr `1e-5`
- TAL / RFM / repeat 1 / epoch 1 / lr `1e-5`
- TAL / RFM / repeat 1 / epoch 5 / lr `1e-5`

## Per Repeat

| Dataset | Method | Repeat | Seed | Epochs | LR | Original WER | Updated WER | Abs Delta | Rel Delta % | Status | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| tedlium | RFM | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/RFM/tedlium_epoch1_lr1e-5.txt` |
| tedlium | RFM | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/RFM/tedlium_epoch5_lr1e-5.txt` |
| earnings22 | RFM | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/RFM/earnings22_epoch1_lr1e-5.txt` |
| earnings22 | RFM | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/RFM/earnings22_epoch5_lr1e-5.txt` |
| chime6 | RFM | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/RFM/chime6_epoch1_lr1e-5.txt` |
| chime6 | RFM | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/RFM/chime6_epoch5_lr1e-5.txt` |
| rev16 | RFM | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/RFM/rev16_epoch1_lr1e-5.txt` |
| rev16 | RFM | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/RFM/rev16_epoch5_lr1e-5.txt` |
| TAL | RFM | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/RFM/TAL_epoch1_lr1e-5.txt` |
| TAL | RFM | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/RFM/TAL_epoch5_lr1e-5.txt` |
