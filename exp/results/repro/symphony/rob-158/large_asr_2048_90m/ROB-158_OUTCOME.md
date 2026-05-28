# ROB-158 UFMR Large-ASR Evaluation

UFMR test-split evals using the SAP-style 2048-seq-len 90M ASR checkpoint. The UFMR policy, datasets, repeats, candidate count, and LR/epoch cells match the ROB-108 UFMR repro setup so the intended variable is ASR model size.

Completed cells: 0/15

The `repeat` column is retained even though ROB-108 starts with one repeat, so the table can be extended without changing schema.

## Aggregate

| Dataset | Method | Epochs | LR | N | Mean Original WER | Mean Updated WER | Updated WER Std | Mean Abs Delta | Mean Rel Delta % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## Missing Cells

- tedlium / UFMR / repeat 1 / epoch 1 / lr `1e-5`
- tedlium / UFMR / repeat 1 / epoch 1 / lr `3e-5`
- tedlium / UFMR / repeat 1 / epoch 5 / lr `1e-5`
- earnings22 / UFMR / repeat 1 / epoch 1 / lr `1e-5`
- earnings22 / UFMR / repeat 1 / epoch 1 / lr `3e-5`
- earnings22 / UFMR / repeat 1 / epoch 5 / lr `1e-5`
- chime6 / UFMR / repeat 1 / epoch 1 / lr `1e-5`
- chime6 / UFMR / repeat 1 / epoch 1 / lr `3e-5`
- chime6 / UFMR / repeat 1 / epoch 5 / lr `1e-5`
- rev16 / UFMR / repeat 1 / epoch 1 / lr `1e-5`
- rev16 / UFMR / repeat 1 / epoch 1 / lr `3e-5`
- rev16 / UFMR / repeat 1 / epoch 5 / lr `1e-5`
- TAL / UFMR / repeat 1 / epoch 1 / lr `1e-5`
- TAL / UFMR / repeat 1 / epoch 1 / lr `3e-5`
- TAL / UFMR / repeat 1 / epoch 5 / lr `1e-5`

## Per Repeat

| Dataset | Method | Repeat | Seed | Epochs | LR | Original WER | Updated WER | Abs Delta | Rel Delta % | Status | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| tedlium | UFMR | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/UFMR/tedlium_epoch1_lr1e-5.txt` |
| tedlium | UFMR | 1 | 123456 | 1 | `3e-5` |  |  |  |  | missing | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/UFMR/tedlium_epoch1_lr3e-5.txt` |
| tedlium | UFMR | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/UFMR/tedlium_epoch5_lr1e-5.txt` |
| earnings22 | UFMR | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/UFMR/earnings22_epoch1_lr1e-5.txt` |
| earnings22 | UFMR | 1 | 123456 | 1 | `3e-5` |  |  |  |  | missing | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/UFMR/earnings22_epoch1_lr3e-5.txt` |
| earnings22 | UFMR | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/UFMR/earnings22_epoch5_lr1e-5.txt` |
| chime6 | UFMR | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/UFMR/chime6_epoch1_lr1e-5.txt` |
| chime6 | UFMR | 1 | 123456 | 1 | `3e-5` |  |  |  |  | missing | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/UFMR/chime6_epoch1_lr3e-5.txt` |
| chime6 | UFMR | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/UFMR/chime6_epoch5_lr1e-5.txt` |
| rev16 | UFMR | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/UFMR/rev16_epoch1_lr1e-5.txt` |
| rev16 | UFMR | 1 | 123456 | 1 | `3e-5` |  |  |  |  | missing | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/UFMR/rev16_epoch1_lr3e-5.txt` |
| rev16 | UFMR | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/UFMR/rev16_epoch5_lr1e-5.txt` |
| TAL | UFMR | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/UFMR/TAL_epoch1_lr1e-5.txt` |
| TAL | UFMR | 1 | 123456 | 1 | `3e-5` |  |  |  |  | missing | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/UFMR/TAL_epoch1_lr3e-5.txt` |
| TAL | UFMR | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/UFMR/TAL_epoch5_lr1e-5.txt` |
