# ROB-108 Test Policy Evaluations

Test split policy evals for RFM, RMM, UFMR, and UVQLM across TED-LIUM, Earnings22, CHiME-6, Rev16, and This American Life. NoAug rows are unadapted baselines.

Completed cells: 0/65

The `repeat` column is retained even though ROB-108 starts with one repeat, so the table can be extended without changing schema.

## Aggregate

| Dataset | Method | Epochs | LR | N | Mean Original WER | Mean Updated WER | Updated WER Std | Mean Abs Delta | Mean Rel Delta % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## Missing Cells

- tedlium / NoAug / repeat 1 / epoch 0 / lr `baseline`
- tedlium / RFM / repeat 1 / epoch 1 / lr `1e-5`
- tedlium / RFM / repeat 1 / epoch 1 / lr `3e-5`
- tedlium / RFM / repeat 1 / epoch 5 / lr `1e-5`
- tedlium / RMM / repeat 1 / epoch 1 / lr `1e-5`
- tedlium / RMM / repeat 1 / epoch 1 / lr `3e-5`
- tedlium / RMM / repeat 1 / epoch 5 / lr `1e-5`
- tedlium / UFMR / repeat 1 / epoch 1 / lr `1e-5`
- tedlium / UFMR / repeat 1 / epoch 1 / lr `3e-5`
- tedlium / UFMR / repeat 1 / epoch 5 / lr `1e-5`
- tedlium / UVQLM / repeat 1 / epoch 1 / lr `1e-5`
- tedlium / UVQLM / repeat 1 / epoch 1 / lr `3e-5`
- tedlium / UVQLM / repeat 1 / epoch 5 / lr `1e-5`
- earnings22 / NoAug / repeat 1 / epoch 0 / lr `baseline`
- earnings22 / RFM / repeat 1 / epoch 1 / lr `1e-5`
- earnings22 / RFM / repeat 1 / epoch 1 / lr `3e-5`
- earnings22 / RFM / repeat 1 / epoch 5 / lr `1e-5`
- earnings22 / RMM / repeat 1 / epoch 1 / lr `1e-5`
- earnings22 / RMM / repeat 1 / epoch 1 / lr `3e-5`
- earnings22 / RMM / repeat 1 / epoch 5 / lr `1e-5`
- earnings22 / UFMR / repeat 1 / epoch 1 / lr `1e-5`
- earnings22 / UFMR / repeat 1 / epoch 1 / lr `3e-5`
- earnings22 / UFMR / repeat 1 / epoch 5 / lr `1e-5`
- earnings22 / UVQLM / repeat 1 / epoch 1 / lr `1e-5`
- earnings22 / UVQLM / repeat 1 / epoch 1 / lr `3e-5`
- earnings22 / UVQLM / repeat 1 / epoch 5 / lr `1e-5`
- chime6 / NoAug / repeat 1 / epoch 0 / lr `baseline`
- chime6 / RFM / repeat 1 / epoch 1 / lr `1e-5`
- chime6 / RFM / repeat 1 / epoch 1 / lr `3e-5`
- chime6 / RFM / repeat 1 / epoch 5 / lr `1e-5`
- chime6 / RMM / repeat 1 / epoch 1 / lr `1e-5`
- chime6 / RMM / repeat 1 / epoch 1 / lr `3e-5`
- chime6 / RMM / repeat 1 / epoch 5 / lr `1e-5`
- chime6 / UFMR / repeat 1 / epoch 1 / lr `1e-5`
- chime6 / UFMR / repeat 1 / epoch 1 / lr `3e-5`
- chime6 / UFMR / repeat 1 / epoch 5 / lr `1e-5`
- chime6 / UVQLM / repeat 1 / epoch 1 / lr `1e-5`
- chime6 / UVQLM / repeat 1 / epoch 1 / lr `3e-5`
- chime6 / UVQLM / repeat 1 / epoch 5 / lr `1e-5`
- rev16 / NoAug / repeat 1 / epoch 0 / lr `baseline`
- ... 25 more missing cells

## Per Repeat

| Dataset | Method | Repeat | Seed | Epochs | LR | Original WER | Updated WER | Abs Delta | Rel Delta % | Status | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| tedlium | NoAug | 1 | 123456 | 0 | `baseline` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/NoAug/tedlium_baseline.txt` |
| tedlium | RFM | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/RFM/tedlium_epoch1_lr1e-5.txt` |
| tedlium | RFM | 1 | 123456 | 1 | `3e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/RFM/tedlium_epoch1_lr3e-5.txt` |
| tedlium | RFM | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/RFM/tedlium_epoch5_lr1e-5.txt` |
| tedlium | RMM | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/RMM/tedlium_epoch1_lr1e-5.txt` |
| tedlium | RMM | 1 | 123456 | 1 | `3e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/RMM/tedlium_epoch1_lr3e-5.txt` |
| tedlium | RMM | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/RMM/tedlium_epoch5_lr1e-5.txt` |
| tedlium | UFMR | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/UFMR/tedlium_epoch1_lr1e-5.txt` |
| tedlium | UFMR | 1 | 123456 | 1 | `3e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/UFMR/tedlium_epoch1_lr3e-5.txt` |
| tedlium | UFMR | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/UFMR/tedlium_epoch5_lr1e-5.txt` |
| tedlium | UVQLM | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/UVQLM/tedlium_epoch1_lr1e-5.txt` |
| tedlium | UVQLM | 1 | 123456 | 1 | `3e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/UVQLM/tedlium_epoch1_lr3e-5.txt` |
| tedlium | UVQLM | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/UVQLM/tedlium_epoch5_lr1e-5.txt` |
| earnings22 | NoAug | 1 | 123456 | 0 | `baseline` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/NoAug/earnings22_baseline.txt` |
| earnings22 | RFM | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/RFM/earnings22_epoch1_lr1e-5.txt` |
| earnings22 | RFM | 1 | 123456 | 1 | `3e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/RFM/earnings22_epoch1_lr3e-5.txt` |
| earnings22 | RFM | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/RFM/earnings22_epoch5_lr1e-5.txt` |
| earnings22 | RMM | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/RMM/earnings22_epoch1_lr1e-5.txt` |
| earnings22 | RMM | 1 | 123456 | 1 | `3e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/RMM/earnings22_epoch1_lr3e-5.txt` |
| earnings22 | RMM | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/RMM/earnings22_epoch5_lr1e-5.txt` |
| earnings22 | UFMR | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/UFMR/earnings22_epoch1_lr1e-5.txt` |
| earnings22 | UFMR | 1 | 123456 | 1 | `3e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/UFMR/earnings22_epoch1_lr3e-5.txt` |
| earnings22 | UFMR | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/UFMR/earnings22_epoch5_lr1e-5.txt` |
| earnings22 | UVQLM | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/UVQLM/earnings22_epoch1_lr1e-5.txt` |
| earnings22 | UVQLM | 1 | 123456 | 1 | `3e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/UVQLM/earnings22_epoch1_lr3e-5.txt` |
| earnings22 | UVQLM | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/UVQLM/earnings22_epoch5_lr1e-5.txt` |
| chime6 | NoAug | 1 | 123456 | 0 | `baseline` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/NoAug/chime6_baseline.txt` |
| chime6 | RFM | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/RFM/chime6_epoch1_lr1e-5.txt` |
| chime6 | RFM | 1 | 123456 | 1 | `3e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/RFM/chime6_epoch1_lr3e-5.txt` |
| chime6 | RFM | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/RFM/chime6_epoch5_lr1e-5.txt` |
| chime6 | RMM | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/RMM/chime6_epoch1_lr1e-5.txt` |
| chime6 | RMM | 1 | 123456 | 1 | `3e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/RMM/chime6_epoch1_lr3e-5.txt` |
| chime6 | RMM | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/RMM/chime6_epoch5_lr1e-5.txt` |
| chime6 | UFMR | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/UFMR/chime6_epoch1_lr1e-5.txt` |
| chime6 | UFMR | 1 | 123456 | 1 | `3e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/UFMR/chime6_epoch1_lr3e-5.txt` |
| chime6 | UFMR | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/UFMR/chime6_epoch5_lr1e-5.txt` |
| chime6 | UVQLM | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/UVQLM/chime6_epoch1_lr1e-5.txt` |
| chime6 | UVQLM | 1 | 123456 | 1 | `3e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/UVQLM/chime6_epoch1_lr3e-5.txt` |
| chime6 | UVQLM | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/UVQLM/chime6_epoch5_lr1e-5.txt` |
| rev16 | NoAug | 1 | 123456 | 0 | `baseline` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/NoAug/rev16_baseline.txt` |
| rev16 | RFM | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/RFM/rev16_epoch1_lr1e-5.txt` |
| rev16 | RFM | 1 | 123456 | 1 | `3e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/RFM/rev16_epoch1_lr3e-5.txt` |
| rev16 | RFM | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/RFM/rev16_epoch5_lr1e-5.txt` |
| rev16 | RMM | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/RMM/rev16_epoch1_lr1e-5.txt` |
| rev16 | RMM | 1 | 123456 | 1 | `3e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/RMM/rev16_epoch1_lr3e-5.txt` |
| rev16 | RMM | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/RMM/rev16_epoch5_lr1e-5.txt` |
| rev16 | UFMR | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/UFMR/rev16_epoch1_lr1e-5.txt` |
| rev16 | UFMR | 1 | 123456 | 1 | `3e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/UFMR/rev16_epoch1_lr3e-5.txt` |
| rev16 | UFMR | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/UFMR/rev16_epoch5_lr1e-5.txt` |
| rev16 | UVQLM | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/UVQLM/rev16_epoch1_lr1e-5.txt` |
| rev16 | UVQLM | 1 | 123456 | 1 | `3e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/UVQLM/rev16_epoch1_lr3e-5.txt` |
| rev16 | UVQLM | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/UVQLM/rev16_epoch5_lr1e-5.txt` |
| TAL | NoAug | 1 | 123456 | 0 | `baseline` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/NoAug/TAL_baseline.txt` |
| TAL | RFM | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/RFM/TAL_epoch1_lr1e-5.txt` |
| TAL | RFM | 1 | 123456 | 1 | `3e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/RFM/TAL_epoch1_lr3e-5.txt` |
| TAL | RFM | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/RFM/TAL_epoch5_lr1e-5.txt` |
| TAL | RMM | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/RMM/TAL_epoch1_lr1e-5.txt` |
| TAL | RMM | 1 | 123456 | 1 | `3e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/RMM/TAL_epoch1_lr3e-5.txt` |
| TAL | RMM | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/RMM/TAL_epoch5_lr1e-5.txt` |
| TAL | UFMR | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/UFMR/TAL_epoch1_lr1e-5.txt` |
| TAL | UFMR | 1 | 123456 | 1 | `3e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/UFMR/TAL_epoch1_lr3e-5.txt` |
| TAL | UFMR | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/UFMR/TAL_epoch5_lr1e-5.txt` |
| TAL | UVQLM | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/UVQLM/TAL_epoch1_lr1e-5.txt` |
| TAL | UVQLM | 1 | 123456 | 1 | `3e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/UVQLM/TAL_epoch1_lr3e-5.txt` |
| TAL | UVQLM | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/sweeps/rob108_test_policy_evals/UVQLM/TAL_epoch5_lr1e-5.txt` |
