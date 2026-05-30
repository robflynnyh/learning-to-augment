# ROB-158 UFMR Large-ASR Evaluation

UFMR test-split evals using the SAP-style 2048-seq-len 90M ASR checkpoint. The UFMR policy, datasets, repeats, candidate count, and LR/epoch cells match the ROB-108 UFMR repro setup so the intended variable is ASR model size.

Completed cells: 15/15

The `repeat` column is retained even though ROB-108 starts with one repeat, so the table can be extended without changing schema.

## Interpretation

The large-ASR UFMR run completed all 15 requested cells. UFMR improves WER in 13/15 large-ASR cells; both failures are at the five-epoch `1e-5` setting on `rev16` and `TAL`.

Compared with the ROB-108 small-ASR UFMR rows, the one-epoch large-ASR setting still transfers cleanly: all 10 one-epoch cells improve WER, and 8/10 have a stronger relative WER reduction than the small-ASR run. The five-epoch setting is less stable across model size: `tedlium`, `earnings22`, and `chime6` still improve, but `rev16` degrades from 0.152681 to 0.490438 WER and `TAL` degrades from 0.139755 to 0.184432 WER.

The detailed cross-model comparison is recorded in `rob158_vs_rob108_ufmr_comparison.csv`. Overall, these results support UFMR transfer to the larger 2048-context 90M ASR model for one-epoch adaptation, but they do not support treating the ROB-108 five-epoch recipe as model-size invariant without additional tuning or stability checks.

A follow-up RFM comparison on the same large-ASR checkpoint is recorded in
`ROB-158_RFM_OUTCOME.md`, `rob158_vs_rob108_rfm_comparison.csv`, and
`rob158_large_asr_ufmr_vs_rfm_comparison.csv`. RFM improved all 10 overlapping
large-ASR cells, while UFMR remained stronger in all five one-epoch `1e-5`
matched cells. At five epochs, RFM avoided the large UFMR regressions on
`rev16` and `TAL`, making it the safer five-epoch large-ASR baseline in this
single-repeat comparison.

## Aggregate

| Dataset | Method | Epochs | LR | N | Mean Original WER | Mean Updated WER | Updated WER Std | Mean Abs Delta | Mean Rel Delta % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TAL | UFMR | 1 | `1e-5` | 1 | 0.139755 | 0.131779 | 0.000000 | -0.007976 | -5.71 |
| TAL | UFMR | 1 | `3e-5` | 1 | 0.139755 | 0.134390 | 0.000000 | -0.005365 | -3.84 |
| TAL | UFMR | 5 | `1e-5` | 1 | 0.139755 | 0.184432 | 0.000000 | 0.044677 | 31.97 |
| chime6 | UFMR | 1 | `1e-5` | 1 | 0.852697 | 0.591558 | 0.000000 | -0.261139 | -30.63 |
| chime6 | UFMR | 1 | `3e-5` | 1 | 0.852697 | 0.595788 | 0.000000 | -0.256909 | -30.13 |
| chime6 | UFMR | 5 | `1e-5` | 1 | 0.852697 | 0.570074 | 0.000000 | -0.282623 | -33.14 |
| earnings22 | UFMR | 1 | `1e-5` | 1 | 0.195495 | 0.149256 | 0.000000 | -0.046239 | -23.65 |
| earnings22 | UFMR | 1 | `3e-5` | 1 | 0.195495 | 0.148091 | 0.000000 | -0.047404 | -24.25 |
| earnings22 | UFMR | 5 | `1e-5` | 1 | 0.195495 | 0.150685 | 0.000000 | -0.044810 | -22.92 |
| rev16 | UFMR | 1 | `1e-5` | 1 | 0.152681 | 0.140875 | 0.000000 | -0.011806 | -7.73 |
| rev16 | UFMR | 1 | `3e-5` | 1 | 0.152681 | 0.143847 | 0.000000 | -0.008834 | -5.79 |
| rev16 | UFMR | 5 | `1e-5` | 1 | 0.152681 | 0.490438 | 0.000000 | 0.337757 | 221.22 |
| tedlium | UFMR | 1 | `1e-5` | 1 | 0.065426 | 0.059862 | 0.000000 | -0.005564 | -8.50 |
| tedlium | UFMR | 1 | `3e-5` | 1 | 0.065426 | 0.059755 | 0.000000 | -0.005671 | -8.67 |
| tedlium | UFMR | 5 | `1e-5` | 1 | 0.065426 | 0.059968 | 0.000000 | -0.005458 | -8.34 |

## Per Repeat

| Dataset | Method | Repeat | Seed | Epochs | LR | Original WER | Updated WER | Abs Delta | Rel Delta % | Status | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| tedlium | UFMR | 1 | 123456 | 1 | `1e-5` | 0.065426 | 0.059862 | -0.005564 | -8.50 | complete | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/UFMR/tedlium_epoch1_lr1e-5.txt` |
| tedlium | UFMR | 1 | 123456 | 1 | `3e-5` | 0.065426 | 0.059755 | -0.005671 | -8.67 | complete | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/UFMR/tedlium_epoch1_lr3e-5.txt` |
| tedlium | UFMR | 1 | 123456 | 5 | `1e-5` | 0.065426 | 0.059968 | -0.005458 | -8.34 | complete | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/UFMR/tedlium_epoch5_lr1e-5.txt` |
| earnings22 | UFMR | 1 | 123456 | 1 | `1e-5` | 0.195495 | 0.149256 | -0.046239 | -23.65 | complete | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/UFMR/earnings22_epoch1_lr1e-5.txt` |
| earnings22 | UFMR | 1 | 123456 | 1 | `3e-5` | 0.195495 | 0.148091 | -0.047403 | -24.25 | complete | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/UFMR/earnings22_epoch1_lr3e-5.txt` |
| earnings22 | UFMR | 1 | 123456 | 5 | `1e-5` | 0.195495 | 0.150685 | -0.044809 | -22.92 | complete | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/UFMR/earnings22_epoch5_lr1e-5.txt` |
| chime6 | UFMR | 1 | 123456 | 1 | `1e-5` | 0.852697 | 0.591558 | -0.261139 | -30.63 | complete | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/UFMR/chime6_epoch1_lr1e-5.txt` |
| chime6 | UFMR | 1 | 123456 | 1 | `3e-5` | 0.852697 | 0.595788 | -0.256909 | -30.13 | complete | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/UFMR/chime6_epoch1_lr3e-5.txt` |
| chime6 | UFMR | 1 | 123456 | 5 | `1e-5` | 0.852697 | 0.570074 | -0.282622 | -33.14 | complete | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/UFMR/chime6_epoch5_lr1e-5.txt` |
| rev16 | UFMR | 1 | 123456 | 1 | `1e-5` | 0.152681 | 0.140875 | -0.011806 | -7.73 | complete | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/UFMR/rev16_epoch1_lr1e-5.txt` |
| rev16 | UFMR | 1 | 123456 | 1 | `3e-5` | 0.152681 | 0.143847 | -0.008834 | -5.79 | complete | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/UFMR/rev16_epoch1_lr3e-5.txt` |
| rev16 | UFMR | 1 | 123456 | 5 | `1e-5` | 0.152681 | 0.490438 | 0.337757 | 221.22 | complete | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/UFMR/rev16_epoch5_lr1e-5.txt` |
| TAL | UFMR | 1 | 123456 | 1 | `1e-5` | 0.139755 | 0.131779 | -0.007976 | -5.71 | complete | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/UFMR/TAL_epoch1_lr1e-5.txt` |
| TAL | UFMR | 1 | 123456 | 1 | `3e-5` | 0.139755 | 0.134390 | -0.005365 | -3.84 | complete | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/UFMR/TAL_epoch1_lr3e-5.txt` |
| TAL | UFMR | 1 | 123456 | 5 | `1e-5` | 0.139755 | 0.184432 | 0.044676 | 31.97 | complete | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/UFMR/TAL_epoch5_lr1e-5.txt` |
