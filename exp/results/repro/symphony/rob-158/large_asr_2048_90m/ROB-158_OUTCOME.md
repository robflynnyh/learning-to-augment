# ROB-158 UFMR and RFM Large-ASR Evaluation

Test-split evals using the SAP-style 2048-seq-len 90M ASR checkpoint. The
datasets, repeat count, candidate count, and evaluation harness match the
ROB-108 repro setup so the intended variable is ASR model size. UFMR uses the
full ROB-108 UFMR cell set; the follow-up RFM comparison uses the requested
reduced matrix at LR `1e-5` for 1 and 5 adaptation epochs.

Completed cells:

- UFMR: 15/15
- RFM: 10/10

## Interpretation

The large-ASR UFMR run completed all 15 requested cells. UFMR improves WER in
13/15 large-ASR cells; both failures are at the five-epoch `1e-5` setting on
`rev16` and `TAL`.

Compared with the ROB-108 small-ASR UFMR rows, the one-epoch large-ASR setting
still transfers cleanly: all 10 one-epoch cells improve WER, and 8/10 have a
stronger relative WER reduction than the small-ASR run. The five-epoch setting
is less stable across model size: `tedlium`, `earnings22`, and `chime6` still
improve, but `rev16` degrades from 0.152681 to 0.490438 WER and `TAL` degrades
from 0.139755 to 0.184432 WER.

The large-ASR RFM follow-up completed all 10 requested cells. RFM improves WER
in every large-ASR cell, with mean relative WER reductions of `13.64%` at one
epoch and `14.99%` at five epochs. Compared with the ROB-108 small-ASR RFM
rows, the larger ASR model improves in all 10 matched cells and has a stronger
relative WER reduction in 8/10 cells. The only weaker relative reductions are
`tedlium` at one and five epochs; the large-ASR updated WER is still lower
there because the unadapted large-ASR baseline starts lower.

On the shared large-ASR `1e-5` cells, UFMR has the lower updated WER in all
five one-epoch comparisons. At five epochs, RFM has the lower updated WER in
4/5 comparisons, with `chime6` the only five-epoch cell where UFMR remains
better. Overall, these results support UFMR transfer to the larger 2048-context
90M ASR model for one-epoch adaptation, but they do not support treating the
ROB-108 five-epoch UFMR recipe as model-size invariant without additional
tuning or stability checks. RFM is the more stable five-epoch large-ASR
baseline in this single-repeat result because it avoids the UFMR `rev16` and
`TAL` regressions.

Detailed comparison files:

- `rob158_vs_rob108_ufmr_comparison.csv`
- `rob158_vs_rob108_rfm_comparison.csv`
- `rob158_large_asr_ufmr_vs_rfm_comparison.csv`

The `repeat` column is retained even though ROB-108 starts with one repeat, so
the tables can be extended without changing schema.

## Large-ASR UFMR vs RFM

Shared cells are LR `1e-5` at 1 and 5 adaptation epochs.

| Dataset | Epochs | UFMR Updated WER | UFMR Rel Delta % | RFM Updated WER | RFM Rel Delta % | Updated WER Winner |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| TAL | 1 | 0.131779 | -5.71 | 0.132645 | -5.09 | UFMR |
| TAL | 5 | 0.184432 | 31.97 | 0.132284 | -5.35 | RFM |
| chime6 | 1 | 0.591558 | -30.63 | 0.607067 | -28.81 | UFMR |
| chime6 | 5 | 0.570074 | -33.14 | 0.590694 | -30.73 | UFMR |
| earnings22 | 1 | 0.149256 | -23.65 | 0.155954 | -20.23 | UFMR |
| earnings22 | 5 | 0.150685 | -22.92 | 0.150481 | -23.03 | RFM |
| rev16 | 1 | 0.140875 | -7.73 | 0.142269 | -6.82 | UFMR |
| rev16 | 5 | 0.490438 | 221.22 | 0.142565 | -6.63 | RFM |
| tedlium | 1 | 0.059862 | -8.50 | 0.060677 | -7.26 | UFMR |
| tedlium | 5 | 0.059968 | -8.34 | 0.059401 | -9.21 | RFM |

## UFMR Aggregate

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

## RFM Aggregate

| Dataset | Method | Epochs | LR | N | Mean Original WER | Mean Updated WER | Updated WER Std | Mean Abs Delta | Mean Rel Delta % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TAL | RFM | 1 | `1e-5` | 1 | 0.139755 | 0.132645 | 0.000000 | -0.007110 | -5.09 |
| TAL | RFM | 5 | `1e-5` | 1 | 0.139755 | 0.132284 | 0.000000 | -0.007471 | -5.35 |
| chime6 | RFM | 1 | `1e-5` | 1 | 0.852697 | 0.607067 | 0.000000 | -0.245630 | -28.81 |
| chime6 | RFM | 5 | `1e-5` | 1 | 0.852697 | 0.590694 | 0.000000 | -0.262003 | -30.73 |
| earnings22 | RFM | 1 | `1e-5` | 1 | 0.195495 | 0.155954 | 0.000000 | -0.039541 | -20.23 |
| earnings22 | RFM | 5 | `1e-5` | 1 | 0.195495 | 0.150481 | 0.000000 | -0.045014 | -23.03 |
| rev16 | RFM | 1 | `1e-5` | 1 | 0.152681 | 0.142269 | 0.000000 | -0.010412 | -6.82 |
| rev16 | RFM | 5 | `1e-5` | 1 | 0.152681 | 0.142565 | 0.000000 | -0.010116 | -6.63 |
| tedlium | RFM | 1 | `1e-5` | 1 | 0.065426 | 0.060677 | 0.000000 | -0.004749 | -7.26 |
| tedlium | RFM | 5 | `1e-5` | 1 | 0.065426 | 0.059401 | 0.000000 | -0.006025 | -9.21 |

## UFMR Per Repeat

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

## RFM Per Repeat

| Dataset | Method | Repeat | Seed | Epochs | LR | Original WER | Updated WER | Abs Delta | Rel Delta % | Status | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| tedlium | RFM | 1 | 123456 | 1 | `1e-5` | 0.065426 | 0.060677 | -0.004749 | -7.26 | complete | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/RFM/tedlium_epoch1_lr1e-5.txt` |
| tedlium | RFM | 1 | 123456 | 5 | `1e-5` | 0.065426 | 0.059401 | -0.006025 | -9.21 | complete | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/RFM/tedlium_epoch5_lr1e-5.txt` |
| earnings22 | RFM | 1 | 123456 | 1 | `1e-5` | 0.195495 | 0.155954 | -0.039540 | -20.23 | complete | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/RFM/earnings22_epoch1_lr1e-5.txt` |
| earnings22 | RFM | 1 | 123456 | 5 | `1e-5` | 0.195495 | 0.150481 | -0.045014 | -23.03 | complete | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/RFM/earnings22_epoch5_lr1e-5.txt` |
| chime6 | RFM | 1 | 123456 | 1 | `1e-5` | 0.852697 | 0.607067 | -0.245629 | -28.81 | complete | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/RFM/chime6_epoch1_lr1e-5.txt` |
| chime6 | RFM | 1 | 123456 | 5 | `1e-5` | 0.852697 | 0.590694 | -0.262002 | -30.73 | complete | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/RFM/chime6_epoch5_lr1e-5.txt` |
| rev16 | RFM | 1 | 123456 | 1 | `1e-5` | 0.152681 | 0.142269 | -0.010412 | -6.82 | complete | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/RFM/rev16_epoch1_lr1e-5.txt` |
| rev16 | RFM | 1 | 123456 | 5 | `1e-5` | 0.152681 | 0.142565 | -0.010116 | -6.63 | complete | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/RFM/rev16_epoch5_lr1e-5.txt` |
| TAL | RFM | 1 | 123456 | 1 | `1e-5` | 0.139755 | 0.132645 | -0.007111 | -5.09 | complete | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/RFM/TAL_epoch1_lr1e-5.txt` |
| TAL | RFM | 1 | 123456 | 5 | `1e-5` | 0.139755 | 0.132284 | -0.007471 | -5.35 | complete | `exp/results/repro/symphony/rob-158/large_asr_2048_90m/results/RFM/TAL_epoch5_lr1e-5.txt` |
