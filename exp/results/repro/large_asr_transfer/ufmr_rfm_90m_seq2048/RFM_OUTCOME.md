# ROB-158 RFM Large-ASR Evaluation

RFM test-split evals using the same SAP-style 2048-seq-len 90M ASR checkpoint as the ROB-158 UFMR run. Per the follow-up request, RFM is evaluated only at LR 1e-5 for 1 and 5 adaptation epochs.

Completed cells: 10/10

The `repeat` column is retained even though ROB-108 starts with one repeat, so the table can be extended without changing schema.

## Interpretation

The large-ASR RFM follow-up completed all 10 requested cells: five test
datasets at LR `1e-5` for 1 and 5 adaptation epochs. RFM improves WER in every
large-ASR cell, with mean relative WER reductions of `13.64%` at one epoch and
`14.99%` at five epochs.

Compared with the ROB-108 small-ASR RFM rows, the larger ASR model still
improves in all 10 matched cells and has a stronger relative WER reduction in
8/10 cells. The only weaker relative reductions are `tedlium` at one and five
epochs; the large-ASR updated WER is still lower there because the unadapted
large-ASR baseline starts lower.

On the shared large-ASR `1e-5` cells, UFMR has the lower updated WER in all
five one-epoch comparisons. At five epochs, RFM has the lower updated WER in
4/5 comparisons, with `chime6` the only five-epoch cell where UFMR remains
better. The practical readout is that UFMR transfers cleanly for short
one-epoch adaptation, but RFM is the more stable five-epoch large-ASR baseline
in this single-repeat result because it avoids the UFMR `rev16` and `TAL`
regressions.

The detailed comparison files are:

- `rob158_vs_rob108_rfm_comparison.csv`
- `rob158_large_asr_ufmr_vs_rfm_comparison.csv`

## Aggregate

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

## Per Repeat

| Dataset | Method | Repeat | Seed | Epochs | LR | Original WER | Updated WER | Abs Delta | Rel Delta % | Status | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| tedlium | RFM | 1 | 123456 | 1 | `1e-5` | 0.065426 | 0.060677 | -0.004749 | -7.26 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/RFM/tedlium_epoch1_lr1e-5.txt` |
| tedlium | RFM | 1 | 123456 | 5 | `1e-5` | 0.065426 | 0.059401 | -0.006025 | -9.21 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/RFM/tedlium_epoch5_lr1e-5.txt` |
| earnings22 | RFM | 1 | 123456 | 1 | `1e-5` | 0.195495 | 0.155954 | -0.039540 | -20.23 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/RFM/earnings22_epoch1_lr1e-5.txt` |
| earnings22 | RFM | 1 | 123456 | 5 | `1e-5` | 0.195495 | 0.150481 | -0.045014 | -23.03 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/RFM/earnings22_epoch5_lr1e-5.txt` |
| chime6 | RFM | 1 | 123456 | 1 | `1e-5` | 0.852697 | 0.607067 | -0.245629 | -28.81 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/RFM/chime6_epoch1_lr1e-5.txt` |
| chime6 | RFM | 1 | 123456 | 5 | `1e-5` | 0.852697 | 0.590694 | -0.262002 | -30.73 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/RFM/chime6_epoch5_lr1e-5.txt` |
| rev16 | RFM | 1 | 123456 | 1 | `1e-5` | 0.152681 | 0.142269 | -0.010412 | -6.82 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/RFM/rev16_epoch1_lr1e-5.txt` |
| rev16 | RFM | 1 | 123456 | 5 | `1e-5` | 0.152681 | 0.142565 | -0.010116 | -6.63 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/RFM/rev16_epoch5_lr1e-5.txt` |
| TAL | RFM | 1 | 123456 | 1 | `1e-5` | 0.139755 | 0.132645 | -0.007111 | -5.09 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/RFM/TAL_epoch1_lr1e-5.txt` |
| TAL | RFM | 1 | 123456 | 5 | `1e-5` | 0.139755 | 0.132284 | -0.007471 | -5.35 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/RFM/TAL_epoch5_lr1e-5.txt` |
