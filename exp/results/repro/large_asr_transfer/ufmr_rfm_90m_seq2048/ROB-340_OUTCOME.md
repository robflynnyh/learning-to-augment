# ROB-340 UFMR/RFM Large-ASR Repeat Completion

Thesis-table large-ASR UFMR/RFM cells only: 90M 2048-context ASR, test split, one adaptation epoch, lr=1e-5. Repeat 1 is the existing ROB-158 seed 123456 artifact; repeats 2 and 3 are new ROB-340 Stanage jobs with seeds 123457 and 123458.

Completed cells: 30/30

The `repeat` column is retained even though ROB-108 starts with one repeat, so the table can be extended without changing schema.

## Aggregate

| Dataset | Method | Epochs | LR | N | Mean Original WER | Mean Updated WER | Updated WER Std | Mean Abs Delta | Mean Rel Delta % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TAL | RFM | 1 | `1e-5` | 3 | 0.139752 | 0.132840 | 0.000280 | -0.006911 | -4.95 |
| TAL | UFMR | 1 | `1e-5` | 3 | 0.139751 | 0.132470 | 0.001396 | -0.007281 | -5.21 |
| chime6 | RFM | 1 | `1e-5` | 3 | 0.852673 | 0.610110 | 0.005240 | -0.242563 | -28.45 |
| chime6 | UFMR | 1 | `1e-5` | 3 | 0.852673 | 0.587463 | 0.004683 | -0.265210 | -31.10 |
| earnings22 | RFM | 1 | `1e-5` | 3 | 0.195474 | 0.156206 | 0.000321 | -0.039268 | -20.09 |
| earnings22 | UFMR | 1 | `1e-5` | 3 | 0.195474 | 0.150066 | 0.001583 | -0.045409 | -23.23 |
| rev16 | RFM | 1 | `1e-5` | 3 | 0.152592 | 0.142531 | 0.000280 | -0.010061 | -6.59 |
| rev16 | UFMR | 1 | `1e-5` | 3 | 0.152592 | 0.140664 | 0.000239 | -0.011928 | -7.82 |
| tedlium | RFM | 1 | `1e-5` | 3 | 0.065426 | 0.059945 | 0.000764 | -0.005481 | -8.38 |
| tedlium | UFMR | 1 | `1e-5` | 3 | 0.065426 | 0.059448 | 0.000517 | -0.005978 | -9.14 |

## Per Repeat

| Dataset | Method | Repeat | Seed | Epochs | LR | Original WER | Updated WER | Abs Delta | Rel Delta % | Status | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| tedlium | UFMR | 1 | 123456 | 1 | `1e-5` | 0.065426 | 0.059862 | -0.005564 | -8.50 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/UFMR/tedlium_epoch1_lr1e-5.txt` |
| tedlium | UFMR | 2 | 123457 | 1 | `1e-5` | 0.065426 | 0.058869 | -0.006557 | -10.02 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/UFMR/tedlium_epoch1_lr1e-5_repeat2.txt` |
| tedlium | UFMR | 3 | 123458 | 1 | `1e-5` | 0.065426 | 0.059614 | -0.005813 | -8.88 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/UFMR/tedlium_epoch1_lr1e-5_repeat3.txt` |
| tedlium | RFM | 1 | 123456 | 1 | `1e-5` | 0.065426 | 0.060677 | -0.004749 | -7.26 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/RFM/tedlium_epoch1_lr1e-5.txt` |
| tedlium | RFM | 2 | 123457 | 1 | `1e-5` | 0.065426 | 0.059153 | -0.006273 | -9.59 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/RFM/tedlium_epoch1_lr1e-5_repeat2.txt` |
| tedlium | RFM | 3 | 123458 | 1 | `1e-5` | 0.065426 | 0.060004 | -0.005423 | -8.29 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/RFM/tedlium_epoch1_lr1e-5_repeat3.txt` |
| earnings22 | UFMR | 1 | 123456 | 1 | `1e-5` | 0.195495 | 0.149256 | -0.046239 | -23.65 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/UFMR/earnings22_epoch1_lr1e-5.txt` |
| earnings22 | UFMR | 2 | 123457 | 1 | `1e-5` | 0.195454 | 0.149051 | -0.046402 | -23.74 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/UFMR/earnings22_epoch1_lr1e-5_repeat2.txt` |
| earnings22 | UFMR | 3 | 123458 | 1 | `1e-5` | 0.195474 | 0.151890 | -0.043584 | -22.30 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/UFMR/earnings22_epoch1_lr1e-5_repeat3.txt` |
| earnings22 | RFM | 1 | 123456 | 1 | `1e-5` | 0.195495 | 0.155954 | -0.039540 | -20.23 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/RFM/earnings22_epoch1_lr1e-5.txt` |
| earnings22 | RFM | 2 | 123457 | 1 | `1e-5` | 0.195474 | 0.156097 | -0.039377 | -20.14 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/RFM/earnings22_epoch1_lr1e-5_repeat2.txt` |
| earnings22 | RFM | 3 | 123458 | 1 | `1e-5` | 0.195454 | 0.156567 | -0.038887 | -19.90 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/RFM/earnings22_epoch1_lr1e-5_repeat3.txt` |
| chime6 | UFMR | 1 | 123456 | 1 | `1e-5` | 0.852697 | 0.591558 | -0.261139 | -30.63 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/UFMR/chime6_epoch1_lr1e-5.txt` |
| chime6 | UFMR | 2 | 123457 | 1 | `1e-5` | 0.852661 | 0.582358 | -0.270303 | -31.70 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/UFMR/chime6_epoch1_lr1e-5_repeat2.txt` |
| chime6 | UFMR | 3 | 123458 | 1 | `1e-5` | 0.852661 | 0.588474 | -0.264188 | -30.98 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/UFMR/chime6_epoch1_lr1e-5_repeat3.txt` |
| chime6 | RFM | 1 | 123456 | 1 | `1e-5` | 0.852697 | 0.607067 | -0.245629 | -28.81 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/RFM/chime6_epoch1_lr1e-5.txt` |
| chime6 | RFM | 2 | 123457 | 1 | `1e-5` | 0.852661 | 0.616161 | -0.236500 | -27.74 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/RFM/chime6_epoch1_lr1e-5_repeat2.txt` |
| chime6 | RFM | 3 | 123458 | 1 | `1e-5` | 0.852661 | 0.607103 | -0.245559 | -28.80 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/RFM/chime6_epoch1_lr1e-5_repeat3.txt` |
| rev16 | UFMR | 1 | 123456 | 1 | `1e-5` | 0.152681 | 0.140875 | -0.011806 | -7.73 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/UFMR/rev16_epoch1_lr1e-5.txt` |
| rev16 | UFMR | 2 | 123457 | 1 | `1e-5` | 0.152548 | 0.140405 | -0.012143 | -7.96 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/UFMR/rev16_epoch1_lr1e-5_repeat2.txt` |
| rev16 | UFMR | 3 | 123458 | 1 | `1e-5` | 0.152548 | 0.140712 | -0.011837 | -7.76 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/UFMR/rev16_epoch1_lr1e-5_repeat3.txt` |
| rev16 | RFM | 1 | 123456 | 1 | `1e-5` | 0.152681 | 0.142269 | -0.010412 | -6.82 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/RFM/rev16_epoch1_lr1e-5.txt` |
| rev16 | RFM | 2 | 123457 | 1 | `1e-5` | 0.152548 | 0.142826 | -0.009723 | -6.37 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/RFM/rev16_epoch1_lr1e-5_repeat2.txt` |
| rev16 | RFM | 3 | 123458 | 1 | `1e-5` | 0.152548 | 0.142499 | -0.010049 | -6.59 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/RFM/rev16_epoch1_lr1e-5_repeat3.txt` |
| TAL | UFMR | 1 | 123456 | 1 | `1e-5` | 0.139755 | 0.131779 | -0.007976 | -5.71 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/UFMR/TAL_epoch1_lr1e-5.txt` |
| TAL | UFMR | 2 | 123457 | 1 | `1e-5` | 0.139747 | 0.131554 | -0.008193 | -5.86 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/UFMR/TAL_epoch1_lr1e-5_repeat2.txt` |
| TAL | UFMR | 3 | 123458 | 1 | `1e-5` | 0.139750 | 0.134077 | -0.005673 | -4.06 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/UFMR/TAL_epoch1_lr1e-5_repeat3.txt` |
| TAL | RFM | 1 | 123456 | 1 | `1e-5` | 0.139755 | 0.132645 | -0.007111 | -5.09 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/RFM/TAL_epoch1_lr1e-5.txt` |
| TAL | RFM | 2 | 123457 | 1 | `1e-5` | 0.139750 | 0.133161 | -0.006589 | -4.71 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/RFM/TAL_epoch1_lr1e-5_repeat2.txt` |
| TAL | RFM | 3 | 123458 | 1 | `1e-5` | 0.139750 | 0.132715 | -0.007034 | -5.03 | complete | `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/results/RFM/TAL_epoch1_lr1e-5_repeat3.txt` |
