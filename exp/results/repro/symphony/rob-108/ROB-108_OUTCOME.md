# ROB-108 Test Policy Evaluations

Test split policy evals for RFM, RMM, UFMR, and UVQLM across TED-LIUM, Earnings22, CHiME-6, Rev16, and This American Life. NoAug rows are unadapted baselines.

Completed cells: 65/65

The `repeat` column is retained even though ROB-108 starts with one repeat, so the table can be extended without changing schema.

## Aggregate

| Dataset | Method | Epochs | LR | N | Mean Original WER | Mean Updated WER | Updated WER Std | Mean Abs Delta | Mean Rel Delta % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TAL | NoAug | 0 | `baseline` | 1 | 0.165691 | 0.165691 | 0.000000 | 0.000000 | 0.00 |
| TAL | RFM | 1 | `1e-5` | 1 | 0.165697 | 0.161865 | 0.000000 | -0.003832 | -2.31 |
| TAL | RFM | 1 | `3e-5` | 1 | 0.165694 | 0.164538 | 0.000000 | -0.001156 | -0.70 |
| TAL | RFM | 5 | `1e-5` | 1 | 0.165697 | 0.161172 | 0.000000 | -0.004525 | -2.73 |
| TAL | RMM | 1 | `1e-5` | 1 | 0.165691 | 0.159962 | 0.000000 | -0.005729 | -3.46 |
| TAL | RMM | 1 | `3e-5` | 1 | 0.165700 | 0.161783 | 0.000000 | -0.003917 | -2.36 |
| TAL | RMM | 5 | `1e-5` | 1 | 0.165702 | 0.155485 | 0.000000 | -0.010217 | -6.17 |
| TAL | UFMR | 1 | `1e-5` | 1 | 0.165691 | 0.158299 | 0.000000 | -0.007392 | -4.46 |
| TAL | UFMR | 1 | `3e-5` | 1 | 0.165691 | 0.161679 | 0.000000 | -0.004012 | -2.42 |
| TAL | UFMR | 5 | `1e-5` | 1 | 0.165697 | 0.162618 | 0.000000 | -0.003079 | -1.86 |
| TAL | UVQLM | 1 | `1e-5` | 1 | 0.165691 | 0.159613 | 0.000000 | -0.006078 | -3.67 |
| TAL | UVQLM | 1 | `3e-5` | 1 | 0.165691 | 0.161978 | 0.000000 | -0.003713 | -2.24 |
| TAL | UVQLM | 5 | `1e-5` | 1 | 0.165691 | 0.155045 | 0.000000 | -0.010646 | -6.43 |
| chime6 | NoAug | 0 | `baseline` | 1 | 0.843620 | 0.843620 | 0.000000 | 0.000000 | 0.00 |
| chime6 | RFM | 1 | `1e-5` | 1 | 0.843585 | 0.666937 | 0.000000 | -0.176648 | -20.94 |
| chime6 | RFM | 1 | `3e-5` | 1 | 0.843620 | 0.803401 | 0.000000 | -0.040219 | -4.77 |
| chime6 | RFM | 5 | `1e-5` | 1 | 0.843620 | 0.655252 | 0.000000 | -0.188368 | -22.33 |
| chime6 | RMM | 1 | `1e-5` | 1 | 0.843620 | 0.831671 | 0.000000 | -0.011949 | -1.42 |
| chime6 | RMM | 1 | `3e-5` | 1 | 0.843620 | 1.000000 | 0.000000 | 0.156380 | 18.54 |
| chime6 | RMM | 5 | `1e-5` | 1 | 0.843620 | 1.000000 | 0.000000 | 0.156380 | 18.54 |
| chime6 | UFMR | 1 | `1e-5` | 1 | 0.843620 | 0.642034 | 0.000000 | -0.201586 | -23.90 |
| chime6 | UFMR | 1 | `3e-5` | 1 | 0.843638 | 0.644889 | 0.000000 | -0.198749 | -23.56 |
| chime6 | UFMR | 5 | `1e-5` | 1 | 0.843585 | 0.620233 | 0.000000 | -0.223352 | -26.48 |
| chime6 | UVQLM | 1 | `1e-5` | 1 | 0.843638 | 0.834544 | 0.000000 | -0.009094 | -1.08 |
| chime6 | UVQLM | 1 | `3e-5` | 1 | 0.843620 | 0.857984 | 0.000000 | 0.014364 | 1.70 |
| chime6 | UVQLM | 5 | `1e-5` | 1 | 0.843585 | 1.000000 | 0.000000 | 0.156415 | 18.54 |
| earnings22 | NoAug | 0 | `baseline` | 1 | 0.235198 | 0.235198 | 0.000000 | 0.000000 | 0.00 |
| earnings22 | RFM | 1 | `1e-5` | 1 | 0.235239 | 0.195535 | 0.000000 | -0.039704 | -16.88 |
| earnings22 | RFM | 1 | `3e-5` | 1 | 0.235218 | 0.197721 | 0.000000 | -0.037497 | -15.94 |
| earnings22 | RFM | 5 | `1e-5` | 1 | 0.235218 | 0.183179 | 0.000000 | -0.052039 | -22.12 |
| earnings22 | RMM | 1 | `1e-5` | 1 | 0.235218 | 0.196209 | 0.000000 | -0.039009 | -16.58 |
| earnings22 | RMM | 1 | `3e-5` | 1 | 0.235218 | 0.198374 | 0.000000 | -0.036844 | -15.66 |
| earnings22 | RMM | 5 | `1e-5` | 1 | 0.235218 | 0.183996 | 0.000000 | -0.051222 | -21.78 |
| earnings22 | UFMR | 1 | `1e-5` | 1 | 0.235198 | 0.185589 | 0.000000 | -0.049609 | -21.09 |
| earnings22 | UFMR | 1 | `3e-5` | 1 | 0.235218 | 0.187877 | 0.000000 | -0.047341 | -20.13 |
| earnings22 | UFMR | 5 | `1e-5` | 1 | 0.235239 | 0.186957 | 0.000000 | -0.048282 | -20.52 |
| earnings22 | UVQLM | 1 | `1e-5` | 1 | 0.235239 | 0.195535 | 0.000000 | -0.039704 | -16.88 |
| earnings22 | UVQLM | 1 | `3e-5` | 1 | 0.235198 | 0.198129 | 0.000000 | -0.037069 | -15.76 |
| earnings22 | UVQLM | 5 | `1e-5` | 1 | 0.235218 | 0.182893 | 0.000000 | -0.052325 | -22.25 |
| rev16 | NoAug | 0 | `baseline` | 1 | 0.172514 | 0.172514 | 0.000000 | 0.000000 | 0.00 |
| rev16 | RFM | 1 | `1e-5` | 1 | 0.172509 | 0.165059 | 0.000000 | -0.007450 | -4.32 |
| rev16 | RFM | 1 | `3e-5` | 1 | 0.172509 | 0.168291 | 0.000000 | -0.004218 | -2.45 |
| rev16 | RFM | 5 | `1e-5` | 1 | 0.172509 | 0.164022 | 0.000000 | -0.008487 | -4.92 |
| rev16 | RMM | 1 | `1e-5` | 1 | 0.172504 | 0.164109 | 0.000000 | -0.008395 | -4.87 |
| rev16 | RMM | 1 | `3e-5` | 1 | 0.172509 | 0.167903 | 0.000000 | -0.004606 | -2.67 |
| rev16 | RMM | 5 | `1e-5` | 1 | 0.172509 | 0.266462 | 0.000000 | 0.093953 | 54.46 |
| rev16 | UFMR | 1 | `1e-5` | 1 | 0.172504 | 0.162485 | 0.000000 | -0.010019 | -5.81 |
| rev16 | UFMR | 1 | `3e-5` | 1 | 0.172514 | 0.164896 | 0.000000 | -0.007618 | -4.42 |
| rev16 | UFMR | 5 | `1e-5` | 1 | 0.172509 | 0.163113 | 0.000000 | -0.009396 | -5.45 |
| rev16 | UVQLM | 1 | `1e-5` | 1 | 0.172509 | 0.163833 | 0.000000 | -0.008676 | -5.03 |
| rev16 | UVQLM | 1 | `3e-5` | 1 | 0.172509 | 0.167822 | 0.000000 | -0.004687 | -2.72 |
| rev16 | UVQLM | 5 | `1e-5` | 1 | 0.172509 | 0.159656 | 0.000000 | -0.012853 | -7.45 |
| tedlium | NoAug | 0 | `baseline` | 1 | 0.085345 | 0.085345 | 0.000000 | 0.000000 | 0.00 |
| tedlium | RFM | 1 | `1e-5` | 1 | 0.085345 | 0.078292 | 0.000000 | -0.007053 | -8.26 |
| tedlium | RFM | 1 | `3e-5` | 1 | 0.085345 | 0.078575 | 0.000000 | -0.006770 | -7.93 |
| tedlium | RFM | 5 | `1e-5` | 1 | 0.085345 | 0.075917 | 0.000000 | -0.009428 | -11.05 |
| tedlium | RMM | 1 | `1e-5` | 1 | 0.085345 | 0.077725 | 0.000000 | -0.007620 | -8.93 |
| tedlium | RMM | 1 | `3e-5` | 1 | 0.085345 | 0.077158 | 0.000000 | -0.008187 | -9.59 |
| tedlium | RMM | 5 | `1e-5` | 1 | 0.085345 | 0.075031 | 0.000000 | -0.010314 | -12.09 |
| tedlium | UFMR | 1 | `1e-5` | 1 | 0.085345 | 0.076555 | 0.000000 | -0.008790 | -10.30 |
| tedlium | UFMR | 1 | `3e-5` | 1 | 0.085345 | 0.075173 | 0.000000 | -0.010172 | -11.92 |
| tedlium | UFMR | 5 | `1e-5` | 1 | 0.085345 | 0.075846 | 0.000000 | -0.009499 | -11.13 |
| tedlium | UVQLM | 1 | `1e-5` | 1 | 0.085345 | 0.076768 | 0.000000 | -0.008577 | -10.05 |
| tedlium | UVQLM | 1 | `3e-5` | 1 | 0.085345 | 0.076874 | 0.000000 | -0.008471 | -9.93 |
| tedlium | UVQLM | 5 | `1e-5` | 1 | 0.085345 | 0.074287 | 0.000000 | -0.011058 | -12.96 |

## Per Repeat

| Dataset | Method | Repeat | Seed | Epochs | LR | Original WER | Updated WER | Abs Delta | Rel Delta % | Status | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| tedlium | NoAug | 1 | 123456 | 0 | `baseline` | 0.085345 | 0.085345 | 0.000000 | 0.00 | complete | `exp/results/repro/NoAug/tedlium_baseline.txt` |
| tedlium | RFM | 1 | 123456 | 1 | `1e-5` | 0.085345 | 0.078292 | -0.007053 | -8.26 | complete | `exp/results/repro/RFM/tedlium_epoch1_lr1e-5.txt` |
| tedlium | RFM | 1 | 123456 | 1 | `3e-5` | 0.085345 | 0.078575 | -0.006769 | -7.93 | complete | `exp/results/repro/RFM/tedlium_epoch1_lr3e-5.txt` |
| tedlium | RFM | 1 | 123456 | 5 | `1e-5` | 0.085345 | 0.075917 | -0.009428 | -11.05 | complete | `exp/results/repro/RFM/tedlium_epoch5_lr1e-5.txt` |
| tedlium | RMM | 1 | 123456 | 1 | `1e-5` | 0.085345 | 0.077725 | -0.007620 | -8.93 | complete | `exp/results/repro/RMM/tedlium_epoch1_lr1e-5.txt` |
| tedlium | RMM | 1 | 123456 | 1 | `3e-5` | 0.085345 | 0.077158 | -0.008187 | -9.59 | complete | `exp/results/repro/RMM/tedlium_epoch1_lr3e-5.txt` |
| tedlium | RMM | 1 | 123456 | 5 | `1e-5` | 0.085345 | 0.075031 | -0.010314 | -12.08 | complete | `exp/results/repro/RMM/tedlium_epoch5_lr1e-5.txt` |
| tedlium | UFMR | 1 | 123456 | 1 | `1e-5` | 0.085345 | 0.076555 | -0.008790 | -10.30 | complete | `exp/results/repro/UFMR/tedlium_epoch1_lr1e-5.txt` |
| tedlium | UFMR | 1 | 123456 | 1 | `3e-5` | 0.085345 | 0.075173 | -0.010172 | -11.92 | complete | `exp/results/repro/UFMR/tedlium_epoch1_lr3e-5.txt` |
| tedlium | UFMR | 1 | 123456 | 5 | `1e-5` | 0.085345 | 0.075846 | -0.009498 | -11.13 | complete | `exp/results/repro/UFMR/tedlium_epoch5_lr1e-5.txt` |
| tedlium | UVQLM | 1 | 123456 | 1 | `1e-5` | 0.085345 | 0.076768 | -0.008577 | -10.05 | complete | `exp/results/repro/UVQLM/tedlium_epoch1_lr1e-5.txt` |
| tedlium | UVQLM | 1 | 123456 | 1 | `3e-5` | 0.085345 | 0.076874 | -0.008471 | -9.93 | complete | `exp/results/repro/UVQLM/tedlium_epoch1_lr3e-5.txt` |
| tedlium | UVQLM | 1 | 123456 | 5 | `1e-5` | 0.085345 | 0.074287 | -0.011058 | -12.96 | complete | `exp/results/repro/UVQLM/tedlium_epoch5_lr1e-5.txt` |
| earnings22 | NoAug | 1 | 123456 | 0 | `baseline` | 0.235198 | 0.235198 | 0.000000 | 0.00 | complete | `exp/results/repro/NoAug/earnings22_baseline.txt` |
| earnings22 | RFM | 1 | 123456 | 1 | `1e-5` | 0.235239 | 0.195535 | -0.039703 | -16.88 | complete | `exp/results/repro/RFM/earnings22_epoch1_lr1e-5.txt` |
| earnings22 | RFM | 1 | 123456 | 1 | `3e-5` | 0.235218 | 0.197721 | -0.037498 | -15.94 | complete | `exp/results/repro/RFM/earnings22_epoch1_lr3e-5.txt` |
| earnings22 | RFM | 1 | 123456 | 5 | `1e-5` | 0.235218 | 0.183179 | -0.052039 | -22.12 | complete | `exp/results/repro/RFM/earnings22_epoch5_lr1e-5.txt` |
| earnings22 | RMM | 1 | 123456 | 1 | `1e-5` | 0.235218 | 0.196209 | -0.039009 | -16.58 | complete | `exp/results/repro/RMM/earnings22_epoch1_lr1e-5.txt` |
| earnings22 | RMM | 1 | 123456 | 1 | `3e-5` | 0.235218 | 0.198374 | -0.036844 | -15.66 | complete | `exp/results/repro/RMM/earnings22_epoch1_lr3e-5.txt` |
| earnings22 | RMM | 1 | 123456 | 5 | `1e-5` | 0.235218 | 0.183996 | -0.051222 | -21.78 | complete | `exp/results/repro/RMM/earnings22_epoch5_lr1e-5.txt` |
| earnings22 | UFMR | 1 | 123456 | 1 | `1e-5` | 0.235198 | 0.185589 | -0.049609 | -21.09 | complete | `exp/results/repro/UFMR/earnings22_epoch1_lr1e-5.txt` |
| earnings22 | UFMR | 1 | 123456 | 1 | `3e-5` | 0.235218 | 0.187877 | -0.047342 | -20.13 | complete | `exp/results/repro/UFMR/earnings22_epoch1_lr3e-5.txt` |
| earnings22 | UFMR | 1 | 123456 | 5 | `1e-5` | 0.235239 | 0.186957 | -0.048281 | -20.52 | complete | `exp/results/repro/UFMR/earnings22_epoch5_lr1e-5.txt` |
| earnings22 | UVQLM | 1 | 123456 | 1 | `1e-5` | 0.235239 | 0.195535 | -0.039703 | -16.88 | complete | `exp/results/repro/UVQLM/earnings22_epoch1_lr1e-5.txt` |
| earnings22 | UVQLM | 1 | 123456 | 1 | `3e-5` | 0.235198 | 0.198129 | -0.037069 | -15.76 | complete | `exp/results/repro/UVQLM/earnings22_epoch1_lr3e-5.txt` |
| earnings22 | UVQLM | 1 | 123456 | 5 | `1e-5` | 0.235218 | 0.182893 | -0.052325 | -22.25 | complete | `exp/results/repro/UVQLM/earnings22_epoch5_lr1e-5.txt` |
| chime6 | NoAug | 1 | 123456 | 0 | `baseline` | 0.843620 | 0.843620 | 0.000000 | 0.00 | complete | `exp/results/repro/NoAug/chime6_baseline.txt` |
| chime6 | RFM | 1 | 123456 | 1 | `1e-5` | 0.843585 | 0.666937 | -0.176648 | -20.94 | complete | `exp/results/repro/RFM/chime6_epoch1_lr1e-5.txt` |
| chime6 | RFM | 1 | 123456 | 1 | `3e-5` | 0.843620 | 0.803401 | -0.040219 | -4.77 | complete | `exp/results/repro/RFM/chime6_epoch1_lr3e-5.txt` |
| chime6 | RFM | 1 | 123456 | 5 | `1e-5` | 0.843620 | 0.655252 | -0.188368 | -22.33 | complete | `exp/results/repro/RFM/chime6_epoch5_lr1e-5.txt` |
| chime6 | RMM | 1 | 123456 | 1 | `1e-5` | 0.843620 | 0.831671 | -0.011949 | -1.42 | complete | `exp/results/repro/RMM/chime6_epoch1_lr1e-5.txt` |
| chime6 | RMM | 1 | 123456 | 1 | `3e-5` | 0.843620 | 1.000000 | 0.156380 | 18.54 | complete | `exp/results/repro/RMM/chime6_epoch1_lr3e-5.txt` |
| chime6 | RMM | 1 | 123456 | 5 | `1e-5` | 0.843620 | 1.000000 | 0.156380 | 18.54 | complete | `exp/results/repro/RMM/chime6_epoch5_lr1e-5.txt` |
| chime6 | UFMR | 1 | 123456 | 1 | `1e-5` | 0.843620 | 0.642034 | -0.201586 | -23.90 | complete | `exp/results/repro/UFMR/chime6_epoch1_lr1e-5.txt` |
| chime6 | UFMR | 1 | 123456 | 1 | `3e-5` | 0.843638 | 0.644889 | -0.198749 | -23.56 | complete | `exp/results/repro/UFMR/chime6_epoch1_lr3e-5.txt` |
| chime6 | UFMR | 1 | 123456 | 5 | `1e-5` | 0.843585 | 0.620233 | -0.223352 | -26.48 | complete | `exp/results/repro/UFMR/chime6_epoch5_lr1e-5.txt` |
| chime6 | UVQLM | 1 | 123456 | 1 | `1e-5` | 0.843638 | 0.834544 | -0.009094 | -1.08 | complete | `exp/results/repro/UVQLM/chime6_epoch1_lr1e-5.txt` |
| chime6 | UVQLM | 1 | 123456 | 1 | `3e-5` | 0.843620 | 0.857984 | 0.014364 | 1.70 | complete | `exp/results/repro/UVQLM/chime6_epoch1_lr3e-5.txt` |
| chime6 | UVQLM | 1 | 123456 | 5 | `1e-5` | 0.843585 | 1.000000 | 0.156415 | 18.54 | complete | `exp/results/repro/UVQLM/chime6_epoch5_lr1e-5.txt` |
| rev16 | NoAug | 1 | 123456 | 0 | `baseline` | 0.172514 | 0.172514 | 0.000000 | 0.00 | complete | `exp/results/repro/NoAug/rev16_baseline.txt` |
| rev16 | RFM | 1 | 123456 | 1 | `1e-5` | 0.172509 | 0.165059 | -0.007450 | -4.32 | complete | `exp/results/repro/RFM/rev16_epoch1_lr1e-5.txt` |
| rev16 | RFM | 1 | 123456 | 1 | `3e-5` | 0.172509 | 0.168291 | -0.004218 | -2.45 | complete | `exp/results/repro/RFM/rev16_epoch1_lr3e-5.txt` |
| rev16 | RFM | 1 | 123456 | 5 | `1e-5` | 0.172509 | 0.164022 | -0.008487 | -4.92 | complete | `exp/results/repro/RFM/rev16_epoch5_lr1e-5.txt` |
| rev16 | RMM | 1 | 123456 | 1 | `1e-5` | 0.172504 | 0.164109 | -0.008395 | -4.87 | complete | `exp/results/repro/RMM/rev16_epoch1_lr1e-5.txt` |
| rev16 | RMM | 1 | 123456 | 1 | `3e-5` | 0.172509 | 0.167903 | -0.004606 | -2.67 | complete | `exp/results/repro/RMM/rev16_epoch1_lr3e-5.txt` |
| rev16 | RMM | 1 | 123456 | 5 | `1e-5` | 0.172509 | 0.266462 | 0.093953 | 54.46 | complete | `exp/results/repro/RMM/rev16_epoch5_lr1e-5.txt` |
| rev16 | UFMR | 1 | 123456 | 1 | `1e-5` | 0.172504 | 0.162485 | -0.010019 | -5.81 | complete | `exp/results/repro/UFMR/rev16_epoch1_lr1e-5.txt` |
| rev16 | UFMR | 1 | 123456 | 1 | `3e-5` | 0.172514 | 0.164896 | -0.007619 | -4.42 | complete | `exp/results/repro/UFMR/rev16_epoch1_lr3e-5.txt` |
| rev16 | UFMR | 1 | 123456 | 5 | `1e-5` | 0.172509 | 0.163113 | -0.009396 | -5.45 | complete | `exp/results/repro/UFMR/rev16_epoch5_lr1e-5.txt` |
| rev16 | UVQLM | 1 | 123456 | 1 | `1e-5` | 0.172509 | 0.163833 | -0.008676 | -5.03 | complete | `exp/results/repro/UVQLM/rev16_epoch1_lr1e-5.txt` |
| rev16 | UVQLM | 1 | 123456 | 1 | `3e-5` | 0.172509 | 0.167822 | -0.004688 | -2.72 | complete | `exp/results/repro/UVQLM/rev16_epoch1_lr3e-5.txt` |
| rev16 | UVQLM | 1 | 123456 | 5 | `1e-5` | 0.172509 | 0.159656 | -0.012853 | -7.45 | complete | `exp/results/repro/UVQLM/rev16_epoch5_lr1e-5.txt` |
| TAL | NoAug | 1 | 123456 | 0 | `baseline` | 0.165691 | 0.165691 | 0.000000 | 0.00 | complete | `exp/results/repro/NoAug/TAL_baseline.txt` |
| TAL | RFM | 1 | 123456 | 1 | `1e-5` | 0.165697 | 0.161865 | -0.003832 | -2.31 | complete | `exp/results/repro/RFM/TAL_epoch1_lr1e-5.txt` |
| TAL | RFM | 1 | 123456 | 1 | `3e-5` | 0.165694 | 0.164538 | -0.001156 | -0.70 | complete | `exp/results/repro/RFM/TAL_epoch1_lr3e-5.txt` |
| TAL | RFM | 1 | 123456 | 5 | `1e-5` | 0.165697 | 0.161172 | -0.004525 | -2.73 | complete | `exp/results/repro/RFM/TAL_epoch5_lr1e-5.txt` |
| TAL | RMM | 1 | 123456 | 1 | `1e-5` | 0.165691 | 0.159962 | -0.005729 | -3.46 | complete | `exp/results/repro/RMM/TAL_epoch1_lr1e-5.txt` |
| TAL | RMM | 1 | 123456 | 1 | `3e-5` | 0.165700 | 0.161783 | -0.003916 | -2.36 | complete | `exp/results/repro/RMM/TAL_epoch1_lr3e-5.txt` |
| TAL | RMM | 1 | 123456 | 5 | `1e-5` | 0.165702 | 0.155485 | -0.010218 | -6.17 | complete | `exp/results/repro/RMM/TAL_epoch5_lr1e-5.txt` |
| TAL | UFMR | 1 | 123456 | 1 | `1e-5` | 0.165691 | 0.158299 | -0.007392 | -4.46 | complete | `exp/results/repro/UFMR/TAL_epoch1_lr1e-5.txt` |
| TAL | UFMR | 1 | 123456 | 1 | `3e-5` | 0.165691 | 0.161679 | -0.004012 | -2.42 | complete | `exp/results/repro/UFMR/TAL_epoch1_lr3e-5.txt` |
| TAL | UFMR | 1 | 123456 | 5 | `1e-5` | 0.165697 | 0.162618 | -0.003079 | -1.86 | complete | `exp/results/repro/UFMR/TAL_epoch5_lr1e-5.txt` |
| TAL | UVQLM | 1 | 123456 | 1 | `1e-5` | 0.165691 | 0.159613 | -0.006079 | -3.67 | complete | `exp/results/repro/UVQLM/TAL_epoch1_lr1e-5.txt` |
| TAL | UVQLM | 1 | 123456 | 1 | `3e-5` | 0.165691 | 0.161978 | -0.003713 | -2.24 | complete | `exp/results/repro/UVQLM/TAL_epoch1_lr3e-5.txt` |
| TAL | UVQLM | 1 | 123456 | 5 | `1e-5` | 0.165691 | 0.155045 | -0.010646 | -6.43 | complete | `exp/results/repro/UVQLM/TAL_epoch5_lr1e-5.txt` |
