# ROB-337 Learnt Augmentation Repeat Eval

## Metadata

- Scope: UFMR, RFM, RMM, and UC-MLM/UVQLM test-set learnt augmentation WER cells.
- Datasets: TED-LIUM, Earnings-22, Rev16, TAL, CHiME-6; all `test` split.
- Adaptation: `epochs=1` and `epochs=5`, `lr=1e-5`, 6L/2048 ASR.
- Repeat policy: repeat 1 is the existing ROB-108 artifact; repeats 2 and 3 are ROB-337 Stanage jobs.
- Branch: `symphony/ROB-337-learnt-augmentation-repeats`
- Commit: `d8b3d5089c932083365f920c2aa9d6da1b126226`
- Finalizer log: `/mnt/parscratch/users/acp21rjf/symphony-workspaces-learning-to-augment/ROB-337/exp/results/repro/learnt_augmentation_repeats/rob337/logs/stanage/rob337_stanage_finalizer-10782851.log`
- Queued jobs: `/mnt/parscratch/users/acp21rjf/symphony-workspaces-learning-to-augment/ROB-337/exp/results/repro/learnt_augmentation_repeats/rob337/queued_jobs.tsv`
- Queued command: `REPO_DIR=/mnt/parscratch/users/acp21rjf/symphony-workspaces-learning-to-augment/ROB-337 scripts/submit_rob337_learnt_augmentation_repeats_stanage.sh`

Completed per-repeat rows: `120/120`.

## Aggregate

| Method | Dataset | Epochs | N | Repeats | Seeds | Mean Original WER | Mean Updated WER | Updated WER Std | Mean Rel Delta % |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| RFM | TAL | 1 | 3 | 1,2,3 | 123456,223456,323456 | 0.165693030 | 0.161937575 | 0.000065294 | -2.27 |
| RFM | TAL | 5 | 3 | 1,2,3 | 123456,223456,323456 | 0.165691150 | 0.161267494 | 0.000453137 | -2.67 |
| RFM | CHiME-6 | 1 | 3 | 1,2,3 | 123456,223456,323456 | 0.843602397 | 0.662689461 | 0.006161963 | -21.45 |
| RFM | CHiME-6 | 5 | 3 | 1,2,3 | 123456,223456,323456 | 0.843608272 | 0.649882505 | 0.005079710 | -22.96 |
| RFM | Earnings-22 | 1 | 3 | 1,2,3 | 123456,223456,323456 | 0.235225238 | 0.196011955 | 0.000479991 | -16.67 |
| RFM | Earnings-22 | 5 | 3 | 1,2,3 | 123456,223456,323456 | 0.235225238 | 0.184676865 | 0.001490969 | -21.49 |
| RFM | Rev16 | 1 | 3 | 1,2,3 | 123456,223456,323456 | 0.172551783 | 0.165018153 | 0.000506769 | -4.37 |
| RFM | Rev16 | 5 | 3 | 1,2,3 | 123456,223456,323456 | 0.172553485 | 0.163969641 | 0.000182001 | -4.97 |
| RFM | TED-LIUM | 1 | 3 | 1,2,3 | 123456,223456,323456 | 0.085344675 | 0.077795499 | 0.000557017 | -8.85 |
| RFM | TED-LIUM | 5 | 3 | 1,2,3 | 123456,223456,323456 | 0.085344675 | 0.075846181 | 0.000122775 | -11.13 |
| RMM | TAL | 1 | 3 | 1,2,3 | 123456,223456,323456 | 0.165689271 | 0.159919816 | 0.000185504 | -3.48 |
| RMM | TAL | 5 | 3 | 1,2,3 | 123456,223456,323456 | 0.165693030 | 0.155883359 | 0.000349224 | -5.92 |
| RMM | CHiME-6 | 1 | 3 | 1,2,3 | 123456,223456,323456 | 0.843608272 | 0.824015979 | 0.017933958 | -2.32 |
| RMM | CHiME-6 | 5 | 3 | 1,2,3 | 123456,223456,323456 | 0.843625896 | 1.000000000 | 0.000000000 | 18.54 |
| RMM | Earnings-22 | 1 | 3 | 1,2,3 | 123456,223456,323456 | 0.235225238 | 0.196740396 | 0.000991333 | -16.36 |
| RMM | Earnings-22 | 5 | 3 | 1,2,3 | 123456,223456,323456 | 0.235211622 | 0.183471873 | 0.000459569 | -22.00 |
| RMM | Rev16 | 1 | 3 | 1,2,3 | 123456,223456,323456 | 0.172551783 | 0.164504110 | 0.000383535 | -4.66 |
| RMM | Rev16 | 5 | 3 | 1,2,3 | 123456,223456,323456 | 0.172551783 | 0.195685440 | 0.061294397 | 13.41 |
| RMM | TED-LIUM | 1 | 3 | 1,2,3 | 123456,223456,323456 | 0.085344675 | 0.077712801 | 0.000266013 | -8.94 |
| RMM | TED-LIUM | 5 | 3 | 1,2,3 | 123456,223456,323456 | 0.085344675 | 0.074995570 | 0.000093771 | -12.13 |
| UFMR | TAL | 1 | 3 | 1,2,3 | 123456,223456,323456 | 0.165689271 | 0.158672697 | 0.000365441 | -4.23 |
| UFMR | TAL | 5 | 3 | 1,2,3 | 123456,223456,323456 | 0.165693030 | 0.162630210 | 0.000011395 | -1.85 |
| UFMR | CHiME-6 | 1 | 3 | 1,2,3 | 123456,223456,323456 | 0.843625896 | 0.637157796 | 0.006141169 | -24.47 |
| UFMR | CHiME-6 | 5 | 3 | 1,2,3 | 123456,223456,323456 | 0.843602397 | 0.620790741 | 0.002834949 | -26.41 |
| UFMR | Earnings-22 | 1 | 3 | 1,2,3 | 123456,223456,323456 | 0.235204815 | 0.187318315 | 0.001686688 | -20.36 |
| UFMR | Earnings-22 | 5 | 3 | 1,2,3 | 123456,223456,323456 | 0.235225238 | 0.185582311 | 0.001421507 | -21.10 |
| UFMR | Rev16 | 1 | 3 | 1,2,3 | 123456,223456,323456 | 0.172550081 | 0.162447936 | 0.000318253 | -5.85 |
| UFMR | Rev16 | 5 | 3 | 1,2,3 | 123456,223456,323456 | 0.172551783 | 0.163321129 | 0.000193527 | -5.35 |
| UFMR | TED-LIUM | 1 | 3 | 1,2,3 | 123456,223456,323456 | 0.085344675 | 0.076897631 | 0.000562999 | -9.90 |
| UFMR | TED-LIUM | 5 | 3 | 1,2,3 | 123456,223456,323456 | 0.085344675 | 0.076755863 | 0.000803789 | -10.06 |
| UC-MLM | TAL | 1 | 3 | 1,2,3 | 123456,223456,323456 | 0.165689271 | 0.159408563 | 0.000178233 | -3.79 |
| UC-MLM | TAL | 5 | 3 | 1,2,3 | 123456,223456,323456 | 0.165691150 | 0.155068549 | 0.000072010 | -6.41 |
| UC-MLM | CHiME-6 | 1 | 3 | 1,2,3 | 123456,223456,323456 | 0.843631770 | 0.819051815 | 0.014214622 | -2.91 |
| UC-MLM | CHiME-6 | 5 | 3 | 1,2,3 | 123456,223456,323456 | 0.843602397 | 0.948825050 | 0.088637613 | 12.47 |
| UC-MLM | Earnings-22 | 1 | 3 | 1,2,3 | 123456,223456,323456 | 0.235218430 | 0.195576251 | 0.000613728 | -16.85 |
| UC-MLM | Earnings-22 | 5 | 3 | 1,2,3 | 123456,223456,323456 | 0.235218430 | 0.183165520 | 0.000581026 | -22.13 |
| UC-MLM | Rev16 | 1 | 3 | 1,2,3 | 123456,223456,323456 | 0.172551783 | 0.163622406 | 0.000183072 | -5.17 |
| UC-MLM | Rev16 | 5 | 3 | 1,2,3 | 123456,223456,323456 | 0.172551783 | 0.159921974 | 0.000239348 | -7.32 |
| UC-MLM | TED-LIUM | 1 | 3 | 1,2,3 | 123456,223456,323456 | 0.085344675 | 0.076992144 | 0.000649987 | -9.79 |
| UC-MLM | TED-LIUM | 5 | 3 | 1,2,3 | 123456,223456,323456 | 0.085344675 | 0.074877429 | 0.000530844 | -12.26 |

## Missing Or Incomplete Cells

None. All affected cells have `N=3`.

## Per Repeat

| Method | Dataset | Epochs | Repeat | Seed | Original WER | Updated WER | Status | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| UFMR | TED-LIUM | 1 | 1 | 123456 | 0.085344675 | 0.076555024 | complete | `exp/results/repro/UFMR/tedlium_epoch1_lr1e-5.txt` |
| UFMR | TED-LIUM | 1 | 2 | 223456 | 0.085344675 | 0.076590466 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UFMR/tedlium_epoch1_lr1e-5_repeat2.txt` |
| UFMR | TED-LIUM | 1 | 3 | 323456 | 0.085344675 | 0.077547404 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UFMR/tedlium_epoch1_lr1e-5_repeat3.txt` |
| UFMR | TED-LIUM | 5 | 1 | 123456 | 0.085344675 | 0.075846181 | complete | `exp/results/repro/UFMR/tedlium_epoch5_lr1e-5.txt` |
| UFMR | TED-LIUM | 5 | 2 | 223456 | 0.085344675 | 0.077051214 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UFMR/tedlium_epoch5_lr1e-5_repeat2.txt` |
| UFMR | TED-LIUM | 5 | 3 | 323456 | 0.085344675 | 0.077370193 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UFMR/tedlium_epoch5_lr1e-5_repeat3.txt` |
| UFMR | Earnings-22 | 1 | 1 | 123456 | 0.235198007 | 0.185589118 | complete | `exp/results/repro/UFMR/earnings22_epoch1_lr1e-5.txt` |
| UFMR | Earnings-22 | 1 | 2 | 223456 | 0.235198007 | 0.188959010 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UFMR/earnings22_epoch1_lr1e-5_repeat2.txt` |
| UFMR | Earnings-22 | 1 | 3 | 323456 | 0.235218430 | 0.187406817 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UFMR/earnings22_epoch1_lr1e-5_repeat3.txt` |
| UFMR | Earnings-22 | 5 | 1 | 123456 | 0.235238854 | 0.186957499 | complete | `exp/results/repro/UFMR/earnings22_epoch5_lr1e-5.txt` |
| UFMR | Earnings-22 | 5 | 2 | 223456 | 0.235238854 | 0.185670813 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UFMR/earnings22_epoch5_lr1e-5_repeat2.txt` |
| UFMR | Earnings-22 | 5 | 3 | 323456 | 0.235198007 | 0.184118620 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UFMR/earnings22_epoch5_lr1e-5_repeat3.txt` |
| UFMR | Rev16 | 1 | 1 | 123456 | 0.172504123 | 0.162485383 | complete | `exp/results/repro/UFMR/rev16_epoch1_lr1e-5.txt` |
| UFMR | Rev16 | 1 | 2 | 223456 | 0.172570507 | 0.162112616 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UFMR/rev16_epoch1_lr1e-5_repeat2.txt` |
| UFMR | Rev16 | 1 | 3 | 323456 | 0.172575613 | 0.162745809 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UFMR/rev16_epoch1_lr1e-5_repeat3.txt` |
| UFMR | Rev16 | 5 | 1 | 123456 | 0.172509230 | 0.163113469 | complete | `exp/results/repro/UFMR/rev16_epoch5_lr1e-5.txt` |
| UFMR | Rev16 | 5 | 2 | 223456 | 0.172575613 | 0.163496449 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UFMR/rev16_epoch5_lr1e-5_repeat2.txt` |
| UFMR | Rev16 | 5 | 3 | 323456 | 0.172570507 | 0.163353470 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UFMR/rev16_epoch5_lr1e-5_repeat3.txt` |
| UFMR | TAL | 1 | 1 | 123456 | 0.165691150 | 0.158298655 | complete | `exp/results/repro/UFMR/TAL_epoch1_lr1e-5.txt` |
| UFMR | TAL | 1 | 2 | 223456 | 0.165691150 | 0.159028882 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UFMR/TAL_epoch1_lr1e-5_repeat2.txt` |
| UFMR | TAL | 1 | 3 | 323456 | 0.165685512 | 0.158690553 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UFMR/TAL_epoch1_lr1e-5_repeat3.txt` |
| UFMR | TAL | 5 | 1 | 123456 | 0.165696789 | 0.162617992 | complete | `exp/results/repro/UFMR/TAL_epoch5_lr1e-5.txt` |
| UFMR | TAL | 5 | 2 | 223456 | 0.165691150 | 0.162640548 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UFMR/TAL_epoch5_lr1e-5_repeat2.txt` |
| UFMR | TAL | 5 | 3 | 323456 | 0.165691150 | 0.162632089 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UFMR/TAL_epoch5_lr1e-5_repeat3.txt` |
| UFMR | CHiME-6 | 1 | 1 | 123456 | 0.843620021 | 0.642033839 | complete | `exp/results/repro/UFMR/chime6_epoch1_lr1e-5.txt` |
| UFMR | CHiME-6 | 1 | 2 | 223456 | 0.843637645 | 0.639178710 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UFMR/chime6_epoch1_lr1e-5_repeat2.txt` |
| UFMR | CHiME-6 | 1 | 3 | 323456 | 0.843620021 | 0.630260839 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UFMR/chime6_epoch1_lr1e-5_repeat3.txt` |
| UFMR | CHiME-6 | 5 | 1 | 123456 | 0.843584773 | 0.620232640 | complete | `exp/results/repro/UFMR/chime6_epoch5_lr1e-5.txt` |
| UFMR | CHiME-6 | 5 | 2 | 223456 | 0.843584773 | 0.623863236 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UFMR/chime6_epoch5_lr1e-5_repeat2.txt` |
| UFMR | CHiME-6 | 5 | 3 | 323456 | 0.843637645 | 0.618276348 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UFMR/chime6_epoch5_lr1e-5_repeat3.txt` |
| RFM | TED-LIUM | 1 | 1 | 123456 | 0.085344675 | 0.078291689 | complete | `exp/results/repro/RFM/tedlium_epoch1_lr1e-5.txt` |
| RFM | TED-LIUM | 1 | 2 | 223456 | 0.085344675 | 0.077192982 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RFM/tedlium_epoch1_lr1e-5_repeat2.txt` |
| RFM | TED-LIUM | 1 | 3 | 323456 | 0.085344675 | 0.077901825 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RFM/tedlium_epoch1_lr1e-5_repeat3.txt` |
| RFM | TED-LIUM | 5 | 1 | 123456 | 0.085344675 | 0.075917065 | complete | `exp/results/repro/RFM/tedlium_epoch5_lr1e-5.txt` |
| RFM | TED-LIUM | 5 | 2 | 223456 | 0.085344675 | 0.075704413 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RFM/tedlium_epoch5_lr1e-5_repeat2.txt` |
| RFM | TED-LIUM | 5 | 3 | 323456 | 0.085344675 | 0.075917065 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RFM/tedlium_epoch5_lr1e-5_repeat3.txt` |
| RFM | Earnings-22 | 1 | 1 | 123456 | 0.235238854 | 0.195535404 | complete | `exp/results/repro/RFM/earnings22_epoch1_lr1e-5.txt` |
| RFM | Earnings-22 | 1 | 2 | 223456 | 0.235238854 | 0.196005147 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RFM/earnings22_epoch1_lr1e-5_repeat2.txt` |
| RFM | Earnings-22 | 1 | 3 | 323456 | 0.235198007 | 0.196495313 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RFM/earnings22_epoch1_lr1e-5_repeat3.txt` |
| RFM | Earnings-22 | 5 | 1 | 123456 | 0.235218430 | 0.183179135 | complete | `exp/results/repro/RFM/earnings22_epoch5_lr1e-5.txt` |
| RFM | Earnings-22 | 5 | 2 | 223456 | 0.235218430 | 0.184690481 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RFM/earnings22_epoch5_lr1e-5_repeat2.txt` |
| RFM | Earnings-22 | 5 | 3 | 323456 | 0.235238854 | 0.186160979 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RFM/earnings22_epoch5_lr1e-5_repeat3.txt` |
| RFM | Rev16 | 1 | 1 | 123456 | 0.172509230 | 0.165059004 | complete | `exp/results/repro/RFM/rev16_epoch1_lr1e-5.txt` |
| RFM | Rev16 | 1 | 2 | 223456 | 0.172575613 | 0.165503260 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RFM/rev16_epoch1_lr1e-5_repeat2.txt` |
| RFM | Rev16 | 1 | 3 | 323456 | 0.172570507 | 0.164492195 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RFM/rev16_epoch1_lr1e-5_repeat3.txt` |
| RFM | Rev16 | 5 | 1 | 123456 | 0.172509230 | 0.164022407 | complete | `exp/results/repro/RFM/rev16_epoch5_lr1e-5.txt` |
| RFM | Rev16 | 5 | 2 | 223456 | 0.172575613 | 0.164119428 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RFM/rev16_epoch5_lr1e-5_repeat2.txt` |
| RFM | Rev16 | 5 | 3 | 323456 | 0.172575613 | 0.163767087 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RFM/rev16_epoch5_lr1e-5_repeat3.txt` |
| RFM | TAL | 1 | 1 | 123456 | 0.165696789 | 0.161865210 | complete | `exp/results/repro/RFM/TAL_epoch1_lr1e-5.txt` |
| RFM | TAL | 1 | 2 | 223456 | 0.165691150 | 0.161955431 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RFM/TAL_epoch1_lr1e-5_repeat2.txt` |
| RFM | TAL | 1 | 3 | 323456 | 0.165691150 | 0.161992083 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RFM/TAL_epoch1_lr1e-5_repeat3.txt` |
| RFM | TAL | 5 | 1 | 123456 | 0.165696789 | 0.161171634 | complete | `exp/results/repro/RFM/TAL_epoch5_lr1e-5.txt` |
| RFM | TAL | 5 | 2 | 223456 | 0.165685512 | 0.160869957 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RFM/TAL_epoch5_lr1e-5_repeat2.txt` |
| RFM | TAL | 5 | 3 | 323456 | 0.165691150 | 0.161760891 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RFM/TAL_epoch5_lr1e-5_repeat3.txt` |
| RFM | CHiME-6 | 1 | 1 | 123456 | 0.843584773 | 0.666936905 | complete | `exp/results/repro/RFM/chime6_epoch1_lr1e-5.txt` |
| RFM | CHiME-6 | 1 | 2 | 223456 | 0.843584773 | 0.665509341 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RFM/chime6_epoch1_lr1e-5_repeat2.txt` |
| RFM | CHiME-6 | 1 | 3 | 323456 | 0.843637645 | 0.655622136 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RFM/chime6_epoch1_lr1e-5_repeat3.txt` |
| RFM | CHiME-6 | 5 | 1 | 123456 | 0.843620021 | 0.655252027 | complete | `exp/results/repro/RFM/chime6_epoch5_lr1e-5.txt` |
| RFM | CHiME-6 | 5 | 2 | 223456 | 0.843620021 | 0.649242157 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RFM/chime6_epoch5_lr1e-5_repeat2.txt` |
| RFM | CHiME-6 | 5 | 3 | 323456 | 0.843584773 | 0.645153331 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RFM/chime6_epoch5_lr1e-5_repeat3.txt` |
| RMM | TED-LIUM | 1 | 1 | 123456 | 0.085344675 | 0.077724615 | complete | `exp/results/repro/RMM/tedlium_epoch1_lr1e-5.txt` |
| RMM | TED-LIUM | 1 | 2 | 223456 | 0.085344675 | 0.077972710 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RMM/tedlium_epoch1_lr1e-5_repeat2.txt` |
| RMM | TED-LIUM | 1 | 3 | 323456 | 0.085344675 | 0.077441077 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RMM/tedlium_epoch1_lr1e-5_repeat3.txt` |
| RMM | TED-LIUM | 5 | 1 | 123456 | 0.085344675 | 0.075031012 | complete | `exp/results/repro/RMM/tedlium_epoch5_lr1e-5.txt` |
| RMM | TED-LIUM | 5 | 2 | 223456 | 0.085344675 | 0.074889243 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RMM/tedlium_epoch5_lr1e-5_repeat2.txt` |
| RMM | TED-LIUM | 5 | 3 | 323456 | 0.085344675 | 0.075066454 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RMM/tedlium_epoch5_lr1e-5_repeat3.txt` |
| RMM | Earnings-22 | 1 | 1 | 123456 | 0.235218430 | 0.196209383 | complete | `exp/results/repro/RMM/earnings22_epoch1_lr1e-5.txt` |
| RMM | Earnings-22 | 1 | 2 | 223456 | 0.235218430 | 0.197884117 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RMM/earnings22_epoch1_lr1e-5_repeat2.txt` |
| RMM | Earnings-22 | 1 | 3 | 323456 | 0.235238854 | 0.196127688 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RMM/earnings22_epoch1_lr1e-5_repeat3.txt` |
| RMM | Earnings-22 | 5 | 1 | 123456 | 0.235218430 | 0.183996079 | complete | `exp/results/repro/RMM/earnings22_epoch5_lr1e-5.txt` |
| RMM | Earnings-22 | 5 | 2 | 223456 | 0.235198007 | 0.183281253 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RMM/earnings22_epoch5_lr1e-5_repeat2.txt` |
| RMM | Earnings-22 | 5 | 3 | 323456 | 0.235218430 | 0.183138288 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RMM/earnings22_epoch5_lr1e-5_repeat3.txt` |
| RMM | Rev16 | 1 | 1 | 123456 | 0.172504123 | 0.164109216 | complete | `exp/results/repro/RMM/rev16_epoch1_lr1e-5.txt` |
| RMM | Rev16 | 1 | 2 | 223456 | 0.172575613 | 0.164875174 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RMM/rev16_epoch1_lr1e-5_repeat2.txt` |
| RMM | Rev16 | 1 | 3 | 323456 | 0.172575613 | 0.164527940 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RMM/rev16_epoch1_lr1e-5_repeat3.txt` |
| RMM | Rev16 | 5 | 1 | 123456 | 0.172509230 | 0.266461730 | complete | `exp/results/repro/RMM/rev16_epoch5_lr1e-5.txt` |
| RMM | Rev16 | 5 | 2 | 223456 | 0.172570507 | 0.160095592 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RMM/rev16_epoch5_lr1e-5_repeat2.txt` |
| RMM | Rev16 | 5 | 3 | 323456 | 0.172575613 | 0.160498997 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RMM/rev16_epoch5_lr1e-5_repeat3.txt` |
| RMM | TAL | 1 | 1 | 123456 | 0.165691150 | 0.159962107 | complete | `exp/results/repro/RMM/TAL_epoch1_lr1e-5.txt` |
| RMM | TAL | 1 | 2 | 223456 | 0.165685512 | 0.159716818 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RMM/TAL_epoch1_lr1e-5_repeat2.txt` |
| RMM | TAL | 1 | 3 | 323456 | 0.165691150 | 0.160080522 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RMM/TAL_epoch1_lr1e-5_repeat3.txt` |
| RMM | TAL | 5 | 1 | 123456 | 0.165702428 | 0.155484882 | complete | `exp/results/repro/RMM/TAL_epoch5_lr1e-5.txt` |
| RMM | TAL | 5 | 2 | 223456 | 0.165691150 | 0.156136166 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RMM/TAL_epoch5_lr1e-5_repeat2.txt` |
| RMM | TAL | 5 | 3 | 323456 | 0.165685512 | 0.156029029 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RMM/TAL_epoch5_lr1e-5_repeat3.txt` |
| RMM | CHiME-6 | 1 | 1 | 123456 | 0.843620021 | 0.831670779 | complete | `exp/results/repro/RMM/chime6_epoch1_lr1e-5.txt` |
| RMM | CHiME-6 | 1 | 2 | 223456 | 0.843620021 | 0.836852309 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RMM/chime6_epoch1_lr1e-5_repeat2.txt` |
| RMM | CHiME-6 | 1 | 3 | 323456 | 0.843584773 | 0.803524850 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RMM/chime6_epoch1_lr1e-5_repeat3.txt` |
| RMM | CHiME-6 | 5 | 1 | 123456 | 0.843620021 | 1.000000000 | complete | `exp/results/repro/RMM/chime6_epoch5_lr1e-5.txt` |
| RMM | CHiME-6 | 5 | 2 | 223456 | 0.843637645 | 1.000000000 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RMM/chime6_epoch5_lr1e-5_repeat2.txt` |
| RMM | CHiME-6 | 5 | 3 | 323456 | 0.843620021 | 1.000000000 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/RMM/chime6_epoch5_lr1e-5_repeat3.txt` |
| UC-MLM | TED-LIUM | 1 | 1 | 123456 | 0.085344675 | 0.076767677 | complete | `exp/results/repro/UVQLM/tedlium_epoch1_lr1e-5.txt` |
| UC-MLM | TED-LIUM | 1 | 2 | 223456 | 0.085344675 | 0.076484140 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UVQLM/tedlium_epoch1_lr1e-5_repeat2.txt` |
| UC-MLM | TED-LIUM | 1 | 3 | 323456 | 0.085344675 | 0.077724615 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UVQLM/tedlium_epoch1_lr1e-5_repeat3.txt` |
| UC-MLM | TED-LIUM | 5 | 1 | 123456 | 0.085344675 | 0.074286727 | complete | `exp/results/repro/UVQLM/tedlium_epoch5_lr1e-5.txt` |
| UC-MLM | TED-LIUM | 5 | 2 | 223456 | 0.085344675 | 0.075314549 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UVQLM/tedlium_epoch5_lr1e-5_repeat2.txt` |
| UC-MLM | TED-LIUM | 5 | 3 | 323456 | 0.085344675 | 0.075031012 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UVQLM/tedlium_epoch5_lr1e-5_repeat3.txt` |
| UC-MLM | Earnings-22 | 1 | 1 | 123456 | 0.235238854 | 0.195535404 | complete | `exp/results/repro/UVQLM/earnings22_epoch1_lr1e-5.txt` |
| UC-MLM | Earnings-22 | 1 | 2 | 223456 | 0.235198007 | 0.196209383 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UVQLM/earnings22_epoch1_lr1e-5_repeat2.txt` |
| UC-MLM | Earnings-22 | 1 | 3 | 323456 | 0.235218430 | 0.194983967 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UVQLM/earnings22_epoch1_lr1e-5_repeat3.txt` |
| UC-MLM | Earnings-22 | 5 | 1 | 123456 | 0.235218430 | 0.182893205 | complete | `exp/results/repro/UVQLM/earnings22_epoch5_lr1e-5.txt` |
| UC-MLM | Earnings-22 | 5 | 2 | 223456 | 0.235238854 | 0.183832690 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UVQLM/earnings22_epoch5_lr1e-5_repeat2.txt` |
| UC-MLM | Earnings-22 | 5 | 3 | 323456 | 0.235198007 | 0.182770664 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UVQLM/earnings22_epoch5_lr1e-5_repeat3.txt` |
| UC-MLM | Rev16 | 1 | 1 | 123456 | 0.172509230 | 0.163833470 | complete | `exp/results/repro/UVQLM/rev16_epoch1_lr1e-5.txt` |
| UC-MLM | Rev16 | 1 | 2 | 223456 | 0.172570507 | 0.163506661 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UVQLM/rev16_epoch1_lr1e-5_repeat2.txt` |
| UC-MLM | Rev16 | 1 | 3 | 323456 | 0.172575613 | 0.163527087 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UVQLM/rev16_epoch1_lr1e-5_repeat3.txt` |
| UC-MLM | Rev16 | 5 | 1 | 123456 | 0.172509230 | 0.159656442 | complete | `exp/results/repro/UVQLM/rev16_epoch5_lr1e-5.txt` |
| UC-MLM | Rev16 | 5 | 2 | 223456 | 0.172575613 | 0.159988357 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UVQLM/rev16_epoch5_lr1e-5_repeat2.txt` |
| UC-MLM | Rev16 | 5 | 3 | 323456 | 0.172570507 | 0.160121124 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UVQLM/rev16_epoch5_lr1e-5_repeat3.txt` |
| UC-MLM | TAL | 1 | 1 | 123456 | 0.165691150 | 0.159612500 | complete | `exp/results/repro/UVQLM/TAL_epoch1_lr1e-5.txt` |
| UC-MLM | TAL | 1 | 2 | 223456 | 0.165691150 | 0.159282629 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UVQLM/TAL_epoch1_lr1e-5_repeat2.txt` |
| UC-MLM | TAL | 1 | 3 | 323456 | 0.165685512 | 0.159330559 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UVQLM/TAL_epoch1_lr1e-5_repeat3.txt` |
| UC-MLM | TAL | 5 | 1 | 123456 | 0.165691150 | 0.155045054 | complete | `exp/results/repro/UVQLM/TAL_epoch5_lr1e-5.txt` |
| UC-MLM | TAL | 5 | 2 | 223456 | 0.165691150 | 0.155149372 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UVQLM/TAL_epoch5_lr1e-5_repeat2.txt` |
| UC-MLM | TAL | 5 | 3 | 323456 | 0.165691150 | 0.155011221 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UVQLM/TAL_epoch5_lr1e-5_repeat3.txt` |
| UC-MLM | CHiME-6 | 1 | 1 | 123456 | 0.843637645 | 0.834543532 | complete | `exp/results/repro/UVQLM/chime6_epoch1_lr1e-5.txt` |
| UC-MLM | CHiME-6 | 1 | 2 | 223456 | 0.843637645 | 0.816002820 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UVQLM/chime6_epoch1_lr1e-5_repeat2.txt` |
| UC-MLM | CHiME-6 | 1 | 3 | 323456 | 0.843620021 | 0.806609094 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UVQLM/chime6_epoch1_lr1e-5_repeat3.txt` |
| UC-MLM | CHiME-6 | 5 | 1 | 123456 | 0.843584773 | 1.000000000 | complete | `exp/results/repro/UVQLM/chime6_epoch5_lr1e-5.txt` |
| UC-MLM | CHiME-6 | 5 | 2 | 223456 | 0.843584773 | 1.000000000 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UVQLM/chime6_epoch5_lr1e-5_repeat2.txt` |
| UC-MLM | CHiME-6 | 5 | 3 | 323456 | 0.843637645 | 0.846475150 | complete | `exp/results/repro/learnt_augmentation_repeats/rob337/UVQLM/chime6_epoch5_lr1e-5_repeat3.txt` |

CSV artifacts:

```text
/mnt/parscratch/users/acp21rjf/symphony-workspaces-learning-to-augment/ROB-337/exp/results/repro/learnt_augmentation_repeats/rob337/rob337_learnt_augmentation_repeats.csv
/mnt/parscratch/users/acp21rjf/symphony-workspaces-learning-to-augment/ROB-337/exp/results/repro/learnt_augmentation_repeats/rob337/rob337_learnt_augmentation_repeats_aggregate.csv
```
