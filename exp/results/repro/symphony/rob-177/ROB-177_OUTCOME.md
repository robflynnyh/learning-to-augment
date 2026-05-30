# ROB-177 UFMR Candidate-Repeat Ablation

Earnings22 test-set UFMR ablation varying the number of randomly sampled frequency masks scored by the UFMR ranker per adaptation step.

All ROB-177 ablation rows use UFMR, epoch `1`, adaptation LR `1e-5`, the 2048-sequence ASR checkpoint, and `use_random: false`.
The `candidate_repeats` column is the UFMR mask-candidate count in `evaluation.augmentation_config.repeats`; it is not a seed repeat.
Each ROB-177 candidate-repeat setting has three seed trials when complete.

Completed rows: 25/25
Completed ROB-177 ablation rows: 24/24

## Per-Trial Results

| Candidate Repeats | Trial | Seed | Source | Original WER | Updated WER | Abs Delta | Rel Delta % | Status | Result |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 2 | 1 | 123456 | ROB-177 ablation | 0.235218 | 0.192002 | -0.043216 | -18.37 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats2_epoch1_lr1e-5.txt` |
| 2 | 2 | 123457 | ROB-177 ablation | 0.235239 | 0.193248 | -0.041991 | -17.85 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats2_seed123457_epoch1_lr1e-5.txt` |
| 2 | 3 | 123458 | ROB-177 ablation | 0.235218 | 0.192166 | -0.043053 | -18.30 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats2_seed123458_epoch1_lr1e-5.txt` |
| 5 | 1 | 123456 | ROB-177 ablation | 0.235239 | 0.187203 | -0.048036 | -20.42 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats5_epoch1_lr1e-5.txt` |
| 5 | 2 | 123457 | ROB-177 ablation | 0.235239 | 0.190184 | -0.045054 | -19.15 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats5_seed123457_epoch1_lr1e-5.txt` |
| 5 | 3 | 123458 | ROB-177 ablation | 0.235218 | 0.189531 | -0.045688 | -19.42 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats5_seed123458_epoch1_lr1e-5.txt` |
| 10 | 1 | 123456 | ROB-177 ablation | 0.235218 | 0.187039 | -0.048179 | -20.48 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats10_epoch1_lr1e-5.txt` |
| 10 | 2 | 123457 | ROB-177 ablation | 0.235239 | 0.189858 | -0.045381 | -19.29 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats10_seed123457_epoch1_lr1e-5.txt` |
| 10 | 3 | 123458 | ROB-177 ablation | 0.235218 | 0.186815 | -0.048404 | -20.58 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats10_seed123458_epoch1_lr1e-5.txt` |
| 15 |  |  | ROB-108 default reference | 0.235198 | 0.185589 | -0.049609 | -21.09 | complete | `exp/results/repro/UFMR/earnings22_epoch1_lr1e-5.txt` |
| 20 | 1 | 123456 | ROB-177 ablation | 0.235239 | 0.186161 | -0.049078 | -20.86 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats20_epoch1_lr1e-5.txt` |
| 20 | 2 | 123457 | ROB-177 ablation | 0.235239 | 0.186651 | -0.048588 | -20.65 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats20_seed123457_epoch1_lr1e-5.txt` |
| 20 | 3 | 123458 | ROB-177 ablation | 0.235239 | 0.187693 | -0.047546 | -20.21 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats20_seed123458_epoch1_lr1e-5.txt` |
| 40 | 1 | 123456 | ROB-177 ablation | 0.235198 | 0.184650 | -0.050548 | -21.49 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats40_epoch1_lr1e-5.txt` |
| 40 | 2 | 123457 | ROB-177 ablation | 0.235218 | 0.186467 | -0.048751 | -20.73 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats40_seed123457_epoch1_lr1e-5.txt` |
| 40 | 3 | 123458 | ROB-177 ablation | 0.235239 | 0.185467 | -0.049772 | -21.16 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats40_seed123458_epoch1_lr1e-5.txt` |
| 100 | 1 | 123456 | ROB-177 ablation | 0.235218 | 0.185814 | -0.049405 | -21.00 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats100_epoch1_lr1e-5.txt` |
| 100 | 2 | 123457 | ROB-177 ablation | 0.235218 | 0.184874 | -0.050344 | -21.40 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats100_seed123457_epoch1_lr1e-5.txt` |
| 100 | 3 | 123458 | ROB-177 ablation | 0.235239 | 0.185140 | -0.050099 | -21.30 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats100_seed123458_epoch1_lr1e-5.txt` |
| 200 | 1 | 123456 | ROB-177 ablation | 0.235198 | 0.186753 | -0.048445 | -20.60 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats200_epoch1_lr1e-5.txt` |
| 200 | 2 | 123457 | ROB-177 ablation | 0.235218 | 0.185038 | -0.050181 | -21.33 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats200_seed123457_epoch1_lr1e-5.txt` |
| 200 | 3 | 123458 | ROB-177 ablation | 0.235218 | 0.187631 | -0.047587 | -20.23 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats200_seed123458_epoch1_lr1e-5.txt` |
| 1000 | 1 | 123456 | ROB-177 ablation | 0.235218 | 0.192206 | -0.043012 | -18.29 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats1000_seed123456_epoch1_lr1e-5.txt` |
| 1000 | 2 | 123457 | ROB-177 ablation | 0.235239 | 0.189490 | -0.045749 | -19.45 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats1000_seed123457_epoch1_lr1e-5.txt` |
| 1000 | 3 | 123458 | ROB-177 ablation | 0.235198 | 0.190082 | -0.045116 | -19.18 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats1000_seed123458_epoch1_lr1e-5.txt` |

## Candidate-Repeat Summary

| Candidate Repeats | Complete Trials | Mean Updated WER | Std Updated WER | Best Updated WER | Worst Updated WER |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 3 | 0.192472 | 0.000553 | 0.192002 | 0.193248 |
| 5 | 3 | 0.188973 | 0.001279 | 0.187203 | 0.190184 |
| 10 | 3 | 0.187904 | 0.001385 | 0.186815 | 0.189858 |
| 20 | 3 | 0.186835 | 0.000639 | 0.186161 | 0.187693 |
| 40 | 3 | 0.185528 | 0.000743 | 0.184650 | 0.186467 |
| 100 | 3 | 0.185276 | 0.000396 | 0.184874 | 0.185814 |
| 200 | 3 | 0.186474 | 0.001077 | 0.185038 | 0.187631 |
| 1000 | 3 | 0.190593 | 0.001166 | 0.189490 | 0.192206 |
