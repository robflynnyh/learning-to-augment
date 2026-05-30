# ROB-177 UFMR Candidate-Repeat Ablation

Earnings22 test-set UFMR ablation varying the number of randomly sampled frequency masks scored by the UFMR ranker per adaptation step.

All ROB-177 ablation rows use UFMR, epoch `1`, adaptation LR `1e-5`, the 2048-sequence ASR checkpoint, and `use_random: false`.
The `candidate_repeats` column is the UFMR mask-candidate count in `evaluation.augmentation_config.repeats`; it is not a seed repeat.
Each ROB-177 candidate-repeat setting has three seed trials when complete.

Completed rows: 8/25
Completed ROB-177 ablation rows: 7/24

## Per-Trial Results

| Candidate Repeats | Trial | Seed | Source | Original WER | Updated WER | Abs Delta | Rel Delta % | Status | Result |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 2 | 1 | 123456 | ROB-177 ablation | 0.235218 | 0.192002 | -0.043216 | -18.37 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats2_epoch1_lr1e-5.txt` |
| 2 | 2 | 123457 | ROB-177 ablation |  |  |  |  | missing | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats2_seed123457_epoch1_lr1e-5.txt` |
| 2 | 3 | 123458 | ROB-177 ablation |  |  |  |  | missing | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats2_seed123458_epoch1_lr1e-5.txt` |
| 5 | 1 | 123456 | ROB-177 ablation | 0.235239 | 0.187203 | -0.048036 | -20.42 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats5_epoch1_lr1e-5.txt` |
| 5 | 2 | 123457 | ROB-177 ablation |  |  |  |  | missing | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats5_seed123457_epoch1_lr1e-5.txt` |
| 5 | 3 | 123458 | ROB-177 ablation |  |  |  |  | missing | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats5_seed123458_epoch1_lr1e-5.txt` |
| 10 | 1 | 123456 | ROB-177 ablation | 0.235218 | 0.187039 | -0.048179 | -20.48 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats10_epoch1_lr1e-5.txt` |
| 10 | 2 | 123457 | ROB-177 ablation |  |  |  |  | missing | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats10_seed123457_epoch1_lr1e-5.txt` |
| 10 | 3 | 123458 | ROB-177 ablation |  |  |  |  | missing | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats10_seed123458_epoch1_lr1e-5.txt` |
| 15 |  |  | ROB-108 default reference | 0.235198 | 0.185589 | -0.049609 | -21.09 | complete | `exp/results/repro/UFMR/earnings22_epoch1_lr1e-5.txt` |
| 20 | 1 | 123456 | ROB-177 ablation | 0.235239 | 0.186161 | -0.049078 | -20.86 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats20_epoch1_lr1e-5.txt` |
| 20 | 2 | 123457 | ROB-177 ablation |  |  |  |  | missing | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats20_seed123457_epoch1_lr1e-5.txt` |
| 20 | 3 | 123458 | ROB-177 ablation |  |  |  |  | missing | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats20_seed123458_epoch1_lr1e-5.txt` |
| 40 | 1 | 123456 | ROB-177 ablation | 0.235198 | 0.184650 | -0.050548 | -21.49 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats40_epoch1_lr1e-5.txt` |
| 40 | 2 | 123457 | ROB-177 ablation |  |  |  |  | missing | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats40_seed123457_epoch1_lr1e-5.txt` |
| 40 | 3 | 123458 | ROB-177 ablation |  |  |  |  | missing | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats40_seed123458_epoch1_lr1e-5.txt` |
| 100 | 1 | 123456 | ROB-177 ablation | 0.235218 | 0.185814 | -0.049405 | -21.00 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats100_epoch1_lr1e-5.txt` |
| 100 | 2 | 123457 | ROB-177 ablation |  |  |  |  | missing | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats100_seed123457_epoch1_lr1e-5.txt` |
| 100 | 3 | 123458 | ROB-177 ablation |  |  |  |  | missing | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats100_seed123458_epoch1_lr1e-5.txt` |
| 200 | 1 | 123456 | ROB-177 ablation | 0.235198 | 0.186753 | -0.048445 | -20.60 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats200_epoch1_lr1e-5.txt` |
| 200 | 2 | 123457 | ROB-177 ablation |  |  |  |  | missing | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats200_seed123457_epoch1_lr1e-5.txt` |
| 200 | 3 | 123458 | ROB-177 ablation |  |  |  |  | missing | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats200_seed123458_epoch1_lr1e-5.txt` |
| 1000 | 1 | 123456 | ROB-177 ablation |  |  |  |  | missing | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats1000_seed123456_epoch1_lr1e-5.txt` |
| 1000 | 2 | 123457 | ROB-177 ablation |  |  |  |  | missing | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats1000_seed123457_epoch1_lr1e-5.txt` |
| 1000 | 3 | 123458 | ROB-177 ablation |  |  |  |  | missing | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats1000_seed123458_epoch1_lr1e-5.txt` |

## Candidate-Repeat Summary

| Candidate Repeats | Complete Trials | Mean Updated WER | Std Updated WER | Best Updated WER | Worst Updated WER |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1 | 0.192002 | 0.000000 | 0.192002 | 0.192002 |
| 5 | 1 | 0.187203 | 0.000000 | 0.187203 | 0.187203 |
| 10 | 1 | 0.187039 | 0.000000 | 0.187039 | 0.187039 |
| 20 | 1 | 0.186161 | 0.000000 | 0.186161 | 0.186161 |
| 40 | 1 | 0.184650 | 0.000000 | 0.184650 | 0.184650 |
| 100 | 1 | 0.185814 | 0.000000 | 0.185814 | 0.185814 |
| 200 | 1 | 0.186753 | 0.000000 | 0.186753 | 0.186753 |

## Missing ROB-177 Cells

- candidate repeats `2`, trial `2`, seed `123457`
- candidate repeats `2`, trial `3`, seed `123458`
- candidate repeats `5`, trial `2`, seed `123457`
- candidate repeats `5`, trial `3`, seed `123458`
- candidate repeats `10`, trial `2`, seed `123457`
- candidate repeats `10`, trial `3`, seed `123458`
- candidate repeats `20`, trial `2`, seed `123457`
- candidate repeats `20`, trial `3`, seed `123458`
- candidate repeats `40`, trial `2`, seed `123457`
- candidate repeats `40`, trial `3`, seed `123458`
- candidate repeats `100`, trial `2`, seed `123457`
- candidate repeats `100`, trial `3`, seed `123458`
- candidate repeats `200`, trial `2`, seed `123457`
- candidate repeats `200`, trial `3`, seed `123458`
- candidate repeats `1000`, trial `1`, seed `123456`
- candidate repeats `1000`, trial `2`, seed `123457`
- candidate repeats `1000`, trial `3`, seed `123458`
