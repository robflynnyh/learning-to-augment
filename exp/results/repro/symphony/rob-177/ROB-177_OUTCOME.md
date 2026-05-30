# ROB-177 UFMR Candidate-Repeat Ablation

`earnings22`, `tedlium` test-set UFMR ablation varying the number of randomly sampled frequency masks scored by the UFMR ranker per adaptation step.

All ROB-177 ablation rows use UFMR, epoch `1`, adaptation LR `1e-5`, the 2048-sequence ASR checkpoint, and `use_random: false`.
The `candidate_repeats` column is the UFMR mask-candidate count in `evaluation.augmentation_config.repeats`; it is not a seed repeat.
Candidate-repeat settings: `1`, `2`, `5`, `10`, `15`, `20`, `40`, `100`, `200`, `1000`.
Each ROB-177 candidate-repeat setting has three seed trials when complete.

Completed rows: 62/62
Completed ROB-177 ablation rows: 60/60

## Per-Trial Results

| Dataset | Candidate Repeats | Trial | Seed | Source | Original WER | Updated WER | Abs Delta | Rel Delta % | Status | Result |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| earnings22 | 1 | 1 | 123456 | ROB-177 ablation | 0.235239 | 0.194759 | -0.040480 | -17.21 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats1_seed123456_epoch1_lr1e-5.txt` |
| earnings22 | 1 | 2 | 123457 | ROB-177 ablation | 0.235239 | 0.197537 | -0.037702 | -16.03 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats1_seed123457_epoch1_lr1e-5.txt` |
| earnings22 | 1 | 3 | 123458 | ROB-177 ablation | 0.235239 | 0.195249 | -0.039989 | -17.00 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats1_seed123458_epoch1_lr1e-5.txt` |
| earnings22 | 2 | 1 | 123456 | ROB-177 ablation | 0.235218 | 0.192002 | -0.043216 | -18.37 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats2_epoch1_lr1e-5.txt` |
| earnings22 | 2 | 2 | 123457 | ROB-177 ablation | 0.235239 | 0.193248 | -0.041991 | -17.85 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats2_seed123457_epoch1_lr1e-5.txt` |
| earnings22 | 2 | 3 | 123458 | ROB-177 ablation | 0.235218 | 0.192166 | -0.043053 | -18.30 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats2_seed123458_epoch1_lr1e-5.txt` |
| earnings22 | 5 | 1 | 123456 | ROB-177 ablation | 0.235239 | 0.187203 | -0.048036 | -20.42 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats5_epoch1_lr1e-5.txt` |
| earnings22 | 5 | 2 | 123457 | ROB-177 ablation | 0.235239 | 0.190184 | -0.045054 | -19.15 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats5_seed123457_epoch1_lr1e-5.txt` |
| earnings22 | 5 | 3 | 123458 | ROB-177 ablation | 0.235218 | 0.189531 | -0.045688 | -19.42 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats5_seed123458_epoch1_lr1e-5.txt` |
| earnings22 | 10 | 1 | 123456 | ROB-177 ablation | 0.235218 | 0.187039 | -0.048179 | -20.48 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats10_epoch1_lr1e-5.txt` |
| earnings22 | 10 | 2 | 123457 | ROB-177 ablation | 0.235239 | 0.189858 | -0.045381 | -19.29 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats10_seed123457_epoch1_lr1e-5.txt` |
| earnings22 | 10 | 3 | 123458 | ROB-177 ablation | 0.235218 | 0.186815 | -0.048404 | -20.58 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats10_seed123458_epoch1_lr1e-5.txt` |
| earnings22 | 15 |  |  | ROB-108 default reference | 0.235198 | 0.185589 | -0.049609 | -21.09 | complete | `exp/results/repro/UFMR/earnings22_epoch1_lr1e-5.txt` |
| earnings22 | 15 | 1 | 123456 | ROB-177 ablation | 0.235218 | 0.186753 | -0.048465 | -20.60 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats15_seed123456_epoch1_lr1e-5.txt` |
| earnings22 | 15 | 2 | 123457 | ROB-177 ablation | 0.235218 | 0.186059 | -0.049160 | -20.90 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats15_seed123457_epoch1_lr1e-5.txt` |
| earnings22 | 15 | 3 | 123458 | ROB-177 ablation | 0.235239 | 0.189653 | -0.045585 | -19.38 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats15_seed123458_epoch1_lr1e-5.txt` |
| earnings22 | 20 | 1 | 123456 | ROB-177 ablation | 0.235239 | 0.186161 | -0.049078 | -20.86 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats20_epoch1_lr1e-5.txt` |
| earnings22 | 20 | 2 | 123457 | ROB-177 ablation | 0.235239 | 0.186651 | -0.048588 | -20.65 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats20_seed123457_epoch1_lr1e-5.txt` |
| earnings22 | 20 | 3 | 123458 | ROB-177 ablation | 0.235239 | 0.187693 | -0.047546 | -20.21 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats20_seed123458_epoch1_lr1e-5.txt` |
| earnings22 | 40 | 1 | 123456 | ROB-177 ablation | 0.235198 | 0.184650 | -0.050548 | -21.49 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats40_epoch1_lr1e-5.txt` |
| earnings22 | 40 | 2 | 123457 | ROB-177 ablation | 0.235218 | 0.186467 | -0.048751 | -20.73 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats40_seed123457_epoch1_lr1e-5.txt` |
| earnings22 | 40 | 3 | 123458 | ROB-177 ablation | 0.235239 | 0.185467 | -0.049772 | -21.16 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats40_seed123458_epoch1_lr1e-5.txt` |
| earnings22 | 100 | 1 | 123456 | ROB-177 ablation | 0.235218 | 0.185814 | -0.049405 | -21.00 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats100_epoch1_lr1e-5.txt` |
| earnings22 | 100 | 2 | 123457 | ROB-177 ablation | 0.235218 | 0.184874 | -0.050344 | -21.40 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats100_seed123457_epoch1_lr1e-5.txt` |
| earnings22 | 100 | 3 | 123458 | ROB-177 ablation | 0.235239 | 0.185140 | -0.050099 | -21.30 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats100_seed123458_epoch1_lr1e-5.txt` |
| earnings22 | 200 | 1 | 123456 | ROB-177 ablation | 0.235198 | 0.186753 | -0.048445 | -20.60 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats200_epoch1_lr1e-5.txt` |
| earnings22 | 200 | 2 | 123457 | ROB-177 ablation | 0.235218 | 0.185038 | -0.050181 | -21.33 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats200_seed123457_epoch1_lr1e-5.txt` |
| earnings22 | 200 | 3 | 123458 | ROB-177 ablation | 0.235218 | 0.187631 | -0.047587 | -20.23 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats200_seed123458_epoch1_lr1e-5.txt` |
| earnings22 | 1000 | 1 | 123456 | ROB-177 ablation | 0.235218 | 0.192206 | -0.043012 | -18.29 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats1000_seed123456_epoch1_lr1e-5.txt` |
| earnings22 | 1000 | 2 | 123457 | ROB-177 ablation | 0.235239 | 0.189490 | -0.045749 | -19.45 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats1000_seed123457_epoch1_lr1e-5.txt` |
| earnings22 | 1000 | 3 | 123458 | ROB-177 ablation | 0.235198 | 0.190082 | -0.045116 | -19.18 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats1000_seed123458_epoch1_lr1e-5.txt` |
| tedlium | 1 | 1 | 123456 | ROB-177 ablation | 0.085345 | 0.078256 | -0.007088 | -8.31 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/tedlium_test_candidate_repeats1_seed123456_epoch1_lr1e-5.txt` |
| tedlium | 1 | 2 | 123457 | ROB-177 ablation | 0.085345 | 0.078540 | -0.006805 | -7.97 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/tedlium_test_candidate_repeats1_seed123457_epoch1_lr1e-5.txt` |
| tedlium | 1 | 3 | 123458 | ROB-177 ablation | 0.085345 | 0.077477 | -0.007868 | -9.22 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/tedlium_test_candidate_repeats1_seed123458_epoch1_lr1e-5.txt` |
| tedlium | 2 | 1 | 123456 | ROB-177 ablation | 0.085345 | 0.078114 | -0.007230 | -8.47 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/tedlium_test_candidate_repeats2_seed123456_epoch1_lr1e-5.txt` |
| tedlium | 2 | 2 | 123457 | ROB-177 ablation | 0.085345 | 0.076909 | -0.008435 | -9.88 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/tedlium_test_candidate_repeats2_seed123457_epoch1_lr1e-5.txt` |
| tedlium | 2 | 3 | 123458 | ROB-177 ablation | 0.085345 | 0.078114 | -0.007230 | -8.47 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/tedlium_test_candidate_repeats2_seed123458_epoch1_lr1e-5.txt` |
| tedlium | 5 | 1 | 123456 | ROB-177 ablation | 0.085345 | 0.077689 | -0.007656 | -8.97 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/tedlium_test_candidate_repeats5_seed123456_epoch1_lr1e-5.txt` |
| tedlium | 5 | 2 | 123457 | ROB-177 ablation | 0.085345 | 0.076484 | -0.008861 | -10.38 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/tedlium_test_candidate_repeats5_seed123457_epoch1_lr1e-5.txt` |
| tedlium | 5 | 3 | 123458 | ROB-177 ablation | 0.085345 | 0.076909 | -0.008435 | -9.88 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/tedlium_test_candidate_repeats5_seed123458_epoch1_lr1e-5.txt` |
| tedlium | 10 | 1 | 123456 | ROB-177 ablation | 0.085345 | 0.076626 | -0.008719 | -10.22 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/tedlium_test_candidate_repeats10_seed123456_epoch1_lr1e-5.txt` |
| tedlium | 10 | 2 | 123457 | ROB-177 ablation | 0.085345 | 0.077370 | -0.007974 | -9.34 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/tedlium_test_candidate_repeats10_seed123457_epoch1_lr1e-5.txt` |
| tedlium | 10 | 3 | 123458 | ROB-177 ablation | 0.085345 | 0.076945 | -0.008400 | -9.84 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/tedlium_test_candidate_repeats10_seed123458_epoch1_lr1e-5.txt` |
| tedlium | 15 |  |  | ROB-108 default reference | 0.085345 | 0.076555 | -0.008790 | -10.30 | complete | `exp/results/repro/UFMR/tedlium_epoch1_lr1e-5.txt` |
| tedlium | 15 | 1 | 123456 | ROB-177 ablation | 0.085345 | 0.076661 | -0.008683 | -10.17 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/tedlium_test_candidate_repeats15_seed123456_epoch1_lr1e-5.txt` |
| tedlium | 15 | 2 | 123457 | ROB-177 ablation | 0.085345 | 0.076236 | -0.009109 | -10.67 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/tedlium_test_candidate_repeats15_seed123457_epoch1_lr1e-5.txt` |
| tedlium | 15 | 3 | 123458 | ROB-177 ablation | 0.085345 | 0.078363 | -0.006982 | -8.18 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/tedlium_test_candidate_repeats15_seed123458_epoch1_lr1e-5.txt` |
| tedlium | 20 | 1 | 123456 | ROB-177 ablation | 0.085345 | 0.076768 | -0.008577 | -10.05 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/tedlium_test_candidate_repeats20_seed123456_epoch1_lr1e-5.txt` |
| tedlium | 20 | 2 | 123457 | ROB-177 ablation | 0.085345 | 0.076626 | -0.008719 | -10.22 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/tedlium_test_candidate_repeats20_seed123457_epoch1_lr1e-5.txt` |
| tedlium | 20 | 3 | 123458 | ROB-177 ablation | 0.085345 | 0.076803 | -0.008542 | -10.01 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/tedlium_test_candidate_repeats20_seed123458_epoch1_lr1e-5.txt` |
| tedlium | 40 | 1 | 123456 | ROB-177 ablation | 0.085345 | 0.077122 | -0.008223 | -9.63 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/tedlium_test_candidate_repeats40_seed123456_epoch1_lr1e-5.txt` |
| tedlium | 40 | 2 | 123457 | ROB-177 ablation | 0.085345 | 0.077335 | -0.008010 | -9.39 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/tedlium_test_candidate_repeats40_seed123457_epoch1_lr1e-5.txt` |
| tedlium | 40 | 3 | 123458 | ROB-177 ablation | 0.085345 | 0.078079 | -0.007266 | -8.51 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/tedlium_test_candidate_repeats40_seed123458_epoch1_lr1e-5.txt` |
| tedlium | 100 | 1 | 123456 | ROB-177 ablation | 0.085345 | 0.076413 | -0.008931 | -10.47 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/tedlium_test_candidate_repeats100_seed123456_epoch1_lr1e-5.txt` |
| tedlium | 100 | 2 | 123457 | ROB-177 ablation | 0.085345 | 0.076980 | -0.008364 | -9.80 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/tedlium_test_candidate_repeats100_seed123457_epoch1_lr1e-5.txt` |
| tedlium | 100 | 3 | 123458 | ROB-177 ablation | 0.085345 | 0.076059 | -0.009286 | -10.88 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/tedlium_test_candidate_repeats100_seed123458_epoch1_lr1e-5.txt` |
| tedlium | 200 | 1 | 123456 | ROB-177 ablation | 0.085345 | 0.076874 | -0.008471 | -9.93 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/tedlium_test_candidate_repeats200_seed123456_epoch1_lr1e-5.txt` |
| tedlium | 200 | 2 | 123457 | ROB-177 ablation | 0.085345 | 0.076590 | -0.008754 | -10.26 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/tedlium_test_candidate_repeats200_seed123457_epoch1_lr1e-5.txt` |
| tedlium | 200 | 3 | 123458 | ROB-177 ablation | 0.085345 | 0.076201 | -0.009144 | -10.71 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/tedlium_test_candidate_repeats200_seed123458_epoch1_lr1e-5.txt` |
| tedlium | 1000 | 1 | 123456 | ROB-177 ablation | 0.085345 | 0.078504 | -0.006840 | -8.01 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/tedlium_test_candidate_repeats1000_seed123456_epoch1_lr1e-5.txt` |
| tedlium | 1000 | 2 | 123457 | ROB-177 ablation | 0.085345 | 0.080135 | -0.005210 | -6.10 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/tedlium_test_candidate_repeats1000_seed123457_epoch1_lr1e-5.txt` |
| tedlium | 1000 | 3 | 123458 | ROB-177 ablation | 0.085345 | 0.078433 | -0.006911 | -8.10 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/tedlium_test_candidate_repeats1000_seed123458_epoch1_lr1e-5.txt` |

## Candidate-Repeat Summary

| Dataset | Candidate Repeats | Complete Trials | Mean Updated WER | Std Updated WER | Best Updated WER | Worst Updated WER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| earnings22 | 1 | 3 | 0.195848 | 0.001211 | 0.194759 | 0.197537 |
| earnings22 | 2 | 3 | 0.192472 | 0.000553 | 0.192002 | 0.193248 |
| earnings22 | 5 | 3 | 0.188973 | 0.001279 | 0.187203 | 0.190184 |
| earnings22 | 10 | 3 | 0.187904 | 0.001385 | 0.186815 | 0.189858 |
| earnings22 | 15 | 3 | 0.187488 | 0.001557 | 0.186059 | 0.189653 |
| earnings22 | 20 | 3 | 0.186835 | 0.000639 | 0.186161 | 0.187693 |
| earnings22 | 40 | 3 | 0.185528 | 0.000743 | 0.184650 | 0.186467 |
| earnings22 | 100 | 3 | 0.185276 | 0.000396 | 0.184874 | 0.185814 |
| earnings22 | 200 | 3 | 0.186474 | 0.001077 | 0.185038 | 0.187631 |
| earnings22 | 1000 | 3 | 0.190593 | 0.001166 | 0.189490 | 0.192206 |
| tedlium | 1 | 3 | 0.078091 | 0.000449 | 0.077477 | 0.078540 |
| tedlium | 2 | 3 | 0.077712 | 0.000568 | 0.076909 | 0.078114 |
| tedlium | 5 | 3 | 0.077027 | 0.000499 | 0.076484 | 0.077689 |
| tedlium | 10 | 3 | 0.076980 | 0.000305 | 0.076626 | 0.077370 |
| tedlium | 15 | 3 | 0.077087 | 0.000919 | 0.076236 | 0.078363 |
| tedlium | 20 | 3 | 0.076732 | 0.000077 | 0.076626 | 0.076803 |
| tedlium | 40 | 3 | 0.077512 | 0.000410 | 0.077122 | 0.078079 |
| tedlium | 100 | 3 | 0.076484 | 0.000379 | 0.076059 | 0.076980 |
| tedlium | 200 | 3 | 0.076555 | 0.000276 | 0.076201 | 0.076874 |
| tedlium | 1000 | 3 | 0.079024 | 0.000786 | 0.078433 | 0.080135 |
