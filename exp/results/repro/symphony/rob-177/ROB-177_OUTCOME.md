# ROB-177 UFMR Candidate-Repeat Ablation

Earnings22 test-set UFMR ablation varying the number of randomly sampled frequency masks scored by the UFMR ranker per adaptation step.

All ROB-177 ablation rows use UFMR, epoch `1`, adaptation LR `1e-5`, the 2048-sequence ASR checkpoint, and `use_random: false`.
The `candidate_repeats` column is the UFMR mask-candidate count in `evaluation.augmentation_config.repeats`; it is not a seed repeat.

Completed rows: 8/8
Completed ROB-177 ablation rows: 7/7

| Candidate Repeats | Source | Original WER | Updated WER | Abs Delta | Rel Delta % | Status | Result |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 2 | ROB-177 ablation | 0.235218 | 0.192002 | -0.043216 | -18.37 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats2_epoch1_lr1e-5.txt` |
| 5 | ROB-177 ablation | 0.235239 | 0.187203 | -0.048036 | -20.42 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats5_epoch1_lr1e-5.txt` |
| 10 | ROB-177 ablation | 0.235218 | 0.187039 | -0.048179 | -20.48 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats10_epoch1_lr1e-5.txt` |
| 15 | ROB-108 default reference | 0.235198 | 0.185589 | -0.049609 | -21.09 | complete | `exp/results/repro/UFMR/earnings22_epoch1_lr1e-5.txt` |
| 20 | ROB-177 ablation | 0.235239 | 0.186161 | -0.049078 | -20.86 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats20_epoch1_lr1e-5.txt` |
| 40 | ROB-177 ablation | 0.235198 | 0.184650 | -0.050548 | -21.49 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats40_epoch1_lr1e-5.txt` |
| 100 | ROB-177 ablation | 0.235218 | 0.185814 | -0.049405 | -21.00 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats100_epoch1_lr1e-5.txt` |
| 200 | ROB-177 ablation | 0.235198 | 0.186753 | -0.048445 | -20.60 | complete | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats200_epoch1_lr1e-5.txt` |
