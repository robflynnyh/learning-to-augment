# ROB-177 UFMR Candidate-Repeat Ablation

Earnings22 test-set UFMR ablation varying the number of randomly sampled frequency masks scored by the UFMR ranker per adaptation step.

All ROB-177 ablation rows use UFMR, epoch `1`, adaptation LR `1e-5`, the 2048-sequence ASR checkpoint, and `use_random: false`.
The `candidate_repeats` column is the UFMR mask-candidate count in `evaluation.augmentation_config.repeats`; it is not a seed repeat.

Completed rows: 1/8
Completed ROB-177 ablation rows: 0/7

| Candidate Repeats | Source | Original WER | Updated WER | Abs Delta | Rel Delta % | Status | Result |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 2 | ROB-177 ablation |  |  |  |  | missing | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats2_epoch1_lr1e-5.txt` |
| 5 | ROB-177 ablation |  |  |  |  | missing | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats5_epoch1_lr1e-5.txt` |
| 10 | ROB-177 ablation |  |  |  |  | missing | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats10_epoch1_lr1e-5.txt` |
| 15 | ROB-108 default reference | 0.235198 | 0.185589 | -0.049609 | -21.09 | complete | `exp/results/repro/UFMR/earnings22_epoch1_lr1e-5.txt` |
| 20 | ROB-177 ablation |  |  |  |  | missing | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats20_epoch1_lr1e-5.txt` |
| 40 | ROB-177 ablation |  |  |  |  | missing | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats40_epoch1_lr1e-5.txt` |
| 100 | ROB-177 ablation |  |  |  |  | missing | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats100_epoch1_lr1e-5.txt` |
| 200 | ROB-177 ablation |  |  |  |  | missing | `exp/results/repro/symphony/rob-177/results/UFMR/earnings22_test_candidate_repeats200_epoch1_lr1e-5.txt` |

## Missing ROB-177 Cells

- candidate repeats `2`
- candidate repeats `5`
- candidate repeats `10`
- candidate repeats `20`
- candidate repeats `40`
- candidate repeats `100`
- candidate repeats `200`
