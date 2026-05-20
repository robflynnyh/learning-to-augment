# Oracle Reproduction Outcome

Issue: ROB-60
Date: 2026-05-19

## Scope

The sequential CPU oracle reproduction sweep completed for the active reduced
configuration set generated under `exp/results/repro/oracle/`:

- RMM, historical matched setup: `lr=1e-6`, `search_lr=4e-2`
- RMM, newer/default setup: `lr=8e-6`, `search_lr=9e-2`
- RFM, historical matched setup: `lr=1e-6`, `search_lr=4e-2`
- RFM, newer/default setup: `lr=8e-6`, `search_lr=9e-2`

Each active configuration produced results for repeats
`1, 2, 3, 4, 5, 10, 20, 50` on `tedlium3_segmented_data`, split `test`,
with `cpu_rollout_search`.

The empty files `tedlium_lr1e-6_searchlr9e-2.txt` and
`tedlium_lr8e-6_searchlr4e-2.txt` under both `RMM/` and `RFM/` are inactive
cross-combination leftovers and are not part of this sweep.

Follow-up Mimas GPU sweeps requested by Robert also completed for both RMM and
RFM at `lr=1e-5`, `search_lr=2e-1`; `lr=8e-6`, `search_lr=2e-1`; and
`lr=1e-5`, `search_lr=9e-2`, using the same repeats
`1, 2, 3, 4, 5, 10, 20, 50`.

The follow-up UVQLM oracle sweep also completed on Mimas at `lr=1e-5`,
`search_lr=2e-1`, using `UnconditionalMaskGenerator` proposals and the same
repeat grid. Later multi-policy follow-ups completed for RMM, RFM, and UVQLM
at `lr=3e-5`, `lr=6e-5`, and `lr=1e-4`, all with `search_lr=2e-1` and the
same repeat grid.

The random additive-noise oracle follow-up also completed on Mimas at
`lr=3e-5`, `search_lr=2e-1`, using `AdditivePolicy` with `use_random: true`
and the same repeat grid.

## Summary

Baseline original WER was `9.586%` for all rows.

| Method | Setup | Best repeat | Best WER | Abs. gain | Relative gain | Repeat 50 WER |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RMM | `lr=8e-6`, `search_lr=9e-2` | 20 | 8.732% | 0.854 pp | 8.9% | 8.789% |
| RFM | `lr=8e-6`, `search_lr=9e-2` | 10 | 8.941% | 0.645 pp | 6.7% | 9.005% |
| RMM | `lr=1e-6`, `search_lr=4e-2` | 20 | 8.991% | 0.595 pp | 6.2% | 9.069% |
| RFM | `lr=1e-6`, `search_lr=4e-2` | 50 | 9.186% | 0.400 pp | 4.2% | 9.186% |

## Follow-up GPU Sweeps

Robert requested additional oracle sweeps at `lr=1e-5`, `search_lr=2e-1`;
`lr=8e-6`, `search_lr=2e-1`; `lr=1e-5`, `search_lr=9e-2`; and `lr=3e-5`,
`search_lr=2e-1`. After `3e-5/2e-1` became the strongest setting, higher-LR
`1e-4/2e-1` and intermediate `6e-5/2e-1` follow-ups were also run for RMM,
RFM, and UVQLM. A random additive-noise `RAN` follow-up was then run at
`3e-5/2e-1`. The detached GPU runs completed successfully between 2026-05-11
and 2026-05-14. The baseline original WER is approximately `9.586%`; the exact
original WER varies slightly between repeated rows in the generated text files.

| Method | Setup | Best repeat | Best WER | Abs. gain | Relative gain | Repeat 50 WER |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RMM | `lr=3e-5`, `search_lr=2e-1` | 50 | 8.428% | 1.155 pp | 12.1% | 8.428% |
| UVQLM | `lr=3e-5`, `search_lr=2e-1` | 50 | 8.467% | 1.116 pp | 11.6% | 8.467% |
| UVQLM | `lr=1e-5`, `search_lr=2e-1` | 50 | 8.488% | 1.095 pp | 11.4% | 8.488% |
| UVQLM | `lr=6e-5`, `search_lr=2e-1` | 50 | 8.555% | 1.028 pp | 10.7% | 8.555% |
| RMM | `lr=1e-5`, `search_lr=2e-1` | 50 | 8.569% | 1.014 pp | 10.6% | 8.569% |
| RMM | `lr=6e-5`, `search_lr=2e-1` | 50 | 8.580% | 1.003 pp | 10.5% | 8.580% |
| RMM | `lr=1e-5`, `search_lr=9e-2` | 10 | 8.612% | 0.971 pp | 10.1% | 8.665% |
| RMM | `lr=8e-6`, `search_lr=2e-1` | 50 | 8.626% | 0.960 pp | 10.0% | 8.626% |
| RFM | `lr=3e-5`, `search_lr=2e-1` | 50 | 8.817% | 0.769 pp | 8.0% | 8.817% |
| UVQLM | `lr=1e-4`, `search_lr=2e-1` | 50 | 8.821% | 0.762 pp | 8.0% | 8.821% |
| RFM | `lr=1e-5`, `search_lr=2e-1` | 50 | 8.842% | 0.744 pp | 7.8% | 8.842% |
| RAN | `lr=3e-5`, `search_lr=2e-1` | 5 | 8.885% | 0.702 pp | 7.3% | 8.949% |
| RFM | `lr=8e-6`, `search_lr=2e-1` | 50 | 8.892% | 0.691 pp | 7.2% | 8.892% |
| RFM | `lr=6e-5`, `search_lr=2e-1` | 50 | 8.906% | 0.677 pp | 7.1% | 8.906% |
| RMM | `lr=1e-4`, `search_lr=2e-1` | 50 | 8.927% | 0.659 pp | 6.9% | 8.927% |
| RFM | `lr=1e-5`, `search_lr=9e-2` | 10 | 8.977% | 0.606 pp | 6.3% | 8.995% |
| RFM | `lr=1e-4`, `search_lr=2e-1` | 50 | 9.204% | 0.383 pp | 4.0% | 9.204% |

The `3e-5/2e-1` follow-up remains the best completed oracle setting.
RMM reaches `8.428%` WER at repeat 50, beating UVQLM at the same setting by
`0.039` pp, the previous UVQLM `1e-5/2e-1` best by `0.060` pp, and the matching
RFM `3e-5/2e-1` result by `0.390` pp. The higher `1e-4/2e-1` follow-up did not
improve any method: best WERs were UVQLM `8.821%`, RMM `8.927%`, and RFM
`9.204%`, all at repeat 50. The intermediate `6e-5/2e-1` follow-up improved
over `1e-4/2e-1` for all three methods, but did not recover the `3e-5/2e-1`
best: RMM was worse by `0.152` pp, UVQLM by `0.089` pp, and RFM by `0.089` pp.
The random additive-noise RAN follow-up reached `8.885%` WER at repeat 5 and
`8.949%` at repeat 50. It improved over no adaptation and narrowly beat the
older `8e-6/9e-2` RFM oracle best, but it underperformed the matching
`3e-5/2e-1` learned/random mask-search cells: worse than RMM by `0.457` pp,
UVQLM by `0.418` pp, and RFM by `0.067` pp.
The `search_lr=2e-1` RMM/RFM sweeps improved over the previous newer/default
oracle setup for both methods. The later `1e-5/9e-2` cell
improved RMM versus `8e-6/9e-2`, but did not improve RFM and did not beat the
`1e-5/2e-1` setting:

| Method | Previous best `8e-6/9e-2` | `8e-6/2e-1` best | Delta | `1e-5/9e-2` best | Delta | `1e-5/2e-1` best | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMM | 8.732% | 8.626% | -0.106 pp | 8.612% | -0.120 pp | 8.569% | -0.163 pp |
| RFM | 8.941% | 8.892% | -0.049 pp | 8.977% | +0.036 pp | 8.842% | -0.099 pp |

All `search_lr=2e-1` RMM, RFM, and UVQLM follow-up curves reached their best
result at repeat 50, including the `3e-5/2e-1`, `6e-5/2e-1`, and `1e-4/2e-1`
curves. The RAN curve is the exception among the later `2e-1` follow-ups,
peaking at repeat 5. This differs from the previous RMM `8e-6/9e-2` result,
which peaked at repeat 20, and the `1e-5/9e-2` RMM/RFM curves, which peaked at
repeat 10. The best overall completed setting is RMM at `lr=3e-5`,
`search_lr=2e-1`, repeat 50.

## Interpretation

The newer/default setup is better than the historical matched setup for both
masking policies. RMM is consistently ahead of RFM in the matched comparison:

- Historical matched best WER: RMM `8.991%` vs RFM `9.186%`
- Newer/default best WER: RMM `8.732%` vs RFM `8.941%`

Increasing repeat count generally helps, but the optimum does not always occur
at repeat 50. In the original reduced sweep, RMM peaked at repeat 20 in both
setups, while RFM peaked at repeat 10 for the newer/default setup and repeat 50
for the historical setup. In the later `search_lr=2e-1` GPU sweeps, both RMM
and RFM were still improving through repeat 50.

## Newer/default Oracle Plot

Robert requested an updated plot for the newer/default setup with RMM and RFM
oracle curves plus no-adaptation and UFMR references. The missing matching UFMR
segmented policy eval was run with `lr=8e-6`, 15 policy repeats, and produced:

- No adaptation: `9.586%` WER
- UFMR `lr=8e-6`: `9.041%` WER
- RFM oracle best: `8.941%` WER at repeat 10
- RMM oracle best: `8.732%` WER at repeat 20

The plot shows that the newer/default RMM oracle remains best, while the RFM
oracle only narrowly beats the matching UFMR line at its best repeat.
After review, the plot was regenerated with a log-scaled repeat axis to make the
early-repeat region easier to read across repeats `1` through `50`.

A second log-scaled comparison plot now includes the follow-up `1e-5/2e-1`,
`8e-6/2e-1`, `1e-5/9e-2`, `3e-5/2e-1`, `6e-5/2e-1`, and `1e-4/2e-1` RMM/RFM
curves, the `1e-5/2e-1`, `3e-5/2e-1`, `6e-5/2e-1`, and `1e-4/2e-1` UVQLM
curves, the `3e-5/2e-1` RAN curve, the previous `8e-6/9e-2` curves, and the
same UFMR/no-adaptation references:

- `exp/results/repro/oracle/oracle_lr_sweep_vs_ufmr.pdf`
- `exp/results/repro/oracle/oracle_lr_sweep_vs_ufmr.csv`

Robert later requested a less crowded version of this plot containing only the
best LR setup per method, where "best" is selected by repeat-50 WER. The
repeat-50 selection from `oracle_lr_sweep_vs_ufmr.csv` is:

| Method | Selected setup | Repeat 50 WER | Best repeat in selected curve | Best WER |
| --- | --- | ---: | ---: | ---: |
| RMM | `lr=3e-5`, `search_lr=2e-1` | 8.428% | 50 | 8.428% |
| UVQLM | `lr=3e-5`, `search_lr=2e-1` | 8.467% | 50 | 8.467% |
| RFM | `lr=3e-5`, `search_lr=2e-1` | 8.817% | 50 | 8.817% |
| RAN | `lr=3e-5`, `search_lr=2e-1` | 8.949% | 5 | 8.885% |

The reduced log-scaled comparison includes those four selected curves plus the
same no-adaptation and UFMR references. After review feedback, the legend uses
method-only labels; the selected LR/search-LR details are recorded in the table
above rather than repeated inside the plot legend.

- `exp/results/repro/oracle/oracle_best_repeat50_lrs_vs_ufmr.pdf`
- `exp/results/repro/oracle/oracle_best_repeat50_lrs_vs_ufmr.csv`

After earlier review feedback, the plotting script was refreshed to use a
higher-contrast color cycle, varied markers and line styles, and a distinct
UFMR reference line. Both `oracle_lr_sweep_vs_ufmr.pdf` and
`oracle_best_repeat50_lrs_vs_ufmr.pdf` were regenerated with this styling.

Artifacts:

- `exp/results/repro/oracle/newer_default_oracle_vs_ufmr.pdf`
- `exp/results/repro/oracle/newer_default_oracle_vs_ufmr.csv`
- `exp/results/repro/policy/UFMR_segmented/tedlium_lr8e-6.txt`
- `exp/results/repro/policy/UFMR_segmented/logs/tedlium_lr8e-6.log`

## Evidence

Result files:

- `exp/results/repro/oracle/RMM/tedlium_lr1e-6_searchlr4e-2.txt`
- `exp/results/repro/oracle/RMM/tedlium_lr8e-6_searchlr9e-2.txt`
- `exp/results/repro/oracle/RMM/tedlium_lr1e-5_searchlr2e-1.txt`
- `exp/results/repro/oracle/RMM/tedlium_lr1e-5_searchlr9e-2.txt`
- `exp/results/repro/oracle/RMM/tedlium_lr8e-6_searchlr2e-1.txt`
- `exp/results/repro/oracle/RMM/tedlium_lr3e-5_searchlr2e-1.txt`
- `exp/results/repro/oracle/RMM/tedlium_lr6e-5_searchlr2e-1.txt`
- `exp/results/repro/oracle/RMM/tedlium_lr1e-4_searchlr2e-1.txt`
- `exp/results/repro/oracle/RFM/tedlium_lr1e-6_searchlr4e-2.txt`
- `exp/results/repro/oracle/RFM/tedlium_lr8e-6_searchlr9e-2.txt`
- `exp/results/repro/oracle/RFM/tedlium_lr1e-5_searchlr2e-1.txt`
- `exp/results/repro/oracle/RFM/tedlium_lr1e-5_searchlr9e-2.txt`
- `exp/results/repro/oracle/RFM/tedlium_lr8e-6_searchlr2e-1.txt`
- `exp/results/repro/oracle/RFM/tedlium_lr3e-5_searchlr2e-1.txt`
- `exp/results/repro/oracle/RFM/tedlium_lr6e-5_searchlr2e-1.txt`
- `exp/results/repro/oracle/RFM/tedlium_lr1e-4_searchlr2e-1.txt`
- `exp/results/repro/oracle/UVQLM/tedlium_lr1e-5_searchlr2e-1.txt`
- `exp/results/repro/oracle/UVQLM/tedlium_lr3e-5_searchlr2e-1.txt`
- `exp/results/repro/oracle/UVQLM/tedlium_lr6e-5_searchlr2e-1.txt`
- `exp/results/repro/oracle/UVQLM/tedlium_lr1e-4_searchlr2e-1.txt`
- `exp/results/repro/oracle/RAN/tedlium_lr3e-5_searchlr2e-1.txt`

Launcher and log:

- `exp/results/repro/oracle/launch_single_sequential.sh`
- `exp/results/repro/oracle/logs/single_sequential.log`
- `exp/results/repro/oracle/jobs/lr1e-5_searchlr2e-1_gpu.sh`
- `exp/results/repro/oracle/logs/lr1e-5_searchlr2e-1_gpu.log`
- `exp/results/repro/oracle/logs/lr1e-5_searchlr2e-1_queue.log`
- `exp/results/repro/oracle/jobs/lr8e-6_searchlr2e-1_gpu.sh`
- `exp/results/repro/oracle/logs/lr8e-6_searchlr2e-1_gpu.log`
- `exp/results/repro/oracle/logs/lr8e-6_searchlr2e-1_queue.log`
- `exp/results/repro/oracle/jobs/lr1e-5_searchlr9e-2_gpu.sh`
- `exp/results/repro/oracle/logs/lr1e-5_searchlr9e-2_gpu.log`
- `exp/results/repro/oracle/logs/lr1e-5_searchlr9e-2_queue.log`
- `exp/results/repro/oracle/jobs/uvqlm_lr1e-5_searchlr2e-1_gpu.sh`
- `exp/results/repro/oracle/logs/uvqlm_lr1e-5_searchlr2e-1_gpu.log`
- `exp/results/repro/oracle/logs/uvqlm_lr1e-5_searchlr2e-1_queue.log`
- `exp/results/repro/oracle/jobs/lr3e-5_searchlr2e-1_all_policies_gpu.sh`
- `exp/results/repro/oracle/logs/lr3e-5_searchlr2e-1_all_policies_gpu.log`
- `exp/results/repro/oracle/logs/lr3e-5_searchlr2e-1_all_policies_queue.log`
- `exp/results/repro/oracle/jobs/lr1e-4_searchlr2e-1_all_policies_gpu.sh`
- `exp/results/repro/oracle/logs/lr1e-4_searchlr2e-1_all_policies_gpu.log`
- `exp/results/repro/oracle/logs/lr1e-4_searchlr2e-1_all_policies_queue.log`
- `exp/results/repro/oracle/jobs/lr6e-5_searchlr2e-1_all_policies_gpu.sh`
- `exp/results/repro/oracle/logs/lr6e-5_searchlr2e-1_all_policies_gpu.log`
- `exp/results/repro/oracle/logs/lr6e-5_searchlr2e-1_all_policies_queue.log`
- `exp/results/repro/oracle/jobs/ran_lr3e-5_searchlr2e-1_gpu.sh`
- `exp/results/repro/oracle/logs/ran_lr3e-5_searchlr2e-1_gpu.log`
- `exp/results/repro/oracle/logs/ran_lr3e-5_searchlr2e-1_queue.log`
