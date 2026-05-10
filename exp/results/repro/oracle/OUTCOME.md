# Oracle Reproduction Outcome

Issue: ROB-60
Date: 2026-05-10

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

## Summary

Baseline original WER was `9.586%` for all rows.

| Method | Setup | Best repeat | Best WER | Abs. gain | Relative gain | Repeat 50 WER |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RMM | `lr=8e-6`, `search_lr=9e-2` | 20 | 8.732% | 0.854 pp | 8.9% | 8.789% |
| RFM | `lr=8e-6`, `search_lr=9e-2` | 10 | 8.941% | 0.645 pp | 6.7% | 9.005% |
| RMM | `lr=1e-6`, `search_lr=4e-2` | 20 | 8.991% | 0.595 pp | 6.2% | 9.069% |
| RFM | `lr=1e-6`, `search_lr=4e-2` | 50 | 9.186% | 0.400 pp | 4.2% | 9.186% |

## Interpretation

The newer/default setup is better than the historical matched setup for both
masking policies. RMM is consistently ahead of RFM in the matched comparison:

- Historical matched best WER: RMM `8.991%` vs RFM `9.186%`
- Newer/default best WER: RMM `8.732%` vs RFM `8.941%`

Increasing repeat count generally helps, but the optimum does not always occur
at repeat 50. RMM peaked at repeat 20 in both setups; RFM peaked at repeat 10
for the newer/default setup and repeat 50 for the historical setup.

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

Artifacts:

- `exp/results/repro/oracle/newer_default_oracle_vs_ufmr.pdf`
- `exp/results/repro/oracle/newer_default_oracle_vs_ufmr.csv`
- `exp/results/repro/policy/UFMR_segmented/tedlium_lr8e-6.txt`
- `exp/results/repro/policy/UFMR_segmented/logs/tedlium_lr8e-6.log`

## Evidence

Result files:

- `exp/results/repro/oracle/RMM/tedlium_lr1e-6_searchlr4e-2.txt`
- `exp/results/repro/oracle/RMM/tedlium_lr8e-6_searchlr9e-2.txt`
- `exp/results/repro/oracle/RFM/tedlium_lr1e-6_searchlr4e-2.txt`
- `exp/results/repro/oracle/RFM/tedlium_lr8e-6_searchlr9e-2.txt`

Launcher and log:

- `exp/results/repro/oracle/launch_single_sequential.sh`
- `exp/results/repro/oracle/logs/single_sequential.log`
