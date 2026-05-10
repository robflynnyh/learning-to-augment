# Research Diary

Keep concise dated notes for project changes that future agents need in order to
reproduce results, interpret metrics, or avoid known failure modes.

## 2026-05-10

- Added Symphony project wiring for the Learn-to-Augment Linear project,
  including split instruction files, detached-job callbacks, wrapper templates,
  and a blocked follow-up issue helper.
- Finalized ROB-60 oracle reproduction results in
  `exp/results/repro/oracle/OUTCOME.md`: the active sequential CPU sweep
  completed for RMM/RFM at `1e-6/4e-2` and `8e-6/9e-2`; best result was RMM
  `lr=8e-6`, `search_lr=9e-2`, repeat 20, at `8.732%` WER on TEDLIUM3
  segmented test.
