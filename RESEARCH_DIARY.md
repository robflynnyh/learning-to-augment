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
- Added the ROB-60 newer/default oracle plot at
  `exp/results/repro/oracle/newer_default_oracle_vs_ufmr.pdf` with source CSV
  beside it. The missing matching UFMR segmented policy eval was run at
  `lr=8e-6`, giving `9.041%` WER versus no-adaptation `9.586%`, RFM oracle
  best `8.941%`, and RMM oracle best `8.732%`.
- Regenerated the ROB-60 newer/default oracle plot with a log-scaled repeat
  axis after review feedback that the linear-axis version was visually
  compressed.
- Prepared a follow-up ROB-60 oracle sweep for Robert's requested
  `lr=1e-5`, `search_lr=2e-1` setting. The sweep adds RMM/RFM configs for
  repeats `1 2 3 4 5 10 20 50` and uses a Mimas `with-gpu` screen wrapper with
  a Linear completion callback.

## 2026-05-11

- Finalized the ROB-60 follow-up GPU oracle sweep at `lr=1e-5`,
  `search_lr=2e-1`. RMM reached `8.569%` WER at repeat 50 and RFM reached
  `8.842%` WER at repeat 50, improving over the previous `8e-6/9e-2` oracle
  bests by `0.163` pp and `0.099` pp respectively. Added the small result text
  files plus `exp/results/repro/oracle/oracle_lr_sweep_vs_ufmr.{csv,pdf}`;
  large screen/GPU logs remain uncommitted and are referenced from
  `exp/results/repro/oracle/OUTCOME.md`.
- Prepared a second ROB-60 follow-up oracle sweep for Robert's requested
  `lr=8e-6`, `search_lr=2e-1` setting. Added RMM/RFM configs for repeats
  `1 2 3 4 5 10 20 50` and a Mimas `with-gpu` screen wrapper with a Linear
  completion callback.
