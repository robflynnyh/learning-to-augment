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
- Replaced the superseded ROB-62 `16384` / `9e-5` scaffold with the corrected
  result-repo setup under
  `exp/results/repro/policy/ROB-62_result_repo_2048_1epoch/`. This uses the
  mirrored Spotify 2048 checkpoint, `5e-6`, one evaluation epoch, RFM
  `FrequencyMaskingRanker`, and RMM `MixedMaskingRanker`; use
  `scripts/launch_rob62_result_repo_eval.sh` for the queued run.

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
- Finalized the second ROB-60 follow-up oracle sweep at `lr=8e-6`,
  `search_lr=2e-1`. RMM reached `8.626%` WER at repeat 50 and RFM reached
  `8.892%` WER at repeat 50, improving over the previous `8e-6/9e-2` oracle
  bests by `0.106` pp and `0.049` pp respectively. Updated
  `exp/results/repro/oracle/OUTCOME.md` and regenerated
  `exp/results/repro/oracle/oracle_lr_sweep_vs_ufmr.{csv,pdf}` to include the
  completed curves; large screen/GPU logs remain uncommitted.
- Prepared a third ROB-60 follow-up oracle sweep for Robert's requested
  `lr=1e-5`, `search_lr=9e-2` setting after confirming that no completed
  artifacts existed for that cell. Added RMM/RFM configs for repeats
  `1 2 3 4 5 10 20 50` and a Mimas `with-gpu` screen wrapper with a Linear
  completion callback.
- Completed the ROB-62 corrected result-repo comparison on Mimas for TED-LIUM,
  Earnings-22, Rev16, and CHiME-6. Small result summaries and comparison tables
  are under `exp/results/repro/policy/ROB-62_result_repo_2048_1epoch/`; TAL
  remains missing because it was not mirrored on this host.
- Added ROB-71 grid-style config support via `exp/run_config_grid.py` and
  converted the repetitive oracle eval YAML fanout to `tedlium_grid.yaml`
  configs that materialize one-run YAMLs under ignored `.generated/`
  directories at launch time.
- Followed up ROB-71 by converting the remaining repro configs under
  `exp/results/repro/` to grid YAMLs and adding paired-case/product grid support
  for runs that sweep repeats over matched `lr` / `single_step_lr` settings.
- Added ROB-73 unconditional VQ mask LM sample rendering under
  `exp/results/repro/unconditional_lm/ROB-73_sample/`, using the trained UMLM
  and BVAE checkpoints cached from Stanage into the local Mimas checkpoint cache.
- Extended the ROB-73 render to 10 unconditional VQ mask LM samples with an
  overview grid, per-sample PNG/PDF masks, toy-spectrogram visualizations, and
  per-sample VQ-code metadata.
- Reorganized `exp/results/` for ROB-74: older reference artifacts now live
  under `exp/results/historical_results/`, while newer resumed-work outputs
  remain under `exp/results/repro/`. Treat the historical numbers as reference
  material until they are reverified.

## 2026-05-12

- Finalized the third ROB-60 follow-up oracle sweep at `lr=1e-5`,
  `search_lr=9e-2`. RMM reached `8.612%` WER at repeat 10 and RFM reached
  `8.977%` WER at repeat 10. The RMM result improves on the previous
  `8e-6/9e-2` RMM best, but neither method beats the existing `1e-5/2e-1`
  best. Added the small result text files, updated
  `exp/results/repro/oracle/OUTCOME.md`, and regenerated
  `exp/results/repro/oracle/oracle_lr_sweep_vs_ufmr.{csv,pdf}` to include the
  completed curves; large screen/GPU logs remain uncommitted.
- Prepared Robert's requested UVQLM oracle follow-up after the policy-scope
  discussion. The queued setup uses repo-local UMLM/BVAE checkpoints under
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/`, evaluates
  `UnconditionalMaskGenerator` proposals at `lr=1e-5`, `search_lr=2e-1`, and
  sweeps repeats `1 2 3 4 5 10 20 50` with a Mimas `with-gpu` screen wrapper
  and Linear completion callback.

## 2026-05-13

- Finalized the ROB-60 UVQLM oracle follow-up at `lr=1e-5`,
  `search_lr=2e-1`. UVQLM reached `8.488%` WER at repeat 50, improving over
  the previous best RMM `1e-5/2e-1` oracle result by `0.081` pp and over the
  matching RFM result by `0.354` pp. Added the small UVQLM result text file,
  updated `exp/results/repro/oracle/OUTCOME.md`, and regenerated
  `exp/results/repro/oracle/oracle_lr_sweep_vs_ufmr.{csv,pdf}` to include the
  completed UVQLM curve; large screen/GPU logs remain uncommitted.
- Prepared Robert's requested ROB-60 `lr=3e-5`, `search_lr=2e-1` oracle
  follow-up for RMM, RFM, and UVQLM after confirming no completed artifacts
  existed for that cell. Added the grid case to the reproducible oracle configs,
  kept new result paths workspace-relative, and added a Mimas `with-gpu` screen
  wrapper with a Linear completion callback.
- Finalized the ROB-60 `lr=3e-5`, `search_lr=2e-1` oracle follow-up. RMM
  reached `8.428%` WER at repeat 50, UVQLM reached `8.467%`, and RFM reached
  `8.817%`; all three curves were best at repeat 50. The new best overall
  oracle result is RMM `3e-5/2e-1`, improving over the previous UVQLM
  `1e-5/2e-1` best by `0.060` pp. Added the small result text files, updated
  `exp/results/repro/oracle/OUTCOME.md`, and regenerated
  `exp/results/repro/oracle/oracle_lr_sweep_vs_ufmr.{csv,pdf}` to include the
  completed curves; large screen/GPU logs remain uncommitted.
- Prepared a higher-LR ROB-60 oracle follow-up after Robert noted that
  `3e-5` is the best setting so far. Added a reproducible `lr=1e-4`,
  `search_lr=2e-1` cell for RMM, RFM, and UVQLM, plus a Mimas `with-gpu`
  screen wrapper with a Linear completion callback.

## 2026-05-14

- Finalized the ROB-60 higher-LR `lr=1e-4`, `search_lr=2e-1` oracle follow-up.
  The setting did not improve the best result: UVQLM reached `8.821%` WER,
  RMM reached `8.927%`, and RFM reached `9.204%`, all at repeat 50. RMM
  `3e-5/2e-1` remains best overall at `8.428%`. Added the small result text
  files, updated `exp/results/repro/oracle/OUTCOME.md`, and regenerated
  `exp/results/repro/oracle/oracle_lr_sweep_vs_ufmr.{csv,pdf}` to include the
  completed curves; large screen/GPU logs remain uncommitted.
