# Research Diary

Keep concise dated notes for project changes that future agents need in order to
reproduce results, interpret metrics, or avoid known failure modes.

## 2026-05-10

- Added Symphony project wiring for the Learn-to-Augment Linear project,
  including split instruction files, detached-job callbacks, wrapper templates,
  and a blocked follow-up issue helper.
- ROB-60 oracle result details live in `exp/results/repro/oracle/OUTCOME.md`;
  keep that file as the durable result summary rather than expanding diary
  notes. Generated plots/CSVs are beside it, and large screen/GPU logs remain
  uncommitted.
- Replaced the superseded ROB-62 `16384` / `9e-5` scaffold with the corrected
  result-repo setup under
  `exp/results/repro/policy/ROB-62_result_repo_2048_1epoch/`. This uses the
  mirrored Spotify 2048 checkpoint, `5e-6`, one evaluation epoch, RFM
  `FrequencyMaskingRanker`, and RMM `MixedMaskingRanker`; use
  `scripts/launch_rob62_result_repo_eval.sh` for the queued run.

## 2026-05-11

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

- Added the ROB-80 TED-LIUM dev policy LR sweep wrapper/summarizer for RFM,
  RMM, and UFMR. Durable tables live in `exp/results/repro/sweeps/`; UFMR
  `1.6e-4` diverged at 5 epochs, so later policy sweeps should stay on the
  lower LR range unless explicitly revisiting instability.

## 2026-05-13

- Added and fixed the ROB-80 segmented-dev policy sweep. Use
  `scripts/launch_rob80_tedlium_segmented_policy_sweep.sh`; it runs
  `oracle_eval.py` with `rollout_setting: policy` because `exp/eval.py` cannot
  consume `tedlium3_segmented_data` utterance-list outputs.
- Updated Symphony rules for ROB-83 to forbid `/tmp` on Mimas for working files,
  logs, downloads, result staging, callbacks, and experiment scratch space. Use
  durable repo-local paths, `exp/results/`, or suitable `/store/...` locations
  for artifacts that later Symphony turns may need to inspect.

## 2026-05-14

- Added ROB-80 no-audio and audio-conditioned CMultiStepVQLM sweep support.
  The compatible no-audio checkpoint is `CMultiStepMLM/no_audio_modelsignals.pt`
  with `condition_on_audio: false` / `use_signal_inputs: true`; the
  audio-conditioned checkpoint is `CMultiStepMLM/curbest.pt` with
  `condition_on_audio: true` / `use_signal_inputs: false`.
- Added fixed-vs-random reward conditioning for CMultiStepVQLM using
  `conditioning_reward_range`; durable comparison tables are under
  `exp/results/repro/sweeps/no_audio_cmultistep_vqlm/` and
  `exp/results/repro/sweeps/audio_cmultistep_vqlm/`.
- Finalized ROB-60 oracle follow-ups across RMM, RFM, UVQLM, and RAN. Best
  completed segmented-test oracle result is RMM `lr=3e-5`, `search_lr=2e-1`,
  repeat 50 at `8.428%` WER; RAN peaks at repeat 5 with `8.885%` WER and
  remains weaker than the matching learned/random mask-search cells. Use
  `exp/results/repro/oracle/OUTCOME.md` plus `oracle_lr_sweep_vs_ufmr.{csv,pdf}`
  for the full table and plot.

## 2026-05-20

- ROB-104 mirrored only This American Life dev/test data from Stanage to
  `/store/store5/data/this_american_life/` on Mimas: 34 valid episodes, 36 test
  episodes, 70 referenced MP3 files, valid/test transcript JSONs, and the
  speaker map. The train transcript and train-only audio are intentionally not
  present. Mimas eval launchers now export `L2A_TAL_DIR` to this path.

## 2026-05-15

- Finalized ROB-80's two-repeat reporting contract. Read the result files, not
  this diary, for metrics: `ROB-80_OUTCOME.md`,
  `segmented_dev/ROB-80_SEGMENTED_OUTCOME.md`,
  `no_audio_cmultistep_vqlm/ROB-80_NOAUDIO_REWARD_CONDITIONING_REPEAT_COMPARISON.md`,
  and `audio_cmultistep_vqlm/ROB-80_AUDIO_REWARD_CONDITIONING_COMPARISON.md`.
  `scripts/launch_rob80_tedlium_missing_repeat2_sweep.sh` only exists to fill
  missing repeat-2 cells and should not be used for a fresh full sweep.

## 2026-05-19

- Added the ROB-60 reduced plot
  `exp/results/repro/oracle/oracle_best_repeat50_lrs_vs_ufmr.{csv,pdf}` for
  the best repeat-50 LR setup per method, then refreshed both oracle plots with
  clearer styling. The reduced plot legend intentionally uses method-only
  labels; selected LR/search-LR details remain in `OUTCOME.md`.

## 2026-05-20

- Resolved ROB-80 PR merge conflicts against current `origin/main`, keeping the
  stricter `VectorQuantize` compatibility helper while preserving the latest
  ROB-60 oracle diary/result artifacts from the base branch.
- Added ROB-82 UVQLM TED-LIUM dev sweep support under
  `exp/results/repro/sweeps/uvqlm/tedlium_dev/`. A later Linear comment on
  2026-05-20 dropped the originally requested segmented-dev half before
  completion, so the launcher and committed artifact set are TED-LIUM dev only.
  The completed 12-cell dev sweep is summarized in
  `ROB-82_TEDLIUM_DEV_UVQLM_OUTCOME.md`; the best averaged row is 5 epochs at
  `5e-6` with updated WER `0.086824` across two repeats.
- Added the ROB-106 low-reward conditioning comparison scaffold for TED-LIUM dev
  no-audio CMultiStepVQLM. It reuses the ROB-80 best comparable setting
  (`epochs=5`, `lr=5e-6`, repeats 1 and 2) and compares fixed reward `1.0`,
  fixed reward `0.0`, and uniform `[0.0, 1.0]` reward conditioning under
  `exp/results/repro/sweeps/no_audio_cmultistep_vqlm/rob106_low_reward_conditioning/`.
- Completed the ROB-106 TED-LIUM dev low-reward conditioning comparison. Across
  two repeats, fixed reward `1.0` did not beat fixed reward `0.0`
  (`0.087764` vs. `0.087625` updated WER), while uniform `[0.0, 1.0]`
  conditioning was best (`0.086244` updated WER).
- Added a ROB-106 CPU-node size-scan path for the Stanage UVQLM rollout folder
  after a follow-up Linear comment requested an exact folder size rather than a
  sample estimate. The scanner reports apparent bytes, allocated bytes, counts,
  and top-level breakdowns; the Slurm wrapper keeps the Linear callback `EXIT`
  trap.
- Recorded the completed ROB-106 UVQLM rollout size scan in
  `exp/results/repro/sweeps/no_audio_cmultistep_vqlm/rob106_low_reward_conditioning/uvqlm_rollout_size/OUTCOME.md`:
  266408 files, 146.22 GiB apparent, and 146.73 GiB allocated.
- Added `scripts/launch_rob106_uvqlm_store4_sync.sh` for the follow-up
  callback-backed Mimas copy from the Stanage UVQLM rollout folder to
  `/store/store4/data/l2augment_rollout_uvqmlm/`. The wrapper uses a real
  `EXIT` trap, logs store4 space before copying, and can be resumed with the
  same command if interrupted.
