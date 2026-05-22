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

## 2026-05-21

- Verified ROB-109 UVQLM dev rollout provenance and reward tensor semantics in
  `exp/results/repro/unconditional_lm/ROB-109_rollout_verification/OUTCOME.md`.
  The Mimas UMLM/BVAE checkpoints are byte-identical to the Stanage checkpoint
  paths used by the original UMLM generation config, and the synced dev rollout
  files use reward shape `(10, 2, 2)` as 10 masks by `[CER, WER]` by
  `[before, after]`.
- Extended ROB-109 after review with behavior-level VQ-sequence verification:
  `exp/results/scripts/verify_rob109_uvqlm_rollout_sequences.py` re-decodes all
  5,070 saved dev `generation` sequences through the current Mimas BVAE
  checkpoint and scores them with the current Mimas UMLM checkpoint. The summary
  JSON in the ROB-109 result directory records 12 mismatched mask pixels out of
  460,274,400 and saved-sequence mean NLL `2.8811` versus random-code mean NLL
  `10.7293`.
- Added the ROB-111 review plan for a no-audio reward-conditioned mask LM under
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/`. The plan
  trains from saved ROB-109 VQ `generation` sequences, keeps the large rollout
  dataset in place, uses per-utterance WER-delta controls, and uses
  fixed-length audio-derived generation without EOS supervision.
- Implemented the ROB-114 no-audio reward-conditioned mask LM path:
  `RewardConditionedMaskLMDataset`, `RewardConditionedMaskLM` collate,
  `RewardConditionedMaskLM`, active full/smoke configs, and smoke validation
  script under
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/`. The model
  conditions the first decoder input on normalized reward and trains only on
  fixed-length saved VQ generations, without EOS targets.
- Validated the initial ROB-114 implementation with the Torch 2.8 `speech-diff`
  environment: 2-file
  train/dev stats, finite CPU CE/backward, fixed 7-token generation at two
  reward values, real-rollout augment length check, and a one-file
  `exp/train_freq_mask.py` smoke with W&B disabled.
- Addressed ROB-114 PR review by switching reward controls to bounded per-file
  min-max normalization with degenerate groups mapped to `0.5`, using multistep
  evaluation in the active config, increasing full-run early-stop tolerance to
  5, and validating the path in the normal `flash_attn_pytorch2` project env.
- Removed the remaining ROB-114 import/load guards after PR review: `lcasr`,
  `torchaudio`, and `eval` now fail at import time as in the project env, and
  ROB-109 rollout files are loaded with the normal `torch.load` path. The
  no-guard direct-load validation was rerun under the bashrc Python 3.10 /
  Torch `2.6.0+cu124` runtime.
- Resolved the ROB-114 environment command docs to use `bash -ic python`, where
  `python` is aliased to `/usr/bin/python3.10`, and set trusted local policy
  checkpoint loads to `weights_only=False` for Torch 2.6.
- Started ROB-117 no-audio reward-conditioned mask LM training from the merged
  ROB-114 commit. The bashrc Python 3.10 / Torch 2.6 smoke passed, the full
  config still has W&B logging and dev-loss early stopping enabled, and
  `scripts/launch_rob117_reward_conditioned_mask_lm_training.sh` records the
  detached Mimas `with-gpu 1,2` launch plus the callback trap contract.
- Diagnosed and restarted the first ROB-117 full-training attempt after a
  DataLoader multiprocessing `OSError: AF_UNIX path too long` before the first
  dev batch. The BVAE load in the log is the frozen binary mask VAE used for
  the mask-token codebook/decoder, not audio conditioning. The launcher now
  defaults to short durable local scratch at `/exp/exp4/acp21rjf/rob117-scratch`,
  and the callback path, two-worker DataLoader smoke, and smoke-config training
  run were revalidated before requeue.
- Finalized ROB-117 after the corrected Mimas retry completed with exit status
  0. The trained no-audio reward-conditioned mask LM checkpoint is
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance.pt`;
  final logged dev loss before the metric reset fix was `2.6927917954301535`,
  W&B run was `5ny25k7g`, and post-training sanity confirmed fixed-length
  generation at reward controls `0.0` and `1.0` on a real TED-LIUM dev rollout.
  The logged value was cumulative across validation passes in that process, not
  a standalone epoch-100 checkpoint loss.
- Extended ROB-117 post-training validation after a Linear follow-up by
  sampling the trained checkpoint at reward controls `0.0` and `1.0` on three
  TED-LIUM dev recordings. The sampled check produced valid fixed-length masks
  for all recordings and different token sequences between the two reward
  controls.
- Added the ROB-117 clarified WER diagnostic for those same recordings using
  `cpu_rollout_policy`, sampled reward-conditioned masks, and one `lr=1e-5`
  adaptation epoch. All six reward-control runs improved WER after adaptation;
  reward `1.0` was better on two recordings and reward `0.0` was better on one,
  so this remains a small diagnostic rather than a full dev-set conclusion.
- Prepared the ROB-117 follow-up full-training run requested after the 100-epoch
  model kept decreasing in dev loss. The follow-up config uses the same rollout
  root and model family with `training.epochs: 500`, `policy.lr: 1e-3`, W&B
  enabled, dev-loss early stopping tolerance 5, and a separate
  `no_audio_tedlium_per_utterance_500ep_lr1e3.pt` checkpoint path.
- Adjusted the ROB-117 follow-up after Robert clarified that it should resume
  from the completed 100-epoch checkpoint and resume the same W&B run. The new
  resume config starts at epoch `100`, targets epoch `500`, uses LR `1e-3`,
  loads `no_audio_tedlium_per_utterance.pt`, resumes W&B run `5ny25k7g`, and
  writes to `no_audio_tedlium_per_utterance_resume100_500ep_lr1e3.pt`.
- Completed the ROB-117 resume-100 500-epoch LR `1e-3` run. It exited cleanly
  from detached Mimas screen `rob117-reward-conditioned-mask-lm-resume100-500ep-lr1e3`
  after dev-loss patience fired; first resumed validation pass reported
  `2.653739192269065`, the best available standalone validation estimate for
  the loaded 100-epoch checkpoint. The final logged validation value before
  rollback was `2.6558073686830923`, cumulative within the resumed process
  before the metric reset fix. The saved checkpoint
  `no_audio_tedlium_per_utterance_resume100_500ep_lr1e3.pt` passed a
  fixed-length generation sanity check at reward controls `0.0` and `1.0`, so
  it is usable for downstream eval/oracle comparison, but the LR `1e-3` resume
  did not validate an improvement over the 100-epoch state.
- Fixed `exp/train_freq_mask.py` so validation-loss accumulation resets per
  validation pass. Older ROB-117 `avg_val_loss` logs should be read as
  cumulative within-process averages.
  The old early-stopping signal was smoothed by that cumulative average; for
  the resumed LR `1e-3` run, reconstructed per-validation losses still support
  the same rollback-to-starting-checkpoint decision.
