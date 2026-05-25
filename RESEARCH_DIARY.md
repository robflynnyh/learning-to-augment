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
- Completed ROB-117 training for the no-audio reward-conditioned mask LM. The
  corrected 100-epoch Mimas run wrote
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance.pt`;
  the first failure was a DataLoader `AF_UNIX path too long` issue fixed by
  using `/exp/exp4/acp21rjf/rob117-scratch`.
- Ran the requested resume-from-100 follow-up with LR `1e-3`, target epoch cap
  `500`, and resumed W&B run `5ny25k7g`. Early stopping restored the first
  resumed state, so
  `no_audio_tedlium_per_utterance_resume100_500ep_lr1e3.pt` is usable for
  downstream eval/oracle comparison but not evidence of improvement over the
  100-epoch checkpoint.
- Fixed validation logging in `exp/train_freq_mask.py` so future dev losses are
  per-validation-pass values. Older ROB-117 logs were cumulative within each
  process; reconstructed resumed-run losses still support the rollback decision.
- Added small post-training diagnostics for fixed-length reward-controlled
  generation, sampled reward `0.0` vs `1.0` masks, and a three-recording
  adaptation-WER check. The 10-mask sample now has committed PDF/PNG
  visualizations under
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/visualizations/`.
- Started ROB-124 as the controlled ROB-117 capacity/dropout follow-up. The new
  config trains the same no-audio reward-conditioned mask LM with
  `hidden_dim: 384` and `dropout: 0.1`, using the same UVQLM rollout data,
  reward normalization, Mimas callback wrapper discipline, and a separate
  result root under
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout/`.
- Completed the ROB-124 384-dim/dropout training run. The callback-backed Mimas
  run wrote
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt`;
  post-training fixed-length reward `0.0`/`1.0` sanity passed, and the best
  logged dev loss `2.624727` modestly improved over the ROB-117 resumed
  baseline estimate `2.653739`.
- Set up ROB-120 Earnings-22 reward-control evaluation for the ROB-117
  `no_audio_tedlium_per_utterance_resume100_500ep_lr1e3.pt` checkpoint. The
  result root is
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob120_earnings_reward_controls/`;
  the wrapper generates fixed `0.0`, fixed `1.0`, uniform `[0.0, 1.0]`, and
  uniform `[0.5, 1.0]` configs, then evaluates Earnings test adaptation at
  `lr=1e-5`. Checkpoint-load/generation preflight and a cropped Earnings CPU
  adaptation smoke passed; the full GPU comparison should be interpreted only
  from the wrapper-generated CSV/`OUTCOME.md`, not from the cropped smoke.
- Started the ROB-124 follow-up Earnings-22 reward-control evaluation for the
  384-dim/dropout checkpoint. The wrapper mirrors ROB-120's four reward-control
  conditions but uses
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt`
  with `hidden_dim: 384` and `dropout: 0.1`; the result root is
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_earnings_reward_controls/`.
- Added the ROB-124 512-dim/dropout follow-up scaffold after the latest Linear
  comment requested another capacity comparison. It keeps the completed
  384/dropout contract but sets `hidden_dim: 512`, writes to
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_512_dropout/`,
  and uses checkpoint path
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_512d_dropout0p1_500ep_lr1e3.pt`.
  The one-file smoke passed under the bashrc Python 3.10/Torch 2.6 path with
  CUDA hidden because all Mimas GPUs were busy; the full run should use the
  validated `with-gpu 1,2` callback wrapper. The full run was queued on
  2026-05-23 as screen `rob124-reward-conditioned-mask-lm-512d-dropout0p1`,
  ticket `32c3350a`, from commit `c36c89ee6ea5ef5be0433cd8c404026fc3009c0f`.
- Completed the ROB-124 512-dim/dropout follow-up. The callback-backed Mimas
  run wrote
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_512d_dropout0p1_500ep_lr1e3.pt`;
  post-training fixed-length reward `0.0`/`1.0` sanity passed with `29/29`
  reward-control token mismatches. Its best logged dev loss `2.625860` still
  improves over the ROB-117 resumed baseline `2.653739`, but is slightly worse
  than the ROB-124 384/dropout checkpoint's `2.624727`, so the 384/dropout
  model remains the better current capacity point.
- Started the ROB-124 RMM proposal plus reward-1 LM-rerank eval requested after
  the 384/dropout checkpoint was confirmed as the preferred model. The new
  policy generates 15 RMM candidate masks at each adaptation step, encodes each
  mask with the mask BVAE, scores the VQ tokens with the 384/dropout
  reward-conditioned mask LM at fixed reward `1.0`, then uses the lowest-CE
  mask. The result root is
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_rmm_lm_rerank/`.
  Prequeue checks included config generation, callback check-only, a synthetic
  15-candidate policy augment smoke, and a cropped CPU Earnings multistep
  rollout smoke under bashrc Python 3.10 / Torch 2.6.
- Completed the ROB-124 RMM proposal plus reward-1 LM-rerank eval. The
  callback-backed Mimas run exited with status `0` and wrote
  `rob124_384_dropout_rmm_lm_rerank.csv` plus `OUTCOME.md`; updated WER was
  `0.202377` from original WER `0.235239`. This is better than unadapted
  Earnings-22 but worse than the previous ROB-124 fixed reward `1.0` condition
  (`+0.006923` absolute WER) and worse than the best prior ROB-124 condition
  (`+0.007434`), so direct reward-conditioned sampling remains the stronger
  use of the 384/dropout checkpoint for this matched eval.
- Started the 2026-05-24 ROB-124 all-dataset follow-up requested after ROB-108:
  evaluate the preferred 384/dropout checkpoint with reward sampled from
  `[0.5, 1.0]` on TED-LIUM, Earnings22, CHiME-6, Rev16, and TAL test splits for
  1 and 5 adaptation epochs at `lr=1e-5`. The result root is
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_reward_sampling/`.
  This setup also fixes `RewardConditionedMaskLM.augment` so adaptation-time
  calls honor `conditioning_reward_range`; without that fix, the new sampled
  reward eval would use the default fixed reward instead.
- Paused that all-dataset follow-up before it started after the later Linear
  comments asked whether previous sampled-reward comparisons were wrong and
  suggested redoing the matched comparison first. The queued all-dataset ticket
  was cancelled, and a corrected Earnings-22 rerun root was added at
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_earnings_reward_controls_corrected_sampling/`.
  The corrected wrapper reuses the original ROB-124 Earnings reward-control
  launcher but writes to a separate root and uses the fixed reward-range
  sampling path.
- Completed the corrected ROB-124 Earnings-22 matched reward-control rerun on
  2026-05-24. All four cells completed; true uniform `[0.5, 1.0]` was best at
  updated WER `0.194433`, beating the matched ROB-120 row by `0.002001`
  absolute WER. This supports resuming the paused all-dataset `[0.5, 1.0]`
  sampled-reward eval for the 384/dropout checkpoint.
- Completed the ROB-124 all-dataset `[0.5, 1.0]` sampled-reward follow-up on
  2026-05-25. The callback-backed Mimas run exited with status `0`, completed
  all 10 cells, and wrote
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_reward_sampling/OUTCOME.md`.
  Nine of ten cells improved WER versus the unadapted original WER; the only
  regression was CHiME-6 at 5 adaptation epochs, which moved from `0.843620` to
  `1.000000`. The result supports the 384/dropout checkpoint and sampled
  reward `[0.5, 1.0]`, especially at 1 adaptation epoch, but longer adaptation
  should be treated as dataset-sensitive.
- Started ROB-132 audio+reward-conditioned mask LM work from the ROB-124 PR
  head because the implementation depends on ROB-124's dropout and
  reward-range fixes. Added `AudioRewardConditionedMaskLM`, a 384-dim
  transformer decoder with rotary self-attention and cross-attention to frozen
  HuBERT-base SSL features. The HuBERT sidecar builder maps TED-LIUM rollout
  filenames back to STM utterance indices, loads raw `.sph` segments from
  `/store/store4/data/TEDLIUM_release-3/legacy`, and stores only
  mask-token-aligned fp16 features under `/store/store5`. A CPU smoke of the
  actual wrapper built two train/two dev sidecars, ran one tiny training epoch,
  and saved a smoke checkpoint; deterministic generation sanity passed at
  reward controls `0.0` and `1.0`.
- Updated ROB-132 after the follow-up comment that on-the-fly SSL computation
  is acceptable. The training dataset now maps each TED-LIUM rollout to its raw
  utterance segment and extracts frozen HuBERT-base features directly in
  `__getitem__`; the sidecar builder remains only for verification/debug reuse.
  The main and smoke configs use `ssl_feature_mode: on_the_fly`, so full
  training no longer requires a precomputed SSL feature cache.
- Corrected ROB-132 after follow-up Linear comments on positional information
  and HuBERT placement. The first full run from commit
  `df72289324412e43d1f274037062382b9153953d` was interrupted because it used
  content-only cross-attention to SSL memory and `ssl_device: cpu`. The model
  now keeps native HuBERT frame sequences, applies RoPE to cross-attention Q/K
  as well as decoder self-attention, and scales native SSL key positions onto
  the mask-token time grid for cross-attention. The ROB-132 configs run frozen
  HuBERT extraction on CUDA with `num_workers: 0` to avoid CUDA work inside
  DataLoader workers.
- Validated the corrected ROB-132 path with a queued native-HuBERT CUDA smoke
  on Mimas ticket `26b4debf` at commit
  `90f8c466a34147442a5cf89cf26b156be31c684b`; the wrapper completed one tiny
  train/validation pass and refreshed the ignored smoke checkpoint. A follow-up
  GPU generation sanity wrote
  `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384/smoke/post_training_generation_sanity_native_gpu.json`.
- Fixed the ROB-132 native-HuBERT full-run OOM from commit
  `c9746ad723cf809f52915c18eff8ac680cd80fdc`. Each TED-LIUM rollout has 10
  candidate mask sequences, so the old audio collate duplicated the same
  HuBERT tensor 10 times and projected 480 audio memories for a `batch_size:
  48` validation batch. The collate/model path now keeps one padded SSL tensor
  per rollout and uses `audio_item_idxs` to map projected audio memory back to
  candidate rows. The training loop also runs validation under
  `torch.no_grad()` and keeps early-stopping state snapshots on CPU. Targeted
  Mimas checks after the fix passed two full-size dev batches at peaks 5.7 GB
  and 7.25 GB, one train backward batch at 8.22 GB, callback check-only, and
  the real smoke wrapper with callbacks disabled.
