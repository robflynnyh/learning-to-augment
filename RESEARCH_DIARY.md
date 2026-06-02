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

- ROB-108 setup: added a dedicated Mimas wrapper and summarizer for test-split
  RFM/RMM/UFMR/UVQLM evals across TED-LIUM, Earnings22, CHiME-6, Rev16, and
  TAL. Result root is `exp/results/repro/`, with per-policy artifacts under
  top-level method directories such as `exp/results/repro/RFM/`;
  ROB-108-specific README and aggregate files are kept under
  `exp/results/repro/symphony/rob-108/`.
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
- ROB-124 trained the controlled 384-dim/dropout no-audio reward-conditioned
  mask LM. Checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt`;
  best logged dev loss `2.624727`, slightly better than the ROB-117 resumed
  baseline estimate `2.653739`.
- The 512-dim/dropout ablation is kept under
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/old_ablations/rob124_512_dropout/`.
  It was usable but slightly worse than 384d on dev loss (`2.625860`), so 384d
  remains the preferred capacity point.
- ROB-124 downstream reward-conditioning artifacts now live under
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/`.
  The corrected Earnings matched rerun found true sampled `[0.5, 1.0]` best at
  WER `0.194433`; the RMM reward-1 LM reranker improved over unadapted
  Earnings-22 but remained worse than direct reward-conditioned sampling.
- Important caveat: `RewardConditionedMaskLM.augment` had to be fixed so
  adaptation-time calls honor `conditioning_reward_range`. Earlier sampled-range
  labels should not be trusted unless they come from the corrected result roots.
- The all-dataset sampled `[0.5, 1.0]` sweep improved 9/10 cells, with CHiME-6
  at 5 adaptation epochs collapsing to `1.000000` WER. Use 1-epoch adaptation
  or dataset-specific reward/epoch choices rather than a blanket 5-epoch
  setting.
- A later sampled `[0.0, 1.0]` queue was cancelled before GPU evaluation after
  clarification that the intended comparison was two separate fixed-reward
  sweeps. The cancelled scaffold was removed during result-folder cleanup.

## 2026-05-26

- ROB-144 regenerated the thesis handoff artifact for
  `fig:selftrain:layer-drop` under
  `exp/results/repro/selftrain_layer_drop_axis/`. The source rows are copied
  from the existing dynamic-ASR `ctc_self_training_extra_ablation_sweeps`
  summary, and the new plot keeps the same bars and labels while using axes set
  to plus/minus 20% around each dataset panel's average WER. The handoff
  includes both PDF and PNG artifacts.

## 2026-05-27

- Completed the ROB-124 fixed reward `1.0`/`0.0` all-dataset follow-up. The
  durable summary is
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/OUTCOME.md`;
  fixed reward `0.0` improved 10/10 cells and fixed reward `1.0` improved 9/10,
  with the same CHiME-6 5-epoch collapse noted above.

## 2026-05-28

- Cleaned the ROB-124 result tree: current 384/dropout downstream evals are
  grouped under `rob124_384_dropout_reward_conditioning/`, old ablations under
  `old_ablations/`, and stale/cancelled roots were removed. The top-level
  `OUTCOME.md` is the compact interpretation source.
- Added 10k-sample reward-control average-mask visualizations under
  `visualizations/reward_conditioned_average_masks_10k/`. The decoded mask is a
  multiplicative keep mask; figures report masked percentage. The current grid
  compares UC-MLM at `49.13%` masked with `RC-MLM (reward=0.0)` at `70.18%`
  and `RC-MLM (reward=1.0)` at `33.19%`.
- Added the ROB-158 UFMR large-ASR eval scaffold. It reuses the ROB-108 UFMR
  policy matrix but swaps only the ASR checkpoint to the SAP-style
  `/store/store5/data/acp21rjf_checkpoints/SAP_LCASR/n_seq_sched_2048_rp_1/step_105360.pt`
  2048-seq-len, approximately 90M-parameter checkpoint. Launcher:
  `scripts/launch_rob158_ufmr_large_asr_eval.sh`; result root:
  `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/`.

## 2026-05-29

- ROB-158 completed all 15 UFMR large-ASR cells. One-epoch adaptation improved
  all five datasets for both tested LRs and was usually stronger than the
  ROB-108 small-ASR UFMR relative deltas. The five-epoch `1e-5` setting improved
  `tedlium`, `earnings22`, and `chime6` but degraded `rev16` and `TAL`, so the
  large-model handoff supports UFMR transfer most clearly for one-epoch
  adaptation rather than the full ROB-108 recipe.
- Added the ROB-158 RFM large-ASR follow-up scaffold after the UFMR handoff.
  It uses the same 2048-seq-len 90M ASR checkpoint and test datasets but only
  runs RFM at `1e-5` for 1 and 5 adaptation epochs, matching the latest Linear
  clarification to drop `3e-5` RFM trials.

## 2026-05-30

- ROB-158 completed the RFM large-ASR follow-up. RFM improved all 10 requested
  `1e-5` cells and beat the ROB-108 small-ASR RFM relative delta in 8/10 cells.
  On the large-ASR matched cells, UFMR was stronger for all five one-epoch
  comparisons, while RFM was safer at five epochs because it avoided the UFMR
  `rev16` and `TAL` regressions. Final artifacts are under
  `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/`.
- Consolidated the ROB-158 primary outcome so `OUTCOME.md` contains
  both UFMR and RFM aggregate/per-repeat results plus the direct shared-cell
  large-ASR comparison.
- ROB-177 adds a UFMR ablation over `candidate_repeats`
  `1 2 5 10 15 20 40 100 200 1000` on Earnings22 and TED-LIUM test, with
  three seed trials per setting. Here `candidate_repeats` is the UFMR
  `evaluation.augmentation_config.repeats` mask-candidate count, not a seed
  repeat. Use `scripts/launch_rob177_ufmr_repeat_ablation.sh`; the
  UFMR candidate-repeat investigation artifacts live under
  `exp/results/repro/UFMR/candidate_repeat_investigation/`.

## 2026-06-01

- Flattened the ROB-177 UFMR candidate-repeat investigation result layout so
  committed per-trial files and generated configs live directly under
  `exp/results/repro/UFMR/candidate_repeat_investigation/results/` rather than
  a redundant nested method directory.
- Moved the ROB-158 large-ASR transfer artifacts out of the Symphony issue
  staging path into
  `exp/results/repro/large_asr_transfer/ufmr_rfm_90m_seq2048/`. The combined
  result summary is `OUTCOME.md`; method-specific launcher summaries are
  `UFMR_OUTCOME.md` and `RFM_OUTCOME.md`.

## 2026-06-02

- ROB-186 added the EGGROLL-trained forward-only plasticity scaffold for ASR
  test-time adaptation. New entry point: `exp/train_plasticity_eggroll.py`;
  tiny config: `exp/configs/plasticity_eggroll.yaml`; design notes:
  `docs/plasticity_eggroll.md`. This path trains only a shared updater centre
  and keeps transcripts out of the inner rollout except for final WER reward.
- ROB-186 follow-up made the plasticity path GPU-run ready: W&B logging is
  enabled in the real config, `scripts/launch_rob186_plasticity_eggroll_gpu.sh`
  provides callback/config/one-step smoke modes, and checkpoints are explicitly
  updater-only with bounded retention under `exp/results/plasticity_eggroll/`.
- ROB-186 GPU smoke follow-up fixed real LCASR runtime issues before queueing:
  the plasticity target is now the callable attention output projection
  `layers.0.attend.fn.out_proj`, the launcher exports Mimas TEDLIUM/Earnings22
  dataset roots, the config uses 2048-frame chunks, and plasticity rollout
  defaults to no explicit `length` for padded chunks to match existing LCASR
  rollout behavior. One-step smoke succeeded under
  `exp/results/plasticity_eggroll/smoke_20260602T141936Z/`.
