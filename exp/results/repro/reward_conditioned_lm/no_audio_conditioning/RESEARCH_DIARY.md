# ROB-117 Research Diary

## 2026-05-22

This file is an issue-local pointer for future agents; the root
`RESEARCH_DIARY.md` remains the project-level diary.

- Trained the no-audio reward-conditioned mask LM introduced by ROB-114 on the
  `/store/store4/data/l2augment_rollout_uvqmlm/{train,dev}` rollout tree. The
  corrected Mimas retry completed 100 epochs and wrote
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance.pt`.
- A resume-from-100 follow-up used LR `1e-3`, target epoch cap `500`, and W&B
  run `5ny25k7g`. Dev-loss patience fired without improvement over the first
  resumed validation pass, so
  `no_audio_tedlium_per_utterance_resume100_500ep_lr1e3.pt` is usable for
  downstream eval/oracle comparison but should not be described as a validated
  improvement over the 100-epoch checkpoint.
- `exp/train_freq_mask.py` now resets validation losses per validation pass.
  Older ROB-117 `avg_val_loss` logs were cumulative within each process; the
  reconstructed resumed-run validation losses still support the same rollback
  decision.
- Post-training diagnostics confirmed fixed-length generation at reward
  controls `0.0` and `1.0`, added the small sampled reward-control and
  adaptation-WER JSON checks, and generated the requested 10 sampled masks.
  The committed mask summary is
  `post_training_10_sampled_masks_reward_0_vs_1.json`.
- Visual mask artifacts for the 10 samples are under `visualizations/`,
  including `reward_conditioned_mask_samples_grid.{png,pdf}` and individual
  per-sample mask PDF/PNG files.
