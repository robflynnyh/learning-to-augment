# ROB-117 Research Diary

## 2026-05-22

- Completed the corrected Mimas retry for the no-audio reward-conditioned mask
  LM training run. The wrapper exited with status `0`, the run reached
  `100/100` epochs, and the final dev loss was `2.6927917954301535`.
- Final checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance.pt`.
- W&B run: `dry-thunder-2166`
  (`https://wandb.ai/wobrob101/l2augment/runs/5ny25k7g`).
- Post-training sanity loaded the checkpoint and confirmed fixed-length
  generation on `/store/store4/data/l2augment_rollout_uvqmlm/dev/AlGore_2009_0.pt`
  at reward controls `0.0` and `1.0`: both produced 29 VQ tokens, mask shape
  `[1, 80, 1042]`, and augmented audio shape `[1, 80, 1042]`.
- Interpretation: the checkpoint is usable for downstream fixed-length
  eval/oracle comparison, but downstream WER/oracle scoring is still needed to
  measure augmentation quality.
- Followed up on the Linear request to sample at reward controls `0.0` and
  `1.0` on three different TED-LIUM dev recordings. The sampled diagnostic
  wrote
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/post_training_sampled_reward_0_vs_1_check.json`;
  all three recordings produced fixed-length masks and different reward-0 vs
  reward-1 token sequences.
- Added the clarified WER-before/after-adaptation diagnostic for the same three
  recordings. With sampled ROB-117 masks and one `lr=1e-5` adaptation epoch, all
  six reward-control runs improved WER after adaptation; reward `1.0` was better
  than reward `0.0` on two recordings and worse on one. Full results are in
  `post_training_adaptation_wer_reward_0_vs_1.json`.
