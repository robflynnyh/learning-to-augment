# ROB-117 Research Diary

## 2026-05-22

- Completed the corrected Mimas retry for the no-audio reward-conditioned mask
  LM training run. The wrapper exited with status `0`, the run reached
  `100/100` epochs. The final logged dev loss before the metric reset fix was
  `2.6927917954301535`, but that was cumulative across validation passes in the
  process rather than the standalone epoch-100 checkpoint loss.
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
- Prepared the requested follow-up full-training config after Robert noted the
  100-epoch run was still decreasing in loss. The new config is
  `tedlium_per_utterance_500ep_lr1e3.yaml`, uses `training.epochs: 500` and
  `policy.lr: 1e-3`, and saves to a distinct `500ep_lr1e3` checkpoint path so
  the completed 100-epoch checkpoint remains available.
- Stopped the fresh `500ep_lr1e3` run after Robert clarified that the follow-up
  should resume from the 100-epoch checkpoint and continue the same W&B run.
  Added a resume config and launcher that load
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance.pt`,
  start logging at epoch `100`, keep the target epoch cap at `500`, use LR
  `1e-3`, and resume W&B run `5ny25k7g`.
  The resume smoke and actual launcher callback-only check passed before
  queueing.
- The resume-100 500-epoch LR `1e-3` run exited cleanly with status `0` from
  detached Mimas screen `rob117-reward-conditioned-mask-lm-resume100-500ep-lr1e3`.
  Dev-loss patience fired after five non-improving validation passes, restoring
  the best previous state. The first resumed validation pass reported
  `2.653739192269065`, the best available standalone validation estimate for
  the loaded 100-epoch checkpoint. The final logged validation value before
  rollback was `2.6558073686830923`; before the metric reset fix this was
  cumulative within the resumed process.
- The resumed checkpoint is
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_resume100_500ep_lr1e3.pt`.
  Post-training sanity loaded it and confirmed fixed-length generation at
  reward controls `0.0` and `1.0` on `AlGore_2009_0.pt`; both produced 29 VQ
  tokens, `[1, 80, 1042]` masks, and `[1, 80, 1042]` augmented audio. Treat the
  checkpoint as usable for downstream eval/oracle comparison, but not as a
  validated improvement over the 100-epoch checkpoint.
- Fixed `exp/train_freq_mask.py` so validation losses are reset for each
  validation pass instead of accumulating across epochs. Older ROB-117 logs
  before this fix should be interpreted as cumulative within-process averages.
  The old early-stopping signal was smoothed by that cumulative average; for
  the resumed LR `1e-3` run, reconstructed per-validation losses still support
  the same rollback-to-starting-checkpoint decision.
- Generated Robert's requested 10-mask sample from the trained policy: 5
  sampled masks at reward `0.0` and 5 at reward `1.0`, using the resumed
  checkpoint and `AlGore_2009_0.pt`. The committed summary is
  `post_training_10_sampled_masks_reward_0_vs_1.json`; the ignored local tensor
  bundle `post_training_10_sampled_masks_reward_0_vs_1.pt` contains the actual
  decoded `[10, 1, 80, 1042]` masks, reward controls, seeds, and generations.
