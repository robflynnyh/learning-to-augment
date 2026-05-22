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
  the best previous state. First resumed validation loss was
  `2.653739192269065`; final logged validation loss before rollback was
  `2.6558073686830923`.
- The resumed checkpoint is
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_resume100_500ep_lr1e3.pt`.
  Post-training sanity loaded it and confirmed fixed-length generation at
  reward controls `0.0` and `1.0` on `AlGore_2009_0.pt`; both produced 29 VQ
  tokens, `[1, 80, 1042]` masks, and `[1, 80, 1042]` augmented audio. Treat the
  checkpoint as usable for downstream eval/oracle comparison, but not as a
  validated improvement over the 100-epoch checkpoint.
