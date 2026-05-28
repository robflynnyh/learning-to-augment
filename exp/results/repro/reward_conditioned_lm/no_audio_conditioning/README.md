# No-Audio Reward-Conditioned Mask LM Results

This directory is the result root for the no-audio reward-conditioned mask LM
family introduced in ROB-114 and trained in ROB-117/ROB-124.

## Current Layout

- `OUTCOME.md`: compact source of truth for the current ROB-124 interpretation.
- `rob124_384_dropout/`: main ROB-124 384-hidden-dim, dropout-0.1 training
  artifacts and post-training sanity check.
- `rob124_384_dropout_reward_conditioning/`: active 384/dropout downstream
  reward-conditioning evals.
- `old_ablations/`: older comparison points moved out of the active result
  surface, including the ROB-120 baseline eval and the 512/dropout capacity
  follow-up.
- `scripts/`, `smoke/`, and `visualizations/`: reusable diagnostics and
  reward-conditioned mask visualizations.

## Cleanup Notes

The stale initial `rob124_384_dropout_earnings_reward_controls/` root was
removed. It was superseded by the corrected Earnings rerun now stored at
`rob124_384_dropout_reward_conditioning/earnings_reward_controls/`, because the
corrected rerun fixed adaptation-time reward-range sampling.

The cancelled sampled `[0.0, 1.0]` all-dataset scaffold
`rob124_384_dropout_all_dataset_reward_sampling_0to1/` was also removed. It
never ran GPU eval and was replaced by the completed fixed-reward `0.0` and
`1.0` sweep under
`rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/`.

## Main Artifacts

- ROB-117 baseline checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_resume100_500ep_lr1e3.pt`
- ROB-124 384/dropout checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt`
- ROB-124 512/dropout checkpoint, kept as an old ablation:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_512d_dropout0p1_500ep_lr1e3.pt`
- ROB-124 reward-control average masks:
  `visualizations/reward_conditioned_average_masks_10k/`

See `OUTCOME.md` for the consolidated interpretation and the subdirectory
OUTCOME files for detailed tables.
