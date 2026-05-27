# ROB-132 Audio SSL Test-Set Fixed-Reward Eval

This result root tracks the follow-up test-set evaluation requested after the
completed TED-LIUM dev fixed-reward ROB-132 handoff.

Scope:

- Policy checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/audio_ssl_hubert_base_tedlium_per_utterance_transformer384_dropout0p1_500ep_lr1e3.pt`
- Policy shape: `AudioRewardConditionedMaskLM`, HuBERT SSL conditioning,
  transformer decoder, `hidden_dim=384`, `dropout=0.1`.
- Reward controls: fixed `conditioning_reward: 1.0` and fixed
  `conditioning_reward: 0.0` as separate runs.
- Datasets: TED-LIUM test and Earnings22 test.
- Adaptation cells: `epochs=1` and `epochs=5`, both with `lr=1e-5`.
- Total cells: `2 datasets * 2 epochs * 2 fixed rewards = 8`.
- Result CSV: `rob132_audio_ssl_self_train_test_sets_fixed_rewards.csv`.
- Outcome: `OUTCOME.md`.

The wrapper is:

```bash
scripts/launch_rob132_audio_ssl_self_train_test_sets.sh
```

Implementation note: configs set both
`evaluation.augmentation_config.conditioning_reward` and
`policy.config.default_conditioning_reward` to the same scalar. They do not set
`conditioning_reward_range`, so these are fixed-reward controls, not sampled
reward-range runs. Frozen HuBERT features are extracted on the fly from raw
audio and passed to `AudioRewardConditionedMaskLM.augment()`.

## Pre-Launch Validation

- Config-only wrapper generation produced exactly eight configs for the
  fixed-reward/dataset/epoch matrix.
- The actual wrapper `EXIT` callback path passed with
  `ROB132_TESTSETS_CALLBACK_ONLY=1 ROB132_TESTSETS_CALLBACK_CHECK_ONLY=1`.
- A bounded GPU check on Earnings22 test index `0` extracted HuBERT features
  from the raw MP3 segment and generated a short reward-`1.0` mask from the
  trained ROB-132 checkpoint.
- Full single-index adaptation was not used as a smoke because both TED-LIUM
  and Earnings22 index-level self-training slices are long enough to behave like
  real eval work rather than a quick preflight.
