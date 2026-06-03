# ROB-132 Audio SSL Test-Set Fixed-Reward Eval

This result root consolidates all ROB-132 fixed-reward test-set self-training
eval artifacts for the trained HuBERT audio+reward-conditioned transformer
mask LM.

Scope:

- Policy checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/audio_ssl_hubert_base_tedlium_per_utterance_transformer384_dropout0p1_500ep_lr1e3.pt`
- Policy shape: `AudioRewardConditionedMaskLM`, HuBERT SSL conditioning,
  transformer decoder, `hidden_dim=384`, `dropout=0.1`.
- Reward controls: fixed `conditioning_reward: 1.0` and fixed
  `conditioning_reward: 0.0` as separate runs.
- Datasets: TED-LIUM, Earnings22, Rev16, TAL, and CHiME-6 test.
- Adaptation cells: `epochs=1` and `epochs=5`, both with `lr=1e-5`.
- Total cells: `5 datasets * 2 epochs * 2 fixed rewards = 20`.
- Completed cells: `16/20`.
- Result CSV: `rob132_audio_ssl_test_fixed_rewards.csv`.
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

## Deferred Cells

The four missing rows are the Rev16/TAL epoch-5 cells at rewards `1.0` and
`0.0`. They were intentionally cancelled after runtime estimates showed they
were likely to exceed the 4-day Stanage limit. Do not rerun them without a new
explicit instruction.

## Pre-Launch Validation

- Config-only wrapper generation produced exactly eight configs for the
  initial TED-LIUM/Earnings22 fixed-reward/dataset/epoch matrix.
- The actual wrapper `EXIT` callback path passed with
  `ROB132_TESTSETS_CALLBACK_ONLY=1 ROB132_TESTSETS_CALLBACK_CHECK_ONLY=1`.
- A bounded GPU check on Earnings22 test index `0` extracted HuBERT features
  from the raw MP3 segment and generated a short reward-`1.0` mask from the
  trained ROB-132 checkpoint.
- Full single-index adaptation was not used as a smoke because both TED-LIUM
  and Earnings22 index-level self-training slices are long enough to behave like
  real eval work rather than a quick preflight.
- The Rev16/TAL/CHiME-6 extension used separate Stanage jobs per cell plus a
  finalizer. Smoke checks covered Rev16, TAL, and CHiME-6, including the
  CHiME-6 multi-channel raw-audio SSL path.
