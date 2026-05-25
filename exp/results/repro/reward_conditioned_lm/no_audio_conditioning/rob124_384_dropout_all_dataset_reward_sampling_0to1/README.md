# ROB-124 384-Dropout All-Dataset Reward Sampling [0.0, 1.0] Eval

This result root tracks a 2026-05-25 sampled-reward interpretation that was
cancelled before GPU eval after the latest Linear clarification. The active
follow-up is now the separate fixed-reward result root:
`rob124_384_dropout_all_dataset_fixed_rewards_0_and_1`.

Scope:

- Policy checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt`
- Policy shape: `RewardConditionedMaskLM`, `hidden_dim=384`, `dropout=0.1`.
- Reward control in this cancelled scaffold: sample `conditioning_reward`
  uniformly from `[0.0, 1.0]` during each adaptation mask generation step.
- Datasets: TED-LIUM, Earnings22, CHiME-6, Rev16, and This American Life; all
  use the `test` split.
- Adaptation cells in this cancelled scaffold: `epochs=1` and `epochs=5`,
  both with `lr=1e-5`.
- Result CSV: `rob124_384_dropout_all_dataset_reward_sampling_0to1.csv`.
- Outcome: `OUTCOME.md`.

The wrapper is:

```bash
scripts/launch_rob124_384_dropout_all_dataset_reward_sampling_0to1.sh
```

The detached Mimas command was queued and then stopped while still waiting in
`with-gpu`; no sampled `[0.0, 1.0]` eval cells were run from this scaffold.

```bash
screen -L -Logfile /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_reward_sampling_0to1/logs/rob124_384_dropout_all_dataset_reward_sampling_0to1.screen.log -dmS rob124-384-dropout-all-dataset-sampling-0to1 bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob124_384_dropout_all_dataset_reward_sampling_0to1.sh'
```

Implementation note: this eval depends on the ROB-124 fix that lets
`RewardConditionedMaskLM.augment` sample from `conditioning_reward_range`.
Without that fix, adaptation-time calls would silently use
`default_conditioning_reward` and the sampled-range condition would be
mislabeled.
