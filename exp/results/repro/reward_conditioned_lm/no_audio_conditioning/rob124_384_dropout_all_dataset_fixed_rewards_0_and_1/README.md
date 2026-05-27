# ROB-124 384-Dropout All-Dataset Fixed Reward 0 And 1 Eval

This result root tracks the corrected 2026-05-25 follow-up requested after the
completed ROB-124 `[0.5, 1.0]` all-dataset sweep and after the cancelled
sampled `[0.0, 1.0]` queue.

Scope:

- Policy checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt`
- Policy shape: `RewardConditionedMaskLM`, `hidden_dim=384`, `dropout=0.1`.
- Reward controls: fixed `conditioning_reward: 1.0` and fixed
  `conditioning_reward: 0.0` as separate runs.
- Datasets: TED-LIUM, Earnings22, CHiME-6, Rev16, and This American Life; all
  use the `test` split.
- Adaptation cells: `epochs=1` and `epochs=5`, both with `lr=1e-5`.
- Total cells: `5 datasets * 2 epochs * 2 fixed rewards = 20`.
- Result CSV: `rob124_384_dropout_all_dataset_fixed_rewards_0_and_1.csv`.
- Outcome: `OUTCOME.md`.

The wrapper is:

```bash
scripts/launch_rob124_384_dropout_all_dataset_fixed_rewards.sh
```

The detached Mimas command is:

```bash
screen -L -Logfile /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1/logs/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1.screen.log -dmS rob124-384-dropout-fixed-rewards-0-and-1 bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob124_384_dropout_all_dataset_fixed_rewards.sh'
```

Implementation note: fixed-reward configs set
`evaluation.augmentation_config.conditioning_reward` and
`policy.config.default_conditioning_reward` to the same scalar. They do not set
`conditioning_reward_range`, so they are not sampled reward-range runs.
