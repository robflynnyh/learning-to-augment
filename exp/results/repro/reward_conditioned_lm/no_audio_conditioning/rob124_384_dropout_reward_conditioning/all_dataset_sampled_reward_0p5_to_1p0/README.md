# ROB-124 384-Dropout All-Dataset Reward Sampling Eval

This result root tracks the ROB-108-style test-set eval requested on
2026-05-24 for the trained ROB-124 384/dropout reward-conditioned mask LM.

Scope:

- Policy checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt`
- Policy shape: `RewardConditionedMaskLM`, `hidden_dim=384`, `dropout=0.1`.
- Reward control: sample `conditioning_reward` uniformly from `[0.5, 1.0]`
  during each adaptation mask generation step.
- Datasets: TED-LIUM, Earnings22, CHiME-6, Rev16, and This American Life; all
  use the `test` split.
- Adaptation cells: `epochs=1` and `epochs=5`, both with `lr=1e-5`, matching
  the safer ROB-108 learning-rate setting.
- Result CSV: `rob124_384_dropout_all_dataset_reward_sampling.csv`.
- Outcome: `OUTCOME.md`.

Final status: the callback-backed Mimas run exited with status `0` and
completed all `10/10` cells. The result is broadly positive at 1 adaptation
epoch and mixed at 5 epochs because the CHiME-6 5-epoch cell regressed to WER
`1.0`. Treat the 5-epoch setting as dataset-sensitive rather than uniformly
better.

The wrapper is:

```bash
scripts/launch_rob124_384_dropout_all_dataset_reward_sampling.sh
```

The detached Mimas command should be:

```bash
screen -L -Logfile /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_sampled_reward_0p5_to_1p0/logs/rob124_384_dropout_all_dataset_reward_sampling.screen.log -dmS rob124-384-dropout-all-dataset-sampling bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob124_384_dropout_all_dataset_reward_sampling.sh'
```

Implementation note: this eval depends on the ROB-124 fix that lets
`RewardConditionedMaskLM.augment` sample from `conditioning_reward_range`.
Without that fix, adaptation-time calls would silently use
`default_conditioning_reward` and the `[0.5, 1.0]` condition would be mislabeled.
