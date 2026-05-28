# ROB-124 384-Dropout All-Dataset Fixed Reward 0 And 1 Eval

## Metadata

- Checkpoint: `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt`
- Policy: `RewardConditionedMaskLM`, `hidden_dim=384`, `dropout=0.1`
- Reward controls: fixed `conditioning_reward: 1.0` and fixed `conditioning_reward: 0.0` as separate runs
- Datasets: `tedlium`, `earnings22`, `chime6`, `rev16`, `TAL`; all `test` split
- Adaptation: `epochs=1` and `epochs=5`, `lr=1e-5`, multistep rollout
- Branch: `symphony/ROB-124-384-dropout-mask-lm`
- Commit: `96a8a3183ad0649a0835799e212fcee1af8490d6`
- Main log: `/exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/logs/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1.log`
- Screen log: `/exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/logs/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1.screen.log`
- Queued command: `screen -L -Logfile /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/logs/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1.screen.log -dmS rob124-384-dropout-fixed-rewards-0-and-1 bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob124_384_dropout_all_dataset_fixed_rewards.sh'`

Completed cells: `20/20`.

## Interpretation

The fixed-reward sweep completed cleanly and produced usable downstream eval
artifacts for both controls. Reward `0.0` improved all `10/10` cells. Reward
`1.0` improved `9/10` cells, with the only regression coming from CHiME-6 at
5 adaptation epochs (`0.843620 -> 1.000000` WER).

The result is consistent with the earlier ROB-124 `[0.5, 1.0]` all-dataset
follow-up: the 384/dropout checkpoint is useful, but 5-epoch adaptation is not
a safe blanket setting because CHiME-6 can collapse under high-reward
conditioning. For downstream eval/oracle comparisons, prefer the 384/dropout
checkpoint with 1-epoch adaptation or dataset-specific reward/epoch selection
rather than assuming larger or longer adaptation is uniformly better.

## Aggregate

| Reward | Dataset | Epochs | N | Mean Original WER | Mean Updated WER | Updated WER Std | Mean Abs Delta | Mean Rel Delta % |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | TAL | 1 | 1 | 0.165694 | 0.162852 | 0.000000 | -0.002842 | -1.72 |
| 0.0 | TAL | 5 | 1 | 0.165700 | 0.159556 | 0.000000 | -0.006144 | -3.71 |
| 0.0 | chime6 | 1 | 1 | 0.843585 | 0.813359 | 0.000000 | -0.030226 | -3.58 |
| 0.0 | chime6 | 5 | 1 | 0.843620 | 0.808213 | 0.000000 | -0.035407 | -4.20 |
| 0.0 | earnings22 | 1 | 1 | 0.235218 | 0.202091 | 0.000000 | -0.033127 | -14.08 |
| 0.0 | earnings22 | 5 | 1 | 0.235239 | 0.185201 | 0.000000 | -0.050038 | -21.27 |
| 0.0 | rev16 | 1 | 1 | 0.172504 | 0.166402 | 0.000000 | -0.006102 | -3.54 |
| 0.0 | rev16 | 5 | 1 | 0.172504 | 0.163593 | 0.000000 | -0.008911 | -5.17 |
| 0.0 | tedlium | 1 | 1 | 0.085345 | 0.079320 | 0.000000 | -0.006025 | -7.06 |
| 0.0 | tedlium | 5 | 1 | 0.085345 | 0.074712 | 0.000000 | -0.010633 | -12.46 |
| 1.0 | TAL | 1 | 1 | 0.165702 | 0.159077 | 0.000000 | -0.006625 | -4.00 |
| 1.0 | TAL | 5 | 1 | 0.165702 | 0.155694 | 0.000000 | -0.010008 | -6.04 |
| 1.0 | chime6 | 1 | 1 | 0.843620 | 0.830649 | 0.000000 | -0.012971 | -1.54 |
| 1.0 | chime6 | 5 | 1 | 0.843620 | 1.000000 | 0.000000 | 0.156380 | 18.54 |
| 1.0 | earnings22 | 1 | 1 | 0.235218 | 0.194004 | 0.000000 | -0.041214 | -17.52 |
| 1.0 | earnings22 | 5 | 1 | 0.235198 | 0.186386 | 0.000000 | -0.048812 | -20.75 |
| 1.0 | rev16 | 1 | 1 | 0.172509 | 0.163599 | 0.000000 | -0.008910 | -5.16 |
| 1.0 | rev16 | 5 | 1 | 0.172504 | 0.159958 | 0.000000 | -0.012546 | -7.27 |
| 1.0 | tedlium | 1 | 1 | 0.085345 | 0.078292 | 0.000000 | -0.007053 | -8.26 |
| 1.0 | tedlium | 5 | 1 | 0.085345 | 0.076236 | 0.000000 | -0.009109 | -10.67 |

## Per Cell

| Reward | Dataset | Repeat | Seed | Epochs | LR | Original WER | Updated WER | Abs Delta | Rel Delta % | Status | Result |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1.0 | tedlium | 1 | 123456 | 1 | `1e-5` | 0.085345 | 0.078292 | -0.007053 | -8.26 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/tedlium_test_epoch1_lr1e-5.txt` |
| 1.0 | tedlium | 1 | 123456 | 5 | `1e-5` | 0.085345 | 0.076236 | -0.009109 | -10.67 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/tedlium_test_epoch5_lr1e-5.txt` |
| 1.0 | earnings22 | 1 | 123456 | 1 | `1e-5` | 0.235218 | 0.194004 | -0.041215 | -17.52 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/earnings22_test_epoch1_lr1e-5.txt` |
| 1.0 | earnings22 | 1 | 123456 | 5 | `1e-5` | 0.235198 | 0.186386 | -0.048812 | -20.75 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/earnings22_test_epoch5_lr1e-5.txt` |
| 1.0 | chime6 | 1 | 123456 | 1 | `1e-5` | 0.843620 | 0.830649 | -0.012971 | -1.54 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/chime6_test_epoch1_lr1e-5.txt` |
| 1.0 | chime6 | 1 | 123456 | 5 | `1e-5` | 0.843620 | 1.000000 | 0.156380 | 18.54 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/chime6_test_epoch5_lr1e-5.txt` |
| 1.0 | rev16 | 1 | 123456 | 1 | `1e-5` | 0.172509 | 0.163599 | -0.008911 | -5.17 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/rev16_test_epoch1_lr1e-5.txt` |
| 1.0 | rev16 | 1 | 123456 | 5 | `1e-5` | 0.172504 | 0.159958 | -0.012546 | -7.27 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/rev16_test_epoch5_lr1e-5.txt` |
| 1.0 | TAL | 1 | 123456 | 1 | `1e-5` | 0.165702 | 0.159077 | -0.006626 | -4.00 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/TAL_test_epoch1_lr1e-5.txt` |
| 1.0 | TAL | 1 | 123456 | 5 | `1e-5` | 0.165702 | 0.155694 | -0.010009 | -6.04 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/TAL_test_epoch5_lr1e-5.txt` |
| 0.0 | tedlium | 1 | 123456 | 1 | `1e-5` | 0.085345 | 0.079320 | -0.006025 | -7.06 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/tedlium_test_epoch1_lr1e-5.txt` |
| 0.0 | tedlium | 1 | 123456 | 5 | `1e-5` | 0.085345 | 0.074712 | -0.010633 | -12.46 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/tedlium_test_epoch5_lr1e-5.txt` |
| 0.0 | earnings22 | 1 | 123456 | 1 | `1e-5` | 0.235218 | 0.202091 | -0.033127 | -14.08 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/earnings22_test_epoch1_lr1e-5.txt` |
| 0.0 | earnings22 | 1 | 123456 | 5 | `1e-5` | 0.235239 | 0.185201 | -0.050038 | -21.27 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/earnings22_test_epoch5_lr1e-5.txt` |
| 0.0 | chime6 | 1 | 123456 | 1 | `1e-5` | 0.843585 | 0.813359 | -0.030226 | -3.58 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/chime6_test_epoch1_lr1e-5.txt` |
| 0.0 | chime6 | 1 | 123456 | 5 | `1e-5` | 0.843620 | 0.808213 | -0.035407 | -4.20 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/chime6_test_epoch5_lr1e-5.txt` |
| 0.0 | rev16 | 1 | 123456 | 1 | `1e-5` | 0.172504 | 0.166402 | -0.006102 | -3.54 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/rev16_test_epoch1_lr1e-5.txt` |
| 0.0 | rev16 | 1 | 123456 | 5 | `1e-5` | 0.172504 | 0.163593 | -0.008911 | -5.17 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/rev16_test_epoch5_lr1e-5.txt` |
| 0.0 | TAL | 1 | 123456 | 1 | `1e-5` | 0.165694 | 0.162852 | -0.002842 | -1.72 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/TAL_test_epoch1_lr1e-5.txt` |
| 0.0 | TAL | 1 | 123456 | 5 | `1e-5` | 0.165700 | 0.159556 | -0.006143 | -3.71 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/TAL_test_epoch5_lr1e-5.txt` |

CSV artifact:

```text
/exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1.csv
```
