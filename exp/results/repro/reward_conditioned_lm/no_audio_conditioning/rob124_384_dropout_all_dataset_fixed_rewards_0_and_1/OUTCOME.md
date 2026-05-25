# ROB-124 384-Dropout All-Dataset Fixed Reward 0 And 1 Eval

## Metadata

- Checkpoint: `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt`
- Policy: `RewardConditionedMaskLM`, `hidden_dim=384`, `dropout=0.1`
- Reward controls: fixed `conditioning_reward: 1.0` and fixed `conditioning_reward: 0.0` as separate runs
- Datasets: `tedlium`, `earnings22`, `chime6`, `rev16`, `TAL`; all `test` split
- Adaptation: `epochs=1` and `epochs=5`, `lr=1e-5`, multistep rollout
- Branch: `symphony/ROB-124-384-dropout-mask-lm`
- Commit: `0929f85c7bdada0c58833c27c03617a175ae2c79`
- Main log: `/exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1/logs/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1.log`
- Screen log: `/exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1/logs/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1.screen.log`
- Queued command: `screen -L -Logfile /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1/logs/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1.screen.log -dmS rob124-384-dropout-fixed-rewards-0-and-1 bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob124_384_dropout_all_dataset_fixed_rewards.sh'`

Completed cells: `0/20`.

## Aggregate

| Reward | Dataset | Epochs | N | Mean Original WER | Mean Updated WER | Updated WER Std | Mean Abs Delta | Mean Rel Delta % |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## Missing Cells

- reward 1.0 / tedlium / epoch 1 / repeat 1 / lr `1e-5`
- reward 1.0 / tedlium / epoch 5 / repeat 1 / lr `1e-5`
- reward 1.0 / earnings22 / epoch 1 / repeat 1 / lr `1e-5`
- reward 1.0 / earnings22 / epoch 5 / repeat 1 / lr `1e-5`
- reward 1.0 / chime6 / epoch 1 / repeat 1 / lr `1e-5`
- reward 1.0 / chime6 / epoch 5 / repeat 1 / lr `1e-5`
- reward 1.0 / rev16 / epoch 1 / repeat 1 / lr `1e-5`
- reward 1.0 / rev16 / epoch 5 / repeat 1 / lr `1e-5`
- reward 1.0 / TAL / epoch 1 / repeat 1 / lr `1e-5`
- reward 1.0 / TAL / epoch 5 / repeat 1 / lr `1e-5`
- reward 0.0 / tedlium / epoch 1 / repeat 1 / lr `1e-5`
- reward 0.0 / tedlium / epoch 5 / repeat 1 / lr `1e-5`
- reward 0.0 / earnings22 / epoch 1 / repeat 1 / lr `1e-5`
- reward 0.0 / earnings22 / epoch 5 / repeat 1 / lr `1e-5`
- reward 0.0 / chime6 / epoch 1 / repeat 1 / lr `1e-5`
- reward 0.0 / chime6 / epoch 5 / repeat 1 / lr `1e-5`
- reward 0.0 / rev16 / epoch 1 / repeat 1 / lr `1e-5`
- reward 0.0 / rev16 / epoch 5 / repeat 1 / lr `1e-5`
- reward 0.0 / TAL / epoch 1 / repeat 1 / lr `1e-5`
- reward 0.0 / TAL / epoch 5 / repeat 1 / lr `1e-5`

## Per Cell

| Reward | Dataset | Repeat | Seed | Epochs | LR | Original WER | Updated WER | Abs Delta | Rel Delta % | Status | Result |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1.0 | tedlium | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/tedlium_test_epoch1_lr1e-5.txt` |
| 1.0 | tedlium | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/tedlium_test_epoch5_lr1e-5.txt` |
| 1.0 | earnings22 | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/earnings22_test_epoch1_lr1e-5.txt` |
| 1.0 | earnings22 | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/earnings22_test_epoch5_lr1e-5.txt` |
| 1.0 | chime6 | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/chime6_test_epoch1_lr1e-5.txt` |
| 1.0 | chime6 | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/chime6_test_epoch5_lr1e-5.txt` |
| 1.0 | rev16 | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/rev16_test_epoch1_lr1e-5.txt` |
| 1.0 | rev16 | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/rev16_test_epoch5_lr1e-5.txt` |
| 1.0 | TAL | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/TAL_test_epoch1_lr1e-5.txt` |
| 1.0 | TAL | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/TAL_test_epoch5_lr1e-5.txt` |
| 0.0 | tedlium | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/tedlium_test_epoch1_lr1e-5.txt` |
| 0.0 | tedlium | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/tedlium_test_epoch5_lr1e-5.txt` |
| 0.0 | earnings22 | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/earnings22_test_epoch1_lr1e-5.txt` |
| 0.0 | earnings22 | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/earnings22_test_epoch5_lr1e-5.txt` |
| 0.0 | chime6 | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/chime6_test_epoch1_lr1e-5.txt` |
| 0.0 | chime6 | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/chime6_test_epoch5_lr1e-5.txt` |
| 0.0 | rev16 | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/rev16_test_epoch1_lr1e-5.txt` |
| 0.0 | rev16 | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/rev16_test_epoch5_lr1e-5.txt` |
| 0.0 | TAL | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/TAL_test_epoch1_lr1e-5.txt` |
| 0.0 | TAL | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/TAL_test_epoch5_lr1e-5.txt` |

CSV artifact:

```text
/exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1.csv
```
