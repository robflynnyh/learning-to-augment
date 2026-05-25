# ROB-124 384-Dropout All-Dataset Reward Sampling [0.0, 1.0] Eval

## Metadata

- Checkpoint: `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt`
- Policy: `RewardConditionedMaskLM`, `hidden_dim=384`, `dropout=0.1`
- Reward control: sampled uniformly from `[0.0, 1.0]` during each adaptation mask generation step
- Datasets: `tedlium`, `earnings22`, `chime6`, `rev16`, `TAL`; all `test` split
- Adaptation: `epochs=1` and `epochs=5`, `lr=1e-5`, multistep rollout
- Branch: `symphony/ROB-124-384-dropout-mask-lm`
- Commit: `9ed50f5ab0b3681e53c90f91dc036ed3b25a3d42`
- Main log: `/exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_reward_sampling_0to1/logs/rob124_384_dropout_all_dataset_reward_sampling_0to1.log`
- Screen log: `/exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_reward_sampling_0to1/logs/rob124_384_dropout_all_dataset_reward_sampling_0to1.screen.log`
- Queued command: `screen -L -Logfile /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_reward_sampling_0to1/logs/rob124_384_dropout_all_dataset_reward_sampling_0to1.screen.log -dmS rob124-384-dropout-all-dataset-sampling-0to1 bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob124_384_dropout_all_dataset_reward_sampling_0to1.sh'`

Completed cells: `0/10`.

## Aggregate

| Dataset | Epochs | N | Mean Original WER | Mean Updated WER | Updated WER Std | Mean Abs Delta | Mean Rel Delta % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## Missing Cells

- tedlium / epoch 1 / repeat 1 / lr `1e-5`
- tedlium / epoch 5 / repeat 1 / lr `1e-5`
- earnings22 / epoch 1 / repeat 1 / lr `1e-5`
- earnings22 / epoch 5 / repeat 1 / lr `1e-5`
- chime6 / epoch 1 / repeat 1 / lr `1e-5`
- chime6 / epoch 5 / repeat 1 / lr `1e-5`
- rev16 / epoch 1 / repeat 1 / lr `1e-5`
- rev16 / epoch 5 / repeat 1 / lr `1e-5`
- TAL / epoch 1 / repeat 1 / lr `1e-5`
- TAL / epoch 5 / repeat 1 / lr `1e-5`

## Per Cell

| Dataset | Repeat | Seed | Epochs | LR | Original WER | Updated WER | Abs Delta | Rel Delta % | Status | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| tedlium | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_reward_sampling_0to1/RewardConditionedMaskLMUniform0to1/tedlium_test_epoch1_lr1e-5.txt` |
| tedlium | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_reward_sampling_0to1/RewardConditionedMaskLMUniform0to1/tedlium_test_epoch5_lr1e-5.txt` |
| earnings22 | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_reward_sampling_0to1/RewardConditionedMaskLMUniform0to1/earnings22_test_epoch1_lr1e-5.txt` |
| earnings22 | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_reward_sampling_0to1/RewardConditionedMaskLMUniform0to1/earnings22_test_epoch5_lr1e-5.txt` |
| chime6 | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_reward_sampling_0to1/RewardConditionedMaskLMUniform0to1/chime6_test_epoch1_lr1e-5.txt` |
| chime6 | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_reward_sampling_0to1/RewardConditionedMaskLMUniform0to1/chime6_test_epoch5_lr1e-5.txt` |
| rev16 | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_reward_sampling_0to1/RewardConditionedMaskLMUniform0to1/rev16_test_epoch1_lr1e-5.txt` |
| rev16 | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_reward_sampling_0to1/RewardConditionedMaskLMUniform0to1/rev16_test_epoch5_lr1e-5.txt` |
| TAL | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_reward_sampling_0to1/RewardConditionedMaskLMUniform0to1/TAL_test_epoch1_lr1e-5.txt` |
| TAL | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_reward_sampling_0to1/RewardConditionedMaskLMUniform0to1/TAL_test_epoch5_lr1e-5.txt` |

CSV artifact:

```text
/exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_reward_sampling_0to1/rob124_384_dropout_all_dataset_reward_sampling_0to1.csv
```
