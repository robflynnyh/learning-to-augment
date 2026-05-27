# ROB-132 Audio SSL Test-Set Fixed-Reward Eval

## Metadata

- Checkpoint: `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/audio_ssl_hubert_base_tedlium_per_utterance_transformer384_dropout0p1_500ep_lr1e3.pt`
- Policy: `AudioRewardConditionedMaskLM`, HuBERT SSL conditioning, transformer decoder
- Reward controls: fixed `conditioning_reward: 1.0` and fixed `conditioning_reward: 0.0` as separate runs
- Datasets: `tedlium` and `earnings22`; both `test` split
- Adaptation: `epochs=1` and `epochs=5`, `lr=1e-5`, multistep rollout
- Branch: `symphony/ROB-132-audio-conditioned-vq-mask-lm`
- Commit: `2a7477353b07696c2a023645c274374da71528c4`
- Main log: `/exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-132/exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_tedlium_earnings22/logs/rob132_audio_ssl_self_train_test_sets_tedlium_smoke.log`
- Screen log: `/exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-132/exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_tedlium_earnings22/logs/rob132_audio_ssl_self_train_test_sets_tedlium_smoke.screen.log`
- Queued command: `screen -L -Logfile /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-132/exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_tedlium_earnings22/logs/rob132_audio_ssl_self_train_test_sets_tedlium_smoke.screen.log -dmS rob132-audio-ssl-testsets-tedlium-smoke bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-132 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob132_audio_ssl_self_train_test_sets.sh'`

Completed cells: `0/8`.

## Aggregate

| Reward | Dataset | Epochs | N | Mean Original WER | Mean Updated WER | Updated WER Std | Mean Abs Delta | Mean Rel Delta % |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## Missing Cells

- reward 1.0 / tedlium / epoch 1 / lr `1e-5`
- reward 1.0 / tedlium / epoch 5 / lr `1e-5`
- reward 1.0 / earnings22 / epoch 1 / lr `1e-5`
- reward 1.0 / earnings22 / epoch 5 / lr `1e-5`
- reward 0.0 / tedlium / epoch 1 / lr `1e-5`
- reward 0.0 / tedlium / epoch 5 / lr `1e-5`
- reward 0.0 / earnings22 / epoch 1 / lr `1e-5`
- reward 0.0 / earnings22 / epoch 5 / lr `1e-5`

## Per Cell

| Reward | Dataset | Epochs | LR | Original WER | Updated WER | Abs Delta | Rel Delta % | Status | Result |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1.0 | tedlium | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_tedlium_earnings22/AudioRewardConditionedMaskLMReward1p0/tedlium_test_reward1p0_epoch1_lr1e-5.txt` |
| 1.0 | tedlium | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_tedlium_earnings22/AudioRewardConditionedMaskLMReward1p0/tedlium_test_reward1p0_epoch5_lr1e-5.txt` |
| 1.0 | earnings22 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_tedlium_earnings22/AudioRewardConditionedMaskLMReward1p0/earnings22_test_reward1p0_epoch1_lr1e-5.txt` |
| 1.0 | earnings22 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_tedlium_earnings22/AudioRewardConditionedMaskLMReward1p0/earnings22_test_reward1p0_epoch5_lr1e-5.txt` |
| 0.0 | tedlium | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_tedlium_earnings22/AudioRewardConditionedMaskLMReward0p0/tedlium_test_reward0p0_epoch1_lr1e-5.txt` |
| 0.0 | tedlium | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_tedlium_earnings22/AudioRewardConditionedMaskLMReward0p0/tedlium_test_reward0p0_epoch5_lr1e-5.txt` |
| 0.0 | earnings22 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_tedlium_earnings22/AudioRewardConditionedMaskLMReward0p0/earnings22_test_reward0p0_epoch1_lr1e-5.txt` |
| 0.0 | earnings22 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_tedlium_earnings22/AudioRewardConditionedMaskLMReward0p0/earnings22_test_reward0p0_epoch5_lr1e-5.txt` |

CSV artifact:

```text
/exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-132/exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_tedlium_earnings22/rob132_audio_ssl_self_train_test_sets_fixed_rewards.csv
```
