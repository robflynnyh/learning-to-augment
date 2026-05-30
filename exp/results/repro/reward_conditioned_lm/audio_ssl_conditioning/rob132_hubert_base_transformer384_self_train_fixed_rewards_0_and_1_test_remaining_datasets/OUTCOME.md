# ROB-132 Audio SSL Test-Set Fixed-Reward Eval

## Metadata

- Checkpoint: `/mnt/parscratch/users/acp21rjf/l2augment_model/reward_conditioned_mask_lm/audio_ssl_hubert_base_tedlium_per_utterance_transformer384_dropout0p1_500ep_lr1e3.pt`
- Policy: `AudioRewardConditionedMaskLM`, HuBERT SSL conditioning, transformer decoder
- Reward controls: fixed `conditioning_reward: 1.0` and fixed `conditioning_reward: 0.0` as separate runs
- Datasets: `rev16`, `TAL`, `chime6`; all `test` split
- Adaptation: `epochs=1` and `epochs=5`, `lr=1e-5`, multistep rollout
- Branch: `symphony/ROB-132-audio-conditioned-vq-mask-lm`
- Commit: `69ed356f7f7e097016855b98570ff54070bdd0ae`
- Main log: `slurm-finalizer-pending`
- Screen log: `slurm`
- Queued command: `scripts/submit_rob132_audio_ssl_testsets_stanage.sh`

Completed cells: `0/12`.

## Aggregate

| Reward | Dataset | Epochs | N | Mean Original WER | Mean Updated WER | Updated WER Std | Mean Abs Delta | Mean Rel Delta % |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## Missing Cells

- reward 1.0 / rev16 / epoch 1 / lr `1e-5`
- reward 1.0 / rev16 / epoch 5 / lr `1e-5`
- reward 1.0 / TAL / epoch 1 / lr `1e-5`
- reward 1.0 / TAL / epoch 5 / lr `1e-5`
- reward 1.0 / chime6 / epoch 1 / lr `1e-5`
- reward 1.0 / chime6 / epoch 5 / lr `1e-5`
- reward 0.0 / rev16 / epoch 1 / lr `1e-5`
- reward 0.0 / rev16 / epoch 5 / lr `1e-5`
- reward 0.0 / TAL / epoch 1 / lr `1e-5`
- reward 0.0 / TAL / epoch 5 / lr `1e-5`
- reward 0.0 / chime6 / epoch 1 / lr `1e-5`
- reward 0.0 / chime6 / epoch 5 / lr `1e-5`

## Per Cell

| Reward | Dataset | Epochs | LR | Original WER | Updated WER | Abs Delta | Rel Delta % | Status | Result |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1.0 | rev16 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_remaining_datasets/AudioRewardConditionedMaskLMReward1p0/rev16_test_reward1p0_epoch1_lr1e-5.txt` |
| 1.0 | rev16 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_remaining_datasets/AudioRewardConditionedMaskLMReward1p0/rev16_test_reward1p0_epoch5_lr1e-5.txt` |
| 1.0 | TAL | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_remaining_datasets/AudioRewardConditionedMaskLMReward1p0/TAL_test_reward1p0_epoch1_lr1e-5.txt` |
| 1.0 | TAL | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_remaining_datasets/AudioRewardConditionedMaskLMReward1p0/TAL_test_reward1p0_epoch5_lr1e-5.txt` |
| 1.0 | chime6 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_remaining_datasets/AudioRewardConditionedMaskLMReward1p0/chime6_test_reward1p0_epoch1_lr1e-5.txt` |
| 1.0 | chime6 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_remaining_datasets/AudioRewardConditionedMaskLMReward1p0/chime6_test_reward1p0_epoch5_lr1e-5.txt` |
| 0.0 | rev16 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_remaining_datasets/AudioRewardConditionedMaskLMReward0p0/rev16_test_reward0p0_epoch1_lr1e-5.txt` |
| 0.0 | rev16 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_remaining_datasets/AudioRewardConditionedMaskLMReward0p0/rev16_test_reward0p0_epoch5_lr1e-5.txt` |
| 0.0 | TAL | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_remaining_datasets/AudioRewardConditionedMaskLMReward0p0/TAL_test_reward0p0_epoch1_lr1e-5.txt` |
| 0.0 | TAL | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_remaining_datasets/AudioRewardConditionedMaskLMReward0p0/TAL_test_reward0p0_epoch5_lr1e-5.txt` |
| 0.0 | chime6 | 1 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_remaining_datasets/AudioRewardConditionedMaskLMReward0p0/chime6_test_reward0p0_epoch1_lr1e-5.txt` |
| 0.0 | chime6 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_remaining_datasets/AudioRewardConditionedMaskLMReward0p0/chime6_test_reward0p0_epoch5_lr1e-5.txt` |

CSV artifact:

```text
exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_remaining_datasets/rob132_audio_ssl_self_train_remaining_datasets_fixed_rewards.csv
```
