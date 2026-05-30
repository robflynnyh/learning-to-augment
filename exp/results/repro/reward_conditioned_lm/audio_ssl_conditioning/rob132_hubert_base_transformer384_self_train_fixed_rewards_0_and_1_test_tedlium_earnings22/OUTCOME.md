# ROB-132 Audio SSL Test-Set Fixed-Reward Eval

## Metadata

- Checkpoint: `/mnt/parscratch/users/acp21rjf/l2augment_model/reward_conditioned_mask_lm/audio_ssl_hubert_base_tedlium_per_utterance_transformer384_dropout0p1_500ep_lr1e3.pt`
- Policy: `AudioRewardConditionedMaskLM`, HuBERT SSL conditioning, transformer decoder
- Reward controls: fixed `conditioning_reward: 1.0` and fixed `conditioning_reward: 0.0` as separate runs
- Datasets: `tedlium` and `earnings22`; both `test` split
- Adaptation: `epochs=1` and `epochs=5`, `lr=1e-5`, multistep rollout
- Branch: `symphony/ROB-132-audio-conditioned-vq-mask-lm`
- Commit: `f8ef2f1ba4633c8efcf157fa803b661ee4fa73a8`
- Main log: `/mnt/parscratch/users/acp21rjf/symphony-workspaces-learning-to-augment/ROB-132/exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_tedlium_earnings22/logs/stanage/rob132_stanage_finalizer-10273343.log`
- Screen log: `slurm`
- Queued command: `rob132-stanage-split-manual`

Completed cells: `8/8`.

## Aggregate

| Reward | Dataset | Epochs | N | Mean Original WER | Mean Updated WER | Updated WER Std | Mean Abs Delta | Mean Rel Delta % |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | earnings22 | 1 | 1 | 0.235198 | 0.199007 | 0.000000 | -0.036191 | -15.39 |
| 0.0 | earnings22 | 5 | 1 | 0.235218 | 0.183731 | 0.000000 | -0.051487 | -21.89 |
| 0.0 | tedlium | 1 | 1 | 0.085345 | 0.077477 | 0.000000 | -0.007868 | -9.22 |
| 0.0 | tedlium | 5 | 1 | 0.085345 | 0.074783 | 0.000000 | -0.010562 | -12.38 |
| 1.0 | earnings22 | 1 | 1 | 0.235239 | 0.193432 | 0.000000 | -0.041807 | -17.77 |
| 1.0 | earnings22 | 5 | 1 | 0.235239 | 0.187121 | 0.000000 | -0.048118 | -20.45 |
| 1.0 | tedlium | 1 | 1 | 0.085345 | 0.077370 | 0.000000 | -0.007975 | -9.34 |
| 1.0 | tedlium | 5 | 1 | 0.085345 | 0.075563 | 0.000000 | -0.009782 | -11.46 |

## Per Cell

| Reward | Dataset | Epochs | LR | Original WER | Updated WER | Abs Delta | Rel Delta % | Status | Result |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1.0 | tedlium | 1 | `1e-5` | 0.085345 | 0.077370 | -0.007974 | -9.34 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_tedlium_earnings22/AudioRewardConditionedMaskLMReward1p0/tedlium_test_reward1p0_epoch1_lr1e-5.txt` |
| 1.0 | tedlium | 5 | `1e-5` | 0.085345 | 0.075563 | -0.009782 | -11.46 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_tedlium_earnings22/AudioRewardConditionedMaskLMReward1p0/tedlium_test_reward1p0_epoch5_lr1e-5.txt` |
| 1.0 | earnings22 | 1 | `1e-5` | 0.235239 | 0.193432 | -0.041807 | -17.77 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_tedlium_earnings22/AudioRewardConditionedMaskLMReward1p0/earnings22_test_reward1p0_epoch1_lr1e-5.txt` |
| 1.0 | earnings22 | 5 | `1e-5` | 0.235239 | 0.187121 | -0.048118 | -20.45 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_tedlium_earnings22/AudioRewardConditionedMaskLMReward1p0/earnings22_test_reward1p0_epoch5_lr1e-5.txt` |
| 0.0 | tedlium | 1 | `1e-5` | 0.085345 | 0.077477 | -0.007868 | -9.22 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_tedlium_earnings22/AudioRewardConditionedMaskLMReward0p0/tedlium_test_reward0p0_epoch1_lr1e-5.txt` |
| 0.0 | tedlium | 5 | `1e-5` | 0.085345 | 0.074783 | -0.010562 | -12.38 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_tedlium_earnings22/AudioRewardConditionedMaskLMReward0p0/tedlium_test_reward0p0_epoch5_lr1e-5.txt` |
| 0.0 | earnings22 | 1 | `1e-5` | 0.235198 | 0.199007 | -0.036191 | -15.39 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_tedlium_earnings22/AudioRewardConditionedMaskLMReward0p0/earnings22_test_reward0p0_epoch1_lr1e-5.txt` |
| 0.0 | earnings22 | 5 | `1e-5` | 0.235218 | 0.183731 | -0.051488 | -21.89 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_tedlium_earnings22/AudioRewardConditionedMaskLMReward0p0/earnings22_test_reward0p0_epoch5_lr1e-5.txt` |

CSV artifact:

```text
/mnt/parscratch/users/acp21rjf/symphony-workspaces-learning-to-augment/ROB-132/exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_tedlium_earnings22/rob132_audio_ssl_self_train_test_sets_fixed_rewards.csv
```
