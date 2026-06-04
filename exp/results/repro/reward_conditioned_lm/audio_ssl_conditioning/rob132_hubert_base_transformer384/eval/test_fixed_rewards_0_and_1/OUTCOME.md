# ROB-132 Audio SSL Test-Set Fixed-Reward Eval

## Metadata

- Checkpoint: `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/audio_ssl_hubert_base_tedlium_per_utterance_transformer384_dropout0p1_500ep_lr1e3.pt`
- Policy: `AudioRewardConditionedMaskLM`, HuBERT SSL conditioning, transformer decoder
- Reward controls: fixed `conditioning_reward: 1.0` and fixed `conditioning_reward: 0.0` as separate runs
- Datasets: `tedlium`, `earnings22`, `rev16`, `TAL`, `chime6`; all `test` split
- Adaptation: `epochs=1` and `epochs=5`, `lr=1e-5`, multistep rollout
- Branch: `symphony/ROB-201-complete-rev16-tal-epoch5-audio-ssl`
- Commit: `fc922ba`
- Main log: `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384/eval/test_fixed_rewards_0_and_1/logs/rob201_rev16_tal_epoch5_mimas_corrected.log`
- Screen log: `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384/eval/test_fixed_rewards_0_and_1/logs/rob201_rev16_tal_epoch5_mimas_corrected.screen.log`
- Queued command: `ROB-201 corrected Mimas Rev16/TAL epoch-5 preparation; previous batch-size-4 run cancelled after equivalence validation failed`

Completed cells: `16/20`.

## Aggregate

| Reward | Dataset | Epochs | N | Mean Original WER | Mean Updated WER | Updated WER Std | Mean Abs Delta | Mean Rel Delta % |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | TAL | 1 | 1 | 0.165691 | 0.160867 | 0.000000 | -0.004824 | -2.91 |
| 0.0 | chime6 | 1 | 1 | 0.843638 | 0.827194 | 0.000000 | -0.016444 | -1.95 |
| 0.0 | chime6 | 5 | 1 | 0.843620 | 1.000000 | 0.000000 | 0.156380 | 18.54 |
| 0.0 | earnings22 | 1 | 1 | 0.235198 | 0.199007 | 0.000000 | -0.036191 | -15.39 |
| 0.0 | earnings22 | 5 | 1 | 0.235218 | 0.183731 | 0.000000 | -0.051487 | -21.89 |
| 0.0 | rev16 | 1 | 1 | 0.172576 | 0.165023 | 0.000000 | -0.007553 | -4.38 |
| 0.0 | tedlium | 1 | 1 | 0.085345 | 0.077477 | 0.000000 | -0.007868 | -9.22 |
| 0.0 | tedlium | 5 | 1 | 0.085345 | 0.074783 | 0.000000 | -0.010562 | -12.38 |
| 1.0 | TAL | 1 | 1 | 0.165691 | 0.158428 | 0.000000 | -0.007263 | -4.38 |
| 1.0 | chime6 | 1 | 1 | 0.843638 | 0.822259 | 0.000000 | -0.021379 | -2.53 |
| 1.0 | chime6 | 5 | 1 | 0.843620 | 1.000000 | 0.000000 | 0.156380 | 18.54 |
| 1.0 | earnings22 | 1 | 1 | 0.235239 | 0.193432 | 0.000000 | -0.041807 | -17.77 |
| 1.0 | earnings22 | 5 | 1 | 0.235239 | 0.187121 | 0.000000 | -0.048118 | -20.45 |
| 1.0 | rev16 | 1 | 1 | 0.172576 | 0.162358 | 0.000000 | -0.010218 | -5.92 |
| 1.0 | tedlium | 1 | 1 | 0.085345 | 0.077370 | 0.000000 | -0.007975 | -9.34 |
| 1.0 | tedlium | 5 | 1 | 0.085345 | 0.075563 | 0.000000 | -0.009782 | -11.46 |

## Missing Cells

- reward 1.0 / rev16 / epoch 5 / lr `1e-5`
- reward 1.0 / TAL / epoch 5 / lr `1e-5`
- reward 0.0 / rev16 / epoch 5 / lr `1e-5`
- reward 0.0 / TAL / epoch 5 / lr `1e-5`

## Per Cell

| Reward | Dataset | Epochs | LR | Original WER | Updated WER | Abs Delta | Rel Delta % | Status | Result |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1.0 | tedlium | 1 | `1e-5` | 0.085345 | 0.077370 | -0.007974 | -9.34 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384/eval/test_fixed_rewards_0_and_1/AudioRewardConditionedMaskLMReward1p0/tedlium_test_reward1p0_epoch1_lr1e-5.txt` |
| 1.0 | tedlium | 5 | `1e-5` | 0.085345 | 0.075563 | -0.009782 | -11.46 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384/eval/test_fixed_rewards_0_and_1/AudioRewardConditionedMaskLMReward1p0/tedlium_test_reward1p0_epoch5_lr1e-5.txt` |
| 1.0 | earnings22 | 1 | `1e-5` | 0.235239 | 0.193432 | -0.041807 | -17.77 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384/eval/test_fixed_rewards_0_and_1/AudioRewardConditionedMaskLMReward1p0/earnings22_test_reward1p0_epoch1_lr1e-5.txt` |
| 1.0 | earnings22 | 5 | `1e-5` | 0.235239 | 0.187121 | -0.048118 | -20.45 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384/eval/test_fixed_rewards_0_and_1/AudioRewardConditionedMaskLMReward1p0/earnings22_test_reward1p0_epoch5_lr1e-5.txt` |
| 1.0 | rev16 | 1 | `1e-5` | 0.172576 | 0.162358 | -0.010218 | -5.92 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384/eval/test_fixed_rewards_0_and_1/AudioRewardConditionedMaskLMReward1p0/rev16_test_reward1p0_epoch1_lr1e-5.txt` |
| 1.0 | rev16 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384/eval/test_fixed_rewards_0_and_1/AudioRewardConditionedMaskLMReward1p0/rev16_test_reward1p0_epoch5_lr1e-5.txt` |
| 1.0 | TAL | 1 | `1e-5` | 0.165691 | 0.158428 | -0.007263 | -4.38 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384/eval/test_fixed_rewards_0_and_1/AudioRewardConditionedMaskLMReward1p0/TAL_test_reward1p0_epoch1_lr1e-5.txt` |
| 1.0 | TAL | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384/eval/test_fixed_rewards_0_and_1/AudioRewardConditionedMaskLMReward1p0/TAL_test_reward1p0_epoch5_lr1e-5.txt` |
| 1.0 | chime6 | 1 | `1e-5` | 0.843638 | 0.822259 | -0.021378 | -2.53 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384/eval/test_fixed_rewards_0_and_1/AudioRewardConditionedMaskLMReward1p0/chime6_test_reward1p0_epoch1_lr1e-5.txt` |
| 1.0 | chime6 | 5 | `1e-5` | 0.843620 | 1.000000 | 0.156380 | 18.54 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384/eval/test_fixed_rewards_0_and_1/AudioRewardConditionedMaskLMReward1p0/chime6_test_reward1p0_epoch5_lr1e-5.txt` |
| 0.0 | tedlium | 1 | `1e-5` | 0.085345 | 0.077477 | -0.007868 | -9.22 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384/eval/test_fixed_rewards_0_and_1/AudioRewardConditionedMaskLMReward0p0/tedlium_test_reward0p0_epoch1_lr1e-5.txt` |
| 0.0 | tedlium | 5 | `1e-5` | 0.085345 | 0.074783 | -0.010562 | -12.38 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384/eval/test_fixed_rewards_0_and_1/AudioRewardConditionedMaskLMReward0p0/tedlium_test_reward0p0_epoch5_lr1e-5.txt` |
| 0.0 | earnings22 | 1 | `1e-5` | 0.235198 | 0.199007 | -0.036191 | -15.39 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384/eval/test_fixed_rewards_0_and_1/AudioRewardConditionedMaskLMReward0p0/earnings22_test_reward0p0_epoch1_lr1e-5.txt` |
| 0.0 | earnings22 | 5 | `1e-5` | 0.235218 | 0.183731 | -0.051488 | -21.89 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384/eval/test_fixed_rewards_0_and_1/AudioRewardConditionedMaskLMReward0p0/earnings22_test_reward0p0_epoch5_lr1e-5.txt` |
| 0.0 | rev16 | 1 | `1e-5` | 0.172576 | 0.165023 | -0.007552 | -4.38 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384/eval/test_fixed_rewards_0_and_1/AudioRewardConditionedMaskLMReward0p0/rev16_test_reward0p0_epoch1_lr1e-5.txt` |
| 0.0 | rev16 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384/eval/test_fixed_rewards_0_and_1/AudioRewardConditionedMaskLMReward0p0/rev16_test_reward0p0_epoch5_lr1e-5.txt` |
| 0.0 | TAL | 1 | `1e-5` | 0.165691 | 0.160867 | -0.004824 | -2.91 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384/eval/test_fixed_rewards_0_and_1/AudioRewardConditionedMaskLMReward0p0/TAL_test_reward0p0_epoch1_lr1e-5.txt` |
| 0.0 | TAL | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384/eval/test_fixed_rewards_0_and_1/AudioRewardConditionedMaskLMReward0p0/TAL_test_reward0p0_epoch5_lr1e-5.txt` |
| 0.0 | chime6 | 1 | `1e-5` | 0.843638 | 0.827194 | -0.016443 | -1.95 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384/eval/test_fixed_rewards_0_and_1/AudioRewardConditionedMaskLMReward0p0/chime6_test_reward0p0_epoch1_lr1e-5.txt` |
| 0.0 | chime6 | 5 | `1e-5` | 0.843620 | 1.000000 | 0.156380 | 18.54 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384/eval/test_fixed_rewards_0_and_1/AudioRewardConditionedMaskLMReward0p0/chime6_test_reward0p0_epoch5_lr1e-5.txt` |

CSV artifact:

```text
exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384/eval/test_fixed_rewards_0_and_1/rob132_audio_ssl_test_fixed_rewards.csv
```
