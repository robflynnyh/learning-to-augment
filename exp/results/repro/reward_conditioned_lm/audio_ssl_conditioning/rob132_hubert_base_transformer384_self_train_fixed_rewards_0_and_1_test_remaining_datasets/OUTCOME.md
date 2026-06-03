# ROB-132 Audio SSL Test-Set Fixed-Reward Eval

## Metadata

- Checkpoint: `/mnt/parscratch/users/acp21rjf/l2augment_model/reward_conditioned_mask_lm/audio_ssl_hubert_base_tedlium_per_utterance_transformer384_dropout0p1_500ep_lr1e3.pt`
- Policy: `AudioRewardConditionedMaskLM`, HuBERT SSL conditioning, transformer decoder
- Reward controls: fixed `conditioning_reward: 1.0` and fixed `conditioning_reward: 0.0` as separate runs
- Datasets: `rev16`, `TAL`, `chime6`; all `test` split
- Adaptation: `epochs=1` and `epochs=5`, `lr=1e-5`, multistep rollout
- Branch: `symphony/ROB-132-audio-conditioned-vq-mask-lm`
- Commit: `51a837d2e9c038a5675501f58e5da292f9cdcbdc`
- Main log: `/mnt/parscratch/users/acp21rjf/symphony-workspaces-learning-to-augment/ROB-132/exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_remaining_datasets/logs/stanage/rob132_stanage_finalizer-10279999.log`
- Screen log: `slurm`
- Queued command: `scripts/submit_rob132_audio_ssl_testsets_stanage.sh`

Completed cells: `8/12`.

## Interpretation

The completed 1-epoch Rev16/TAL/CHiME-6 cells all improved WER versus their
unadapted original rows. Fixed reward `1.0` was stronger than fixed reward
`0.0` for the completed Rev16, TAL, and CHiME-6 1-epoch cells:

- Rev16 epoch 1: reward `1.0` improved WER by `5.92%` relative, versus `4.38%`
  for reward `0.0`.
- TAL epoch 1: reward `1.0` improved WER by `4.38%` relative, versus `2.91%`
  for reward `0.0`.
- CHiME-6 epoch 1: reward `1.0` improved WER by `2.53%` relative, versus
  `1.95%` for reward `0.0`.

Both CHiME-6 5-epoch cells collapsed to WER `1.000000`. This matches the
broader ROB-124 fixed-reward finding that CHiME-6 is not safe under blanket
5-epoch adaptation. The four Rev16/TAL 5-epoch cells are intentionally
deferred, not accidentally absent: jobs `10279988`, `10279990`, `10279994`,
and `10279996` were cancelled after runtime estimates indicated they were
likely to hit the 4-day Stanage walltime. They should not be rerun without a
new explicit follow-up instruction.

## Aggregate

| Reward | Dataset | Epochs | N | Mean Original WER | Mean Updated WER | Updated WER Std | Mean Abs Delta | Mean Rel Delta % |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | TAL | 1 | 1 | 0.165691 | 0.160867 | 0.000000 | -0.004824 | -2.91 |
| 0.0 | chime6 | 1 | 1 | 0.843638 | 0.827194 | 0.000000 | -0.016444 | -1.95 |
| 0.0 | chime6 | 5 | 1 | 0.843620 | 1.000000 | 0.000000 | 0.156380 | 18.54 |
| 0.0 | rev16 | 1 | 1 | 0.172576 | 0.165023 | 0.000000 | -0.007553 | -4.38 |
| 1.0 | TAL | 1 | 1 | 0.165691 | 0.158428 | 0.000000 | -0.007263 | -4.38 |
| 1.0 | chime6 | 1 | 1 | 0.843638 | 0.822259 | 0.000000 | -0.021379 | -2.53 |
| 1.0 | chime6 | 5 | 1 | 0.843620 | 1.000000 | 0.000000 | 0.156380 | 18.54 |
| 1.0 | rev16 | 1 | 1 | 0.172576 | 0.162358 | 0.000000 | -0.010218 | -5.92 |

## Missing Cells

- reward 1.0 / rev16 / epoch 5 / lr `1e-5`
- reward 1.0 / TAL / epoch 5 / lr `1e-5`
- reward 0.0 / rev16 / epoch 5 / lr `1e-5`
- reward 0.0 / TAL / epoch 5 / lr `1e-5`

## Per Cell

| Reward | Dataset | Epochs | LR | Original WER | Updated WER | Abs Delta | Rel Delta % | Status | Result |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1.0 | rev16 | 1 | `1e-5` | 0.172576 | 0.162358 | -0.010218 | -5.92 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_remaining_datasets/AudioRewardConditionedMaskLMReward1p0/rev16_test_reward1p0_epoch1_lr1e-5.txt` |
| 1.0 | rev16 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_remaining_datasets/AudioRewardConditionedMaskLMReward1p0/rev16_test_reward1p0_epoch5_lr1e-5.txt` |
| 1.0 | TAL | 1 | `1e-5` | 0.165691 | 0.158428 | -0.007263 | -4.38 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_remaining_datasets/AudioRewardConditionedMaskLMReward1p0/TAL_test_reward1p0_epoch1_lr1e-5.txt` |
| 1.0 | TAL | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_remaining_datasets/AudioRewardConditionedMaskLMReward1p0/TAL_test_reward1p0_epoch5_lr1e-5.txt` |
| 1.0 | chime6 | 1 | `1e-5` | 0.843638 | 0.822259 | -0.021378 | -2.53 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_remaining_datasets/AudioRewardConditionedMaskLMReward1p0/chime6_test_reward1p0_epoch1_lr1e-5.txt` |
| 1.0 | chime6 | 5 | `1e-5` | 0.843620 | 1.000000 | 0.156380 | 18.54 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_remaining_datasets/AudioRewardConditionedMaskLMReward1p0/chime6_test_reward1p0_epoch5_lr1e-5.txt` |
| 0.0 | rev16 | 1 | `1e-5` | 0.172576 | 0.165023 | -0.007552 | -4.38 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_remaining_datasets/AudioRewardConditionedMaskLMReward0p0/rev16_test_reward0p0_epoch1_lr1e-5.txt` |
| 0.0 | rev16 | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_remaining_datasets/AudioRewardConditionedMaskLMReward0p0/rev16_test_reward0p0_epoch5_lr1e-5.txt` |
| 0.0 | TAL | 1 | `1e-5` | 0.165691 | 0.160867 | -0.004824 | -2.91 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_remaining_datasets/AudioRewardConditionedMaskLMReward0p0/TAL_test_reward0p0_epoch1_lr1e-5.txt` |
| 0.0 | TAL | 5 | `1e-5` |  |  |  |  | missing | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_remaining_datasets/AudioRewardConditionedMaskLMReward0p0/TAL_test_reward0p0_epoch5_lr1e-5.txt` |
| 0.0 | chime6 | 1 | `1e-5` | 0.843638 | 0.827194 | -0.016443 | -1.95 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_remaining_datasets/AudioRewardConditionedMaskLMReward0p0/chime6_test_reward0p0_epoch1_lr1e-5.txt` |
| 0.0 | chime6 | 5 | `1e-5` | 0.843620 | 1.000000 | 0.156380 | 18.54 | complete | `exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_remaining_datasets/AudioRewardConditionedMaskLMReward0p0/chime6_test_reward0p0_epoch5_lr1e-5.txt` |

CSV artifact:

```text
/mnt/parscratch/users/acp21rjf/symphony-workspaces-learning-to-augment/ROB-132/exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384_self_train_fixed_rewards_0_and_1_test_remaining_datasets/rob132_audio_ssl_self_train_remaining_datasets_fixed_rewards.csv
```
