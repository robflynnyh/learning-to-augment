# ROB-338 RC-MLM Fixed-Reward WER Repeats

## Metadata

- Checkpoint: `/mnt/parscratch/users/acp21rjf/l2augment_model/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt`
- Policy: `RewardConditionedMaskLM`, `hidden_dim=384`, `dropout=0.1`
- Reward controls: fixed `conditioning_reward: 1.0` and fixed `conditioning_reward: 0.0` as separate runs
- Datasets: `tedlium`, `earnings22`, `chime6`, `rev16`, `TAL`; all `test` split
- Adaptation: `epochs=1` and `epochs=5`, `lr=1e-5`, multistep rollout
- Branch: `symphony/ROB-338-rc-mlm-repeats`
- Commit: `fbd45153372f3cb373a4cb5f2f784233a6d5c41f`
- Main log: `/mnt/parscratch/users/acp21rjf/symphony-workspaces-learning-to-augment/ROB-338/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/logs/stanage/rob338/rob338_stanage_finalizer-10774945.log`
- Screen log: `slurm`
- Queued command: `scripts/submit_rob338_rc_mlm_repeats_stanage.sh`

Completed cells: `60/60`.

## Aggregate

| Reward | Dataset | Epochs | N | Mean Original WER | Mean Updated WER | Updated WER Std | Mean Abs Delta | Mean Rel Delta % |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | TAL | 1 | 3 | 0.165691 | 0.162876 | 0.000024 | -0.002815 | -1.70 |
| 0.0 | TAL | 5 | 3 | 0.165692 | 0.159748 | 0.000322 | -0.005945 | -3.59 |
| 0.0 | chime6 | 1 | 3 | 0.843597 | 0.812678 | 0.002640 | -0.030919 | -3.67 |
| 0.0 | chime6 | 5 | 3 | 0.843614 | 0.874762 | 0.108515 | 0.031148 | 3.69 |
| 0.0 | earnings22 | 1 | 3 | 0.235225 | 0.202050 | 0.001695 | -0.033175 | -14.10 |
| 0.0 | earnings22 | 5 | 3 | 0.235225 | 0.185746 | 0.001218 | -0.049480 | -21.04 |
| 0.0 | rev16 | 1 | 3 | 0.172550 | 0.166846 | 0.000389 | -0.005704 | -3.31 |
| 0.0 | rev16 | 5 | 3 | 0.172550 | 0.162895 | 0.000638 | -0.009655 | -5.60 |
| 0.0 | tedlium | 1 | 3 | 0.085345 | 0.079060 | 0.000362 | -0.006285 | -7.36 |
| 0.0 | tedlium | 5 | 3 | 0.085345 | 0.075315 | 0.000525 | -0.010030 | -11.75 |
| 1.0 | TAL | 1 | 3 | 0.165695 | 0.158823 | 0.000327 | -0.006871 | -4.15 |
| 1.0 | TAL | 5 | 3 | 0.165693 | 0.156741 | 0.002085 | -0.008952 | -5.40 |
| 1.0 | chime6 | 1 | 3 | 0.843626 | 0.827241 | 0.004231 | -0.016385 | -1.94 |
| 1.0 | chime6 | 5 | 3 | 0.843608 | 1.000000 | 0.000000 | 0.156392 | 18.54 |
| 1.0 | earnings22 | 1 | 3 | 0.235211 | 0.195440 | 0.001725 | -0.039771 | -16.91 |
| 1.0 | earnings22 | 5 | 3 | 0.235218 | 0.186461 | 0.001197 | -0.048758 | -20.73 |
| 1.0 | rev16 | 1 | 3 | 0.172554 | 0.163201 | 0.000384 | -0.009353 | -5.42 |
| 1.0 | rev16 | 5 | 3 | 0.172550 | 0.159687 | 0.000235 | -0.012863 | -7.45 |
| 1.0 | tedlium | 1 | 3 | 0.085345 | 0.077855 | 0.000379 | -0.007490 | -8.78 |
| 1.0 | tedlium | 5 | 3 | 0.085345 | 0.075894 | 0.000309 | -0.009451 | -11.07 |

## Per Cell

| Reward | Dataset | Repeat | Seed | Epochs | LR | Original WER | Updated WER | Abs Delta | Rel Delta % | Status | Result |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1.0 | tedlium | 1 | 123456 | 1 | `1e-5` | 0.085345 | 0.078292 | -0.007053 | -8.26 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/tedlium_test_epoch1_lr1e-5.txt` |
| 1.0 | tedlium | 1 | 123456 | 5 | `1e-5` | 0.085345 | 0.076236 | -0.009109 | -10.67 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/tedlium_test_epoch5_lr1e-5.txt` |
| 1.0 | tedlium | 2 | 123457 | 1 | `1e-5` | 0.085345 | 0.077654 | -0.007691 | -9.01 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/tedlium_test_epoch1_lr1e-5_repeat2.txt` |
| 1.0 | tedlium | 2 | 123457 | 5 | `1e-5` | 0.085345 | 0.075634 | -0.009711 | -11.38 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/tedlium_test_epoch5_lr1e-5_repeat2.txt` |
| 1.0 | tedlium | 3 | 123458 | 1 | `1e-5` | 0.085345 | 0.077618 | -0.007726 | -9.05 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/tedlium_test_epoch1_lr1e-5_repeat3.txt` |
| 1.0 | tedlium | 3 | 123458 | 5 | `1e-5` | 0.085345 | 0.075811 | -0.009534 | -11.17 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/tedlium_test_epoch5_lr1e-5_repeat3.txt` |
| 1.0 | earnings22 | 1 | 123456 | 1 | `1e-5` | 0.235218 | 0.194004 | -0.041215 | -17.52 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/earnings22_test_epoch1_lr1e-5.txt` |
| 1.0 | earnings22 | 1 | 123456 | 5 | `1e-5` | 0.235198 | 0.186386 | -0.048812 | -20.75 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/earnings22_test_epoch5_lr1e-5.txt` |
| 1.0 | earnings22 | 2 | 123457 | 1 | `1e-5` | 0.235218 | 0.197353 | -0.037865 | -16.10 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/earnings22_test_epoch1_lr1e-5_repeat2.txt` |
| 1.0 | earnings22 | 2 | 123457 | 5 | `1e-5` | 0.235239 | 0.187693 | -0.047546 | -20.21 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/earnings22_test_epoch5_lr1e-5_repeat2.txt` |
| 1.0 | earnings22 | 3 | 123458 | 1 | `1e-5` | 0.235198 | 0.194964 | -0.040234 | -17.11 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/earnings22_test_epoch1_lr1e-5_repeat3.txt` |
| 1.0 | earnings22 | 3 | 123458 | 5 | `1e-5` | 0.235218 | 0.185303 | -0.049915 | -21.22 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/earnings22_test_epoch5_lr1e-5_repeat3.txt` |
| 1.0 | rev16 | 1 | 123456 | 1 | `1e-5` | 0.172509 | 0.163599 | -0.008911 | -5.17 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/rev16_test_epoch1_lr1e-5.txt` |
| 1.0 | rev16 | 1 | 123456 | 5 | `1e-5` | 0.172504 | 0.159958 | -0.012546 | -7.27 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/rev16_test_epoch5_lr1e-5.txt` |
| 1.0 | rev16 | 2 | 123457 | 1 | `1e-5` | 0.172576 | 0.162833 | -0.009743 | -5.65 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/rev16_test_epoch1_lr1e-5_repeat2.txt` |
| 1.0 | rev16 | 2 | 123457 | 5 | `1e-5` | 0.172571 | 0.159570 | -0.013001 | -7.53 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/rev16_test_epoch5_lr1e-5_repeat2.txt` |
| 1.0 | rev16 | 3 | 123458 | 1 | `1e-5` | 0.172576 | 0.163170 | -0.009406 | -5.45 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/rev16_test_epoch1_lr1e-5_repeat3.txt` |
| 1.0 | rev16 | 3 | 123458 | 5 | `1e-5` | 0.172576 | 0.159534 | -0.013042 | -7.56 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/rev16_test_epoch5_lr1e-5_repeat3.txt` |
| 1.0 | TAL | 1 | 123456 | 1 | `1e-5` | 0.165702 | 0.159077 | -0.006626 | -4.00 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/TAL_test_epoch1_lr1e-5.txt` |
| 1.0 | TAL | 1 | 123456 | 5 | `1e-5` | 0.165702 | 0.155694 | -0.010009 | -6.04 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/TAL_test_epoch5_lr1e-5.txt` |
| 1.0 | TAL | 2 | 123457 | 1 | `1e-5` | 0.165691 | 0.158939 | -0.006752 | -4.08 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/TAL_test_epoch1_lr1e-5_repeat2.txt` |
| 1.0 | TAL | 2 | 123457 | 5 | `1e-5` | 0.165686 | 0.159142 | -0.006544 | -3.95 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/TAL_test_epoch5_lr1e-5_repeat2.txt` |
| 1.0 | TAL | 3 | 123458 | 1 | `1e-5` | 0.165691 | 0.158454 | -0.007237 | -4.37 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/TAL_test_epoch1_lr1e-5_repeat3.txt` |
| 1.0 | TAL | 3 | 123458 | 5 | `1e-5` | 0.165691 | 0.155386 | -0.010305 | -6.22 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/TAL_test_epoch5_lr1e-5_repeat3.txt` |
| 1.0 | chime6 | 1 | 123456 | 1 | `1e-5` | 0.843620 | 0.830649 | -0.012971 | -1.54 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/chime6_test_epoch1_lr1e-5.txt` |
| 1.0 | chime6 | 1 | 123456 | 5 | `1e-5` | 0.843620 | 1.000000 | 0.156380 | 18.54 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/chime6_test_epoch5_lr1e-5.txt` |
| 1.0 | chime6 | 2 | 123457 | 1 | `1e-5` | 0.843620 | 0.828569 | -0.015051 | -1.78 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/chime6_test_epoch1_lr1e-5_repeat2.txt` |
| 1.0 | chime6 | 2 | 123457 | 5 | `1e-5` | 0.843585 | 1.000000 | 0.156415 | 18.54 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/chime6_test_epoch5_lr1e-5_repeat2.txt` |
| 1.0 | chime6 | 3 | 123458 | 1 | `1e-5` | 0.843638 | 0.822506 | -0.021131 | -2.50 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/chime6_test_epoch1_lr1e-5_repeat3.txt` |
| 1.0 | chime6 | 3 | 123458 | 5 | `1e-5` | 0.843620 | 1.000000 | 0.156380 | 18.54 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward1/chime6_test_epoch5_lr1e-5_repeat3.txt` |
| 0.0 | tedlium | 1 | 123456 | 1 | `1e-5` | 0.085345 | 0.079320 | -0.006025 | -7.06 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/tedlium_test_epoch1_lr1e-5.txt` |
| 0.0 | tedlium | 1 | 123456 | 5 | `1e-5` | 0.085345 | 0.074712 | -0.010633 | -12.46 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/tedlium_test_epoch5_lr1e-5.txt` |
| 0.0 | tedlium | 2 | 123457 | 1 | `1e-5` | 0.085345 | 0.078646 | -0.006699 | -7.85 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/tedlium_test_epoch1_lr1e-5_repeat2.txt` |
| 0.0 | tedlium | 2 | 123457 | 5 | `1e-5` | 0.085345 | 0.075563 | -0.009782 | -11.46 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/tedlium_test_epoch5_lr1e-5_repeat2.txt` |
| 0.0 | tedlium | 3 | 123458 | 1 | `1e-5` | 0.085345 | 0.079213 | -0.006131 | -7.18 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/tedlium_test_epoch1_lr1e-5_repeat3.txt` |
| 0.0 | tedlium | 3 | 123458 | 5 | `1e-5` | 0.085345 | 0.075669 | -0.009676 | -11.34 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/tedlium_test_epoch5_lr1e-5_repeat3.txt` |
| 0.0 | earnings22 | 1 | 123456 | 1 | `1e-5` | 0.235218 | 0.202091 | -0.033127 | -14.08 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/earnings22_test_epoch1_lr1e-5.txt` |
| 0.0 | earnings22 | 1 | 123456 | 5 | `1e-5` | 0.235239 | 0.185201 | -0.050038 | -21.27 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/earnings22_test_epoch5_lr1e-5.txt` |
| 0.0 | earnings22 | 2 | 123457 | 1 | `1e-5` | 0.235239 | 0.203725 | -0.031514 | -13.40 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/earnings22_test_epoch1_lr1e-5_repeat2.txt` |
| 0.0 | earnings22 | 2 | 123457 | 5 | `1e-5` | 0.235198 | 0.184895 | -0.050303 | -21.39 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/earnings22_test_epoch5_lr1e-5_repeat2.txt` |
| 0.0 | earnings22 | 3 | 123458 | 1 | `1e-5` | 0.235218 | 0.200335 | -0.034883 | -14.83 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/earnings22_test_epoch1_lr1e-5_repeat3.txt` |
| 0.0 | earnings22 | 3 | 123458 | 5 | `1e-5` | 0.235239 | 0.187141 | -0.048098 | -20.45 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/earnings22_test_epoch5_lr1e-5_repeat3.txt` |
| 0.0 | rev16 | 1 | 123456 | 1 | `1e-5` | 0.172504 | 0.166402 | -0.006102 | -3.54 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/rev16_test_epoch1_lr1e-5.txt` |
| 0.0 | rev16 | 1 | 123456 | 5 | `1e-5` | 0.172504 | 0.163593 | -0.008911 | -5.17 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/rev16_test_epoch5_lr1e-5.txt` |
| 0.0 | rev16 | 2 | 123457 | 1 | `1e-5` | 0.172571 | 0.167122 | -0.005449 | -3.16 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/rev16_test_epoch1_lr1e-5_repeat2.txt` |
| 0.0 | rev16 | 2 | 123457 | 5 | `1e-5` | 0.172576 | 0.162342 | -0.010233 | -5.93 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/rev16_test_epoch5_lr1e-5_repeat2.txt` |
| 0.0 | rev16 | 3 | 123458 | 1 | `1e-5` | 0.172576 | 0.167015 | -0.005561 | -3.22 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/rev16_test_epoch1_lr1e-5_repeat3.txt` |
| 0.0 | rev16 | 3 | 123458 | 5 | `1e-5` | 0.172571 | 0.162751 | -0.009820 | -5.69 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/rev16_test_epoch5_lr1e-5_repeat3.txt` |
| 0.0 | TAL | 1 | 123456 | 1 | `1e-5` | 0.165694 | 0.162852 | -0.002842 | -1.72 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/TAL_test_epoch1_lr1e-5.txt` |
| 0.0 | TAL | 1 | 123456 | 5 | `1e-5` | 0.165700 | 0.159556 | -0.006143 | -3.71 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/TAL_test_epoch5_lr1e-5.txt` |
| 0.0 | TAL | 2 | 123457 | 1 | `1e-5` | 0.165688 | 0.162900 | -0.002788 | -1.68 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/TAL_test_epoch1_lr1e-5_repeat2.txt` |
| 0.0 | TAL | 2 | 123457 | 5 | `1e-5` | 0.165691 | 0.160120 | -0.005571 | -3.36 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/TAL_test_epoch5_lr1e-5_repeat2.txt` |
| 0.0 | TAL | 3 | 123458 | 1 | `1e-5` | 0.165691 | 0.162875 | -0.002817 | -1.70 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/TAL_test_epoch1_lr1e-5_repeat3.txt` |
| 0.0 | TAL | 3 | 123458 | 5 | `1e-5` | 0.165686 | 0.159567 | -0.006118 | -3.69 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/TAL_test_epoch5_lr1e-5_repeat3.txt` |
| 0.0 | chime6 | 1 | 123456 | 1 | `1e-5` | 0.843585 | 0.813359 | -0.030226 | -3.58 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/chime6_test_epoch1_lr1e-5.txt` |
| 0.0 | chime6 | 1 | 123456 | 5 | `1e-5` | 0.843620 | 0.808213 | -0.035407 | -4.20 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/chime6_test_epoch5_lr1e-5.txt` |
| 0.0 | chime6 | 2 | 123457 | 1 | `1e-5` | 0.843585 | 0.809764 | -0.033821 | -4.01 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/chime6_test_epoch1_lr1e-5_repeat2.txt` |
| 0.0 | chime6 | 2 | 123457 | 5 | `1e-5` | 0.843638 | 0.816091 | -0.027547 | -3.27 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/chime6_test_epoch5_lr1e-5_repeat2.txt` |
| 0.0 | chime6 | 3 | 123458 | 1 | `1e-5` | 0.843620 | 0.814910 | -0.028710 | -3.40 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/chime6_test_epoch1_lr1e-5_repeat3.txt` |
| 0.0 | chime6 | 3 | 123458 | 5 | `1e-5` | 0.843585 | 0.999982 | 0.156398 | 18.54 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/RewardConditionedMaskLMReward0/chime6_test_epoch5_lr1e-5_repeat3.txt` |

CSV artifact:

```text
/mnt/parscratch/users/acp21rjf/symphony-workspaces-learning-to-augment/ROB-338/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_reward_conditioning/all_dataset_fixed_rewards_0_and_1/rob124_384_dropout_all_dataset_fixed_rewards_0_and_1.csv
```
