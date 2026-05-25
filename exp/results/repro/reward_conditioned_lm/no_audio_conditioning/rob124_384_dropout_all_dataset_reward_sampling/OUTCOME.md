# ROB-124 384-Dropout All-Dataset Reward Sampling Eval

## Metadata

- Checkpoint: `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt`
- Policy: `RewardConditionedMaskLM`, `hidden_dim=384`, `dropout=0.1`
- Reward control: sampled uniformly from `[0.5, 1.0]` during each adaptation mask generation step
- Datasets: `tedlium`, `earnings22`, `chime6`, `rev16`, `TAL`; all `test` split
- Adaptation: `epochs=1` and `epochs=5`, `lr=1e-5`, multistep rollout
- Branch: `symphony/ROB-124-384-dropout-mask-lm`
- Commit: `ab31d28c57a5c0f828a457d519108b2e0a541910`
- Main log: `/exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_reward_sampling/logs/rob124_384_dropout_all_dataset_reward_sampling.log`
- Screen log: `/exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_reward_sampling/logs/rob124_384_dropout_all_dataset_reward_sampling.screen.log`
- Queued command: `screen -L -Logfile /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_reward_sampling/logs/rob124_384_dropout_all_dataset_reward_sampling.screen.log -dmS rob124-384-dropout-all-dataset-sampling bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob124_384_dropout_all_dataset_reward_sampling.sh'`

Completed cells: `10/10`.

## Interpretation

The all-dataset `[0.5, 1.0]` sampled-reward eval completed cleanly. Nine of ten
cells improved WER versus the unadapted original WER, including every
1-epoch cell. The exception is CHiME-6 at 5 adaptation epochs, which regressed
from `0.843620` to `1.000000`; this makes the 5-epoch setting unsafe as a
blanket default even though the other 5-epoch cells improved.

For downstream use, the 384/dropout checkpoint remains usable and remains the
preferred checkpoint from the ROB-124 capacity/dropout comparison. This
all-dataset follow-up supports using sampled reward `[0.5, 1.0]`, especially at
1 adaptation epoch, but CHiME-6 should be handled separately before claiming
that longer adaptation is robust across datasets.

## Aggregate

| Dataset | Epochs | N | Mean Original WER | Mean Updated WER | Updated WER Std | Mean Abs Delta | Mean Rel Delta % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TAL | 1 | 1 | 0.165708 | 0.160010 | 0.000000 | -0.005698 | -3.44 |
| TAL | 5 | 1 | 0.165700 | 0.155231 | 0.000000 | -0.010469 | -6.32 |
| chime6 | 1 | 1 | 0.843585 | 0.812689 | 0.000000 | -0.030896 | -3.66 |
| chime6 | 5 | 1 | 0.843620 | 1.000000 | 0.000000 | 0.156380 | 18.54 |
| earnings22 | 1 | 1 | 0.235198 | 0.196107 | 0.000000 | -0.039091 | -16.62 |
| earnings22 | 5 | 1 | 0.235239 | 0.182260 | 0.000000 | -0.052979 | -22.52 |
| rev16 | 1 | 1 | 0.172509 | 0.163920 | 0.000000 | -0.008589 | -4.98 |
| rev16 | 5 | 1 | 0.172504 | 0.159958 | 0.000000 | -0.012546 | -7.27 |
| tedlium | 1 | 1 | 0.085345 | 0.076590 | 0.000000 | -0.008755 | -10.26 |
| tedlium | 5 | 1 | 0.085345 | 0.074180 | 0.000000 | -0.011165 | -13.08 |

## Per Cell

| Dataset | Repeat | Seed | Epochs | LR | Original WER | Updated WER | Abs Delta | Rel Delta % | Status | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| tedlium | 1 | 123456 | 1 | `1e-5` | 0.085345 | 0.076590 | -0.008754 | -10.26 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_reward_sampling/RewardConditionedMaskLMUniform0p5to1/tedlium_test_epoch1_lr1e-5.txt` |
| tedlium | 1 | 123456 | 5 | `1e-5` | 0.085345 | 0.074180 | -0.011164 | -13.08 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_reward_sampling/RewardConditionedMaskLMUniform0p5to1/tedlium_test_epoch5_lr1e-5.txt` |
| earnings22 | 1 | 123456 | 1 | `1e-5` | 0.235198 | 0.196107 | -0.039091 | -16.62 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_reward_sampling/RewardConditionedMaskLMUniform0p5to1/earnings22_test_epoch1_lr1e-5.txt` |
| earnings22 | 1 | 123456 | 5 | `1e-5` | 0.235239 | 0.182260 | -0.052979 | -22.52 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_reward_sampling/RewardConditionedMaskLMUniform0p5to1/earnings22_test_epoch5_lr1e-5.txt` |
| chime6 | 1 | 123456 | 1 | `1e-5` | 0.843585 | 0.812689 | -0.030895 | -3.66 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_reward_sampling/RewardConditionedMaskLMUniform0p5to1/chime6_test_epoch1_lr1e-5.txt` |
| chime6 | 1 | 123456 | 5 | `1e-5` | 0.843620 | 1.000000 | 0.156380 | 18.54 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_reward_sampling/RewardConditionedMaskLMUniform0p5to1/chime6_test_epoch5_lr1e-5.txt` |
| rev16 | 1 | 123456 | 1 | `1e-5` | 0.172509 | 0.163920 | -0.008589 | -4.98 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_reward_sampling/RewardConditionedMaskLMUniform0p5to1/rev16_test_epoch1_lr1e-5.txt` |
| rev16 | 1 | 123456 | 5 | `1e-5` | 0.172504 | 0.159958 | -0.012546 | -7.27 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_reward_sampling/RewardConditionedMaskLMUniform0p5to1/rev16_test_epoch5_lr1e-5.txt` |
| TAL | 1 | 123456 | 1 | `1e-5` | 0.165708 | 0.160010 | -0.005698 | -3.44 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_reward_sampling/RewardConditionedMaskLMUniform0p5to1/TAL_test_epoch1_lr1e-5.txt` |
| TAL | 1 | 123456 | 5 | `1e-5` | 0.165700 | 0.155231 | -0.010468 | -6.32 | complete | `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_reward_sampling/RewardConditionedMaskLMUniform0p5to1/TAL_test_epoch5_lr1e-5.txt` |

CSV artifact:

```text
/exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout_all_dataset_reward_sampling/rob124_384_dropout_all_dataset_reward_sampling.csv
```
