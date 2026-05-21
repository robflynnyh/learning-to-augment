# ROB-106 TED-LIUM Dev No-Audio CMultiStepVQLM Low-Reward Conditioning

Compares the ROB-80 best comparable no-audio CMultiStepVQLM setting, TED-LIUM dev with 5 adaptation epochs and lr=5e-6, under fixed reward 1.0, fixed reward 0.0, and random reward sampled uniformly from [0.0, 1.0]. Rewards are in the trained MultiStepDataset min-max normalized range.

Policy checkpoint: `/store/store5/data/acp21rjf_checkpoints/l2augment/models/CMultiStepMLM/no_audio_modelsignals.pt`

ASR checkpoint: `/store/store5/data/acp21rjf_checkpoints/l2augment/asr/step_105360.pt`

Launch command: `/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob106_tedlium_low_reward_conditioning.sh`

Interpretation: fixed reward `1.0` did not outperform fixed reward `0.0` on the two-repeat mean; uniform random conditioning over `[0.0, 1.0]` was best in this small comparison.

Completed cells: 6/6

## Aggregate

| Method | Conditioning | Epochs | LR | N | Mean Original WER | Mean Updated WER | Updated WER Std | Mean Abs Delta | Mean Rel Delta % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CMultiStepVQLMReward1 | fixed_1.0 | 5 | `5e-6` | 2 | 0.100088 | 0.087764 | 0.000625 | -0.012324 | -12.31 |
| CMultiStepVQLMReward0 | fixed_0.0 | 5 | `5e-6` | 2 | 0.100088 | 0.087625 | 0.000508 | -0.012463 | -12.45 |
| CMultiStepVQLMRandomReward0to1 | uniform_0.0_1.0 | 5 | `5e-6` | 2 | 0.100088 | 0.086244 | 0.001525 | -0.013844 | -13.83 |

## Per Repeat

| Method | Conditioning | Repeat | Seed | Epochs | LR | Config | Original WER | Updated WER | Abs Delta | Rel Delta % | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| CMultiStepVQLMReward1 | fixed_1.0 | 1 | 123456 | 5 | `5e-6` | `exp/results/repro/sweeps/no_audio_cmultistep_vqlm/rob106_low_reward_conditioning/CMultiStepVQLMReward1/configs/tedlium_dev_epoch5_lr5e-6.yaml` | 0.100088 | 0.088206 | -0.011882 | -11.87 | complete |
| CMultiStepVQLMReward1 | fixed_1.0 | 2 | 123457 | 5 | `5e-6` | `exp/results/repro/sweeps/no_audio_cmultistep_vqlm/rob106_low_reward_conditioning/CMultiStepVQLMReward1/configs/tedlium_dev_epoch5_lr5e-6_repeat2.yaml` | 0.100088 | 0.087322 | -0.012767 | -12.76 | complete |
| CMultiStepVQLMReward0 | fixed_0.0 | 1 | 123456 | 5 | `5e-6` | `exp/results/repro/sweeps/no_audio_cmultistep_vqlm/rob106_low_reward_conditioning/CMultiStepVQLMReward0/configs/tedlium_dev_epoch5_lr5e-6.yaml` | 0.100088 | 0.087266 | -0.012822 | -12.81 | complete |
| CMultiStepVQLMReward0 | fixed_0.0 | 2 | 123457 | 5 | `5e-6` | `exp/results/repro/sweeps/no_audio_cmultistep_vqlm/rob106_low_reward_conditioning/CMultiStepVQLMReward0/configs/tedlium_dev_epoch5_lr5e-6_repeat2.yaml` | 0.100088 | 0.087985 | -0.012103 | -12.09 | complete |
| CMultiStepVQLMRandomReward0to1 | uniform_0.0_1.0 | 1 | 123456 | 5 | `5e-6` | `exp/results/repro/sweeps/no_audio_cmultistep_vqlm/rob106_low_reward_conditioning/CMultiStepVQLMRandomReward0to1/configs/tedlium_dev_epoch5_lr5e-6.yaml` | 0.100088 | 0.085166 | -0.014922 | -14.91 | complete |
| CMultiStepVQLMRandomReward0to1 | uniform_0.0_1.0 | 2 | 123457 | 5 | `5e-6` | `exp/results/repro/sweeps/no_audio_cmultistep_vqlm/rob106_low_reward_conditioning/CMultiStepVQLMRandomReward0to1/configs/tedlium_dev_epoch5_lr5e-6_repeat2.yaml` | 0.100088 | 0.087322 | -0.012767 | -12.76 | complete |
