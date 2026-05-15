# ROB-80 TED-LIUM Dev Audio-Conditioned CMultiStepVQLM Reward Conditioning Comparison

Audio-conditioned CMultiStepVQLM uses the legacy score-conditioned audio checkpoint with condition_on_audio true and use_signal_inputs false. This compares fixed reward 1.0 against uniform [0.5, 1.0] random reward conditioning.

Completed cells: 12/12

## Aggregate

| Method | Conditioning | Epochs | LR | N | Mean Original WER | Mean Updated WER | Updated WER Std | Mean Abs Delta | Mean Rel Delta % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 1 | `5e-6` | 1 | 0.100088 | 0.090472 | 0.000000 | -0.009616 | -9.61 |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 1 | `1e-5` | 1 | 0.100088 | 0.090030 | 0.000000 | -0.010058 | -10.05 |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 1 | `2e-5` | 1 | 0.100088 | 0.090748 | 0.000000 | -0.009340 | -9.33 |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 5 | `5e-6` | 1 | 0.100088 | 0.089256 | 0.000000 | -0.010832 | -10.82 |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 5 | `1e-5` | 1 | 0.100088 | 0.088427 | 0.000000 | -0.011661 | -11.65 |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 5 | `2e-5` | 1 | 0.100088 | 0.089311 | 0.000000 | -0.010777 | -10.77 |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 1 | `5e-6` | 1 | 0.100088 | 0.088427 | 0.000000 | -0.011661 | -11.65 |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 1 | `1e-5` | 1 | 0.100088 | 0.087598 | 0.000000 | -0.012490 | -12.48 |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 1 | `2e-5` | 1 | 0.100088 | 0.089809 | 0.000000 | -0.010279 | -10.27 |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 5 | `5e-6` | 1 | 0.100088 | 0.086880 | 0.000000 | -0.013208 | -13.20 |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 5 | `1e-5` | 1 | 0.100088 | 0.087488 | 0.000000 | -0.012600 | -12.59 |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 5 | `2e-5` | 1 | 0.100088 | 0.089090 | 0.000000 | -0.010998 | -10.99 |

## Per Repeat

| Method | Conditioning | Repeat | Seed | Epochs | LR | Original WER | Updated WER | Abs Delta | Rel Delta % | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 1 | 123456 | 1 | `5e-6` | 0.100088 | 0.090472 | -0.009616 | -9.61 | complete |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 1 | 123456 | 1 | `1e-5` | 0.100088 | 0.090030 | -0.010059 | -10.05 | complete |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 1 | 123456 | 1 | `2e-5` | 0.100088 | 0.090748 | -0.009340 | -9.33 | complete |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 1 | 123456 | 5 | `5e-6` | 0.100088 | 0.089256 | -0.010832 | -10.82 | complete |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 1 | 123456 | 5 | `1e-5` | 0.100088 | 0.088427 | -0.011661 | -11.65 | complete |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 1 | 123456 | 5 | `2e-5` | 0.100088 | 0.089311 | -0.010777 | -10.77 | complete |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 1 | 123456 | 1 | `5e-6` | 0.100088 | 0.088427 | -0.011661 | -11.65 | complete |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 1 | 123456 | 1 | `1e-5` | 0.100088 | 0.087598 | -0.012490 | -12.48 | complete |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 1 | 123456 | 1 | `2e-5` | 0.100088 | 0.089809 | -0.010280 | -10.27 | complete |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 1 | 123456 | 5 | `5e-6` | 0.100088 | 0.086880 | -0.013209 | -13.20 | complete |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 1 | 123456 | 5 | `1e-5` | 0.100088 | 0.087488 | -0.012601 | -12.59 | complete |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 1 | 123456 | 5 | `2e-5` | 0.100088 | 0.089090 | -0.010998 | -10.99 | complete |
