# ROB-80 TED-LIUM Dev Audio-Conditioned CMultiStepVQLM Reward Conditioning Comparison

Audio-conditioned CMultiStepVQLM uses the legacy score-conditioned audio checkpoint with condition_on_audio true and use_signal_inputs false. This two-repeat comparison covers fixed reward 1.0 and uniform [0.5, 1.0] random reward conditioning.

Completed cells: 24/24

## Aggregate

| Method | Conditioning | Epochs | LR | N | Mean Original WER | Mean Updated WER | Updated WER Std | Mean Abs Delta | Mean Rel Delta % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 1 | `5e-6` | 2 | 0.100088 | 0.090168 | 0.000430 | -0.009920 | -9.91 |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 1 | `1e-5` | 2 | 0.100088 | 0.090057 | 0.000039 | -0.010030 | -10.02 |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 1 | `2e-5` | 2 | 0.100088 | 0.090444 | 0.000430 | -0.009644 | -9.64 |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 5 | `5e-6` | 2 | 0.100088 | 0.089201 | 0.000078 | -0.010887 | -10.88 |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 5 | `1e-5` | 2 | 0.100088 | 0.088095 | 0.000469 | -0.011993 | -11.98 |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 5 | `2e-5` | 2 | 0.100088 | 0.089450 | 0.000196 | -0.010638 | -10.63 |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 1 | `5e-6` | 2 | 0.100088 | 0.089671 | 0.001759 | -0.010417 | -10.41 |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 1 | `1e-5` | 2 | 0.100088 | 0.088841 | 0.001759 | -0.011247 | -11.24 |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 1 | `2e-5` | 2 | 0.100088 | 0.090030 | 0.000313 | -0.010058 | -10.05 |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 5 | `5e-6` | 2 | 0.100088 | 0.087653 | 0.001094 | -0.012435 | -12.42 |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 5 | `1e-5` | 2 | 0.100088 | 0.088068 | 0.000820 | -0.012020 | -12.01 |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 5 | `2e-5` | 2 | 0.100088 | 0.089311 | 0.000313 | -0.010777 | -10.77 |

## Per Repeat

| Method | Conditioning | Repeat | Seed | Epochs | LR | Original WER | Updated WER | Abs Delta | Rel Delta % | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 1 | 123456 | 1 | `5e-6` | 0.100088 | 0.090472 | -0.009616 | -9.61 | complete |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 1 | 123456 | 1 | `1e-5` | 0.100088 | 0.090030 | -0.010059 | -10.05 | complete |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 1 | 123456 | 1 | `2e-5` | 0.100088 | 0.090748 | -0.009340 | -9.33 | complete |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 1 | 123456 | 5 | `5e-6` | 0.100088 | 0.089256 | -0.010832 | -10.82 | complete |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 1 | 123456 | 5 | `1e-5` | 0.100088 | 0.088427 | -0.011661 | -11.65 | complete |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 1 | 123456 | 5 | `2e-5` | 0.100088 | 0.089311 | -0.010777 | -10.77 | complete |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 2 | 123457 | 1 | `5e-6` | 0.100088 | 0.089864 | -0.010224 | -10.22 | complete |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 2 | 123457 | 1 | `1e-5` | 0.100088 | 0.090085 | -0.010003 | -9.99 | complete |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 2 | 123457 | 1 | `2e-5` | 0.100088 | 0.090140 | -0.009948 | -9.94 | complete |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 2 | 123457 | 5 | `5e-6` | 0.100088 | 0.089146 | -0.010943 | -10.93 | complete |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 2 | 123457 | 5 | `1e-5` | 0.100088 | 0.087764 | -0.012325 | -12.31 | complete |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 2 | 123457 | 5 | `2e-5` | 0.100088 | 0.089588 | -0.010501 | -10.49 | complete |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 1 | 123456 | 1 | `5e-6` | 0.100088 | 0.088427 | -0.011661 | -11.65 | complete |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 1 | 123456 | 1 | `1e-5` | 0.100088 | 0.087598 | -0.012490 | -12.48 | complete |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 1 | 123456 | 1 | `2e-5` | 0.100088 | 0.089809 | -0.010280 | -10.27 | complete |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 1 | 123456 | 5 | `5e-6` | 0.100088 | 0.086880 | -0.013209 | -13.20 | complete |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 1 | 123456 | 5 | `1e-5` | 0.100088 | 0.087488 | -0.012601 | -12.59 | complete |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 1 | 123456 | 5 | `2e-5` | 0.100088 | 0.089090 | -0.010998 | -10.99 | complete |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 2 | 123457 | 1 | `5e-6` | 0.100088 | 0.090914 | -0.009174 | -9.17 | complete |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 2 | 123457 | 1 | `1e-5` | 0.100088 | 0.090085 | -0.010003 | -9.99 | complete |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 2 | 123457 | 1 | `2e-5` | 0.100088 | 0.090251 | -0.009838 | -9.83 | complete |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 2 | 123457 | 5 | `5e-6` | 0.100088 | 0.088427 | -0.011661 | -11.65 | complete |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 2 | 123457 | 5 | `1e-5` | 0.100088 | 0.088648 | -0.011440 | -11.43 | complete |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 2 | 123457 | 5 | `2e-5` | 0.100088 | 0.089532 | -0.010556 | -10.55 | complete |
