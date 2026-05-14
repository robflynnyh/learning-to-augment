# ROB-80 TED-LIUM Dev Audio-Conditioned CMultiStepVQLM Reward Conditioning Comparison

Audio-conditioned CMultiStepVQLM uses the legacy score-conditioned audio checkpoint with condition_on_audio true and use_signal_inputs false. This compares fixed reward 1.0 against uniform [0.5, 1.0] random reward conditioning.

Completed cells: 0/12

## Aggregate

| Method | Conditioning | Epochs | LR | N | Mean Original WER | Mean Updated WER | Updated WER Std | Mean Abs Delta | Mean Rel Delta % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## Per Repeat

| Method | Conditioning | Repeat | Seed | Epochs | LR | Original WER | Updated WER | Abs Delta | Rel Delta % | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 1 | 123456 | 1 | `5e-6` |  |  |  |  | missing |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 1 | 123456 | 1 | `2e-5` |  |  |  |  | missing |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 1 | 123456 | 5 | `5e-6` |  |  |  |  | missing |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing |
| CMultiStepVQLMAudio | audio_fixed_1.0 | 1 | 123456 | 5 | `2e-5` |  |  |  |  | missing |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 1 | 123456 | 1 | `5e-6` |  |  |  |  | missing |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 1 | 123456 | 1 | `1e-5` |  |  |  |  | missing |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 1 | 123456 | 1 | `2e-5` |  |  |  |  | missing |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 1 | 123456 | 5 | `5e-6` |  |  |  |  | missing |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 1 | 123456 | 5 | `1e-5` |  |  |  |  | missing |
| CMultiStepVQLMAudioRandomReward | audio_uniform_0.5_1.0 | 1 | 123456 | 5 | `2e-5` |  |  |  |  | missing |
