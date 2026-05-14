# ROB-80 Audio-Conditioned CMultiStepVQLM Sweep

This directory stores the TED-LIUM dev audio-conditioned CMultiStepVQLM
follow-up for ROB-80.

- `CMultiStepVQLMAudio/` contains the fixed reward-conditioning baseline with
  `default_conditioning_reward: 1.0`.
- `CMultiStepVQLMAudioRandomReward/` contains the variant where each generated
  mask samples the conditioning reward uniformly from `[0.5, 1.0]`.
- The audio-conditioned checkpoint is
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/CMultiStepMLM/curbest.pt`.
- The audio VAE checkpoint is
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/autoenc_audio/model_gpu.pt`.
- This checkpoint is the legacy score-conditioned audio model, so generated
  configs set `condition_on_audio: true` and `use_signal_inputs: false`.

The queued launcher is
`scripts/launch_rob80_tedlium_audio_cmultistep_sweep.sh`. It writes
`ROB-80_AUDIO_REWARD_CONDITIONING_COMPARISON.md` and
`rob80_tedlium_audio_reward_conditioning_comparison.csv`.
