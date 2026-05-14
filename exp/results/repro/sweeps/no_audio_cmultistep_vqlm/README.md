# ROB-80 No-Audio CMultiStepVQLM Sweeps

This directory stores TED-LIUM dev sweeps for the no-audio
`ConditionalMultiStepMaskGenerator` follow-ups in ROB-80.

- `CMultiStepVQLM/` contains the fixed reward-conditioning baseline with
  `default_conditioning_reward: 1.0`.
- `CMultiStepVQLMRandomReward/` is reserved for the follow-up where each
  generation samples the conditioning reward uniformly from `[0.5, 1.0]`.
- `ROB-80_NOAUDIO_CMULTISTEP_OUTCOME.md` and
  `rob80_tedlium_noaudio_cmultistep_sweep.csv` summarize the fixed baseline.
- `ROB-80_NOAUDIO_REWARD_CONDITIONING_COMPARISON.md` and
  `rob80_tedlium_noaudio_reward_conditioning_comparison.csv` compare the fixed
  and random reward-conditioning variants when the random run has completed.

The launcher is `scripts/launch_rob80_tedlium_noaudio_cmultistep_sweep.sh`.
Use `ROB80_METHOD=CMultiStepVQLMRandomReward` and
`ROB80_CONDITIONING_REWARD_RANGE="0.5 1.0"` for the randomized variant.
