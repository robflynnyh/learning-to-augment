# ROB-106 Low-Reward Conditioning

This directory holds the TED-LIUM dev CMultiStepVQLM low-reward conditioning
comparison requested in ROB-106.

The comparison reuses the best comparable no-audio ROB-80 setting:

- dataset: `tedlium`
- split: `dev`
- adaptation epochs: `5`
- adaptation LR: `5e-6`
- repeats: `1 2`
- policy checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/CMultiStepMLM/no_audio_modelsignals.pt`

Compared conditioning modes:

- `CMultiStepVQLMReward1`: fixed reward `1.0`
- `CMultiStepVQLMReward0`: fixed reward `0.0`
- `CMultiStepVQLMRandomReward0to1`: uniform random reward from `[0.0, 1.0]`

Launch command:

```bash
/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob106_tedlium_low_reward_conditioning.sh
```

The follow-up UVQLM rollout folder size scan is recorded under
`uvqlm_rollout_size/OUTCOME.md`.
