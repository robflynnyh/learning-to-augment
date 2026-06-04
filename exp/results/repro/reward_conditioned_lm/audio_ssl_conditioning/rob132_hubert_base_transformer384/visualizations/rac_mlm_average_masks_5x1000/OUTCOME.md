# Outcome

Generated the requested RAC-MLM average-mask PDF from the ROB-132 HuBERT-base
audio SSL checkpoint.

## Inputs

- Config:
  `exp/configs/reward_conditioned_lm/audio_ssl_conditioning/tedlium_per_utterance_hubert_base_transformer384_dropout0p1_500ep_lr1e3.yaml`
- Checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/audio_ssl_hubert_base_tedlium_per_utterance_transformer384_dropout0p1_500ep_lr1e3.pt`
- Rollouts:
  - `/store/store4/data/l2augment_rollout_uvqmlm/train/911Mothers_2010W_0.pt`
  - `/store/store4/data/l2augment_rollout_uvqmlm/train/AJJacobs_2007P_0.pt`
  - `/store/store4/data/l2augment_rollout_uvqmlm/train/AJJacobs_2011P_0.pt`
  - `/store/store4/data/l2augment_rollout_uvqmlm/train/AJJacobs_2014A_0.pt`
  - `/store/store4/data/l2augment_rollout_uvqmlm/train/AalaElKhani_2016X_0.pt`

## Summary

Each row in `rac_mlm_average_masks_5x1000.pdf` is the average of `1000`
sampled RAC-MLM masks for one TED-LIUM audio segment at conditioning reward
`1.0`.

Average masked percentages:

- Sample 1, `911Mothers_2010W`: `55.08%`
- Sample 2, `AJJacobs_2007P`: `49.38%`
- Sample 3, `AJJacobs_2011P`: `44.07%`
- Sample 4, `AJJacobs_2014A`: `40.52%`
- Sample 5, `AalaElKhani_2016X`: `41.73%`

The PDF plots masked probability. The underlying NPZ stores average keep masks.
