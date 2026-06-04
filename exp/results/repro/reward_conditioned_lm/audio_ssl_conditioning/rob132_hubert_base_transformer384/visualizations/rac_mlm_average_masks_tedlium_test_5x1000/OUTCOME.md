# Outcome

Generated the requested RAC-MLM average-mask PDF from the ROB-132 HuBERT-base
audio SSL checkpoint using TED-LIUM test audio.

## Inputs

- Config:
  `exp/configs/reward_conditioned_lm/audio_ssl_conditioning/tedlium_per_utterance_hubert_base_transformer384_dropout0p1_500ep_lr1e3.yaml`
- Checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/audio_ssl_hubert_base_tedlium_per_utterance_transformer384_dropout0p1_500ep_lr1e3.pt`
- Source split:
  `TEDLIUM_release-3/legacy/test`
- Source segments:
  - `AimeeMullins_2009P`, utterance 0, `17.82-28.81`
  - `BillGates_2010`, utterance 0, `15.861-19.986`
  - `DanBarber_2010`, utterance 0, `16.09-19.402`
  - `DanielKahneman_2010`, utterance 0, `15.8-18.33`
  - `EricMead_2009P`, utterance 0, `18.434-31.66`

## Summary

Each row in `rac_mlm_average_masks_tedlium_test_5x1000.pdf` is the average of
`1000` sampled RAC-MLM masks for one TED-LIUM test audio segment at
conditioning reward `1.0`.

Average masked percentages:

- Sample 1, `AimeeMullins_2009P`: `34.67%`
- Sample 2, `BillGates_2010`: `44.23%`
- Sample 3, `DanBarber_2010`: `43.61%`
- Sample 4, `DanielKahneman_2010`: `46.59%`
- Sample 5, `EricMead_2009P`: `37.94%`

The PDF plots masked probability only. The masks are not overlaid on, or
applied to, the audio. The underlying NPZ stores average keep masks.
