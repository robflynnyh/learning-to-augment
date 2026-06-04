# Outcome

Generated the ROB-196 follow-up RAC-MLM visualizations from the ROB-132
HuBERT-base audio SSL checkpoint using TED-LIUM test audio.

## Inputs

- Config:
  `exp/configs/reward_conditioned_lm/audio_ssl_conditioning/tedlium_per_utterance_hubert_base_transformer384_dropout0p1_500ep_lr1e3.yaml`
- Checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/audio_ssl_hubert_base_tedlium_per_utterance_transformer384_dropout0p1_500ep_lr1e3.pt`
- Source split:
  `TEDLIUM_release-3/legacy/test`
- Source segments:
  - Sample 3, `DanBarber_2010`, utterance 0, `16.09-19.402`
  - Sample 4, `DanielKahneman_2010`, utterance 0, `15.8-18.33`
  - Sample 5, `EricMead_2009P`, utterance 0, `18.434-31.66`

## Summary

The averaged figure has three rows and two columns. Rows are TED-LIUM test
samples 3-5; columns are conditioning rewards `1.0` and `0.0`. Each averaged
panel streams `1000` sampled RAC-MLM masks without storing every generated
mask.

Average masked percentages:

- Sample 3, `DanBarber_2010`, reward `1.0`: `43.61%`
- Sample 3, `DanBarber_2010`, reward `0.0`: `58.91%`
- Sample 4, `DanielKahneman_2010`, reward `1.0`: `46.59%`
- Sample 4, `DanielKahneman_2010`, reward `0.0`: `62.68%`
- Sample 5, `EricMead_2009P`, reward `1.0`: `37.94%`
- Sample 5, `EricMead_2009P`, reward `0.0`: `68.86%`

The single-generation figure uses the same audio and reward panels, taking the
first generated mask from each panel's streamed sampling pass. The PDFs plot
masked probability only; masks are not overlaid on, or applied to, the audio.

The follow-up render removes the colorbar from the single-generation mask
figure, removes the averaged-mask figure suptitle, and uses panel titles of the
form `{recording_id} - reward {value}`. It also adds a separate spectrogram-only
PDF/PNG for the same three TED-LIUM test audio segments.
