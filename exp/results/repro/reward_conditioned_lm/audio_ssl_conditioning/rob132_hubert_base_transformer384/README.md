# ROB-132 Audio SSL-Conditioned Mask LM

This experiment extends the ROB-124 reward-conditioned mask LM from
`p(mask | reward)` to `p(mask | reward, SSL)`.

## Planned Model

- Policy class: `AudioRewardConditionedMaskLM`
- Audio representation: frozen torchaudio `HUBERT_BASE` SSL features
- Decoder: 4-layer causal transformer, `hidden_dim=384`, `num_heads=8`,
  dropout `0.1`, rotary decoder self-attention, and rotary cross-attention
  over projected SSL features
- Mask target: same binary mask VAE codebook sequence used by ROB-124
- Reward normalization: ROB-124 per-utterance WER min-max normalization, with
  degenerate reward groups mapped to `0.5`

## SSL Feature Policy

The SSL model is frozen. Full-rate SSL features are not committed and are not
stored under `/store/store4`. Training computes frozen HuBERT features
on-the-fly on the allocated GPU from the mapped raw TED-LIUM utterance
segments. The sidecar builder is retained only as a verification/debug helper;
it can write mask-token-aligned fp16 sidecars under:

```text
/store/store5/data/acp21rjf_checkpoints/l2augment/ssl_feature_cache/rob132_hubert_base_tedlium_per_utterance/
```

Training keeps only two model files: one final checkpoint and one overwritten
temporary/best-so-far checkpoint.

## Main Configs

- Full training config:
  `exp/configs/reward_conditioned_lm/audio_ssl_conditioning/tedlium_per_utterance_hubert_base_transformer384_dropout0p1_500ep_lr1e3.yaml`
- Smoke config:
  `exp/configs/reward_conditioned_lm/audio_ssl_conditioning/tedlium_per_utterance_hubert_base_transformer384_dropout0p1_smoke.yaml`
- Launch wrapper:
  `scripts/launch_rob132_audio_ssl_mask_lm_training.sh`

## Full Launch Command

```bash
screen -L -Logfile /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-132/exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384/logs/rob132_audio_ssl_hubert_base_transformer384_dropout0p1_500ep_lr1e3.screen.log -dmS rob132-audio-ssl-mask-lm-transformer384 bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-132 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob132_audio_ssl_mask_lm_training.sh'
```

## Initial Caveat

The ROB-124 rollout files contain 80-channel spectrogram tensors, not raw
waveform. `AudioRewardConditionedMaskLMDataset` maps TED-LIUM rollout filenames
back to STM utterance indices, loads the corresponding raw waveform segment
from `/store/store4/data/TEDLIUM_release-3/legacy`, and extracts native-rate
HuBERT features on-the-fly. Optional mask-token-aligned sidecars are supported
only for smoke/debug reuse, not for the main training path.

## Mapping Check

The rollout stem contract was checked on a random 20-file sample: 10 train
rollouts and 10 dev rollouts. Each rollout stem was parsed as
`<recording_id>_<utterance_idx>`, then matched against the STM-derived
segmented TED-LIUM loader. All 20 sampled files matched the saved rollout
spectrogram length exactly.

Artifact:
`exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384/mapping_verification_sample.json`

## Smoke Validation

The callback wrapper smoke was rerun after switching the configs to
`ssl_feature_mode: on_the_fly`. With CUDA hidden, it loaded raw TED-LIUM
utterance segments, extracted frozen HuBERT features in the dataset, completed
one tiny validation/train cycle, and saved the ignored smoke checkpoint.

Generation sanity without a cache also passed at reward controls `0.0` and
`1.0` for one train rollout and one dev rollout.

Artifact:
`exp/results/repro/reward_conditioned_lm/audio_ssl_conditioning/rob132_hubert_base_transformer384/smoke/post_training_generation_sanity_on_the_fly.json`

The requested queued GPU smoke was run through `with-gpu 1,2` on Mimas ticket
`f102175c`; it acquired GPU 1, confirmed CUDA with torchaudio `HUBERT_BASE`,
ran the on-the-fly SSL train/validation path, and saved the smoke checkpoint.
The HuBERT checkpoint is stored outside the repository at:

```text
/exp/exp4/acp21rjf/rob132-audio-ssl-scratch/torch/hub/checkpoints/hubert_fairseq_base_ls960.pth
```

That checkpoint is 361M. The ignored smoke model checkpoint is 54M.

## 2026-05-25 Position And SSL Device Correction

A first full training run was interrupted after a Linear follow-up pointed out
two issues: cross-attention did not include an explicit query/KV positional
signal, and HuBERT extraction was configured on CPU. The interrupted run should
be treated only as a content-only cross-attention baseline attempt.

The corrected implementation applies RoPE to both decoder self-attention and
mask-to-SSL cross-attention Q/K tensors. The main path now keeps native HuBERT
frame sequences instead of resizing them to the mask-token length. For
cross-attention RoPE, native SSL key positions are scaled onto the mask-token
time grid per example using `generation_lengths` and `audio_feature_lengths`.
The configs now set `ssl_device: cuda`; DataLoader workers are disabled for
this path so CUDA HuBERT extraction happens in the main training process
instead of inside forked workers.
