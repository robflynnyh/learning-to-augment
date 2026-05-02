# Learn-to-Augment Stanage Checkpoints — High-Level Summary

This is a readable summary of the checkpoint inventory under:

- `/mnt/parscratch/users/acp21rjf/l2augment_model/`
- `/mnt/parscratch/users/acp21rjf/l2augment_value/`

It is based on embedded checkpoint configs plus exact repo config path matches where available. Anything not recoverable from those sources is marked **unknown** rather than guessed.

A more exhaustive machine-style manifest exists at `CHECKPOINT_MANIFEST_STANAGE.md`.

## Big picture / experimentation story

The checkpoint set looks like an exploration of learned augmentation policies for ASR adaptation. The basic progression was:

1. **Start with UFMR** — an unconditional policy that generates/ranks masks without conditioning on the audio, previous masks, or any other context. Despite being simple, this worked quite well.
2. **Try conditional variants** — after UFMR, the experiments branch into policies conditioned on:
   - the **audio** being augmented,
   - **previously generated masks** / mask history,
   - or **both audio and previous masks**.
3. **Support those conditional policies with latent models** — audio VAEs and binary/mask VAEs encode audio and masks so more structured/generative policies can operate in latent/codebook space.

Most policies use a frozen ASR checkpoint as the reward/evaluation backbone:

`/mnt/parscratch/users/acp21rjf/spotify/rotary_pos_6l_256d_seq_sched/n_seq_sched_2048_rp_1/step_105360.pt`

That backbone appears to be a Conformer ASR model: 6 layers, d_model 256, 8 heads, rotary embeddings, self-conditioning, chunk size 2048, trained with MADGRAD lr 0.003. One embedded name says `rotary_pos_3l_256d...`, so exact ASR naming provenance is inconsistent/unknown.

The learned augmentation policies fall into four broad groups:

1. **Unconditional mask rankers/generators** — especially UFMR, which samples SpecAugment-style frequency masks and chooses/ranks them without conditioning.
2. **Audio/mask representation models** — VAE / BVAE checkpoints that compress audio or masks into latent/codebook representations for conditional policies.
3. **Conditional mask generators / language models** — generate masks conditioned on audio, previous mask tokens, reward signals, or combinations of these.
4. **Value / decision-style models** — experimental policies/value models for multi-step or reward-conditioned augmentation.

## What is probably most important

### UFMR checkpoints — the simple unconditional baseline that worked well

Directory: `l2augment_model/ufm/*`

These are **Unconditional Frequency Mask Rankers**. They do not look at the audio and they do not condition on previously generated masks. Instead, they sample random frequency masks, score each mask using a tiny network over the 80-bin mask vector, and apply the highest-scoring mask.

This is the first/simple line of experimentation, and according to rob it worked quite well despite having no conditioning signal. The later checkpoint families should be read as attempts to improve on / generalize this by adding conditioning on audio, mask history, or both.

Training setup, from embedded configs and `exp/configs/UFMR_README.md`:

- Model class: `UnconditionalFrequencyMaskingRanker`
- Input: mask only, 80 mel-bin frequency mask
- Mask source: SpecAugment-like frequency masks, `n_freq_masks=6`, `freq_mask_param=34`, time masking off
- Reward data: offline rollout `.pt` files containing candidate masks and WER/CER before/after scores
- Objective: usually MSE between predicted mask score and standardized reward; `multloss` uses a multiplicative objective
- Optimizer: MADGRAD in training code
- Common hyperparameters: lr `1e-4`, batch size `84`, epochs `100`, seed `1234`
- Frozen ASR backbone: `spotify/.../n_seq_sched_2048_rp_1/step_105360.pt`

Checkpoints found:

- `ufm/mseloss/model.pt`, `tmp_model.pt` — UFM ranker, MSE reward objective. Exact distinction from `mseloss2e1` is unknown from embedded config.
- `ufm/mseloss2e1/model.pt`, `tmp_model.pt` — UFM ranker, MSE reward objective. Name suggests a second run/variant, but exact reason is unknown.
- `ufm/multloss/model.pt`, `tmp_model.pt` — UFM ranker, `loss_type=mult`.
- `ufm/test_cer/model.pt`, `tmp_model.pt` — despite the name, the README says this may still be WER-only due to default reward weighting; directory name is misleading.
- `ufm/test_wer/model.pt`, `tmp_model.pt` — WER-only UFM ranker; matching eval configs exist under `configs_in_paper/UFRM/UFRM_eval/...`.
- `ufm/test_wer_cer/model.pt`, `tmp_model.pt` — likely joint WER/CER reward variant; config path `exp/configs/UFMR_wer_cer.yaml` matched.
- `ufm/test/tmp_model.pt` — UFM test run, lr `1e-5`; no final `model.pt` found in current inventory.

These UFMR files are the only Learn-to-Augment policy checkpoints already copied locally under `/store/store5/data/acp21rjf_checkpoints/l2augment/ufmr/`.

## Support models: audio and mask VAEs

These look like representation-learning components used by later/generative policies.

### `autoenc_audio/` — audio VAE without vector quantization

Model class: `VQVAE`, but configs say `use_vq=False`, so functionally this is an audio VAE-style encoder/decoder.

Purpose: compress 80-bin mel audio into a latent representation, likely used as audio conditioning for mask generators/rankers.

Checkpoints:

- `autoenc_audio/model_cpu.pt`, `tmp_model_cpu.pt`
  - latent dim 256
  - batch size 48
  - lr `6e-4`
  - epochs 200
  - max steps 10000
  - matched config: `configs_in_paper/audio_vae_train/VAE_cpu.yaml`
- `autoenc_audio/model_gpu.pt`, `tmp_model_gpu.pt`
  - latent dim 256
  - batch size 256
  - lr `1e-3`
  - epochs 200
  - matched config: `configs_in_paper/audio_vae_train/VAE_gpu.yaml`

The names `cpu`/`gpu` appear to describe training/evaluation variant or intended launch target, not the saved tensor device; all configs say training device `cuda`.

### `bvae/` — binary / vector-quantized mask VAE family

Model class: `BVAE`.

Purpose: learn a discrete/codebook representation of binary masks. These are used by the mask language models/generators via `mask_vae_config`.

Likely most important checkpoint:

- `bvae/bvae_USINGTHISFORNOW_2048gpu.pt`
  - latent dim 128
  - codebook size 2048
  - lr `1e-3`
  - batch size 40
  - epochs 200
  - this filename strongly suggests it was the preferred mask BVAE at some point.

Other variants:

- `tmp_model2048gpu.pt` — same 2048-codebook GPU-style setup as above.
- `tmp_model2048cpu.pt` — 2048-codebook but batch size 4, lr `6e-4`, max steps 10000.
- `model3.pt`, `tmp_model3.pt` — latent dim 128, codebook size 64, batch size 4, epochs 200; matched `DT_train/DT.yaml` path references.
- `model.pt`, `tmp_model.pt` — latent dim 32, batch size 4, epochs 100, max steps 10000.
- `tmp_model2.pt` — latent dim 64, batch size 4, epochs 100, max steps 10000.

Some configs reference `model2048cpu.pt`/`model2048gpu.pt`, but only temporary/renamed copies were found in the inventory. Exact final-vs-temp provenance is unknown.

### `vae/`, `vae_tmp/`, `ssvae/`

These contain older or alternate audio VAE-style checkpoints.

- `vae/model.pt`, `vae/tmp_model.pt`, `vae_tmp/model.pt`
  - embedded policy class unknown
  - batch size 84
  - epochs 1000
  - likely generic audio VAE checkpoints used by conditional policies
  - matched configs include `exp/configs/vae.yaml`, `CM_train/CM.yaml`, and `DT_train/DT.yaml`
- `ssvae/model.pt`, `ssvae/tmp_model.pt`
  - embedded policy class unknown
  - batch size 84
  - epochs 1000
  - likely single-state VAE variant used by conditional frequency-mask rankers
  - exact role/provenance unknown

## Conditional and unconditional mask generators

These are more complex policies that generate masks rather than simply ranking random frequency masks. This is where the experimentation moves beyond UFMR: instead of choosing masks unconditionally, these models try to exploit structure from the current audio, from previous masks in a sequence, or from both.

### `UMLM/` — unconditional mask language model

Model class: `UnconditionalMaskGenerator`.

Purpose: generate masks from a learned mask-codebook representation without audio conditioning.

Checkpoints:

- `UMLM/modelgpu.pt`, `tmp_modelgpu.pt`
  - mask VAE config: latent dim 128, codebook size 2048, VQ enabled
  - lr `6e-4`
  - batch size 40
  - epochs 200
  - matched config: `configs_in_paper/unconditional_mask_lm/UMLM.yaml`
- `UMLM/tmp_modelcpu.pt`
  - same core setup but batch size 4
  - no exact train config hit found

`UMLM/modelcpu.pt` is referenced by configs but was not found in the current inventory.

### `MLM/` — conditional single-step-ish mask language model

Model class: `ConditionalMaskGenerator`.

Purpose: generate a mask-code sequence conditioned on audio latent features and a target/default reward. It uses an audio VAE and a mask BVAE/codebook.

Checkpoints:

- `MLM/modelcpu.pt`, `tmp_modelcpu.pt`
  - codebook size 2048
  - hidden size 256
  - default conditioning reward 1.0
  - audio VAE: latent dim 256, non-VQ, batchnorm
  - mask VAE: latent dim 128, codebook size 2048, VQ enabled
  - lr `6e-4`
  - batch size 4
  - epochs 10
  - matched config: `configs_in_paper/conditional_mask_lm/MLM.yaml`

### `CMultiStepMLM/` — conditional multi-step mask language model family

Model class: `ConditionalMultiStepMaskGenerator`.

Purpose: generate a sequence of masks over multiple steps. Some variants are audio-conditioned, some explicitly disable audio conditioning. The model uses learned embeddings for reward/signal conditioning plus a GRU decoder over mask-codebook tokens.

Core audio-conditioned variant:

- `CMultiStepMLM/modelcpu.pt`, `tmp_modelcpu.pt`, `curbest.pt`
  - codebook size 2048
  - hidden dim 512
  - embedding dim 256
  - audio VAE: latent dim 256, non-VQ
  - mask VAE: latent dim 128, codebook size 2048, VQ enabled
  - lr `1e-3`
  - batch size 1
  - epochs 10
  - max steps 10000
  - matched configs: `conditional_multistep_mask_lm/MLM.yaml`, `MLM_continue.yaml`, and eval configs

No-audio variants:

- `no_audio_modelcpu.pt`, `no_audio_tmp_modelcpu.pt`
  - condition_on_audio=False
  - hidden dim 512, embedding dim 512
  - epochs 10
  - no exact train config hit for this exact path
- `no_audio_modelcpu2.pt`, `no_audio_tmp_modelcpu2.pt`
  - condition_on_audio=False
  - epochs field says 10000 and max_steps 10000
  - matched config: `MLM_noaudio.yaml`
- `no_audio_modelgpu_big.pt`, `no_audio_tmp_modelgpu_big.pt`
  - condition_on_audio=False
  - decoder_layers=8
  - epochs 100
  - matched config: `MLM_noaudio_big.yaml`
- `no_audio_modelsignals.pt`, `no_audio_tmp_modelsignals.pt`
  - condition_on_audio=False
  - includes more loader parallelism: workers 8, prefetch 4
  - matched config: `MLM_noaudio_signals.yaml`

Interpretation: this directory seems to contain the main family of multi-step mask-generation experiments, exploring whether audio conditioning helps and whether a larger/no-audio decoder can model useful augmentation sequences.

## Conditional rankers and generative policies

### `cfm/` and `cfmsimple/` — conditional frequency-mask rankers

Model class: `ConditionalFrequencyMaskingRanker`.

Purpose: like UFMR, but audio-conditioned. It encodes audio with a VAE, encodes candidate frequency masks, concatenates audio+mask representations, and predicts mask reward.

Checkpoints:

- `cfm/tmp_model.pt`
  - loss MSE
  - latent dim 256
  - mel bins 80
  - VAE: input 80, hidden 128, latent 256, 6 layers, min input size 256
  - lr `1e-4`
  - batch size 84
  - epochs 3
  - matched config: `exp/configs/conditional_freq_mask.yaml`
- `cfmsimple/tmp_model.pt`
  - similar, but lr `3e-4`, epochs 100, no min-input-size field in embedded VAE config
  - no exact train config hit found

No final `model.pt` for either was found in the current inventory, though configs reference them.

### `cm/test/` — conditional masking policy

Model class: `ConditionalMaskingPolicy`.

Purpose: audio-conditioned generative masking policy. It passes audio through a VAE and predicts a Bernoulli mask over the mel/time grid, trained with a PPO-style clipped objective from rollout rewards/probabilities.

Checkpoints:

- `cm/test/model.pt`, `tmp_model.pt`
  - hidden dim 256
  - mel bins 80
  - latent dim 16
  - output dim 1
  - VAE: input 80, hidden 256, latent 16, 5 layers
  - lr `2e-5`
  - batch size 42
  - epochs 1
  - matched config: `configs_in_paper/CM_train/CM.yaml`

This looks like an early/small conditional masking run rather than a mature long training run.

### `dt/test/` — decision-transformer-ish conditional masking policy

Model class: `DTConditionalMaskingPolicy`.

Purpose: generate mask-code sequences conditioned on audio and a desired/default reward. It uses an audio VAE and a mask BVAE/codebook, then decodes a sequence of mask tokens.

Checkpoints:

- `dt/test/best_model.pt`, `tmp_model.pt`
  - default conditioning reward 1.0
  - hidden dim 256
  - mel bins 80
  - output dim 80
  - audio VAE: input 80, hidden 256, latent 16, 5 layers
  - mask VAE: latent dim 128, codebook size 64
  - lr `1e-3`
  - batch size 4
  - epochs 10
  - max steps 2500
  - matched config: `configs_in_paper/DT_train/DT.yaml`

The `best_model.pt` name suggests this is the preferred checkpoint in that subfamily.

## Multi-step frequency-mask ranker family

Directory: `multistep_FM_ranker/`

Model class: `MultiStepMaskRanker`.

Purpose: repeatedly choose frequency masks in a stateful/multi-step setting. It scores masks using a GRU state so later choices can depend on earlier selected masks. Most configs found have `condition_on_audio=False`, so these are generally mask-history-conditioned rather than audio-conditioned.

Checkpoints:

- `modelcpu.pt`, `tmp_modelcpu.pt`
  - hidden dim 512
  - condition_on_audio=False
  - lr `1e-3`
  - batch size 32
  - epochs 100
  - matched config: `configs_in_paper/multistep_FM_ranker/MLM3.yaml`
- `modelcpul2.pt`, `tmp_modelcpul2.pt`
  - hidden dim 1024
  - GRU layers 2
  - condition_on_audio=False
  - lr `1e-3`
  - batch size 32
  - epochs 100
  - matched config: `configs_in_paper/multistep_FM_ranker/MLM.yaml`
- `tmp_modelcpuaudio.pt`
  - name says audio, but embedded config still says `condition_on_audio=False`; it includes an `audio_vae_config`, but does not enable audio conditioning
  - matched config: `configs_in_paper/multistep_FM_ranker/MLM2.yaml`
- `curbest.pt`, `modelcpu2.pt`, `tmp_modelcpu2.pt`
  - lr `3e-3`
  - batch size 4
  - epochs 100
  - policy config absent in checkpoint, so exact architecture defaults are inferred from class defaults

Likely takeaway: this was a family of experiments around sequential mask choice, including hidden-size/layer-size variations and possibly an attempted audio-conditioned variant whose embedded config does not actually enable audio conditioning.

## Value model

### `l2augment_value/model.pt`

This is separate from `l2augment_model/`.

Embedded config has no policy class, but state-dict prefixes include value-model-ish modules:

- `data_mask_combine`
- `data_with_state_ds`
- `encode`
- `gru`
- `init_state`
- `reward_prediction`
- `sequence_to_state`

Training setup:

- lr `0.002`
- batch size 24
- epochs 25
- seed 1234
- ASR backbone: `spotify/.../n_seq_sched_4096_rp_1/step_105360.pt`, i.e. a 4096 sequence-length variant rather than the common 2048 backbone.

High-level role: probably a learned reward/value predictor for stateful augmentation sequences. Exact training data and intended downstream usage are unknown from embedded config.

## Things I would treat as likely canonical

Based on names, matched configs, and transfer status:

1. **ASR backbone**
   - `/store/store5/data/acp21rjf_checkpoints/l2augment/asr/step_105360.pt`
2. **UFMR policies**
   - especially `ufm/test_wer/model.pt`, `ufm/test_cer/model.pt`, and `ufm/test_wer_cer/model.pt`
   - copied locally and documented by `UFMR_README.md`
3. **Mask BVAE preferred checkpoint**
   - `bvae/bvae_USINGTHISFORNOW_2048gpu.pt`
4. **Audio VAE GPU checkpoint**
   - `autoenc_audio/model_gpu.pt`
5. **Conditional multi-step generator main checkpoint**
   - `CMultiStepMLM/modelcpu.pt` or `CMultiStepMLM/curbest.pt`
6. **No-audio large multi-step generator**
   - `CMultiStepMLM/no_audio_modelgpu_big.pt`
7. **Decision-transformer-style best model**
   - `dt/test/best_model.pt`

The others are probably ablations, temporary checkpoints, incomplete runs, or alternate baselines unless you remember otherwise.

## Open questions / weak provenance

These are the questions I would ask if we wanted to make this publication-grade, but the summary above does not depend on you remembering them:

1. For UFMR, were `test_cer` and `test_wer` intentionally separate reward objectives, or did `test_cer` become a misleading directory name as the README suggests?
2. Is `bvae_USINGTHISFORNOW_2048gpu.pt` definitely the canonical mask BVAE for the paper/evals?
3. For `CMultiStepMLM`, should `curbest.pt` be treated as the selected checkpoint over `modelcpu.pt`, or is it only an eval artifact?
4. Are the `no_audio_*` CMultiStepMLM models intended as ablations for “mask prior only” / no audio conditioning?
5. Was `l2augment_value/model.pt` actually used downstream, or just an abandoned value-prediction experiment?
6. Should temporary checkpoints `tmp_model*.pt` be preserved in the release manifest, or only final/best checkpoints?

## Short human-readable inventory

- **UFMR**: simple, robust frequency-mask rankers; likely the main checkpoints already copied and easiest to reuse.
- **Audio VAE**: compresses mel audio for conditioning; `model_gpu.pt` is probably the stronger/high-throughput version.
- **BVAE / mask VAE**: compresses binary masks into discrete codes; `bvae_USINGTHISFORNOW_2048gpu.pt` looks canonical by name.
- **UMLM**: unconditional mask generator using the mask codebook.
- **MLM**: audio-conditioned mask generator using audio VAE + mask BVAE.
- **CMultiStepMLM**: multi-step mask generator; contains main audio-conditioned and no-audio ablation families.
- **CFM**: audio-conditioned frequency-mask ranker; likely less central/incomplete because only tmp checkpoints are present.
- **CM**: direct conditional mask generator trained briefly with PPO-style objective.
- **DT**: decision-transformer-style mask generator; `best_model.pt` is probably the selected checkpoint.
- **Multi-step FM ranker**: sequential ranker for repeated frequency-mask choices; several ablations by hidden size/layers/audio-conditioning flags.
- **Value model**: separate reward/value predictor trained against a 4096-sequence ASR backbone; role unknown.
