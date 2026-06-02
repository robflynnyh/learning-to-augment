# EGGROLL Plasticity Path

This path trains a label-free forward-only updater for ASR test-time adaptation.
It is separate from the existing augmentation and pseudo-label CTC adaptation
rollouts.

## Training

`exp/train_plasticity_eggroll.py` loads a frozen ASR checkpoint, wraps only the
configured `nn.Linear` target modules with `FastWeightLinear`, and trains a
`PlasticityPolicy` centre updater with EGGROLL rank-1 perturbations.

For each step:

1. sample `B` labelled recordings;
2. segment each recording into `T` chunks;
3. sample `N` rank-1 EGGROLL perturbations of updater matrices;
4. run the ASR over all `B * N` streams per chunk;
5. update the per-recording/per-candidate fast-weight state from activations;
6. decode transcripts after each causal chunk;
7. compute `WER[B, N]` only after rollout;
8. compute `quality = 1.0 - wer.clamp(max=1.0)`;
9. group-normalise quality over candidates for each recording;
10. average rewards over recordings and update the shared updater centre.

The updater receives selected layer activations and final ASR layer outputs only
as generic tensors. It is not passed reference text, WER/CER, pseudo-labels, CTC
loss, entropy, blank ratio, confidence, or other hand-engineered ASR diagnostic
features.

## GPU Run Setup

The queue-safe Mimas launcher is:

```bash
/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- \
  bash scripts/launch_rob186_plasticity_eggroll_gpu.sh
```

Use these smoke modes before queueing a real run:

```bash
ROB186_CALLBACK_ONLY=1 ROB186_CALLBACK_DRY_RUN=1 \
  bash scripts/launch_rob186_plasticity_eggroll_gpu.sh

ROB186_CONFIG_ONLY=1 ROB186_DISABLE_CALLBACK=1 \
  bash scripts/launch_rob186_plasticity_eggroll_gpu.sh

ROB186_SMOKE=1 ROB186_DISABLE_CALLBACK=1 \
  bash scripts/launch_rob186_plasticity_eggroll_gpu.sh
```

The real config enables W&B logging with `training.wandb_enabled: true` and
`training.wandb_mode: online`. Smoke mode overrides W&B to `offline` and runs a
single training step with two antithetic candidates.

The default config targets `layers.0.attend.fn.out_proj`, a standard
`nn.Linear` in the Mimas 2048-context LCASR checkpoint. Feed-forward fused dense
internals such as `layers.0.ff2.fn.fn.fc2` are not suitable MVP targets because
LCASR reads their raw `.weight` tensors inside the fused block.

The launcher exports the Mimas TEDLIUM and Earnings22 dataset roots. The rollout
defaults to `rollout.pass_lengths: false`, matching the existing rollout calls
that run LCASR on padded chunks without an explicit `length`; passing padded
chunk lengths can produce rotary-cache length mismatches in this checkpoint.

## Checkpoints

Plasticity checkpoints are adaptation-only. The payload is marked with:

```text
checkpoint_type: plasticity_updater_only
contains_asr_model_state: false
```

It stores the updater centre state, optional updater optimizer state, config,
step, and last metrics. It does not store the frozen seed ASR model weights.

By default step checkpoints are written under
`exp/results/plasticity_eggroll/checkpoints/`, `latest.pt` is refreshed each
save, and only the last three `updater_step_*.pt` files are kept. Change
`training.keep_last_checkpoints` if a run needs a different retention policy.

## Inference

Inference uses the learned centre updater with `B = 1` and `N = 1`.

There is no EGGROLL population, transcript/reference input, WER/CER reward,
pseudo-label CTC objective, ASR model cloning, or backward pass through the ASR
model. The frozen/shared ASR parameters remain unchanged; only the factorised
fast state evolves over chunks.

## Tensor Layout

The rollout keeps recordings and candidates batched:

```text
chunks: [B, T, C, S]
chunk_t: [B, C, S]
chunk_bn: [B, N, C, S]
chunk_flat inside ASR: [B*N, C, S]
```

Fast weights are held only for selected modules:

```text
A: [B, N, D_out, R_state]
B: [B, N, D_in, R_state]
```

Most ASR modules run normally over the flattened `B * N` batch. Selected
`FastWeightLinear` modules reshape internally, apply their per-stream low-rank
delta, and reshape back.

## Current MVP Limits

- only selected `nn.Linear` modules are adaptable;
- fast-rank capping drops oldest columns;
- fast-weight norm clipping uses exact dense norms for selected modules;
- the implemented decode mode is `causal_chunk`;
- EGGROLL perturbations apply only to 2D updater matrices via `EggrollLinear`.
