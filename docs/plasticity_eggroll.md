# EGGROLL Plasticity Path

This path trains a label-free forward-only updater for ASR test-time adaptation.
It is separate from the existing augmentation and pseudo-label CTC adaptation
rollouts.

## Training

`exp/train_plasticity_eggroll.py` loads a frozen ASR checkpoint, wraps only the
configured `nn.Linear` target modules with `FastWeightLinear`, and trains a
`PlasticityPolicy` centre updater with EGGROLL rank-1 perturbations. The main
ROB-186 setup observes and adapts every attention `out_proj` layer selected in
`plasticity.target_modules`; the default Mimas checkpoint exposes
`layers.0.attend.fn.out_proj` through `layers.5.attend.fn.out_proj`.

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
/store/store5/software/simple-gpu-schedule/with-gpu 1,2 --num 2 -- \
  bash scripts/launch_rob186_plasticity_eggroll_gpu.sh
```

The current default uses two allocated GPUs with logical `cuda:0` and `cuda:1`.
The optimizer and EGGROLL centre update remain on `training.device`; rollout
workers shard the candidate axis across the configured devices and explicitly
bind each worker thread to its CUDA device before running LCASR.

For a single-device debug run, override `rollout.devices` and request one GPU:

```bash
ROB186_EXTRA_OVERRIDES='rollout.devices=[cuda:0]' \
  /store/store5/software/simple-gpu-schedule/with-gpu 1,2 --num 1 -- \
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

The default real-run config targets all six standard `nn.Linear` attention
`out_proj` modules in the Mimas 2048-context LCASR checkpoint. The training
script also supports `plasticity.target_modules: auto_attention_out_proj` to
discover names from `asr_model.named_modules()` at startup. A single target such
as `layers.0.attend.fn.out_proj` is only a smoke/debug setting; it does not
exercise cross-layer communication. Feed-forward fused dense internals such as
`layers.0.ff2.fn.fn.fc2` are not suitable MVP targets because LCASR reads their
raw `.weight` tensors inside the fused block.

The default dataset is unsegmented TED-LIUM via `training.dataset: tedlium`.
Do not use `tedlium3_segmented_data` for ROB-186 training: that loader returns
pre-segmented utterances, while this method must adapt over chunks cut from full
long-form recordings. The training script rejects that known segmented loader
unless `training.allow_segmented_dataset_for_debug: true` is set explicitly for
debug tests.

The launcher exports the Mimas TEDLIUM and Earnings22 dataset roots. The rollout
defaults to `rollout.pass_lengths: false`, matching the existing rollout calls
that run LCASR on padded chunks without an explicit `length`; passing padded
chunk lengths can produce rotary-cache length mismatches in this checkpoint.

Each training-step log includes `chunks_per_recording_mean`,
`chunks_per_recording_min`, `chunks_per_recording_max`,
`chunk_length_frames_mean`, `rollout_chunk_steps`, and
`plasticity_num_modules` so long-form recordings can be checked for the expected
number of causal update chunks and adapted modules.

Fast weights now accumulate as dense per-recording/per-candidate matrices:

```text
F: [B, N, D_out, D_in]
```

The updater still emits a low-rank chunk update with `P`, `Q`, `eta`, and `rho`,
but the state update materialises the accumulated dense matrix:

```text
F <- rho * F + eta * P Q^T
```

This removes the former `max_fast_rank` history truncation. The dense state is
still norm-capped against the frozen base weight norm with
`plasticity.max_fast_norm_ratio`, which is set to `5.0e-2` for the default run.
Step logs include `fast_state_norm_ratio_mean`,
`fast_state_norm_ratio_max`, `fast_weight_clipped_fraction`,
`fast_update_norm_ratio_mean`, and `fast_update_norm_ratio_max` to make the
looser cap auditable.

The default real run uses `rollout.batch_size_recordings: 8` with 32 EGGROLL
candidates, giving `rollout_streams = 256` split into 128 streams per logical
CUDA device. The training dtype is `bfloat16`, and the startup event logs the
resolved dtype so W&B and local logs can verify the run did not fall back to
`float32`.

The default config also sets:

```yaml
rollout:
  devices:
    - cuda:0
    - cuda:1
training:
  dtype: bfloat16
```

This is not distributed training. Each shard returns WERs for its candidate
subset, then the trainer recombines `WER[B, N]` and runs candidate
group-normalisation over the full `N` axis on the primary device. Step logs
include `rollout_num_devices` and per-device stream counts.

`reward_std` is the standard deviation of `reward_per_candidate`. With
`B = 1`, group-normalising over candidates makes this metric close to 1 when
candidate qualities differ and 0 when all candidates tie. The script also logs
`quality_std_over_candidates_mean`, `quality_std_over_candidates_max`,
`reward_group_std_mean`, and `reward_active_recording_fraction` so ties and
pre-normalisation spread are visible.

## EGGROLL Paper Notes

Sarkar et al. (2026) define EGGROLL as low-rank ES over matrix parameters:
sample `A` and `B`, form `E = A B^T / sqrt(r)`, evaluate `M + sigma E`, and
average reward-weighted perturbations across workers. The ROB-186 MVP uses
rank 1, so the `1 / sqrt(r)` factor is numerically 1, and its implicit
`EggrollLinear` forward path matches the paper's efficient decomposition:
shared `x M^T` plus a cheap candidate-specific low-rank term.

The paper's Algorithm 1 absorbs the ES `1 / sigma` factor into the learning
rate. This implementation follows the ROB-186 issue formula and computes
`delta / sigma` before passing it to a PyTorch optimizer. To make that scale
auditable, logs include `eggroll_lr`, `eggroll_sigma`, and
`eggroll_lr_over_sigma`. The default `lr = 1e-4` and `sigma = 1e-3` therefore
correspond to an Algorithm-1-style scale of `0.1`, close to the `0.125` scale
reported for the paper's RWKV reasoning experiment, while staying conservative
for this new ASR updater path. The default optimizer is `adamw` with zero weight
decay, matching the paper's small-policy ES optimizer choice without adding
regularisation to the updater centre.

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
model. The frozen/shared ASR parameters remain unchanged; only the learned
fast-weight update process evolves a dense fast state over chunks.

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
F: [B, N, D_out, D_in]
```

Most ASR modules run normally over the flattened `B * N` batch. Selected
`FastWeightLinear` modules reshape internally, apply their per-stream dense
fast-state delta, and reshape back. Each chunk update emitted by the updater is
low-rank, but the accumulated state can become full-rank over the recording.

## Current MVP Limits

- only selected `nn.Linear` modules are adaptable;
- fast-weight norm clipping uses exact dense norms for selected modules;
- the implemented decode mode is `causal_chunk`;
- EGGROLL perturbations apply only to 2D updater matrices via `EggrollLinear`.
