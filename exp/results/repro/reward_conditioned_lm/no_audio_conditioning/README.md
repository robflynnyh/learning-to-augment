# ROB-111 Reward-Conditioned Mask LM Plan

## Instruction And Context Check

Read the required instruction files before writing this plan:

1. `symphony/instructions/linear-context.md`
2. `symphony/instructions/repository.md`
3. `symphony/instructions/work-loop.md`
4. `symphony/instructions/experiment-execution.md`
5. `symphony/instructions/validation-and-handoff.md`

Instructions that directly affect this issue:

- Recent Linear comments must be checked before planning. The latest human
  Linear comment said PR review comments were added, so this revision treats
  PR #17 review comments as binding task context.
- If comments request rework on an existing PR or branch, inspect that PR or
  branch before editing.
- The work should stay narrowly scoped and should not launch long GPU jobs
  unless requested. This issue asks for a reviewable plan, not a training run.
- A nontrivial experiment family should have a small repo-local README and
  should record exact assumptions, commands, configs, checkpoints, output paths,
  and validation once implemented.
- Documentation-only completion should be validated with `git diff --check`.
- Completed work should be committed on an issue branch, pushed, opened or
  refreshed as a PR, then handed to Linear in `In Review`.

Review comments that changed or constrained the plan:

- Use the name "reward-conditioned mask LM" rather than the earlier
  reward-conditioned VQ LM wording.
- Keep this plan and future artifacts under
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/`.
- Do not train against an EOS target for the fixed-length inference path.
  Generation should receive the required sequence length derived from the input
  audio and suppress EOS for that many prediction steps.
- Include the required generation length as an explicit model input at
  generation/evaluation time.
- Use the normal rollout training set in place. Do not touch or duplicate the
  large dataset.
- Raw rollout files can be read without torch 2.8 for this task because the
  model uses saved VQ `generation` sequences. If masks are ever needed, cast
  them with `.float()`.
- Build in W&B logging and early stopping on dev loss.
- Use per-utterance reward normalization only. A later PR review comment changed
  the normalization to bounded per-file min-max controls.

## Current Repository Evidence

The closest current implementations are:

- `UnconditionalMaskGenerator` in `l2augment/modelling/models.py` is the UVQLM
  policy. It trains an autoregressive LM over BVAE VQ mask codes, uses the
  BVAE codebook size as a special BOS/EOS token, and decodes generated code
  sequences through the frozen BVAE mask decoder.
- `UnconditionalMaskGenerator.augment()` already computes the required VQ
  generation length from input audio via
  `mask_enc.calc_downsampled_length(lengths)` and passes it to `generate()` as
  `target_prediction_steps`. In that fixed-length path, `generate()` suppresses
  EOS by setting the EOS logit to negative infinity.
- `ConditionalMaskGenerator` is a single-mask reward-conditioned LM, but it is
  also audio-conditioned: it encodes audio with an audio VAE, uses that hidden
  state to initialize the decoder, and uses the reward embedding as the first
  token.
- `ConditionalMultiStepMaskGenerator` supports reward-conditioned generation,
  optional audio conditioning, and sampled conditioning reward ranges, but it is
  designed for sequential multi-mask rollouts and hidden-state carryover. That
  is not the target model for ROB-111.
- `CustomDataset` already reads ROB-109-style rollout files, computes reward as
  `before - after`, keeps the saved `generation` tensor when present, supports
  WER-only reward via `cer_weight: 0.0` and `wer_weight: 1.0`, and can skip
  audio loading with `load_audio: false`.
- `DTLM_fn` already collates saved `generation` sequences and scalar rewards in
  the shape expected by `ConditionalMaskGenerator`, but it still assumes audio
  is present. A no-audio collate path should reuse this logic without the audio
  fields.
- `exp/train_freq_mask.py` already initializes W&B and implements dev-loss
  early stopping via `training.tolerance`.

ROB-109 verified the rollout source for this issue:

- Rollout root: `/store/store4/data/l2augment_rollout_uvqmlm/`
- Dev split: 507 files and 5,070 sampled masks.
- Train split: 265,901 files.
- Each dev file has keys `audio`, `generation`, `mask`, and `reward`.
- Each file has 10 sampled generations.
- `reward[:, metric, stage]` stores before/after CER and WER, with the training
  reward normally computed as `reward[:, metric, 0] - reward[:, metric, 1]`.
- The saved `generation` tensor is the target sequence for this model. Training
  does not need to decode or re-encode `mask`.

## Design Assumptions

1. The target is a no-audio, single-mask reward-conditioned mask LM. It should
   be architecturally closest to UVQLM, not to audio-conditioned
   `ConditionalMaskGenerator` and not to hierarchical
   `ConditionalMultiStepMaskGenerator`.
2. The training target is the saved VQ `generation` sequence from ROB-109, not a
   re-encoded mask and not a text or mask-parameter label.
3. The conditioning scalar should default to WER delta because ROB-109 rewards
   include CER and WER and the repo's evaluation summaries use WER as the main
   criterion.
4. The first implementation should use bounded per-utterance min-max reward
   normalization across the sampled masks for each rollout file. This
   intentionally discards absolute reward scale and frames the model around
   per-utterance candidate preference, which matches the latest review
   direction.
5. The model should condition by replacing UVQLM's fixed BOS embedding with an
   MLP/timestep-encoder output from the normalized reward. This keeps the
   architecture close to UVQLM while making reward control the only new signal.
6. The training objective should not include an EOS position for the main
   fixed-length model. During inference the required generation length is known
   from input audio, so the model should produce exactly that many VQ tokens.
7. A later audio-conditioned variant can be added, but it should be a separate
   experiment because it changes the question from reward-conditioned UVQLM to
   a conditional audio policy.

## Proposed Model

Add a new policy class, tentatively named
`RewardConditionedMaskLM`.

Core architecture:

- Reuse the UVQLM `BinaryVariationalAutoEncoder` setup, VQ codebook size,
  token embedding table, GRU decoder, and prediction head.
- Add `reward_encoder`, either the existing `timestep_encoder(hidden_dim)` or a
  small MLP:

```text
normalized_reward -> Linear/SiLU/Linear or timestep_encoder -> hidden_dim BOS
```

- During teacher-forced training:
  - Input: normalized reward scalar, saved VQ code sequence, and sequence
    length.
  - BOS: `reward_encoder(normalized_reward)`.
  - Remaining inputs: embeddings of previous VQ codes.
  - Target: saved VQ codes only.
  - Loss: masked cross entropy over valid VQ-code positions only; do not append
    or supervise EOS in the fixed-length training path.
- During generation:
  - Input: a requested normalized reward plus the required output audio length.
  - Compute `target_prediction_steps` from the input audio length using the
    frozen BVAE downsampling contract, matching `UnconditionalMaskGenerator`.
  - BOS: encoded reward.
  - Autoregressively sample or greedily decode exactly
    `target_prediction_steps` VQ codes with EOS suppressed.
  - Decode through the frozen BVAE mask decoder, interpolate to
    `target_output_length`, and return
    `(augmented_audio, mask_pred, {"generation": generation, "conditioning_reward": ...})`.

Recommended config knobs:

- `default_conditioning_reward`: normalized scalar used by `augment`.
- `conditioning_reward_range`: optional `[low, high]` range in normalized units
  for random reward sampling at eval time, matching the existing CMultistep
  sweep pattern.
- `reward_encoder`: `timestep` or `mlp`.
- `reward_normalization`: `per_utterance_minmax`.
- `reward_metric`: `wer`, `cer`, or weighted combination.
- `generation_sample`: boolean for sample versus greedy diagnostics.
- `suppress_eos_for_fixed_length`: default `true`; keep an explicit knob only
  if a later ablation wants open-ended generation.

## Data Plan

Use the ROB-109 UVQLM rollout root in place:

```text
/store/store4/data/l2augment_rollout_uvqmlm/{train,dev}
```

Do not duplicate this dataset. It is large, and the normal training set can be
read directly for the fields needed here. Training should load each rollout
file and use:

```text
generation: LongTensor[num_samples, seq_len]
reward_raw: reward[:, 1, 0] - reward[:, 1, 1] for WER delta
```

The raw `mask` tensor can stay untouched because the model trains on
`generation`. If a diagnostic later needs to inspect masks from these rollout
files, cast them to float with `.float()` before use.

Per-utterance min-max normalization:

- For each rollout file, map the 10 sampled rewards within that file to `[0, 1]`.
- Record how all-equal or near-zero-range reward groups are handled. ROB-114
  maps these degenerate groups to the neutral value `0.5` and logs their count.
- At generation time, choose conditioning values in normalized rank-like units
  such as `0`, `0.5`, and `1`, because the absolute WER-delta interpretation is
  intentionally discarded in this mode.

Implementation should avoid a bulky derived dataset. If caching is needed for
speed, cache only small stats/index files that record source paths, sequence
lengths, and reward summaries rather than copying per-rollout tensors.

## Training Infra Plan

Add the smallest set of reusable pieces:

1. A no-audio generation/reward dataset adapter, either as a new
   `RewardConditionedMaskLMDataset` or as narrow options on `CustomDataset`.
   It should:
   - read rollout files directly from `/store/store4/data/l2augment_rollout_uvqmlm/`;
   - return `generation`, normalized `reward`, raw reward, source path, and
     generation length;
   - avoid loading `audio` and `mask` for training.

2. A no-audio collate function, for example `RewardConditionedMaskLM_fn`.
   It should:
   - pad variable-length VQ sequences;
   - concatenate the 10 sampled generations across files;
   - return `generations`, `generation_lengths`, `rewards`, `raw_rewards`, and
     `source_paths`;
   - avoid the current `DTLM_fn` audio assumptions.

3. `RewardConditionedMaskLM` in `l2augment/modelling/models.py`.
   It should:
   - reuse the UVQLM code path;
   - register in `policy_dict`;
   - share helper logic with `UnconditionalMaskGenerator` where practical;
   - expose fixed-length generation based on input audio length.

4. A training config under active configs, for example:

```text
exp/configs/reward_conditioned_lm/no_audio_conditioning/tedlium_per_utterance.yaml
```

5. A result directory for subsequent implementation:

```text
exp/results/repro/reward_conditioned_lm/no_audio_conditioning/
```

The existing `exp/train_freq_mask.py` should be reusable after adding the new
dataset and collate function because it already loads policy, dataset, and
collate implementations from config dictionaries, initializes W&B, and stops
early when dev loss does not improve for `training.tolerance` epochs.

## Evaluation Plan

Initial validation before any long run:

1. Stats/index smoke:
   - Read 2 train files and 2 dev files from the real rollout root.
   - Confirm `generation` and raw WER reward can be loaded without touching or
     duplicating masks.
   - Confirm bounded per-utterance min-max normalization produces finite rewards
     and records degenerate reward-group counts.

2. Model unit smoke:
   - Instantiate the model on CPU with the BVAE config and a tiny synthetic
     batch of VQ codes.
   - Run `forward_pass`.
   - Assert finite CE loss and backward pass.
   - Run fixed-length `generate` at two reward values and confirm exactly the
     requested number of VQ tokens are produced.

3. One-file training smoke:
   - Run `exp/train_freq_mask.py` with `max_steps: 1`, batch size 2, W&B in
     disabled/offline mode if needed, and the real rollout root limited to a
     tiny file list.
   - Confirm checkpoint save path works in a durable non-committed location.
   - Confirm W&B keys include train loss, dev loss, reward-normalization setting,
     and generated sequence diagnostics.

4. One-recording eval smoke:
   - Plug the trained smoke checkpoint into an eval/oracle config.
   - Run one recording only, using Mimas with `with-gpu` if GPU is required.
   - Confirm `augment` computes required generation length from audio, returns
     a mask and generation metadata, and does not attempt open-ended EOS-based
     generation.

Only after the above should a full training job be queued. If a full run is
requested later, follow the project instruction to use a detached Mimas screen
session through `/store/store5/software/simple-gpu-schedule/with-gpu` with an
actual tested Linear callback wrapper.

## ROB-114 Implementation

Implemented repo artifacts:

- Dataset adapter:
  `l2augment.utils.datasets.RewardConditionedMaskLMDataset`
- Collate path:
  `l2augment.utils.collate_functions.RewardConditionedMaskLM_fn`
- Policy class:
  `l2augment.modelling.models.RewardConditionedMaskLM`
- Active full-run config:
  `exp/configs/reward_conditioned_lm/no_audio_conditioning/tedlium_per_utterance.yaml`
- One-file harness-smoke config:
  `exp/configs/reward_conditioned_lm/no_audio_conditioning/tedlium_per_utterance_smoke.yaml`
- Smoke script:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/scripts/smoke_reward_conditioned_mask_lm.py`

The dataset reads the in-place ROB-109 rollout root and returns saved
`generation` tensors, bounded per-utterance min-max WER-delta rewards, raw
WER-delta rewards, source paths, and VQ sequence lengths. It does not return
audio or raw masks for training.

The model keeps the UVQLM BVAE codebook, token embeddings, GRU decoder, and
prediction head. It replaces the fixed BOS token with a reward encoder output.
Teacher-forced training predicts only the saved VQ sequence positions and does
not append or supervise EOS. Fixed-length generation receives
`target_prediction_steps`, suppresses EOS while that explicit length is active,
and returns generation metadata from `augment()`.

Smoke commands:

```bash
bash -ic 'export PYTHONPATH="$PWD:/exp/exp4/acp21rjf/long-context-asr:/exp/exp4/acp21rjf/language_modelling${PYTHONPATH:+:$PYTHONPATH}"; python exp/results/repro/reward_conditioned_lm/no_audio_conditioning/scripts/smoke_reward_conditioned_mask_lm.py'

bash -ic 'export PYTHONPATH="$PWD:$PWD/exp:/exp/exp4/acp21rjf/long-context-asr:/exp/exp4/acp21rjf/language_modelling${PYTHONPATH:+:$PYTHONPATH}"; export WANDB_MODE=disabled; python exp/train_freq_mask.py --config exp/configs/reward_conditioned_lm/no_audio_conditioning/tedlium_per_utterance_smoke.yaml'
```

Rollout files are read with the project-standard `torch.load`; the training path
consumes only saved VQ `generation` and reward fields.

PR-review follow-up removed all local import/load compatibility guards. The
smoke commands above use the bashrc-resolved Python 3.10 / Torch 2.6 runtime.
Torch 2.6 requires trusted local training checkpoints to be loaded with
`weights_only=False`; ROB-109 rollout files still use normal `torch.load`.

## First Real Experiment Proposal

Recommended first full experiment:

- Dataset: in-place ROB-109 UVQLM train/dev rollouts.
- Reward metric: WER delta.
- Reward normalization: bounded per-utterance min-max normalization across each
  rollout file's sampled masks.
- Model: no-audio reward-conditioned mask LM.
- Architecture: UVQLM GRU decoder, hidden dim 256, four layers, BVAE codebook
  2048, reward BOS from `timestep_encoder`.
- Training: start from scratch for the baseline because the BOS mechanism
  changes. Optional follow-up: initialize shared embeddings/decoder/prediction
  from UMLM and randomly initialize reward encoder.
- W&B: log config, train/dev CE loss, normalization setting, reward distribution
  summaries, sequence NLL by reward bins, generated sequence length checks, and
  generated mask samples at representative rewards.
- Early stopping: use dev CE loss with `training.tolerance`; restore the best
  checkpoint before saving final output.
- Diagnostics:
  - Train/dev CE loss.
  - Sequence NLL by raw-reward and normalized-reward bins.
  - Generated mask samples at normalized reward values such as `0`, `0.5`, and
    `1`.
  - Fixed-length generation checks against audio-derived target length.
  - Oracle/eval WER for fixed reward values and sampled reward ranges.

Suggested reward probes:

```text
per-utterance min-max normalized: 0.0, 0.5, 1.0
```

## Risks And Checks

- Per-utterance min-max normalization removes absolute WER-delta scale. Report
  generated behavior as rank-like candidate preference within an utterance, not
  as targeting a globally interpretable raw reward value.
- If the model is trained only on UVQLM samples, high reward samples may be
  rare. Consider balanced sampling by reward bin after the baseline run if
  conditioning is ignored.
- Fixed-length generation removes EOS supervision. Verify the model never
  relies on EOS for stopping in eval/oracle paths, and keep sequence length
  checks in diagnostics.
- If initializing from UMLM, checkpoint loading needs a partial-state load that
  excludes the old fixed BOS behavior and includes the new reward encoder.
- Reward conditioning should be tested by comparing generated sequence
  statistics across requested rewards before spending GPU time on full WER
  evaluation.
- Keep rollout files read-only and do not write bulky caches beside the source
  dataset.

## Review Questions

1. Is `RewardConditionedMaskLM` the preferred class/config name, or should the
   implementation use a shorter existing-family name?
2. For per-utterance min-max normalization, should degenerate 10-sample groups
   keep the current neutral `0.5` value or be skipped?
3. Should the first training run start from scratch or initialize shared UVQLM
   layers from the existing UMLM checkpoint?
4. Should the conditioning reward at eval time be fixed to high target values,
   swept over several fixed values, or sampled from a range?

## ROB-117 Training Launch

ROB-117 queues the active full config introduced in ROB-114:

```bash
exp/configs/reward_conditioned_lm/no_audio_conditioning/tedlium_per_utterance.yaml
```

The full config trains on
`/store/store4/data/l2augment_rollout_uvqmlm/{train,dev}`, keeps W&B logging
enabled through `training.wandb_project: l2augment`, and uses dev-loss early
stopping with `training.tolerance: 5`.

Preflight on 2026-05-21:

```bash
bash -ic 'export PYTHONPATH="$PWD:$PWD/exp:/exp/exp4/acp21rjf/long-context-asr:/exp/exp4/acp21rjf/language_modelling${PYTHONPATH:+:$PYTHONPATH}"; export WANDB_MODE=disabled; python exp/train_freq_mask.py --config exp/configs/reward_conditioned_lm/no_audio_conditioning/tedlium_per_utterance_smoke.yaml'
```

This passed under the bashrc Python 3.10 / Torch 2.6 runtime and logged to:

```text
exp/results/repro/reward_conditioned_lm/no_audio_conditioning/logs/rob117_smoke_20260521.log
```

The long Mimas launch wrapper is:

```bash
scripts/launch_rob117_reward_conditioned_mask_lm_training.sh
```

It keeps logs, W&B files, caches, and scratch under this result directory and
uses the Linear completion callback trap. The actual trap path was smoke-tested
with:

```bash
ROB117_CALLBACK_ONLY=1 ROB117_CALLBACK_CHECK_ONLY=1 CALLBACK_TARGET_STATE=Todo scripts/launch_rob117_reward_conditioned_mask_lm_training.sh
```

The detached queue command is:

```bash
screen -L -Logfile exp/results/repro/reward_conditioned_lm/no_audio_conditioning/logs/rob117_no_audio_reward_conditioned_mask_lm_training.screen.log -dmS rob117-reward-conditioned-mask-lm bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-117 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob117_reward_conditioned_mask_lm_training.sh'
```

Expected checkpoint:

```text
/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance.pt
```

## ROB-117 Training Outcome

The corrected retry completed on 2026-05-22 through detached Mimas `screen`
session `rob117-reward-conditioned-mask-lm-retry` and `with-gpu 1,2`.

Run evidence:

- Branch: `symphony/ROB-117-train-no-audio-reward-conditioned-mask-lm`
- Commit: `80f389d6a89ebc830d07572639657c5abef1cb8d`
- Main log:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/logs/rob117_no_audio_reward_conditioned_mask_lm_training_retry_20260521.log`
- Screen log:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/logs/rob117_no_audio_reward_conditioned_mask_lm_training_retry_20260521.screen.log`
- W&B: project `l2augment`, run `dry-thunder-2166`
  (`https://wandb.ai/wobrob101/l2augment/runs/5ny25k7g`)
- Final checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance.pt`

The training process reached `100/100` epochs and exited with status `0`.
The final checkpoint is 20M (`20842210` bytes). The final logged dev loss
before the metric reset fix was `2.6927917954301535`, but that value was a
cumulative average over validation passes in the training process rather than
the standalone epoch-100 checkpoint loss. Early stopping did not trigger because
the run reached the configured maximum epoch count first.

Post-training sanity:

```bash
bash -ic 'export PYTHONPATH="$PWD:$PWD/exp:/exp/exp4/acp21rjf/long-context-asr:/exp/exp4/acp21rjf/language_modelling${PYTHONPATH:+:$PYTHONPATH}"; python exp/results/repro/reward_conditioned_lm/no_audio_conditioning/scripts/post_training_sanity_check.py'
```

The sanity check loaded the trained checkpoint on CPU and ran fixed-length
generation/augment on
`/store/store4/data/l2augment_rollout_uvqmlm/dev/AlGore_2009_0.pt` at reward
controls `0.0` and `1.0`. Both controls produced exactly 29 VQ tokens for the
1042-frame sample, returned masks of shape `[1, 80, 1042]`, and returned
augmented audio of shape `[1, 80, 1042]`.

Sanity artifact:

```text
exp/results/repro/reward_conditioned_lm/no_audio_conditioning/post_training_sanity_check.json
```

Usability assessment: the checkpoint is loadable and usable for downstream
fixed-length generation/eval/oracle comparison. The immediate caveat is that
the reward-control behavior has only been checked as a load/shape/generation
sanity test here; downstream WER/oracle comparison should still evaluate the
actual augmentation quality.

Follow-up sampled reward-control check after the ROB-117 Linear request:

```bash
bash -ic 'export PYTHONPATH="$PWD:$PWD/exp:/exp/exp4/acp21rjf/long-context-asr:/exp/exp4/acp21rjf/language_modelling${PYTHONPATH:+:$PYTHONPATH}"; python exp/results/repro/reward_conditioned_lm/no_audio_conditioning/scripts/post_training_sanity_check.py --sample --seed 20260522 --rollout /store/store4/data/l2augment_rollout_uvqmlm/dev/AlGore_2009_0.pt --rollout /store/store4/data/l2augment_rollout_uvqmlm/dev/BarrySchwartz_2005G_0.pt --rollout /store/store4/data/l2augment_rollout_uvqmlm/dev/BlaiseAguerayArcas_2007_0.pt --output exp/results/repro/reward_conditioned_lm/no_audio_conditioning/post_training_sampled_reward_0_vs_1_check.json'
```

This sampled run used the same checkpoint and checked three different
TED-LIUM dev recordings. Reward `0.0` versus `1.0` produced valid fixed-length
masks for every recording and different sampled token sequences:

| Recording | Frames | Tokens | Token mismatches | Mask active fraction at reward 0.0 | Mask active fraction at reward 1.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `AlGore_2009_0.pt` | 1042 | 29 | 29/29 | 0.0649 | 0.9410 |
| `BarrySchwartz_2005G_0.pt` | 732 | 19 | 10/19 | 0.0953 | 0.3332 |
| `BlaiseAguerayArcas_2007_0.pt` | 352 | 8 | 7/8 | 0.2389 | 0.3484 |

Artifact:

```text
exp/results/repro/reward_conditioned_lm/no_audio_conditioning/post_training_sampled_reward_0_vs_1_check.json
```

Follow-up adaptation WER check after Robert clarified the request:

```bash
/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash -ic 'export TMPDIR=/exp/exp4/acp21rjf/rob117-scratch/tmp; export L2A_TEDLIUM3_LEGACY_DIR=/store/store4/data/TEDLIUM_release-3/legacy/; export PYTHONPATH="$PWD:$PWD/exp:/exp/exp4/acp21rjf/long-context-asr:/exp/exp4/acp21rjf/language_modelling${PYTHONPATH:+:$PYTHONPATH}"; python exp/results/repro/reward_conditioned_lm/no_audio_conditioning/scripts/post_training_adaptation_wer.py'
```

This ran on Mimas GPU 2 through the scheduler. It uses the trained checkpoint,
TED-LIUM dev recordings matching the sampled diagnostic, `cpu_rollout_policy`,
sampled masks, one adaptation epoch, and `lr=1e-5`.

| Recording | Reward | WER before adaptation | WER after adaptation | Delta |
| --- | ---: | ---: | ---: | ---: |
| `AlGore_2009` | 0.0 | 0.1335 | 0.1241 | -0.0094 |
| `AlGore_2009` | 1.0 | 0.1335 | 0.1224 | -0.0111 |
| `BarrySchwartz_2005G` | 0.0 | 0.0513 | 0.0486 | -0.0027 |
| `BarrySchwartz_2005G` | 1.0 | 0.0513 | 0.0468 | -0.0046 |
| `BlaiseAguerayArcas_2007` | 0.0 | 0.1487 | 0.1373 | -0.0114 |
| `BlaiseAguerayArcas_2007` | 1.0 | 0.1487 | 0.1400 | -0.0087 |

Artifacts:

```text
exp/results/repro/reward_conditioned_lm/no_audio_conditioning/post_training_adaptation_wer_reward_0_vs_1.json
exp/results/repro/reward_conditioned_lm/no_audio_conditioning/logs/rob117_post_training_adaptation_wer_20260522.log
```

## ROB-117 500-Epoch LR 1e-3 Follow-Up Queue

After the 100-epoch run reached the configured epoch limit with dev loss still
decreasing, Robert requested a second full training run with `training.epochs:
500` and `policy.lr: 1e-3`.

Follow-up config:

```text
exp/configs/reward_conditioned_lm/no_audio_conditioning/tedlium_per_utterance_500ep_lr1e3.yaml
```

The config keeps the same in-place ROB-109 rollout data root, batch size,
W&B project, and dev-loss early stopping tolerance as the completed run. It
writes to a distinct checkpoint so the 100-epoch checkpoint remains intact:

```text
/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_500ep_lr1e3.pt
```

Prequeue validation:

```text
exp/results/repro/reward_conditioned_lm/no_audio_conditioning/logs/rob117_smoke_500ep_lr1e3_prequeue_20260522.log
exp/results/repro/reward_conditioned_lm/no_audio_conditioning/logs/rob117_callback_check_500ep_lr1e3_specific_20260522.log
```

The smoke used the documented bashrc Python 3.10 / Torch 2.6 runtime. The
callback check ran the actual follow-up launcher in callback-only/check-only
mode with the follow-up config and checkpoint paths.

Detached launch command:

```bash
screen -L -Logfile /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-117/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/logs/rob117_no_audio_reward_conditioned_mask_lm_training_500ep_lr1e3_20260522.screen.log -dmS rob117-reward-conditioned-mask-lm-500ep-lr1e3 bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-117 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob117_reward_conditioned_mask_lm_training_500ep_lr1e3.sh'
```

Completion check:

```bash
screen -ls | grep rob117-reward-conditioned-mask-lm-500ep-lr1e3 || true
tail -120 exp/results/repro/reward_conditioned_lm/no_audio_conditioning/logs/rob117_no_audio_reward_conditioned_mask_lm_training_500ep_lr1e3_20260522.log
ls -lh /store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_500ep_lr1e3.pt
```

## ROB-117 Resume-100 500-Epoch LR 1e-3 Queue

Robert later clarified that the quicker follow-up should resume from the
completed 100-epoch checkpoint and resume the same W&B run so loss is
continuous. The fresh `500ep_lr1e3` run was stopped before a final checkpoint
was written.

Resume config:

```text
exp/configs/reward_conditioned_lm/no_audio_conditioning/tedlium_per_utterance_resume100_500ep_lr1e3.yaml
```

Key settings:

- Resume source:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance.pt`
- Start epoch: `100`
- Target epoch cap: `500`
- LR: `1e-3`
- W&B: project `l2augment`, run id `5ny25k7g`, `resume: must`
- Output checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_resume100_500ep_lr1e3.pt`

Prequeue validation:

```text
exp/results/repro/reward_conditioned_lm/no_audio_conditioning/logs/rob117_smoke_resume100_500ep_lr1e3_prequeue_20260522.log
exp/results/repro/reward_conditioned_lm/no_audio_conditioning/logs/rob117_callback_check_resume100_500ep_lr1e3_20260522.log
```

The smoke used the documented bashrc Python 3.10 / Torch 2.6 runtime through
`with-gpu 1,2`, loaded the 100-epoch checkpoint, logged from epoch `100`, ran
one tiny train step at LR `1e-3`, and saved smoke checkpoints. The callback
check ran the actual resumed launcher in callback-only/check-only mode.

Detached launch command:

```bash
screen -L -Logfile /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-117/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/logs/rob117_no_audio_reward_conditioned_mask_lm_training_resume100_500ep_lr1e3_20260522.screen.log -dmS rob117-reward-conditioned-mask-lm-resume100-500ep-lr1e3 bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-117 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob117_reward_conditioned_mask_lm_training_resume100_500ep_lr1e3.sh'
```

Completion check:

```bash
screen -ls | grep rob117-reward-conditioned-mask-lm-resume100-500ep-lr1e3 || true
tail -120 exp/results/repro/reward_conditioned_lm/no_audio_conditioning/logs/rob117_no_audio_reward_conditioned_mask_lm_training_resume100_500ep_lr1e3_20260522.log
ls -lh /store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_resume100_500ep_lr1e3.pt
```

## ROB-117 Resume-100 500-Epoch LR 1e-3 Outcome

The resumed follow-up run completed on 2026-05-22 through detached Mimas
`screen` session `rob117-reward-conditioned-mask-lm-resume100-500ep-lr1e3`
and `/store/store5/software/simple-gpu-schedule/with-gpu 1,2`.

Run evidence:

- Branch: `symphony/ROB-117-train-no-audio-reward-conditioned-mask-lm`
- Commit: `84ce9c7c615cc67a0a775590f8f0b4ac529257d5`
- Main log:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/logs/rob117_no_audio_reward_conditioned_mask_lm_training_resume100_500ep_lr1e3_20260522.log`
- Screen log:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/logs/rob117_no_audio_reward_conditioned_mask_lm_training_resume100_500ep_lr1e3_20260522.screen.log`
- W&B: project `l2augment`, resumed run `dry-thunder-2166` /
  `5ny25k7g`
  (`https://wandb.ai/wobrob101/l2augment/runs/5ny25k7g`)
- Final checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_resume100_500ep_lr1e3.pt`

The wrapper exited with status `0`. The run loaded the 100-epoch checkpoint,
started from epoch `100`, and used LR `1e-3`. The first resumed validation pass
reported `2.653739192269065`, which is the best available standalone validation
estimate for the loaded 100-epoch checkpoint. Dev-loss patience triggered after
five non-improving validation passes, so the training loop restored the best
previous state and saved it to the final checkpoint path. The final logged
validation value before rollback was `2.6558073686830923`; before the metric
reset fix this was also cumulative within the resumed process. The old
early-stopping signal was therefore not a strictly correct per-validation
criterion in general, because it compared cumulative running averages. For this
specific resumed run, recovering the per-validation estimates from the
cumulative logs gives approximately `2.653739`, `2.657236`, `2.656021`,
`2.656333`, `2.655912`, and `2.655604`, so the same rollback decision is still
supported by the reconstructed per-validation losses.

Post-training sanity for the resumed checkpoint:

```bash
bash -ic 'export PYTHONPATH="$PWD:$PWD/exp:/exp/exp4/acp21rjf/long-context-asr:/exp/exp4/acp21rjf/language_modelling${PYTHONPATH:+:$PYTHONPATH}"; python exp/results/repro/reward_conditioned_lm/no_audio_conditioning/scripts/post_training_sanity_check.py --config exp/configs/reward_conditioned_lm/no_audio_conditioning/tedlium_per_utterance_resume100_500ep_lr1e3.yaml --checkpoint /store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_resume100_500ep_lr1e3.pt --output exp/results/repro/reward_conditioned_lm/no_audio_conditioning/post_training_sanity_check_resume100_500ep_lr1e3.json'
```

The sanity check loaded the resumed checkpoint on CPU and ran fixed-length
generation/augment on
`/store/store4/data/l2augment_rollout_uvqmlm/dev/AlGore_2009_0.pt` at reward
controls `0.0` and `1.0`. Both controls produced exactly 29 VQ tokens for the
1042-frame sample, returned masks of shape `[1, 80, 1042]`, and returned
augmented audio of shape `[1, 80, 1042]`. Reward `0.0` versus `1.0` produced
29/29 different greedy tokens, with mask active fractions `0.3000` and `1.0000`
respectively.

Artifact:

```text
exp/results/repro/reward_conditioned_lm/no_audio_conditioning/post_training_sanity_check_resume100_500ep_lr1e3.json
```

Usability assessment: the resumed checkpoint is loadable and usable for
downstream fixed-length generation/eval/oracle comparison. The caveat is that
this LR `1e-3` resume did not improve validation loss over the starting
100-epoch state before early stopping, so downstream comparisons should treat
the resumed checkpoint as an additional usable candidate rather than as a
validated improvement.

Metric note: `exp/train_freq_mask.py` now resets `val_losses` at the start of
each validation pass. Older ROB-117 logs before that fix report cumulative
within-process averages, not pure per-epoch dev losses, and older early
stopping decisions should be treated as smoothed/cumulative rather than exact
per-validation patience checks.
