# ROB-111 Reward-Conditioned VQ LM Plan

## Instruction And Context Check

Read the required instruction files before writing this plan:

1. `symphony/instructions/linear-context.md`
2. `symphony/instructions/repository.md`
3. `symphony/instructions/work-loop.md`
4. `symphony/instructions/experiment-execution.md`
5. `symphony/instructions/validation-and-handoff.md`

Instructions that directly affect this issue:

- Recent Linear comments must be checked before planning. ROB-111 had no recent
  comments, so no thread comment changed or constrained the issue beyond the
  description.
- The work should stay narrowly scoped and should not launch long GPU jobs
  unless requested. This issue asks for a reviewable plan, not a training run.
- A nontrivial experiment family should have a small repo-local README and
  should record exact assumptions, commands, configs, checkpoints, output paths,
  and validation once implemented.
- Documentation-only completion should be validated with `git diff --check`.
- Completed work should be committed on an issue branch, pushed, opened as a
  PR, then handed to Linear in `In Review`.

## Current Repository Evidence

The closest current implementations are:

- `UnconditionalMaskGenerator` in `l2augment/modelling/models.py` is the UVQLM
  policy. It trains an autoregressive LM over BVAE VQ mask codes, using an EOS
  embedding as the BOS token, and decodes generated code sequences through the
  frozen BVAE mask decoder.
- `ConditionalMaskGenerator` is a single-mask reward-conditioned LM, but it is
  also audio-conditioned: it encodes audio with an audio VAE, uses that hidden
  state to initialize the decoder, and uses the reward embedding as the first
  token.
- `ConditionalMultiStepMaskGenerator` already supports reward-conditioned
  generation, optional audio conditioning, and sampled conditioning reward
  ranges, but it is designed for sequential multi-mask rollouts and hidden-state
  carryover. That is not the target model for ROB-111.
- `CustomDataset` already reads ROB-109-style rollout files, computes reward as
  `before - after`, keeps the saved `generation` tensor when present, and can
  select WER-only reward via `cer_weight: 0.0` and `wer_weight: 1.0`.
- `DTLM_fn` already collates saved `generation` sequences and scalar rewards in
  the shape expected by `ConditionalMaskGenerator`. It currently assumes audio
  is also present.

ROB-109 verified the rollout source for this issue:

- Rollout root: `/store/store4/data/l2augment_rollout_uvqmlm/`
- Dev split: 507 files and 5,070 sampled masks.
- Train split: 265,901 files.
- Each dev file has keys `audio`, `generation`, `mask`, and `reward`.
- Each file has 10 sampled generations.
- `reward[:, metric, stage]` stores before/after CER and WER, with the training
  reward normally computed as `reward[:, metric, 0] - reward[:, metric, 1]`.
- The rollout files contain float8 masks. The ROB-109 verification needed a
  torch 2.8 environment to deserialize them, so this plan should not assume the
  default training environment can load raw rollout `.pt` files directly.

## Design Assumptions

1. The target is a no-audio, single-mask model: reward-conditioned UVQLM, not
   audio-conditioned `ConditionalMaskGenerator` and not hierarchical
   `ConditionalMultiStepMaskGenerator`.
2. The training target is the saved VQ `generation` sequence from ROB-109, not a
   re-encoded mask. This avoids float8 mask decoding during training and matches
   the verified UVQLM generation provenance.
3. The conditioning scalar should default to WER delta because ROB-109 rewards
   include CER and WER and the repo's evaluation summaries use WER as the main
   criterion.
4. "Normalize reward of all the rollouts" should mean global training-split
   normalization, not per-file normalization across the 10 samples from one
   utterance. Per-file normalization would preserve only within-utterance rank
   and would make an absolute target reward ambiguous at generation time.
5. The first implementation should condition by replacing UVQLM's learned EOS
   BOS embedding with an MLP/timestep-encoder output from the normalized reward.
   This keeps the architecture close to UVQLM while making reward control the
   only new signal.
6. A later audio-conditioned variant can be added, but it should be a separate
   experiment because it changes the question from reward-conditioned UVQLM to a
   conditional audio policy.

## Proposed Model

Add a new policy class, tentatively named
`RewardConditionedUnconditionalMaskGenerator`.

Core architecture:

- Reuse the UVQLM `BinaryVariationalAutoEncoder` setup, VQ codebook size,
  token embedding table, GRU decoder, and prediction head.
- Add `reward_encoder`, either the existing `timestep_encoder(hidden_dim)` or a
  small MLP:

```text
normalized_reward -> Linear/SiLU/Linear or timestep_encoder -> hidden_dim BOS
```

- During teacher-forced training:
  - Input: normalized reward scalar and saved VQ code sequence.
  - BOS: `reward_encoder(normalized_reward)`.
  - Remaining inputs: embeddings of previous VQ codes.
  - Target: saved VQ codes plus EOS at the sequence length.
  - Loss: masked cross entropy over code/EOS positions, identical in structure
    to `UnconditionalMaskGenerator.forward_pass`.
- During generation:
  - Input: a requested normalized reward, or a raw reward plus stats for
    normalization.
  - BOS: encoded reward.
  - Autoregressively sample or greedily decode one VQ code sequence.
  - Decode through the frozen BVAE mask decoder and return
    `(augmented_audio, mask_pred, {"generation": generation, "conditioning_reward": ...})`.

Recommended config knobs:

- `default_conditioning_reward`: normalized scalar used by `augment`.
- `conditioning_reward_range`: optional `[low, high]` range in normalized units
  for random reward sampling at eval time, matching the existing CMultistep
  sweep pattern.
- `reward_encoder`: `timestep` or `mlp`.
- `reward_stats_path`: JSON file recording raw reward normalization stats.
- `reward_metric`: `wer`, `cer`, or weighted combination.
- `generation_sample`: boolean for sample versus greedy diagnostics if needed.

## Data Plan

Use the ROB-109 UVQLM rollout root:

```text
/store/store4/data/l2augment_rollout_uvqmlm/{train,dev}
```

Because the raw rollout files contain float8 masks, build a slim derived
training dataset that stores only fields needed for this model:

```text
{
  "generation": LongTensor[num_samples, seq_len],
  "reward_raw": FloatTensor[num_samples],
  "source_path": str,
  "reward_metric": "wer",
  "reward_stats": optional metadata
}
```

Proposed derived data path:

```text
/store/store5/data/acp21rjf_l2augment/rob111_reward_conditioned_vq_lm_slim/{train,dev}
```

Proposed builder:

```text
scripts/build_rob111_reward_conditioned_vq_dataset.py
```

Builder responsibilities:

- Read raw rollout files in a torch 2.8-compatible environment.
- Extract `generation`.
- Compute raw reward as WER delta:
  `rollout["reward"][:, 1, 0] - rollout["reward"][:, 1, 1]`.
- Replace NaNs with zero, as current dataset code does.
- Compute global train-split stats: mean, std, min, max, selected quantiles,
  positive/zero/negative counts.
- Normalize train and dev with the same train stats.
- Write slim `.pt` records and a committed or indexed `reward_stats.json`
  summary. Do not commit the full slim dataset if it is large.

This separates data extraction from training and keeps training independent of
float8 rollout deserialization.

## Training Infra Plan

Add the smallest set of reusable pieces:

1. `RewardConditionedVQDataset` in `l2augment/utils/datasets.py`
   - Reads slim records.
   - Returns `generation`, `reward`, and `source_path`.
   - Does not load audio or masks.

2. `RewardConditionedDTLM_fn` in `l2augment/utils/collate_functions.py`
   - Pads variable-length VQ sequences.
   - Concatenates rollout samples across files.
   - Returns `generations`, `generation_lengths`, and `rewards`.

3. `RewardConditionedUnconditionalMaskGenerator` in
   `l2augment/modelling/models.py`
   - Reuses the UVQLM code path.
   - Registers in `policy_dict`.
   - Shares as much helper logic as practical with `UnconditionalMaskGenerator`
     without refactoring the whole model file.

4. A training config under active configs, for example:

```text
exp/configs/reward_conditioned_unconditional_mask_lm/rob111_tedlium.yaml
```

5. A result directory for subsequent implementation:

```text
exp/results/repro/unconditional_lm/ROB-111_reward_conditioned_vq_lm/
```

The existing `exp/train_freq_mask.py` should be reusable after adding the new
dataset and collate function, because it already loads policy, dataset, and
collate implementations from config dictionaries.

## Evaluation Plan

Initial validation before any long run:

1. Builder smoke:
   - Run the slim-dataset builder on 2 train files and 2 dev files.
   - Confirm generated records contain only `generation`, normalized reward,
     raw reward, and metadata.

2. Model unit smoke:
   - Instantiate the model on CPU with the BVAE config and a tiny synthetic
     batch of VQ codes.
   - Run `forward_pass`.
   - Assert finite CE loss and backward pass.
   - Run `generate` at two reward values and confirm non-empty code sequences
     or a controlled EOS-only failure path.

3. One-file training smoke:
   - Run `exp/train_freq_mask.py` with `max_steps: 1`, batch size 2, and the
     slim mini dataset.
   - Confirm checkpoint save path works in a durable non-committed location.

4. One-recording eval smoke:
   - Plug the trained smoke checkpoint into an eval/oracle config.
   - Run one recording only, using Mimas with `with-gpu` if GPU is required.
   - Confirm `augment` returns a mask and generation metadata without error.

Only after the above should a full training job be queued. If a full run is
requested later, follow the project instruction to use a detached Mimas screen
session through `/store/store5/software/simple-gpu-schedule/with-gpu` with an
actual tested Linear callback wrapper.

## First Real Experiment Proposal

Recommended first full experiment:

- Dataset: slim ROB-109 UVQLM train/dev rollouts.
- Reward metric: WER delta.
- Reward normalization: global train mean/std, with stats saved.
- Model: reward-conditioned no-audio UVQLM.
- Architecture: UVQLM GRU decoder, hidden dim 256, four layers, BVAE codebook
  2048, reward BOS from `timestep_encoder`.
- Training: start from scratch, not from the unconditional UMLM checkpoint,
  because the BOS mechanism changes. Optional follow-up: initialize shared
  embeddings/decoder/prediction from UMLM and randomly initialize reward encoder.
- Diagnostics:
  - Train/dev CE loss.
  - Sequence NLL by reward bins.
  - Generated mask samples at normalized reward values such as p10, p50, p90.
  - Oracle/eval WER for fixed reward values and sampled reward ranges.

Suggested reward probes:

```text
normalized: -1.0, 0.0, 1.0, 2.0
raw WER delta: use reward_stats.json to map interpretable deltas into normalized units
```

## Risks And Checks

- Raw rollout loading can fail in older torch because of float8 masks. Mitigate
  by using a one-time torch 2.8 builder and training on slim records.
- Global reward normalization may be dominated by many negative deltas. Record
  quantiles and consider clipped z-score or percentile scaling if the
  distribution is too skewed.
- If the model is trained only on UVQLM samples, high reward samples are rare.
  Consider balanced sampling by reward bin after the baseline run if
  conditioning is ignored.
- If initializing from UMLM, checkpoint loading needs a partial-state load that
  excludes the old BOS-only behavior and includes the new reward encoder.
- Reward conditioning should be tested by comparing generated sequence
  statistics across requested rewards before spending GPU time on full WER
  evaluation.

## Review Questions

1. Should the first model be strictly no-audio, as assumed here, or should it
   include audio conditioning like `ConditionalMaskGenerator`?
2. Should reward normalization be global z-score, clipped z-score, or min-max
   over the train split?
3. Should the first training run start from scratch or initialize shared UVQLM
   layers from the existing UMLM checkpoint?
4. Should the conditioning reward at eval time be fixed to a high target, swept
   over several fixed values, or sampled from a range?
