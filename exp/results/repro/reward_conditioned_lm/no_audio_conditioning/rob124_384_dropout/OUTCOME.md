# ROB-124 Outcome

Status: training completed successfully and post-training fixed-length
generation sanity passed.

Launch branch/commit:

- Branch: `symphony/ROB-124-384-dropout-mask-lm`
- Commit: `63f16096c7dcfa8e91f1bb8cf4d8f21465afaa3d`

## What Changed

- Added configurable `dropout` to `RewardConditionedMaskLM`, defaulting to
  `0.0` for ROB-117 compatibility.
- Updated checkpoint saving to create parent directories before `torch.save`;
  this keeps new issue-specific smoke/full checkpoint paths from failing when
  the result subdirectory has not existed before.
- Added a 384-dimensional dropout config and matching tiny smoke config.
- Added a callback-backed Mimas launch wrapper using durable result/log paths
  and the short scratch-root pattern from ROB-117.

## Baseline For Comparison

- ROB-117 checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_resume100_500ep_lr1e3.pt`
- ROB-117 policy: 4-layer GRU, `hidden_dim: 256`, no dropout.
- ROB-117 trainable policy parameters: `2.63M`.
- ROB-117 best available standalone dev loss from the resumed run:
  `2.653739192269065`.
- ROB-117 assessment from PR #18: loadable and usable for downstream
  fixed-length generation/eval/oracle comparison, but the LR `1e-3` resume did
  not improve validation loss before early stopping restored the best previous
  state.

## ROB-124 Run Contract

- Policy: 4-layer GRU, `hidden_dim: 384`, `dropout: 0.1`.
- Data: `/store/store4/data/l2augment_rollout_uvqmlm/{train,dev}`.
- Reward normalization: per-utterance min-max WER delta, degenerate groups
  mapped to `0.5`.
- W&B: enabled by `training.wandb_project: l2augment` in the full config.
- Early stopping: dev-loss patience with `training.tolerance: 5`.
- Result root:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout/`
- Checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt`

## Queue Handoff

- Queued: 2026-05-22 15:25 UTC
- Screen: `rob124-reward-conditioned-mask-lm-384d-dropout0p1`
- Queue ticket: `b530a207`
- Pool: `1,2`
- Queued command:
  `screen -L -Logfile /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout/logs/rob124_no_audio_reward_conditioned_mask_lm_384d_dropout0p1_500ep_lr1e3.screen.log -dmS rob124-reward-conditioned-mask-lm-384d-dropout0p1 bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob124_reward_conditioned_mask_lm_training_384d_dropout0p1.sh'`
- Queued commit: `63f16096c7dcfa8e91f1bb8cf4d8f21465afaa3d`
- Callback target state: `Todo`

The callback returned with exit status `0` on 2026-05-22 and moved ROB-124 back
to `Todo` for finalization.

## Training Result

- W&B run: `wandering-planet-2168` / `b8cm3a2g`
  (<https://wandb.ai/wobrob101/l2augment/runs/b8cm3a2g>)
- Runtime: bashrc Python `/usr/bin/python3.10`, Torch `2.6.0+cu124`
- GPU assigned by `with-gpu 1,2`: `CUDA_VISIBLE_DEVICES=2`
- Full log:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout/logs/rob124_no_audio_reward_conditioned_mask_lm_384d_dropout0p1_500ep_lr1e3.log`
- Screen log:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout/logs/rob124_no_audio_reward_conditioned_mask_lm_384d_dropout0p1_500ep_lr1e3.screen.log`
- Final checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt`
- Checkpoint size: `30M`
- Trainable policy parameters: `5,124,993`
- Best logged dev loss: `2.6247272274710913`
- Final validation before rollback: `2.62546706199646`
- Early stopping: patience fired and the training loop restored the best
  previous state before writing the final checkpoint.

## Prequeue Validation

- Syntax/config checks passed:
  - `bash -n scripts/launch_rob124_reward_conditioned_mask_lm_training_384d_dropout0p1.sh`
  - `bash -ic 'python -m py_compile l2augment/modelling/models.py l2augment/utils/helpers.py exp/train_freq_mask.py'`
  - config parse for full and smoke YAMLs
  - `git diff --check`
- Tiny training smoke passed under bashrc Python 3.10 / Torch 2.6 after the
  checkpoint parent-directory fix:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout/logs/rob124_smoke_384d_dropout0p1_prequeue_retry_20260522.log`.
- Actual wrapper callback-only `EXIT` trap check passed:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout/logs/rob124_callback_only_check_20260522.log`.

## Post-Training Validation

The same ROB-117 checkpoint-load/fixed-length generation sanity script passed
against the ROB-124 checkpoint:

```bash
bash -ic 'export TMPDIR=/exp/exp4/acp21rjf/rob124-scratch/tmp; mkdir -p "$TMPDIR"; export PYTHONPATH="$PWD:$PWD/exp:/exp/exp4/acp21rjf/long-context-asr:/exp/exp4/acp21rjf/language_modelling${PYTHONPATH:+:$PYTHONPATH}"; python exp/results/repro/reward_conditioned_lm/no_audio_conditioning/scripts/post_training_sanity_check.py --config exp/configs/reward_conditioned_lm/no_audio_conditioning/tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.yaml --checkpoint /store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt --output exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout/post_training_sanity_check_384d_dropout0p1_500ep_lr1e3.json'
```

Artifact:
`exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout/post_training_sanity_check_384d_dropout0p1_500ep_lr1e3.json`

Result on `/store/store4/data/l2augment_rollout_uvqmlm/dev/AlGore_2009_0.pt`:

- Checkpoint loaded on CPU.
- Reward `0.0` and `1.0` both generated exactly `29` VQ tokens.
- Decoded masks and augmented audio matched `[1, 80, 1042]`.
- Reward `0.0` mask active fraction: `0.30000001192092896`.
- Reward `1.0` mask active fraction: `1.0`.
- Reward `0.0` vs `1.0` greedy generation mismatch: `29/29` tokens.

## Baseline Comparison

ROB-124 improves the no-audio LM dev loss under the same rollout data and
reward normalization:

- ROB-117 best available standalone dev loss: `2.653739192269065`.
- ROB-124 best logged dev loss: `2.6247272274710913`.
- Absolute delta: `-0.02901196479797357`.

This is a modest but real held-out LM loss improvement. The checkpoint is usable
for downstream eval/oracle comparison, and the result justifies evaluating this
384/dropout policy downstream before trying a substantially larger architecture.
