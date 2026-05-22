# ROB-124 Outcome

Status: queued for callback-backed Mimas training. Final training outcome is
pending the detached run and callback.

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

## Pending Completion Work

When the callback returns ROB-124 to `Todo`:

1. Inspect the main log, screen log, W&B run, and checkpoint path.
2. If training succeeded, run the same post-training checkpoint-load and
   fixed-length generation sanity checks at reward controls `0.0` and `1.0`.
3. Compare ROB-124 dev-loss behavior against the ROB-117 baseline checkpoint.
4. Update this file with the final result, checkpoint usability, and whether
   the 384/dropout ablation justifies larger architectures.
