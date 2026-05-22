# ROB-124 384-Dim Dropout Reward-Conditioned Mask LM

ROB-124 is the controlled capacity/dropout ablation following the completed
ROB-117 no-audio reward-conditioned mask LM handoff.

## Context

- Baseline checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_resume100_500ep_lr1e3.pt`
- Baseline PR: <https://github.com/robflynnyh/learning-to-augment/pull/18>
- Baseline merged head: `deddca9d3d78acd3b9734b88b858f9698b0f81a8`
- Baseline merge commit: `ddd349c30acb056cc791788e85e46153fa344c1b`

The active checkout for ROB-124 was verified to contain the ROB-117 merge
commit before editing.

ROB-124 launch branch/commit:

- Branch: `symphony/ROB-124-384-dropout-mask-lm`
- Commit: `63f16096c7dcfa8e91f1bb8cf4d8f21465afaa3d`

## Config

- Full training config:
  `exp/configs/reward_conditioned_lm/no_audio_conditioning/tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.yaml`
- Smoke config:
  `exp/configs/reward_conditioned_lm/no_audio_conditioning/tedlium_per_utterance_384d_dropout0p1_smoke.yaml`
- Policy class: `RewardConditionedMaskLM`
- Policy change from ROB-117 baseline:
  - `hidden_dim: 384`
  - `dropout: 0.1`
- Reward/data contract is unchanged:
  - rollout root: `/store/store4/data/l2augment_rollout_uvqmlm/{train,dev}`
  - reward metric: WER
  - normalization: per-utterance min-max
  - degenerate reward groups: `0.5`
  - no-audio training on saved VQ `generation` sequences

Dropout is exposed in `RewardConditionedMaskLM` with a default of `0.0`, so
older configs/checkpoints keep the ROB-117 behavior. For ROB-124, the same
`dropout: 0.1` value is applied to the 4-layer GRU's inter-layer dropout and
the prediction head.

## Final Run

- Result root:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout/`
- Wrapper:
  `scripts/launch_rob124_reward_conditioned_mask_lm_training_384d_dropout0p1.sh`
- Scratch root:
  `/exp/exp4/acp21rjf/rob124-scratch`
- Checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt`
- Temporary checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3_tmp.pt`

Queue command:

```bash
screen -L -Logfile /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124/exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout/logs/rob124_no_audio_reward_conditioned_mask_lm_384d_dropout0p1_500ep_lr1e3.screen.log -dmS rob124-reward-conditioned-mask-lm-384d-dropout0p1 bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-124 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob124_reward_conditioned_mask_lm_training_384d_dropout0p1.sh'
```

Expected full-run logs:

- Main log:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout/logs/rob124_no_audio_reward_conditioned_mask_lm_384d_dropout0p1_500ep_lr1e3.log`
- Screen log:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout/logs/rob124_no_audio_reward_conditioned_mask_lm_384d_dropout0p1_500ep_lr1e3.screen.log`

Training exited cleanly on 2026-05-22 with status `0` via the callback-backed
Mimas `screen` wrapper.

- W&B run: `wandering-planet-2168` / `b8cm3a2g`
  (<https://wandb.ai/wobrob101/l2augment/runs/b8cm3a2g>)
- Runtime: bashrc Python `/usr/bin/python3.10`, Torch `2.6.0+cu124`
- GPU assigned by `with-gpu 1,2`: `CUDA_VISIBLE_DEVICES=2`
- Checkpoint size: `30M`
- Trainable policy parameters: `5.12M`
- Best logged dev loss: `2.6247272274710913`
- Final validation before rollback: `2.62546706199646`
- Early stopping: patience fired and the training loop restored the best
  previous state before writing the final checkpoint.

## Queue Handoff

Queued on Mimas at 2026-05-22 15:25 UTC.

- Screen: `rob124-reward-conditioned-mask-lm-384d-dropout0p1`
- Queue ticket: `b530a207`
- Pool: `1,2`
- Queued commit: `63f16096c7dcfa8e91f1bb8cf4d8f21465afaa3d`
- Completion callback target state: `Todo`

Completion check:

```bash
screen -ls | rg 'rob124-reward-conditioned-mask-lm-384d-dropout0p1'
tail -80 exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout/logs/rob124_no_audio_reward_conditioned_mask_lm_384d_dropout0p1_500ep_lr1e3.screen.log
tail -80 exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout/logs/rob124_no_audio_reward_conditioned_mask_lm_384d_dropout0p1_500ep_lr1e3.log
ls -lh /store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt
```

## Post-Training Validation

Before full training:

```bash
bash -n scripts/launch_rob124_reward_conditioned_mask_lm_training_384d_dropout0p1.sh
bash -ic 'export TMPDIR=/exp/exp4/acp21rjf/rob124-scratch/tmp; mkdir -p "$TMPDIR"; export PYTHONPATH="$PWD:$PWD/exp:/exp/exp4/acp21rjf/long-context-asr:/exp/exp4/acp21rjf/language_modelling${PYTHONPATH:+:$PYTHONPATH}"; python exp/train_freq_mask.py --config exp/configs/reward_conditioned_lm/no_audio_conditioning/tedlium_per_utterance_384d_dropout0p1_smoke.yaml'
ROB124_CALLBACK_ONLY=1 ROB124_CALLBACK_CHECK_ONLY=1 CALLBACK_TARGET_STATE=Todo scripts/launch_rob124_reward_conditioned_mask_lm_training_384d_dropout0p1.sh
```

The ROB-117 checkpoint-load/fixed-length generation sanity script passed against
the ROB-124 checkpoint at reward controls `0.0` and `1.0`:

```bash
bash -ic 'export TMPDIR=/exp/exp4/acp21rjf/rob124-scratch/tmp; mkdir -p "$TMPDIR"; export PYTHONPATH="$PWD:$PWD/exp:/exp/exp4/acp21rjf/long-context-asr:/exp/exp4/acp21rjf/language_modelling${PYTHONPATH:+:$PYTHONPATH}"; python exp/results/repro/reward_conditioned_lm/no_audio_conditioning/scripts/post_training_sanity_check.py --config exp/configs/reward_conditioned_lm/no_audio_conditioning/tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.yaml --checkpoint /store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_384d_dropout0p1_500ep_lr1e3.pt --output exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout/post_training_sanity_check_384d_dropout0p1_500ep_lr1e3.json'
```

Artifact:
`exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout/post_training_sanity_check_384d_dropout0p1_500ep_lr1e3.json`

Sanity result on `/store/store4/data/l2augment_rollout_uvqmlm/dev/AlGore_2009_0.pt`:

- Checkpoint loaded on CPU.
- Both reward controls generated exactly `29` VQ tokens.
- Both decoded masks and augmented audio matched the input audio shape
  `[1, 80, 1042]`.
- Reward `0.0` fixed generation mask active fraction: `0.30000001192092896`.
- Reward `1.0` fixed generation mask active fraction: `1.0`.
- Reward `0.0` vs `1.0` greedy generation mismatch: `29/29` tokens.

Comparison to ROB-117 baseline:

- ROB-117 resumed baseline checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_resume100_500ep_lr1e3.pt`
- ROB-117 trainable policy parameters: `2.63M`
- ROB-117 best available standalone dev loss from the resumed run:
  `2.653739192269065`
- ROB-124 trainable policy parameters: `5.12M`
- ROB-124 best logged dev loss: `2.6247272274710913`
- Absolute dev-loss delta: `-0.02901196479797357`

Interpretation: the 384-dim/dropout checkpoint is loadable and usable for
downstream eval/oracle comparison. It improves the matched no-audio LM dev loss
modestly versus the ROB-117 baseline, so it justifies downstream comparison of
this checkpoint before trying substantially larger architectures.

## Prequeue Validation

- `bash -n scripts/launch_rob124_reward_conditioned_mask_lm_training_384d_dropout0p1.sh`
- `bash -ic 'python -m py_compile l2augment/modelling/models.py l2augment/utils/helpers.py exp/train_freq_mask.py'`
- Config parse confirmed both ROB-124 configs set `hidden_dim=384`,
  `dropout=0.1`, and the rollout root
  `/store/store4/data/l2augment_rollout_uvqmlm`.
- Tiny training smoke passed under bashrc Python 3.10 / Torch 2.6:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout/logs/rob124_smoke_384d_dropout0p1_prequeue_retry_20260522.log`.
  The smoke loaded the frozen BVAE and reported `5.12M` trainable policy
  parameters.
- Actual wrapper `EXIT` callback path passed in check-only mode:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/rob124_384_dropout/logs/rob124_callback_only_check_20260522.log`.
- `git diff --check`
