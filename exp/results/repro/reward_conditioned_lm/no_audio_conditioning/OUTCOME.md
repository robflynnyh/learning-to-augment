# ROB-114 No-Audio Reward-Conditioned Mask LM

## Implementation

- Added `RewardConditionedMaskLMDataset` for saved ROB-109 UVQLM rollout
  `generation` tensors under `/store/store4/data/l2augment_rollout_uvqmlm`.
- Added `RewardConditionedMaskLM` collate logic that flattens sampled
  generations across rollout files and pads VQ sequences without audio fields.
- Added `RewardConditionedMaskLM`, a UVQLM-like no-audio mask LM conditioned by
  a bounded per-file reward scalar.
- Added full and smoke configs under
  `exp/configs/reward_conditioned_lm/no_audio_conditioning/`.

## Commands

Stats/model/augment smoke:

```bash
bash -ic 'export PYTHONPATH="$PWD:/exp/exp4/acp21rjf/long-context-asr:/exp/exp4/acp21rjf/language_modelling${PYTHONPATH:+:$PYTHONPATH}"; python exp/results/repro/reward_conditioned_lm/no_audio_conditioning/scripts/smoke_reward_conditioned_mask_lm.py'
```

One-file training harness smoke:

```bash
bash -ic 'export PYTHONPATH="$PWD:$PWD/exp:/exp/exp4/acp21rjf/long-context-asr:/exp/exp4/acp21rjf/language_modelling${PYTHONPATH:+:$PYTHONPATH}"; export WANDB_MODE=disabled; python exp/train_freq_mask.py --config exp/configs/reward_conditioned_lm/no_audio_conditioning/tedlium_per_utterance_smoke.yaml'
```

Full training was not queued because ROB-114 explicitly requires no long GPU
training unless a later human comment asks for it.

## Validation

Completed on 2026-05-21.

- Stats smoke: 2 train rollout files and 2 dev rollout files loaded from the
  in-place ROB-109 rollout root. Bounded min-max and raw WER-delta rewards were
  finite for 20 train samples and 20 dev samples, with normalized ranges
  `[0.0, 1.0]` for both splits. Degenerate reward groups: 0 train, 0 dev.
- CPU model smoke: finite CE loss `7.714467525482178`, backward pass completed,
  and fixed-length generation at normalized rewards `0.0` and `1.0` produced
  exactly 7 VQ tokens each.
- One-recording augment smoke: real dev rollout
  `/store/store4/data/l2augment_rollout_uvqmlm/dev/AlGore_2009_0.pt` had
  1042 audio frames, audio-derived generation length 29, metadata generation
  length 29, 29 generated VQ tokens, mask shape `[1, 80, 1042]`, and augmented
  audio shape `[1, 80, 1042]`.
- One-file training harness smoke: W&B disabled, CPU device, 1 train file,
  1 dev file, one training batch. Initial dev loss `7.758825302124023`, train
  batch loss `7.676178932189941`, final averaged dev loss
  `7.759217977523804`, and both tmp
  and final smoke checkpoints saved.

PR-review follow-up validation after removing import/load guards:

- The no-guard direct-load stats/model/augment smoke passed under the bashrc
  Python `3.10.12` / Torch `2.6.0+cu124` runtime and reproduced the committed
  JSON values.
- The one-file training harness smoke passed under the same runtime after
  `load_model()` was updated to load trusted local policy checkpoints with
  `weights_only=False`, which Torch 2.6 requires for existing checkpoints that
  contain OmegaConf config metadata. ROB-109 rollout files still use normal
  `torch.load`.

Durable smoke artifacts:

- `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/smoke/smoke_reward_conditioned_mask_lm.json`
- `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/smoke/no_audio_reward_conditioned_mask_lm_smoke.pt`
- `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/smoke/no_audio_reward_conditioned_mask_lm_smoke_tmp.pt`

The `.pt` checkpoint artifacts are intentionally ignored by Git.

## ROB-117 Training Queue

Started on 2026-05-21 from branch
`symphony/ROB-117-train-no-audio-reward-conditioned-mask-lm` at commit
`b755600ab27e90fdf411bde9d201d35f208cb494`.

Preflight:

- Local checkout is based on merged ROB-114 / PR #17 code at
  `b755600ab27e90fdf411bde9d201d35f208cb494`.
- Active config:
  `exp/configs/reward_conditioned_lm/no_audio_conditioning/tedlium_per_utterance.yaml`.
- Smoke config:
  `exp/configs/reward_conditioned_lm/no_audio_conditioning/tedlium_per_utterance_smoke.yaml`.
- Dataset root exists at `/store/store4/data/l2augment_rollout_uvqmlm/`, with
  `/train` using 147G and `/dev` using 531M at launch time.
- Full config has W&B logging enabled by `training.wandb_project: l2augment`
  and no disabled `training.wandb_mode`; dev-loss early stopping is enabled by
  `training.tolerance: 5`.
- Bashrc runtime check resolved `python` to `/usr/bin/python3.10`, Python
  `3.10.12`, Torch `2.6.0+cu124`, CUDA available.
- One-file training smoke passed with W&B disabled. Log:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/logs/rob117_smoke_20260521.log`.
- Actual wrapper EXIT-trap callback path passed in check-only mode:
  `ROB117_CALLBACK_ONLY=1 ROB117_CALLBACK_CHECK_ONLY=1 CALLBACK_TARGET_STATE=Todo scripts/launch_rob117_reward_conditioned_mask_lm_training.sh`.

Queued command:

```bash
screen -L -Logfile exp/results/repro/reward_conditioned_lm/no_audio_conditioning/logs/rob117_no_audio_reward_conditioned_mask_lm_training.screen.log -dmS rob117-reward-conditioned-mask-lm bash -lc 'cd /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment/ROB-117 && /store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- scripts/launch_rob117_reward_conditioned_mask_lm_training.sh'
```

Expected outputs:

- Main log:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/logs/rob117_no_audio_reward_conditioned_mask_lm_training.log`
- Screen log:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/logs/rob117_no_audio_reward_conditioned_mask_lm_training.screen.log`
- Result root:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/`
- Checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance.pt`
- Temporary checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/reward_conditioned_mask_lm/no_audio_tedlium_per_utterance_tmp.pt`

Training outcome is pending the detached run and callback. On completion,
inspect the log and checkpoint, then run the requested checkpoint-load and
fixed-length generation sanity check at two reward controls before PR/Linear
handoff.

## ROB-117 First Attempt Restart

The first detached full-training attempt acquired Mimas GPU 2 and started W&B
run `ejufy294` (`comic-field-2165`) on 2026-05-21, but it did not reach a first
dev batch. The process stayed alive with no loss charts because DataLoader
workers repeatedly failed before the initial `wandb.log(...)` calls:

```text
OSError: AF_UNIX path too long
```

Root cause: the wrapper set `TMPDIR` under the full result-root scratch path,
which made Python multiprocessing's Unix socket path too long. The run was
interrupted with `screen -S rob117-reward-conditioned-mask-lm -X stuff ^C`; the
wrapper `EXIT` trap posted a Linear failure callback and the screen exited.

Fix and validation before requeue:

- Updated `scripts/launch_rob117_reward_conditioned_mask_lm_training.sh` so the
  default scratch root is the short durable local path
  `/exp/exp4/acp21rjf/rob117-scratch`.
- Revalidated the actual wrapper callback path with:
  `SCRATCH_ROOT=/exp/exp4/acp21rjf/rob117-scratch ROB117_CALLBACK_ONLY=1 ROB117_CALLBACK_CHECK_ONLY=1 bash scripts/launch_rob117_reward_conditioned_mask_lm_training.sh`.
- Revalidated a DataLoader worker smoke against the active full config with
  `TMPDIR=/exp/exp4/acp21rjf/rob117-scratch/tmp`, 4 dev rollout files,
  `num_workers=2`, and `prefetch_factor=2`; the first batch loaded with
  generation shape `(20, 48)` and reward range `[0.0, 1.0]`.
- Reran the documented one-file training smoke through the actual wrapper using
  the smoke config and the short scratch path. Log:
  `exp/results/repro/reward_conditioned_lm/no_audio_conditioning/logs/rob117_smoke_retry_20260521.log`.
