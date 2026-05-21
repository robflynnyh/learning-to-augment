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
bash -ic 'export PYTHONPATH="$PWD:/exp/exp4/acp21rjf/long-context-asr:/exp/exp4/acp21rjf/language_modelling${PYTHONPATH:+:$PYTHONPATH}"; python - <<'"'"'PY'"'"'
import runpy
import sys
import torch
print("python", sys.executable, sys.version.split()[0])
print("torch", torch.__version__, torch.__file__)
sys.path.append("/store/store4/software/bin/anaconda3/envs/flash_attn_pytorch2/lib/python3.9/site-packages")
sys.argv = ["exp/results/repro/reward_conditioned_lm/no_audio_conditioning/scripts/smoke_reward_conditioned_mask_lm.py"]
runpy.run_path(sys.argv[0], run_name="__main__")
PY'
```

One-file training harness smoke:

```bash
bash -ic 'export PYTHONPATH="$PWD:/exp/exp4/acp21rjf/long-context-asr:/exp/exp4/acp21rjf/language_modelling${PYTHONPATH:+:$PYTHONPATH}"; export WANDB_MODE=disabled; python - <<'"'"'PY'"'"'
import runpy
import sys
import torch
print("python", sys.executable, sys.version.split()[0])
print("torch", torch.__version__, torch.__file__)
sys.path.insert(0, "exp")
sys.path.append("/store/store4/software/bin/anaconda3/envs/flash_attn_pytorch2/lib/python3.9/site-packages")
sys.argv = [
    "exp/train_freq_mask.py",
    "--config",
    "exp/configs/reward_conditioned_lm/no_audio_conditioning/tedlium_per_utterance_smoke.yaml",
]
runpy.run_path(sys.argv[0], run_name="__main__")
PY'
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

- The literal `/store/store4/software/bin/anaconda3/envs/flash_attn_pytorch2/bin/python`
  interpreter in this Symphony shell reports Python `3.9.17` and Torch `2.0.1`.
- The human interactive transcript is reproduced by `bash -ic python`, where
  `python` is aliased to `/usr/bin/python3.10` and imports Torch
  `2.6.0+cu124` from `~/.local`.
- The no-guard direct-load stats/model/augment smoke passed under that Python
  `3.10.12` / Torch `2.6.0+cu124` runtime and reproduced the committed JSON
  values.
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
