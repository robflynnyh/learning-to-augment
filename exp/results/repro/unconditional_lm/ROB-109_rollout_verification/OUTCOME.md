# ROB-109 UVQLM Rollout Verification

Verified on 2026-05-21 against:

- Rollout root: `/store/store4/data/l2augment_rollout_uvqmlm/dev`
- Local Mimas UMLM checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/UMLM/modelgpu.pt`
- Local Mimas BVAE checkpoint:
  `/store/store5/data/acp21rjf_checkpoints/l2augment/models/bvae/bvae_USINGTHISFORNOW_2048gpu.pt`

## Summary

The `/store/store4/data/l2augment_rollout_uvqmlm/dev` files are consistent with
the UVQLM/UMLM generation path and the checkpoint currently used for UVQLM work
on Mimas.

Evidence:

- `exp/configs/configs_in_paper/unconditional_mask_lm/UMLM.yaml` uses
  `policy.class: UnconditionalMaskGenerator`, writes to
  `/mnt/parscratch/users/acp21rjf/l2augment_rollout_uvqmlm/`, and points at the
  Stanage checkpoint paths
  `/mnt/parscratch/users/acp21rjf/l2augment_model/UMLM/modelgpu.pt` and
  `/mnt/parscratch/users/acp21rjf/l2augment_model/bvae/bvae_USINGTHISFORNOW_2048gpu.pt`.
- ROB-106 copied
  `stanage.shef.ac.uk:/mnt/parscratch/users/acp21rjf/l2augment_rollout_uvqmlm/`
  to `/store/store4/data/l2augment_rollout_uvqmlm/` using
  `scripts/launch_rob106_uvqlm_store4_sync.sh`.
- The local Mimas UMLM checkpoint is byte-identical to the Stanage
  `UMLM/modelgpu.pt` checkpoint:
  `7cbe29f0e924c9950b4125d840e0ce46a62a79d65552a26a0b858427ba4959dc`.
- The local Mimas BVAE checkpoint is byte-identical to the Stanage
  `bvae_USINGTHISFORNOW_2048gpu.pt` checkpoint:
  `87fc7bede3862bfbe6e7373ed621bd1603bc659d20ffd8ec152f4c51b7fcbaa1`.
- A sample synced rollout file is also byte-identical between Stanage and
  Mimas:
  `AlGore_2009_0.pt`
  `c233d24923a24cf1be77711add20d03e54e3f1eab24729c5201b96602f5964eb`.
- The files include the `generation` tensor returned by
  `UnconditionalMaskGenerator.augment`, matching the UVQLM sampled-code path.

Important limitation: the rollout `.pt` files do not embed checkpoint filenames
or checkpoint hashes. The verification is therefore provenance plus bytewise
checkpoint/source matching, not embedded metadata read from each rollout file.

The local UMLM checkpoint's embedded `config.generation.save_dir` still records
`/mnt/parscratch/users/acp21rjf/l2augment_rollout_mmr9e2/`, which appears to be
the rollout source used when training the UMLM checkpoint. That embedded field
does not control `exp/generate.py` output: the generation job loads the runtime
YAML, constructs the policy from that YAML, and loads only
`checkpoint['model_state_dict']` from `training.model_save_path`.

## Dev Rollout Contents

Inspection used torch 2.8 because the tensors include `torch.float8_e5m2`, which
the active default torch 2.0.1 environment cannot deserialize.

Command:

```bash
/store/store4/software/bin/anaconda3/envs/speech-diff/bin/python - <<'PY'
from pathlib import Path
import torch

root = Path('/store/store4/data/l2augment_rollout_uvqmlm/dev')
files = sorted(root.glob('*.pt'))
print('torch', torch.__version__)
print('file_count', len(files))
for path in files[:5]:
    data = torch.load(path, map_location='cpu')
    print(path.name, sorted(data), data['reward'].shape, data['mask'].shape, data['generation'].shape)
PY
```

Observed:

- Dev files: `507`
- Top-level tree counts: `train 265901`, `dev 507`
- Every dev file has keys: `audio`, `generation`, `mask`, `reward`
- Every dev reward tensor has shape `(10, 2, 2)`
- Every dev mask tensor has shape `(10, 1, 80, T)`, where `T` is the utterance
  frame length
- Every dev file has 10 rollout samples, matching `job_val.sh --repeats 10`

Example file:

```text
AlGore_2009_0.pt
reward:     torch.Size([10, 2, 2]) float32
mask:       torch.Size([10, 1, 80, 1042]) float8_e5m2
audio:      torch.Size([1, 80, 1042]) float16
generation: torch.Size([10, 29]) int64
```

## Reward Format

For each sampled mask, `exp/generate.py` runs `cpu_rollout(...)` and saves:

```python
reward = torch.stack([prev_cer, u_cer], dim=-1)
```

With `generation.return_wer: true`, `cpu_rollout` returns:

```python
prev = torch.stack([prev_cer, prev_wer])
updated = torch.stack([updated_cer, updated_wer])
```

Therefore each reward entry is:

```text
reward[rollout_index, metric_index, stage_index]

metric_index 0 = CER
metric_index 1 = WER

stage_index 0 = before adaptation
stage_index 1 = after one adaptation step on the masked audio
```

So a single mask has four logged values:

```text
[
  [before_CER, after_CER],
  [before_WER, after_WER],
]
```

The actual reward/decrease for a metric is:

```python
delta = reward[:, :, 0] - reward[:, :, 1]
```

Positive `delta` means the one-step adaptation improved the metric; negative
`delta` means it made the metric worse.

Across all 507 dev files and 5070 sampled masks:

```text
delta mean [CER, WER]: [-17.798837661743164, -21.442310333251953]
delta std  [CER, WER]: [34.928070068359375, 37.11581802368164]
delta min  [CER, WER]: [-241.2587432861328, -220.0]
delta max  [CER, WER]: [41.66666793823242, 22.22222137451172]
positive delta counts [CER, WER]: [560, 447] out of 5070
zero delta counts     [CER, WER]: [1662, 1931] out of 5070
```

These aggregate values are descriptive only; the verification task did not run a
new generation job.

## Verification Commands

Checkpoint hash comparison:

```bash
sha256sum \
  /store/store5/data/acp21rjf_checkpoints/l2augment/models/UMLM/modelgpu.pt \
  /store/store5/data/acp21rjf_checkpoints/l2augment/models/bvae/bvae_USINGTHISFORNOW_2048gpu.pt

ssh -o BatchMode=yes -o ConnectTimeout=10 stanage.shef.ac.uk \
  "sha256sum /mnt/parscratch/users/acp21rjf/l2augment_model/UMLM/modelgpu.pt /mnt/parscratch/users/acp21rjf/l2augment_model/bvae/bvae_USINGTHISFORNOW_2048gpu.pt"
```

Sample rollout source/destination hash comparison:

```bash
sha256sum /store/store4/data/l2augment_rollout_uvqmlm/dev/AlGore_2009_0.pt

ssh -o BatchMode=yes -o ConnectTimeout=10 stanage.shef.ac.uk \
  "sha256sum /mnt/parscratch/users/acp21rjf/l2augment_rollout_uvqmlm/dev/AlGore_2009_0.pt"
```

Code paths checked:

- `exp/configs/configs_in_paper/unconditional_mask_lm/UMLM.yaml`
- `exp/launch_scripts/job_val.sh`
- `exp/generate.py`
- `l2augment/rollout/cpu.py`
- `l2augment/modelling/models.py`
- `scripts/launch_rob106_uvqlm_store4_sync.sh`
