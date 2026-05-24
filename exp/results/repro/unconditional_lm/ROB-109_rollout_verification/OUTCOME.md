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
- Re-decoding every saved dev `generation` sequence through the current Mimas
  BVAE checkpoint reproduces the saved masks to within 12 differing binary
  pixels out of 460,274,400 checked pixels across all 5,070 dev rollout
  sequences.
- The saved dev `generation` sequences have much lower negative log-likelihood
  under the current Mimas UMLM checkpoint than random same-length VQ sequences:
  mean NLL `2.8811` vs. random-code mean NLL `10.7293`.

Important limitation: the rollout `.pt` files do not embed checkpoint filenames
or checkpoint hashes. The verification is therefore provenance plus bytewise
checkpoint/source matching plus behavior-level consistency with the current
Mimas checkpoints, not embedded metadata read from each rollout file.

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

## VQ Sequence And Mask Verification

After the follow-up request, I added a behavior-level verification that does not
depend only on source/destination hashes:

1. Load each saved `/store/store4/data/l2augment_rollout_uvqmlm/dev/*.pt` file.
2. Take each saved `generation` VQ-code sequence.
3. Decode that fixed code sequence through the current Mimas BVAE checkpoint
   using the same decoder path used by `UnconditionalMaskGenerator.generate`.
4. Compare the decoded binary mask against the saved `mask` tensor.
5. Score the saved code sequence under the current Mimas UMLM checkpoint and
   compare it with a random same-length VQ-code baseline.

Command:

```bash
/store/store4/software/bin/anaconda3/envs/speech-diff/bin/python \
  exp/results/scripts/verify_rob109_uvqlm_rollout_sequences.py \
  --max-files 0 \
  --output-json \
  exp/results/repro/unconditional_lm/ROB-109_rollout_verification/uvqlm_sequence_verification.json
```

Results:

```text
verified files: 507 / 507
verified sequences: 5070
mask mismatch pixels: 12 / 460274400
mask mismatch mean rate: 2.858976174958292e-08
mask mismatch max rate for any sequence: 2.6881720430107527e-05

saved sequence NLL mean/min/max/std:
  2.8810917640931506 / 0.25088369846343994 / 7.844291687011719 / 1.319888008691828

random same-length code NLL mean/min/max/std:
  10.729314022609703 / 4.61667013168335 / 12.436729431152344 / 0.854364330410143
```

The 12 mismatched pixels are isolated single-pixel differences after recomputing
the BVAE decoder on CPU and rounding the sigmoid output back to a binary mask.
This is behaviorally consistent with the saved masks having been decoded from
those exact VQ sequences using the Mimas/Stanage-equivalent BVAE checkpoint. The
NLL comparison is also consistent with the saved VQ sequences coming from the
Mimas/Stanage-equivalent UMLM checkpoint: they are far more likely under the
trained UMLM than random VQ-code sequences of the same lengths.

The full machine-readable summary is committed at:

```text
exp/results/repro/unconditional_lm/ROB-109_rollout_verification/uvqlm_sequence_verification.json
```

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
