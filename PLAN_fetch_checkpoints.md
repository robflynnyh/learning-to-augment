# PLAN — fetch UFMR checkpoints from Stanage

You (Claude) are about to be re-launched on `acp21rjf@stanage.shef.ac.uk`. The user wants the UFMR checkpoints (and the frozen ASR backbone they depend on) pulled out of `parscratch` and into a long-lived location on Stanage:

**Destination root:** `/store/store5/data/acp21rjf_checkpoints/`

`parscratch` is purgeable; `/store/store5/...` is the persistent store. That's the whole point of moving the files.

## Goal

Produce a self-contained sub-tree under `/store/store5/data/acp21rjf_checkpoints/` that contains everything needed to reproduce `results/UFMR/`:

```
/store/store5/data/acp21rjf_checkpoints/
└── learning-to-augment/
    ├── asr/step_105360.pt                    # frozen long-context ASR backbone
    ├── ufmr/test_wer/model.pt                # UFMR policy (WER-only, explicit)
    ├── ufmr/test_cer/model.pt                # UFMR policy (named test_cer, but also WER-only via defaults)
    └── MANIFEST.md                           # what each file is, source path, sha256, size
```

The `learning-to-augment/` sub-folder keeps these files namespaced so other projects under `acp21rjf_checkpoints/` don't collide. Confirm the layout convention with the user before committing to it if a different scheme is already in use under `/store/store5/data/acp21rjf_checkpoints/` (just `ls` it first).

## Source paths on Stanage

From `exp/configs/UFMR.yaml` and `exp/configs/UFMR_wer.yaml`:

| Purpose | Path |
|---|---|
| Frozen ASR | `/mnt/parscratch/users/acp21rjf/spotify/rotary_pos_6l_256d_seq_sched/n_seq_sched_2048_rp_1/step_105360.pt` |
| UFMR (WER-only, explicit `wer_weight=1.0`) | `/mnt/parscratch/users/acp21rjf/l2augment_model/ufm/test_wer/model.pt` |
| UFMR (defaults, also WER-only) | `/mnt/parscratch/users/acp21rjf/l2augment_model/ufm/test_cer/model.pt` |

The matching `tmp_model.pt` files are mid-training snapshots; ignore unless `model.pt` is missing — in which case `tmp_model.pt` from the same folder is the most-recent epoch and is the fallback.

There is also a (possibly second) ASR checkpoint commented out in the configs:
`/mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_rb/n_seq_sched_2048_rp_1/step_105360.pt` — the **uncommented** one is the canonical one; only fetch the alternative if the canonical one is missing.

## Step-by-step

1. **Confirm you're on Stanage.** `hostname` should match `*.stanage.*` and `/mnt/parscratch/users/acp21rjf/` should be a real directory. If not, abort and tell the user — there is nothing useful to do from elsewhere.

2. **Verify the canonical files exist** before copying anything:
   ```bash
   ls -lh /mnt/parscratch/users/acp21rjf/spotify/rotary_pos_6l_256d_seq_sched/n_seq_sched_2048_rp_1/step_105360.pt
   ls -lh /mnt/parscratch/users/acp21rjf/l2augment_model/ufm/test_wer/model.pt
   ls -lh /mnt/parscratch/users/acp21rjf/l2augment_model/ufm/test_cer/model.pt
   ```
   If any are missing:
   - For `model.pt`, check `tmp_model.pt` in the same dir and surface that to the user before falling back.
   - For the ASR `.pt`, try the alternative `checkpoints_seq_scheduler_rb` path. Do **not** silently substitute — report what you found.

3. **Inspect the destination** before creating anything:
   ```bash
   ls -la /store/store5/data/acp21rjf_checkpoints/ 2>&1 | head -40
   df -h  /store/store5/data/acp21rjf_checkpoints/
   ```
   Make sure the directory is writable and has room. Then create the sub-tree:
   ```bash
   DEST=/store/store5/data/acp21rjf_checkpoints/learning-to-augment
   mkdir -p "$DEST"/{asr,ufmr/test_wer,ufmr/test_cer}
   ```
   If a different naming convention already exists under `acp21rjf_checkpoints/` (e.g. flat layout, project-prefix), follow it instead and adjust paths below — but only after confirming with the user.

4. **Copy with `rsync -avh --progress` and checksums** (don't use plain `cp` — we want a verifiable transfer):
   ```bash
   DEST=/store/store5/data/acp21rjf_checkpoints/learning-to-augment

   rsync -avh --progress --checksum \
     /mnt/parscratch/users/acp21rjf/spotify/rotary_pos_6l_256d_seq_sched/n_seq_sched_2048_rp_1/step_105360.pt \
     "$DEST"/asr/

   rsync -avh --progress --checksum \
     /mnt/parscratch/users/acp21rjf/l2augment_model/ufm/test_wer/model.pt \
     "$DEST"/ufmr/test_wer/

   rsync -avh --progress --checksum \
     /mnt/parscratch/users/acp21rjf/l2augment_model/ufm/test_cer/model.pt \
     "$DEST"/ufmr/test_cer/
   ```

5. **Sanity-check that the policy files actually load** with PyTorch — they're tiny, so this is cheap and catches a corrupt copy now rather than at eval time:
   ```bash
   DEST=/store/store5/data/acp21rjf_checkpoints/learning-to-augment
   python - <<PY
   import torch
   for p in [
       "$DEST/ufmr/test_wer/model.pt",
       "$DEST/ufmr/test_cer/model.pt",
   ]:
       sd = torch.load(p, map_location="cpu", weights_only=False)
       print(p, "OK", "keys=", list(sd)[:3] if hasattr(sd, "keys") else type(sd))
   PY
   ```
   For the ASR checkpoint, just confirm it loads as a dict containing `'config'` and `'model'` (that's what `exp/eval.py:38-40` expects).

6. **Write `$DEST/MANIFEST.md`** so future sessions know exactly what each file is. Include for each entry: original parscratch path, sha256, size in MB, copied-on date, and the config it corresponds to. Use `sha256sum` to compute hashes.

7. **Do NOT modify the YAML configs in the repo**, and do NOT rewrite paths in committed files. The original `/mnt/parscratch/...` strings still need to work for anyone running on a fresh Stanage rollout. Future eval runs should copy a YAML and edit `checkpointing.asr_model` / `training.model_save_path` to point at the `/store/store5/data/acp21rjf_checkpoints/learning-to-augment/...` paths in the local copy.

8. **Report back to the user** with: file sizes, sha256s, the destination paths, and a one-line `rsync` summary (e.g. `total size is X bytes, transferred Y`). If anything was missing or substituted, surface it loudly.

## Out of scope (do not do unless the user asks)

- Do not pull the rollout directory `/mnt/parscratch/users/acp21rjf/l2augment_rollout_test/`. It's only needed to **retrain** UFMR; the user wants to **reproduce eval results**, which only needs the policy + ASR checkpoints. The rollout dir is also potentially huge.
- Do not pull the `tmp_model.pt` snapshots unless `model.pt` is missing.
- Do not pull every checkpoint under `l2augment_model/` — only the two UFMR ones above.
- The checkpoints live outside the repo (`/store/store5/...`), so there is nothing to add to `.gitignore`.

## After you're done

The user will be running eval jobs on Stanage from the same repo. Next steps for them (or for you on the next invocation):

1. Copy one of `exp/configs/configs_in_paper/UFRM/UFRM_eval/{singlestep,singleepoch,multiepoch}/*.yaml` to a local variant.
2. Edit `checkpointing.asr_model` → `/store/store5/data/acp21rjf_checkpoints/learning-to-augment/asr/step_105360.pt` and `training.model_save_path` → the matching `ufmr/test_wer/model.pt` (or `test_cer/model.pt`).
3. Run `python exp/eval.py --config <local cfg>`.

Leave a one-line note in your final message reminding the user of this so they don't have to re-derive it.
