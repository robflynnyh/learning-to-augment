# PLAN — fetch UFMR checkpoints from Stanage  *(executed 2026-04-28)*

> **Status:** done. This file is now a record of what was pulled and where it lives, plus the procedure for re-doing it from scratch (e.g. on another box, or after a parscratch purge).

## What's on disk now

```
/store/store5/data/acp21rjf_checkpoints/l2augment/
├── MANIFEST.md                              # detailed table of every file (sha256, source, role)
├── asr/
│   └── step_105360.pt                       # frozen long-context ASR backbone (191M)
└── ufmr/
    ├── mseloss/{model,tmp_model}.pt         # loss_type=mse, rollout corpus l2augment_rollout
    ├── mseloss2e1/{model,tmp_model}.pt      # loss_type=mse, rollout corpus l2augment_rollout_2e1
    ├── multloss/{model,tmp_model}.pt        # loss_type=mult, rollout corpus l2augment_rollout
    ├── test/tmp_model.pt                    # no model.pt was ever saved; reward weights default; policy.lr=1e-5
    ├── test_cer/{model,tmp_model}.pt        # named "test_cer" but defaults → effectively WER-only
    ├── test_wer/{model,tmp_model}.pt        # explicit wer_weight=1.0, cer_weight=0.0
    └── test_wer_cer/{model,tmp_model}.pt    # blended wer_weight=0.5, cer_weight=0.5
```

The two checkpoint *families* — older `loss_type` experiments (Feb '25) and the Mar 12 reward-weighting set — are distinguished by what's set in the embedded `config` (see `MANIFEST.md` for exact values).

`/mnt/parscratch/...` is purgeable; `/store/store5/...` is the persistent store. That's the whole point of the move.

## Source layout on Stanage

| Purpose | Path |
|---|---|
| Frozen ASR | `/mnt/parscratch/users/acp21rjf/spotify/rotary_pos_6l_256d_seq_sched/n_seq_sched_2048_rp_1/step_105360.pt` |
| All UFMR variants (folder per variant) | `/mnt/parscratch/users/acp21rjf/l2augment_model/ufm/<variant>/{model,tmp_model}.pt` |

There is also an alternative ASR checkpoint at `/mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_rb/n_seq_sched_2048_rp_1/step_105360.pt` (commented out in the YAMLs). The canonical one above is the one currently in use; only fetch the alternative if it's missing.

## Procedure (to re-pull everything)

These steps were used on `mimas`, ssh-ing to Stanage. Adapt paths if running from elsewhere.

1. ✅ **Verify source files exist.**
   ```bash
   ssh acp21rjf@stanage.shef.ac.uk \
     'ls -lh /mnt/parscratch/users/acp21rjf/spotify/rotary_pos_6l_256d_seq_sched/n_seq_sched_2048_rp_1/step_105360.pt && \
      ls -lh /mnt/parscratch/users/acp21rjf/l2augment_model/ufm/'
   ```
   If the canonical ASR `.pt` is missing, fall back to the `checkpoints_seq_scheduler_rb` alternative — but report the substitution loudly, do not silently swap.

2. ✅ **Confirm destination layout.** This box uses a flat per-project convention (`spotifyASR`, `lcasr`, ...) under `/store/store5/data/acp21rjf_checkpoints/`. We added `l2augment/` alongside the others.

3. ✅ **Pull the ASR backbone.**
   ```bash
   DEST=/store/store5/data/acp21rjf_checkpoints/l2augment
   mkdir -p "$DEST/asr"
   rsync -ah --checksum --info=progress2 \
     acp21rjf@stanage.shef.ac.uk:/mnt/parscratch/users/acp21rjf/spotify/rotary_pos_6l_256d_seq_sched/n_seq_sched_2048_rp_1/step_105360.pt \
     "$DEST/asr/"
   ```

4. ✅ **Pull every UFMR variant in one go** (filtered to just `model.pt` and `tmp_model.pt` so we don't accidentally drag rollouts):
   ```bash
   rsync -ah --checksum --info=progress2 \
     --include='*/' --include='model.pt' --include='tmp_model.pt' --exclude='*' \
     acp21rjf@stanage.shef.ac.uk:/mnt/parscratch/users/acp21rjf/l2augment_model/ufm/ \
     "$DEST/ufmr/"
   ```

5. ✅ **Sanity-load and characterise.** Each `model.pt` is a dict with `model_state_dict` + `config`; each `config` records the variant's `policy.config.loss_type` and/or `dataset.{wer_weight,cer_weight}` — that's how you tell the variants apart. The ASR `.pt` is a dict with `model` + `config` (`exp/eval.py:38-40` consumes those keys).
   ```bash
   /store/store4/software/bin/anaconda3/envs/flash_attn_pytorch2/bin/python - <<'PY'
   import torch, json
   from pathlib import Path
   for mp in sorted(Path("/store/store5/data/acp21rjf_checkpoints/l2augment/ufmr").glob("*/model.pt")):
       ck = torch.load(mp, map_location="cpu", weights_only=False)
       cfg = ck.get("config", {})
       print(mp.parent.name, "policy:", dict(cfg.get("policy", {}).get("config", {}) or {}),
                              "dataset:", {k: cfg.get("dataset", {}).get(k) for k in ("wer_weight","cer_weight")})
   PY
   ```

6. ✅ **sha256 + write `MANIFEST.md`** under `$DEST/`. See the existing one for the exact format.

7. ✅ **Don't modify the YAML configs in the repo.** The original `/mnt/parscratch/...` strings still need to work on a fresh Stanage rollout. Use `exp/launch_scripts/run_eval_mimas.sh` instead — it materialises a patched copy of the YAML at runtime (and sets the `L2A_*_DIR` env vars added in `l2augment/utils/data.py` so dataset paths resolve to `/store/store4/data/...` on this box).

## Out of scope (do not pull unless asked)

- `/mnt/parscratch/users/acp21rjf/l2augment_rollout*/` — only needed to **retrain** UFMR. Potentially huge.
- The alternative ASR checkpoint at `checkpoints_seq_scheduler_rb/...` — only fetch if the canonical one is missing.
- TAL (`this_american_life`) dataset — not on this box; would need to come from parscratch/elsewhere if a TAL eval is ever needed.

## After you're done — running an eval on mimas

```
exp/launch_scripts/run_eval_mimas.sh <config.yaml> [variant]
```

`<config.yaml>` is one of `exp/configs/configs_in_paper/UFRM/UFRM_eval/{singlestep,singleepoch,multiepoch}/*.yaml`. `[variant]` is any subdirectory of `ufmr/` (defaults to `test_wer`); the script falls back to `tmp_model.pt` automatically when no `model.pt` exists (the `test/` variant). Results land at `exp/results/UFMR_mimas/<variant>/<regime>/<dataset>.txt`.
