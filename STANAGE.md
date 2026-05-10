# Stanage notes for Learning-to-Augment

Project-specific remote paths and checkpoint transfer notes. General Stanage access belongs in the `stanage-access` skill; this file is only for this repo.

## Remote paths on Stanage

- User parscratch root: `/mnt/parscratch/users/acp21rjf/`
- Learn-to-Augment model checkpoints: `/mnt/parscratch/users/acp21rjf/l2augment_model/`
- Learn-to-Augment value model: `/mnt/parscratch/users/acp21rjf/l2augment_value/model.pt`
- Learn-to-Augment rollouts: `/mnt/parscratch/users/acp21rjf/l2augment_rollout*/`
- Spotify/ASR checkpoints: `/mnt/parscratch/users/acp21rjf/spotify/`

## Local persistent cache on mimas

Current repo-specific cache:

```text
/store/store5/data/acp21rjf_checkpoints/l2augment/
```

Currently transferred there:

- `asr/step_105360.pt` — canonical ASR backbone from `spotify/rotary_pos_6l_256d_seq_sched/n_seq_sched_2048_rp_1/step_105360.pt`
- `ufmr/` — UFMR variants from `l2augment_model/ufm/`
- `MANIFEST.md` — manifest for transferred files

## Previously missed non-UFMR checkpoint families

The original UFMR transfer did **not** include these Learn-to-Augment model families from `/mnt/parscratch/users/acp21rjf/l2augment_model/`:

- `autoenc_audio/`
- `bvae/`
- `cfm/` and `cfmsimple/`
- `cm/test/`
- `vae/` and `vae_tmp/`
- `MLM/`, `UMLM/`
- `CMultiStepMLM/`
- `multistep_FM_ranker/`
- `dt/test/`
- `ssvae/`
- top-level `model*.pt` files under `l2augment_model/`
- separate `l2augment_value/model.pt`

Configs also reference alternate Spotify/ASR checkpoints that were not pulled, including `rotary_pos_*`, `checkpoints_seq_scheduler_rb*`, and 3-layer/3-epoch variants.

## Inventory commands

List all model checkpoints on Stanage:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 acp21rjf@stanage.shef.ac.uk \
  'find /mnt/parscratch/users/acp21rjf/l2augment_model -type f \( -name "*.pt" -o -name "*.ckpt" -o -name "*.pth" \) -printf "%s %p\n" 2>/dev/null | sort -k2'
```

List referenced Spotify/ASR checkpoint paths from this repo:

```bash
python3 - <<'PY'
import re, pathlib
paths=set()
for p in pathlib.Path('.').rglob('*'):
    if p.is_file() and p.suffix in {'.yaml','.yml','.py','.md','.sh'}:
        txt=p.read_text(errors='ignore')
        paths.update(m.group(0).rstrip('`:,') for m in re.finditer(r'/mnt/parscratch/users/acp21rjf/spotify/[^\\s"\\',\\]\\)]+\\.pt`?', txt))
for s in sorted(paths): print(s)
PY
```

## Transfer patterns

Pull non-UFMR model checkpoints while preserving layout:

```bash
DEST=/store/store5/data/acp21rjf_checkpoints/l2augment
mkdir -p "$DEST/models"
rsync -ah --checksum --info=progress2 \
  --include='*/' --include='*.pt' --include='*.ckpt' --include='*.pth' --exclude='*' \
  --exclude='ufm/***' \
  acp21rjf@stanage.shef.ac.uk:/mnt/parscratch/users/acp21rjf/l2augment_model/ \
  "$DEST/models/"
```

Pull value model:

```bash
mkdir -p /store/store5/data/acp21rjf_checkpoints/l2augment/value
rsync -ah --checksum --info=progress2 \
  acp21rjf@stanage.shef.ac.uk:/mnt/parscratch/users/acp21rjf/l2augment_value/model.pt \
  /store/store5/data/acp21rjf_checkpoints/l2augment/value/
```

Do not pull rollout directories unless explicitly needed; they may be huge.

## Symphony callback credentials on Stanage

Stanage jobs launched for Symphony must keep the same Linear callback contract
as Mimas jobs. The Slurm wrapper should call
`scripts/callbacks/linear_experiment_callback.py` from an `EXIT` trap and move
the issue back to `Todo` when the job exits.

The template `scripts/templates/slurm_experiment_wrapper.template.sh` expects
`LINEAR_API_KEY` to be available through:

```text
~/.config/learning-to-augment/linear.env
```

Do not assume this file already exists on Stanage. When a Stanage run is
directly requested, provision it from Mimas without printing or logging the key:

```bash
test -n "${LINEAR_API_KEY:?LINEAR_API_KEY is required}"
printf 'LINEAR_API_KEY=%q\n' "${LINEAR_API_KEY}" | \
  ssh stanage.shef.ac.uk \
    'umask 077; mkdir -p ~/.config/learning-to-augment; cat > ~/.config/learning-to-augment/linear.env; chmod 600 ~/.config/learning-to-augment/linear.env'
```

Before submitting a long Stanage job, validate the callback path from the
Stanage checkout with a non-mutating check:

```bash
LINEAR_ENV_FILE=~/.config/learning-to-augment/linear.env bash -lc \
  'set -a; . "$LINEAR_ENV_FILE"; set +a; python3 scripts/callbacks/linear_experiment_callback.py --issue <issue> --status-code 0 --runner-label slurm:check --queued-command check-only --target-state Todo --check-only'
```

Do not put `LINEAR_API_KEY` in command arguments, Slurm scripts, logs, Git,
Linear comments, or shell history.
