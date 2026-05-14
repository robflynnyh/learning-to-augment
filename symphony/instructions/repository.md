# Repository

`l2augment/` is the importable Python package containing policy models, rollout
functions, dataset helpers, and masking logic.

`exp/` is the experiment harness. The main entry points are `exp/train.py`,
`exp/train_freq_mask.py`, `exp/eval.py`, `exp/oracle_eval.py`,
`exp/generate.py`, and `exp/generate_search.py`.

`exp/configs/` contains active YAML configs. `exp/configs/configs_in_paper/`
contains paper-faithful reproduction configs.

`exp/launch_scripts/` contains cluster and local launch helpers. Read the
relevant launcher before changing a run or starting a long job.

`scripts/callbacks/` contains Linear callback helpers used by detached Symphony
experiment jobs. `scripts/templates/` contains wrapper templates for long Mimas
and Stanage runs.

`scripts/linear/` contains Linear helpers for issue-management tasks that need
to work from detached or scripted contexts without Codex-only tools.

`exp/results/` contains historical outputs, reproduction outputs, summaries,
plotting inputs, and result notes. Prefer programmatic aggregation or parsing
over hand-editing metrics.

For nontrivial new experiment families, create a small README under the relevant
experiment or result directory before launch, and create or update an `OUTCOME.md`
under the result directory when results are available. Keep names and paths
compatible with the existing `exp/results/` layout rather than forcing a new
top-level experiment tree.

`STANAGE.md` records Stanage paths and checkpoint transfer notes for this
repository. Treat remote and `/store/...` checkpoint/data locations as read-only
unless the issue explicitly says otherwise.

The repo depends on an installed `lcasr` environment plus Python packages used
by the experiment harness. Verify the active environment before launching or
validating long experiments.

Keep credentials, large checkpoints, raw audio, W&B output, and bulky temporary
files out of Git. On Mimas, never use `/tmp` for Symphony work, including
notes, scratch files, logs, downloads, generated artifacts, or experiment
intermediates; use a durable repo-local path, `exp/results/`, or an appropriate
`/store/...` path instead.

Commit small, meaningful result artifacts when they are part of the requested
deliverable and are reasonable for Git. For large generated artifacts, commit a
small index or summary recording the external path, size, generation command,
and why the artifact was not committed.

Keep Symphony-specific instructions and runtime config under `symphony/`.
`symphony/.env` is local-only and must not be committed.

Append concise dated entries to `RESEARCH_DIARY.md` for meaningful project
changes, experiment launches, completed runs, fixes, and interpretation updates.
Do not add repetitive launch bookkeeping that will not help a future agent.
