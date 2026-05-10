# Scripts

This directory contains reusable helpers for Symphony-managed work.

## Layout

- `callbacks/`: Linear callback helpers used by detached Mimas or Stanage jobs.
- `linear/`: Linear issue-management helpers that work without Codex-only tools.
- `templates/`: wrapper templates that should be copied to an issue-specific
  path before launching long jobs.

## Detached Experiment Callback

Use `scripts/callbacks/linear_experiment_callback.py` from long-running wrappers
to report completion back to Linear. The helper uses `LINEAR_API_KEY`, posts a
bounded completion comment, and moves the issue back to `Todo` so Symphony can
resume finalization.

Dry-run a callback comment without touching Linear:

```bash
python3 scripts/callbacks/linear_experiment_callback.py \
  --dry-run \
  --issue ROB-000 \
  --status-code 0 \
  --log results/example/logs/run.log \
  --results results/example \
  --runner-label screen:l2a_example_run \
  --queued-command '/store/store5/software/simple-gpu-schedule/with-gpu 1,2 -- bash scripts/archive/rob000/run_example.sh'
```

Validate Linear access and target-state lookup without mutating the issue:

```bash
python3 scripts/callbacks/linear_experiment_callback.py \
  --check-only \
  --issue ROB-000 \
  --status-code 0 \
  --target-state Todo
```

Use `scripts/templates/queued_experiment_wrapper.template.sh` as a starting
point for Mimas screen jobs. Use
`scripts/templates/slurm_experiment_wrapper.template.sh` as a starting point for
Stanage Slurm jobs. Keep the `EXIT` trap intact.

Before queueing a long job, smoke test the exact wrapper or Slurm finalizer that
will be launched. A helper-level `--check-only` proves Linear access; it does
not prove that the real shell trap, environment, log path, and callback
arguments work together.

## Blocked Follow-Up Issues

Use `scripts/linear/create_blocked_issue.py` when Symphony finds a concrete,
actionable problem that is outside the current issue scope.

```bash
python3 scripts/linear/create_blocked_issue.py \
  --blocked-by ROB-000 \
  --title "Fix separate issue found during ROB-000" \
  --description-file /tmp/followup.md
```

The helper creates a new issue in the same team/project as the blocking issue
and links the current issue as blocking the new follow-up.
