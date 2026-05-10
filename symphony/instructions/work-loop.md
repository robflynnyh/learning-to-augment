# Work Loop

Inspect repo state and task context before editing.

Make a concise plan.

Identify validation for the specific change.

If the issue description includes `Branch/ref: <name>`, fetch and check out
that branch or ref before editing.

Confirm the checked-out commit with `git status`,
`git rev-parse --abbrev-ref HEAD`, and `git rev-parse HEAD`.

Create a working branch named `symphony/<issue-identifier>-<short-slug>` from
the checked-out base branch. Do not commit directly to the base branch.

Keep edits narrowly scoped to the issue.

Prefer existing experiment configs, launchers, result directories, and helper
functions over new abstractions.

Use structured parsers for structured data when reasonable.

Record exact commands, configs, checkpoints, output paths, and validation
outcomes for experiment or result changes.

During nontrivial work, periodically post concise Linear progress comments for
meaningful implementation progress, design decisions, experiment-launch
decisions, blockers, or validation changes.

Before each Linear progress or completion comment, re-fetch recent comments and
incorporate any new human reply first.

If a comparison is partial or still running, label it as a snapshot rather than
a final result.
