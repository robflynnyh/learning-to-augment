# Experiment Execution

Do not launch long-running GPU work unless the issue asks for a run.

Mimas is the default execution target for this project. Use Stanage only when
the issue or a later human Linear comment directly asks for Stanage, HPC, or
Slurm. Do not infer Stanage from GPU size, queue length, or the agent's judgment
that Mimas is not suitable.

On Mimas, use `/store/store5/software/simple-gpu-schedule/with-gpu` for
cooperative GPU allocation instead of manual polling. Prefer pool `1,2` unless
the issue or experiment requires a different pool. Use a narrower pool only when
the issue or an inspected launcher makes that need concrete.

Launch long-running Mimas GPU experiments in durable detached `screen` sessions
with log files. The detached command should run
`with-gpu <pool> -- <experiment-wrapper>` so the queue waiter survives after the
agent exits.

For long CPU-only Mimas jobs, prefer a detached `screen` session with a log and
the same callback discipline described below. Do not hide long CPU jobs inside
the live Symphony turn.

For Stanage work, follow `STANAGE.md` and the repo's existing
`exp/launch_scripts/*.sh` Slurm patterns. Use short bounded SSH commands from
Mimas, submit compute through Slurm, keep stdout/stderr and generated artifacts
under a durable issue-specific path on `/mnt/parscratch/users/acp21rjf/`, and
do not run meaningful compute on a Stanage login node.

Do not spend agent turns waiting for a queued or running experiment to start or
finish. After queueing a long experiment, post a Linear comment with the queued
command, screen name or Slurm job ID, log path, expected result path, git branch
and commit, callback/handoff path, and exact completion-check command. Then move
the issue back to the Linear state named `Backlog`.

Every queued long experiment must have a verified completion callback in the
launched wrapper before it is queued. The callback must run when the experiment
process exits for any reason, including success, nonzero exit, Python exception,
shell error, timeout-wrapper exit, or manual termination where the shell can
still run traps.

Prefer an `EXIT` trap or equivalent wrapper-level hook that records the
experiment exit status, then calls `scripts/callbacks/linear_experiment_callback.py`.
The callback uses `LINEAR_API_KEY` to post a Linear comment with success or
failure evidence, log path, output path, and residual risk. It should move the
issue back to the Linear state named `Todo` so Symphony can resume finalization.
Detached experiment processes cannot use Codex-only tools such as
`linear_graphql`.

The wrapper should pass at least `--issue`, `--status-code`, `--log`,
`--results`, `--runner-label`, `--queued-command`, `--branch`, and `--commit` to
the callback when those values are known.

Use `scripts/templates/queued_experiment_wrapper.template.sh` as the starting
point for Mimas launch wrappers. Use
`scripts/templates/slurm_experiment_wrapper.template.sh` as the starting point
for Stanage Slurm launch wrappers. Keep the `EXIT` trap intact.

Do not queue a long GPU or CPU experiment if the launched code lacks this
completion callback. First add or fix the hook, then validate the actual wrapper
`EXIT` trap with the smallest practical smoke test or callback-only dry run.

When Symphony relaunches from a callback comment, inspect the log and results
before deciding whether to finalize, diagnose, or rerun. If a run failed, fix
the concrete issue before queueing another run. Do not blindly relaunch an
unchanged failing command.
