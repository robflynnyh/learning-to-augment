---
tracker:
  kind: linear
  project_slug: "learn-to-augment-b90747e63c8a"
  api_key: $LINEAR_API_KEY
  active_states:
    - Todo
    - In Progress
  terminal_states:
    - Closed
    - Cancelled
    - Canceled
    - Duplicate
    - Done
polling:
  interval_ms: 30000
workspace:
  root: /exp/exp4/acp21rjf/symphony-workspaces-learning-to-augment
hooks:
  timeout_ms: 120000
  after_create: |
    if [ -f /exp/exp4/acp21rjf/learning-to-augment/symphony/.env ]; then
      set -a
      . /exp/exp4/acp21rjf/learning-to-augment/symphony/.env
      set +a
    fi
    git clone --depth 1 "$SOURCE_REPO_URL" .
    if [ -n "${SOURCE_REF:-}" ]; then
      git fetch --depth 1 origin "$SOURCE_REF"
      git checkout -B symphony-source FETCH_HEAD
    fi
agent:
  max_concurrent_agents: 1
  max_turns: 30
codex:
  command: /home/acp21rjf/.npm-global/bin/codex --config shell_environment_policy.inherit=all app-server
  approval_policy: never
  thread_sandbox: danger-full-access
  turn_sandbox_policy:
    type: dangerFullAccess
---

You are working on Linear issue {{ issue.identifier }} for the
learning-to-augment repository.

Title: {{ issue.title }}
Current status: {{ issue.state }}
URL: {{ issue.url }}

Description:
{% if issue.description %}
{{ issue.description }}
{% else %}
No description provided.
{% endif %}

Linear discussion context:
- Before planning or editing, use the `linear_graphql` tool to fetch this
  issue's recent comments, newest last.
- Use this query shape with `{{ issue.id }}`:
  ```graphql
  query IssueComments($id: String!) {
    issue(id: $id) {
      comments(first: 20) {
        nodes {
          body
          createdAt
          user {
            name
          }
        }
      }
    }
  }
  ```
- Treat recent human comments as current task context, especially comments made
  after the latest completion, queue, or blocker comment.
- If a recent human comment asks a question or requests clarification rather
  than implementation, answer it in Linear first and do not move the issue to
  `Done`.
- If recent comments request rework on an existing PR or branch, inspect that
  PR or branch before editing.
- In your plan, explicitly state which recent comments changed or constrained
  the task.

Repository orientation:
- `l2augment/` is the importable Python package containing policy models,
  rollout functions, dataset helpers, and masking logic.
- `exp/` is the experiment harness. The main entry points are `exp/train.py`,
  `exp/train_freq_mask.py`, `exp/eval.py`, `exp/oracle_eval.py`,
  `exp/generate.py`, and `exp/generate_search.py`.
- `exp/configs/` contains active YAML configs. `exp/configs/configs_in_paper/`
  contains paper-faithful reproduction configs.
- `exp/launch_scripts/` contains cluster and local launch helpers. Read the
  relevant launcher before changing a run or starting a long job.
- `exp/results/` contains historical outputs, reproduction outputs, summaries,
  plotting inputs, and result notes. Prefer programmatic aggregation or parsing
  over hand-editing metrics.
- `STANAGE.md` records Stanage paths and checkpoint transfer notes for this
  repository. Treat remote and `/store/...` checkpoint/data locations as
  read-only unless the issue explicitly says otherwise.

Local configuration and artifacts:
- The repo depends on an installed `lcasr` environment plus Python packages
  used by the experiment harness. Verify the active environment before
  launching or validating long experiments.
- Keep credentials, large checkpoints, raw audio, W&B output, and bulky
  temporary files out of Git.
- Commit small, meaningful result artifacts when they are part of the requested
  deliverable and are reasonable for Git. For large generated artifacts, commit
  a small index or summary recording the external path, size, generation
  command, and why the artifact was not committed.
- Keep Symphony-specific instructions and runtime config under `symphony/`.
  `symphony/.env` is local-only and must not be committed.

Before editing:
- Inspect repo state and task context first.
- Make a concise plan.
- Identify validation for the specific change.
- If the issue description includes `Branch/ref: <name>`, fetch and check out
  that branch/ref before editing.
- Confirm the checked-out commit with `git status`,
  `git rev-parse --abbrev-ref HEAD`, and `git rev-parse HEAD`.
- Create a working branch named `symphony/{{ issue.identifier }}-<short-slug>`
  from the checked-out base branch. Do not commit directly to the base branch.

During work:
- Keep edits narrowly scoped to the issue.
- Prefer existing experiment configs, launchers, result directories, and helper
  functions over new abstractions.
- Use structured parsers for structured data when reasonable.
- Record exact commands, configs, checkpoints, output paths, and validation
  outcomes for experiment or result changes.
- During nontrivial work, periodically post concise Linear progress comments
  for meaningful implementation progress, design decisions, experiment-launch
  decisions, blockers, or validation changes.
- Before each Linear progress or completion comment, re-fetch recent comments
  and incorporate any new human reply first.
- If a comparison is partial or still running, label it as a snapshot rather
  than a final result.

Experiment launching:
- Do not launch long-running GPU work unless the issue asks for a run.
- Mimas is the default execution target for this project. Use Stanage only when
  the issue or a later human Linear comment directly asks for
  Stanage/HPC/Slurm. Do not infer Stanage from GPU size, queue length, or the
  agent's judgment that Mimas is not suitable.
- On Mimas, use `/store/store5/software/simple-gpu-schedule/with-gpu` for
  cooperative GPU allocation instead of manual polling. Prefer pool `1,2`
  unless the issue or experiment requires a different pool. Use a narrower pool
  only when the issue or an inspected launcher makes that need concrete.
- Launch long-running Mimas GPU experiments in durable detached `screen`
  sessions with log files. The detached command should run
  `with-gpu <pool> -- <experiment-wrapper>` so the queue waiter survives after
  the agent exits.
- For long CPU-only Mimas jobs, prefer a detached `screen` session with a log
  and the same callback discipline described below. Do not hide long CPU jobs
  inside the live Symphony turn.
- For Stanage work, follow `STANAGE.md` and the repo's existing
  `exp/launch_scripts/*.sh` Slurm patterns. Use short bounded SSH commands
  from Mimas, submit compute through Slurm, keep stdout/stderr and generated
  artifacts under a durable issue-specific path on `/mnt/parscratch/users/acp21rjf/`,
  and do not run meaningful compute on a Stanage login node.
- Do not spend agent turns waiting for a queued or running experiment to start
  or finish. After queueing a long experiment, post a Linear comment with the
  queued command, screen name or Slurm job ID, log path, expected result path,
  git branch and commit, callback/handoff path, and exact completion-check
  command. Then move the issue back to the Linear state named `Backlog`.
- Every queued long experiment must have a verified completion callback in the
  launched wrapper before it is queued. The callback must run when the
  experiment process exits for any reason, including success, nonzero exit,
  Python exception, shell error, timeout-wrapper exit, or manual termination
  where the shell can still run traps.
- Prefer an `EXIT` trap or equivalent wrapper-level hook that records the
  experiment exit status, then calls a real repo script that uses
  `LINEAR_API_KEY` to post a Linear comment with success or failure evidence,
  log path, output path, and residual risk. The callback should move the issue
  back to the Linear state named `Todo` so Symphony can resume finalization.
  Detached experiment processes cannot use Codex-only tools such as
  `linear_graphql`.
- Do not queue a long GPU or CPU experiment if the launched code lacks this
  completion callback. First add or fix the hook, then validate the actual
  wrapper `EXIT` trap with the smallest practical smoke test or callback-only
  dry run.
- When Symphony relaunches from a callback comment, inspect the log and
  results before deciding whether to finalize, diagnose, or rerun. If a run
  failed, fix the concrete issue before queueing another run. Do not blindly
  relaunch an unchanged failing command.

Validation:
- Run the most targeted command or test that demonstrates the task is complete.
- For documentation-only changes, run `git diff --check` and inspect the diff.
- For code changes, run the narrowest relevant script or test. Prefer small
  smoke configs before launching full experiments.
- If validation cannot run, document the exact blocker and the command that
  should be run later.

GitHub handoff:
- Commit completed changes on the issue branch.
- Push the branch to `origin`.
- Open a GitHub pull request using
  `/exp/exp4/acp21rjf/scripts/github-create-pr.sh`, using the issue
  `Branch/ref` as the PR base when provided, otherwise the repository default
  branch.
- Include the PR URL in the Linear completion comment.
- If pushing or PR creation fails, do not move the issue to `Done`; post a
  blocker comment with the exact failing command and error.

Linear handoff:
- Use the `linear_graphql` tool for Linear updates.
- Post one completion comment summarizing files changed, validation, output
  paths if any, GitHub PR URL, and residual risk.
- Move the issue to `Done` only when the requested work is complete and the
  GitHub handoff has succeeded.
