# Claude Agent Loop

An outer-loop orchestrator that coordinates two Claude Code instances in a **Planner/Implementer** cycle to autonomously accomplish complex software goals.

```
python3 claude_loop.py ./my-project "Build a REST API with auth" --effort max
```

## How It Works

The orchestrator runs a feedback loop between two specialized Claude Code roles:

- **Planner** (tech lead): Analyzes the project, creates step-by-step implementation plans, reviews results, and decides when the goal is complete.
- **Implementer** (engineer): Receives a plan, writes code, runs tests, commits changes.

Both roles use your Claude subscription via the CLI -- no API key required.

## Architecture

```
                    +------------------+
                    |   User defines   |
                    |  project + goal  |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |   Orchestrator   |
                    |   (Python CLI)   |
                    +--------+---------+
                             |
               +-------------+-------------+
               |                           |
               v                           v
      +-----------------+        +------------------+
      |     Planner     |        |   Implementer    |
      | (claude --print |        | (claude --print  |
      |  --system-prompt|        |  default prompt) |
      |  "tech lead")   |        |                  |
      +-----------------+        +------------------+
               |                           |
               |    Plan (numbered steps)  |
               +----------->--------------+
               |                           |
               |   Output + git diff       |
               +-----------<--------------+
               |                           |
               v                           |
      +------------------+                 |
      | "GOAL COMPLETE"? |----no---------->+
      +--------+---------+
               | yes
               v
            [Done]
```

## Detailed Flow

```
Iteration N:
+------------------------------------------------------------------+
|                                                                  |
|  1. GATHER CONTEXT                                               |
|     - File listing (excludes .git, node_modules, etc.)           |
|     - Git log (last 10 commits)                                  |
|     - README contents                                            |
|     - Previous implementation output + git diff (iterations 2+)  |
|                                                                  |
|  2. PLANNER (claude --print --system-prompt "tech lead...")      |
|     Input:  project context + goal (or previous results)         |
|     Output: numbered implementation plan                         |
|             OR "GOAL COMPLETE" to stop the loop                  |
|                                                                  |
|  3. IMPLEMENTER (claude --print)                                 |
|     Input:  the plan from step 2                                 |
|     Output: code changes, test results, commit                   |
|                                                                  |
|  4. CAPTURE RESULTS                                              |
|     - Implementation stdout/stderr                               |
|     - git diff HEAD~1                                            |
|     - git log                                                    |
|                                                                  |
|  5. LOG EVERYTHING                                               |
|     - All inputs/outputs saved to ~/.claude-loop/logs/<run_id>/  |
|                                                                  |
+------------------------------------------------------------------+
          |
          v
     Next iteration (back to step 1 with new context)
```

## Installation

No dependencies beyond Python 3.7+ and [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) CLI.

```bash
git clone https://github.com/josephgec/claude-agent.git
cd claude-agent

# Verify claude CLI is available
claude --version
```

## Usage

```bash
# Basic usage
python3 claude_loop.py <project_dir> "<goal>"

# Full options
python3 claude_loop.py <project_dir> "<goal>" \
    --max-iterations 10 \
    --planner-model opus \
    --impl-model sonnet \
    --effort max \
    --skip-permissions \
    --impl-timeout 600
```

### Arguments

| Argument | Description |
|---|---|
| `project_dir` | Path to the project directory |
| `goal` | The goal/task to accomplish |

### Options

| Option | Default | Description |
|---|---|---|
| `--max-iterations` | `10` | Maximum number of plan/implement cycles |
| `--planner-model` | `opus` | Model for the planner role |
| `--impl-model` | `sonnet` | Model for the implementer role |
| `--effort` | model default | Effort level: `low`, `medium`, `high`, `max` |
| `--impl-timeout` | `600` | Timeout in seconds per implementation round |
| `--skip-permissions` | off | Skip permission checks (sandboxed environments only) |
| `--no-auto-commit` | off | Leave changes for manual review instead of committing |
| `--dry-run` | off | Print the planner prompt and exit without calling Claude |
| `--resume [RUN_ID]` | off | Resume a previous run. Omit `RUN_ID` to resume the most recent run. |

### Examples

```bash
# Quick bug fix
python3 claude_loop.py ./my-app "Fix the login page 500 error" --max-iterations 3

# Full feature build with max effort
python3 claude_loop.py ./my-app "Add JWT authentication with refresh tokens" \
    --effort max --planner-model opus --impl-model opus

# Refactoring with fast iterations
python3 claude_loop.py ./my-app "Convert all JavaScript files to TypeScript" \
    --planner-model sonnet --impl-model sonnet --max-iterations 15

# Automated test improvement
python3 claude_loop.py ./my-app "Achieve 90% test coverage" --skip-permissions

# Resume after a crash (or after --resume'ing through a rate-limit wait)
python3 claude_loop.py --resume                      # pick up the most recent run
python3 claude_loop.py --resume 20260115_120000      # resume a specific run ID
```

## Logs

Every run is logged to `~/.claude-loop/logs/<timestamp>/` with one file per phase:

```
~/.claude-loop/logs/20260411_103515/
  iter001_planner_input.md    # What was sent to the planner
  iter001_planner_output.md   # The plan
  iter001_impl_input.md       # What was sent to the implementer
  iter001_impl_output.md      # Implementation results
  iter002_planner_input.md
  iter002_planner_output.md
  iter002_COMPLETE.md         # Final completion summary
  run_state.json              # Resumable state snapshot (see "Rate Limits & Resume")
  summary.md                  # End-of-run summary: status, iterations, duration, diff stat
```

## Rate Limits & Resume

The orchestrator is designed to run unattended through long sessions, including rate-limit waits.

**Automatic rate-limit handling.** When the Claude CLI returns a usage-limit, quota, throttle,
overloaded, or 429 response, the orchestrator:

1. Parses the reset time from the error message. It understands many phrasings, including
   `resets in 45 minutes`, `try again in 30 seconds`, `retry after 2 minutes`, `please wait 5 minutes`,
   `resets 2:30pm`, `resets at 14:30`, and bare `Retry-After: 60` HTTP-style headers.
2. Sleeps until the reset time with a live in-place countdown and 2-second poll chunks so
   `Ctrl+C` remains responsive.
3. Retries automatically. Error backoff (30s → 60s → … capped at 300s) is reset after a
   clean rate-limit recovery so transient errors do not inflate the wait of future retries.

If the message can't be parsed the loop defaults to a 15-minute wait, capped at 4 hours.

**State persistence & crash recovery.** Every iteration writes a `run_state.json` under
`~/.claude-loop/logs/<run_id>/` before each planner call, before each implementer call, and
after each completed iteration. State is also flushed to disk just before any rate-limit sleep.
This means you can kill the process (or lose the machine) during a multi-hour wait and recover
with:

```bash
python3 claude_loop.py --resume                 # resume the most recent run
python3 claude_loop.py --resume <run_id>        # resume a specific run by log-directory name
```

On resume, the orchestrator reloads project path, goal, models, effort, timeout, and the
last completed implementation output from `run_state.json`. It reuses the existing log
directory and picks up at the next iteration if the previous one completed, or re-runs the
current iteration from the planner if it was interrupted mid-phase.

## Run Lifecycle

- **Goal-completion detection** is line-anchored: a line that starts with `GOAL COMPLETE`
  ends the loop. Substrings inside prose (e.g., "we are not yet GOAL COMPLETE") do not
  trigger a false positive.
- **Graceful Ctrl+C.** Hitting `Ctrl+C` at any point — including during a rate-limit
  countdown — prints a clean interrupt message and exits with status 130. The run's
  `run_state.json` is already on disk from the last phase boundary, so you can pick up
  with `--resume`.
- **Non-git projects.** If the project directory isn't a git repository, the orchestrator
  still runs, but prints a warning: diffs, commit tracking, and "recent git log" context
  are all skipped. Run `git init` in the project to get the richer context back.
- **End-of-run summary.** Every completed (or max-iteration-exhausted) run writes a
  `summary.md` to its log directory with the goal, status, duration, per-iteration
  summaries, and — when git is available — an overall `git diff --stat` from the starting
  HEAD to the final HEAD.

## Goal Completion

The loop terminates when:
1. The planner outputs a line starting with `GOAL COMPLETE` (it reviewed the implementation
   and is satisfied)
2. `--max-iterations` is reached
3. The process is interrupted with `Ctrl+C` (state is preserved — resume with `--resume`)

## How the Planner Decides

The planner receives a system prompt that instructs it to act as a senior tech lead:
- **Read before planning.** The planner is told to open the files it intends to change or
  reference before writing the plan — it doesn't plan blind from the file listing alone.
  (The implementer already reads and writes files as part of its normal workflow, so no
  extra nudge is needed there.)
- Create specific, actionable plans with file paths and change descriptions
- Review implementation output for errors, missed requirements, and bugs
- Only declare `GOAL COMPLETE` when the overall goal is fully achieved

## License

MIT
