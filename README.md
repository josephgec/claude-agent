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
  iter002_COMPLETE.md          # Final completion summary
```

## Goal Completion

The loop terminates when:
1. The planner responds with `GOAL COMPLETE` (it reviewed the implementation and is satisfied)
2. `--max-iterations` is reached
3. The process is interrupted

## How the Planner Decides

The planner receives a system prompt that instructs it to act as a senior tech lead:
- Create specific, actionable plans with file paths and change descriptions
- Review implementation output for errors, missed requirements, and bugs
- Only declare `GOAL COMPLETE` when the overall goal is fully achieved

## License

MIT
