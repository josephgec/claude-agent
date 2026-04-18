#!/usr/bin/env python3 -u
"""
Claude Loop Orchestrator
========================
An outer-loop agent that coordinates between two Claude Code instances:
  - Planner: A "tech lead" that reviews work and creates implementation plans
  - Implementer: Claude Code that executes the plans

Both use your Claude subscription via the CLI — no API key needed.

Usage:
    python3 claude_loop.py /path/to/project "Build a REST API with auth" --max-iterations 5
    python3 claude_loop.py /path/to/project "Fix all failing tests" --planner-model opus --impl-model sonnet
"""

from __future__ import annotations

import re
import subprocess
import sys
import os
import time
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Force unbuffered stdout for background execution
os.environ["PYTHONUNBUFFERED"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

LOGS_DIR = os.path.expanduser("~/.claude-loop/logs")
MAX_OUTPUT_CHARS = 80000  # truncate outputs sent to planner if longer
STATE_FILE = "run_state.json"

# Patterns that indicate the response is an error, not real content
RATE_LIMIT_PATTERNS = [
    r"You've hit your limit",
    r"you've hit your limit",
    r"hit your limit",
    r"Usage limit",
    r"usage limit",
    r"rate limit",
    r"Rate limit",
    r"too many requests",
    r"Too many requests",
    r"(?:error|status|code)\s*[:=]?\s*429",
    r"overloaded",
    r"Overloaded",
    r"at capacity",
    r"quota exceeded",
    r"Quota exceeded",
    r"please try again later",
    r"Please try again later",
    r"limit reached",
    r"Limit reached",
    r"throttled",
    r"Throttled",
    r"maximum.*requests",
    r"cooldown",
]

TIMEOUT_PATTERN = r"\(Claude Code timed out after \d+s\)"

ERROR_PATTERNS = [
    r"\(Error running Claude Code:",
    r"permission.*denied",
    r"ECONNREFUSED",
    r"network error",
    r"(?:error|status|code)\s*[:=]?\s*50[023]",
]

# Minimum chars for a response to be considered "real" content
MIN_VALID_RESPONSE_CHARS = 100

# Key project files to look for in structure-aware context
KEY_PROJECT_FILES = [
    "package.json",
    "pyproject.toml",
    "Cargo.toml",
    "go.mod",
    "Makefile",
    "Dockerfile",
    "docker-compose.yml",
    "requirements.txt",
    "setup.py",
    ".env.example",
    "CLAUDE.md",
]

PLANNER_SYSTEM_PROMPT = """\
You are a senior tech lead and software architect. Your job is to:
1. Analyze the current state of a project
2. Create clear, actionable implementation plans
3. Review implementation results and plan next steps

RULES:
- Be specific: name exact files, functions, and line-level changes
- Break work into small, sequential steps that can be executed one at a time
- Each plan should be completable in a single implementation round
- When reviewing results, check for errors, missed requirements, and bugs
- When the overall goal is fully achieved, respond with exactly "GOAL COMPLETE" on its own line, followed by a summary

OUTPUT FORMAT:
Respond with a numbered plan. Each step should specify:
- What file to create/modify
- What changes to make
- Why (brief rationale)
"""

IMPLEMENTER_APPEND_PROMPT = """\
After making all changes, commit your work with a descriptive commit message.
If there are tests, run them and fix any failures before committing.
At the end, briefly summarize what you did and any issues encountered.
"""

NO_AUTO_COMMIT_APPEND_PROMPT = """\
Do NOT commit. Leave changes staged or unstaged for manual review.
At the end, briefly summarize what you did.
"""


def truncate(text: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    """Truncate text with a note if it exceeds max_chars."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n... (truncated, {len(text)} total chars)"


def extract_summary(text: str, max_sentences: int = 3) -> str:
    """Extract the first 2-3 sentences from text as a summary."""
    if not text:
        return ""
    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    selected = sentences[:max_sentences]
    summary = " ".join(selected)
    # Cap at 500 chars
    if len(summary) > 500:
        summary = summary[:497] + "..."
    return summary


def save_state(log_dir: str, state_dict: Dict) -> None:
    """Persist run state to JSON for crash recovery.

    Wrapped in try/except so write failures never crash the loop.
    """
    try:
        state_path = os.path.join(log_dir, STATE_FILE)
        data = dict(state_dict)
        data["saved_at"] = datetime.now().isoformat()
        with open(state_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"    [!] Failed to save state: {e}")


def load_state(run_id: str) -> Optional[Dict]:
    """Read run_state.json for a given run_id. Returns None if missing."""
    state_path = os.path.join(LOGS_DIR, run_id, STATE_FILE)
    if not os.path.isfile(state_path):
        return None
    try:
        with open(state_path) as f:
            return json.load(f)
    except Exception:
        return None


def find_latest_run() -> Optional[str]:
    """Return the run_id of the most recent run that has a run_state.json."""
    if not os.path.isdir(LOGS_DIR):
        return None
    candidates = []
    for name in os.listdir(LOGS_DIR):
        path = os.path.join(LOGS_DIR, name)
        state_file = os.path.join(path, STATE_FILE)
        if os.path.isdir(path) and os.path.isfile(state_file):
            candidates.append((os.path.getmtime(state_file), name))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def format_progress_section(progress_log: List[Dict]) -> str:
    """Format progress log entries into a section for the planner prompt.

    Caps at ~4000 chars. If exceeded, keeps first and last entries with
    a note about omitted iterations in between.
    """
    if not progress_log:
        return ""

    def format_entry(entry: Dict) -> str:
        lines = [f"  Iteration {entry['iteration']}:"]
        if entry.get("plan_summary"):
            lines.append(f"    Plan: {entry['plan_summary']}")
        if entry.get("impl_summary"):
            lines.append(f"    Impl: {entry['impl_summary']}")
        if entry.get("commit_hash"):
            lines.append(f"    Commit: {entry['commit_hash']}")
        if entry.get("head_before"):
            lines.append(f"    HEAD before: {entry['head_before'][:8]}")
        if entry.get("head_after"):
            lines.append(f"    HEAD after: {entry['head_after'][:8]}")
        return "\n".join(lines)

    # Try formatting all entries
    entries = [format_entry(e) for e in progress_log]
    full_text = "\n".join(entries)

    if len(full_text) <= 4000:
        return f"=== PROGRESS SO FAR ===\n{full_text}"

    # Keep first and last, summarize middle
    first = entries[0]
    last = entries[-1]
    omitted = len(entries) - 2
    middle = f"  ({omitted} iterations omitted)"
    return f"=== PROGRESS SO FAR ===\n{first}\n{middle}\n{last}"


def parse_reset_time(text: str) -> Optional[int]:
    """Parse a rate limit message to find seconds until reset.

    Handles formats like:
      "resets 1pm (America/Los_Angeles)"
      "resets 2:30pm"
      "resets at 14:30"
      "resets in 45 minutes" / "try again in 30 seconds" / "retry after 2 minutes"
      "resets in 1 hour"
      "retry-after: 60"
    Returns seconds to wait, or None if unparseable.
    """
    # "resets in N", "try again in N", "retry after N", "wait N", "available in N"
    # with seconds / minutes / hours
    m = re.search(
        r"(?:resets?\s+in|try\s+again\s+in|retry\s+after|wait|available\s+in)"
        r"\s+(\d+)\s*(second|sec|minute|min|hour|hr)",
        text,
        re.IGNORECASE,
    )
    if m:
        amount = int(m.group(1))
        unit = m.group(2).lower()
        if unit.startswith("hour") or unit.startswith("hr"):
            return amount * 3600 + 60  # extra minute buffer
        if unit.startswith("min"):
            return amount * 60 + 60
        # seconds
        return amount + 60

    # "resets Xam/pm" or "resets X:XXam/pm"
    m = re.search(r"resets?\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)", text, re.IGNORECASE)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2)) if m.group(2) else 0
        ampm = m.group(3).lower()
        if ampm == "pm" and hour != 12:
            hour += 12
        if ampm == "am" and hour == 12:
            hour = 0

        now = datetime.now()
        reset_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if reset_time <= now:
            reset_time += timedelta(days=1)

        wait = (reset_time - now).total_seconds()
        return int(wait) + 60  # extra minute buffer

    # "resets at HH:MM" (24-hour, no am/pm)
    m = re.search(r"resets?\s+at\s+(\d{1,2}):(\d{2})(?!\s*(?:am|pm))", text, re.IGNORECASE)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            now = datetime.now()
            reset_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if reset_time <= now:
                reset_time += timedelta(days=1)
            wait = (reset_time - now).total_seconds()
            return int(wait) + 60

    # "retry-after: N" (bare header-style, treat as seconds)
    m = re.search(r"retry[-\s]after\s*[:=]\s*(\d+)", text, re.IGNORECASE)
    if m:
        return int(m.group(1)) + 60

    return None


def classify_response(stdout: str, stderr: str = "", returncode: int = 0) -> str:
    """Classify a Claude response as 'ok', 'rate_limit', 'timeout', or 'error'.

    Uses returncode as primary signal, then scans stderr and first/last 500
    chars of stdout for patterns.

    Returns one of:
      'rate_limit' -- usage cap hit, should wait for reset
      'timeout'    -- claude timed out, worth retrying
      'error'      -- other error, worth retrying with backoff
      'ok'         -- valid response
    """
    # Non-zero returncode is a strong error signal
    if returncode != 0:
        # Check stderr + bounded stdout for specific patterns
        scan_text = stderr + " " + stdout[:500] + " " + stdout[-500:] if len(stdout) > 1000 else stderr + " " + stdout

        for pattern in RATE_LIMIT_PATTERNS:
            if re.search(pattern, scan_text, re.IGNORECASE):
                return "rate_limit"

        if re.search(TIMEOUT_PATTERN, scan_text):
            return "timeout"

        for pattern in ERROR_PATTERNS:
            if re.search(pattern, scan_text, re.IGNORECASE):
                return "error"

        return "error"

    # Returncode 0 but empty/whitespace-only output
    if not stdout or not stdout.strip():
        return "error"

    stripped = stdout.strip()

    # Long responses (>500 chars) with returncode 0 are almost certainly real content
    is_short = len(stripped) < 500

    if is_short:
        for pattern in RATE_LIMIT_PATTERNS:
            if re.search(pattern, stdout, re.IGNORECASE):
                return "rate_limit"

        if re.search(TIMEOUT_PATTERN, stdout):
            return "timeout"

        for pattern in ERROR_PATTERNS:
            if re.search(pattern, stdout, re.IGNORECASE):
                return "error"

        # Very short responses are suspicious -- but "GOAL COMPLETE" is short and valid
        if len(stripped) < MIN_VALID_RESPONSE_CHARS and "GOAL COMPLETE" not in stdout.upper():
            return "error"

    return "ok"


def format_duration(seconds: int) -> str:
    """Format a duration as Xs, Xm Ys, or Xh Ym."""
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{m}m {s}s"
    h, rem = divmod(seconds, 3600)
    m = rem // 60
    return f"{h}h {m}m"


def wait_with_countdown(wait_seconds: int, phase: str) -> None:
    """Sleep for wait_seconds, updating a single-line countdown every 30s.

    Uses 2-second sleep chunks so Ctrl+C is responsive.
    """
    resume_at = datetime.now() + timedelta(seconds=wait_seconds)
    print(f"    Waiting {format_duration(wait_seconds)} until {resume_at.strftime('%H:%M:%S')}...")
    print(f"    (Ctrl+C to abort)")

    remaining = wait_seconds
    last_update = None  # remaining value at which we last redrew the line
    while remaining > 0:
        if last_update is None or (last_update - remaining) >= 30 or remaining < 30:
            sys.stdout.write(f"\r    {format_duration(remaining)} remaining...    ")
            sys.stdout.flush()
            last_update = remaining
        chunk = min(remaining, 2)
        time.sleep(chunk)
        remaining -= chunk

    # Clear the countdown line
    sys.stdout.write("\r" + " " * 60 + "\r")
    sys.stdout.flush()
    print(f"    Wait complete. Resuming {phase}...\n")


def wait_for_reset(text: str, phase: str) -> None:
    """Parse rate limit reset time and sleep until then."""
    wait_seconds = parse_reset_time(text)

    if wait_seconds is None:
        # Can't parse reset time -- default to 15 minutes
        wait_seconds = 900
        print(f"    Could not parse reset time. Defaulting to {format_duration(wait_seconds)} wait.")
    else:
        print(f"    Parsed reset time from message.")

    # Cap at 4 hours
    wait_seconds = min(wait_seconds, 4 * 3600)

    wait_with_countdown(wait_seconds, phase)


def run_claude_with_retry(
    prompt: str,
    cwd: str,
    phase: str,
    model: str = "opus",
    system_prompt: Optional[str] = None,
    append_system_prompt: Optional[str] = None,
    timeout: int = 600,
    skip_permissions: bool = False,
    effort: Optional[str] = None,
    max_retries: int = 3,
    state_save_callback: Optional[Callable[[], None]] = None,
) -> str:
    """Run Claude Code with automatic retry on rate limits and errors.

    On rate limit: parses reset time, sleeps until then, retries.
    On timeout/error: retries up to max_retries with exponential backoff.
    If state_save_callback is provided, it's invoked just before entering a
    rate-limit wait so state is persisted to disk before any long sleep.
    Returns the stdout string to callers.
    """
    error_retries = 0
    backoff = 30  # initial backoff seconds for non-rate-limit errors

    while True:
        stdout, stderr, returncode = run_claude(
            prompt=prompt,
            cwd=cwd,
            model=model,
            system_prompt=system_prompt,
            append_system_prompt=append_system_prompt,
            timeout=timeout,
            skip_permissions=skip_permissions,
            effort=effort,
        )

        status = classify_response(stdout, stderr, returncode)

        if status == "ok":
            return stdout

        # Combine stdout and stderr for messages/parsing
        combined = stdout
        if stderr:
            combined += "\n" + stderr

        if status == "rate_limit":
            print(f"    [!] Rate limit hit during {phase}: {combined.strip()[:200]}")
            if state_save_callback is not None:
                try:
                    state_save_callback()
                except Exception as e:
                    print(f"    [!] state_save_callback failed: {e}")
            wait_for_reset(combined, phase)
            error_retries = 0  # reset error counter after rate limit wait
            backoff = 30       # reset backoff too after clean rate-limit recovery
            continue

        if status == "timeout":
            error_retries += 1
            if error_retries > max_retries:
                print(f"    [!] {phase} timed out {max_retries} times. Returning last output.")
                return stdout
            print(f"    [!] {phase} timed out. Retry {error_retries}/{max_retries} in {backoff}s...")
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)
            continue

        if status == "error":
            error_retries += 1
            if error_retries > max_retries:
                print(f"    [!] {phase} failed {max_retries} times. Returning last output.")
                return stdout
            print(f"    [!] {phase} error ({len(stdout)} chars): {combined.strip()[:120]}")
            print(f"    Retry {error_retries}/{max_retries} in {backoff}s...")
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)
            continue


def run_claude(
    prompt: str,
    cwd: str,
    model: str = "opus",
    system_prompt: Optional[str] = None,
    append_system_prompt: Optional[str] = None,
    timeout: int = 600,
    skip_permissions: bool = False,
    effort: Optional[str] = None,
) -> Tuple[str, str, int]:
    """Run Claude Code CLI in print mode and return (stdout, stderr, returncode)."""
    cmd = ["claude", "--print", "--model", model]

    if effort:
        cmd.extend(["--effort", effort])
    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])
    if append_system_prompt:
        cmd.extend(["--append-system-prompt", append_system_prompt])
    if skip_permissions:
        cmd.append("--dangerously-skip-permissions")

    cmd.append(prompt)

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return (result.stdout, result.stderr, result.returncode)
    except subprocess.TimeoutExpired:
        return (f"(Claude Code timed out after {timeout}s)", "", -1)
    except FileNotFoundError:
        print("Error: 'claude' command not found. Is Claude Code installed?")
        sys.exit(1)
    except Exception as e:
        return (f"(Error running Claude Code: {e})", "", -1)


def git_cmd(args: List[str], cwd: str) -> str:
    """Run a git command and return stdout."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def get_project_context(project_dir: str) -> str:
    """Gather project context: key files, file tree, git log, README, etc."""
    parts = []

    # Key project files (structure-aware context)
    key_file_parts = []
    for filename in KEY_PROJECT_FILES:
        filepath = os.path.join(project_dir, filename)
        if os.path.isfile(filepath):
            try:
                with open(filepath) as f:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= 20:
                            break
                        lines.append(line)
                    content = "".join(lines)
                key_file_parts.append(f"--- {filename} (first 20 lines) ---\n{content}")
            except Exception:
                pass

    if key_file_parts:
        parts.append(f"=== PROJECT STRUCTURE ===\n" + "\n".join(key_file_parts))

    # File listing (exclude common noise)
    try:
        result = subprocess.run(
            [
                "find", ".", "-type", "f",
                "-not", "-path", "./.git/*",
                "-not", "-path", "./node_modules/*",
                "-not", "-path", "./__pycache__/*",
                "-not", "-path", "./.venv/*",
                "-not", "-path", "./venv/*",
                "-not", "-path", "./dist/*",
                "-not", "-path", "./build/*",
                "-not", "-path", "./.next/*",
                "-not", "-name", "*.pyc",
                "-not", "-name", ".DS_Store",
            ],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Pipe through sort for consistent ordering
        raw_tree = result.stdout.strip()
        sorted_lines = sorted(raw_tree.split("\n")) if raw_tree else []
        tree = "\n".join(sorted_lines)
        if len(tree) > 3000:
            tree = tree[:3000] + "\n... (truncated)"
        parts.append(f"=== FILE LISTING ===\n{tree}")
    except Exception:
        parts.append("=== FILE LISTING ===\n(could not list files)")

    # Git log
    log = git_cmd(["log", "-10", "--oneline"], project_dir)
    if log:
        parts.append(f"=== RECENT GIT LOG ===\n{log}")

    # README
    for readme in ["README.md", "README.txt", "README"]:
        readme_path = os.path.join(project_dir, readme)
        if os.path.isfile(readme_path):
            try:
                with open(readme_path) as f:
                    content = f.read(3000)
                parts.append(f"=== {readme} ===\n{content}")
            except Exception:
                pass
            break

    return "\n\n".join(parts)


def log_to_file(log_dir: str, iteration: int, phase: str, content: str):
    """Save iteration data to a log file."""
    filename = f"iter{iteration:03d}_{phase}.md"
    filepath = os.path.join(log_dir, filename)
    with open(filepath, "w") as f:
        f.write(content)


def print_header(
    project_dir: str,
    goal: str,
    max_iterations: int,
    log_dir: str,
    resume_from: Optional[int] = None,
):
    print(f"\n{'=' * 60}")
    print(f"  Claude Loop Orchestrator")
    print(f"{'=' * 60}")
    print(f"  Project:        {project_dir}")
    print(f"  Goal:           {goal}")
    print(f"  Max iterations: {max_iterations}")
    print(f"  Logs:           {log_dir}")
    if resume_from is not None:
        print(f"  Resuming from:  iteration {resume_from}")
    print(f"{'=' * 60}\n")


def run_loop(
    project_dir: str,
    goal: str,
    max_iterations: int = 10,
    planner_model: str = "opus",
    impl_model: str = "sonnet",
    impl_timeout: int = 600,
    skip_permissions: bool = False,
    effort: Optional[str] = None,
    auto_commit: bool = True,
    dry_run: bool = False,
    resume_from_iteration: Optional[int] = None,
    resume_log_dir: Optional[str] = None,
    resume_impl_output: Optional[str] = None,
):
    """Main orchestration loop."""
    progress_log = []  # type: List[Dict]

    if resume_log_dir:
        log_dir = resume_log_dir
        start_iteration = resume_from_iteration or 1
        implementation_output = resume_impl_output
        # The most recent completed impl file predates the iteration we restart on
        last_impl_file = (
            f"iter{start_iteration - 1:03d}_impl_output.md" if start_iteration > 1 else None
        )
        print(f"[*] Resuming from iteration {start_iteration} (run {os.path.basename(log_dir)})")
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(LOGS_DIR, run_id)
        start_iteration = 1
        implementation_output = None
        last_impl_file = None

    os.makedirs(log_dir, exist_ok=True)

    print_header(project_dir, goal, max_iterations, log_dir, resume_from=resume_from_iteration)

    # Gather initial project context
    print("[*] Gathering project context...")
    project_context = get_project_context(project_dir)

    # Determine the append prompt for implementer
    impl_append = NO_AUTO_COMMIT_APPEND_PROMPT if not auto_commit else IMPLEMENTER_APPEND_PROMPT

    def build_state(phase: str, iteration: int, status: str = "running") -> Dict:
        return {
            "project_dir": project_dir,
            "goal": goal,
            "max_iterations": max_iterations,
            "planner_model": planner_model,
            "impl_model": impl_model,
            "impl_timeout": impl_timeout,
            "skip_permissions": skip_permissions,
            "effort": effort,
            "current_iteration": iteration,
            "current_phase": phase,
            "status": status,
            "last_impl_output_file": last_impl_file,
        }

    # Dry run: print planner prompt and exit
    if dry_run:
        planner_input = (
            f"PROJECT: {os.path.basename(project_dir)}\n\n"
            f"{project_context}\n\n"
            f"GOAL: {goal}\n\n"
            f"Create a detailed implementation plan for this goal. "
            f"Be specific about files and changes."
        )
        char_count = len(planner_input)
        token_estimate = char_count // 4  # rough estimate
        print(f"\n{'=' * 60}")
        print(f"  DRY RUN - Planner prompt preview")
        print(f"{'=' * 60}")
        print(f"  Characters: {char_count}")
        print(f"  Estimated tokens: ~{token_estimate}")
        print(f"{'=' * 60}\n")
        print(planner_input)
        return

    for iteration in range(start_iteration, max_iterations + 1):
        print(f"\n{'─' * 60}")
        print(f"  ITERATION {iteration}/{max_iterations}")
        print(f"{'─' * 60}")

        # ── PLANNING PHASE ──
        print(f"\n[{iteration}] Planning (model: {planner_model})...")

        if iteration == 1 and implementation_output is None:
            planner_input = (
                f"PROJECT: {os.path.basename(project_dir)}\n\n"
                f"{project_context}\n\n"
                f"GOAL: {goal}\n\n"
                f"Create a detailed implementation plan for this goal. "
                f"Be specific about files and changes."
            )
        else:
            if not auto_commit:
                diff = git_cmd(["diff"], project_dir)
                diff_stat = git_cmd(["diff", "--stat"], project_dir)
            else:
                diff = git_cmd(["diff", "HEAD~1"], project_dir)
                diff_stat = git_cmd(["diff", "HEAD~1", "--stat"], project_dir)
            recent_log = git_cmd(["log", "-5", "--oneline"], project_dir)

            # Build progress section
            progress_section = format_progress_section(progress_log)

            planner_input = (
                f"PROJECT: {os.path.basename(project_dir)}\n\n"
                f"OVERALL GOAL: {goal}\n\n"
            )

            if progress_section:
                planner_input += f"{progress_section}\n\n"

            planner_input += (
                f"=== IMPLEMENTATION OUTPUT (from last round) ===\n"
                f"{truncate(implementation_output, 40000)}\n\n"
                f"=== GIT DIFF STAT ===\n{diff_stat}\n\n"
                f"=== GIT DIFF ===\n{truncate(diff, 20000)}\n\n"
                f"=== RECENT GIT LOG ===\n{recent_log}\n\n"
                f"Review the implementation results above.\n"
                f"- If the goal is FULLY complete, respond with 'GOAL COMPLETE' and a summary.\n"
                f"- If there are errors or issues, plan fixes.\n"
                f"- If more work remains, provide the next implementation plan."
            )

        log_to_file(log_dir, iteration, "planner_input", planner_input)

        save_state(log_dir, build_state("calling_planner", iteration))

        plan = run_claude_with_retry(
            prompt=planner_input,
            cwd=project_dir,
            phase=f"planner (iter {iteration})",
            model=planner_model,
            system_prompt=PLANNER_SYSTEM_PROMPT,
            timeout=impl_timeout,
            effort=effort,
            state_save_callback=lambda it=iteration: save_state(
                log_dir, build_state("calling_planner", it)
            ),
        )

        log_to_file(log_dir, iteration, "planner_output", plan)

        # Show plan preview
        preview = plan[:800] + ("..." if len(plan) > 800 else "")
        print(f"[{iteration}] Plan received ({len(plan)} chars):\n")
        print(f"    {preview.replace(chr(10), chr(10) + '    ')}\n")

        # Check if goal is complete
        if "GOAL COMPLETE" in plan.upper():
            print(f"\n{'=' * 60}")
            print(f"  GOAL COMPLETE after {iteration} iteration(s)!")
            print(f"{'=' * 60}")
            print(f"\n{plan}\n")
            log_to_file(log_dir, iteration, "COMPLETE", plan)
            save_state(log_dir, build_state("completed", iteration, status="complete"))
            break

        # ── IMPLEMENTATION PHASE ──
        print(f"[{iteration}] Implementing (model: {impl_model})...")

        # Capture HEAD before implementation
        head_before = git_cmd(["rev-parse", "HEAD"], project_dir)

        impl_prompt = (
            f"Execute the following implementation plan completely.\n\n"
            f"PLAN:\n{plan}\n"
        )

        log_to_file(log_dir, iteration, "impl_input", impl_prompt)

        save_state(log_dir, build_state("calling_implementer", iteration))

        implementation_output = run_claude_with_retry(
            prompt=impl_prompt,
            cwd=project_dir,
            phase=f"implementer (iter {iteration})",
            model=impl_model,
            append_system_prompt=impl_append,
            timeout=impl_timeout,
            skip_permissions=skip_permissions,
            effort=effort,
            state_save_callback=lambda it=iteration: save_state(
                log_dir, build_state("calling_implementer", it)
            ),
        )

        log_to_file(log_dir, iteration, "impl_output", implementation_output)
        last_impl_file = f"iter{iteration:03d}_impl_output.md"

        # Capture HEAD after implementation
        head_after = git_cmd(["rev-parse", "HEAD"], project_dir)

        # Validate commits and select appropriate diff
        if head_before and head_after:
            if head_before == head_after:
                print(f"    [!] Warning: implementer did not commit (HEAD unchanged)")
            elif head_before != head_after:
                # Count commits between before and after
                commit_count = git_cmd(
                    ["rev-list", "--count", f"{head_before}..{head_after}"],
                    project_dir,
                )
                if commit_count and int(commit_count) > 1:
                    print(f"    [*] Implementer made {commit_count} commits")

        # Build progress log entry
        entry = {
            "iteration": iteration,
            "plan_summary": extract_summary(plan),
            "impl_summary": extract_summary(implementation_output),
            "commit_hash": head_after[:8] if head_after else "",
            "head_before": head_before,
            "head_after": head_after,
        }
        progress_log.append(entry)

        impl_preview = implementation_output[:600] + (
            "..." if len(implementation_output) > 600 else ""
        )
        print(f"[{iteration}] Implementation done ({len(implementation_output)} chars):\n")
        print(f"    {impl_preview.replace(chr(10), chr(10) + '    ')}\n")

        save_state(log_dir, build_state("iteration_complete", iteration))
    else:
        print(f"\n[!] Reached max iterations ({max_iterations}) without goal completion.")
        save_state(log_dir, build_state("completed", max_iterations, status="max_iterations"))

    print(f"\nAll logs saved to: {log_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Claude Loop: Planner + Implementer orchestration loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s ./my-project "Add user authentication with JWT"
  %(prog)s ./my-project "Fix all failing tests" --max-iterations 3
  %(prog)s ./my-project "Refactor to use TypeScript" --planner-model opus --impl-model sonnet
  %(prog)s ./my-project "Build a CLI tool" --skip-permissions
  %(prog)s ./my-project --dry-run "Preview the prompt"
  %(prog)s --resume                      # resume the most recent run
  %(prog)s --resume 20240101_120000      # resume a specific run by ID
""",
    )
    parser.add_argument(
        "project_dir", nargs="?", default=None,
        help="Path to the project directory (omit when using --resume)",
    )
    parser.add_argument("goal", nargs="?", default=None, help="The goal/task to accomplish")
    parser.add_argument(
        "--max-iterations", type=int, default=10,
        help="Maximum loop iterations (default: 10)",
    )
    parser.add_argument(
        "--planner-model", default="opus",
        help="Model for the planner (default: opus)",
    )
    parser.add_argument(
        "--impl-model", default="sonnet",
        help="Model for the implementer (default: sonnet)",
    )
    parser.add_argument(
        "--impl-timeout", type=int, default=600,
        help="Timeout in seconds for implementation rounds (default: 600)",
    )
    parser.add_argument(
        "--skip-permissions", action="store_true",
        help="Skip permission checks for the implementer (use in sandboxed environments)",
    )
    parser.add_argument(
        "--effort", default=None, choices=["low", "medium", "high", "max"],
        help="Effort level for both planner and implementer (default: model default)",
    )
    parser.add_argument(
        "--no-auto-commit", action="store_true",
        help="Do not auto-commit; leave changes for manual review",
    )
    parser.add_argument(
        "--resume", nargs="?", const="latest", default=None, metavar="RUN_ID",
        help="Resume a previous run. Pass a run ID (log directory name), "
             "or use --resume alone to resume the most recent run.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the planner prompt with estimates and exit",
    )

    args = parser.parse_args()

    if args.resume is not None:
        # Resume mode: load state from disk and rebuild run parameters.
        if args.resume == "latest":
            run_id = find_latest_run()
            if not run_id:
                print("Error: no previous runs with a run_state.json found")
                sys.exit(1)
        else:
            run_id = args.resume

        state = load_state(run_id)
        if state is None:
            print(f"Error: no state file found for run '{run_id}'")
            sys.exit(1)

        project_dir = state["project_dir"]
        if not os.path.isdir(project_dir):
            print(f"Error: project directory '{project_dir}' no longer exists")
            sys.exit(1)

        log_dir = os.path.join(LOGS_DIR, run_id)

        # Read the last completed implementation output if referenced.
        resume_impl_output = None
        last_impl_file = state.get("last_impl_output_file")
        if last_impl_file:
            impl_path = os.path.join(log_dir, last_impl_file)
            if os.path.isfile(impl_path):
                with open(impl_path) as f:
                    resume_impl_output = f.read()

        current_phase = state.get("current_phase", "calling_planner")
        current_iteration = state.get("current_iteration", 1)
        if current_phase == "iteration_complete":
            resume_from_iteration = current_iteration + 1
        else:
            resume_from_iteration = current_iteration

        run_loop(
            project_dir=project_dir,
            goal=state.get("goal", ""),
            max_iterations=state.get("max_iterations", 10),
            planner_model=state.get("planner_model", "opus"),
            impl_model=state.get("impl_model", "sonnet"),
            impl_timeout=state.get("impl_timeout", 600),
            skip_permissions=state.get("skip_permissions", False),
            effort=state.get("effort"),
            resume_from_iteration=resume_from_iteration,
            resume_log_dir=log_dir,
            resume_impl_output=resume_impl_output,
        )
        return

    # Normal (non-resume) mode: require project_dir and goal.
    if not args.project_dir:
        parser.error("project_dir is required unless using --resume")
    if not args.goal:
        parser.error("goal is required unless using --resume")

    project_dir = os.path.abspath(args.project_dir)
    if not os.path.isdir(project_dir):
        print(f"Error: '{project_dir}' is not a directory")
        sys.exit(1)

    run_loop(
        project_dir=project_dir,
        goal=args.goal,
        max_iterations=args.max_iterations,
        planner_model=args.planner_model,
        impl_model=args.impl_model,
        impl_timeout=args.impl_timeout,
        skip_permissions=args.skip_permissions,
        effort=args.effort,
        auto_commit=not args.no_auto_commit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
