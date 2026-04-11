#!/usr/bin/env python3
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

import subprocess
import sys
import os
import time
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

LOGS_DIR = os.path.expanduser("~/.claude-loop/logs")
MAX_OUTPUT_CHARS = 80000  # truncate outputs sent to planner if longer

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


def truncate(text: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    """Truncate text with a note if it exceeds max_chars."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n... (truncated, {len(text)} total chars)"


def run_claude(
    prompt: str,
    cwd: str,
    model: str = "opus",
    system_prompt: Optional[str] = None,
    append_system_prompt: Optional[str] = None,
    timeout: int = 600,
    skip_permissions: bool = False,
    effort: Optional[str] = None,
) -> str:
    """Run Claude Code CLI in print mode and return the output."""
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
        output = result.stdout
        if result.stderr:
            output += f"\n\n=== STDERR ===\n{result.stderr}"
        return output
    except subprocess.TimeoutExpired:
        return f"(Claude Code timed out after {timeout}s)"
    except FileNotFoundError:
        print("Error: 'claude' command not found. Is Claude Code installed?")
        sys.exit(1)
    except Exception as e:
        return f"(Error running Claude Code: {e})"


def git_cmd(args: list[str], cwd: str) -> str:
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
    """Gather project context: file tree, git log, README, etc."""
    parts = []

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
        tree = result.stdout.strip()
        if len(tree) > 5000:
            tree = tree[:5000] + "\n... (truncated)"
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


def print_header(project_dir: str, goal: str, max_iterations: int, log_dir: str):
    print(f"\n{'=' * 60}")
    print(f"  Claude Loop Orchestrator")
    print(f"{'=' * 60}")
    print(f"  Project:        {project_dir}")
    print(f"  Goal:           {goal}")
    print(f"  Max iterations: {max_iterations}")
    print(f"  Logs:           {log_dir}")
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
):
    """Main orchestration loop."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(LOGS_DIR, run_id)
    os.makedirs(log_dir, exist_ok=True)

    print_header(project_dir, goal, max_iterations, log_dir)

    # Gather initial project context
    print("[*] Gathering project context...")
    project_context = get_project_context(project_dir)

    implementation_output = None

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'─' * 60}")
        print(f"  ITERATION {iteration}/{max_iterations}")
        print(f"{'─' * 60}")

        # ── PLANNING PHASE ──
        print(f"\n[{iteration}] Planning (model: {planner_model})...")

        if iteration == 1:
            planner_input = (
                f"PROJECT DIRECTORY: {project_dir}\n\n"
                f"{project_context}\n\n"
                f"GOAL: {goal}\n\n"
                f"Create a detailed implementation plan for this goal. "
                f"Be specific about files and changes."
            )
        else:
            diff = git_cmd(["diff", "HEAD~1"], project_dir)
            diff_stat = git_cmd(["diff", "HEAD~1", "--stat"], project_dir)
            recent_log = git_cmd(["log", "-5", "--oneline"], project_dir)

            planner_input = (
                f"PROJECT DIRECTORY: {project_dir}\n\n"
                f"OVERALL GOAL: {goal}\n\n"
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

        plan = run_claude(
            prompt=planner_input,
            cwd=project_dir,
            model=planner_model,
            system_prompt=PLANNER_SYSTEM_PROMPT,
            timeout=300,
            effort=effort,
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
            break

        # ── IMPLEMENTATION PHASE ──
        print(f"[{iteration}] Implementing (model: {impl_model})...")

        impl_prompt = (
            f"Execute the following implementation plan completely.\n\n"
            f"PLAN:\n{plan}\n"
        )

        log_to_file(log_dir, iteration, "impl_input", impl_prompt)

        implementation_output = run_claude(
            prompt=impl_prompt,
            cwd=project_dir,
            model=impl_model,
            append_system_prompt=IMPLEMENTER_APPEND_PROMPT,
            timeout=impl_timeout,
            skip_permissions=skip_permissions,
            effort=effort,
        )

        log_to_file(log_dir, iteration, "impl_output", implementation_output)

        impl_preview = implementation_output[:600] + (
            "..." if len(implementation_output) > 600 else ""
        )
        print(f"[{iteration}] Implementation done ({len(implementation_output)} chars):\n")
        print(f"    {impl_preview.replace(chr(10), chr(10) + '    ')}\n")
    else:
        print(f"\n[!] Reached max iterations ({max_iterations}) without goal completion.")

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
""",
    )
    parser.add_argument("project_dir", help="Path to the project directory")
    parser.add_argument("goal", help="The goal/task to accomplish")
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

    args = parser.parse_args()

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
    )


if __name__ == "__main__":
    main()
