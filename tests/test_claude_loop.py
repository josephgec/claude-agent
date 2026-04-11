"""Tests for claude_loop.py."""
from __future__ import annotations

import os
import sys
import subprocess
import time

import pytest

# Add parent dir to path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import claude_loop


# ── truncate ──


class TestTruncate:
    def test_short_text_unchanged(self):
        assert claude_loop.truncate("hello", 100) == "hello"

    def test_exact_limit_unchanged(self):
        text = "a" * 100
        assert claude_loop.truncate(text, 100) == text

    def test_long_text_truncated(self):
        text = "a" * 200
        result = claude_loop.truncate(text, 100)
        assert result.startswith("a" * 100)
        assert "truncated" in result
        assert "200" in result

    def test_default_limit(self):
        text = "x" * (claude_loop.MAX_OUTPUT_CHARS + 1000)
        result = claude_loop.truncate(text)
        assert len(result) < len(text)
        assert "truncated" in result

    def test_empty_string(self):
        assert claude_loop.truncate("", 100) == ""


# ── classify_response ──


class TestClassifyResponse:
    def test_ok_response(self):
        text = "Here is a detailed plan:\n1. Create foo.py\n2. Add bar function\n" + "x" * 100
        assert claude_loop.classify_response(text) == "ok"

    def test_goal_complete_short(self):
        assert claude_loop.classify_response("GOAL COMPLETE\nDone.") == "ok"

    def test_rate_limit_hit_your_limit(self):
        assert claude_loop.classify_response("You've hit your limit · resets 1pm (America/Los_Angeles)") == "rate_limit"

    def test_rate_limit_usage_limit(self):
        assert claude_loop.classify_response("Usage limit exceeded. Try again later.") == "rate_limit"

    def test_rate_limit_429(self):
        assert claude_loop.classify_response("Error 429: too many requests") == "rate_limit"

    def test_rate_limit_overloaded(self):
        assert claude_loop.classify_response("The API is overloaded right now") == "rate_limit"

    def test_timeout(self):
        assert claude_loop.classify_response("(Claude Code timed out after 300s)") == "timeout"

    def test_timeout_600(self):
        assert claude_loop.classify_response("(Claude Code timed out after 600s)") == "timeout"

    def test_error_generic(self):
        assert claude_loop.classify_response("(Error running Claude Code: connection reset)") == "error"

    def test_empty_string(self):
        assert claude_loop.classify_response("") == "error"

    def test_none_like_empty(self):
        assert claude_loop.classify_response("   ") == "error"

    def test_short_suspicious_response(self):
        assert claude_loop.classify_response("ok") == "error"

    def test_short_but_goal_complete(self):
        assert claude_loop.classify_response("GOAL COMPLETE") == "ok"


# ── parse_reset_time ──


class TestParseResetTime:
    def test_resets_pm(self):
        seconds = claude_loop.parse_reset_time("You've hit your limit · resets 1pm (America/Los_Angeles)")
        assert seconds is not None
        assert seconds > 0
        assert seconds <= 24 * 3600 + 60  # at most ~24h + buffer

    def test_resets_am(self):
        seconds = claude_loop.parse_reset_time("resets 9am")
        assert seconds is not None
        assert seconds > 0

    def test_resets_with_minutes(self):
        seconds = claude_loop.parse_reset_time("resets 2:30pm")
        assert seconds is not None
        assert seconds > 0

    def test_resets_in_minutes(self):
        seconds = claude_loop.parse_reset_time("resets in 45 minutes")
        assert seconds is not None
        assert 45 * 60 <= seconds <= 46 * 60 + 60

    def test_resets_in_hours(self):
        seconds = claude_loop.parse_reset_time("resets in 2 hours")
        assert seconds is not None
        assert 2 * 3600 <= seconds <= 2 * 3600 + 120

    def test_unparseable(self):
        assert claude_loop.parse_reset_time("something random") is None

    def test_empty(self):
        assert claude_loop.parse_reset_time("") is None

    def test_resets_in_min_abbreviation(self):
        seconds = claude_loop.parse_reset_time("resets in 30 min")
        assert seconds is not None
        assert 30 * 60 <= seconds <= 31 * 60 + 60


# ── run_claude ──


class TestRunClaude:
    def test_builds_basic_command(self, monkeypatch):
        captured_cmd = {}

        def fake_run(cmd, **kwargs):
            captured_cmd["cmd"] = cmd
            captured_cmd["kwargs"] = kwargs
            return subprocess.CompletedProcess(cmd, 0, stdout="output", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        result = claude_loop.run_claude("test prompt", "/tmp", model="sonnet")

        assert captured_cmd["cmd"] == [
            "claude", "--print", "--model", "sonnet", "test prompt"
        ]
        assert captured_cmd["kwargs"]["cwd"] == "/tmp"
        assert captured_cmd["kwargs"]["timeout"] == 600
        assert result == "output"

    def test_includes_system_prompt(self, monkeypatch):
        captured_cmd = {}

        def fake_run(cmd, **kwargs):
            captured_cmd["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        claude_loop.run_claude("p", "/tmp", system_prompt="You are a planner")
        assert "--system-prompt" in captured_cmd["cmd"]
        idx = captured_cmd["cmd"].index("--system-prompt")
        assert captured_cmd["cmd"][idx + 1] == "You are a planner"

    def test_includes_append_system_prompt(self, monkeypatch):
        captured_cmd = {}

        def fake_run(cmd, **kwargs):
            captured_cmd["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        claude_loop.run_claude("p", "/tmp", append_system_prompt="Commit changes")
        assert "--append-system-prompt" in captured_cmd["cmd"]
        idx = captured_cmd["cmd"].index("--append-system-prompt")
        assert captured_cmd["cmd"][idx + 1] == "Commit changes"

    def test_includes_effort(self, monkeypatch):
        captured_cmd = {}

        def fake_run(cmd, **kwargs):
            captured_cmd["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        claude_loop.run_claude("p", "/tmp", effort="max")
        assert "--effort" in captured_cmd["cmd"]
        idx = captured_cmd["cmd"].index("--effort")
        assert captured_cmd["cmd"][idx + 1] == "max"

    def test_effort_not_included_when_none(self, monkeypatch):
        captured_cmd = {}

        def fake_run(cmd, **kwargs):
            captured_cmd["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        claude_loop.run_claude("p", "/tmp", effort=None)
        assert "--effort" not in captured_cmd["cmd"]

    def test_skip_permissions_flag(self, monkeypatch):
        captured_cmd = {}

        def fake_run(cmd, **kwargs):
            captured_cmd["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        claude_loop.run_claude("p", "/tmp", skip_permissions=True)
        assert "--dangerously-skip-permissions" in captured_cmd["cmd"]

    def test_skip_permissions_not_included_by_default(self, monkeypatch):
        captured_cmd = {}

        def fake_run(cmd, **kwargs):
            captured_cmd["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        claude_loop.run_claude("p", "/tmp")
        assert "--dangerously-skip-permissions" not in captured_cmd["cmd"]

    def test_stderr_appended(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout="out", stderr="err")

        monkeypatch.setattr(subprocess, "run", fake_run)

        result = claude_loop.run_claude("p", "/tmp")
        assert "out" in result
        assert "STDERR" in result
        assert "err" in result

    def test_no_stderr_no_section(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout="out", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        result = claude_loop.run_claude("p", "/tmp")
        assert result == "out"
        assert "STDERR" not in result

    def test_timeout_handled(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, 600)

        monkeypatch.setattr(subprocess, "run", fake_run)

        result = claude_loop.run_claude("p", "/tmp", timeout=600)
        assert "timed out" in result
        assert "600" in result

    def test_file_not_found_exits(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            raise FileNotFoundError()

        monkeypatch.setattr(subprocess, "run", fake_run)

        with pytest.raises(SystemExit):
            claude_loop.run_claude("p", "/tmp")

    def test_generic_exception_handled(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            raise RuntimeError("something broke")

        monkeypatch.setattr(subprocess, "run", fake_run)

        result = claude_loop.run_claude("p", "/tmp")
        assert "Error" in result
        assert "something broke" in result

    def test_custom_timeout(self, monkeypatch):
        captured_kwargs = {}

        def fake_run(cmd, **kwargs):
            captured_kwargs.update(kwargs)
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        claude_loop.run_claude("p", "/tmp", timeout=120)
        assert captured_kwargs["timeout"] == 120


# ── run_claude_with_retry ──


class TestRunClaudeWithRetry:
    def test_ok_on_first_try(self, monkeypatch):
        def fake_run_claude(prompt, cwd, **kwargs):
            return "Here is a detailed plan with enough content to pass the minimum threshold." + "x" * 100

        monkeypatch.setattr(claude_loop, "run_claude", fake_run_claude)

        result = claude_loop.run_claude_with_retry("p", "/tmp", "test")
        assert "detailed plan" in result

    def test_retries_on_rate_limit(self, monkeypatch):
        calls = {"n": 0}

        def fake_run_claude(prompt, cwd, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return "You've hit your limit · resets in 0 minutes"
            return "Here is the real response with enough content to pass." + "x" * 100

        monkeypatch.setattr(claude_loop, "run_claude", fake_run_claude)
        monkeypatch.setattr(time, "sleep", lambda x: None)  # skip actual sleep

        result = claude_loop.run_claude_with_retry("p", "/tmp", "test")
        assert calls["n"] == 2
        assert "real response" in result

    def test_retries_on_timeout(self, monkeypatch):
        calls = {"n": 0}

        def fake_run_claude(prompt, cwd, **kwargs):
            calls["n"] += 1
            if calls["n"] <= 2:
                return "(Claude Code timed out after 300s)"
            return "Success after retries with enough content." + "x" * 100

        monkeypatch.setattr(claude_loop, "run_claude", fake_run_claude)
        monkeypatch.setattr(time, "sleep", lambda x: None)

        result = claude_loop.run_claude_with_retry("p", "/tmp", "test")
        assert calls["n"] == 3
        assert "Success" in result

    def test_gives_up_after_max_retries(self, monkeypatch):
        def fake_run_claude(prompt, cwd, **kwargs):
            return "(Claude Code timed out after 300s)"

        monkeypatch.setattr(claude_loop, "run_claude", fake_run_claude)
        monkeypatch.setattr(time, "sleep", lambda x: None)

        result = claude_loop.run_claude_with_retry("p", "/tmp", "test", max_retries=2)
        assert "timed out" in result

    def test_retries_on_short_error(self, monkeypatch):
        calls = {"n": 0}

        def fake_run_claude(prompt, cwd, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return "error"
            return "A proper response with sufficient content." + "x" * 100

        monkeypatch.setattr(claude_loop, "run_claude", fake_run_claude)
        monkeypatch.setattr(time, "sleep", lambda x: None)

        result = claude_loop.run_claude_with_retry("p", "/tmp", "test")
        assert calls["n"] == 2

    def test_passes_all_kwargs(self, monkeypatch):
        captured_kwargs = {}

        def fake_run_claude(prompt, cwd, **kwargs):
            captured_kwargs.update(kwargs)
            return "ok " * 50

        monkeypatch.setattr(claude_loop, "run_claude", fake_run_claude)

        claude_loop.run_claude_with_retry(
            "p", "/tmp", "test",
            model="opus", effort="max", skip_permissions=True,
            system_prompt="sys", append_system_prompt="append",
            timeout=120,
        )
        assert captured_kwargs["model"] == "opus"
        assert captured_kwargs["effort"] == "max"
        assert captured_kwargs["skip_permissions"] is True
        assert captured_kwargs["system_prompt"] == "sys"
        assert captured_kwargs["append_system_prompt"] == "append"
        assert captured_kwargs["timeout"] == 120


# ── git_cmd ──


class TestGitCmd:
    def test_returns_stdout(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout="abc123 commit\n", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        result = claude_loop.git_cmd(["log", "-1", "--oneline"], "/tmp")
        assert result == "abc123 commit"

    def test_strips_whitespace(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout="  hello  \n", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        result = claude_loop.git_cmd(["status"], "/tmp")
        assert result == "hello"

    def test_exception_returns_empty(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            raise OSError("not a git repo")

        monkeypatch.setattr(subprocess, "run", fake_run)

        result = claude_loop.git_cmd(["status"], "/tmp")
        assert result == ""

    def test_passes_cwd(self, monkeypatch):
        captured_kwargs = {}

        def fake_run(cmd, **kwargs):
            captured_kwargs.update(kwargs)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        claude_loop.git_cmd(["log"], "/my/project")
        assert captured_kwargs["cwd"] == "/my/project"

    def test_constructs_git_command(self, monkeypatch):
        captured_cmd = {}

        def fake_run(cmd, **kwargs):
            captured_cmd["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        claude_loop.git_cmd(["diff", "HEAD~1", "--stat"], "/tmp")
        assert captured_cmd["cmd"] == ["git", "diff", "HEAD~1", "--stat"]


# ── get_project_context ──


class TestGetProjectContext:
    def test_includes_file_listing(self, tmp_path):
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "README.md").write_text("# Project")
        subprocess.run(
            ["git", "add", "."], cwd=str(tmp_path), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=str(tmp_path),
            capture_output=True,
            env={**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t",
                 "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t"},
        )

        ctx = claude_loop.get_project_context(str(tmp_path))
        assert "FILE LISTING" in ctx
        assert "main.py" in ctx

    def test_includes_readme(self, tmp_path):
        (tmp_path / "README.md").write_text("# My Project\nSome description")
        ctx = claude_loop.get_project_context(str(tmp_path))
        assert "My Project" in ctx

    def test_includes_git_log(self, tmp_path):
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        (tmp_path / "f.txt").write_text("x")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "first commit"],
            cwd=str(tmp_path),
            capture_output=True,
            env={**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t",
                 "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t"},
        )

        ctx = claude_loop.get_project_context(str(tmp_path))
        assert "GIT LOG" in ctx
        assert "first commit" in ctx

    def test_excludes_git_dir(self, tmp_path):
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        ctx = claude_loop.get_project_context(str(tmp_path))
        assert ".git/" not in ctx.split("FILE LISTING")[1].split("===")[0]

    def test_excludes_node_modules(self, tmp_path):
        nm = tmp_path / "node_modules" / "pkg"
        nm.mkdir(parents=True)
        (nm / "index.js").write_text("module.exports = {}")
        ctx = claude_loop.get_project_context(str(tmp_path))
        assert "node_modules" not in ctx.split("FILE LISTING")[1].split("===")[0]

    def test_truncates_large_file_listing(self, tmp_path):
        for i in range(500):
            (tmp_path / f"file_{i:04d}.txt").write_text("x")
        ctx = claude_loop.get_project_context(str(tmp_path))
        assert "truncated" in ctx or len(ctx) < 10000

    def test_no_readme(self, tmp_path):
        ctx = claude_loop.get_project_context(str(tmp_path))
        assert "README" not in ctx or "FILE LISTING" in ctx

    def test_no_git(self, tmp_path):
        ctx = claude_loop.get_project_context(str(tmp_path))
        assert "FILE LISTING" in ctx


# ── log_to_file ──


class TestLogToFile:
    def test_creates_file(self, tmp_path):
        claude_loop.log_to_file(str(tmp_path), 1, "planner_input", "content here")
        filepath = tmp_path / "iter001_planner_input.md"
        assert filepath.exists()
        assert filepath.read_text() == "content here"

    def test_correct_naming(self, tmp_path):
        claude_loop.log_to_file(str(tmp_path), 5, "impl_output", "data")
        assert (tmp_path / "iter005_impl_output.md").exists()

    def test_overwrites_existing(self, tmp_path):
        claude_loop.log_to_file(str(tmp_path), 1, "test", "first")
        claude_loop.log_to_file(str(tmp_path), 1, "test", "second")
        assert (tmp_path / "iter001_test.md").read_text() == "second"

    def test_large_iteration_number(self, tmp_path):
        claude_loop.log_to_file(str(tmp_path), 100, "phase", "data")
        assert (tmp_path / "iter100_phase.md").exists()


# ── print_header ──


class TestPrintHeader:
    def test_prints_all_fields(self, capsys):
        claude_loop.print_header("/my/project", "build thing", 5, "/logs/run1")
        output = capsys.readouterr().out
        assert "/my/project" in output
        assert "build thing" in output
        assert "5" in output
        assert "/logs/run1" in output
        assert "Claude Loop" in output


# ── run_loop ──


class TestRunLoop:
    def _init_git(self, tmp_path):
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        (tmp_path / "f.txt").write_text("x")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=str(tmp_path),
            capture_output=True,
            env={**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t",
                 "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t"},
        )

    def test_goal_complete_on_first_plan(self, monkeypatch, tmp_path):
        self._init_git(tmp_path)
        call_count = {"n": 0}

        def fake(prompt, cwd, phase, **kwargs):
            call_count["n"] += 1
            if kwargs.get("system_prompt"):
                return "GOAL COMPLETE\nEverything is done."
            return "implemented"

        monkeypatch.setattr(claude_loop, "run_claude_with_retry", fake)
        monkeypatch.setattr(claude_loop, "LOGS_DIR", str(tmp_path / "logs"))

        claude_loop.run_loop(str(tmp_path), "do nothing", max_iterations=5)
        assert call_count["n"] == 1

    def test_full_loop_iteration(self, monkeypatch, tmp_path):
        self._init_git(tmp_path)
        planner_calls = {"n": 0}

        def fake(prompt, cwd, phase, **kwargs):
            if kwargs.get("system_prompt"):
                planner_calls["n"] += 1
                if planner_calls["n"] == 1:
                    return "1. Create foo.py\n2. Add bar function" + "x" * 100
                return "GOAL COMPLETE\nAll done."
            return "Created foo.py with bar function." + "x" * 100

        monkeypatch.setattr(claude_loop, "run_claude_with_retry", fake)
        monkeypatch.setattr(claude_loop, "LOGS_DIR", str(tmp_path / "logs"))

        claude_loop.run_loop(str(tmp_path), "build foo", max_iterations=5)
        assert planner_calls["n"] == 2

    def test_max_iterations_reached(self, monkeypatch, tmp_path, capsys):
        self._init_git(tmp_path)

        def fake(prompt, cwd, phase, **kwargs):
            if kwargs.get("system_prompt"):
                return "1. Do step A\n2. Do step B" + "x" * 100
            return "Done with step A and B." + "x" * 100

        monkeypatch.setattr(claude_loop, "run_claude_with_retry", fake)
        monkeypatch.setattr(claude_loop, "LOGS_DIR", str(tmp_path / "logs"))

        claude_loop.run_loop(str(tmp_path), "endless goal", max_iterations=2)
        output = capsys.readouterr().out
        assert "Reached max iterations" in output

    def test_effort_passed_through(self, monkeypatch, tmp_path):
        self._init_git(tmp_path)
        captured_efforts = []

        def fake(prompt, cwd, phase, **kwargs):
            captured_efforts.append(kwargs.get("effort"))
            if kwargs.get("system_prompt"):
                return "GOAL COMPLETE\nDone."
            return "implemented"

        monkeypatch.setattr(claude_loop, "run_claude_with_retry", fake)
        monkeypatch.setattr(claude_loop, "LOGS_DIR", str(tmp_path / "logs"))

        claude_loop.run_loop(str(tmp_path), "test", max_iterations=1, effort="max")
        assert "max" in captured_efforts

    def test_logs_created(self, monkeypatch, tmp_path):
        self._init_git(tmp_path)

        def fake(prompt, cwd, phase, **kwargs):
            if kwargs.get("system_prompt"):
                return "GOAL COMPLETE\nDone."
            return "implemented"

        log_dir = str(tmp_path / "logs")
        monkeypatch.setattr(claude_loop, "run_claude_with_retry", fake)
        monkeypatch.setattr(claude_loop, "LOGS_DIR", log_dir)

        claude_loop.run_loop(str(tmp_path), "test", max_iterations=1)

        run_dirs = os.listdir(log_dir)
        assert len(run_dirs) == 1
        run_path = os.path.join(log_dir, run_dirs[0])
        files = os.listdir(run_path)
        assert any("planner_input" in f for f in files)
        assert any("planner_output" in f for f in files)
        assert any("COMPLETE" in f for f in files)


# ── main (argparse) ──


class TestMain:
    def test_invalid_directory_exits(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["claude_loop.py", "/nonexistent/path", "goal"],
        )
        with pytest.raises(SystemExit) as exc_info:
            claude_loop.main()
        assert exc_info.value.code == 1

    def test_valid_args_parsed(self, monkeypatch, tmp_path):
        captured_args = {}

        def fake_run_loop(**kwargs):
            captured_args.update(kwargs)

        monkeypatch.setattr(claude_loop, "run_loop", fake_run_loop)
        monkeypatch.setattr(
            sys, "argv",
            [
                "claude_loop.py", str(tmp_path), "build it",
                "--max-iterations", "3",
                "--planner-model", "opus",
                "--impl-model", "sonnet",
                "--effort", "max",
                "--skip-permissions",
            ],
        )

        claude_loop.main()

        assert captured_args["project_dir"] == str(tmp_path)
        assert captured_args["goal"] == "build it"
        assert captured_args["max_iterations"] == 3
        assert captured_args["planner_model"] == "opus"
        assert captured_args["impl_model"] == "sonnet"
        assert captured_args["effort"] == "max"
        assert captured_args["skip_permissions"] is True

    def test_defaults(self, monkeypatch, tmp_path):
        captured_args = {}

        def fake_run_loop(**kwargs):
            captured_args.update(kwargs)

        monkeypatch.setattr(claude_loop, "run_loop", fake_run_loop)
        monkeypatch.setattr(
            sys, "argv",
            ["claude_loop.py", str(tmp_path), "goal"],
        )

        claude_loop.main()

        assert captured_args["max_iterations"] == 10
        assert captured_args["planner_model"] == "opus"
        assert captured_args["impl_model"] == "sonnet"
        assert captured_args["effort"] is None
        assert captured_args["skip_permissions"] is False
        assert captured_args["impl_timeout"] == 600


# ── Constants ──


class TestConstants:
    def test_planner_prompt_has_goal_complete_instruction(self):
        assert "GOAL COMPLETE" in claude_loop.PLANNER_SYSTEM_PROMPT

    def test_implementer_prompt_mentions_commit(self):
        assert "commit" in claude_loop.IMPLEMENTER_APPEND_PROMPT.lower()

    def test_max_output_chars_reasonable(self):
        assert claude_loop.MAX_OUTPUT_CHARS >= 10000

    def test_min_valid_response_chars(self):
        assert claude_loop.MIN_VALID_RESPONSE_CHARS > 0
        assert claude_loop.MIN_VALID_RESPONSE_CHARS <= 200

    def test_rate_limit_patterns_not_empty(self):
        assert len(claude_loop.RATE_LIMIT_PATTERNS) > 0
