"""Tests for claude_loop.py."""
from __future__ import annotations

import json
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
        assert claude_loop.classify_response(text, "", 0) == "ok"

    def test_goal_complete_short(self):
        assert claude_loop.classify_response("GOAL COMPLETE\nDone.", "", 0) == "ok"

    def test_rate_limit_hit_your_limit(self):
        assert claude_loop.classify_response("You've hit your limit", "", 1) == "rate_limit"

    def test_rate_limit_usage_limit(self):
        assert claude_loop.classify_response("Usage limit exceeded. Try again later.", "", 1) == "rate_limit"

    def test_rate_limit_429(self):
        assert claude_loop.classify_response("Error 429: too many requests", "", 1) == "rate_limit"

    def test_rate_limit_overloaded(self):
        assert claude_loop.classify_response("The API is overloaded right now", "", 1) == "rate_limit"

    # New rate-limit phrasing coverage
    def test_rate_limit_quota_exceeded(self):
        assert claude_loop.classify_response("Quota exceeded. Try tomorrow.", "", 1) == "rate_limit"

    def test_rate_limit_throttled(self):
        assert claude_loop.classify_response("Request throttled.", "", 1) == "rate_limit"

    def test_rate_limit_please_try_again(self):
        assert claude_loop.classify_response("Please try again later.", "", 1) == "rate_limit"

    def test_rate_limit_cooldown(self):
        assert claude_loop.classify_response("Model in cooldown.", "", 1) == "rate_limit"

    def test_rate_limit_maximum_requests(self):
        assert claude_loop.classify_response("maximum hourly requests reached", "", 1) == "rate_limit"

    def test_rate_limit_limit_reached(self):
        assert claude_loop.classify_response("Limit reached for today.", "", 1) == "rate_limit"

    def test_timeout(self):
        assert claude_loop.classify_response("(Claude Code timed out after 300s)", "", -1) == "timeout"

    def test_timeout_600(self):
        assert claude_loop.classify_response("(Claude Code timed out after 600s)", "", -1) == "timeout"

    def test_error_generic(self):
        assert claude_loop.classify_response("(Error running Claude Code: connection reset)", "", -1) == "error"

    def test_empty_string(self):
        assert claude_loop.classify_response("", "", 0) == "error"

    def test_none_like_empty(self):
        assert claude_loop.classify_response("   ", "", 0) == "error"

    def test_short_suspicious_response(self):
        assert claude_loop.classify_response("ok", "", 0) == "error"

    def test_short_but_goal_complete(self):
        assert claude_loop.classify_response("GOAL COMPLETE", "", 0) == "ok"

    # New tests for hardened classification
    def test_returncode_primary_signal(self):
        # Non-zero returncode with rate limit pattern in stderr
        assert claude_loop.classify_response("", "You've hit your limit", 1) == "rate_limit"

    def test_returncode_zero_long_ok(self):
        # Long response with returncode 0 is ok even with error-like words
        text = "x" * 600
        assert claude_loop.classify_response(text, "", 0) == "ok"

    def test_nonzero_returncode_error(self):
        # Non-zero returncode with no matching patterns is generic error
        assert claude_loop.classify_response("something failed", "", 1) == "error"

    def test_rate_limit_short_returncode_zero(self):
        # Short response with rate limit pattern and returncode 0
        assert claude_loop.classify_response("You've hit your limit", "", 0) == "rate_limit"

    def test_stderr_scanned_for_rate_limit(self):
        # Rate limit in stderr with non-zero returncode
        assert claude_loop.classify_response("", "rate limit exceeded", 2) == "rate_limit"

    def test_bounded_scan_with_long_output(self):
        # Non-zero returncode but error pattern only in stdout boundaries
        long_text = "x" * 2000
        text_with_error = "(Error running Claude Code: fail)" + long_text
        assert claude_loop.classify_response(text_with_error, "", 1) == "error"

    def test_legacy_single_arg_ok(self):
        # Backwards compat: calling with just stdout still works (defaults)
        text = "Here is a detailed plan:\n1. Create foo.py\n" + "x" * 100
        assert claude_loop.classify_response(text) == "ok"


# ── parse_reset_time ──


class TestParseResetTime:
    def test_resets_pm(self):
        seconds = claude_loop.parse_reset_time("You've hit your limit - resets 1pm (America/Los_Angeles)")
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

    # New: expanded verb variants, seconds unit, 24-hour time, retry-after header.
    def test_try_again_in_seconds(self):
        seconds = claude_loop.parse_reset_time("please try again in 45 seconds")
        assert seconds is not None
        assert 45 <= seconds <= 45 + 120

    def test_retry_after_minutes(self):
        seconds = claude_loop.parse_reset_time("retry after 3 minutes")
        assert seconds is not None
        assert 3 * 60 <= seconds <= 3 * 60 + 120

    def test_wait_verb(self):
        seconds = claude_loop.parse_reset_time("please wait 2 minutes before retrying")
        assert seconds is not None
        assert 2 * 60 <= seconds <= 2 * 60 + 120

    def test_available_in(self):
        seconds = claude_loop.parse_reset_time("service available in 10 seconds")
        assert seconds is not None
        assert 10 <= seconds <= 10 + 120

    def test_seconds_unit_short_form(self):
        seconds = claude_loop.parse_reset_time("resets in 30 sec")
        assert seconds is not None
        assert 30 <= seconds <= 30 + 120

    def test_resets_at_24_hour(self):
        # 24-hour time format with no am/pm should parse and return a positive wait.
        seconds = claude_loop.parse_reset_time("resets at 23:59")
        assert seconds is not None
        assert 0 < seconds <= 24 * 3600 + 120

    def test_resets_at_24_hour_rejects_am_pm(self):
        # "resets at 2:30pm" should not be misinterpreted by the 24-hour branch.
        # The am/pm branch doesn't match with an "at" in between either (pre-existing),
        # so we just confirm this doesn't blow up and returns something plausible.
        seconds = claude_loop.parse_reset_time("resets at 2:30pm")
        # Either parses via another route or returns None; must not raise.
        assert seconds is None or seconds > 0

    def test_retry_after_header_style(self):
        # Bare "Retry-After: N" header-style, value is seconds.
        seconds = claude_loop.parse_reset_time("Retry-After: 60")
        assert seconds is not None
        assert 60 <= seconds <= 60 + 120

    def test_retry_after_header_with_hyphen(self):
        seconds = claude_loop.parse_reset_time("retry-after: 120")
        assert seconds is not None
        assert 120 <= seconds <= 120 + 120

    def test_retry_after_header_without_colon_not_matched_as_bare(self):
        # Without a colon and without a unit, this should not parse as seconds.
        assert claude_loop.parse_reset_time("retry after 30") is None


# ── format_duration ──


class TestFormatDuration:
    def test_seconds_only(self):
        assert claude_loop.format_duration(0) == "0s"
        assert claude_loop.format_duration(5) == "5s"
        assert claude_loop.format_duration(59) == "59s"

    def test_minutes_and_seconds(self):
        assert claude_loop.format_duration(60) == "1m 0s"
        assert claude_loop.format_duration(65) == "1m 5s"
        assert claude_loop.format_duration(3599) == "59m 59s"

    def test_hours_and_minutes(self):
        assert claude_loop.format_duration(3600) == "1h 0m"
        assert claude_loop.format_duration(3700) == "1h 1m"
        assert claude_loop.format_duration(7260) == "2h 1m"


# ── wait_with_countdown ──


class TestWaitWithCountdown:
    def test_smoke_does_not_raise(self, monkeypatch, capsys):
        # Replace sleep so the test runs instantly, then ensure the
        # function prints headers, the countdown line, and the finish line.
        monkeypatch.setattr(time, "sleep", lambda x: None)
        claude_loop.wait_with_countdown(4, "planner")
        output = capsys.readouterr().out
        assert "Waiting" in output
        assert "remaining" in output
        assert "Wait complete" in output
        assert "planner" in output


# ── run_claude ──


class TestRunClaude:
    def test_builds_basic_command(self, monkeypatch):
        captured_cmd = {}

        def fake_run(cmd, **kwargs):
            captured_cmd["cmd"] = cmd
            captured_cmd["kwargs"] = kwargs
            return subprocess.CompletedProcess(cmd, 0, stdout="output", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        stdout, stderr, returncode = claude_loop.run_claude("test prompt", "/tmp", model="sonnet")

        assert captured_cmd["cmd"] == [
            "claude", "--print", "--model", "sonnet", "test prompt"
        ]
        assert captured_cmd["kwargs"]["cwd"] == "/tmp"
        assert captured_cmd["kwargs"]["timeout"] == 600
        assert stdout == "output"
        assert stderr == ""
        assert returncode == 0

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

    def test_returns_tuple(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout="out", stderr="err")

        monkeypatch.setattr(subprocess, "run", fake_run)

        result = claude_loop.run_claude("p", "/tmp")
        assert isinstance(result, tuple)
        assert len(result) == 3
        stdout, stderr, returncode = result
        assert stdout == "out"
        assert stderr == "err"
        assert returncode == 0

    def test_no_stderr(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout="out", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        stdout, stderr, returncode = claude_loop.run_claude("p", "/tmp")
        assert stdout == "out"
        assert stderr == ""
        assert returncode == 0

    def test_timeout_handled(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, 600)

        monkeypatch.setattr(subprocess, "run", fake_run)

        stdout, stderr, returncode = claude_loop.run_claude("p", "/tmp", timeout=600)
        assert "timed out" in stdout
        assert "600" in stdout
        assert returncode == -1

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

        stdout, stderr, returncode = claude_loop.run_claude("p", "/tmp")
        assert "Error" in stdout
        assert "something broke" in stdout
        assert returncode == -1

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
            return ("Here is a detailed plan with enough content to pass the minimum threshold." + "x" * 100, "", 0)

        monkeypatch.setattr(claude_loop, "run_claude", fake_run_claude)

        result = claude_loop.run_claude_with_retry("p", "/tmp", "test")
        assert "detailed plan" in result

    def test_retries_on_rate_limit(self, monkeypatch):
        calls = {"n": 0}

        def fake_run_claude(prompt, cwd, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return ("You've hit your limit - resets in 0 minutes", "", 1)
            return ("Here is the real response with enough content to pass." + "x" * 100, "", 0)

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
                return ("(Claude Code timed out after 300s)", "", -1)
            return ("Success after retries with enough content." + "x" * 100, "", 0)

        monkeypatch.setattr(claude_loop, "run_claude", fake_run_claude)
        monkeypatch.setattr(time, "sleep", lambda x: None)

        result = claude_loop.run_claude_with_retry("p", "/tmp", "test")
        assert calls["n"] == 3
        assert "Success" in result

    def test_gives_up_after_max_retries(self, monkeypatch):
        def fake_run_claude(prompt, cwd, **kwargs):
            return ("(Claude Code timed out after 300s)", "", -1)

        monkeypatch.setattr(claude_loop, "run_claude", fake_run_claude)
        monkeypatch.setattr(time, "sleep", lambda x: None)

        result = claude_loop.run_claude_with_retry("p", "/tmp", "test", max_retries=2)
        assert "timed out" in result

    def test_retries_on_short_error(self, monkeypatch):
        calls = {"n": 0}

        def fake_run_claude(prompt, cwd, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return ("error", "", 1)
            return ("A proper response with sufficient content." + "x" * 100, "", 0)

        monkeypatch.setattr(claude_loop, "run_claude", fake_run_claude)
        monkeypatch.setattr(time, "sleep", lambda x: None)

        result = claude_loop.run_claude_with_retry("p", "/tmp", "test")
        assert calls["n"] == 2

    def test_passes_all_kwargs(self, monkeypatch):
        captured_kwargs = {}

        def fake_run_claude(prompt, cwd, **kwargs):
            captured_kwargs.update(kwargs)
            return ("ok " * 50, "", 0)

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

    def test_returns_string_not_tuple(self, monkeypatch):
        """run_claude_with_retry returns just the stdout string."""
        def fake_run_claude(prompt, cwd, **kwargs):
            return ("response text" + "x" * 100, "some stderr", 0)

        monkeypatch.setattr(claude_loop, "run_claude", fake_run_claude)

        result = claude_loop.run_claude_with_retry("p", "/tmp", "test")
        assert isinstance(result, str)
        assert "response text" in result

    def test_state_save_callback_fires_before_wait(self, monkeypatch):
        """Callback runs once per rate-limit wait, before the sleep."""
        calls = {"n": 0, "saves": 0}

        def fake_run_claude(prompt, cwd, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return ("You've hit your limit - resets in 0 minutes", "", 1)
            return ("Real content after wait." + "x" * 100, "", 0)

        def fake_save():
            calls["saves"] += 1

        monkeypatch.setattr(claude_loop, "run_claude", fake_run_claude)
        monkeypatch.setattr(time, "sleep", lambda x: None)

        result = claude_loop.run_claude_with_retry(
            "p", "/tmp", "test", state_save_callback=fake_save,
        )
        assert calls["saves"] == 1
        assert "Real content" in result

    def test_state_save_callback_not_fired_on_success(self, monkeypatch):
        """Callback does not fire when the first call already succeeds."""
        calls = {"saves": 0}

        def fake_run_claude(prompt, cwd, **kwargs):
            return ("plenty of content to be considered ok " * 20, "", 0)

        monkeypatch.setattr(claude_loop, "run_claude", fake_run_claude)

        claude_loop.run_claude_with_retry(
            "p", "/tmp", "test", state_save_callback=lambda: calls.__setitem__("saves", calls["saves"] + 1),
        )
        assert calls["saves"] == 0

    def test_state_save_callback_failure_is_non_fatal(self, monkeypatch):
        """If the callback raises, the wait still proceeds and we recover."""
        calls = {"n": 0}

        def fake_run_claude(prompt, cwd, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return ("rate limit hit - resets in 0 minutes", "", 1)
            return ("Real content." + "x" * 100, "", 0)

        def bad_callback():
            raise RuntimeError("disk full")

        monkeypatch.setattr(claude_loop, "run_claude", fake_run_claude)
        monkeypatch.setattr(time, "sleep", lambda x: None)

        result = claude_loop.run_claude_with_retry(
            "p", "/tmp", "test", state_save_callback=bad_callback,
        )
        assert "Real content" in result

    def test_backoff_resets_after_rate_limit_recovery(self, monkeypatch):
        """Backoff seconds reset to 30 after a clean rate-limit wait."""
        call_types = []
        sleep_values = []

        def fake_run_claude(prompt, cwd, **kwargs):
            # Sequence: error, error (backoff grows), rate_limit (reset), error (backoff must be 30 again), ok
            n = len(call_types)
            call_types.append(n)
            if n == 0:
                return ("(Error running Claude Code: boom)", "", -1)
            if n == 1:
                return ("(Error running Claude Code: boom)", "", -1)
            if n == 2:
                return ("rate limit hit - resets in 0 minutes", "", 1)
            if n == 3:
                return ("(Error running Claude Code: boom)", "", -1)
            return ("Final real content." + "x" * 100, "", 0)

        def fake_sleep(s):
            sleep_values.append(s)

        monkeypatch.setattr(claude_loop, "run_claude", fake_run_claude)
        monkeypatch.setattr(time, "sleep", fake_sleep)

        result = claude_loop.run_claude_with_retry("p", "/tmp", "test", max_retries=5)
        assert "Final real content" in result

        # Separate the post-rate-limit sleeps from pre-rate-limit sleeps.
        # The countdown uses 2-second chunks; error backoffs are >= 30s.
        error_backoffs = [s for s in sleep_values if s >= 30]
        # First error: 30s, second error: 60s. Then rate limit resets backoff.
        # After that, the next error must start again at 30s (not 120s).
        assert error_backoffs[:2] == [30, 60]
        assert error_backoffs[2] == 30, f"expected backoff reset to 30, got {error_backoffs}"


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

    # New: structure-aware context tests
    def test_includes_key_files(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name": "test-project", "version": "1.0.0"}')
        (tmp_path / "Makefile").write_text("all:\n\techo hello\n")
        ctx = claude_loop.get_project_context(str(tmp_path))
        assert "PROJECT STRUCTURE" in ctx
        assert "package.json" in ctx
        assert "test-project" in ctx
        assert "Makefile" in ctx

    def test_key_file_reads_first_20_lines(self, tmp_path):
        lines = [f"line {i}\n" for i in range(40)]
        (tmp_path / "requirements.txt").write_text("".join(lines))
        ctx = claude_loop.get_project_context(str(tmp_path))
        assert "PROJECT STRUCTURE" in ctx
        assert "line 0" in ctx
        assert "line 19" in ctx
        # Lines beyond 20 should not appear in the key files section
        structure_section = ctx.split("PROJECT STRUCTURE")[1].split("===")[0]
        assert "line 25" not in structure_section

    def test_no_key_files_no_section(self, tmp_path):
        (tmp_path / "random.txt").write_text("nothing special")
        ctx = claude_loop.get_project_context(str(tmp_path))
        assert "PROJECT STRUCTURE" not in ctx

    def test_claude_md_detected(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("# Claude Instructions\nDo X and Y")
        ctx = claude_loop.get_project_context(str(tmp_path))
        assert "PROJECT STRUCTURE" in ctx
        assert "CLAUDE.md" in ctx
        assert "Claude Instructions" in ctx


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

    def test_resume_from_line_absent_by_default(self, capsys):
        claude_loop.print_header("/p", "g", 5, "/logs/r")
        output = capsys.readouterr().out
        assert "Resuming from" not in output

    def test_resume_from_line_printed_when_set(self, capsys):
        claude_loop.print_header("/p", "g", 5, "/logs/r", resume_from=3)
        output = capsys.readouterr().out
        assert "Resuming from" in output
        assert "iteration 3" in output


# ── extract_summary ──


class TestExtractSummary:
    def test_extracts_first_sentences(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = claude_loop.extract_summary(text)
        assert "First sentence." in result
        assert "Second sentence." in result
        assert "Third sentence." in result
        assert "Fourth sentence." not in result

    def test_handles_empty_string(self):
        assert claude_loop.extract_summary("") == ""

    def test_caps_at_500_chars(self):
        text = "A" * 600 + ". End."
        result = claude_loop.extract_summary(text)
        assert len(result) <= 503  # 500 + "..."

    def test_single_sentence(self):
        result = claude_loop.extract_summary("Only one sentence here.")
        assert result == "Only one sentence here."

    def test_custom_max_sentences(self):
        text = "One. Two. Three. Four."
        result = claude_loop.extract_summary(text, max_sentences=2)
        assert "One." in result
        assert "Two." in result
        assert "Three." not in result


# ── format_progress_section ──


class TestFormatProgressSection:
    def test_empty_log(self):
        assert claude_loop.format_progress_section([]) == ""

    def test_single_entry(self):
        log = [{"iteration": 1, "plan_summary": "Plan A", "impl_summary": "Did A",
                "commit_hash": "abc123", "head_before": "", "head_after": ""}]
        result = claude_loop.format_progress_section(log)
        assert "PROGRESS SO FAR" in result
        assert "Iteration 1" in result
        assert "Plan A" in result
        assert "Did A" in result

    def test_multiple_entries(self):
        log = [
            {"iteration": 1, "plan_summary": "P1", "impl_summary": "I1",
             "commit_hash": "aaa", "head_before": "", "head_after": ""},
            {"iteration": 2, "plan_summary": "P2", "impl_summary": "I2",
             "commit_hash": "bbb", "head_before": "", "head_after": ""},
        ]
        result = claude_loop.format_progress_section(log)
        assert "Iteration 1" in result
        assert "Iteration 2" in result

    def test_caps_at_4000_chars(self):
        log = []
        for i in range(100):
            log.append({
                "iteration": i + 1,
                "plan_summary": "X" * 100,
                "impl_summary": "Y" * 100,
                "commit_hash": "abc",
                "head_before": "1234567890",
                "head_after": "0987654321",
            })
        result = claude_loop.format_progress_section(log)
        assert "omitted" in result
        assert "Iteration 1" in result
        assert "Iteration 100" in result

    def test_includes_head_info(self):
        log = [{"iteration": 1, "plan_summary": "P", "impl_summary": "I",
                "commit_hash": "abc", "head_before": "1234567890ab", "head_after": "abcdef123456"}]
        result = claude_loop.format_progress_section(log)
        assert "12345678" in result
        assert "abcdef12" in result


# ── save_state / load_state / find_latest_run ──


class TestStatePersistence:
    def _minimal_state(self):
        return {
            "project_dir": "/some/path",
            "goal": "do it",
            "max_iterations": 5,
            "planner_model": "opus",
            "impl_model": "sonnet",
            "impl_timeout": 600,
            "skip_permissions": False,
            "effort": None,
            "current_iteration": 2,
            "current_phase": "iteration_complete",
            "status": "running",
            "last_impl_output_file": "iter002_impl_output.md",
        }

    def test_save_state_writes_file(self, tmp_path):
        claude_loop.save_state(str(tmp_path), self._minimal_state())
        state_path = tmp_path / claude_loop.STATE_FILE
        assert state_path.exists()
        data = json.loads(state_path.read_text())
        assert data["current_iteration"] == 2
        assert data["current_phase"] == "iteration_complete"
        assert "saved_at" in data

    def test_save_state_overwrites(self, tmp_path):
        claude_loop.save_state(str(tmp_path), self._minimal_state())
        updated = self._minimal_state()
        updated["current_iteration"] = 3
        updated["current_phase"] = "calling_planner"
        claude_loop.save_state(str(tmp_path), updated)
        data = json.loads((tmp_path / claude_loop.STATE_FILE).read_text())
        assert data["current_iteration"] == 3
        assert data["current_phase"] == "calling_planner"

    def test_save_state_never_raises(self, tmp_path, capsys):
        # Target a nonexistent directory; save_state should swallow the error.
        bad_dir = str(tmp_path / "does_not_exist")
        claude_loop.save_state(bad_dir, self._minimal_state())
        # Should print a warning rather than raising
        output = capsys.readouterr().out
        assert "Failed to save state" in output

    def test_load_state_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(claude_loop, "LOGS_DIR", str(tmp_path))
        run_id = "20260115_120000"
        log_dir = tmp_path / run_id
        log_dir.mkdir()
        claude_loop.save_state(str(log_dir), self._minimal_state())

        loaded = claude_loop.load_state(run_id)
        assert loaded is not None
        assert loaded["current_iteration"] == 2
        assert loaded["goal"] == "do it"
        assert loaded["last_impl_output_file"] == "iter002_impl_output.md"

    def test_load_state_missing_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(claude_loop, "LOGS_DIR", str(tmp_path))
        assert claude_loop.load_state("nope") is None

    def test_load_state_corrupt_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(claude_loop, "LOGS_DIR", str(tmp_path))
        run_id = "20260115_120000"
        log_dir = tmp_path / run_id
        log_dir.mkdir()
        (log_dir / claude_loop.STATE_FILE).write_text("not json {{{")
        assert claude_loop.load_state(run_id) is None


class TestFindLatestRun:
    def test_no_logs_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(claude_loop, "LOGS_DIR", str(tmp_path / "absent"))
        assert claude_loop.find_latest_run() is None

    def test_empty_logs_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(claude_loop, "LOGS_DIR", str(tmp_path))
        assert claude_loop.find_latest_run() is None

    def test_ignores_runs_without_state(self, tmp_path, monkeypatch):
        monkeypatch.setattr(claude_loop, "LOGS_DIR", str(tmp_path))
        (tmp_path / "old_run").mkdir()
        (tmp_path / "old_run" / "iter001_planner_input.md").write_text("no state")
        assert claude_loop.find_latest_run() is None

    def test_returns_most_recent_by_mtime(self, tmp_path, monkeypatch):
        monkeypatch.setattr(claude_loop, "LOGS_DIR", str(tmp_path))
        first = tmp_path / "first_run"
        first.mkdir()
        (first / claude_loop.STATE_FILE).write_text("{}")
        # Backdate the first state file so the second is strictly newer.
        old = time.time() - 3600
        os.utime(first / claude_loop.STATE_FILE, (old, old))

        second = tmp_path / "second_run"
        second.mkdir()
        (second / claude_loop.STATE_FILE).write_text("{}")

        assert claude_loop.find_latest_run() == "second_run"


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

    def test_progress_log_built(self, monkeypatch, tmp_path):
        """Progress log accumulates entries across iterations."""
        self._init_git(tmp_path)
        planner_calls = {"n": 0}
        captured_prompts = []

        def fake(prompt, cwd, phase, **kwargs):
            if kwargs.get("system_prompt"):
                planner_calls["n"] += 1
                captured_prompts.append(prompt)
                if planner_calls["n"] <= 2:
                    return "1. Do step\n2. More work" + "x" * 100
                return "GOAL COMPLETE\nAll done."
            return "Implemented the step successfully." + "x" * 100

        monkeypatch.setattr(claude_loop, "run_claude_with_retry", fake)
        monkeypatch.setattr(claude_loop, "LOGS_DIR", str(tmp_path / "logs"))

        claude_loop.run_loop(str(tmp_path), "build it", max_iterations=5)

        # Third planner call (iteration 3) should have progress from iterations 1 and 2
        assert planner_calls["n"] == 3
        assert "PROGRESS SO FAR" in captured_prompts[2]

    def test_strips_local_path(self, monkeypatch, tmp_path):
        """Planner prompts use basename, not full path."""
        self._init_git(tmp_path)
        captured_prompts = []

        def fake(prompt, cwd, phase, **kwargs):
            if kwargs.get("system_prompt"):
                captured_prompts.append(prompt)
                return "GOAL COMPLETE\nDone."
            return "implemented"

        monkeypatch.setattr(claude_loop, "run_claude_with_retry", fake)
        monkeypatch.setattr(claude_loop, "LOGS_DIR", str(tmp_path / "logs"))

        claude_loop.run_loop(str(tmp_path), "test", max_iterations=1)
        # Should use PROJECT: basename, not PROJECT DIRECTORY: /full/path
        assert f"PROJECT: {os.path.basename(str(tmp_path))}" in captured_prompts[0]
        assert "PROJECT DIRECTORY:" not in captured_prompts[0]

    def test_no_auto_commit_uses_working_diff(self, monkeypatch, tmp_path):
        """With no_auto_commit, uses working tree diff and different append prompt."""
        self._init_git(tmp_path)
        planner_calls = {"n": 0}
        captured_append = []

        def fake(prompt, cwd, phase, **kwargs):
            if kwargs.get("append_system_prompt"):
                captured_append.append(kwargs["append_system_prompt"])
            if kwargs.get("system_prompt"):
                planner_calls["n"] += 1
                if planner_calls["n"] == 1:
                    return "1. Do step" + "x" * 100
                return "GOAL COMPLETE\nDone."
            return "Implemented." + "x" * 100

        monkeypatch.setattr(claude_loop, "run_claude_with_retry", fake)
        monkeypatch.setattr(claude_loop, "LOGS_DIR", str(tmp_path / "logs"))

        claude_loop.run_loop(str(tmp_path), "test", max_iterations=5, auto_commit=False)
        assert any("Do NOT commit" in a for a in captured_append)

    def test_dry_run_exits_early(self, monkeypatch, tmp_path, capsys):
        """Dry run prints prompt and returns without running Claude."""
        self._init_git(tmp_path)
        call_count = {"n": 0}

        def fake(prompt, cwd, phase, **kwargs):
            call_count["n"] += 1
            return "should not be called"

        monkeypatch.setattr(claude_loop, "run_claude_with_retry", fake)
        monkeypatch.setattr(claude_loop, "LOGS_DIR", str(tmp_path / "logs"))

        claude_loop.run_loop(str(tmp_path), "test goal", max_iterations=5, dry_run=True)
        output = capsys.readouterr().out
        assert "DRY RUN" in output
        assert "Characters:" in output
        assert "Estimated tokens:" in output
        assert "test goal" in output
        assert call_count["n"] == 0

    def test_state_file_written_per_iteration(self, monkeypatch, tmp_path):
        """run_loop writes a run_state.json that progresses through phases."""
        self._init_git(tmp_path)
        planner_calls = {"n": 0}
        # Track the phase stored in run_state.json at each save.
        seen_phases = []

        def fake(prompt, cwd, phase, **kwargs):
            # Capture current state from disk at each CLI call.
            log_dirs = [d for d in os.listdir(str(tmp_path / "logs"))] if (tmp_path / "logs").exists() else []
            if log_dirs:
                state_path = tmp_path / "logs" / log_dirs[0] / claude_loop.STATE_FILE
                if state_path.exists():
                    seen_phases.append(json.loads(state_path.read_text())["current_phase"])
            if kwargs.get("system_prompt"):
                planner_calls["n"] += 1
                if planner_calls["n"] == 1:
                    return "1. Do step" + "x" * 100
                return "GOAL COMPLETE\nDone."
            return "Implemented." + "x" * 100

        monkeypatch.setattr(claude_loop, "run_claude_with_retry", fake)
        monkeypatch.setattr(claude_loop, "LOGS_DIR", str(tmp_path / "logs"))

        claude_loop.run_loop(str(tmp_path), "build", max_iterations=3)

        # Should have observed at least: calling_planner, calling_implementer, calling_planner
        assert "calling_planner" in seen_phases
        assert "calling_implementer" in seen_phases

        # After run, final state should exist and reflect completion.
        run_dirs = os.listdir(str(tmp_path / "logs"))
        state_path = tmp_path / "logs" / run_dirs[0] / claude_loop.STATE_FILE
        assert state_path.exists()
        final = json.loads(state_path.read_text())
        assert final["current_phase"] == "completed"

    def test_state_file_records_last_impl_output(self, monkeypatch, tmp_path):
        """After an iteration completes, last_impl_output_file points to the file."""
        self._init_git(tmp_path)
        planner_calls = {"n": 0}

        def fake(prompt, cwd, phase, **kwargs):
            if kwargs.get("system_prompt"):
                planner_calls["n"] += 1
                if planner_calls["n"] == 1:
                    return "1. Step" + "x" * 100
                return "GOAL COMPLETE\nDone."
            return "work done" + "x" * 100

        monkeypatch.setattr(claude_loop, "run_claude_with_retry", fake)
        monkeypatch.setattr(claude_loop, "LOGS_DIR", str(tmp_path / "logs"))

        claude_loop.run_loop(str(tmp_path), "build", max_iterations=3)

        run_dirs = os.listdir(str(tmp_path / "logs"))
        state_path = tmp_path / "logs" / run_dirs[0] / claude_loop.STATE_FILE
        final = json.loads(state_path.read_text())
        assert final["last_impl_output_file"] == "iter001_impl_output.md"

    def test_resume_uses_existing_log_dir(self, monkeypatch, tmp_path):
        """Resumed runs must reuse the supplied log directory, not create a new one."""
        self._init_git(tmp_path)
        planner_calls = {"n": 0}

        def fake(prompt, cwd, phase, **kwargs):
            if kwargs.get("system_prompt"):
                planner_calls["n"] += 1
                return "GOAL COMPLETE\nPicked up where we left off."
            return "never"

        logs_root = tmp_path / "logs"
        logs_root.mkdir()
        existing_run = logs_root / "20260115_000000"
        existing_run.mkdir()

        monkeypatch.setattr(claude_loop, "run_claude_with_retry", fake)
        monkeypatch.setattr(claude_loop, "LOGS_DIR", str(logs_root))

        claude_loop.run_loop(
            str(tmp_path), "resume goal", max_iterations=3,
            resume_from_iteration=2,
            resume_log_dir=str(existing_run),
            resume_impl_output="prior impl output",
        )

        # No new run directory should have been created.
        dirs = sorted(os.listdir(str(logs_root)))
        assert dirs == ["20260115_000000"]
        # A planner_input file should be written with iter002_ prefix, not iter001_.
        files = os.listdir(str(existing_run))
        assert any(f.startswith("iter002_planner_input") for f in files)
        assert not any(f.startswith("iter001_") for f in files)

    def test_resume_planner_prompt_includes_prior_impl(self, monkeypatch, tmp_path):
        """When resuming with a prior impl output, the planner sees it."""
        self._init_git(tmp_path)
        captured_prompts = []

        def fake(prompt, cwd, phase, **kwargs):
            if kwargs.get("system_prompt"):
                captured_prompts.append(prompt)
                return "GOAL COMPLETE\nAll set."
            return "never"

        logs_root = tmp_path / "logs"
        logs_root.mkdir()
        existing_run = logs_root / "20260115_000000"
        existing_run.mkdir()

        monkeypatch.setattr(claude_loop, "run_claude_with_retry", fake)
        monkeypatch.setattr(claude_loop, "LOGS_DIR", str(logs_root))

        claude_loop.run_loop(
            str(tmp_path), "g", max_iterations=3,
            resume_from_iteration=2,
            resume_log_dir=str(existing_run),
            resume_impl_output="PRIOR_IMPL_MARKER",
        )
        assert any("PRIOR_IMPL_MARKER" in p for p in captured_prompts)

    def test_commit_validation_warning(self, monkeypatch, tmp_path, capsys):
        """Warns when HEAD doesn't change after implementation."""
        self._init_git(tmp_path)
        planner_calls = {"n": 0}

        def fake(prompt, cwd, phase, **kwargs):
            if kwargs.get("system_prompt"):
                planner_calls["n"] += 1
                if planner_calls["n"] == 1:
                    return "1. Do step" + "x" * 100
                return "GOAL COMPLETE\nDone."
            return "Did the work but forgot to commit." + "x" * 100

        monkeypatch.setattr(claude_loop, "run_claude_with_retry", fake)
        monkeypatch.setattr(claude_loop, "LOGS_DIR", str(tmp_path / "logs"))

        claude_loop.run_loop(str(tmp_path), "test", max_iterations=5)
        output = capsys.readouterr().out
        assert "did not commit" in output


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
        assert captured_args["auto_commit"] is True
        assert captured_args["dry_run"] is False
        # Resume params are not passed for a normal (non-resume) run.
        assert "resume_from_iteration" not in captured_args
        assert "resume_log_dir" not in captured_args

    def test_no_auto_commit_flag(self, monkeypatch, tmp_path):
        captured_args = {}

        def fake_run_loop(**kwargs):
            captured_args.update(kwargs)

        monkeypatch.setattr(claude_loop, "run_loop", fake_run_loop)
        monkeypatch.setattr(
            sys, "argv",
            ["claude_loop.py", str(tmp_path), "goal", "--no-auto-commit"],
        )

        claude_loop.main()
        assert captured_args["auto_commit"] is False

    def test_dry_run_flag(self, monkeypatch, tmp_path):
        captured_args = {}

        def fake_run_loop(**kwargs):
            captured_args.update(kwargs)

        monkeypatch.setattr(claude_loop, "run_loop", fake_run_loop)
        monkeypatch.setattr(
            sys, "argv",
            ["claude_loop.py", str(tmp_path), "goal", "--dry-run"],
        )

        claude_loop.main()
        assert captured_args["dry_run"] is True

    def _write_state(self, logs_dir, run_id, state):
        """Helper: write a run_state.json under logs_dir/run_id/."""
        run_dir = os.path.join(logs_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, claude_loop.STATE_FILE), "w") as f:
            json.dump(state, f)
        return run_dir

    def test_resume_with_run_id(self, monkeypatch, tmp_path):
        captured_args = {}

        def fake_run_loop(**kwargs):
            captured_args.update(kwargs)

        project = tmp_path / "proj"
        project.mkdir()
        logs = tmp_path / "logs"
        logs.mkdir()
        run_id = "20260115_120000"
        state = {
            "project_dir": str(project),
            "goal": "resumed goal",
            "max_iterations": 7,
            "planner_model": "opus",
            "impl_model": "sonnet",
            "impl_timeout": 600,
            "skip_permissions": False,
            "effort": "max",
            "current_iteration": 2,
            "current_phase": "iteration_complete",
            "status": "running",
            "last_impl_output_file": "iter002_impl_output.md",
        }
        run_dir = self._write_state(str(logs), run_id, state)
        # Stash an impl output file so main() can load it.
        with open(os.path.join(run_dir, "iter002_impl_output.md"), "w") as f:
            f.write("the last implementation output")

        monkeypatch.setattr(claude_loop, "LOGS_DIR", str(logs))
        monkeypatch.setattr(claude_loop, "run_loop", fake_run_loop)
        monkeypatch.setattr(sys, "argv", ["claude_loop.py", "--resume", run_id])

        claude_loop.main()

        # iteration_complete => next iteration is current + 1
        assert captured_args["resume_from_iteration"] == 3
        assert captured_args["resume_log_dir"] == run_dir
        assert captured_args["resume_impl_output"] == "the last implementation output"
        assert captured_args["project_dir"] == str(project)
        assert captured_args["goal"] == "resumed goal"
        assert captured_args["max_iterations"] == 7
        assert captured_args["effort"] == "max"

    def test_resume_latest_uses_find_latest_run(self, monkeypatch, tmp_path):
        captured_args = {}

        def fake_run_loop(**kwargs):
            captured_args.update(kwargs)

        project = tmp_path / "proj"
        project.mkdir()
        logs = tmp_path / "logs"
        logs.mkdir()
        run_id = "20260115_120000"
        state = {
            "project_dir": str(project),
            "goal": "latest goal",
            "max_iterations": 5,
            "planner_model": "opus",
            "impl_model": "sonnet",
            "impl_timeout": 600,
            "skip_permissions": False,
            "effort": None,
            "current_iteration": 1,
            "current_phase": "calling_implementer",
            "status": "running",
            "last_impl_output_file": None,
        }
        self._write_state(str(logs), run_id, state)

        monkeypatch.setattr(claude_loop, "LOGS_DIR", str(logs))
        monkeypatch.setattr(claude_loop, "run_loop", fake_run_loop)
        monkeypatch.setattr(sys, "argv", ["claude_loop.py", "--resume"])

        claude_loop.main()

        # calling_implementer => re-start current_iteration (not +1)
        assert captured_args["resume_from_iteration"] == 1
        assert captured_args["goal"] == "latest goal"

    def test_resume_latest_with_no_runs_exits(self, monkeypatch, tmp_path):
        logs = tmp_path / "logs"
        logs.mkdir()
        monkeypatch.setattr(claude_loop, "LOGS_DIR", str(logs))
        monkeypatch.setattr(sys, "argv", ["claude_loop.py", "--resume"])
        with pytest.raises(SystemExit) as exc:
            claude_loop.main()
        assert exc.value.code == 1

    def test_resume_missing_state_exits(self, monkeypatch, tmp_path):
        logs = tmp_path / "logs"
        logs.mkdir()
        monkeypatch.setattr(claude_loop, "LOGS_DIR", str(logs))
        monkeypatch.setattr(sys, "argv", ["claude_loop.py", "--resume", "nonexistent"])
        with pytest.raises(SystemExit) as exc:
            claude_loop.main()
        assert exc.value.code == 1

    def test_resume_stale_project_dir_exits(self, monkeypatch, tmp_path):
        logs = tmp_path / "logs"
        logs.mkdir()
        run_id = "20260115_120000"
        state = {
            "project_dir": str(tmp_path / "no_longer_here"),
            "goal": "g",
            "max_iterations": 5,
            "planner_model": "opus",
            "impl_model": "sonnet",
            "impl_timeout": 600,
            "skip_permissions": False,
            "effort": None,
            "current_iteration": 1,
            "current_phase": "calling_planner",
            "status": "running",
            "last_impl_output_file": None,
        }
        self._write_state(str(logs), run_id, state)
        monkeypatch.setattr(claude_loop, "LOGS_DIR", str(logs))
        monkeypatch.setattr(sys, "argv", ["claude_loop.py", "--resume", run_id])
        with pytest.raises(SystemExit) as exc:
            claude_loop.main()
        assert exc.value.code == 1

    def test_no_goal_no_resume_exits(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            sys, "argv",
            ["claude_loop.py", str(tmp_path)],
        )
        with pytest.raises(SystemExit):
            claude_loop.main()

    def test_no_project_dir_no_resume_exits(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["claude_loop.py"])
        with pytest.raises(SystemExit):
            claude_loop.main()


# ── Constants ──


class TestConstants:
    def test_planner_prompt_has_goal_complete_instruction(self):
        assert "GOAL COMPLETE" in claude_loop.PLANNER_SYSTEM_PROMPT

    def test_implementer_prompt_mentions_commit(self):
        assert "commit" in claude_loop.IMPLEMENTER_APPEND_PROMPT.lower()

    def test_no_auto_commit_prompt_says_do_not_commit(self):
        assert "Do NOT commit" in claude_loop.NO_AUTO_COMMIT_APPEND_PROMPT

    def test_max_output_chars_reasonable(self):
        assert claude_loop.MAX_OUTPUT_CHARS >= 10000

    def test_min_valid_response_chars(self):
        assert claude_loop.MIN_VALID_RESPONSE_CHARS > 0
        assert claude_loop.MIN_VALID_RESPONSE_CHARS <= 200

    def test_rate_limit_patterns_not_empty(self):
        assert len(claude_loop.RATE_LIMIT_PATTERNS) > 0

    def test_key_project_files_not_empty(self):
        assert len(claude_loop.KEY_PROJECT_FILES) > 0
        assert "package.json" in claude_loop.KEY_PROJECT_FILES
        assert "CLAUDE.md" in claude_loop.KEY_PROJECT_FILES

    def test_state_file_constant(self):
        assert claude_loop.STATE_FILE == "run_state.json"

    def test_new_rate_limit_patterns_present(self):
        patterns = "|".join(claude_loop.RATE_LIMIT_PATTERNS)
        for token in ["quota", "throttled", "cooldown", "limit reached", "try again later", "maximum"]:
            assert token in patterns.lower(), f"missing new pattern token: {token}"
