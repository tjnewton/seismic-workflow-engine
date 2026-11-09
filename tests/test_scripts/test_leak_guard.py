"""Tests for scripts/leak_guard.py (commit-message and log modes)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "leak_guard.py"
_spec = importlib.util.spec_from_file_location("leak_guard", _PATH)
leak_guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(leak_guard)

TOKEN = "zzprivatetoken"


@pytest.fixture
def tokens(monkeypatch, tmp_path):
    """Isolate from any real denylist; use one dummy token."""
    monkeypatch.setenv("SWE_LEAK_DENYLIST", str(tmp_path / "no-such-denylist"))
    monkeypatch.setenv("LEAK_TOKENS", TOKEN)
    return leak_guard._load_tokens()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.email=t@example.org", "-c", "user.name=t", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


class TestCheckMessage:
    def test_token_flagged_case_insensitive(self, tokens):
        assert leak_guard.check_message("m", f"Add {TOKEN.upper()} support", tokens) == 1

    def test_clean_message_passes(self, tokens):
        assert leak_guard.check_message("m", "Add station inventory queries", tokens) == 0

    def test_comment_lines_are_scanned(self, tokens):
        """`git commit -m` keeps '#' lines in the commit, so they must be scanned."""
        assert leak_guard.check_message("m", f"Fix loader\n# {TOKEN} notes\n", tokens) == 1

    def test_scissors_diff_ignored_in_editmsg(self, tokens):
        """The `git commit -v` diff below the scissors line is not part of the message."""
        msg = (
            "Remove an old reference\n"
            "# ------------------------ >8 ------------------------\n"
            f"-old line with {TOKEN}\n"
        )
        assert leak_guard.check_message("m", msg, tokens, editmsg=True) == 0
        assert leak_guard.check_message("m", f"Add {TOKEN}\n" + msg, tokens, editmsg=True) == 1

    def test_scissors_with_crlf(self, tokens):
        msg = f"Clean subject\r\n# ------------------------ >8 ------------------------\r\n-{TOKEN}\r\n"
        assert leak_guard.check_message("m", msg, tokens, editmsg=True) == 0

    def test_scissors_with_custom_comment_char(self, tokens):
        msg = f"Clean subject\n; ------------------------ >8 ------------------------\n-{TOKEN}\n"
        assert leak_guard.check_message("m", msg, tokens, editmsg=True) == 0

    def test_secret_flagged_without_tokens(self):
        fake = "ghp_" + "a" * 36
        assert leak_guard.check_message("m", f"oops {fake}", []) == 1


class TestCheckLog:
    def test_scans_each_commit_in_range(self, tokens, tmp_path, monkeypatch):
        _git(tmp_path, "init", "-q")
        _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "clean first commit")
        base = _git(tmp_path, "rev-parse", "HEAD").strip()
        _git(tmp_path, "commit", "-q", "--allow-empty", "-m", f"mention {TOKEN}")
        _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "clean third commit")
        monkeypatch.chdir(tmp_path)
        assert leak_guard.check_log([f"{base}..HEAD"], tokens) == 1
        assert leak_guard.check_log([base], tokens) == 0

    def test_hash_line_kept_by_git_m_is_flagged(self, tokens, tmp_path, monkeypatch):
        """`git commit -m x -m '# token'` keeps the '#' line in the commit."""
        _git(tmp_path, "init", "-q")
        _git(
            tmp_path, "commit", "-q", "--allow-empty", "-m", "Fix loader", "-m", f"# {TOKEN} notes"
        )
        monkeypatch.chdir(tmp_path)
        assert leak_guard.check_log(["HEAD"], tokens) == 1

    def test_record_separator_in_message_does_not_hide_lines(self, tokens, tmp_path, monkeypatch):
        _git(tmp_path, "init", "-q")
        _git(tmp_path, "commit", "-q", "--allow-empty", "-m", f"Subject\n\nbody \x1e\n{TOKEN}")
        monkeypatch.chdir(tmp_path)
        assert leak_guard.check_log(["HEAD"], tokens) == 1


class TestPushRange:
    def test_union_of_pushed_ref_and_all_unpushed_branches(self, monkeypatch):
        monkeypatch.delenv("PRE_COMMIT_REMOTE_NAME", raising=False)
        monkeypatch.setenv("PRE_COMMIT_FROM_REF", "abc")
        monkeypatch.setenv("PRE_COMMIT_TO_REF", "def")
        assert leak_guard._push_range() == ["def", "--branches", "--not", "--remotes"]

    def test_only_target_remote_counts_as_published(self, tokens, tmp_path, monkeypatch):
        """A commit already on a private remote is still scanned when pushing to another."""
        priv, repo = tmp_path / "private.git", tmp_path / "repo"
        _git(tmp_path, "init", "-q", "--bare", str(priv))
        _git(tmp_path, "init", "-q", "-b", "main", str(repo))
        _git(repo, "commit", "-q", "--allow-empty", "-m", f"mention {TOKEN}")
        _git(repo, "remote", "add", "internal", str(priv))
        _git(repo, "push", "-q", "internal", "main")
        monkeypatch.chdir(repo)
        monkeypatch.setenv("PRE_COMMIT_TO_REF", "main")
        monkeypatch.setenv("PRE_COMMIT_REMOTE_NAME", "origin")
        assert leak_guard._push_range()[-1] == "--remotes=origin"
        assert leak_guard.check_log(leak_guard._push_range(), tokens) == 1
        monkeypatch.setenv("PRE_COMMIT_REMOTE_NAME", "internal")
        assert leak_guard.check_log(leak_guard._push_range(), tokens) == 0

    def test_refs_unset_scans_all_unpushed_branches(self, monkeypatch):
        monkeypatch.delenv("PRE_COMMIT_REMOTE_NAME", raising=False)
        monkeypatch.delenv("PRE_COMMIT_FROM_REF", raising=False)
        monkeypatch.delenv("PRE_COMMIT_TO_REF", raising=False)
        assert leak_guard._push_range() == ["--branches", "--not", "--remotes"]

    def test_second_branch_of_multi_ref_push_is_scanned(self, tokens, tmp_path, monkeypatch):
        """`git push origin a b` only reports `a`; a token on `b` must still be caught."""
        _git(tmp_path, "init", "-q", "-b", "a")
        _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "clean on a")
        _git(tmp_path, "checkout", "-q", "-b", "b")
        _git(tmp_path, "commit", "-q", "--allow-empty", "-m", f"mention {TOKEN}")
        _git(tmp_path, "checkout", "-q", "a")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("PRE_COMMIT_REMOTE_NAME", raising=False)
        monkeypatch.setenv("PRE_COMMIT_TO_REF", "a")
        assert leak_guard.check_log(leak_guard._push_range(), tokens) == 1


class TestMain:
    def test_commit_msg_mode_exit_codes(self, tokens, tmp_path, monkeypatch):
        bad, good = tmp_path / "bad", tmp_path / "good"
        bad.write_text(f"Add {TOKEN}\n")
        good.write_text("Add station inventory queries\n")
        monkeypatch.setattr(sys, "argv", ["leak_guard.py", "--commit-msg", str(bad)])
        assert leak_guard.main() == 1
        monkeypatch.setattr(sys, "argv", ["leak_guard.py", "--commit-msg", str(good)])
        assert leak_guard.main() == 0
        latin1 = tmp_path / "latin1"
        latin1.write_bytes("Caf\xe9 fix\n".encode("latin-1"))
        monkeypatch.setattr(sys, "argv", ["leak_guard.py", "--commit-msg", str(latin1)])
        assert leak_guard.main() == 0  # undecodable bytes do not crash the hook
