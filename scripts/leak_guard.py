#!/usr/bin/env python3
"""Leak guard for the public seismic-workflow-engine repo.

Blocks accidental commits of private, oversized, or secret content. Generic by
design: this script contains NO private project names. Distinctive private
name-tokens live in an out-of-repo denylist referenced by ``$SWE_LEAK_DENYLIST``
(default ``~/.config/swe/leak-denylist``), one lowercase token per line, and/or
the ``$LEAK_TOKENS`` env var in CI. Used by ``.pre-commit-config.yaml`` (staged
files; the commit message via ``--commit-msg``; pushed commit messages via
``--push-log``) and the CI ``leak-check`` job (``--tracked``, plus ``--log`` over
a pull request's commits).

Exit non-zero on any violation.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

MAX_BYTES = 25 * 1024 * 1024

# Never appropriate to commit to the public repo: waveforms, archives, pickles,
# large binary science formats.
BLOCKED_EXT = {
    ".ms",
    ".mseed",
    ".msd",
    ".pkl",
    ".pickle",
    ".mrkr",
    ".tif",
    ".tiff",
    ".laz",
    ".las",
    ".mat",
    ".nc",
    ".h5",
    ".hdf5",
    ".npy",
    ".npz",
    ".tar",
    ".tgz",
    ".gz",
    ".zip",
    ".bundle",
}

SECRET_RE = re.compile(
    r"BEGIN (?:RSA|OPENSSH|EC|DSA|PGP) PRIVATE KEY"
    r"|AKIA[0-9A-Z]{16}"
    r"|ghp_[0-9A-Za-z]{36}"
    r"|xox[baprs]-[0-9A-Za-z-]+"
)


def _git(*args: str) -> list[str]:
    out = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    return [line for line in out.stdout.splitlines() if line]


def _load_tokens() -> list[str]:
    tokens: list[str] = []
    path = Path(os.environ.get("SWE_LEAK_DENYLIST", Path.home() / ".config/swe/leak-denylist"))
    if path.is_file():
        tokens += [
            line.strip().lower()
            for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    env = os.environ.get("LEAK_TOKENS", "")
    if env:
        tokens += [t.strip().lower() for t in re.split(r"[,\s]+", env) if t.strip()]
    return tokens


def check(paths: list[str], tokens: list[str]) -> int:
    violations = 0

    def flag(path: str, why: str) -> None:
        nonlocal violations
        print(f"LEAK GUARD: {why} -- {path}", file=sys.stderr)
        violations += 1

    for p in paths:
        f = Path(p)
        if not f.is_file():
            continue
        if f.suffix.lower() in BLOCKED_EXT or f.name.lower().endswith(".tar.gz"):
            flag(p, "blocked file type")
            continue
        try:
            if f.stat().st_size > MAX_BYTES:
                flag(p, f"exceeds {MAX_BYTES // (1024 * 1024)}MB")
                continue
            data = f.read_bytes()
        except OSError:
            continue
        if b"\x00" in data[:8192]:  # binary; skip text scans
            continue
        text = data.decode("utf-8", "replace")
        if SECRET_RE.search(text):
            flag(p, "possible secret")
            continue
        low = text.lower()
        if tokens and any(tok in low for tok in tokens):
            flag(p, "matches private denylist token")
            continue
    return violations


# git writes the scissors line with core.commentChar ("#" by default; any char).
SCISSORS_RE = re.compile(r"(?m)^\S -+ >8 -+\r?$")


def check_message(label: str, text: str, tokens: list[str], *, editmsg: bool = False) -> int:
    """Scan one commit message, including any ``#`` lines.

    Comment lines are scanned too: with ``git commit -m`` git keeps them in the
    commit, and git's own template lines name the branch and files, which are
    leaks if they hold a private token. With ``editmsg=True`` (the commit-msg
    hook's COMMIT_EDITMSG), everything below git's scissors line is dropped:
    that is the ``git commit -v`` diff, which git always removes.
    """
    if editmsg:
        text = SCISSORS_RE.split(text, maxsplit=1)[0]
    low = text.lower()
    why = None
    if SECRET_RE.search(text):
        why = "possible secret in commit message"
    elif tokens and any(tok in low for tok in tokens):
        why = "commit message matches private denylist token"
    if why:
        print(f"LEAK GUARD: {why} -- {label}", file=sys.stderr)
        return 1
    return 0


def check_log(rev_args: list[str], tokens: list[str]) -> int:
    """Scan the messages of every commit selected by ``git log <rev_args>``."""
    # -z separates entries with NUL, which cannot occur in a commit message.
    out = subprocess.run(
        ["git", "log", "-z", "--format=%H%n%B", *rev_args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    violations = 0
    for entry in out.split("\0"):
        entry = entry.strip("\n")
        if not entry:
            continue
        sha, _, msg = entry.partition("\n")
        violations += check_message(f"commit {sha[:12]}", msg, tokens)
    return violations


def _push_range() -> list[str]:
    """Commits a push may publish: every local commit not yet on the target remote.

    pre-commit passes only the first pushed ref (PRE_COMMIT_TO_REF) even for
    ``git push origin a b`` or ``--all``, so that ref alone would miss the
    others. All local branches are scanned as well, plus the given ref in case
    it is not a branch (e.g. a detached HEAD). Only refs of the remote being
    pushed to (PRE_COMMIT_REMOTE_NAME) count as already published: a commit
    on another remote, such as a private mirror, is still scanned. Trade-off:
    an unpushed local branch with a flagged message blocks pushes until it is
    reworded or deleted.
    """
    to_ref = os.environ.get("PRE_COMMIT_TO_REF", "")
    remote = os.environ.get("PRE_COMMIT_REMOTE_NAME", "")
    published = f"--remotes={remote}" if remote else "--remotes"
    return [*([to_ref] if to_ref else []), "--branches", "--not", published]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracked", action="store_true", help="scan all git-tracked files (CI mode)")
    ap.add_argument(
        "--commit-msg", metavar="FILE", help="scan a commit message file (commit-msg hook)"
    )
    ap.add_argument(
        "--log", metavar="RANGE", help="scan commit messages in a git revision range (CI)"
    )
    ap.add_argument(
        "--push-log",
        action="store_true",
        help="scan messages of the commits being pushed (pre-push hook)",
    )
    ap.add_argument("files", nargs="*", help="files to check (pre-commit passes staged files)")
    args = ap.parse_args()

    tokens = _load_tokens()
    if args.commit_msg or args.log or args.push_log:
        if args.commit_msg:
            bad = check_message(
                "commit message",
                Path(args.commit_msg).read_text(encoding="utf-8", errors="replace"),
                tokens,
                editmsg=True,
            )
        else:
            bad = check_log([args.log] if args.log else _push_range(), tokens)
        if bad:
            print(
                "LEAK GUARD: reword the message (git commit --amend, or an interactive "
                "rebase for older commits) before pushing.",
                file=sys.stderr,
            )
            return 1
        return 0

    if args.tracked:
        paths = _git("ls-files")
    elif args.files:
        paths = args.files
    else:
        paths = _git("diff", "--cached", "--name-only", "--diff-filter=AM")

    if check(paths, tokens):
        print(
            "LEAK GUARD: commit/CI blocked. For a genuine public file, override "
            "deliberately with `git commit --no-verify`.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
