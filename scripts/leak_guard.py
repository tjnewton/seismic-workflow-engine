#!/usr/bin/env python3
"""Leak guard for the public seismic-workflow-engine repo.

Blocks accidental commits of private, oversized, or secret content. Generic by
design: this script contains NO private project names. Distinctive private
name-tokens live in an out-of-repo denylist referenced by ``$SWE_LEAK_DENYLIST``
(default ``~/.config/swe/leak-denylist``), one lowercase token per line, and/or
the ``$LEAK_TOKENS`` env var in CI. Used by ``.pre-commit-config.yaml`` (staged
files) and the CI ``leak-check`` job (``--tracked``).

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
    ".ms", ".mseed", ".msd", ".pkl", ".pickle", ".mrkr", ".tif", ".tiff",
    ".laz", ".las", ".mat", ".nc", ".h5", ".hdf5", ".npy", ".npz",
    ".tar", ".tgz", ".gz", ".zip", ".bundle",
}

SECRET_RE = re.compile(
    r"BEGIN (?:RSA|OPENSSH|EC|DSA|PGP) PRIVATE KEY"
    r"|AKIA[0-9A-Z]{16}"
    r"|ghp_[0-9A-Za-z]{36}"
    r"|xox[baprs]-[0-9A-Za-z-]+"
)


def _git(*args: str) -> list[str]:
    out = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    )
    return [line for line in out.stdout.splitlines() if line]


def _load_tokens() -> list[str]:
    tokens: list[str] = []
    path = Path(
        os.environ.get("SWE_LEAK_DENYLIST", Path.home() / ".config/swe/leak-denylist")
    )
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tracked", action="store_true", help="scan all git-tracked files (CI mode)"
    )
    ap.add_argument("files", nargs="*", help="files to check (pre-commit passes staged files)")
    args = ap.parse_args()

    if args.tracked:
        paths = _git("ls-files")
    elif args.files:
        paths = args.files
    else:
        paths = _git("diff", "--cached", "--name-only", "--diff-filter=AM")

    if check(paths, _load_tokens()):
        print(
            "LEAK GUARD: commit/CI blocked. For a genuine public file, override "
            "deliberately with `git commit --no-verify`.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
