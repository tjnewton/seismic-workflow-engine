"""Output caching for pipeline steps.

A step's cache key is a SHA-256 hash of its function path, canonicalized
parameters, source code, and the SWE version.  Prior run directories are
scanned (most recent first) for matching step name + hash.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
from pathlib import Path
from typing import Any

import swe
from swe.pipeline.serialization import SWEJSONEncoder

logger = logging.getLogger(__name__)

__all__ = ["compute_step_hash", "find_cached_result"]


def compute_step_hash(
    function_path: str,
    resolved_params: dict[str, Any],
    func: Any,
) -> str:
    """Compute a deterministic SHA-256 hash for a pipeline step.

    Parameters
    ----------
    function_path : str
        Dotted import path of the callable.
    resolved_params : dict
        Fully resolved keyword arguments.
    func : callable
        The imported function (used to hash its source code).

    Returns
    -------
    str
        Hex-encoded SHA-256 digest.
    """
    h = hashlib.sha256()
    h.update(function_path.encode())

    try:
        canonical = json.dumps(
            resolved_params, cls=SWEJSONEncoder, sort_keys=True
        )
    except (TypeError, ValueError):
        canonical = repr(sorted(resolved_params.items()))
    h.update(canonical.encode())

    try:
        source = inspect.getsource(func)
        h.update(source.encode())
    except (OSError, TypeError):
        logger.debug(
            "Could not retrieve source for %s; hash excludes source.",
            function_path,
        )

    h.update(swe.__version__.encode())

    return h.hexdigest()


def find_cached_result(
    step_name: str,
    step_hash: str,
    base_dir: Path,
) -> Any | None:
    """Search prior runs for a cached result matching the step hash.

    Parameters
    ----------
    step_name : str
        Pipeline step name.
    step_hash : str
        Expected hash (from :func:`compute_step_hash`).
    base_dir : Path
        Root directory containing run directories.

    Returns
    -------
    Any or None
        The deserialized output if found, otherwise ``None``.
    """
    if not base_dir.exists():
        return None

    run_dirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )

    for run_dir in run_dirs:
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue

        try:
            manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        step_info = manifest.get("steps", {}).get(step_name)
        if (
            step_info is not None
            and step_info.get("status") == "completed"
            and step_info.get("hash") == step_hash
        ):
            from swe.pipeline.serialization import (
                _VersionMismatchError,
                deserialize_output,
            )

            try:
                return deserialize_output(run_dir / "steps", step_name)
            except (_VersionMismatchError, FileNotFoundError):
                continue

    return None
