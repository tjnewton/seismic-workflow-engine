"""Output caching for pipeline steps.

A step's cache key is a SHA-256 hash of its function path, canonicalized
parameters, source code, and the SWE version.  Prior run directories are
scanned (most recent first) for matching step name + hash.
"""

from __future__ import annotations

import datetime
import hashlib
import inspect
import json
import logging
import struct
import uuid
from decimal import Decimal
from pathlib import Path, PosixPath, PurePosixPath, PureWindowsPath, WindowsPath
from typing import Any

import numpy as np
from obspy import UTCDateTime

import swe
from swe.pipeline.serialization import OUTPUT_FORMAT

logger = logging.getLogger(__name__)

# prefix of the hash of a step whose params have no exact key: it is never looked up
UNCACHEABLE = "uncacheable:"

__all__ = ["UNCACHEABLE", "compute_step_hash", "find_cached_result"]


def _name(t: type) -> list[str]:
    return [t.__module__, t.__qualname__]


_PATHS = (PurePosixPath, PureWindowsPath, PosixPath, WindowsPath)  # concrete classes only


def _tz_key(tz: Any) -> Any:
    """Fixed UTC offsets by name (``isoformat`` already carries the offset); any other tzinfo
    (an IANA zone's name does not pin its rules) has no exact key."""
    if tz is None:
        return None
    if type(tz) is datetime.timezone:
        return ["timezone", tz.tzname(None)]
    raise _NoExactKey(f"no exact cache key for tzinfo {type(tz).__qualname__}")


def _digest(a: np.ndarray, fortran: bool) -> str:
    """SHA-256 of an array's bytes in its memory order (the key records which one); only an
    array that is neither C- nor F-contiguous is copied, into C order."""
    flat = np.ascontiguousarray(a.T if fortran else a).reshape(-1)  # a.T of F order: a view
    # a uint8 view: datetime64 and timedelta64 arrays expose no buffer of their own
    return hashlib.sha256(flat.view(np.uint8).data).hexdigest()


class _NoExactKey(Exception):
    """A value has no exact cache key (by design: its step is uncacheable)."""


def _key(v: Any) -> str:
    """Exact, type-tagged JSON text of *v*.

    Contract (memoization semantics, as for ``functools.lru_cache``): two values that are not
    equal, compared with their exact types (arrays by dtype, shape and bytes), never share a
    key. Properties outside value equality are not part of the key: memory layout and
    aliasing, writeability, set iteration order. A few of them are keyed anyway (dict order,
    Fortran order, writeability), which can only turn equal values into misses.
    Every value becomes ``[tag, payload]``, so a user dict can never pose as a tagged value.
    Only exact types that cannot carry unkeyed instance state are keyed: immutable builtins,
    the builtin containers, plain ndarrays and numpy scalars, paths, Decimal, dates and
    times, and UTCDateTime with no attributes set on it. Dataclasses, NamedTuples, masked
    arrays and everything else raise ``_NoExactKey`` (their steps are uncacheable).
    A container's text is joined from its elements' text, so no element is serialized twice.
    """
    t: Any = type(v)
    if t in (list, tuple):
        return _tagged(t.__name__, [_key(x) for x in v])
    if t in (set, frozenset):  # sorted by text: iteration order is not part of the key
        return _tagged(t.__name__, sorted(_key(x) for x in v))
    if t is dict:  # in insertion order: a step that iterates the dict sees it
        return _tagged("dict", [f"[{_key(k)}, {_key(x)}]" for k, x in v.items()])
    return json.dumps(_leaf(v))


def _tagged(tag: str, items: list[str]) -> str:
    """``json.dumps([tag, [...]])`` of items that are already JSON text."""
    return f'["{tag}", [{", ".join(items)}]]'


def _leaf(v: Any) -> list[Any]:
    """``[tag, payload]`` of a value that is not a builtin container (see ``_key``)."""
    t: Any = type(v)
    if t is float:  # by bit pattern: NaN payloads and -0.0 stay distinct
        return ["float", struct.pack("<d", v).hex()]
    if t is int:  # as hex: no JSON digit limit
        return ["int", hex(v)]
    if v is None or t in (bool, str):  # exact types: IntEnum, np.float64 go on
        return [t.__name__, v]
    if t is np.ndarray and not v.dtype.hasobject:  # exact: a masked array holds a __dict__
        layout = bool(v.flags.f_contiguous and not v.flags.c_contiguous)  # visible to the step
        meta = [str(v.dtype), v.dtype.char, list(v.shape), layout, bool(v.flags.writeable)]
        return ["ndarray", [*meta, _digest(v, layout)]]
    if isinstance(v, np.generic) and t.__module__ == "numpy" and not v.dtype.hasobject:
        # object fields would be pointers; type and dtype char tell apart same-width aliases
        return ["numpy", [t.__name__, str(v.dtype), v.dtype.char, v.tobytes().hex()]]
    # exact types only: a subclass may carry state its text form omits
    if t is UTCDateTime:
        # no attribute beyond a fresh instance's (newer obspy uses __slots__: no __dict__)
        fresh = UTCDateTime(ns=v.ns, precision=v.precision)
        if set(getattr(v, "__dict__", ())) == set(getattr(fresh, "__dict__", ())):
            return ["UTCDateTime", [v.ns, v.precision]]
    if t in _PATHS or t is Decimal:
        return [_name(t), str(v)]
    if t in (datetime.date, datetime.datetime, datetime.time):
        return [
            _name(t),
            [v.isoformat(), _tz_key(getattr(v, "tzinfo", None)), getattr(v, "fold", 0)],
        ]
    raise _NoExactKey(f"no exact cache key for {t.__qualname__}")


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
        Hex-encoded SHA-256 digest; for params with no exact key (see ``_key``),
        ``UNCACHEABLE`` plus a random suffix, which never matches.
    """
    h = hashlib.sha256()
    h.update(f"step-output-format-{OUTPUT_FORMAT}".encode())  # older (lossy) caches miss
    h.update(function_path.encode())

    try:
        canonical = _key(resolved_params)
    except MemoryError:
        raise
    except Exception as exc:  # any key-building error means no exact key, never a failed step
        # no exact key (e.g. a Stream or Catalog passed by ${step}): uncacheable
        # ponytail: chain the upstream step's own hash instead, if such steps need caching
        by_design = isinstance(exc, (_NoExactKey, RecursionError))  # RecursionError: too deep
        level = logging.DEBUG if by_design else logging.WARNING
        logger.log(level, "%s is uncacheable: %s: %s", function_path, type(exc).__name__, exc)
        return UNCACHEABLE + uuid.uuid4().hex
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
    found = _find_cached(step_name, step_hash, base_dir)
    return None if found is None else found[0]


def _find_cached(step_name: str, step_hash: str, base_dir: Path) -> tuple[Any, Path] | None:
    """As :func:`find_cached_result`, plus the steps directory the output was loaded from."""
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
                return deserialize_output(run_dir / "steps", step_name), run_dir / "steps"
            except (_VersionMismatchError, FileNotFoundError):
                continue
            except Exception as exc:  # e.g. a pickled class renamed since: a miss, not a crash
                logger.warning("Cached output of %s in %s unreadable (%s)", step_name, run_dir, exc)
                continue

    return None
