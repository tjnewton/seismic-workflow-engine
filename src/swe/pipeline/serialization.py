"""Output serialization for pipeline step results.

Strategy: JSON only for values built purely from JSON-native types (dict
with str keys, list, str, int, float, bool, None), which load back equal and
of the same types. Everything else (NamedTuples, tuples, dataclasses, numpy
arrays and scalars, UTCDateTime, ObsPy Stream/Catalog/Trace) is pickled, so a
cache hit returns the same type the step returned.

Pickle files are HMAC-signed so that only files written by the same
SWE installation (with the same secret key) can be loaded.  This
mitigates arbitrary-code-execution risk from untrusted pickle files.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

import swe

logger = logging.getLogger(__name__)

__all__ = [
    "OUTPUT_FORMAT",
    "SWEJSONEncoder",
    "deserialize_output",
    "serialize_output",
]

# Checkpoint format. 2: JSON only for JSON-native values, so a loaded value keeps its type.
# Runs record it (manifest "output_format") and the cache key includes it.
OUTPUT_FORMAT = 2


def _get_hmac_key() -> bytes:
    """Return a per-installation HMAC key for signing pickle files.

    Uses the ``SWE_PICKLE_KEY`` environment variable if set, otherwise
    derives a key from the SWE package location on disk.  This ensures
    that pickles written by one installation cannot be loaded by another.
    """
    env_key = os.environ.get("SWE_PICKLE_KEY")
    if env_key:
        return env_key.encode()
    # Derive from the package install path + version
    identity = f"{swe.__file__}:{swe.__version__}".encode()
    return hashlib.sha256(identity).digest()


def _sig_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sig")


def _sign_file(path: Path) -> None:
    """Write an HMAC-SHA256 signature file alongside *path*."""
    key = _get_hmac_key()
    data = path.read_bytes()
    sig = hmac.new(key, data, hashlib.sha256).hexdigest()
    _sig_path(path).write_text(sig)


def _verify_file(path: Path) -> bool:
    """Return True if the HMAC signature for *path* is valid."""
    sig_path = _sig_path(path)
    if not sig_path.exists():
        return False
    key = _get_hmac_key()
    data = path.read_bytes()
    expected = hmac.new(key, data, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig_path.read_text().strip())


_JSON_LEAVES = (str, int, float, bool, type(None))


def _json_native(x: Any) -> bool:
    """True when *x* round-trips through plain JSON unchanged (exact types, no subclasses)."""
    t = type(x)
    if t in _JSON_LEAVES:
        return True
    if t is list:
        return all(_json_native(v) for v in x)
    if t is dict:
        return all(type(k) is str and _json_native(v) for k, v in x.items())
    return False


class SWEJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles common scientific types."""

    def default(self, o: Any) -> Any:
        # numpy scalars
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()

        # ObsPy UTCDateTime
        try:
            from obspy import UTCDateTime

            if isinstance(o, UTCDateTime):
                return str(o)
        except ImportError:
            pass

        # Frozen dataclasses (uncertainty types)
        if hasattr(type(o), "__dataclass_fields__"):
            from dataclasses import asdict

            d = asdict(o)
            # Convert numpy arrays inside dataclass dicts
            for key, val in d.items():
                if isinstance(val, np.ndarray):
                    d[key] = val.tolist()
            d["__type__"] = type(o).__name__
            return d

        # NamedTuple - subclass of tuple with _asdict
        if isinstance(o, tuple) and hasattr(o, "_asdict"):
            return o._asdict()

        return super().default(o)


def serialize_output(
    steps_dir: Path,
    step_name: str,
    output: Any,
) -> str:
    """Serialize a step output to disk.

    Parameters
    ----------
    steps_dir : Path
        Directory under the run where step artifacts are stored.
    step_name : str
        Name of the pipeline step.
    output : Any
        The value returned by the step function.

    Returns
    -------
    str
        Format used: ``"json"`` or ``"pickle"``, or ``"none"`` when the output can be
        neither pickled nor JSON-encoded.
    """
    json_path, pkl_path, meta_path = _clear_output(steps_dir, step_name)
    try:
        if _json_native(output):
            json_path.write_text(json.dumps(output, indent=2, allow_nan=False))
            return "json"
    except (ValueError, RecursionError):  # NaN/inf, or nesting too deep: pickle it
        pass

    # Pickle everything else, so the loaded value keeps its type.  HMAC signatures are
    # stored alongside the file; unsigned or tampered files are rejected on load (see
    # deserialize_output).  Version metadata is also recorded; mismatches are rejected.
    try:
        with open(pkl_path, "wb") as fh:
            pickle.dump(output, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except OSError:  # disk full, permissions: a real write failure, raised as before
        pkl_path.unlink(missing_ok=True)
        raise
    except Exception as exc:  # any pickling error: the step has run, so fall back, never abort
        pkl_path.unlink(missing_ok=True)  # partial file
        # as before this change for a value pickle cannot handle (e.g. a NamedTuple class
        # defined inside a function): the lossy JSON encoder, so a checkpoint still exists
        try:
            json_path.write_text(json.dumps(output, cls=SWEJSONEncoder, indent=2))
        except (TypeError, ValueError, RecursionError):
            json_path.unlink(missing_ok=True)
            logger.warning(
                "Could not serialize step '%s' output (type: %s): %s; "
                "checkpointing unavailable for this step.",
                step_name,
                type(output).__name__,
                exc,
            )
            return "none"
        logger.warning(
            "Step '%s' output (type: %s) cannot be pickled (%s); stored as JSON, so a cache "
            "hit or resume returns plain lists and dicts.",
            step_name,
            type(output).__name__,
            exc,
        )
        return "json"
    _sign_file(pkl_path)
    meta_path.write_text(
        json.dumps({"swe_version": swe.__version__, "python_version": sys.version})
    )
    logger.debug("Serialized step '%s' using pickle (type: %s)", step_name, type(output).__name__)
    return "pickle"


def _output_files(steps_dir: Path, step_name: str) -> tuple[Path, Path, Path, Path]:
    pkl_path = steps_dir / f"{step_name}.output.pkl"
    return (
        steps_dir / f"{step_name}.output.json",
        pkl_path,
        steps_dir / f"{step_name}.output.pkl.meta",
        _sig_path(pkl_path),
    )


def _clear_output(steps_dir: Path, step_name: str) -> tuple[Path, Path, Path]:
    """Remove a step's earlier output (either format): it would shadow or outlive a new one."""
    steps_dir.mkdir(parents=True, exist_ok=True)
    files = _output_files(steps_dir, step_name)
    for f in files:
        f.unlink(missing_ok=True)
    return files[0], files[1], files[2]


def deserialize_output(steps_dir: Path, step_name: str) -> Any:
    """Deserialize a step output from disk.

    Parameters
    ----------
    steps_dir : Path
        Directory under the run where step artifacts are stored.
    step_name : str
        Name of the pipeline step.

    Returns
    -------
    Any
        The deserialized output.

    Raises
    ------
    FileNotFoundError
        If no serialized output exists for this step.
    """
    json_path, pkl_path, meta_path, _ = _output_files(steps_dir, step_name)
    if json_path.exists():
        return json.loads(json_path.read_text())

    if pkl_path.exists():
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("swe_version") != swe.__version__:
                logger.warning(
                    "Pickle for step '%s' was written with SWE %s "
                    "(current: %s); treating as cache miss.",
                    step_name,
                    meta.get("swe_version"),
                    swe.__version__,
                )
                raise _VersionMismatchError(step_name)

        # Verify HMAC signature before loading
        if not _verify_file(pkl_path):
            logger.warning(
                "Pickle for step '%s' has no valid HMAC signature; rejecting as untrusted.",
                step_name,
            )
            raise FileNotFoundError(
                f"Unsigned or tampered pickle for step '{step_name}' "
                f"in {steps_dir}; refusing to load."
            )

        with open(pkl_path, "rb") as fh:
            return pickle.load(fh)

    raise FileNotFoundError(f"No serialized output found for step '{step_name}' in {steps_dir}")


class _VersionMismatchError(Exception):
    """Raised when a pickle was written by a different SWE version."""
