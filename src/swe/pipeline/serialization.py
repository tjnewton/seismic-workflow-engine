"""Output serialization for pipeline step results.

Strategy: try JSON first (handles dicts, lists, numpy scalars/arrays,
UTCDateTime, NamedTuples). Fall back to pickle only for objects that
cannot be JSON-encoded (ObsPy Stream/Catalog/Trace).

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

__all__ = ["serialize_output", "deserialize_output", "SWEJSONEncoder"]


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


def _sign_file(path: Path) -> None:
    """Write an HMAC-SHA256 signature file alongside *path*."""
    key = _get_hmac_key()
    data = path.read_bytes()
    sig = hmac.new(key, data, hashlib.sha256).hexdigest()
    path.with_suffix(path.suffix + ".sig").write_text(sig)


def _verify_file(path: Path) -> bool:
    """Return True if the HMAC signature for *path* is valid."""
    sig_path = path.with_suffix(path.suffix + ".sig")
    if not sig_path.exists():
        return False
    key = _get_hmac_key()
    data = path.read_bytes()
    expected = hmac.new(key, data, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig_path.read_text().strip())


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
        Format used: ``"json"`` or ``"pickle"``.
    """
    steps_dir.mkdir(parents=True, exist_ok=True)

    # Try JSON first
    try:
        json_path = steps_dir / f"{step_name}.output.json"
        payload = json.dumps(output, cls=SWEJSONEncoder, indent=2)
        json_path.write_text(payload)
        return "json"
    except (TypeError, ValueError, RecursionError):
        pass

    # Fall back to pickle for complex scientific objects (ObsPy Stream,
    # Catalog, Trace) that cannot be JSON-serialized.  HMAC signatures
    # are stored alongside the file; unsigned or tampered files are
    # rejected on load (see deserialize_output).  Version metadata is
    # also recorded; mismatches are treated as cache misses.
    pkl_path = steps_dir / f"{step_name}.output.pkl"
    meta_path = steps_dir / f"{step_name}.output.pkl.meta"
    try:
        with open(pkl_path, "wb") as fh:
            pickle.dump(output, fh, protocol=pickle.HIGHEST_PROTOCOL)  # noqa: S301
    except (pickle.PicklingError, AttributeError, TypeError) as exc:
        logger.warning(
            "Could not serialize step '%s' output (type: %s): %s; "
            "checkpointing unavailable for this step.",
            step_name,
            type(output).__name__,
            exc,
        )
        # Clean up partial files
        pkl_path.unlink(missing_ok=True)
        return "none"

    # Sign the pickle file
    _sign_file(pkl_path)

    meta = {
        "swe_version": swe.__version__,
        "python_version": sys.version,
    }
    meta_path.write_text(json.dumps(meta))
    logger.debug(
        "Serialized step '%s' using pickle (type: %s)",
        step_name,
        type(output).__name__,
    )
    return "pickle"


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
    json_path = steps_dir / f"{step_name}.output.json"
    if json_path.exists():
        return json.loads(json_path.read_text())

    pkl_path = steps_dir / f"{step_name}.output.pkl"
    if pkl_path.exists():
        meta_path = steps_dir / f"{step_name}.output.pkl.meta"
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
                "Pickle for step '%s' has no valid HMAC signature; "
                "rejecting as untrusted.",
                step_name,
            )
            raise FileNotFoundError(
                f"Unsigned or tampered pickle for step '{step_name}' "
                f"in {steps_dir}; refusing to load."
            )

        with open(pkl_path, "rb") as fh:
            return pickle.load(fh)  # noqa: S301

    raise FileNotFoundError(
        f"No serialized output found for step '{step_name}' in {steps_dir}"
    )


class _VersionMismatchError(Exception):
    """Raised when a pickle was written by a different SWE version."""
