"""Injection-recovery: detection probability of a detector, with exact binomial intervals.

Plant a known signal into data (arrays, rasters or traces), run a detector on each realization,
and report the detection probability per parameter set with a Clopper-Pearson interval. The
harness is detector-agnostic: any ``detect(data, params) -> bool`` works.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from fractions import Fraction
from typing import Any

import numpy as np
import pandas as pd
from obspy import Trace, UTCDateTime
from scipy.stats import binomtest

from swe.utils.helpers import is_integer


def clopper_pearson(
    successes: int, trials: int, *, confidence: float = 0.95
) -> tuple[float, float]:
    """Exact (Clopper-Pearson) two-sided binomial confidence interval.

    A thin wrapper around ``scipy.stats.binomtest(...).proportion_ci(method="exact")``.

    Parameters
    ----------
    successes, trials : int
        Number of detections and of trials (``0 <= successes <= trials``, ``trials >= 1``).
    confidence : float
        Two-sided confidence level in (0, 1).

    Returns
    -------
    (float, float)
        Lower and upper bounds; 0 and 1 at the edges (``successes`` of 0 or ``trials``).

    Raises
    ------
    ValueError
        For counts or a confidence level outside their valid ranges.
    """
    if not all(is_integer(v) for v in (successes, trials)):
        raise ValueError(f"successes and trials must be integers, got {successes!r}/{trials!r}")
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError(f"need 0 <= successes <= trials and trials >= 1, got {successes}/{trials}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    ci = binomtest(successes, trials).proportion_ci(confidence_level=confidence, method="exact")
    return float(ci.low), float(ci.high)


def inject(
    data: np.ndarray, signal: np.ndarray, *, offset: int | Sequence[int], scale: float = 1.0
) -> np.ndarray:
    """Return a copy of *data* with ``scale * signal`` added at *offset* (any number of dimensions).

    Parameters
    ----------
    data : ndarray
        Array or raster to inject into (not modified).
    signal : ndarray
        Signal with the same number of dimensions as *data*.
    offset : int or sequence of int
        Index in *data* of ``signal[0, 0, ...]``; integers only (bool rejected).
    scale : float
        Amplitude multiplier.

    Returns
    -------
    ndarray
        ``data`` plus the scaled signal, as float (complex if any input is complex). A masked
        array with masked samples stays masked (same mask and fill value; nothing is added under
        the mask); one with no masked samples comes back as a plain array.

    Raises
    ------
    ValueError
        If the dimensions differ or the signal does not fit inside *data* at *offset*.
    """
    mask = fill = None
    if np.ma.isMaskedArray(data):
        mask, fill = np.ma.getmaskarray(data), data.fill_value
    raw_data = np.asarray(np.ma.getdata(data))
    # masked signal samples contribute nothing (their raw values are not injected)
    signal = np.ma.filled(signal, 0) if np.ma.isMaskedArray(signal) else np.asarray(signal)
    dtype = np.result_type(
        raw_data, signal, np.asarray(scale), float
    )  # complex inputs stay complex
    data = raw_data.astype(dtype)
    signal = signal.astype(dtype)
    raw = np.atleast_1d(np.asarray(offset, dtype=object))
    if raw.ndim != 1 or not all(is_integer(o) for o in raw):
        raise ValueError(f"offset must be an integer or a 1-d sequence of integers, got {offset!r}")
    idx: tuple[int, ...] = tuple(int(o) for o in raw)
    if signal.ndim != data.ndim or len(idx) != data.ndim:
        raise ValueError("signal, data and offset must have the same number of dimensions")
    if any(o < 0 or o + n > m for o, n, m in zip(idx, signal.shape, data.shape, strict=True)):
        raise ValueError(
            f"signal of shape {signal.shape} at {idx} does not fit in data of shape {data.shape}"
        )
    out: np.ndarray = data.copy()
    out[tuple(slice(o, o + n) for o, n in zip(idx, signal.shape, strict=True))] += scale * signal
    if mask is not None and mask.any():  # gaps stay gaps: nothing is added under the mask
        out = np.ma.masked_array(np.where(mask, data, out), mask=mask, fill_value=fill)
    return out


def inject_trace(
    trace: Trace, signal: np.ndarray, *, time: UTCDateTime, scale: float = 1.0
) -> Trace:
    """Return a copy of *trace* with ``scale * signal`` added starting at the sample nearest *time*.

    *signal* is sampled at the trace's sampling rate; a *time* exactly halfway between samples
    rounds to the later one. The returned data are float64 (complex128 if *signal* or *scale* is
    complex, which MiniSEED cannot store), and any MiniSEED encoding from the input is dropped,
    so an unmasked real result can be written to MiniSEED again. Masked gaps stay
    masked; ObsPy cannot write masked data to MiniSEED, so split or fill the trace first.

    Raises
    ------
    ValueError
        If the signal does not fit inside the trace from *time*.
    """
    # exact rational arithmetic in nanoseconds; ties round up to the later sample
    x = Fraction(time.ns - trace.stats.starttime.ns) * Fraction(trace.stats.sampling_rate) / 10**9
    i0 = math.floor(x + Fraction(1, 2))
    out = Trace(data=inject(trace.data, signal, offset=i0, scale=scale), header=trace.stats.copy())
    # the data are float64 now; a stale integer MiniSEED encoding would break writing
    out.stats.pop("mseed", None)
    return out


def _canonical(v: Any) -> Any:
    """Unambiguous JSON-ready form of a parameter value (used only to derive stream keys).

    Whole numbers become exact ints (``1``, ``1.0`` and ``np.float64(1.0)`` agree; large ints
    stay distinct); other floats stay float. Every container is tagged with its kind, so
    a mapping, a NamedTuple and a dataclass never collide, and mapping keys keep their own
    canonical form (``1`` and ``"1"`` differ). Unsupported types raise ``TypeError``.
    """
    if v is None or isinstance(v, str):
        return v
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return int(f) if f.is_integer() else f
    if isinstance(v, (complex, np.complexfloating)):
        return {"complex": [_canonical(v.real), _canonical(v.imag)]}
    if isinstance(v, np.ndarray):
        if v.ndim == 0:  # a 0-d array is a scalar
            return _canonical(v.item())
        return {"array": _canonical(v.tolist())}
    if isinstance(v, tuple) and hasattr(v, "_asdict"):
        return {"namedtuple": type(v).__qualname__, "fields": _canonical(dict(v._asdict()))}
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        fields = {f.name: getattr(v, f.name) for f in dataclasses.fields(v)}
        return {"dataclass": type(v).__qualname__, "fields": _canonical(fields)}
    if isinstance(v, Mapping):
        items = [[_canonical(k), _canonical(x)] for k, x in v.items()]
        return {"map": sorted(items, key=lambda kv: json.dumps(kv[0], sort_keys=True))}
    if isinstance(v, (list, tuple)):
        return {"seq": [_canonical(x) for x in v]}
    raise TypeError(f"unsupported parameter value type {type(v).__name__}")


def _params_key(params: Mapping[str, Any]) -> int:
    """Stable 64-bit key of a parameter set: the same across key order, numeric types and
    processes. Values may be numbers, strings, booleans, None, sequences, mappings, arrays,
    NamedTuples or dataclasses (nested freely).
    """
    if not all(isinstance(k, str) for k in params):
        raise ValueError(f"grid parameter names must be strings, got {list(params)}")
    try:
        text = json.dumps(_canonical(dict(params)), sort_keys=True)
    except TypeError as exc:
        raise ValueError(f"grid parameter values must be JSON-serializable: {exc}") from exc
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def injection_recovery(
    *,
    make_trial: Callable[[np.random.Generator, Mapping[str, Any]], Any],
    detect: Callable[[Any, Mapping[str, Any]], bool],
    grid: Iterable[Mapping[str, Any]],
    n_trials: int,
    random_state: int = 0,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Detection probability over a parameter grid by repeated injection.

    Parameters
    ----------
    make_trial : callable
        ``make_trial(rng, params) -> data``: one realization with the signal for *params*
        planted (noise drawn from *rng*).
    detect : callable
        ``detect(data, params) -> bool``.
    grid : iterable of mapping
        Parameter sets (e.g. amplitude, size, position); any iterable, read once.
    n_trials : int
        Realizations per parameter set.
    random_state : int
        Non-negative root seed (named as elsewhere in SWE). Each grid entry's random stream is
        seeded from ``(random_state, its parameters)``, so a row's result depends only on the
        seed and its own parameters, not on the other entries, their order, or numeric types
        (``1`` vs ``1.0`` vs ``np.float64(1.0)``). Entries with equal parameters get identical
        streams. The seed key itself is stable across processes and NumPy versions, but NumPy
        does not promise identical draws from ``Generator`` methods across versions (NEP 19),
        so archive results with the NumPy version. Parameter names must be strings and values
        JSON-serializable (numbers, strings, booleans, None, sequences, arrays, NamedTuples,
        dataclasses).
    confidence : float
        Level of the Clopper-Pearson interval.

    Returns
    -------
    DataFrame
        One row per parameter set: its parameters, then ``n_trials, n_detected, p_detect,
        ci_low, ci_high``.

    Raises
    ------
    ValueError
        If *n_trials* is not an integer of at least 1, *grid* is empty, *random_state* is not
        a non-negative integer (bool rejected), a parameter name collides with a result column,
        *confidence* is outside (0, 1), entries have different parameter names, two entries
        have equal parameters (they would share a
        random stream; raise *n_trials* instead), or any entry's names or values are invalid.
        All of this is checked before any trial runs.
    """
    grid = list(grid)  # a generator would otherwise be consumed by the checks below
    if not is_integer(random_state) or random_state < 0:
        raise ValueError(f"random_state must be a non-negative integer, got {random_state!r}")
    if not is_integer(n_trials) or n_trials < 1 or not grid:
        raise ValueError(
            f"need an integer n_trials >= 1 and a non-empty grid, got n_trials={n_trials!r}"
        )
    reserved = ["n_trials", "n_detected", "p_detect", "ci_low", "ci_high"]
    clash = sorted({k for p in grid for k in p if k in reserved})
    if clash:
        raise ValueError(f"grid parameter names collide with result columns: {clash}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    names = set(grid[0])
    if any(set(p) != names for p in grid):
        raise ValueError("every grid entry must have the same parameter names")
    keys = [_params_key(p) for p in grid]  # validate every entry before running any trial
    if len(set(keys)) != len(keys):
        raise ValueError(
            "duplicate grid entries would share one random stream; raise n_trials instead"
        )
    rows = []
    for params, key in zip(grid, keys, strict=True):
        rng = np.random.default_rng(np.random.SeedSequence([int(random_state), key]))
        hits = sum(bool(detect(make_trial(rng, params), params)) for _ in range(n_trials))
        lo, hi = clopper_pearson(hits, n_trials, confidence=confidence)
        rows.append(
            {
                **params,
                "n_trials": n_trials,
                "n_detected": hits,
                "p_detect": hits / n_trials,
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    return pd.DataFrame(rows, columns=list(grid[0]) + reserved)
