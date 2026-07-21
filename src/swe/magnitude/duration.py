"""Coda duration magnitude (Md) estimation.

Computes coda duration from the envelope decay of a seismogram and
derives duration magnitude using the Lee et al. (1972) empirical
relation.

References
----------
Lee, W.H.K., Bennett, R.E. & Meagher, K.L. (1972). A method of
    estimating magnitude of local earthquakes from signal duration.
    USGS Open-File Report 72-223.
"""

from __future__ import annotations

import logging
from typing import NamedTuple, TYPE_CHECKING

import numpy as np
from scipy.signal import hilbert

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from obspy.core.trace import Trace
    from obspy.core.utcdatetime import UTCDateTime

logger = logging.getLogger(__name__)

__all__ = [
    "DurationMagnitudeBatchResult",
    "DurationMagnitudeResult",
    "coda_duration",
    "duration_magnitude",
    "duration_magnitudes_batch",
]


class DurationMagnitudeResult(NamedTuple):
    """Result from duration magnitude estimation."""

    md: float
    """Duration magnitude."""
    coda_duration: float
    """Coda duration in seconds from P-onset."""
    snr_threshold_used: float
    """SNR threshold that defined the coda end."""


def coda_duration(
    trace: Trace,
    p_time: UTCDateTime,
    *,
    noise_window: float = 5.0,
    snr_threshold: float = 2.0,
    smooth_window: float = 1.0,
) -> float:
    """Measure coda duration from the signal envelope.

    The coda end is defined as the first time after the P-onset where
    the smoothed envelope amplitude falls below ``snr_threshold`` times
    the pre-event noise level.

    Parameters
    ----------
    trace : obspy.Trace
        Waveform data containing the event.
    p_time : UTCDateTime
        P-wave arrival time.
    noise_window : float
        Duration (seconds) of the pre-P noise window used to estimate
        the noise level.
    snr_threshold : float
        The envelope must drop below ``snr_threshold * noise_rms``
        to end the coda.
    smooth_window : float
        Envelope smoothing window length in seconds.

    Returns
    -------
    float
        Coda duration in seconds from P-onset.  Returns ``0.0`` if
        the coda end cannot be determined (e.g. trace too short).
    """
    sr = trace.stats.sampling_rate
    t_start = trace.stats.starttime

    # Indices
    p_idx = int((p_time - t_start) * sr)
    noise_start = max(0, int((p_time - noise_window - t_start) * sr))
    noise_end = max(0, p_idx)

    if noise_end <= noise_start or p_idx < 0 or p_idx >= len(trace.data):
        logger.warning("Trace too short for coda duration measurement")
        return 0.0

    # Compute envelope via analytic signal
    envelope = np.abs(hilbert(trace.data.astype(np.float64)))

    # Smooth envelope
    smooth_samples = max(1, int(smooth_window * sr))
    kernel = np.ones(smooth_samples) / smooth_samples
    envelope = np.convolve(envelope, kernel, mode="same")

    # Noise level
    noise_rms = np.sqrt(np.mean(envelope[noise_start:noise_end] ** 2))
    if noise_rms == 0:
        noise_rms = 1e-30

    threshold = snr_threshold * noise_rms

    # Find coda end: first sample after P where envelope < threshold
    post_p = envelope[p_idx:]
    below = np.where(post_p < threshold)[0]

    if len(below) == 0:
        # Envelope never drops below threshold; use full trace
        coda_samples = len(post_p)
    else:
        coda_samples = below[0]

    return coda_samples / sr


def duration_magnitude(
    trace: Trace,
    p_time: UTCDateTime,
    *,
    a0: float = -0.87,
    a1: float = 2.00,
    a2: float = 0.0035,
    distance_km: float = 0.0,
    noise_window: float = 5.0,
    snr_threshold: float = 2.0,
    smooth_window: float = 1.0,
) -> DurationMagnitudeResult:
    """Compute duration magnitude (Md) from coda duration.

    Uses the general form of Lee et al. (1972)::

        Md = a0 + a1 * log10(tau) + a2 * delta

    where *tau* is the coda duration in seconds and *delta* is the
    epicentral distance in km.

    Parameters
    ----------
    trace : obspy.Trace
        Waveform data containing the event.
    p_time : UTCDateTime
        P-wave arrival time.
    a0 : float
        Intercept term.
    a1 : float
        Duration coefficient.
    a2 : float
        Distance coefficient.
    distance_km : float
        Epicentral distance in kilometres.
    noise_window : float
        Duration (seconds) of the pre-P noise window.
    snr_threshold : float
        Envelope SNR threshold for coda end.
    smooth_window : float
        Envelope smoothing window in seconds.

    Returns
    -------
    DurationMagnitudeResult
        Duration magnitude, coda duration, and threshold used.
    """
    tau = coda_duration(
        trace,
        p_time,
        noise_window=noise_window,
        snr_threshold=snr_threshold,
        smooth_window=smooth_window,
    )

    if tau <= 0:
        logger.warning("Coda duration is zero; returning Md=NaN")
        return DurationMagnitudeResult(
            md=float("nan"),
            coda_duration=0.0,
            snr_threshold_used=snr_threshold,
        )

    md = a0 + a1 * np.log10(tau) + a2 * distance_km

    return DurationMagnitudeResult(
        md=float(md),
        coda_duration=tau,
        snr_threshold_used=snr_threshold,
    )


class DurationMagnitudeBatchResult(NamedTuple):
    """Result from multi-station duration magnitude estimation."""

    md: float
    """Mean Md across valid stations."""
    md_per_station: list[float]
    """Individual station Md values."""
    station_names: list[str]
    """Station identifiers."""
    md_std: float = 0.0
    """Standard deviation of valid station Md values."""
    md_stderr: float = 0.0
    """Standard error of the mean: ``md_std / sqrt(n_valid)``."""
    n_valid_stations: int = 0
    """Number of stations with valid Md values."""


def duration_magnitudes_batch(
    traces: list[Trace],
    p_times: list[UTCDateTime],
    *,
    a0: float = -0.87,
    a1: float = 2.00,
    a2: float = 0.0035,
    distances_km: list[float] | None = None,
    noise_window: float = 5.0,
    snr_threshold: float = 2.0,
    smooth_window: float = 1.0,
    station_names: list[str] | None = None,
) -> DurationMagnitudeBatchResult:
    """Compute duration magnitudes across multiple stations with uncertainty.

    Parameters
    ----------
    traces : list of Trace
        Waveform data, one per station.
    p_times : list of UTCDateTime
        P-wave arrival times, one per trace.
    a0, a1, a2 : float
        Duration magnitude coefficients.
    distances_km : list of float, optional
        Epicentral distances in km. If not provided, distance term is zero.
    noise_window, snr_threshold, smooth_window : float
        Parameters passed to :func:`duration_magnitude`.
    station_names : list of str, optional
        Station identifiers. If not provided, derived from trace stats.

    Returns
    -------
    DurationMagnitudeBatchResult
        Mean Md, per-station values, and uncertainty.
    """
    mds: list[float] = []
    names: list[str] = []

    for i, trace in enumerate(traces):
        dist = distances_km[i] if distances_km is not None else 0.0
        result = duration_magnitude(
            trace,
            p_times[i],
            a0=a0,
            a1=a1,
            a2=a2,
            distance_km=dist,
            noise_window=noise_window,
            snr_threshold=snr_threshold,
            smooth_window=smooth_window,
        )
        mds.append(result.md)
        if station_names is not None:
            names.append(station_names[i])
        else:
            names.append(
                f"{trace.stats.network}.{trace.stats.station}.{trace.stats.channel}"
            )

    valid = [m for m in mds if not np.isnan(m)]
    n_valid = len(valid)

    if n_valid > 0:
        mean_md = float(np.mean(valid))
    else:
        mean_md = float("nan")

    if n_valid > 1:
        md_std = float(np.std(valid, ddof=1))
        md_stderr = md_std / np.sqrt(n_valid)
    else:
        md_std = 0.0
        md_stderr = 0.0

    return DurationMagnitudeBatchResult(
        md=mean_md,
        md_per_station=mds,
        station_names=names,
        md_std=md_std,
        md_stderr=md_stderr,
        n_valid_stations=n_valid,
    )
