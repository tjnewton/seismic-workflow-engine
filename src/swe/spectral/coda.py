"""Coda-Q estimation from seismogram envelope decay.

Estimates the quality factor Q from coda wave amplitude decay,
both for single frequency bands and across multiple octave bands
with power-law fitting.

References
----------
Aki, K. & Chouet, B. (1975). Origin of coda waves: source, attenuation,
    and scattering effects. J. Geophys. Res., 80(23), 3322-3342.
Sato, H. & Fehler, M. (1998). Seismic Wave Propagation and Scattering
    in the Heterogeneous Earth. Springer.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from scipy.signal import hilbert
from scipy.stats import linregress

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from obspy.core.trace import Trace
    from obspy.core.utcdatetime import UTCDateTime

logger = logging.getLogger(__name__)

__all__ = [
    "CodaQFrequencyResult",
    "CodaQResult",
    "coda_q_frequency_bands",
    "estimate_coda_q",
]

# Default octave frequency bands (Hz)
_DEFAULT_BANDS: list[tuple[float, float]] = [
    (0.5, 1.0),
    (1.0, 2.0),
    (2.0, 4.0),
    (4.0, 8.0),
    (8.0, 16.0),
    (16.0, 32.0),
]


class CodaQResult(NamedTuple):
    """Result from single-band coda-Q estimation."""

    qc: float
    """Coda quality factor."""
    qc_std: float
    """Standard error of Qc from linear regression."""
    central_frequency: float
    """Central frequency of the band (Hz)."""
    decay_rate: float
    """Decay rate from regression slope."""
    fit_r_squared: float
    """R-squared of the linear fit."""
    coda_start_time: float
    """Coda window start time (seconds from trace start)."""
    coda_end_time: float
    """Coda window end time (seconds from trace start)."""
    station: str
    """Station identifier."""


class CodaQFrequencyResult(NamedTuple):
    """Result from multi-band coda-Q estimation."""

    frequencies: NDArray[np.floating]
    """Central frequencies of each band (Hz)."""
    qc_values: NDArray[np.floating]
    """Qc values for each band."""
    qc_errors: NDArray[np.floating]
    """Standard errors of Qc for each band."""
    q0: float
    """Q at 1 Hz from power-law fit Qc(f) = Q0 * f^eta."""
    eta: float
    """Frequency exponent from power-law fit."""
    r_squared_values: NDArray[np.floating]
    """R-squared values for each band's regression."""
    station: str
    """Station identifier."""


def estimate_coda_q(
    trace: Trace,
    origin_time: UTCDateTime,
    *,
    freq_min: float = 1.0,
    freq_max: float = 10.0,
    coda_start_factor: float = 2.0,
    coda_end_snr: float = 2.0,
    noise_window: float = 5.0,
    smooth_window: float = 1.0,
) -> CodaQResult:
    """Estimate coda-Q from the envelope decay of a bandpass-filtered trace.

    The coda window starts at ``coda_start_factor`` times the
    lapse time (time from origin to trace start + 1/4 of the trace
    duration), and ends when the envelope SNR drops below
    ``coda_end_snr``.

    Parameters
    ----------
    trace : obspy.Trace
        Input waveform containing the coda.
    origin_time : UTCDateTime
        Event origin time.
    freq_min : float
        Lower corner of the bandpass filter (Hz).
    freq_max : float
        Upper corner of the bandpass filter (Hz).
    coda_start_factor : float
        Multiplier for the lapse time to define coda start.
    coda_end_snr : float
        SNR threshold for coda end determination.
    noise_window : float
        Pre-event noise window length in seconds.
    smooth_window : float
        Envelope smoothing window length in seconds.

    Returns
    -------
    CodaQResult
        Coda-Q estimate with fit statistics.
    """
    station = f"{trace.stats.network}.{trace.stats.station}.{trace.stats.channel}"
    central_freq = np.sqrt(freq_min * freq_max)

    nan_result = CodaQResult(
        qc=float("nan"),
        qc_std=float("nan"),
        central_frequency=float(central_freq),
        decay_rate=float("nan"),
        fit_r_squared=0.0,
        coda_start_time=0.0,
        coda_end_time=0.0,
        station=station,
    )

    sr = trace.stats.sampling_rate
    t_start = trace.stats.starttime

    # Bandpass filter
    tr = trace.copy()
    tr.filter("bandpass", freqmin=freq_min, freqmax=freq_max, corners=4)

    # Compute envelope via Hilbert transform
    envelope = np.abs(hilbert(tr.data.astype(np.float64)))

    # Smooth envelope
    smooth_samples = max(1, int(smooth_window * sr))
    kernel = np.ones(smooth_samples) / smooth_samples
    envelope = np.convolve(envelope, kernel, mode="same")

    # Time array relative to trace start
    times = np.arange(len(envelope)) / sr

    # Origin time relative to trace start
    origin_rel = float(origin_time - t_start)

    # Lapse time heuristic: origin offset + 1/4 of trace duration
    trace_duration = len(envelope) / sr
    lapse_time = max(origin_rel, 0.0) + trace_duration / 4.0

    # Coda start
    coda_start_rel = coda_start_factor * lapse_time
    if coda_start_rel < 0:
        coda_start_rel = 0.0

    # Noise level from pre-origin window
    noise_end_idx = max(0, int(max(origin_rel, 0.0) * sr))
    noise_start_idx = max(0, int((max(origin_rel, 0.0) - noise_window) * sr))
    if noise_end_idx > noise_start_idx:
        noise_rms = np.sqrt(np.mean(envelope[noise_start_idx:noise_end_idx] ** 2))
    else:
        noise_rms = np.median(envelope) * 0.1

    if noise_rms == 0:
        noise_rms = 1e-30
    threshold = coda_end_snr * noise_rms

    # Coda end: first point after coda_start where envelope < threshold
    coda_start_idx = int(coda_start_rel * sr)
    if coda_start_idx >= len(envelope):
        logger.warning("Coda start beyond trace end for %s", station)
        return nan_result

    post_coda_start = envelope[coda_start_idx:]
    below = np.where(post_coda_start < threshold)[0]
    if len(below) > 0:
        coda_end_idx = coda_start_idx + below[0]
    else:
        coda_end_idx = len(envelope) - 1

    coda_start_time = coda_start_idx / sr
    coda_end_time = coda_end_idx / sr

    # Need enough points for regression
    if coda_end_idx - coda_start_idx < 10:
        logger.warning(
            "Coda window too short (%d samples) for %s",
            coda_end_idx - coda_start_idx,
            station,
        )
        return nan_result

    # Regression on ln(A(t) * t) vs t for single scattering model
    # A(t) = A0 * t^(-1) * exp(-pi*f*t/Qc)
    # => ln(A(t)*t) = ln(A0) - pi*f*t/Qc
    coda_times = times[coda_start_idx:coda_end_idx]
    coda_env = envelope[coda_start_idx:coda_end_idx]

    # Lapse time from origin for the coda window
    t_lapse = coda_times - origin_rel
    # Avoid non-positive lapse times
    valid = (t_lapse > 0) & (coda_env > 0)
    if np.sum(valid) < 10:
        logger.warning("Too few valid coda samples for %s", station)
        return nan_result

    t_valid = t_lapse[valid]
    env_valid = coda_env[valid]

    y = np.log(env_valid * t_valid)
    x = t_valid

    reg = linregress(x, y)
    slope = reg.slope
    r_squared = reg.rvalue**2

    # Qc = -pi * f / slope
    if slope >= 0:
        logger.warning(
            "Non-negative slope (%.4f) for %s; returning NaN Qc",
            slope,
            station,
        )
        return nan_result

    qc = -np.pi * central_freq / slope

    # Standard error of Qc from slope error
    if reg.stderr > 0 and slope != 0:
        qc_std = qc * (reg.stderr / abs(slope))
    else:
        qc_std = 0.0

    return CodaQResult(
        qc=float(qc),
        qc_std=float(qc_std),
        central_frequency=float(central_freq),
        decay_rate=float(slope),
        fit_r_squared=float(r_squared),
        coda_start_time=float(coda_start_time),
        coda_end_time=float(coda_end_time),
        station=station,
    )


def coda_q_frequency_bands(
    trace: Trace,
    origin_time: UTCDateTime,
    *,
    frequency_bands: list[tuple[float, float]] | None = None,
    min_r_squared: float = 0.5,
    coda_start_factor: float = 2.0,
    coda_end_snr: float = 2.0,
    noise_window: float = 5.0,
    smooth_window: float = 1.0,
) -> CodaQFrequencyResult:
    """Estimate coda-Q across multiple frequency bands.

    Computes Qc for each band, then fits a power-law
    ``Qc(f) = Q0 * f^eta`` to the valid results.

    Parameters
    ----------
    trace : obspy.Trace
        Input waveform.
    origin_time : UTCDateTime
        Event origin time.
    frequency_bands : list of (float, float), optional
        List of ``(freq_min, freq_max)`` tuples. Defaults to octave
        bands from 0.5 to 32 Hz.
    min_r_squared : float
        Minimum R-squared to include a band in the power-law fit.
    coda_start_factor : float
        Multiplier for the lapse time to define coda start.
    coda_end_snr : float
        SNR threshold for coda end.
    noise_window : float
        Pre-event noise window length in seconds.
    smooth_window : float
        Envelope smoothing window in seconds.

    Returns
    -------
    CodaQFrequencyResult
        Multi-band Qc results with power-law fit parameters.
    """
    station = f"{trace.stats.network}.{trace.stats.station}.{trace.stats.channel}"

    if frequency_bands is None:
        frequency_bands = _DEFAULT_BANDS

    freqs_out: list[float] = []
    qc_out: list[float] = []
    qc_err_out: list[float] = []
    r2_out: list[float] = []

    for fmin, fmax in frequency_bands:
        # Skip bands above Nyquist
        nyquist = trace.stats.sampling_rate / 2.0
        if fmin >= nyquist:
            continue

        fmax_eff = min(fmax, nyquist * 0.9)
        result = estimate_coda_q(
            trace,
            origin_time,
            freq_min=fmin,
            freq_max=fmax_eff,
            coda_start_factor=coda_start_factor,
            coda_end_snr=coda_end_snr,
            noise_window=noise_window,
            smooth_window=smooth_window,
        )
        freqs_out.append(result.central_frequency)
        qc_out.append(result.qc)
        qc_err_out.append(result.qc_std)
        r2_out.append(result.fit_r_squared)

    freq_arr = np.array(freqs_out, dtype=np.float64)
    qc_arr = np.array(qc_out, dtype=np.float64)
    err_arr = np.array(qc_err_out, dtype=np.float64)
    r2_arr = np.array(r2_out, dtype=np.float64)

    # Power-law fit: log10(Qc) = log10(Q0) + eta * log10(f)
    valid = (~np.isnan(qc_arr)) & (qc_arr > 0) & (r2_arr >= min_r_squared)
    if np.sum(valid) >= 2:
        log_f = np.log10(freq_arr[valid])
        log_q = np.log10(qc_arr[valid])
        reg = linregress(log_f, log_q)
        q0 = 10.0**reg.intercept
        eta = reg.slope
    else:
        q0 = float("nan")
        eta = float("nan")

    return CodaQFrequencyResult(
        frequencies=freq_arr,
        qc_values=qc_arr,
        qc_errors=err_arr,
        q0=float(q0),
        eta=float(eta),
        r_squared_values=r2_arr,
        station=station,
    )
