"""Signal processing utilities for trace analysis."""

from __future__ import annotations

import logging

import numpy as np
from obspy.core.trace import Trace
from scipy import interpolate, stats
from scipy.signal import hilbert

logger = logging.getLogger(__name__)


def get_trace_properties(
    trace: Trace,
    pick_time,
    duration: float,
    *,
    sampling_rate: float | None = None,
    polynomial_degree: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, list[float]]:
    """Compute derivatives, curvature, and MADs around a phase pick.

    Analyzes the waveform within a time window centered on *pick_time*
    using max-pooling, polynomial fitting, and numerical differentiation.

    Parameters
    ----------
    trace : obspy.Trace
        Input waveform.
    pick_time : obspy.UTCDateTime
        Phase pick time that centers the analysis window.
    duration : float
        Window length in seconds (symmetric around *pick_time*).
    sampling_rate : float, optional
        Override the trace sampling rate (Hz).  When *None* the trace's
        own ``stats.sampling_rate`` is used.
    polynomial_degree : int
        Degree of the polynomial fit to the max-pooled amplitudes.

    Returns
    -------
    dy : numpy.ndarray
        First derivative of the polynomial fit.
    d2y : numpy.ndarray
        Second derivative.
    curvature : numpy.ndarray
        Curvature (unsigned) of the polynomial fit.
    fits : list
        ``[max_pool_indices, max_pool_amplitude, poly1d, t_linspace, degree]``.
    mads : list of float
        ``[w1_median, w1_MAD, w2_median, w2_MAD, w3_median, w3_MAD]``
        for the pre-pick, pick, and post-pick windows.
    """
    import obspy.signal.filter

    if sampling_rate is None:
        sampling_rate = trace.stats.sampling_rate

    trace_data = trace.data.copy()
    data_envelope = obspy.signal.filter.envelope(trace.data)

    # Build absolute-time and matplotlib-time arrays
    trace_times_dt = np.asarray([trace.stats.starttime + offset for offset in trace.times()])
    trace_times = trace.times("matplotlib")

    # Index into trace closest to pick_time
    pick_time_index = int(np.abs([t - pick_time for t in trace_times_dt]).argmin())

    # Window bounds
    half_samples = int(duration / 2 * sampling_rate)
    start_idx = pick_time_index - half_samples
    end_idx = pick_time_index + half_samples

    # --- Max-pooling ---
    max_pool_amplitude: list[float] = []
    max_pool_times: list[float] = []
    max_pool_indices: list[int] = []

    # Before pick: keep every sample
    for idx in range(start_idx, pick_time_index + 1):
        max_pool_amplitude.append(abs(trace_data[idx]))
        max_pool_times.append(trace_times[idx])
        max_pool_indices.append(idx)

    # After pick: 2-of-4 max-pooling
    for idx in range(pick_time_index + 1, end_idx + 1, 4):
        chunk_data = np.abs(trace_data[idx : idx + 4])
        chunk_times = trace_times[idx : idx + 4]
        if len(chunk_data) < 2:
            for k in range(len(chunk_data)):
                max_pool_amplitude.append(float(chunk_data[k]))
                max_pool_times.append(chunk_times[k])
                max_pool_indices.append(idx + k)
            continue
        top2 = np.argpartition(chunk_data, -2)[-2:]
        for k in top2:
            max_pool_amplitude.append(float(np.abs(trace_data[idx + k])))
            max_pool_times.append(trace_times[idx + k])
            max_pool_indices.append(idx + k)

    max_pool_amplitude_arr = np.asarray(max_pool_amplitude)

    # Polynomial fit
    p = np.poly1d(np.polyfit(max_pool_indices, max_pool_amplitude_arr, polynomial_degree))
    t = np.linspace(max_pool_indices[0], max_pool_indices[-1], num=1000)

    # Envelope interpolation (for reference; included in fits)
    margin = 10
    env_start = max(start_idx - margin, 0)
    env_end = min(end_idx + margin, len(data_envelope))
    temp_xs = np.linspace(env_start, env_end - 1, num=env_end - env_start)
    temp_envelope = data_envelope[env_start:env_end]
    env_interp = None
    if len(temp_xs) > 3:
        env_interp = interpolate.interp1d(temp_xs, temp_envelope, kind="cubic")

    # Derivatives
    dy = np.gradient(p(t), t)
    d2y = np.gradient(dy, t)
    curvature = np.abs(d2y) / (np.sqrt(1 + dy**2)) ** 1.5

    # MADs for three windows around pick
    mp_pick_matches = np.where(np.array(max_pool_indices) == pick_time_index)[0]
    if len(mp_pick_matches) == 0:
        # pick_time_index not in max_pool_indices; find nearest
        mp_arr = np.array(max_pool_indices)
        mp_pick_idx = int(np.argmin(np.abs(mp_arr - pick_time_index)))
    else:
        mp_pick_idx = int(mp_pick_matches[0])
    w1 = max_pool_amplitude_arr[: max(mp_pick_idx - 10, 0)]
    w2 = max_pool_amplitude_arr[max(mp_pick_idx - 10, 0) : mp_pick_idx + 6]
    w3 = max_pool_amplitude_arr[mp_pick_idx + 6 :]

    def _median_mad(arr: np.ndarray) -> tuple[float, float]:
        if len(arr) == 0:
            return 0.0, 0.0
        return float(np.median(arr)), float(stats.median_abs_deviation(arr))

    w1_med, w1_mad = _median_mad(w1)
    w2_med, w2_mad = _median_mad(w2)
    w3_med, w3_mad = _median_mad(w3)
    mads = [w1_med, w1_mad, w2_med, w2_mad, w3_med, w3_mad]

    fits = [max_pool_indices, max_pool_amplitude_arr, p, t, polynomial_degree, env_interp]
    return dy, d2y, curvature, fits, mads


def instantaneous_frequency(
    trace: Trace,
    *,
    halfwin: float = 0.06,
) -> float:
    """Estimate the instantaneous frequency at the peak amplitude.

    Uses the Hilbert transform to compute the analytic signal, then
    estimates the instantaneous frequency over a window centered on
    the maximum absolute amplitude.

    Parameters
    ----------
    trace : obspy.Trace
        Input waveform.
    halfwin : float
        Half-window length in seconds around the peak amplitude.

    Returns
    -------
    float
        Median instantaneous frequency (Hz).  ``NaN`` if the peak is
        too close to a trace boundary.
    """
    dt = trace.stats.delta
    pts = int(halfwin / dt)
    data = trace.data
    ind = int(np.argmax(np.abs(data)))

    if ind - pts < 0 or ind + pts >= len(data):
        return np.nan

    segment = data[ind - pts : ind + pts]
    analytic = hilbert(segment)
    v = analytic.real
    h = analytic.imag
    dv = np.gradient(v, dt)
    dh = np.gradient(h, dt)
    f_t = (1 / (2 * np.pi)) * ((v * dh - h * dv) / (v**2 + h**2))
    return float(np.median(f_t))
