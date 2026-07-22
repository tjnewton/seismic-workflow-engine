"""Spectral ratio methods: HVSR and SSR.

Computes Horizontal-to-Vertical Spectral Ratios (HVSR) for site
characterisation and Standard Spectral Ratios (SSR) for relative
site response estimation.

References
----------
Nakamura, Y. (1989). A method for dynamic characteristics estimation
    of subsurface using microtremor on the ground surface. QR of RTRI,
    30(1), 25-33.
SESAME (2004). Guidelines for the implementation of the H/V spectral
    ratio technique on ambient vibrations. European Commission Project
    No. EVG1-CT-2000-00026.
Borcherdt, R.D. (1970). Effects of local geology on ground motion near
    San Francisco Bay. Bull. Seismol. Soc. Am., 60(1), 29-61.
Konno, K. & Ohmachi, T. (1998). Ground-motion characteristics estimated
    from spectral ratio between horizontal and vertical components of
    microtremor. Bull. Seismol. Soc. Am., 88(1), 228-241.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from obspy import Stream
    from obspy.core.trace import Trace

logger = logging.getLogger(__name__)

__all__ = [
    "HVSRResult",
    "SSRResult",
    "compute_hvsr",
    "compute_ssr",
]


class HVSRResult(NamedTuple):
    """Result from HVSR computation."""

    frequencies: NDArray[np.floating]
    """Frequency array (Hz)."""
    hvsr_mean: NDArray[np.floating]
    """Mean HVSR curve across windows."""
    hvsr_std: NDArray[np.floating]
    """Standard deviation of HVSR across windows."""
    hvsr_per_window: NDArray[np.floating]
    """Individual HVSR curves, shape ``(n_windows, n_freqs)``."""
    peak_frequency: float
    """Frequency of the maximum HVSR amplitude."""
    peak_amplitude: float
    """Maximum HVSR amplitude."""
    n_windows: int
    """Number of time windows used."""
    station: str
    """Station identifier."""


class SSRResult(NamedTuple):
    """Result from Standard Spectral Ratio computation."""

    frequencies: NDArray[np.floating]
    """Frequency array (Hz)."""
    ssr_mean: NDArray[np.floating]
    """Mean SSR across events."""
    ssr_std: NDArray[np.floating]
    """Standard deviation of SSR across events."""
    ssr_per_event: NDArray[np.floating]
    """Individual SSR curves, shape ``(n_events, n_freqs)``."""
    target_station: str
    """Target station identifier."""
    reference_station: str
    """Reference station identifier."""
    n_events: int
    """Number of events used."""


def _konno_ohmachi_smoothing(
    spectrum: NDArray[np.floating],
    frequencies: NDArray[np.floating],
    *,
    bandwidth: float = 40.0,
) -> NDArray[np.floating]:
    """Apply Konno-Ohmachi (1998) logarithmic smoothing.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Amplitude spectrum to smooth.
    frequencies : numpy.ndarray
        Corresponding frequency array (Hz).
    bandwidth : float
        Smoothing bandwidth parameter (b). Higher values give less
        smoothing.

    Returns
    -------
    numpy.ndarray
        Smoothed spectrum.
    """
    smoothed = np.zeros_like(spectrum, dtype=np.float64)
    freqs = np.asarray(frequencies, dtype=np.float64)

    for i in range(len(freqs)):
        if freqs[i] <= 0:
            smoothed[i] = spectrum[i]
            continue

        # Compute the smoothing window
        ratio = freqs / freqs[i]
        # Avoid log(0) - skip DC
        valid = ratio > 0
        window = np.zeros_like(freqs)
        log_ratio = np.log10(ratio[valid])

        # Konno-Ohmachi window: (sin(b * log10(f/fc)) / (b * log10(f/fc)))^4
        arg = bandwidth * log_ratio
        # Handle the singularity at f == fc (arg == 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            sinc_vals = np.where(
                np.abs(arg) < 1e-10,
                1.0,
                (np.sin(arg) / arg) ** 4,
            )
        window[valid] = sinc_vals

        # Normalise and apply
        w_sum = np.sum(window)
        if w_sum > 0:
            smoothed[i] = np.sum(window * spectrum) / w_sum
        else:
            smoothed[i] = spectrum[i]

    return smoothed


def _identify_components(
    stream: Stream,
) -> tuple[list[Trace], list[Trace]]:
    """Separate horizontal and vertical components from a Stream.

    Vertical: channel code ending in Z or 3.
    Horizontal: channel code ending in N, E, 1, or 2.

    Returns
    -------
    tuple of (horizontal_traces, vertical_traces)
    """
    horizontal: list[Trace] = []
    vertical: list[Trace] = []
    for tr in stream:
        last_char = tr.stats.channel[-1].upper()
        if last_char in ("Z", "3"):
            vertical.append(tr)
        elif last_char in ("N", "E", "1", "2"):
            horizontal.append(tr)
        else:
            logger.warning("Cannot classify channel %s; skipping", tr.stats.channel)
    return horizontal, vertical


def compute_hvsr(
    stream: Stream,
    *,
    window_length: float = 40.0,
    overlap: float = 0.5,
    smoothing_type: str = "konno_ohmachi",
    smoothing_bandwidth: float = 40.0,
    freq_min: float = 0.1,
    freq_max: float = 50.0,
    merge_method: str = "geometric_mean",
    taper_fraction: float = 0.05,
) -> HVSRResult:
    """Compute Horizontal-to-Vertical Spectral Ratio (HVSR).

    Divides the stream into overlapping time windows and computes
    the mean H/V spectral ratio with Konno-Ohmachi smoothing.

    Parameters
    ----------
    stream : obspy.Stream
        Three-component waveform data.
    window_length : float
        Length of each time window in seconds.
    overlap : float
        Fractional overlap between windows (0 to <1).
    smoothing_type : str
        Smoothing method. Currently only ``"konno_ohmachi"`` is supported.
    smoothing_bandwidth : float
        Bandwidth parameter for Konno-Ohmachi smoothing.
    freq_min : float
        Minimum frequency (Hz) for the output.
    freq_max : float
        Maximum frequency (Hz) for the output.
    merge_method : str
        Method to combine horizontal components: ``"geometric_mean"``
        or ``"quadratic_mean"``.
    taper_fraction : float
        Taper fraction applied to each window.

    Returns
    -------
    HVSRResult
        HVSR results including mean curve, per-window curves, and peak.

    Raises
    ------
    ValueError
        If the stream lacks required components or windows are too short.
    """
    horizontal, vertical = _identify_components(stream)

    if len(horizontal) < 2:
        raise ValueError(f"Need at least 2 horizontal components, got {len(horizontal)}")
    if len(vertical) < 1:
        raise ValueError("No vertical component found in stream")

    station = f"{vertical[0].stats.network}.{vertical[0].stats.station}"

    # Use the first vertical trace
    vert_trace = vertical[0]
    horiz_traces = horizontal[:2]

    sr = vert_trace.stats.sampling_rate
    window_samples = int(window_length * sr)
    step_samples = int(window_samples * (1.0 - overlap))

    if window_samples < 4:
        raise ValueError(
            f"Window length {window_length}s is too short (need at least 4 samples at {sr} Hz)"
        )

    # Find common time range
    t_start = max(tr.stats.starttime for tr in [vert_trace, *horiz_traces])
    t_end = min(tr.stats.endtime for tr in [vert_trace, *horiz_traces])

    # Trim all to common range
    vert_data = vert_trace.copy().trim(t_start, t_end).data.astype(np.float64)
    horiz_data = [tr.copy().trim(t_start, t_end).data.astype(np.float64) for tr in horiz_traces]

    total_samples = min(len(vert_data), *(len(h) for h in horiz_data))

    # Generate window start indices
    starts: list[int] = []
    idx = 0
    while idx + window_samples <= total_samples:
        starts.append(idx)
        idx += step_samples

    if len(starts) == 0:
        raise ValueError(f"Trace too short for window_length={window_length}s")

    # Frequency array for the FFT
    freqs = np.fft.rfftfreq(window_samples, d=1.0 / sr)
    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    out_freqs = freqs[freq_mask]

    hvsr_windows: list[NDArray] = []

    for start in starts:
        end = start + window_samples

        # Taper
        taper = np.ones(window_samples)
        n_taper = max(1, int(taper_fraction * window_samples))
        taper[:n_taper] = 0.5 * (1 - np.cos(np.pi * np.arange(n_taper) / n_taper))
        taper[-n_taper:] = 0.5 * (1 - np.cos(np.pi * np.arange(n_taper) / n_taper))[::-1]

        v_win = vert_data[start:end] * taper
        h_wins = [h[start:end] * taper for h in horiz_data]

        # FFT amplitudes
        v_spec = np.abs(np.fft.rfft(v_win))[freq_mask]
        h_specs = [np.abs(np.fft.rfft(hw))[freq_mask] for hw in h_wins]

        # Smooth
        if smoothing_type == "konno_ohmachi":
            v_spec = _konno_ohmachi_smoothing(v_spec, out_freqs, bandwidth=smoothing_bandwidth)
            h_specs = [
                _konno_ohmachi_smoothing(hs, out_freqs, bandwidth=smoothing_bandwidth)
                for hs in h_specs
            ]

        # Combine horizontals
        if merge_method == "geometric_mean":
            h_combined = np.sqrt(h_specs[0] * h_specs[1])
        elif merge_method == "quadratic_mean":
            h_combined = np.sqrt((h_specs[0] ** 2 + h_specs[1] ** 2) / 2.0)
        else:
            raise ValueError(f"Unknown merge_method: {merge_method!r}")

        # H/V ratio, avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            hvsr = np.where(v_spec > 0, h_combined / v_spec, 0.0)

        hvsr_windows.append(hvsr)

    hvsr_array = np.array(hvsr_windows)
    hvsr_mean = np.mean(hvsr_array, axis=0)
    hvsr_std = (
        np.std(hvsr_array, axis=0, ddof=1) if len(hvsr_windows) > 1 else np.zeros_like(hvsr_mean)
    )

    peak_idx = int(np.argmax(hvsr_mean))
    peak_frequency = float(out_freqs[peak_idx])
    peak_amplitude = float(hvsr_mean[peak_idx])

    return HVSRResult(
        frequencies=out_freqs,
        hvsr_mean=hvsr_mean,
        hvsr_std=hvsr_std,
        hvsr_per_window=hvsr_array,
        peak_frequency=peak_frequency,
        peak_amplitude=peak_amplitude,
        n_windows=len(hvsr_windows),
        station=station,
    )


def compute_ssr(
    target_traces: list[Trace],
    reference_traces: list[Trace],
    *,
    smoothing_type: str = "konno_ohmachi",
    smoothing_bandwidth: float = 40.0,
    freq_min: float = 0.1,
    freq_max: float = 50.0,
    water_level: float = 1e-10,
    taper_fraction: float = 0.05,
) -> SSRResult:
    """Compute Standard Spectral Ratios (SSR).

    Computes the spectral ratio between target and reference station
    recordings for each event, then stacks across events.

    Parameters
    ----------
    target_traces : list of Trace
        Target station waveforms, one per event.
    reference_traces : list of Trace
        Reference station waveforms, one per event (paired with target).
    smoothing_type : str
        Smoothing method (``"konno_ohmachi"``).
    smoothing_bandwidth : float
        Bandwidth parameter for smoothing.
    freq_min : float
        Minimum frequency (Hz).
    freq_max : float
        Maximum frequency (Hz).
    water_level : float
        Small value added to the reference spectrum denominator to
        prevent division by zero.
    taper_fraction : float
        Taper fraction applied before FFT.

    Returns
    -------
    SSRResult
        SSR results including mean, standard deviation, and per-event
        ratios.

    Raises
    ------
    ValueError
        If input lists are empty or have mismatched lengths.
    """
    if len(target_traces) == 0:
        raise ValueError("target_traces must not be empty")
    if len(target_traces) != len(reference_traces):
        raise ValueError(
            f"target_traces ({len(target_traces)}) and reference_traces "
            f"({len(reference_traces)}) must have the same length"
        )

    target_station = f"{target_traces[0].stats.network}.{target_traces[0].stats.station}"
    reference_station = f"{reference_traces[0].stats.network}.{reference_traces[0].stats.station}"

    ssr_events: list[NDArray] = []
    out_freqs: NDArray | None = None

    for tgt, ref in zip(target_traces, reference_traces, strict=False):
        # Use the shorter trace length for consistent FFT size
        npts = min(len(tgt.data), len(ref.data))
        sr = tgt.stats.sampling_rate

        # Taper
        taper = np.ones(npts)
        n_taper = max(1, int(taper_fraction * npts))
        taper[:n_taper] = 0.5 * (1 - np.cos(np.pi * np.arange(n_taper) / n_taper))
        taper[-n_taper:] = 0.5 * (1 - np.cos(np.pi * np.arange(n_taper) / n_taper))[::-1]

        tgt_data = tgt.data[:npts].astype(np.float64) * taper
        ref_data = ref.data[:npts].astype(np.float64) * taper

        # FFT
        freqs = np.fft.rfftfreq(npts, d=1.0 / sr)
        tgt_spec = np.abs(np.fft.rfft(tgt_data))
        ref_spec = np.abs(np.fft.rfft(ref_data))

        # Frequency band
        freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
        f_out = freqs[freq_mask]
        tgt_s = tgt_spec[freq_mask]
        ref_s = ref_spec[freq_mask]

        # Smooth
        if smoothing_type == "konno_ohmachi":
            tgt_s = _konno_ohmachi_smoothing(tgt_s, f_out, bandwidth=smoothing_bandwidth)
            ref_s = _konno_ohmachi_smoothing(ref_s, f_out, bandwidth=smoothing_bandwidth)

        # Spectral ratio
        ratio = tgt_s / (ref_s + water_level)
        ssr_events.append(ratio)

        if out_freqs is None:
            out_freqs = f_out

    ssr_array = np.array(ssr_events)
    ssr_mean = np.mean(ssr_array, axis=0)
    ssr_std = np.std(ssr_array, axis=0, ddof=1) if len(ssr_events) > 1 else np.zeros_like(ssr_mean)

    assert out_freqs is not None  # set on the first loop iteration
    return SSRResult(
        frequencies=out_freqs,
        ssr_mean=ssr_mean,
        ssr_std=ssr_std,
        ssr_per_event=ssr_array,
        target_station=target_station,
        reference_station=reference_station,
        n_events=len(ssr_events),
    )
