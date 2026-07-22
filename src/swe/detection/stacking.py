"""Waveform stacking for matched-filter template improvement.

Stacks phase-aligned detections from EQcorrscan ``Party`` objects to
build higher-SNR templates for iterative matched-filter workflows.

Includes advanced stacking methods (nth-root, median, SNR-selective)
beyond the basic linear and phase-weighted stacks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
from obspy import Stream, Trace, UTCDateTime, read
from scipy.ndimage import shift as ndimage_shift

logger = logging.getLogger(__name__)


class StackResult(NamedTuple):
    """Result from waveform stacking."""

    pws: Stream
    """Phase-weighted stack."""
    linear: Stream
    """Linear stack."""
    ccs: list[float]
    """Per-detection cross-correlation coefficients."""


try:
    from eqcorrscan import Party, Tribe
    from eqcorrscan.utils.stacking import PWS_stack, linstack
except ImportError as exc:  # pragma: no cover
    raise ImportError("eqcorrscan is required for swe.detection.stacking") from exc


class Stack(Tribe):
    """A :class:`~eqcorrscan.Tribe` subclass that treats stacks as templates.

    This enables using stacked waveforms (higher-SNR averages of
    matched-filter detections) as new templates for iterative detection.
    The ``template_gen`` method is inherited from
    :class:`eqcorrscan.Tribe`.
    """


def time_shift(
    trace: Trace,
    shift_seconds: float,
) -> Trace:
    """Shift a trace in time using FFT-based interpolation.

    Parameters
    ----------
    trace : obspy.Trace
        Input waveform.
    shift_seconds : float
        Time shift in seconds (positive = forward / delay).

    Returns
    -------
    obspy.Trace
        A copy of *trace* with shifted data and adjusted ``starttime``.
    """
    tr = trace.copy()
    shift_samples = shift_seconds * tr.stats.sampling_rate
    tr.data = ndimage_shift(tr.data.astype(float), shift_samples, mode="constant", cval=0.0)
    tr.stats.starttime += shift_seconds
    return tr


def stack_waveforms(
    party: Party,
    streams_dir: str | Path,
    main_trace: tuple[str, str, str],
    *,
    lowcut: float | None = None,
    highcut: float | None = None,
    filt_order: int = 4,
) -> StackResult:
    """Stack waveforms from a Party's detections for SNR improvement.

    Parameters
    ----------
    party : eqcorrscan.Party
        Detection results.
    streams_dir : str or Path
        Directory containing day-long MiniSEED files.
    main_trace : tuple of str
        ``(network, station, channel)`` identifying the primary trace.
    lowcut, highcut : float, optional
        Bandpass filter corners (Hz).  Both must be set to apply filtering.
    filt_order : int
        Bandpass filter order.

    Returns
    -------
    pws : obspy.Stream
        Phase-weighted stack.
    linear : obspy.Stream
        Linear stack.
    ccs : list of float
        Cross-correlation coefficients for each detection used.
    """
    import glob

    streams_dir = Path(streams_dir)
    family = sorted(party.families, key=lambda f: len(f))[-1]

    all_streams: list[Stream] = []
    ccs: list[float] = []

    for detection in family.detections:
        _net, sta, _cha = main_trace
        det_time = detection.detect_time
        day_str = det_time.strftime("%Y-%m-%d")
        pattern = str(streams_dir / f"*{sta}*{day_str}*.ms")
        files = sorted(glob.glob(pattern))
        if not files:
            logger.debug("No file found for %s on %s", sta, day_str)
            continue

        try:
            st = Stream()
            for f in files:
                st += read(f)
            st.merge(method=1, fill_value=0)

            if lowcut is not None and highcut is not None:
                st.detrend("demean")
                st.filter(
                    "bandpass",
                    freqmin=lowcut,
                    freqmax=highcut,
                    corners=filt_order,
                )

            template_start = det_time - 2.0
            template_end = det_time + 12.0
            st.trim(template_start, template_end, pad=True, fill_value=0)
            all_streams.append(st)
            ccs.append(detection.detect_val)
        except (OSError, ValueError, TypeError) as exc:
            logger.warning(
                "Failed to process detection at %s: %s: %s", det_time, type(exc).__name__, exc
            )

    if not all_streams:
        return StackResult(Stream(), Stream(), [])

    pws = PWS_stack(all_streams) if len(all_streams) > 1 else all_streams[0]
    linear = linstack(all_streams) if len(all_streams) > 1 else all_streams[0]
    return StackResult(pws, linear, ccs)


def stack_template_detections(
    party: Party,
    streams_dir: str | Path,
    main_trace: tuple[str, str, str],
    template_times: dict[str, list],
    *,
    align_type: str = "self",
    shift_len: float = 2.0,
    lowcut: float | None = None,
    highcut: float | None = None,
    filt_order: int = 4,
    animate_stacks: bool = False,
) -> StackResult:
    """Stack detections with multiple alignment strategies.

    Parameters
    ----------
    party : eqcorrscan.Party
        Detection results containing at least one family.
    streams_dir : str or Path
        Directory containing day-long MiniSEED files.
    main_trace : tuple of str
        ``(network, station, channel)`` for the primary detection trace.
    template_times : dict
        ``{trace_id: [starttime, endtime]}`` for cross-correlation alignment.
    align_type : str
        Alignment method: ``'zero'``, ``'self'``, ``'fixed'``, ``'med'``,
        ``'max'``, ``'stack'``, or ``'super_stack'``.
    shift_len : float
        Maximum allowed alignment shift in seconds.
    lowcut, highcut : float, optional
        Bandpass filter corners (Hz).
    filt_order : int
        Bandpass filter order.
    animate_stacks : bool
        If *True*, save a stacking animation (requires ``celluloid``).

    Returns
    -------
    pws : obspy.Stream
        Phase-weighted stack.
    linear : obspy.Stream
        Linear stack.
    ccs : list of float
        Per-detection cross-correlation coefficients.
    """
    import glob

    from obspy.signal.cross_correlation import correlate, xcorr_max

    streams_dir = Path(streams_dir)
    family = sorted(party.families, key=lambda f: len(f))[-1]

    all_streams: list[Stream] = []
    ccs: list[float] = []

    for detection in family.detections:
        _net, sta, _cha = main_trace
        det_time = detection.detect_time
        day_str = det_time.strftime("%Y-%m-%d")
        pattern = str(streams_dir / f"*{sta}*{day_str}*.ms")
        files = sorted(glob.glob(pattern))
        if not files:
            continue

        try:
            st = Stream()
            for f in files:
                st += read(f)
            st.merge(method=1, fill_value=0)

            if lowcut is not None and highcut is not None:
                st.detrend("demean")
                st.filter(
                    "bandpass",
                    freqmin=lowcut,
                    freqmax=highcut,
                    corners=filt_order,
                )

            template_start = det_time - 2.0
            template_end = det_time + 12.0
            st.trim(template_start, template_end, pad=True, fill_value=0)
            all_streams.append(st)
            ccs.append(detection.detect_val)
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Skipping detection at %s: %s: %s", det_time, type(exc).__name__, exc)

    if not all_streams:
        return StackResult(Stream(), Stream(), [])

    # Alignment
    if align_type == "zero":
        pass  # No alignment, use as-is
    elif align_type in ("self", "fixed", "med", "max", "stack", "super_stack"):
        try:
            for i, st in enumerate(all_streams):
                for tr in st:
                    trace_id = f"{tr.stats.network}.{tr.stats.station}.{tr.stats.channel}"
                    if trace_id in template_times:
                        _ref_start, _ref_end = template_times[trace_id]
                        # Cross-correlate with first stream as reference
                        if (
                            i > 0
                            and len(
                                all_streams[0].select(
                                    network=tr.stats.network,
                                    station=tr.stats.station,
                                    channel=tr.stats.channel,
                                )
                            )
                            > 0
                        ):
                            ref_tr = all_streams[0].select(
                                network=tr.stats.network,
                                station=tr.stats.station,
                                channel=tr.stats.channel,
                            )[0]
                            cc = correlate(
                                ref_tr.data.astype(float),
                                tr.data.astype(float),
                                int(shift_len * tr.stats.sampling_rate),
                            )
                            shift_val, _cc_val = xcorr_max(cc)
                            shift_secs = shift_val / tr.stats.sampling_rate
                            if abs(shift_secs) <= shift_len:
                                shifted = time_shift(tr, -shift_secs)
                                tr.data = shifted.data
        except (ValueError, IndexError) as exc:
            logger.warning(
                "Alignment failed, falling back to zero alignment: %s: %s", type(exc).__name__, exc
            )

    pws = PWS_stack(all_streams) if len(all_streams) > 1 else all_streams[0]
    linear = linstack(all_streams) if len(all_streams) > 1 else all_streams[0]
    return StackResult(pws, linear, ccs)


# ---------------------------------------------------------------------------
# Advanced stacking methods
# ---------------------------------------------------------------------------


def nth_root_stack(
    traces: list[Trace],
    *,
    n: int = 2,
) -> Trace:
    """Compute an nth-root stack of aligned traces.

    The nth-root stack suppresses incoherent noise more aggressively than
    a linear stack by compressing amplitudes before averaging, then
    re-expanding.  It is particularly effective for weak, repeating
    signals such as LFEs and tremor.

    Parameters
    ----------
    traces : list of obspy.Trace
        Aligned waveforms with identical sampling rates and lengths.
    n : int
        Root order (default 2).  Higher values increase noise suppression
        but distort relative amplitudes.

    Returns
    -------
    obspy.Trace
        The nth-root stacked trace, inheriting metadata from the first
        input trace.

    Raises
    ------
    ValueError
        If *traces* is empty or *n* < 1.

    References
    ----------
    Muirhead, K.J. (1968). Eliminating false alarms when detecting
        seismic events automatically. Nature, 217, 533-534.
    """
    if not traces:
        raise ValueError("Cannot stack an empty list of traces.")
    if n < 1:
        raise ValueError(f"Root order n must be >= 1, got {n}")

    # Pad/truncate to common length
    min_len = min(len(tr.data) for tr in traces)
    data = np.array([tr.data[:min_len].astype(np.float64) for tr in traces])

    # nth-root transform: sign-preserving
    signs = np.sign(data)
    root_data = signs * np.abs(data) ** (1.0 / n)

    # Average and restore power
    avg = np.mean(root_data, axis=0)
    avg_signs = np.sign(avg)
    stacked = avg_signs * np.abs(avg) ** n

    result = traces[0].copy()
    result.data = stacked[:min_len]
    return result


def median_stack(
    traces: list[Trace],
) -> Trace:
    """Compute a median stack of aligned traces.

    The sample-by-sample median is robust to outliers and amplitude
    variations, making it useful when individual detections have
    variable SNR or contain glitches.

    Parameters
    ----------
    traces : list of obspy.Trace
        Aligned waveforms with identical sampling rates and lengths.

    Returns
    -------
    obspy.Trace
        The median-stacked trace, inheriting metadata from the first
        input trace.

    Raises
    ------
    ValueError
        If *traces* is empty.
    """
    if not traces:
        raise ValueError("Cannot stack an empty list of traces.")

    min_len = min(len(tr.data) for tr in traces)
    data = np.array([tr.data[:min_len].astype(np.float64) for tr in traces])

    stacked = np.median(data, axis=0)

    result = traces[0].copy()
    result.data = stacked
    return result


def selective_stack(
    traces: list[Trace],
    *,
    snr_threshold: float = 2.0,
    noise_window: tuple[float, float] = (0.0, 0.1),
    signal_window: tuple[float, float] = (0.2, 0.8),
    method: str = "linear",
    nth_root_n: int = 2,
) -> Trace:
    """Stack only traces that exceed an SNR threshold.

    Computes per-trace SNR from specified noise and signal windows,
    then stacks only those traces that meet the threshold.  This filters
    out low-quality detections before stacking, improving the final
    template.

    Parameters
    ----------
    traces : list of obspy.Trace
        Aligned waveforms with identical sampling rates and lengths.
    snr_threshold : float
        Minimum SNR required for a trace to be included (default 2.0).
    noise_window : tuple of float
        ``(start_fraction, end_fraction)`` defining the noise window
        as fractions of total trace length.
    signal_window : tuple of float
        ``(start_fraction, end_fraction)`` defining the signal window
        as fractions of total trace length.
    method : str
        Stacking method for selected traces: ``"linear"``, ``"median"``,
        or ``"nth_root"``.
    nth_root_n : int
        Root order if *method* is ``"nth_root"``.

    Returns
    -------
    obspy.Trace
        The stacked trace from high-SNR inputs.

    Raises
    ------
    ValueError
        If *traces* is empty or no traces pass the SNR threshold.
    """
    if not traces:
        raise ValueError("Cannot stack an empty list of traces.")

    selected: list[Trace] = []
    snr_values: list[float] = []

    for tr in traces:
        snr = _compute_trace_snr(tr, noise_window, signal_window)
        snr_values.append(snr)
        if snr >= snr_threshold:
            selected.append(tr)

    logger.info(
        "SNR-selective stack: %d / %d traces above threshold %.1f (SNR range: %.1f - %.1f)",
        len(selected),
        len(traces),
        snr_threshold,
        min(snr_values) if snr_values else 0.0,
        max(snr_values) if snr_values else 0.0,
    )

    if not selected:
        raise ValueError(
            f"No traces exceed SNR threshold {snr_threshold:.1f}. "
            f"Max SNR was {max(snr_values):.2f}."
        )

    if method == "linear":
        return _simple_linear_stack(selected)
    elif method == "median":
        return median_stack(selected)
    elif method == "nth_root":
        return nth_root_stack(selected, n=nth_root_n)
    else:
        raise ValueError(
            f"Unknown stacking method: {method!r}. Supported: 'linear', 'median', 'nth_root'"
        )


def trace_snr(
    trace: Trace,
    *,
    signal_window: tuple[float, float],
    noise_window: tuple[float, float],
    reference_time: UTCDateTime | None = None,
) -> float:
    """Two-window signal-to-noise ratio (signal RMS / noise RMS).

    Unlike the whole-trace peak/RMS metric in :func:`swe.utils.helpers.snr`,
    this compares an explicit signal window against an explicit noise window, so
    a single detection and a stack can be measured over the *identical*
    intervals (a fair, apples-to-apples comparison).  Windows are given in
    seconds relative to ``reference_time``.

    Parameters
    ----------
    trace : obspy.Trace
        Input waveform.
    signal_window : tuple of float
        ``(start, end)`` of the signal window in seconds, relative to
        ``reference_time``.
    noise_window : tuple of float
        ``(start, end)`` of the noise window in seconds, relative to
        ``reference_time``.
    reference_time : obspy.UTCDateTime, optional
        Zero-time for the window offsets.  Defaults to ``trace.stats.starttime``.

    Returns
    -------
    float
        ``signal_rms / noise_rms`` (linear scale).  ``inf`` when the noise window
        is silent but the signal window is not; ``0.0`` when both are silent.

    Raises
    ------
    ValueError
        If a window start is not before its end, or a window lies entirely
        outside the trace.
    """
    if signal_window[0] >= signal_window[1]:
        raise ValueError(f"signal_window start must precede end: {signal_window}")
    if noise_window[0] >= noise_window[1]:
        raise ValueError(f"noise_window start must precede end: {noise_window}")

    sr = trace.stats.sampling_rate
    ref = reference_time if reference_time is not None else trace.stats.starttime
    base = float(ref - trace.stats.starttime)
    n = len(trace.data)
    data = trace.data.astype(np.float64)

    def _slice(window: tuple[float, float], label: str) -> np.ndarray:
        start = round((base + window[0]) * sr)
        end = round((base + window[1]) * sr)
        start_c = max(0, min(start, n))
        end_c = max(0, min(end, n))
        if end_c <= start_c:
            raise ValueError(
                f"{label} {window} (s) lies outside the trace "
                f"(0-{n / sr:.3f} s relative to reference)"
            )
        return np.asarray(data[start_c:end_c])

    signal = _slice(signal_window, "signal_window")
    noise = _slice(noise_window, "noise_window")

    noise_rms = float(np.sqrt(np.mean(noise**2)))
    signal_rms = float(np.sqrt(np.mean(signal**2)))
    if noise_rms < 1e-30:
        return float("inf") if signal_rms > 0 else 0.0
    return signal_rms / noise_rms


def _compute_trace_snr(
    trace: Trace,
    noise_window: tuple[float, float],
    signal_window: tuple[float, float],
) -> float:
    """Compute SNR as signal RMS / noise RMS for a trace.

    Parameters
    ----------
    trace : obspy.Trace
        Input waveform.
    noise_window : tuple of float
        ``(start_frac, end_frac)`` as fractions of trace length.
    signal_window : tuple of float
        ``(start_frac, end_frac)`` as fractions of trace length.

    Returns
    -------
    float
        Signal-to-noise ratio (linear scale).
    """
    n = len(trace.data)
    data = trace.data.astype(np.float64)

    n_start = int(noise_window[0] * n)
    n_end = int(noise_window[1] * n)
    s_start = int(signal_window[0] * n)
    s_end = int(signal_window[1] * n)

    # Clamp to valid ranges
    n_start = max(0, min(n_start, n - 1))
    n_end = max(n_start + 1, min(n_end, n))
    s_start = max(0, min(s_start, n - 1))
    s_end = max(s_start + 1, min(s_end, n))

    noise_rms = np.sqrt(np.mean(data[n_start:n_end] ** 2))
    signal_rms = np.sqrt(np.mean(data[s_start:s_end] ** 2))

    if noise_rms < 1e-30:
        return float("inf") if signal_rms > 0 else 0.0
    return float(signal_rms / noise_rms)


def _simple_linear_stack(traces: list[Trace]) -> Trace:
    """Compute a simple linear (mean) stack.

    Parameters
    ----------
    traces : list of obspy.Trace
        Aligned waveforms.

    Returns
    -------
    obspy.Trace
        The mean-stacked trace.
    """
    min_len = min(len(tr.data) for tr in traces)
    data = np.array([tr.data[:min_len].astype(np.float64) for tr in traces])

    result = traces[0].copy()
    result.data = np.mean(data, axis=0)
    return result
