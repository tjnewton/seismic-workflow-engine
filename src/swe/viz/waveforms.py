"""Waveform visualization functions for ObsPy Stream and Trace objects.

Every public function accepts ObsPy objects directly (no file-path I/O) and
returns a :class:`matplotlib.figure.Figure` without calling ``plt.show()``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as spsig
from matplotlib.dates import AutoDateLocator, DateFormatter
from matplotlib import gridspec
from scipy import fftpack as spfft

from swe.utils.helpers import max_amplitude, snr

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from obspy import Stream, Trace

logger = logging.getLogger(__name__)

__all__ = [
    "plot_stream",
    "plot_time_series_and_spectrogram",
    "plot_time_series",
    "plot_party_detections",
    "plot_stack",
    "plot_template_and_stack",
    "plot_stream_absolute",
    "plot_event_picks",
    "plot_spectrogram",
    "plot_spectra",
    "plot_multi_spectra",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _trace_label(trace) -> str:
    """Build a *net.sta.cha* label from trace stats."""
    return (
        f"{trace.stats.network}.{trace.stats.station}.{trace.stats.channel}"
    )


def _normalize(data: np.ndarray, offset: float = 0.0) -> np.ndarray:
    """Min-max normalise *data* into [0, 1.25] and shift by *offset*."""
    dmin = np.nanmin(data)
    dmax = np.nanmax(np.abs(data))
    denom = dmax - dmin
    if denom == 0:
        return np.zeros_like(data) + offset
    return (data - dmin) / denom * 1.25 + offset


def _spectrogram_params(trace):
    """Return *(window_length, nfft, overlap)* for a spectrogram."""
    duration = trace.stats.endtime - trace.stats.starttime
    if duration <= 0:
        duration = 1.0
    num_windows = max((duration / 60) * 40 - 2, 4)
    window_duration = duration / num_windows
    window_length = max(int(window_duration * trace.stats.sampling_rate), 4)
    nfft = window_length * 2
    overlap = window_length // 2
    return window_length, nfft, overlap


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_stream(
    stream: Stream,
    *,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Multi-panel waveform, spectrogram, and PSD for an ObsPy Stream.

    Parameters
    ----------
    stream : obspy.Stream
        Input waveform data.
    figsize : tuple of float, optional
        ``(width, height)`` in inches.  Defaults to ``(20, 4.5 * n_traces)``.

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure.
    """
    st = stream.copy()
    n = len(st)
    if n == 0:
        fig, ax = plt.subplots()
        ax.set_title("Empty stream")
        return fig

    if figsize is None:
        figsize = (20, 4.5 * n)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, 1)
    amplitude_ax = fig.add_subplot(gs[0, :])
    freq_outer = fig.add_subplot(gs[1:3, :])
    psd_ax = fig.add_subplot(gs[3, :])

    freq_inner = gridspec.GridSpecFromSubplotSpec(
        n, 1, subplot_spec=gs[1:3, :],
    )
    # Remove the outer axes ticks (we use it only as a frame)
    freq_outer.set_yticks([])
    freq_outer.set_xticks([])

    y_labels: list[str] = []

    for idx, trace in enumerate(st):
        time = trace.times("matplotlib")

        norm_amp = _normalize(trace.data, offset=idx)
        amplitude_ax.plot(time, norm_amp, color="k", linewidth=0.7)
        y_labels.append(f"{_trace_label(trace)}.{idx}")

        # Spectrogram
        wlen, nfft, ovlp = _spectrogram_params(trace)
        f_stft, t_stft, stft = spsig.spectrogram(
            trace.data,
            trace.stats.sampling_rate,
            nperseg=wlen,
            noverlap=ovlp,
            nfft=nfft,
        )
        spec_ax = fig.add_subplot(freq_inner[(idx + 1) * -1, 0])
        with np.errstate(divide="ignore"):
            spec_ax.pcolormesh(
                t_stft, f_stft, 20 * np.log10(np.abs(stft)),
                cmap="magma", shading="auto",
            )
        spec_ax.set_ylabel(
            f"{_trace_label(trace)}.{idx}", rotation=0, labelpad=40,
        )
        spec_ax.tick_params(
            axis="x", which="both", bottom=False, top=False,
            labelbottom=False,
        )
        spec_ax.set_ylim([0, trace.stats.sampling_rate / 2])

        # PSD
        f_psd, pxx = spsig.welch(
            trace.data,
            trace.stats.sampling_rate,
            nperseg=wlen,
            noverlap=ovlp,
            nfft=nfft,
        )
        psd_ax.plot(f_psd, pxx, label=str(idx))

    # Amplitude axes
    amplitude_ax.set_yticks(np.arange(0.5, n + 0.5))
    amplitude_ax.set_yticklabels(y_labels)
    amplitude_ax.set_ylabel("Station.Channel")
    amplitude_ax.set_xlim([time[0], time[-1]])
    amplitude_ax.set_xlabel("Time (Min:Sec)")
    amplitude_ax.xaxis.set_major_formatter(DateFormatter("%M:%S"))
    amplitude_ax.xaxis.set_major_locator(AutoDateLocator(minticks=10, maxticks=15))
    amplitude_ax.set_ylim((0, n + 0.5))

    # PSD axes
    psd_ax.set_xlabel("Frequency [Hz]")
    psd_ax.set_ylabel("PSD [V**2/Hz]")
    psd_ax.set_title("Power Spectrum")
    psd_ax.set_xlim([0, trace.stats.sampling_rate / 2])
    psd_ax.legend()

    fig.tight_layout()
    return fig


def plot_time_series_and_spectrogram(
    stream: Stream,
    *,
    title: str = "",
    figsize: tuple[float, float] | None = None,
    time_markers: dict | None = None,
) -> Figure:
    """Combined time-series and spectrogram for a pre-loaded Stream.

    Parameters
    ----------
    stream : obspy.Stream
        Already-trimmed and optionally filtered waveform data.
    title : str
        Title string placed above the amplitude panel.
    figsize : tuple of float, optional
        ``(width, height)`` in inches.  Defaults to ``(7, 3.5 * n_traces)``.
    time_markers : dict, optional
        Mapping of ``"NET.STA"`` to a two-element list of
        :class:`~obspy.core.utcdatetime.UTCDateTime` defining a
        highlight window.

    Returns
    -------
    matplotlib.figure.Figure
    """
    st = stream.copy()
    n = len(st)
    if n == 0:
        fig, ax = plt.subplots()
        ax.set_title("Empty stream")
        return fig

    if time_markers is None:
        time_markers = {}
    if figsize is None:
        figsize = (7, 3.5 * n)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 1)
    amplitude_ax = fig.add_subplot(gs[0, :])
    freq_outer = fig.add_subplot(gs[1:, :])
    freq_inner = gridspec.GridSpecFromSubplotSpec(
        n, 1, subplot_spec=gs[1:, :],
    )
    freq_outer.set_yticks([])
    freq_outer.set_xticks([])

    y_labels: list[str] = []
    for idx, trace in enumerate(st):
        time = trace.times("matplotlib")
        trace_start = trace.stats.starttime
        norm_amp = _normalize(trace.data, offset=idx - 0.7)
        amplitude_ax.plot(time, norm_amp, color="k", linewidth=0.7)

        net_sta = f"{trace.stats.network}.{trace.stats.station}"
        if net_sta in time_markers:
            tm = time_markers[net_sta]
            x_vals = [
                tm[0].matplotlib_date, tm[0].matplotlib_date,
                tm[1].matplotlib_date, tm[1].matplotlib_date,
                tm[0].matplotlib_date,
            ]
            y_vals = [
                min(norm_amp), max(norm_amp),
                max(norm_amp), min(norm_amp),
                min(norm_amp),
            ]
            amplitude_ax.plot(x_vals, y_vals, color="r", linewidth=2.0)

        y_labels.append(f"{net_sta}.{trace.stats.channel}")

        # Spectrogram
        wlen, nfft, ovlp = _spectrogram_params(trace)
        f_stft, t_stft, stft = spsig.spectrogram(
            trace.data,
            trace.stats.sampling_rate,
            nperseg=wlen,
            noverlap=ovlp,
            nfft=nfft,
        )
        spec_ax = fig.add_subplot(freq_inner[(idx + 1) * -1, 0])
        with np.errstate(divide="ignore"):
            spec_ax.pcolormesh(
                t_stft, f_stft, 20 * np.log10(np.abs(stft)),
                cmap="magma", shading="auto",
            )
        if net_sta in time_markers:
            tm = time_markers[net_sta]
            x_s = [
                tm[0] - trace_start, tm[0] - trace_start,
                tm[1] - trace_start, tm[1] - trace_start,
                tm[0] - trace_start,
            ]
            y_s = [
                f_stft.min(), f_stft.max(),
                f_stft.max(), f_stft.min(),
                f_stft.min(),
            ]
            spec_ax.plot(x_s, y_s, "r-", linewidth=2.0)

        spec_ax.set_ylabel(
            _trace_label(trace), rotation=0, labelpad=40,
        )
        spec_ax.tick_params(
            axis="x", which="both", bottom=False, top=False,
            labelbottom=False,
        )

    # Amplitude axes
    amplitude_ax.set_yticks(np.arange(0.5, n + 0.5))
    amplitude_ax.set_yticklabels(y_labels)
    amplitude_ax.set_ylabel("Station.Channel")
    start_t = st[0].stats.starttime
    end_t = st[0].stats.endtime
    amplitude_ax.set_xlim([start_t.matplotlib_date, end_t.matplotlib_date])
    amplitude_ax.set_xlabel(
        f"Time on {start_t.month:02}-{start_t.day:02}-{start_t.year}"
    )
    amplitude_ax.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
    amplitude_ax.xaxis.set_major_locator(AutoDateLocator(minticks=5, maxticks=10))
    amplitude_ax.set_ylim((-0.5, n + 0.3))
    if title:
        amplitude_ax.set_title(title)

    freq_outer.set_yticks([])
    freq_outer.set_xticks([])
    fig.tight_layout()
    return fig


def plot_time_series(
    stream: Stream,
    *,
    title: str = "",
    figsize: tuple[float, float] | None = None,
    time_markers: dict | None = None,
) -> Figure:
    """Multi-station normalised time-series plot.

    Parameters
    ----------
    stream : obspy.Stream
        Already-trimmed and optionally filtered waveform data.
    title : str
        Title for the figure.
    figsize : tuple of float, optional
        ``(width, height)`` in inches.  Defaults to ``(20, 3.5 * n_traces)``.
    time_markers : dict, optional
        Mapping of ``"NET.STA"`` to a two-element list of
        :class:`~obspy.core.utcdatetime.UTCDateTime` highlight windows.

    Returns
    -------
    matplotlib.figure.Figure
    """
    st = stream.copy()
    n = len(st)
    if n == 0:
        fig, ax = plt.subplots()
        ax.set_title("Empty stream")
        return fig

    if time_markers is None:
        time_markers = {}
    if figsize is None:
        figsize = (20, 3.5 * n)

    fig, amplitude_ax = plt.subplots(figsize=figsize)

    y_labels: list[str] = []
    for idx, trace in enumerate(st):
        time = trace.times("matplotlib")
        amp_max = max_amplitude(trace)
        if amp_max == 0:
            amp_max = 1.0
        norm_amp = _normalize(trace.data, offset=idx)
        amplitude_ax.plot(time, norm_amp, color="k", linewidth=0.7)

        net_sta = f"{trace.stats.network}.{trace.stats.station}"
        if net_sta in time_markers:
            tm = time_markers[net_sta]
            x_vals = [
                tm[0].matplotlib_date, tm[0].matplotlib_date,
                tm[1].matplotlib_date, tm[1].matplotlib_date,
                tm[0].matplotlib_date,
            ]
            y_vals = [
                min(norm_amp), max(norm_amp),
                max(norm_amp), min(norm_amp),
                min(norm_amp),
            ]
            amplitude_ax.plot(x_vals, y_vals, color="r", linewidth=2.0)

        y_labels.append(f"{net_sta}.{trace.stats.channel}")

    amplitude_ax.set_yticks(np.arange(0.5, n + 0.5))
    amplitude_ax.set_yticklabels(y_labels)
    amplitude_ax.set_ylabel("Station.Channel")
    start_t = st[0].stats.starttime
    end_t = st[-1].stats.endtime
    amplitude_ax.set_xlim([start_t.matplotlib_date, end_t.matplotlib_date])
    amplitude_ax.set_xlabel("Time (Hr:Min:Sec)")
    amplitude_ax.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
    amplitude_ax.xaxis.set_major_locator(AutoDateLocator(minticks=10, maxticks=35))
    amplitude_ax.set_ylim((0, n + 0.5))
    if title:
        amplitude_ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_party_detections(
    party,
    stream: Stream,
    *,
    figsize: tuple[float, float] | None = None,
    title: str = "",
    max_detections: int = 60,
) -> Figure:
    """Plot EQcorrscan Party detections as stacked normalised waveforms.

    This function expects the caller to supply a pre-loaded
    :class:`obspy.Stream` containing one trace per detection (already
    trimmed around the detection time).

    Parameters
    ----------
    party
        An EQcorrscan ``Party`` object with at least one family.
    stream : obspy.Stream
        Stream with one trace per detection, ordered to match the
        detection list.
    figsize : tuple of float, optional
        Figure dimensions.  Defaults to ``(12, 0.5 * n_detections)``.
    title : str
        Title placed above the axes.
    max_detections : int
        Cap on the number of detections to plot.

    Returns
    -------
    matplotlib.figure.Figure
    """
    detections = party.families[0].detections[:max_detections]
    n = min(len(detections), len(stream))
    if n == 0:
        fig, ax = plt.subplots()
        ax.set_title("No detections")
        return fig

    if figsize is None:
        figsize = (12, max(0.5 * n, 2))

    fig, amplitude_ax = plt.subplots(figsize=figsize)

    y_labels: list[str] = []
    for idx in range(n):
        trace = stream[idx]
        time = trace.times("matplotlib")
        if idx == 0:
            plot_start = trace.stats.starttime
            plot_end = trace.stats.endtime

        norm_amp = _normalize(trace.data, offset=idx)
        amplitude_ax.plot(time, norm_amp, color="k", linewidth=0.7)
        y_labels.append(_trace_label(trace))

    amplitude_ax.set_yticks(np.arange(0.5, n + 0.5))
    amplitude_ax.set_yticklabels(y_labels)
    amplitude_ax.set_ylabel("Station.Channel")
    amplitude_ax.set_xlim(
        [plot_start.matplotlib_date, plot_end.matplotlib_date],
    )
    amplitude_ax.set_xlabel("Time (Hr:Min:Sec)")
    amplitude_ax.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
    amplitude_ax.xaxis.set_major_locator(AutoDateLocator(minticks=8, maxticks=35))
    amplitude_ax.set_ylim((0, n + 0.5))
    if title:
        amplitude_ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_stack(
    stack_stream: Stream,
    *,
    figsize: tuple[float, float] | None = None,
    title: str = "",
    show_snr: bool = True,
) -> Figure:
    """Plot a stacked waveform stream with normalised amplitudes.

    Parameters
    ----------
    stack_stream : obspy.Stream
        Stacked waveform data.
    figsize : tuple of float, optional
        Figure dimensions.  Defaults to ``(20, 2.5 * n_traces)``.
    title : str
        Title placed above the axes.
    show_snr : bool
        If *True*, append per-trace SNR to the y-tick labels.

    Returns
    -------
    matplotlib.figure.Figure
    """
    st = stack_stream.copy()
    n = len(st)
    if n == 0:
        fig, ax = plt.subplots()
        ax.set_title("Empty stack")
        return fig

    if figsize is None:
        figsize = (20, max(2.5 * n, 3))

    fig, amplitude_ax = plt.subplots(figsize=figsize)
    time = st[0].times("matplotlib")

    y_labels: list[str] = []
    for idx, trace in enumerate(st):
        # Guard against all-NaN traces
        if np.all(np.isnan(trace.data)):
            trace.data = np.zeros_like(trace.data)

        norm_amp = _normalize(trace.data, offset=idx)
        amplitude_ax.plot(time, norm_amp, color="k", linewidth=0.7)

        label = _trace_label(trace)
        if show_snr:
            trace_snr = snr(trace)
            snr_val = round(trace_snr[0], 1) if trace_snr else 0.0
            label += f" SNR:{snr_val}"
        y_labels.append(label)

    amplitude_ax.set_yticks(np.arange(0.5, n + 0.5))
    amplitude_ax.set_yticklabels(y_labels)
    amplitude_ax.set_ylabel("Station.Channel")
    amplitude_ax.set_xlim([time[0], time[-1]])
    amplitude_ax.set_xlabel("Time (Hr:Min:Sec)")
    amplitude_ax.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
    amplitude_ax.xaxis.set_major_locator(AutoDateLocator(minticks=10, maxticks=35))
    amplitude_ax.set_ylim((0, n + 0.5))
    if title:
        amplitude_ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_template_and_stack(
    template_stream: Stream,
    stack_stream: Stream,
    *,
    figsize: tuple[float, float] | None = None,
    title: str = "",
) -> Figure:
    """Plot template waveforms alongside their stacked counterparts.

    The function interleaves traces from *template_stream* and
    *stack_stream* so that each template trace is visually compared
    against the corresponding stacked trace.

    Parameters
    ----------
    template_stream : obspy.Stream
        Template waveforms.
    stack_stream : obspy.Stream
        Stacked waveforms (same channels as the template).
    figsize : tuple of float, optional
        Figure dimensions.
    title : str
        Title for the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from obspy import Stream as ObspyStream

    combined = ObspyStream()
    for tr in template_stream:
        t_copy = tr.copy()
        t_copy.stats.station += "_TMPL"
        combined += t_copy
        sel = stack_stream.select(
            network=tr.stats.network,
            station=tr.stats.station,
            channel=tr.stats.channel,
        )
        if sel:
            s_copy = sel[0].copy()
            s_copy.stats.station += "_STACK"
            combined += s_copy

    return plot_stack(
        combined, figsize=figsize, title=title, show_snr=False,
    )


def plot_stream_absolute(
    stream: Stream,
    *,
    figsize: tuple[float, float] | None = None,
    title: str = "",
) -> Figure:
    """Plot traces on an absolute (common) time axis.

    Useful for visualising shifted traces of varying length where each
    trace may have a different start time.

    Parameters
    ----------
    stream : obspy.Stream
        Waveform data.
    figsize : tuple of float, optional
        Figure dimensions.  Defaults to ``(10, 0.5 * n_traces)``.
    title : str
        Title for the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(stream)
    if n == 0:
        fig, ax = plt.subplots()
        ax.set_title("Empty stream")
        return fig

    if figsize is None:
        figsize = (10, max(0.5 * n, 2))

    fig, amplitude_ax = plt.subplots(figsize=figsize)

    y_labels: list[str] = []
    for idx, trace in enumerate(stream):
        trace_times = trace.times("matplotlib")
        norm_amp = _normalize(trace.data, offset=idx)
        amplitude_ax.plot(trace_times, norm_amp, color="k", linewidth=0.7)
        y_labels.append(_trace_label(trace))

    amplitude_ax.set_yticks(np.arange(0.5, n + 0.5))
    amplitude_ax.set_yticklabels(y_labels)
    amplitude_ax.set_ylabel("Station.Channel")
    amplitude_ax.set_xlabel("Time (Hr:Min:Sec)")
    amplitude_ax.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
    amplitude_ax.xaxis.set_major_locator(AutoDateLocator(minticks=12, maxticks=30))
    amplitude_ax.set_ylim((0, n + 0.5))
    if title:
        amplitude_ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_event_picks(
    stream: Stream,
    picks: list[dict],
    *,
    figsize: tuple[float, float] | None = None,
    title: str = "Normalised event waveforms and picks",
) -> Figure:
    """Plot waveforms with event picks marked.

    Parameters
    ----------
    stream : obspy.Stream
        Pre-loaded, trimmed waveform data (one trace per pick).
    picks : list of dict
        Each dict must contain a ``"time"`` key whose value is a
        :class:`~obspy.core.utcdatetime.UTCDateTime`.  The list must
        be the same length as *stream*.
    figsize : tuple of float, optional
        Figure dimensions.  Defaults to ``(5, 0.3 * n_picks)``.
    title : str
        Title for the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = min(len(stream), len(picks))
    if n == 0:
        fig, ax = plt.subplots()
        ax.set_title("No picks")
        return fig

    if figsize is None:
        figsize = (5, max(0.3 * n, 2))

    fig, amplitude_ax = plt.subplots(figsize=figsize)

    y_labels: list[str] = []
    for idx in range(n):
        trace = stream[idx]
        pick_time = picks[idx]["time"]
        trace_times = trace.times("matplotlib")

        amp_max = max_amplitude(trace)
        if amp_max is None or amp_max == 0:
            continue

        norm_amp = _normalize(trace.data, offset=idx)
        amplitude_ax.plot(trace_times, norm_amp, color="k", linewidth=0.7)

        # Mark the pick
        pick_idx = np.abs(trace_times - pick_time.matplotlib_date).argmin()
        amplitude_ax.plot(
            pick_time.matplotlib_date, norm_amp[pick_idx],
            "ro", linewidth=0.7,
        )
        y_labels.append(_trace_label(trace))

    amplitude_ax.set_yticks(np.arange(0.5, n + 0.5))
    amplitude_ax.set_yticklabels(y_labels)
    amplitude_ax.set_ylabel("Network.Station.Channel")
    amplitude_ax.set_xlabel("Time (seconds)")
    amplitude_ax.xaxis.set_major_formatter(DateFormatter("%S.%f"))
    amplitude_ax.xaxis.set_major_locator(AutoDateLocator(minticks=2, maxticks=5))
    amplitude_ax.set_ylim((0, n + 0.5))
    if title:
        amplitude_ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_spectrogram(
    trace: Trace,
    *,
    figsize: tuple[float, float] | None = None,
    freq_max: float | None = None,
    cmap: str = "magma",
) -> Figure:
    """Generate a spectrogram for a single ObsPy Trace.

    Parameters
    ----------
    trace : obspy.Trace
        Input waveform.
    figsize : tuple of float, optional
        Figure dimensions.  Defaults to ``(12, 4)``.
    freq_max : float, optional
        Upper frequency limit for the y-axis.  Defaults to Nyquist / 2.
    cmap : str
        Matplotlib colormap name.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if figsize is None:
        figsize = (12, 4)

    wlen, nfft, ovlp = _spectrogram_params(trace)
    f_stft, t_stft, stft = spsig.spectrogram(
        trace.data,
        trace.stats.sampling_rate,
        nperseg=wlen,
        noverlap=ovlp,
        nfft=nfft,
    )

    fig, ax = plt.subplots(figsize=figsize)
    with np.errstate(divide="ignore"):
        ax.pcolormesh(
            t_stft, f_stft, 20 * np.log10(np.abs(stft)),
            cmap=cmap, shading="auto",
        )
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title(_trace_label(trace))
    if freq_max is not None:
        ax.set_ylim([0, freq_max])
    else:
        ax.set_ylim([0, trace.stats.sampling_rate / 2])
    fig.tight_layout()
    return fig


def plot_spectra(
    trace: Trace,
    *,
    figsize: tuple[float, float] | None = None,
    freq_max: float | None = None,
) -> Figure:
    """Plot the amplitude spectrum (FFT) of a single Trace.

    Parameters
    ----------
    trace : obspy.Trace
        Input waveform.
    figsize : tuple of float, optional
        Figure dimensions.  Defaults to ``(10, 4)``.
    freq_max : float, optional
        Upper frequency limit for the x-axis.  Defaults to Nyquist.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if figsize is None:
        figsize = (10, 4)

    fs = trace.stats.sampling_rate
    n_fft = len(trace.data)
    freqs = np.arange(0, fs / 2, fs / n_fft)
    spectrum = spfft.fft(trace.data, n_fft)
    spectrum = spectrum[: int(n_fft / 2)]

    # Trim frequency array if needed
    if len(freqs) > len(spectrum):
        freqs = freqs[: len(spectrum)]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(freqs, np.abs(spectrum), "k")
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_title(_trace_label(trace))
    if freq_max is not None:
        ax.set_xlim([0, freq_max])
    else:
        ax.set_xlim([0, fs / 2])
    ax.grid(True, which="both", axis="x")
    fig.tight_layout()
    return fig


def plot_multi_spectra(
    stream: Stream,
    *,
    figsize: tuple[float, float] | None = None,
    labels: list[str] | None = None,
    freq_range: tuple[float, float] = (0.1, 100.0),
) -> Figure:
    """Overlay amplitude spectra from multiple traces on a log-log plot.

    Parameters
    ----------
    stream : obspy.Stream
        Input waveform data (one trace per spectrum).
    figsize : tuple of float, optional
        Figure dimensions.  Defaults to ``(7, 4)``.
    labels : list of str, optional
        Legend labels, one per trace.  Defaults to trace IDs.
    freq_range : tuple of float
        ``(f_min, f_max)`` for the x-axis limits.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if figsize is None:
        figsize = (7, 4)
    if labels is None:
        labels = [_trace_label(tr) for tr in stream]

    fig, ax = plt.subplots(figsize=figsize)
    for idx, trace in enumerate(stream):
        fs = trace.stats.sampling_rate
        n_fft = len(trace.data)
        freqs = np.arange(0, fs / 2, fs / n_fft)
        spectrum = spfft.fft(trace.data, n_fft)
        spectrum = spectrum[: int(n_fft / 2)]

        if len(freqs) > len(spectrum):
            freqs = freqs[: len(spectrum)]

        label = labels[idx] if idx < len(labels) else f"Trace {idx}"
        ax.loglog(freqs, np.abs(spectrum), label=label)

    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Frequency (Hz)")
    ax.legend()
    ax.set_xlim(freq_range)
    ax.grid(True, which="both", axis="both")
    fig.tight_layout()
    return fig
