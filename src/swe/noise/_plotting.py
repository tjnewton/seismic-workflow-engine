"""Internal plotting helpers for the noise module."""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from obspy.core.trace import Trace
from obspy.core.utcdatetime import UTCDateTime

from swe.utils.helpers import max_amplitude, nearest_ind


def plot_stalta(
    obspy_trace: Trace,
    cft: np.ndarray,
    start_time: UTCDateTime,
    end_time: UTCDateTime,
    sta_samples: int | None = None,
    lta_samples: int | None = None,
    figsize: tuple[float, float] = (19, 5),
) -> Figure:
    """Plot a trace and its STA/LTA characteristic function for a time window.

    Parameters
    ----------
    obspy_trace : obspy.Trace
        The seismic trace.
    cft : numpy.ndarray
        Characteristic function from ``classic_sta_lta``.
    start_time, end_time : UTCDateTime
        Time bounds to display.
    sta_samples, lta_samples : int, optional
        Window lengths used (for axis label only).
    figsize : tuple
        Figure size.
    """
    all_times = np.array(obspy_trace.times(type="utcdatetime"))
    start_idx = nearest_ind(all_times, start_time)
    end_idx = nearest_ind(all_times, end_time)

    mpl_times = np.array(obspy_trace.times(type="matplotlib"))[start_idx : end_idx + 1]
    subset_data = obspy_trace.data[start_idx : end_idx + 1]

    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=figsize)
    ax[0].plot(mpl_times, subset_data, "k-", linewidth=0.4)
    ax[0].xaxis_date()
    ax[0].set_ylabel("Amplitude")

    ax[1].plot(mpl_times, cft[start_idx : end_idx + 1], "k-", linewidth=0.4)
    ax[1].xaxis_date()
    ylabel = "STA/LTA"
    if sta_samples is not None and lta_samples is not None:
        ylabel = f"STA:{sta_samples} / LTA:{lta_samples}"
    ax[1].set_ylabel(ylabel)

    plt.xlabel("Time")
    return fig


def plot_noise_windows(
    noise_windows: np.ndarray,
    noise_windows_times: list[tuple] | None = None,
    indices: list[int] | None = None,
    figsize_width: float = 19,
) -> Figure:
    """Plot normalized noise windows.

    Parameters
    ----------
    noise_windows : numpy.ndarray
        2-D array of noise window amplitudes.
    noise_windows_times : list of tuple, optional
        If provided, plots windows with timestamps (expanded view).
    indices : list of int, optional
        Window index labels for the compact view.
    figsize_width : float
        Figure width in inches.
    """
    max_vals = max_amplitude(noise_windows)
    # Guard against all-zero windows producing division by zero
    max_vals = np.where(max_vals == 0, 1.0, max_vals)

    if noise_windows_times is not None:
        fig_height = 1.5 * len(noise_windows)
        fig, ax = plt.subplots(
            nrows=len(noise_windows),
            sharex=False,
            figsize=(figsize_width, fig_height),
        )
        if len(noise_windows) == 1:
            ax = [ax]
        for i in range(len(noise_windows)):
            ax[i].plot(
                noise_windows_times[i][2],
                noise_windows[i] / max_vals[i],
                "k-",
                linewidth=0.4,
            )
            ax[i].xaxis_date()
            xlabels = [
                noise_windows_times[i][0].matplotlib_date,
                noise_windows_times[i][1].matplotlib_date,
            ]
            ax[i].set_xticks(xlabels)
            ax[i].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S.%f"))
            ax[i].set_xlim(xlabels)
        plt.xlabel("Time")
        return fig

    else:
        if indices is None:
            indices = list(range(len(noise_windows)))

        fig_height = 0.8 * len(noise_windows)
        fig, ax = plt.subplots(
            nrows=1,
            sharex=True,
            figsize=(figsize_width, fig_height),
        )
        for i in range(len(noise_windows)):
            normalized = (noise_windows[i] / max_vals[i]) * 0.7
            ax.plot(
                range(len(noise_windows[i])),
                normalized + indices[i],
                "k-",
                linewidth=0.4,
            )
        ax.set_xlim((0, len(noise_windows[0])))
        ax.set_ylim((min(indices) - 1, max(indices) + 1))
        plt.ylabel("Window #")
        plt.xlabel("Sample")
        plt.yticks(indices)
        return fig
