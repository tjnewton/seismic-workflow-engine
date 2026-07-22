"""Noise window metrics and distribution analysis."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def analyze_noise_windows(
    noise_windows: np.ndarray,
    noise_windows_times: list[tuple],
    output_csv: str | Path | None = None,
) -> dict[str, list[tuple[float, float, float]]]:
    """Compute per-station noise metrics (max, median, RMS of absolute values).

    Parameters
    ----------
    noise_windows : numpy.ndarray
        2-D array of noise window amplitudes.
    noise_windows_times : list of tuple
        Metadata ``(start, end, station, channel)`` for each window.
    output_csv : str, Path, or None
        If provided, save metrics to this CSV file.

    Returns
    -------
    metric_map : dict
        Mapping ``{station: [(max, median, rms), ...]}`` with one entry
        per noise window belonging to that station.
    """
    metric_map: dict[str, list[tuple[float, float, float]]] = {}

    for idx, meta in enumerate(noise_windows_times):
        station = meta[2]
        window = noise_windows[idx]
        abs_window = np.abs(window)
        metric = (
            float(np.max(abs_window)),
            float(np.median(abs_window)),
            float(np.sqrt(np.mean(np.square(window)))),
        )
        metric_map.setdefault(station, []).append(metric)

    if output_csv is not None:
        _save_metrics_csv(metric_map, output_csv)

    return metric_map


def plot_noise_time_histogram(
    noise_windows_times: list[tuple],
    bin_width: int,
    major_x_tick: str = "day",
    minor_x_tick: str = "hour",
) -> Figure:
    """Plot a temporal histogram of noise window occurrence.

    Parameters
    ----------
    noise_windows_times : list of tuple
        Metadata from :func:`find_noise_windows`.
    bin_width : int
        Bin width in seconds.
    major_x_tick : str
        Major tick convention: ``"month"``, ``"day"``, or ``"hour"``.
    minor_x_tick : str
        Minor tick convention: ``"day"``, ``"hour"``, or ``"minute"``.
    """
    start_time = min(t[0] for t in noise_windows_times)
    end_time = max(t[1] for t in noise_windows_times)

    n_bins = int((end_time - start_time) / bin_width) + 1
    bin_edges = [start_time + i * bin_width for i in range(n_bins)]

    # Vectorized counting via numpy searchsorted on float timestamps
    window_starts = np.array([t[0].timestamp for t in noise_windows_times])
    edge_timestamps = np.array([b.timestamp for b in bin_edges])
    bin_indices = np.searchsorted(edge_timestamps, window_starts, side="right") - 1
    counts = np.bincount(bin_indices.clip(0, n_bins - 1), minlength=n_bins)

    bins_dt = [b.datetime for b in bin_edges]

    plt.style.use("Solarize_Light2")
    fig = plt.figure(figsize=(20, 4), dpi=150)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(bottom=0.2)

    ax.bar(
        bins_dt,
        counts,
        align="edge",
        width=1 / (len(bins_dt) + 1),
        linewidth=1.0,
        edgecolor="black",
        alpha=0.8,
    )
    ax2 = ax.twiny()

    _MAJOR_LOCATOR = {
        "month": mdates.MonthLocator(),
        "hour": mdates.HourLocator(byhour=[0, 6, 12, 18, 24]),
        "day": mdates.DayLocator(bymonthday=[start_time.datetime.day]),
    }
    _MAJOR_FMT = {
        "month": mdates.DateFormatter("%b"),
        "hour": mdates.DateFormatter("%-H"),
        "day": mdates.DateFormatter("%b %-d %Y"),
    }
    _MINOR_LOCATOR = {
        "day": mdates.DayLocator(bymonthday=[1, 8, 15, 22]),
        "minute": mdates.MinuteLocator(byminute=[0, 30]),
        "hour": mdates.HourLocator(byhour=list(range(24))),
    }
    _MINOR_FMT = {
        "day": mdates.DateFormatter("%-d"),
        "minute": mdates.DateFormatter("%-M"),
        "hour": mdates.DateFormatter("%-H"),
    }

    ax2.xaxis.set_major_locator(_MAJOR_LOCATOR[major_x_tick])
    ax2.xaxis.set_major_formatter(_MAJOR_FMT[major_x_tick])

    ax.xaxis.set_major_locator(_MINOR_LOCATOR[minor_x_tick])
    ax.xaxis.set_major_formatter(_MINOR_FMT[minor_x_tick])
    ax.xaxis.set_tick_params(labelsize=8)

    _X_LIMITS = {
        "hour": {
            "start": start_time.datetime.replace(microsecond=0, second=0, minute=0),
            "end": end_time.datetime.replace(
                microsecond=999999,
                second=59,
                minute=59,
                hour=23,
                day=start_time.datetime.day,
            ),
        }
    }
    if minor_x_tick in _X_LIMITS:
        ax.set_xlim(_X_LIMITS[minor_x_tick]["start"], _X_LIMITS[minor_x_tick]["end"])

    ax2.set_xlim(ax.get_xlim())
    ax2.spines["bottom"].set_position(("axes", -0.05))
    ax2.spines["bottom"].set_visible(False)
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.xaxis.set_tick_params(grid_visible=False)
    for label in ax2.xaxis.get_ticklabels():
        label.set_horizontalalignment("left")

    ax.set_title("Temporal Histogram of Noise")
    ax.set_title(
        f"{len(noise_windows_times):,} noise windows",
        loc="right",
        fontsize=8,
        color="gray",
    )
    ax2.set_xlabel("Hour of Day")
    ax.set_ylabel("Count")
    return fig


def plot_noise_station_histogram(
    noise_windows_times: list[tuple],
    stations: list[str],
) -> Figure:
    """Plot a spatial histogram of noise windows across stations.

    Parameters
    ----------
    noise_windows_times : list of tuple
        Metadata from :func:`find_noise_windows`.
    stations : list of str
        Ordered list of station names for the x-axis.
    """
    station_counts = Counter(t[2] for t in noise_windows_times)
    counts = [station_counts.get(s, 0) for s in stations]

    plt.style.use("Solarize_Light2")
    fig = plt.figure(figsize=(20, 4), dpi=150)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(bottom=0.2)

    x_coords = list(range(len(stations)))
    ax.bar(
        x_coords,
        counts,
        align="center",
        width=1,
        linewidth=1.0,
        edgecolor="black",
        alpha=0.8,
    )

    ax.xaxis.set_major_locator(plt.FixedLocator(x_coords))
    ax.xaxis.set_major_formatter(plt.FixedFormatter(stations))
    ax.xaxis.set_tick_params(labelsize=8)
    ax.set_xlim(-0.6, x_coords[-1] + 0.6)

    ax.set_title("Histogram of Noise by Station")
    ax.set_title(
        f"{len(noise_windows_times):,} noise windows",
        loc="right",
        fontsize=8,
        color="gray",
    )
    if noise_windows_times:
        ax.set_title(
            noise_windows_times[0][0].datetime.strftime("%b %-d %Y"),
            loc="left",
            fontsize=8,
            color="gray",
        )
    ax.set_xlabel("Station")
    ax.set_ylabel("Count")
    return fig


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _save_metrics_csv(
    metric_map: dict[str, list[tuple[float, float, float]]],
    output_path: str | Path,
) -> None:
    """Write aggregated per-station metrics to a CSV file."""
    rows = []
    for station, metrics in sorted(metric_map.items()):
        medians = np.array([m[1] for m in metrics])
        rms_values = np.array([m[2] for m in metrics])
        rows.append(
            [
                station,
                str(np.mean(medians)),
                str(np.median(medians)),
                str(np.mean(rms_values)),
                str(np.median(rms_values)),
            ]
        )
    arr = np.array(rows)
    np.savetxt(str(output_path), arr, delimiter=",", fmt="%s")
    logger.info("Saved noise metrics to %s", output_path)
