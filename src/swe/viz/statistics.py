"""Statistical distribution visualizations for seismic analysis.

Every public function accepts plain arrays/lists directly (no file I/O) and
returns a :class:`matplotlib.figure.Figure` without calling ``plt.show()``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import AutoDateLocator, DateFormatter

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

__all__ = [
    "plot_date_distribution",
    "plot_distribution",
    "plot_moment_distribution",
]


def plot_distribution(
    values: Sequence[float] | np.ndarray,
    *,
    bins: int = 500,
    title: str = "",
    xlabel: str = "Value",
    figsize: tuple[float, float] = (9, 4),
) -> Figure:
    """Histogram of a one-dimensional value distribution.

    Parameters
    ----------
    values : array-like
        Data values to histogram.
    bins : int
        Number of histogram bins.
    title : str
        Title for the figure.
    xlabel : str
        Label for the x-axis.
    figsize : tuple of float
        ``(width, height)`` in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    data = np.asarray(values, dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    _n, bin_edges, _patches = ax.hist(
        data,
        bins=bins,
        facecolor="darkred",
        alpha=0.6,
    )
    ax.set_facecolor("dimgray")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Counts per bin")
    ax.set_xlim((bin_edges[0], bin_edges[-1]))
    ax.grid(True)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_moment_distribution(
    magnitudes: Sequence[float] | np.ndarray,
    *,
    bins: int = 100,
    title: str = "Relative moment distribution",
    figsize: tuple[float, float] = (9, 4),
) -> Figure:
    """Histogram of moment magnitudes with quantile annotations.

    Parameters
    ----------
    magnitudes : array-like
        Moment-magnitude (or relative-moment) values.
    bins : int
        Number of histogram bins.
    title : str
        Title for the figure.
    figsize : tuple of float
        ``(width, height)`` in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    data = np.asarray(magnitudes, dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    _n, bin_edges, _patches = ax.hist(
        data,
        bins=bins,
        facecolor="darkred",
        alpha=0.6,
    )
    ax.set_facecolor("dimgray")
    ax.set_xlabel(f"Relative moment ({bins} bins)")
    ax.set_ylabel("Counts per bin")
    ax.set_xlim((bin_edges[0], bin_edges[-1]))
    ax.grid(True)

    # Quantile overlay on twin axis
    ax2 = ax.twinx()
    q05 = float(np.quantile(data, 0.05))
    q25 = float(np.quantile(data, 0.25))
    q50 = float(np.quantile(data, 0.50))
    q75 = float(np.quantile(data, 0.75))
    q95 = float(np.quantile(data, 0.95))
    quants = [
        (q05, 0.6, 0.16, "5th"),
        (q25, 0.8, 0.26, "25th"),
        (q50, 1.0, 0.36, "50th"),
        (q75, 0.8, 0.46, "75th"),
        (q95, 0.6, 0.56, "95th"),
    ]
    for qval, alpha, ymax, qlabel in quants:
        ax2.axvline(qval, alpha=alpha, ymax=ymax, linestyle=":", color="k")
        ax2.text(qval, ymax + 0.02, qlabel, size=9, alpha=alpha)

    ax2.set_ylim(0, 1)
    ax2.set_yticklabels([])
    ax2.set_ylabel("")

    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_date_distribution(
    dates: Sequence | np.ndarray,
    *,
    bins: int = 100,
    title: str = "",
    figsize: tuple[float, float] = (10, 5),
) -> Figure:
    """Histogram of event dates (event rate over time).

    Parameters
    ----------
    dates : array-like
        Sequence of :class:`~obspy.core.utcdatetime.UTCDateTime` objects
        **or** Matplotlib date floats.  If items have a ``matplotlib_date``
        attribute they are converted automatically.
    bins : int
        Number of histogram bins.
    title : str
        Title for the figure.  Defaults to ``"Detections (n=<N>)"``.
    figsize : tuple of float
        ``(width, height)`` in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Convert UTCDateTime objects to matplotlib floats if needed
    converted: list[float] = []
    for d in dates:
        if hasattr(d, "matplotlib_date"):
            converted.append(d.matplotlib_date)
        else:
            converted.append(float(d))
    mpl_dates = np.asarray(converted, dtype=float)

    if not title:
        title = f"Detections (n={len(mpl_dates)})"

    fig, ax = plt.subplots(figsize=figsize)
    _n, bin_edges, _patches = ax.hist(
        mpl_dates,
        bins=bins,
        facecolor="darkred",
        alpha=0.6,
    )
    ax.set_facecolor("dimgray")
    ax.set_xlabel("Date")
    ax.set_ylabel("Detections per bin")
    ax.set_title(title)
    ax.set_xlim((bin_edges[0], bin_edges[-1]))
    ax.xaxis.set_major_formatter(DateFormatter("%m-%d-%y"))
    ax.xaxis.set_major_locator(AutoDateLocator())
    ax.grid(True)
    fig.tight_layout()
    return fig
