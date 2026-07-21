"""Spatial map visualizations for seismic analysis.

Provides station/event maps, cross-section views, detection timelines,
and NonLinLoc PDF scatter plots using plain matplotlib (no Cartopy).
All coordinates are in local Cartesian (km).  Users should convert from
geographic coordinates using :func:`swe.utils.helpers.geo_to_local`
before calling these functions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

__all__ = [
    "plot_station_event_map",
    "plot_cross_sections",
    "plot_detection_timeline",
    "plot_nll_scatter",
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _scatter_sizes(
    magnitudes: NDArray[np.floating] | None,
    n_events: int,
    mag_scale: float,
) -> NDArray[np.floating]:
    """Compute marker sizes from magnitudes.

    Parameters
    ----------
    magnitudes : array-like or None
        Event magnitudes.  If *None*, uniform size is used.
    n_events : int
        Number of events (used for uniform sizing).
    mag_scale : float
        Scaling factor for magnitude-based sizing.

    Returns
    -------
    numpy.ndarray
        Marker sizes suitable for ``scatter()``.
    """
    if magnitudes is None or len(magnitudes) == 0:
        return np.full(n_events, mag_scale)
    mags = np.asarray(magnitudes, dtype=np.float64)
    # Shift to positive range and scale
    mags_shifted = mags - np.nanmin(mags) + 0.5
    return mags_shifted ** 2 * mag_scale


def _annotate_stations(
    ax,
    coords: NDArray[np.floating],
    names: list[str],
) -> None:
    """Draw station triangles with labels on an axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    coords : numpy.ndarray
        Station coordinates, shape ``(N, 2)`` or ``(N, 3)``.
    names : list of str
        Station names.
    """
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        marker="^",
        c="blue",
        s=80,
        edgecolors="black",
        linewidths=0.5,
        zorder=10,
        label="Stations",
    )
    for i, name in enumerate(names):
        ax.annotate(
            name,
            (coords[i, 0], coords[i, 1]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            zorder=11,
        )


def _convert_times(times) -> NDArray[np.floating]:
    """Convert UTCDateTime or datetime-like objects to matplotlib date floats.

    Parameters
    ----------
    times : array-like
        Sequence of times.  Items with a ``matplotlib_date`` attribute
        (e.g. UTCDateTime) are converted; plain floats pass through.

    Returns
    -------
    numpy.ndarray
        Matplotlib date floats.
    """
    converted = []
    for t in times:
        if hasattr(t, "matplotlib_date"):
            converted.append(t.matplotlib_date)
        else:
            converted.append(float(t))
    return np.asarray(converted, dtype=np.float64)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_station_event_map(
    station_coords: NDArray[np.floating],
    station_names: list[str],
    *,
    event_coords: NDArray[np.floating] | None = None,
    event_magnitudes: NDArray[np.floating] | None = None,
    topo_grid: tuple[NDArray, NDArray, NDArray] | None = None,
    topo_cmap: str = "terrain",
    topo_alpha: float = 0.4,
    mag_scale: float = 10.0,
    cmap: str = "hot_r",
    figsize: tuple[float, float] = (8, 8),
    title: str = "",
) -> Figure:
    """Plot station triangles and event circles on a local-coordinate map.

    Parameters
    ----------
    station_coords : numpy.ndarray
        Station positions, shape ``(N_sta, 2)`` or ``(N_sta, 3)``
        as ``[x_km, y_km, ...]``.
    station_names : list of str
        Station identifiers.
    event_coords : numpy.ndarray, optional
        Event positions, shape ``(N_ev, 2)`` or ``(N_ev, 3)``.
    event_magnitudes : numpy.ndarray, optional
        Magnitudes for sizing event markers.
    topo_grid : tuple, optional
        ``(x_1d, y_1d, z_2d)`` arrays for a topography overlay.
    topo_cmap : str
        Colormap for topography.
    topo_alpha : float
        Transparency for topography overlay.
    mag_scale : float
        Scaling factor for magnitude-based marker sizes.
    cmap : str
        Colormap for event depth (3rd column) if available.
    figsize : tuple of float
        Figure dimensions in inches.
    title : str
        Title for the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Optional topography
    if topo_grid is not None:
        x_topo, y_topo, z_topo = topo_grid
        ax.contourf(
            x_topo, y_topo, z_topo,
            levels=50,
            cmap=topo_cmap,
            alpha=topo_alpha,
        )

    # Stations
    sta_coords = np.atleast_2d(station_coords)
    _annotate_stations(ax, sta_coords, station_names)

    # Events
    if event_coords is not None:
        ev = np.atleast_2d(event_coords)
        n_ev = ev.shape[0]
        sizes = _scatter_sizes(event_magnitudes, n_ev, mag_scale)

        if ev.shape[1] >= 3:
            scatter = ax.scatter(
                ev[:, 0], ev[:, 1],
                c=ev[:, 2],
                s=sizes,
                cmap=cmap,
                alpha=0.7,
                edgecolors="grey",
                linewidths=0.3,
                zorder=5,
                label="Events",
            )
            fig.colorbar(scatter, ax=ax, label="Depth (km)", shrink=0.7)
        else:
            ax.scatter(
                ev[:, 0], ev[:, 1],
                s=sizes,
                c="red",
                alpha=0.7,
                edgecolors="grey",
                linewidths=0.3,
                zorder=5,
                label="Events",
            )

    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_cross_sections(
    event_coords: NDArray[np.floating],
    *,
    station_coords: NDArray[np.floating] | None = None,
    station_names: list[str] | None = None,
    color_values: NDArray[np.floating] | None = None,
    magnitudes: NDArray[np.floating] | None = None,
    cmap: str = "viridis",
    mag_scale: float = 10.0,
    figsize: tuple[float, float] = (10, 10),
    title: str = "",
) -> Figure:
    """Three-panel cross-section figure: map view, N-S depth, E-W depth.

    Layout::

        +-----------+------+
        | Map View  | N-S  |
        | (x vs y)  | Depth|
        +-----------+------+
        | E-W Depth | cbar |
        | (x vs z)  |      |
        +-----------+------+

    Parameters
    ----------
    event_coords : numpy.ndarray
        Event positions, shape ``(N, 3)`` as ``[x_km, y_km, z_km]``.
    station_coords : numpy.ndarray, optional
        Station positions, shape ``(N_sta, 2+)``.
    station_names : list of str, optional
        Station identifiers (required if *station_coords* is given).
    color_values : numpy.ndarray, optional
        Values used to colour events (e.g. time, magnitude).
    magnitudes : numpy.ndarray, optional
        Magnitudes for sizing event markers.
    cmap : str
        Colormap for *color_values*.
    mag_scale : float
        Scaling factor for magnitude-based marker sizes.
    figsize : tuple of float
        Figure dimensions in inches.
    title : str
        Title for the map-view panel.

    Returns
    -------
    matplotlib.figure.Figure
    """
    ev = np.atleast_2d(event_coords)
    if ev.shape[1] < 3:
        raise ValueError(
            "event_coords must have at least 3 columns (x, y, z)"
        )

    n_ev = ev.shape[0]
    sizes = _scatter_sizes(magnitudes, n_ev, mag_scale)

    if color_values is not None:
        c = np.asarray(color_values, dtype=np.float64)
    else:
        c = ev[:, 2]

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[3, 1],
        height_ratios=[3, 1],
        hspace=0.05,
        wspace=0.05,
    )

    # Map view (top-left)
    ax_map = fig.add_subplot(gs[0, 0])
    sc = ax_map.scatter(
        ev[:, 0], ev[:, 1],
        c=c, s=sizes, cmap=cmap,
        alpha=0.7, edgecolors="grey", linewidths=0.3,
    )
    if station_coords is not None and station_names is not None:
        _annotate_stations(ax_map, np.atleast_2d(station_coords), station_names)
    ax_map.set_ylabel("Northing (km)")
    ax_map.set_aspect("equal")
    ax_map.tick_params(labelbottom=False)
    if title:
        ax_map.set_title(title)

    # N-S depth (top-right)
    ax_ns = fig.add_subplot(gs[0, 1], sharey=ax_map)
    ax_ns.scatter(
        ev[:, 2], ev[:, 1],
        c=c, s=sizes, cmap=cmap,
        alpha=0.7, edgecolors="grey", linewidths=0.3,
    )
    ax_ns.set_xlabel("Depth (km)")
    ax_ns.tick_params(labelleft=False)

    # E-W depth (bottom-left)
    ax_ew = fig.add_subplot(gs[1, 0], sharex=ax_map)
    ax_ew.scatter(
        ev[:, 0], ev[:, 2],
        c=c, s=sizes, cmap=cmap,
        alpha=0.7, edgecolors="grey", linewidths=0.3,
    )
    ax_ew.set_xlabel("Easting (km)")
    ax_ew.set_ylabel("Depth (km)")
    ax_ew.invert_yaxis()

    # Colorbar (bottom-right)
    ax_cb = fig.add_subplot(gs[1, 1])
    ax_cb.axis("off")
    fig.colorbar(sc, ax=ax_cb, fraction=0.8, label="Color value")

    fig.tight_layout()
    return fig


def plot_detection_timeline(
    event_coords: NDArray[np.floating],
    event_times,
    *,
    station_coords: NDArray[np.floating] | None = None,
    station_names: list[str] | None = None,
    magnitudes: NDArray[np.floating] | None = None,
    cmap: str = "plasma",
    mag_scale: float = 10.0,
    figsize: tuple[float, float] = (10, 8),
    title: str = "",
) -> Figure:
    """Plot events coloured by time to show spatial migration.

    Parameters
    ----------
    event_coords : numpy.ndarray
        Event positions, shape ``(N, 2)`` or ``(N, 3)``.
    event_times : array-like
        Event origin times (UTCDateTime or matplotlib date floats).
    station_coords : numpy.ndarray, optional
        Station positions.
    station_names : list of str, optional
        Station identifiers.
    magnitudes : numpy.ndarray, optional
        Magnitudes for sizing event markers.
    cmap : str
        Colormap for time progression.
    mag_scale : float
        Scaling factor for magnitude-based marker sizes.
    figsize : tuple of float
        Figure dimensions in inches.
    title : str
        Title for the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    ev = np.atleast_2d(event_coords)
    n_ev = ev.shape[0]
    sizes = _scatter_sizes(magnitudes, n_ev, mag_scale)
    time_floats = _convert_times(event_times)

    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(
        ev[:, 0], ev[:, 1],
        c=time_floats, s=sizes, cmap=cmap,
        alpha=0.7, edgecolors="grey", linewidths=0.3,
        zorder=5,
    )

    if station_coords is not None and station_names is not None:
        _annotate_stations(ax, np.atleast_2d(station_coords), station_names)

    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.set_aspect("equal")
    fig.colorbar(scatter, ax=ax, label="Time")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_nll_scatter(
    pdfs,
    *,
    hypocenters: NDArray[np.floating] | None = None,
    station_coords_km: NDArray[np.floating] | None = None,
    station_names: list[str] | None = None,
    cmap: str = "hot_r",
    figsize: tuple[float, float] = (10, 10),
    title: str = "",
) -> Figure:
    """Plot NonLinLoc PDF scatter points in a 3-panel cross-section.

    Parameters
    ----------
    pdfs : pandas.DataFrame
        DataFrame with columns ``x``, ``y``, ``z``, and optionally
        ``weight``.  Typically from
        :attr:`~swe.location.nonlinloc.NLLLocationResult.pdfs`.
    hypocenters : numpy.ndarray, optional
        Best-fit hypocenter(s), shape ``(M, 3)`` or ``(3,)``.
    station_coords_km : numpy.ndarray, optional
        Station coordinates in km for overlay.
    station_names : list of str, optional
        Station identifiers.
    cmap : str
        Colormap for PDF density.
    figsize : tuple of float
        Figure dimensions in inches.
    title : str
        Title for the map-view panel.

    Returns
    -------
    matplotlib.figure.Figure
    """
    x = pdfs["x"].values
    y = pdfs["y"].values
    z = pdfs["z"].values
    w = pdfs["weight"].values if "weight" in pdfs.columns else None

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[3, 1],
        height_ratios=[3, 1],
        hspace=0.05,
        wspace=0.05,
    )

    scatter_kw = dict(
        s=2, cmap=cmap, alpha=0.5, edgecolors="none",
    )

    # Map view (top-left)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_map.scatter(x, y, c=w, **scatter_kw)
    ax_map.set_ylabel("Northing (km)")
    ax_map.set_aspect("equal")
    ax_map.tick_params(labelbottom=False)
    if title:
        ax_map.set_title(title)

    # N-S depth (top-right)
    ax_ns = fig.add_subplot(gs[0, 1], sharey=ax_map)
    ax_ns.scatter(z, y, c=w, **scatter_kw)
    ax_ns.set_xlabel("Depth (km)")
    ax_ns.tick_params(labelleft=False)

    # E-W depth (bottom-left)
    ax_ew = fig.add_subplot(gs[1, 0], sharex=ax_map)
    ax_ew.scatter(x, z, c=w, **scatter_kw)
    ax_ew.set_xlabel("Easting (km)")
    ax_ew.set_ylabel("Depth (km)")
    ax_ew.invert_yaxis()

    # Bottom-right: empty
    ax_empty = fig.add_subplot(gs[1, 1])
    ax_empty.axis("off")

    # Overlay hypocenters
    if hypocenters is not None:
        hyp = np.atleast_2d(hypocenters)
        hyp_kw = dict(
            marker="*", c="lime", s=200,
            edgecolors="black", linewidths=0.8, zorder=10,
        )
        ax_map.scatter(hyp[:, 0], hyp[:, 1], **hyp_kw)
        ax_ns.scatter(hyp[:, 2], hyp[:, 1], **hyp_kw)
        ax_ew.scatter(hyp[:, 0], hyp[:, 2], **hyp_kw)

    # Overlay stations
    if station_coords_km is not None and station_names is not None:
        _annotate_stations(
            ax_map,
            np.atleast_2d(station_coords_km),
            station_names,
        )

    fig.tight_layout()
    return fig
