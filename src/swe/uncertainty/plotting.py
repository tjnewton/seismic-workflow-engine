"""Publication-quality uncertainty visualization.

Provides functions to create figures showing:
- Pick time uncertainty error bars on waveforms
- Location uncertainty ellipsoids (map and cross-section views)
- Magnitude probability density distributions
- Stress tensor confidence regions on stereonet projections
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from swe.uncertainty.types import (
    LocationUncertainty,
    MagnitudeUncertainty,
    UncertainValue,
)

logger = logging.getLogger(__name__)

__all__ = [
    "plot_location_uncertainty",
    "plot_magnitude_pdf",
    "plot_pick_uncertainty",
    "plot_uncertainty_summary",
]


def plot_pick_uncertainty(
    pick_times: NDArray[np.floating],
    pick_uncertainties: NDArray[np.floating],
    *,
    station_labels: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    title: str = "Pick Time Uncertainties",
) -> Figure:
    """Plot pick times with error bars.

    Parameters
    ----------
    pick_times : NDArray
        Pick times as seconds relative to some reference time.
    pick_uncertainties : NDArray
        Symmetric 1-sigma uncertainty for each pick (seconds).
    station_labels : list of str, optional
        Labels for each pick (e.g. station codes).
    figsize : tuple, optional
        Figure size ``(width, height)`` in inches.
    title : str
        Figure title.

    Returns
    -------
    Figure
        Matplotlib figure with pick error bars.
    """
    pick_times = np.asarray(pick_times, dtype=np.float64)
    pick_uncertainties = np.asarray(pick_uncertainties, dtype=np.float64)

    n_picks = len(pick_times)
    if station_labels is None:
        station_labels = [f"Stn {i}" for i in range(n_picks)]

    fig, ax = plt.subplots(figsize=figsize or (10, max(4, n_picks * 0.4)))

    y_pos = np.arange(n_picks)
    ax.errorbar(
        pick_times,
        y_pos,
        xerr=pick_uncertainties,
        fmt="o",
        color="steelblue",
        ecolor="gray",
        elinewidth=1.5,
        capsize=3,
        capthick=1.5,
        markersize=5,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(station_labels)
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()

    return fig


def plot_location_uncertainty(
    locations: list[tuple[float, float, float]],
    uncertainties: list[LocationUncertainty],
    *,
    figsize: tuple[float, float] | None = None,
    title: str = "Location Uncertainties",
) -> Figure:
    """Plot event locations with uncertainty ellipses.

    Creates a two-panel figure: map view (horizontal) and depth
    cross-section (vertical).

    Parameters
    ----------
    locations : list of (lon, lat, depth_km)
        Event locations.
    uncertainties : list of LocationUncertainty
        Corresponding uncertainty estimates.
    figsize : tuple, optional
        Figure size ``(width, height)`` in inches.
    title : str
        Figure title.

    Returns
    -------
    Figure
        Two-panel figure (map view + depth section).
    """
    from matplotlib.patches import Ellipse

    fig, (ax_map, ax_depth) = plt.subplots(
        1,
        2,
        figsize=figsize or (12, 5),
    )

    [loc[0] for loc in locations]
    [loc[1] for loc in locations]
    [loc[2] for loc in locations]

    # --- Map view ---
    for _i, (loc, unc) in enumerate(zip(locations, uncertainties, strict=False)):
        ax_map.plot(loc[0], loc[1], "ko", markersize=4, zorder=3)
        if unc.horiz_uncertainty_m > 0:
            # Convert meters to approximate degrees
            h_deg = unc.horiz_uncertainty_m / 111_000.0

            if unc.covariance_matrix is not None and unc.covariance_matrix.shape == (3, 3):
                # Use covariance for ellipse orientation
                cov_xy = unc.covariance_matrix[:2, :2]
                eigenvalues, eigenvectors = np.linalg.eigh(cov_xy)
                angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
                width = 2 * np.sqrt(eigenvalues[1]) / 111_000.0
                height = 2 * np.sqrt(max(eigenvalues[0], 0)) / 111_000.0
            else:
                angle = 0
                width = 2 * h_deg
                height = 2 * h_deg

            ellipse = Ellipse(
                (loc[0], loc[1]),
                width=width,
                height=height,
                angle=angle,
                facecolor="lightcoral",
                edgecolor="darkred",
                alpha=0.3,
                linewidth=0.8,
                zorder=2,
            )
            ax_map.add_patch(ellipse)

    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.set_title(f"{title} - Map View")
    ax_map.set_aspect("equal")

    # --- Depth cross-section ---
    for _i, (loc, unc) in enumerate(zip(locations, uncertainties, strict=False)):
        ax_depth.plot(loc[0], loc[2], "ko", markersize=4, zorder=3)
        if unc.depth_uncertainty_m > 0:
            d_km = unc.depth_uncertainty_m / 1000.0
            h_deg = unc.horiz_uncertainty_m / 111_000.0 if unc.horiz_uncertainty_m > 0 else 0.01
            ellipse = Ellipse(
                (loc[0], loc[2]),
                width=2 * h_deg,
                height=2 * d_km,
                facecolor="lightskyblue",
                edgecolor="navy",
                alpha=0.3,
                linewidth=0.8,
                zorder=2,
            )
            ax_depth.add_patch(ellipse)

    ax_depth.set_xlabel("Longitude")
    ax_depth.set_ylabel("Depth (km)")
    ax_depth.set_title(f"{title} - Depth Section")
    ax_depth.invert_yaxis()

    fig.tight_layout()
    return fig


def plot_magnitude_pdf(
    magnitudes: NDArray[np.floating],
    uncertainties: NDArray[np.floating],
    *,
    figsize: tuple[float, float] | None = None,
    n_kde_points: int = 200,
    title: str = "Magnitude Distribution",
) -> Figure:
    """Plot magnitude estimates as individual Gaussians and combined PDF.

    Parameters
    ----------
    magnitudes : NDArray
        Best-estimate magnitudes.
    uncertainties : NDArray
        1-sigma magnitude uncertainties.
    figsize : tuple, optional
        Figure size.
    n_kde_points : int
        Number of points for the combined density curve.
    title : str
        Figure title.

    Returns
    -------
    Figure
        Figure with magnitude PDF and individual event curves.
    """
    magnitudes = np.asarray(magnitudes, dtype=np.float64)
    uncertainties = np.asarray(uncertainties, dtype=np.float64)

    # Clamp very small uncertainties
    uncertainties = np.maximum(uncertainties, 0.01)

    fig, ax = plt.subplots(figsize=figsize or (8, 5))

    # x-axis range
    x_min = np.min(magnitudes) - 3 * np.max(uncertainties)
    x_max = np.max(magnitudes) + 3 * np.max(uncertainties)
    x = np.linspace(x_min, x_max, n_kde_points)

    # Individual Gaussians
    combined = np.zeros_like(x)
    for mag, sigma in zip(magnitudes, uncertainties, strict=False):
        pdf = np.exp(-0.5 * ((x - mag) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        combined += pdf
        ax.plot(x, pdf, color="steelblue", alpha=0.15, linewidth=0.5)

    # Normalize combined
    if len(magnitudes) > 0:
        combined /= len(magnitudes)

    ax.fill_between(x, combined, alpha=0.3, color="steelblue", label="Combined PDF")
    ax.plot(x, combined, color="navy", linewidth=1.5)

    # Mark individual magnitudes
    ax.scatter(
        magnitudes,
        np.zeros_like(magnitudes),
        marker="|",
        color="darkred",
        s=50,
        zorder=5,
        label="Events",
    )

    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Probability Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    return fig


def plot_uncertainty_summary(
    results: dict[str, Any],
    *,
    figsize: tuple[float, float] | None = None,
    title: str = "Pipeline Uncertainty Summary",
) -> Figure:
    """Create a summary dashboard of uncertainties across pipeline stages.

    Parameters
    ----------
    results : dict
        Mapping of ``stage_name -> uncertainty_value``.  Values can be:
        - :class:`UncertainValue`
        - :class:`LocationUncertainty`
        - :class:`MagnitudeUncertainty`
        - ``float`` (interpreted as 1-sigma uncertainty)
        - ``dict`` with ``"value"`` and ``"uncertainty"`` keys
    figsize : tuple, optional
        Figure size.
    title : str
        Figure title.

    Returns
    -------
    Figure
        Multi-bar summary figure.
    """
    stages: list[str] = []
    values: list[float] = []
    errors: list[float] = []

    for stage, data in results.items():
        if isinstance(data, UncertainValue):
            stages.append(stage)
            values.append(data.value)
            errors.append(data.uncertainty)
        elif isinstance(data, LocationUncertainty):
            stages.append(f"{stage} (horiz)")
            values.append(data.horiz_uncertainty_m)
            errors.append(0)
            stages.append(f"{stage} (depth)")
            values.append(data.depth_uncertainty_m)
            errors.append(0)
        elif isinstance(data, MagnitudeUncertainty):
            stages.append(stage)
            values.append(data.mag_uncertainty)
            errors.append(data.station_std)
        elif isinstance(data, dict):
            stages.append(stage)
            values.append(data.get("value", 0.0))
            errors.append(data.get("uncertainty", 0.0))
        elif isinstance(data, (int, float)):
            stages.append(stage)
            values.append(float(data))
            errors.append(0)

    fig, ax = plt.subplots(figsize=figsize or (10, max(4, len(stages) * 0.5)))

    y_pos = np.arange(len(stages))
    ax.barh(
        y_pos,
        values,
        xerr=errors if any(e > 0 for e in errors) else None,
        color="steelblue",
        edgecolor="navy",
        alpha=0.7,
        capsize=3,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(stages)
    ax.set_xlabel("Uncertainty")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()

    return fig
