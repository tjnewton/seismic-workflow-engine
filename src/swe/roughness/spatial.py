"""Spatial frequency-roughness correlation.

Maps event frequencies onto 2-D grids and correlates with local
roughness metrics derived from elevation data.
"""

from __future__ import annotations

import logging
from typing import NamedTuple, TYPE_CHECKING

import numpy as np
from scipy import ndimage
from scipy.stats import pearsonr

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from swe.topo.raster import ElevationGrid

logger = logging.getLogger(__name__)

__all__ = [
    "SpatialCorrelationResult",
    "compute_spatial_frequency_map",
    "correlate_frequency_roughness",
]


class SpatialCorrelationResult(NamedTuple):
    """Result of spatial frequency-roughness correlation.

    Attributes
    ----------
    event_x : ndarray
        Event x-coordinates.
    event_y : ndarray
        Event y-coordinates.
    event_frequency : ndarray
        Frequency at each event location (Hz).
    local_roughness : ndarray
        Local roughness metric at each event location.
    correlation_coefficient : float
        Pearson correlation coefficient.
    p_value : float
        Two-sided p-value for the correlation.
    predicted_frequency : ndarray
        Frequency predicted by linear regression on roughness.
    roughness_metric : str
        Name of the roughness metric used.
    """

    event_x: NDArray[np.floating]
    event_y: NDArray[np.floating]
    event_frequency: NDArray[np.floating]
    local_roughness: NDArray[np.floating]
    correlation_coefficient: float
    p_value: float
    predicted_frequency: NDArray[np.floating]
    roughness_metric: str


def compute_spatial_frequency_map(
    event_locations: NDArray[np.floating],
    event_frequencies: NDArray[np.floating],
    *,
    grid_spacing: float | None = None,
    smoothing_radius: float | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Bin event frequencies into a 2-D spatial grid.

    Parameters
    ----------
    event_locations : ndarray, shape (N, 2)
        Event coordinates ``[x, y]``.
    event_frequencies : ndarray, shape (N,)
        Frequency value at each event (Hz).
    grid_spacing : float, optional
        Grid cell size.  If ``None``, uses 1/20 of the spatial extent.
    smoothing_radius : float, optional
        Gaussian smoothing radius (same units as coordinates).
        If ``None``, no smoothing is applied.

    Returns
    -------
    x_grid : ndarray
        1-D array of x grid coordinates.
    y_grid : ndarray
        1-D array of y grid coordinates.
    freq_grid : ndarray
        2-D grid of median frequencies, shape ``(len(y_grid), len(x_grid))``.
        Cells with no events contain ``NaN``.
    """
    locs = np.asarray(event_locations, dtype=np.float64)
    freqs = np.asarray(event_frequencies, dtype=np.float64)

    if locs.ndim != 2 or locs.shape[1] != 2:
        raise ValueError("event_locations must have shape (N, 2)")
    if len(locs) != len(freqs):
        raise ValueError("event_locations and event_frequencies must have same length")

    x_min, y_min = locs.min(axis=0)
    x_max, y_max = locs.max(axis=0)

    if grid_spacing is None:
        extent = max(x_max - x_min, y_max - y_min)
        grid_spacing = extent / 20.0 if extent > 0 else 1.0

    x_edges = np.arange(x_min, x_max + grid_spacing, grid_spacing)
    y_edges = np.arange(y_min, y_max + grid_spacing, grid_spacing)
    x_grid = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_grid = 0.5 * (y_edges[:-1] + y_edges[1:])

    freq_grid = np.full((len(y_grid), len(x_grid)), np.nan)

    ix = np.clip(
        np.searchsorted(x_edges, locs[:, 0]) - 1, 0, len(x_grid) - 1
    )
    iy = np.clip(
        np.searchsorted(y_edges, locs[:, 1]) - 1, 0, len(y_grid) - 1
    )

    for j in range(len(y_grid)):
        for i in range(len(x_grid)):
            mask = (ix == i) & (iy == j)
            if np.any(mask):
                freq_grid[j, i] = np.median(freqs[mask])

    if smoothing_radius is not None and smoothing_radius > 0:
        sigma = smoothing_radius / grid_spacing
        # Smooth only valid cells
        valid = ~np.isnan(freq_grid)
        filled = np.where(valid, freq_grid, 0.0)
        weight = valid.astype(np.float64)
        smoothed_vals = ndimage.gaussian_filter(filled, sigma=sigma)
        smoothed_wt = ndimage.gaussian_filter(weight, sigma=sigma)
        with np.errstate(invalid="ignore", divide="ignore"):
            freq_grid = np.where(
                smoothed_wt > 0, smoothed_vals / smoothed_wt, np.nan
            )

    logger.info(
        "Spatial frequency map: %d x %d grid, spacing=%.2f",
        len(y_grid),
        len(x_grid),
        grid_spacing,
    )

    return x_grid, y_grid, freq_grid


def correlate_frequency_roughness(
    event_locations: NDArray[np.floating],
    event_frequencies: NDArray[np.floating],
    elevation_grid: ElevationGrid,
    *,
    roughness_metric: str = "rms_height",
    neighborhood_radius: float = 5.0,
    slip_velocity: float | None = None,
) -> SpatialCorrelationResult:
    """Correlate event frequencies with local surface roughness.

    For each event, computes a roughness metric in a circular
    neighbourhood on the elevation grid, then reports the Pearson
    correlation with event frequency.

    Parameters
    ----------
    event_locations : ndarray, shape (N, 2)
        Event coordinates ``[x, y]`` in the same CRS as
        *elevation_grid*.
    event_frequencies : ndarray, shape (N,)
        Frequency at each event (Hz).
    elevation_grid : ElevationGrid
        Gridded elevation data.
    roughness_metric : str
        Roughness metric to compute.  One of ``"rms_height"``,
        ``"dominant_wavelength"``, ``"slope_variance"``.
    neighborhood_radius : float
        Radius of the circular neighbourhood (same units as grid
        coordinates).
    slip_velocity : float, optional
        If provided (m/s), predicts frequency via f = V / lambda
        and includes in the result.

    Returns
    -------
    SpatialCorrelationResult
    """
    valid_metrics = ("rms_height", "dominant_wavelength", "slope_variance")
    if roughness_metric not in valid_metrics:
        raise ValueError(
            f"roughness_metric must be one of {valid_metrics}, "
            f"got {roughness_metric!r}"
        )

    locs = np.asarray(event_locations, dtype=np.float64)
    freqs = np.asarray(event_frequencies, dtype=np.float64)
    elev = elevation_grid.elevation
    gx = np.asarray(elevation_grid.x)
    gy = np.asarray(elevation_grid.y)
    dx = abs(np.median(np.diff(gx)))

    local_roughness = np.full(len(locs), np.nan)

    for i, (ex, ey) in enumerate(locs):
        # Find grid indices within neighbourhood
        ix = np.searchsorted(gx, ex)
        iy = np.searchsorted(gy, ey)

        # Build circular mask in grid units
        r_cells = int(np.ceil(neighborhood_radius / dx))
        ix_min = max(0, ix - r_cells)
        ix_max = min(len(gx), ix + r_cells + 1)
        iy_min = max(0, iy - r_cells)
        iy_max = min(len(gy), iy + r_cells + 1)

        if ix_max <= ix_min or iy_max <= iy_min:
            continue

        sub_x = gx[ix_min:ix_max]
        sub_y = gy[iy_min:iy_max]
        sub_elev = elev[iy_min:iy_max, ix_min:ix_max]

        # Circular mask
        yy, xx = np.meshgrid(sub_y, sub_x, indexing="ij")
        dist = np.sqrt((xx - ex) ** 2 + (yy - ey) ** 2)
        circle = dist <= neighborhood_radius

        if np.sum(circle) < 4:
            continue

        heights = sub_elev[circle]

        if roughness_metric == "rms_height":
            local_roughness[i] = np.std(heights)
        elif roughness_metric == "slope_variance":
            gy_grad, gx_grad = np.gradient(sub_elev, dx)
            slopes = np.sqrt(gx_grad**2 + gy_grad**2)
            local_roughness[i] = np.var(slopes[circle])
        elif roughness_metric == "dominant_wavelength":
            # 1-D PSD along longest axis through the centre
            centre_row = sub_elev.shape[0] // 2
            profile = sub_elev[centre_row, :]
            if len(profile) >= 4:
                from swe.roughness.surface import compute_roughness_psd_1d

                psd_result = compute_roughness_psd_1d(
                    np.arange(len(profile)) * dx,
                    profile,
                    detrend=True,
                )
                local_roughness[i] = psd_result.dominant_wavelength

    # Filter out NaN entries
    valid = ~np.isnan(local_roughness) & ~np.isnan(freqs)
    if np.sum(valid) < 3:
        logger.warning(
            "Fewer than 3 valid event-roughness pairs; correlation unreliable"
        )
        predicted = np.full(len(locs), np.nan)
        return SpatialCorrelationResult(
            event_x=locs[:, 0],
            event_y=locs[:, 1],
            event_frequency=freqs,
            local_roughness=local_roughness,
            correlation_coefficient=np.nan,
            p_value=np.nan,
            predicted_frequency=predicted,
            roughness_metric=roughness_metric,
        )

    r, p = pearsonr(local_roughness[valid], freqs[valid])

    # Linear prediction
    from scipy.stats import linregress as _linregress

    reg = _linregress(local_roughness[valid], freqs[valid])
    predicted = np.full(len(locs), np.nan)
    predicted[valid] = reg.slope * local_roughness[valid] + reg.intercept

    logger.info(
        "Frequency-roughness correlation: r=%.3f, p=%.4f (%s)",
        r,
        p,
        roughness_metric,
    )

    return SpatialCorrelationResult(
        event_x=locs[:, 0],
        event_y=locs[:, 1],
        event_frequency=freqs,
        local_roughness=local_roughness,
        correlation_coefficient=r,
        p_value=p,
        predicted_frequency=predicted,
        roughness_metric=roughness_metric,
    )
