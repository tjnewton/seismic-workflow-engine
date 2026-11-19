"""Grid-based travel-time earthquake location.

Provides a grid-search locator that uses precomputed travel-time grids
(e.g. from :func:`swe.velocity.raytrace.compute_travel_time_grid`)
instead of constant-velocity straight-ray approximations.

Uses duck typing on the *travel_time_grid* argument to avoid circular
imports with :mod:`swe.velocity`.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from swe.location.amplitude import LocationResult

logger = logging.getLogger(__name__)


def locate_by_travel_time_grid(
    observed_times: NDArray[np.floating],
    travel_time_grid,
    *,
    reference_station_index: int = 0,
) -> LocationResult:
    """Locate an event using a precomputed travel-time grid.

    For each candidate source position on the grid, reads precomputed
    travel times and minimizes the L1 residual of differential times
    relative to a reference station.  This is the same algorithm as
    :func:`~swe.location.amplitude.locate_by_travel_time` but uses
    full 3-D ray-traced travel times instead of ``dist / velocity``.

    Parameters
    ----------
    observed_times : ndarray, shape (n_stations,)
        Observed arrival times (seconds) at each station.
    travel_time_grid
        A precomputed travel-time object with attributes:

        - ``grid_x``, ``grid_y``, ``grid_z`` : 1-D ndarray of node
          coordinates.
        - ``travel_times`` : ndarray of shape
          ``(n_stations, nx, ny, nz)``.

        Typically a :class:`~swe.velocity.model.TravelTimeGrid`, but
        any object with these attributes works (duck typing).
    reference_station_index : int
        Index of the reference station for differential times.

    Returns
    -------
    LocationResult
        Best-fit ``(x, y, z, misfit)``.

    Raises
    ------
    ValueError
        If the number of observed times does not match the number of
        stations in the grid.
    """
    observed_times = np.asarray(observed_times, dtype=np.float64)

    tt = travel_time_grid.travel_times
    n_stations = tt.shape[0]

    if observed_times.shape[0] != n_stations:
        msg = (
            f"observed_times length ({observed_times.shape[0]}) does not "
            f"match travel_time_grid stations ({n_stations})"
        )
        raise ValueError(msg)

    grid_x = travel_time_grid.grid_x
    grid_y = travel_time_grid.grid_y
    grid_z = travel_time_grid.grid_z

    logger.info(
        "Travel-time grid-search: %d x %d x %d nodes, %d stations",
        len(grid_x),
        len(grid_y),
        len(grid_z),
        n_stations,
    )

    # Observed differential times relative to the reference station.
    obs_dt = observed_times - observed_times[reference_station_index]

    # Reference station travel times at every grid node.
    ref_tt = tt[reference_station_index]

    # Accumulate L1 residuals.
    residuals = np.zeros(ref_tt.shape)
    for ii in range(n_stations):
        if ii == reference_station_index:
            continue
        pred_dt = tt[ii] - ref_tt
        residuals += np.abs(obs_dt[ii] - pred_dt)

    # Find global minimum.
    min_idx = np.unravel_index(np.argmin(residuals), residuals.shape)
    best_x = float(grid_x[min_idx[0]])
    best_y = float(grid_y[min_idx[1]])
    best_z = float(grid_z[min_idx[2]])
    best_residual = float(residuals[min_idx])

    logger.info(
        "Best grid-TT location: x=%.1f  y=%.1f  z=%.1f  residual=%.4f",
        best_x,
        best_y,
        best_z,
        best_residual,
    )

    return LocationResult(
        x=best_x,
        y=best_y,
        z=best_z,
        misfit=best_residual,
    )
