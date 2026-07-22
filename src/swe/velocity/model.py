"""Velocity model data types and I/O.

Provides containers for 1-D and 3-D velocity models and precomputed
travel-time grids, along with functions for reading, building, saving,
and loading these objects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class VelocityModel(NamedTuple):
    """Grid-based velocity model stored as slowness.

    Attributes
    ----------
    x : ndarray
        1-D array of grid node coordinates along the x-axis.
    y : ndarray
        1-D array of grid node coordinates along the y-axis.
    z : ndarray
        1-D array of grid node coordinates along the z-axis.
    slowness : ndarray
        Cell-centred slowness values with shape
        ``(len(x)-1, len(y)-1, len(z)-1)``.
    model_type : str
        Either ``"1d"`` or ``"3d"``.
    """

    x: NDArray[np.floating]
    y: NDArray[np.floating]
    z: NDArray[np.floating]
    slowness: NDArray[np.floating]
    model_type: str


class TravelTimeGrid(NamedTuple):
    """Precomputed travel times from stations to every grid node.

    Attributes
    ----------
    grid_x : ndarray
        1-D array of grid node x-coordinates.
    grid_y : ndarray
        1-D array of grid node y-coordinates.
    grid_z : ndarray
        1-D array of grid node z-coordinates.
    station_coords : ndarray
        Station positions, shape ``(n_stations, 3)``.
    travel_times : ndarray
        Travel times, shape ``(n_stations, nx, ny, nz)``.
    """

    grid_x: NDArray[np.floating]
    grid_y: NDArray[np.floating]
    grid_z: NDArray[np.floating]
    station_coords: NDArray[np.floating]
    travel_times: NDArray[np.floating]


# ---------------------------------------------------------------------------
# I/O functions
# ---------------------------------------------------------------------------


def read_velocity_model_1d(
    model_path: str | Path,
    *,
    grid_bounds: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ],
    grid_spacing: tuple[float, float, float],
    max_cells_per_axis: int = 200,
) -> VelocityModel:
    """Read a 1-D velocity profile and expand it onto a 3-D grid.

    The input file is a two-column text file with ``depth  velocity``
    rows.  The 1-D profile is interpolated at cell centres and
    broadcast laterally to fill the 3-D slowness grid.

    Parameters
    ----------
    model_path : str or Path
        Path to a two-column text file (depth, velocity).
    grid_bounds : tuple of 3 (min, max) tuples
        ``((x_min, x_max), (y_min, y_max), (z_min, z_max))``.
    grid_spacing : tuple of 3 floats
        ``(dx, dy, dz)`` grid spacing.
    max_cells_per_axis : int
        Safety cap on grid dimensions to prevent excessive memory use.

    Returns
    -------
    VelocityModel
        Velocity model with ``model_type="1d"``.

    Raises
    ------
    ValueError
        If grid dimensions exceed *max_cells_per_axis*.
    """
    model_data = np.loadtxt(model_path)
    depths = model_data[:, 0]
    velocities = model_data[:, 1]

    dx, dy, dz = grid_spacing
    x = np.arange(grid_bounds[0][0], grid_bounds[0][1] + dx / 2, dx)
    y = np.arange(grid_bounds[1][0], grid_bounds[1][1] + dy / 2, dy)
    z = np.arange(grid_bounds[2][0], grid_bounds[2][1] + dz / 2, dz)

    for name, arr in [("x", x), ("y", y), ("z", z)]:
        if len(arr) - 1 > max_cells_per_axis:
            raise ValueError(
                f"Grid dimension along {name} ({len(arr) - 1} cells) exceeds "
                f"max_cells_per_axis={max_cells_per_axis}."
            )

    # Interpolate slowness at cell centres along z.
    z_cell = 0.5 * (z[:-1] + z[1:])
    slow_1d = np.interp(z_cell, depths, 1.0 / velocities)

    # Broadcast to 3-D: constant in x-y, varies in z.
    slowness = np.broadcast_to(
        slow_1d[np.newaxis, np.newaxis, :],
        (len(x) - 1, len(y) - 1, len(z) - 1),
    ).copy()

    logger.info(
        "Read 1-D velocity model from %s: grid %d x %d x %d cells",
        model_path,
        len(x) - 1,
        len(y) - 1,
        len(z) - 1,
    )

    return VelocityModel(
        x=x,
        y=y,
        z=z,
        slowness=slowness,
        model_type="1d",
    )


def read_velocity_model_3d(model_path: str | Path) -> VelocityModel:
    """Load a 3-D velocity model from an NPZ archive.

    Parameters
    ----------
    model_path : str or Path
        Path to an ``.npz`` file containing keys ``x``, ``y``, ``z``,
        and ``slowness``.

    Returns
    -------
    VelocityModel
        Velocity model with ``model_type="3d"``.
    """
    data = np.load(model_path)
    model = VelocityModel(
        x=data["x"],
        y=data["y"],
        z=data["z"],
        slowness=data["slowness"],
        model_type="3d",
    )
    logger.info("Loaded 3-D velocity model from %s", model_path)
    return model


def build_velocity_model_3d(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    z: NDArray[np.floating],
    slowness: NDArray[np.floating],
) -> VelocityModel:
    """Construct a 3-D velocity model from arrays.

    Parameters
    ----------
    x, y, z : ndarray
        1-D arrays of grid node coordinates.
    slowness : ndarray
        Cell-centred slowness, shape ``(len(x)-1, len(y)-1, len(z)-1)``.

    Returns
    -------
    VelocityModel
        Velocity model with ``model_type="3d"``.

    Raises
    ------
    ValueError
        If *slowness* shape does not match ``(nx-1, ny-1, nz-1)``.
    """
    expected = (len(x) - 1, len(y) - 1, len(z) - 1)
    if slowness.shape != expected:
        raise ValueError(
            f"slowness shape {slowness.shape} does not match expected {expected} from grid axes."
        )
    return VelocityModel(
        x=np.asarray(x, dtype=np.float64),
        y=np.asarray(y, dtype=np.float64),
        z=np.asarray(z, dtype=np.float64),
        slowness=np.asarray(slowness, dtype=np.float64),
        model_type="3d",
    )


def save_velocity_model(
    model: VelocityModel,
    output_path: str | Path,
) -> Path:
    """Save a velocity model to a compressed NPZ archive.

    Parameters
    ----------
    model : VelocityModel
        Velocity model to save.
    output_path : str or Path
        Destination file path.

    Returns
    -------
    Path
        The output path.
    """
    output_path = Path(output_path)
    np.savez_compressed(
        output_path,
        x=model.x,
        y=model.y,
        z=model.z,
        slowness=model.slowness,
    )
    logger.info("Saved velocity model to %s", output_path)
    return output_path


def save_travel_time_grid(
    ttg: TravelTimeGrid,
    output_path: str | Path,
) -> Path:
    """Save a travel-time grid to a compressed NPZ archive.

    Parameters
    ----------
    ttg : TravelTimeGrid
        Travel-time grid to cache.
    output_path : str or Path
        Destination file path.

    Returns
    -------
    Path
        The output path.
    """
    output_path = Path(output_path)
    np.savez_compressed(
        output_path,
        grid_x=ttg.grid_x,
        grid_y=ttg.grid_y,
        grid_z=ttg.grid_z,
        station_coords=ttg.station_coords,
        travel_times=ttg.travel_times,
    )
    logger.info("Saved travel-time grid to %s", output_path)
    return output_path


def load_travel_time_grid(path: str | Path) -> TravelTimeGrid:
    """Load a cached travel-time grid from an NPZ archive.

    Parameters
    ----------
    path : str or Path
        Path to the ``.npz`` file.

    Returns
    -------
    TravelTimeGrid
    """
    data = np.load(path)
    ttg = TravelTimeGrid(
        grid_x=data["grid_x"],
        grid_y=data["grid_y"],
        grid_z=data["grid_z"],
        station_coords=data["station_coords"],
        travel_times=data["travel_times"],
    )
    logger.info("Loaded travel-time grid from %s", path)
    return ttg
