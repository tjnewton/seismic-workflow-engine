"""Ray tracing for seismic travel-time computation.

Integration with `ttcrpy <https://ttcrpy.readthedocs.io/>`_ for
graph-based shortest-path ray tracing through 2-D and 3-D velocity models.
It replaces an earlier MATLAB-based ray-tracing interface.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from swe.velocity.model import TravelTimeGrid, VelocityModel

logger = logging.getLogger(__name__)


def _require_ttcrpy():
    """Import and return Grid3d, raising a clear error if unavailable."""
    try:
        from ttcrpy.rgrid import Grid3d
    except ImportError as exc:
        raise ImportError(
            "ttcrpy is required for ray tracing. Install it with: pip install ttcrpy"
        ) from exc
    return Grid3d


def compute_travel_times(
    velocity_model: str | Path | VelocityModel,
    sources: np.ndarray,
    receivers: np.ndarray,
    *,
    model_type: str = "1d",
) -> np.ndarray:
    """Compute seismic travel times through a velocity model.

    Parameters
    ----------
    velocity_model : str, Path, or VelocityModel
        Path to a velocity model file **or** a pre-built
        :class:`~swe.velocity.model.VelocityModel`.  For
        ``model_type="1d"`` with a file path, a two-column text file
        with ``depth  velocity`` rows is expected.  When a
        ``VelocityModel`` is passed, the file path and *model_type*
        parameter are ignored and the model's own grid/slowness are
        used directly.
    sources : numpy.ndarray
        Source locations, shape ``(N, 3)`` with columns
        ``[x, y, z]``.
    receivers : numpy.ndarray
        Receiver locations, shape ``(M, 3)``.
    model_type : str
        Type of velocity model.  Only used when *velocity_model* is a
        file path.  Currently ``"1d"`` is supported for file paths.

    Returns
    -------
    numpy.ndarray
        Travel-time matrix of shape ``(N, M)`` in seconds.

    Raises
    ------
    ImportError
        If ttcrpy is not installed.
    ValueError
        If *model_type* is not supported (file-path mode).
    """
    Grid3d = _require_ttcrpy()

    sources = np.asarray(sources, dtype=np.float64)
    receivers = np.asarray(receivers, dtype=np.float64)

    if sources.ndim == 1:
        sources = sources.reshape(1, -1)
    if receivers.ndim == 1:
        receivers = receivers.reshape(1, -1)

    # ------------------------------------------------------------------
    # Resolve velocity model
    # ------------------------------------------------------------------
    if isinstance(velocity_model, VelocityModel):
        x, y, z = velocity_model.x, velocity_model.y, velocity_model.z
        slow_3d = velocity_model.slowness
    else:
        if model_type != "1d":
            raise ValueError(
                f"Unsupported model_type={model_type!r}. "
                "Currently only '1d' is supported for file paths."
            )

        # Read 1-D velocity model: two columns (depth, velocity)
        model_data = np.loadtxt(velocity_model)
        depths = model_data[:, 0]
        velocities = model_data[:, 1]

        # Build a 3-D grid that spans sources and receivers
        all_pts = np.vstack([sources, receivers])
        x_min, y_min, z_min = all_pts.min(axis=0)
        x_max, y_max, z_max = all_pts.max(axis=0)

        # Extend bounds by depth range to ensure grid coverage
        depth_range = depths[-1] - depths[0]
        pad = max(depth_range * 0.1, 1.0)
        x_min -= pad
        x_max += pad
        y_min -= pad
        y_max += pad
        z_min = min(z_min, depths[0]) - pad
        z_max = max(z_max, depths[-1]) + pad

        # Grid resolution: use the minimum layer thickness
        if len(depths) > 1:
            dz = float(np.min(np.diff(depths)))
            dz = max(dz, 0.01)
        else:
            dz = 1.0

        nx = max(int(np.ceil((x_max - x_min) / dz)), 2)
        ny = max(int(np.ceil((y_max - y_min) / dz)), 2)
        nz = max(int(np.ceil((z_max - z_min) / dz)), 2)

        # Cap grid size to keep computation reasonable
        max_cells = 200
        nx = min(nx, max_cells)
        ny = min(ny, max_cells)
        nz = min(nz, max_cells)

        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        z = np.linspace(z_min, z_max, nz)

        # Build cell-centred slowness grid
        z_cell = 0.5 * (z[:-1] + z[1:])
        slow_1d = np.interp(z_cell, depths, 1.0 / velocities)
        slow_3d = np.broadcast_to(
            slow_1d[np.newaxis, np.newaxis, :],
            (nx - 1, ny - 1, nz - 1),
        ).copy()

    grid = Grid3d(x, y, z, cell_slowness=True)

    n_src = sources.shape[0]
    n_rcv = receivers.shape[0]
    tt_matrix = np.empty((n_src, n_rcv))

    for i in range(n_src):
        tt, _ = grid.raytrace(
            sources[i : i + 1, :],
            receivers,
            slowness=slow_3d.ravel(order="C"),
        )
        tt_matrix[i, :] = tt

    logger.info(
        "Computed %d x %d travel-time matrix",
        n_src,
        n_rcv,
    )
    return tt_matrix


def compute_travel_time_grid(
    velocity_model: VelocityModel,
    station_coords: np.ndarray,
) -> TravelTimeGrid:
    """Compute travel times from every station to all grid nodes.

    Uses ttcrpy reciprocity: each station is treated as a source and
    travel times to every grid node are computed via shortest-path
    ray tracing.

    Parameters
    ----------
    velocity_model : VelocityModel
        A pre-built velocity model (1-D or 3-D).
    station_coords : numpy.ndarray
        Station positions, shape ``(n_stations, 3)``.

    Returns
    -------
    TravelTimeGrid
        Precomputed travel times suitable for
        :func:`swe.location.travel_time.locate_by_travel_time_grid`.

    Raises
    ------
    ImportError
        If ttcrpy is not installed.
    """
    Grid3d = _require_ttcrpy()

    station_coords = np.asarray(station_coords, dtype=np.float64)
    if station_coords.ndim == 1:
        station_coords = station_coords.reshape(1, -1)

    x, y, z = velocity_model.x, velocity_model.y, velocity_model.z
    slow_flat = velocity_model.slowness.ravel(order="C")

    grid = Grid3d(x, y, z, cell_slowness=True)

    # Build receiver grid: all node positions.
    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
    all_nodes = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    nx, ny, nz = len(x), len(y), len(z)
    n_stations = station_coords.shape[0]
    travel_times = np.empty((n_stations, nx, ny, nz))

    for idx in range(n_stations):
        src = station_coords[idx : idx + 1, :]
        tt, _ = grid.raytrace(
            src,
            all_nodes,
            slowness=slow_flat,
        )
        travel_times[idx] = tt.reshape(nx, ny, nz)

    logger.info(
        "Computed travel-time grid: %d stations, %d x %d x %d nodes",
        n_stations,
        nx,
        ny,
        nz,
    )

    return TravelTimeGrid(
        grid_x=x,
        grid_y=y,
        grid_z=z,
        station_coords=station_coords,
        travel_times=travel_times,
    )
