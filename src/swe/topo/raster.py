"""Raster and DEM elevation queries.

Provides functions for extracting elevations and gridded values from
raster data (e.g. GeoTIFF DEMs) using GDAL / rasterio.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ElevationGrid(NamedTuple):
    """Gridded elevation data extracted from a raster.

    Attributes
    ----------
    x : ndarray
        1-D array of x-coordinates (easting or longitude).
    y : ndarray
        1-D array of y-coordinates (northing or latitude).
    elevation : ndarray
        2-D elevation grid, shape ``(len(y), len(x))``.
    crs : str or None
        Coordinate reference system string (e.g. ``"EPSG:32610"``),
        or ``None`` if not available.
    """

    x: NDArray[np.floating]
    y: NDArray[np.floating]
    elevation: NDArray[np.floating]
    crs: str | None


def elevations_from_raster(
    raster_path: str | Path,
    coordinates: np.ndarray,
    *,
    target_crs: str | None = None,
) -> np.ndarray:
    """Query DEM elevations at the given coordinates.

    Parameters
    ----------
    raster_path : str or Path
        Path to a raster file (e.g. GeoTIFF) readable by GDAL / rasterio.
    coordinates : numpy.ndarray
        Array of shape ``(N, 2)`` with columns ``[x, y]``
        (e.g. ``[longitude, latitude]`` or ``[easting, northing]``).
    target_crs : str, optional
        CRS of the input *coordinates* (e.g. ``"EPSG:4326"``).  If
        provided and different from the raster CRS, coordinates are
        reprojected before sampling.  If ``None``, coordinates are
        assumed to already match the raster CRS.

    Returns
    -------
    numpy.ndarray
        Elevation values at each coordinate, shape ``(N,)``.

    Raises
    ------
    ImportError
        If ``rasterio`` (or ``pyproj`` when *target_crs* is used) is
        not installed.
    """
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "rasterio is required for elevations_from_raster(). "
            "Install it with: conda install -c conda-forge rasterio"
        ) from exc

    coordinates = np.asarray(coordinates)
    if coordinates.ndim == 1:
        coordinates = coordinates.reshape(1, -1)

    with rasterio.open(raster_path) as src:
        if target_crs is not None and str(src.crs) != target_crs:
            try:
                from pyproj import Transformer
            except ImportError as exc:
                raise ImportError(
                    "pyproj is required for CRS reprojection. "
                    "Install it with: conda install -c conda-forge pyproj"
                ) from exc

            transformer = Transformer.from_crs(
                target_crs, str(src.crs), always_xy=True
            )
            xs, ys = transformer.transform(
                coordinates[:, 0], coordinates[:, 1]
            )
            xy_coords = list(zip(xs, ys))
        else:
            xy_coords = list(zip(coordinates[:, 0], coordinates[:, 1]))

        elevations = np.array(
            [val[0] for val in src.sample(xy_coords)]
        )

    logger.info(
        "Sampled %d elevations from %s", len(elevations), raster_path
    )
    return elevations


def grid_from_raster(
    raster_path: str | Path,
    *,
    bounds: tuple[float, float, float, float] | None = None,
    resolution: float | None = None,
    target_crs: str | None = None,
    fill_nodata: bool = True,
) -> ElevationGrid:
    """Extract a gridded elevation surface from a raster file.

    Parameters
    ----------
    raster_path : str or Path
        Path to a raster file (e.g. GeoTIFF) readable by rasterio.
    bounds : tuple of 4 floats, optional
        ``(left, bottom, right, top)`` bounding box in the raster's
        CRS (or *target_crs* if given).  If ``None``, the full raster
        extent is used.
    resolution : float, optional
        Output grid cell size.  If ``None``, the native raster
        resolution is used.
    target_crs : str, optional
        CRS of the input *bounds* (e.g. ``"EPSG:4326"``).  If provided
        and different from the raster CRS, bounds are reprojected
        before windowing.
    fill_nodata : bool
        If ``True``, fill nodata gaps using ``rasterio.fill.fillnodata``.

    Returns
    -------
    ElevationGrid
        Gridded elevation data.

    Raises
    ------
    ImportError
        If ``rasterio`` (or ``pyproj`` when *target_crs* is used) is
        not installed.
    """
    try:
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError as exc:
        raise ImportError(
            "rasterio is required for grid_from_raster(). "
            "Install it with: conda install -c conda-forge rasterio"
        ) from exc

    with rasterio.open(raster_path) as src:
        crs_str = str(src.crs) if src.crs else None

        if bounds is not None:
            # Reproject bounds if needed.
            if target_crs is not None and crs_str != target_crs:
                try:
                    from pyproj import Transformer
                except ImportError as exc:
                    raise ImportError(
                        "pyproj is required for CRS reprojection. "
                        "Install it with: conda install -c conda-forge pyproj"
                    ) from exc
                transformer = Transformer.from_crs(
                    target_crs, crs_str, always_xy=True
                )
                left, bottom = transformer.transform(bounds[0], bounds[1])
                right, top = transformer.transform(bounds[2], bounds[3])
                bounds = (left, bottom, right, top)

            window = from_bounds(*bounds, transform=src.transform)
            data = src.read(1, window=window)
            win_transform = src.window_transform(window)
        else:
            data = src.read(1)
            win_transform = src.transform

        # Fill nodata gaps.
        if fill_nodata and src.nodata is not None:
            from rasterio.fill import fillnodata as _fillnodata

            mask = data != src.nodata
            data = _fillnodata(data, mask)

        # Build coordinate axes from the window transform.
        nrows, ncols = data.shape
        x = np.array(
            [win_transform.c + (col + 0.5) * win_transform.a for col in range(ncols)]
        )
        y = np.array(
            [win_transform.f + (row + 0.5) * win_transform.e for row in range(nrows)]
        )

        # Resample if resolution requested.
        if resolution is not None and resolution > 0:
            new_x = np.arange(x[0], x[-1], resolution)
            new_y = np.arange(y[0], y[-1], -resolution if y[0] > y[-1] else resolution)
            if len(new_x) > 1 and len(new_y) > 1:
                from scipy.interpolate import RegularGridInterpolator

                interp = RegularGridInterpolator(
                    (y, x), data.astype(np.float64),
                    bounds_error=False, fill_value=None,
                )
                yy, xx = np.meshgrid(new_y, new_x, indexing="ij")
                data = interp((yy, xx))
                x = new_x
                y = new_y

    logger.info(
        "Extracted elevation grid (%d x %d) from %s",
        data.shape[0], data.shape[1], raster_path,
    )

    return ElevationGrid(
        x=x, y=y, elevation=data.astype(np.float64), crs=crs_str,
    )


def terrain_correction(
    station_coords: NDArray[np.floating],
    raster_path: str | Path,
    *,
    target_crs: str | None = None,
) -> NDArray[np.floating]:
    """Replace station elevations with DEM-sampled values.

    Creates a copy of *station_coords* with the z-column (index 2)
    overwritten by elevations sampled from the raster.

    Parameters
    ----------
    station_coords : ndarray, shape (N, 3)
        Station coordinates ``[x, y, z]``.
    raster_path : str or Path
        Path to a raster DEM.
    target_crs : str, optional
        CRS of the station coordinates if different from the raster CRS.

    Returns
    -------
    ndarray, shape (N, 3)
        Copy of *station_coords* with updated elevations.
    """
    coords = np.array(station_coords, dtype=np.float64)
    xy = coords[:, :2]
    elevations = elevations_from_raster(
        raster_path, xy, target_crs=target_crs,
    )
    result = coords.copy()
    result[:, 2] = elevations
    logger.info(
        "Terrain-corrected %d station elevations from %s",
        len(elevations), raster_path,
    )
    return result
