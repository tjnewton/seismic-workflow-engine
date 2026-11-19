"""LAS/LAZ point-cloud loading.

Provides a lightweight wrapper around `laspy <https://laspy.readthedocs.io/>`_
for loading point-cloud data with optional spatial filtering, classification
filtering, and subsampling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class PointCloud(NamedTuple):
    """Container for point-cloud data.

    Attributes
    ----------
    points : ndarray
        Point coordinates, shape ``(N, 3)`` with columns ``[x, y, z]``.
    colors : ndarray or None
        RGB colors normalized to ``[0, 1]``, shape ``(N, 3)``, or
        ``None`` if color data is not available.
    classification : ndarray or None
        LAS classification codes, shape ``(N,)``, or ``None`` if not
        available.
    """

    points: NDArray[np.floating]
    colors: NDArray[np.floating] | None
    classification: NDArray[np.integer] | None


def load_point_cloud(
    las_path: str | Path,
    *,
    bounds: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]
    | None = None,
    classification_filter: list[int] | None = None,
    subsample: int | None = None,
) -> PointCloud:
    """Load a LAS/LAZ point cloud.

    Parameters
    ----------
    las_path : str or Path
        Path to a ``.las`` or ``.laz`` file.
    bounds : tuple of 3 (min, max) tuples, optional
        Spatial bounding box
        ``((x_min, x_max), (y_min, y_max), (z_min, z_max))``.
        Points outside the box are excluded.
    classification_filter : list of int, optional
        Keep only points whose classification code is in this list.
    subsample : int, optional
        If given, uniformly subsample to every *n*-th point after all
        other filters have been applied.

    Returns
    -------
    PointCloud
        Loaded (and optionally filtered/subsampled) point cloud.

    Raises
    ------
    ImportError
        If ``laspy`` is not installed.
    """
    try:
        import laspy
    except ImportError as exc:
        raise ImportError(
            "laspy is required for load_point_cloud(). Install it with: pip install laspy"
        ) from exc

    las = laspy.read(las_path)

    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    points = np.column_stack([x, y, z])

    # Extract optional RGB colors (normalize from 16-bit).
    colors: NDArray[np.floating] | None = None
    if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
        try:
            red = np.asarray(las.red, dtype=np.float64)
            green = np.asarray(las.green, dtype=np.float64)
            blue = np.asarray(las.blue, dtype=np.float64)
            max_val = max(red.max(), green.max(), blue.max(), 1.0)
            colors = np.column_stack([red, green, blue]) / max_val
        except (AttributeError, ValueError, TypeError, IndexError):
            colors = None

    # Extract optional classification.
    classification: NDArray[np.integer] | None = None
    if hasattr(las, "classification"):
        try:
            classification = np.asarray(las.classification)
        except (AttributeError, ValueError, TypeError, IndexError):
            classification = None

    # Apply spatial bounds filter.
    if bounds is not None:
        mask = (
            (points[:, 0] >= bounds[0][0])
            & (points[:, 0] <= bounds[0][1])
            & (points[:, 1] >= bounds[1][0])
            & (points[:, 1] <= bounds[1][1])
            & (points[:, 2] >= bounds[2][0])
            & (points[:, 2] <= bounds[2][1])
        )
        points = points[mask]
        if colors is not None:
            colors = colors[mask]
        if classification is not None:
            classification = classification[mask]

    # Apply classification filter.
    if classification_filter is not None and classification is not None:
        mask = np.isin(classification, classification_filter)
        points = points[mask]
        if colors is not None:
            colors = colors[mask]
        classification = classification[mask]

    # Apply uniform subsampling.
    if subsample is not None and subsample > 1:
        points = points[::subsample]
        if colors is not None:
            colors = colors[::subsample]
        if classification is not None:
            classification = classification[::subsample]

    logger.info(
        "Loaded %d points from %s",
        len(points),
        las_path,
    )

    return PointCloud(
        points=points,
        colors=colors,
        classification=classification,
    )
