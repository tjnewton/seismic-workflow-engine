"""Shared utility functions for the seismic-workflow-engine package."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np
import obspy
from obspy import Stream
from obspy.core.trace import Trace

if TYPE_CHECKING:
    from obspy.core.utcdatetime import UTCDateTime

logger = logging.getLogger(__name__)


def is_integer(value: object) -> bool:
    """True for a Python or NumPy integer; bool (and NumPy bool) is not an integer here."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def nearest(items: np.ndarray, pivot: float | UTCDateTime):
    """Find the value in *items* closest to *pivot*."""
    return min(items, key=lambda x: abs(x - pivot))


def nearest_ind(items: np.ndarray, pivot: float | UTCDateTime) -> int:
    """Return the index of the value in *items* closest to *pivot*."""
    time_diff = np.abs([date - pivot for date in items])
    return int(time_diff.argmin(0))


def max_amplitude(
    time_series: Stream | Trace | np.ndarray,
) -> float | np.ndarray:
    """Return the maximum absolute amplitude of a time series.

    Parameters
    ----------
    time_series : obspy.Stream, obspy.Trace, or numpy.ndarray
        Input data.  For 2-D arrays the maximum is computed per row.

    Returns
    -------
    float or numpy.ndarray
        Scalar for Stream/Trace/1-D arrays, 1-D array for 2-D input.

    Raises
    ------
    ValueError
        If the input array has more than 2 dimensions.
    """
    if isinstance(time_series, obspy.core.stream.Stream):
        if len(time_series) == 0:
            return 0.0
        trace_max = [np.nanmax(np.abs(tr.data)) for tr in time_series]
        return float(np.nanmax(trace_max))

    if isinstance(time_series, obspy.core.trace.Trace):
        return float(np.nanmax(np.abs(time_series.data)))

    if isinstance(time_series, np.ndarray):
        if time_series.ndim == 1:
            return float(np.nanmax(np.abs(time_series)))
        if time_series.ndim == 2:
            return np.asarray(np.nanmax(np.abs(time_series), axis=1))
        raise ValueError(f"Expected 1-D or 2-D array, got {time_series.ndim}-D")

    raise TypeError(f"Unsupported type: {type(time_series)}")


def snr(obspy_object: Stream | Trace) -> list[float]:
    """Compute per-trace signal-to-noise ratio.

    Signal-to-noise ratio is defined as the ratio of the maximum absolute
    amplitude to the RMS amplitude of the entire trace.

    Parameters
    ----------
    obspy_object : obspy.Stream or obspy.Trace
        Input waveform data.

    Returns
    -------
    list of float
        One SNR value per trace.  Empty list when the trace contains no data.
    """
    if isinstance(obspy_object, obspy.core.stream.Stream):
        snrs = []
        for trace in obspy_object:
            rms = np.sqrt(np.mean(trace.data**2))
            if rms == 0:
                snrs.append(0.0)
            else:
                snrs.append(float(np.max(np.abs(trace.data)) / rms))
        return snrs

    if isinstance(obspy_object, obspy.core.trace.Trace):
        if len(obspy_object.data) == 0:
            return []
        rms = np.sqrt(np.mean(obspy_object.data**2))
        if rms == 0:
            return [0.0]
        return [float(np.max(np.abs(obspy_object.data)) / rms)]

    raise TypeError(f"Unsupported type: {type(obspy_object)}")


def geo_to_local(
    location_dict: dict[str, list[float]],
    *,
    origin: tuple[float, float, float] | None = None,
) -> tuple[dict[str, np.ndarray], tuple[float, float, float]]:
    """Convert geographic station coordinates to a local Cartesian frame.

    Parameters
    ----------
    location_dict : dict
        ``{station: [longitude, latitude, elevation_km]}``: the same format
        used by :func:`~swe.association.associate_picks`.
    origin : tuple of float, optional
        ``(longitude, latitude, elevation_km)`` reference point.  If *None*,
        the centroid of all stations is used.

    Returns
    -------
    local_coords : dict
        ``{station: np.array([x_m, y_m, z_m])}`` where *x* is east,
        *y* is north, and *z* is depth below the origin elevation, all in
        meters.
    origin_used : tuple of float
        ``(longitude, latitude, elevation_km)`` of the reference point.
    """
    if not location_dict:
        if origin is None:
            origin = (0.0, 0.0, 0.0)
        return {}, origin

    if origin is None:
        lons = [v[0] for v in location_dict.values()]
        lats = [v[1] for v in location_dict.values()]
        elevs = [v[2] if len(v) > 2 else 0.0 for v in location_dict.values()]
        origin = (float(np.mean(lons)), float(np.mean(lats)), float(np.mean(elevs)))

    origin_lon, origin_lat, origin_elev = origin
    local_coords: dict[str, np.ndarray] = {}

    for station, coords in location_dict.items():
        lon = coords[0]
        lat = coords[1]
        elev = coords[2] if len(coords) > 2 else 0.0
        ew_dist = (
            ellipsoidal_distance(
                [origin_lon, origin_lat, 0.0],
                [lon, origin_lat, 0.0],
                two_dimensional=True,
            )
            * 1000.0
        )
        if lon < origin_lon:
            ew_dist = -ew_dist

        ns_dist = (
            ellipsoidal_distance(
                [origin_lon, origin_lat, 0.0],
                [origin_lon, lat, 0.0],
                two_dimensional=True,
            )
            * 1000.0
        )
        if lat < origin_lat:
            ns_dist = -ns_dist

        z = (origin_elev - elev) * 1000.0
        local_coords[station] = np.array([ew_dist, ns_dist, z])

    return local_coords, origin


def ellipsoidal_distance(
    point1: list[float],
    point2: list[float],
    *,
    two_dimensional: bool = False,
) -> float:
    """WGS-84 distance between two geographic points.

    Uses `geopy`'s implementation of the Karney (2013) geodesic algorithm
    for the horizontal component.

    Parameters
    ----------
    point1, point2 : list of float
        ``[longitude, latitude, depth_km]``.  Depth is in kilometers
        (positive downward).
    two_dimensional : bool
        If *True*, ignore depth and return the surface distance only.

    Returns
    -------
    float
        Distance in kilometers.
    """
    from geopy import Point
    from geopy.distance import distance as geodesic_distance

    lon1, lat1, depth1 = point1
    lon2, lat2, depth2 = point2
    p1 = Point(lat1, lon1)
    p2 = Point(lat2, lon2)
    dist_2d = geodesic_distance(p1, p2).km
    if two_dimensional:
        return float(abs(dist_2d))
    dist_3d = math.sqrt(dist_2d**2 + (depth1 - depth2) ** 2)
    return abs(dist_3d)
