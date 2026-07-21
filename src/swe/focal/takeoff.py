"""Takeoff angle and azimuth computation for ray paths.

Computes the takeoff angle and azimuth from a seismic source to
recording stations, using either a homogeneous half-space or a
layered 1-D velocity model.

Coordinate convention: x=East, y=North, z=Down (positive downward),
matching the convention in :mod:`swe.stress.inversion`.
"""

from __future__ import annotations

import logging
from typing import NamedTuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

__all__ = [
    "RayParameters",
    "compute_takeoff_angles",
    "compute_takeoff_angles_1d",
]


class RayParameters(NamedTuple):
    """Ray geometry from source to stations."""

    stations: list[str]
    """Station identifiers."""
    azimuths: NDArray[np.floating]
    """Source-to-station azimuths in degrees (clockwise from North)."""
    takeoff_angles: NDArray[np.floating]
    """Takeoff angles in degrees (0=up, 90=horizontal, 180=down)."""
    distances: NDArray[np.floating]
    """Source-to-station distances in km."""


def compute_takeoff_angles(
    source: dict[str, float],
    station_coords: dict[str, NDArray[np.floating] | list[float]],
    *,
    velocity: float = 6.0,
) -> RayParameters:
    """Compute takeoff angles using straight-ray (homogeneous) approximation.

    Parameters
    ----------
    source : dict
        Source location with keys ``'x'``, ``'y'``, ``'z'`` in km.
        Convention: x=East, y=North, z=Down (depth positive).
    station_coords : dict
        Mapping of station name to ``[x, y, z]`` array in km.
    velocity : float
        Homogeneous P-wave velocity in km/s (used only for documentation;
        straight-ray geometry is independent of velocity).

    Returns
    -------
    RayParameters
        Stations, azimuths, takeoff angles, and distances.
    """
    src = np.array([source["x"], source["y"], source["z"]], dtype=np.float64)

    stations: list[str] = []
    azimuths: list[float] = []
    takeoffs: list[float] = []
    dists: list[float] = []

    for name, coords in station_coords.items():
        sta = np.asarray(coords, dtype=np.float64)
        diff = sta - src  # [dx_east, dy_north, dz_down]

        dist = np.linalg.norm(diff)
        if dist < 1e-12:
            logger.warning("Station %s at source location, skipping", name)
            continue

        # Azimuth: angle from North (y-axis) clockwise to East (x-axis)
        azimuth = np.degrees(np.arctan2(diff[0], diff[1]))
        if azimuth < 0:
            azimuth += 360.0

        # Takeoff angle: angle from vertical UP (-z direction)
        # The ray direction from source to station is diff/dist
        # Vertical up in z-down convention is [0, 0, -1]
        # cos(takeoff) = dot(ray_dir, up) = -diff[2] / dist
        cos_takeoff = -diff[2] / dist
        cos_takeoff = np.clip(cos_takeoff, -1.0, 1.0)
        takeoff = np.degrees(np.arccos(cos_takeoff))

        stations.append(name)
        azimuths.append(float(azimuth))
        takeoffs.append(float(takeoff))
        dists.append(float(dist))

    return RayParameters(
        stations=stations,
        azimuths=np.array(azimuths, dtype=np.float64),
        takeoff_angles=np.array(takeoffs, dtype=np.float64),
        distances=np.array(dists, dtype=np.float64),
    )


def compute_takeoff_angles_1d(
    source: dict[str, float],
    station_coords: dict[str, NDArray[np.floating] | list[float]],
    *,
    velocity_profile: NDArray[np.floating] | list[list[float]],
) -> RayParameters:
    """Compute takeoff angles through a layered 1-D velocity model.

    Uses Snell's law ray tracing through horizontal layers to
    determine the takeoff angle at the source depth.

    Parameters
    ----------
    source : dict
        Source location with keys ``'x'``, ``'y'``, ``'z'`` in km.
    station_coords : dict
        Mapping of station name to ``[x, y, z]`` array in km.
    velocity_profile : array-like
        Shape ``(N, 2)`` with columns ``[depth_km, velocity_km_s]``.
        Depths must be monotonically increasing.  The velocity at
        each depth applies from that depth down to the next layer
        boundary.

    Returns
    -------
    RayParameters
        Stations, azimuths, takeoff angles, and distances.
    """
    profile = np.asarray(velocity_profile, dtype=np.float64)
    if profile.ndim != 2 or profile.shape[1] != 2:
        raise ValueError("velocity_profile must have shape (N, 2)")

    depths = profile[:, 0]
    velocities = profile[:, 1]

    src = np.array([source["x"], source["y"], source["z"]], dtype=np.float64)

    stations: list[str] = []
    azimuths_out: list[float] = []
    takeoffs_out: list[float] = []
    dists_out: list[float] = []

    for name, coords in station_coords.items():
        sta = np.asarray(coords, dtype=np.float64)
        diff = sta - src

        horiz_dist = np.sqrt(diff[0] ** 2 + diff[1] ** 2)
        total_dist = np.linalg.norm(diff)

        if total_dist < 1e-12:
            logger.warning("Station %s at source location, skipping", name)
            continue

        # Azimuth (same as homogeneous case)
        azimuth = np.degrees(np.arctan2(diff[0], diff[1]))
        if azimuth < 0:
            azimuth += 360.0

        # For 1-D layered model: trace a ray from source depth to station depth
        src_depth = src[2]
        sta_depth = sta[2]

        # Get velocity at source depth
        src_vel = _velocity_at_depth(src_depth, depths, velocities)

        # Binary search for the ray parameter p that matches the
        # horizontal distance
        takeoff = _trace_ray_1d(
            src_depth, sta_depth, horiz_dist, depths, velocities
        )

        stations.append(name)
        azimuths_out.append(float(azimuth))
        takeoffs_out.append(float(takeoff))
        dists_out.append(float(total_dist))

    return RayParameters(
        stations=stations,
        azimuths=np.array(azimuths_out, dtype=np.float64),
        takeoff_angles=np.array(takeoffs_out, dtype=np.float64),
        distances=np.array(dists_out, dtype=np.float64),
    )


def _velocity_at_depth(
    depth: float,
    layer_depths: NDArray[np.floating],
    layer_velocities: NDArray[np.floating],
) -> float:
    """Return velocity at a given depth from a 1-D profile."""
    idx = np.searchsorted(layer_depths, depth, side="right") - 1
    idx = max(0, min(idx, len(layer_velocities) - 1))
    return float(layer_velocities[idx])


def _trace_ray_1d(
    src_depth: float,
    sta_depth: float,
    target_horiz_dist: float,
    layer_depths: NDArray[np.floating],
    layer_velocities: NDArray[np.floating],
    *,
    n_iter: int = 50,
) -> float:
    """Find takeoff angle via bisection for a 1-D layered model.

    Returns takeoff angle in degrees (0=up, 180=down).
    """
    # Determine if ray goes up or down
    going_up = sta_depth < src_depth

    # If source and station at same depth and zero horizontal distance
    if abs(target_horiz_dist) < 1e-12:
        return 0.0 if going_up else 180.0

    # Binary search over takeoff angle
    # For upgoing rays: takeoff angles 0-90 (0=straight up, 90=horizontal)
    # For downgoing rays that refract back up: start with straight-ray estimate
    if going_up:
        theta_low = 0.01
        theta_high = 89.99
    else:
        theta_low = 90.01
        theta_high = 179.99

    src_vel = _velocity_at_depth(src_depth, layer_depths, layer_velocities)

    for _ in range(n_iter):
        theta_mid = (theta_low + theta_high) / 2.0
        horiz = _compute_horizontal_distance(
            src_depth, sta_depth, theta_mid, layer_depths, layer_velocities
        )

        if abs(horiz - target_horiz_dist) < 1e-6:
            return theta_mid

        if horiz < target_horiz_dist:
            if going_up:
                theta_low = theta_mid
            else:
                theta_high = theta_mid
        else:
            if going_up:
                theta_high = theta_mid
            else:
                theta_low = theta_mid

    return (theta_low + theta_high) / 2.0


def _compute_horizontal_distance(
    src_depth: float,
    sta_depth: float,
    takeoff_deg: float,
    layer_depths: NDArray[np.floating],
    layer_velocities: NDArray[np.floating],
) -> float:
    """Compute horizontal distance for a ray with given takeoff angle.

    Uses Snell's law through horizontal layers.
    """
    takeoff_rad = np.radians(takeoff_deg)
    src_vel = _velocity_at_depth(src_depth, layer_depths, layer_velocities)

    # Ray parameter: p = sin(theta) / v at the source
    # For takeoff angles > 90 (downgoing), sin(180-theta) = sin(theta)
    p = np.sin(takeoff_rad) / src_vel

    going_up = sta_depth < src_depth

    if going_up:
        z_start = src_depth
        z_end = sta_depth
    else:
        z_start = src_depth
        z_end = sta_depth

    # Accumulate horizontal distance layer by layer
    horiz_dist = 0.0
    z_current = z_start

    # Build list of layer boundaries between source and station
    if going_up:
        # Traverse layers upward
        boundaries = layer_depths[(layer_depths > z_end) & (layer_depths < z_start)]
        boundaries = np.sort(boundaries)[::-1]  # descending
        targets = list(boundaries) + [z_end]

        for z_next in targets:
            v = _velocity_at_depth(
                min(z_current, z_next) + 1e-10, layer_depths, layer_velocities
            )
            dz = abs(z_current - z_next)
            sin_theta = p * v
            if abs(sin_theta) >= 1.0:
                # Total internal reflection - return large distance
                return 1e6
            cos_theta = np.sqrt(1.0 - sin_theta**2)
            if cos_theta < 1e-12:
                return 1e6
            horiz_dist += dz * sin_theta / cos_theta
            z_current = z_next
    else:
        # Traverse layers downward
        boundaries = layer_depths[(layer_depths > z_start) & (layer_depths < z_end)]
        boundaries = np.sort(boundaries)
        targets = list(boundaries) + [z_end]

        for z_next in targets:
            v = _velocity_at_depth(z_current + 1e-10, layer_depths, layer_velocities)
            dz = abs(z_next - z_current)
            sin_theta = p * v
            if abs(sin_theta) >= 1.0:
                return 1e6
            cos_theta = np.sqrt(1.0 - sin_theta**2)
            if cos_theta < 1e-12:
                return 1e6
            horiz_dist += dz * sin_theta / cos_theta
            z_current = z_next

    return horiz_dist
