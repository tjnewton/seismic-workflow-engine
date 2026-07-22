"""Radiation patterns for a general moment tensor.

Computes P-, SV-, and SH-wave far-field radiation patterns for an
arbitrary symmetric moment tensor.  The double-couple special case
reproduces the same results as :func:`swe.focal.solve._radiation_pattern`.

References
----------
Aki, K. & Richards, P.G. (2002). Quantitative Seismology (2nd ed.).
    University Science Books. Eq. 4.29.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from swe.moment_tensor.tensor import MomentTensor, to_matrix

logger = logging.getLogger(__name__)

__all__ = [
    "radiation_pattern_p",
    "radiation_pattern_sh",
    "radiation_pattern_sv",
]


def _direction_cosines(
    azimuth_rad: NDArray[np.floating],
    takeoff_rad: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Compute ray direction cosines in NED coordinates.

    Uses the same takeoff angle convention as :mod:`swe.focal.takeoff`
    and :func:`swe.focal.solve._radiation_pattern`: ``0=up, π=down``.

    Parameters
    ----------
    azimuth_rad : NDArray
        Azimuths in radians (clockwise from North).
    takeoff_rad : NDArray
        Takeoff angles in radians (0=up, π=down).

    Returns
    -------
    gamma_x, gamma_y, gamma_z : NDArray
        Direction cosines (North, East, Down).
    """
    sin_t = np.sin(takeoff_rad)
    cos_t = np.cos(takeoff_rad)
    sin_a = np.sin(azimuth_rad)
    cos_a = np.cos(azimuth_rad)

    # NED convention: x=North, y=East, z=Down
    # takeoff=0 -> straight up -> gamma_z = cos(0) = +1
    # takeoff=π -> straight down -> gamma_z = cos(π) = -1
    # This is consistent with the M tensor from strike_dip_rake_to_tensor
    # and with Aki & Richards eq. 4.88 in focal.solve._radiation_pattern.
    gamma_x = sin_t * cos_a  # North component
    gamma_y = sin_t * sin_a  # East component
    gamma_z = cos_t  # Down component

    return gamma_x, gamma_y, gamma_z


def radiation_pattern_p(
    mt: MomentTensor,
    azimuth: NDArray[np.floating],
    takeoff: NDArray[np.floating],
) -> NDArray[np.floating]:
    r"""Compute P-wave radiation pattern for a general moment tensor.

    .. math::
        A_P(\hat{\gamma}) = \sum_{ij} M_{ij}\, \gamma_i\, \gamma_j

    (Aki & Richards eq. 4.29, far-field P displacement.)

    Parameters
    ----------
    mt : MomentTensor
        Moment tensor (6 components).
    azimuth : NDArray
        Source-to-station azimuths in degrees.
    takeoff : NDArray
        Takeoff angles in degrees (0=up, 180=down).

    Returns
    -------
    NDArray
        P-wave radiation amplitude at each station.
        Positive = compressional, negative = dilatational.
    """
    azimuth = np.asarray(azimuth, dtype=np.float64)
    takeoff = np.asarray(takeoff, dtype=np.float64)

    az_rad = np.radians(azimuth)
    to_rad = np.radians(takeoff)

    gx, gy, gz = _direction_cosines(az_rad, to_rad)
    m = to_matrix(mt)

    # A_P = gamma_i M_ij gamma_j
    gamma = np.stack([gx, gy, gz], axis=-1)  # (..., 3)
    rad = np.einsum("...i,ij,...j->...", gamma, m, gamma)

    return np.asarray(rad)


def radiation_pattern_sv(
    mt: MomentTensor,
    azimuth: NDArray[np.floating],
    takeoff: NDArray[np.floating],
) -> NDArray[np.floating]:
    r"""Compute SV-wave radiation pattern for a general moment tensor.

    The SV direction lies in the ray-vertical plane, perpendicular to
    the ray direction.

    Parameters
    ----------
    mt : MomentTensor
        Moment tensor.
    azimuth : NDArray
        Source-to-station azimuths in degrees.
    takeoff : NDArray
        Takeoff angles in degrees (0=up, 180=down).

    Returns
    -------
    NDArray
        SV-wave radiation amplitude.
    """
    azimuth = np.asarray(azimuth, dtype=np.float64)
    takeoff = np.asarray(takeoff, dtype=np.float64)

    az_rad = np.radians(azimuth)
    to_rad = np.radians(takeoff)

    gx, gy, gz = _direction_cosines(az_rad, to_rad)

    # SV unit vector: in the vertical plane containing the ray,
    # perpendicular to the ray.  Obtained by rotating the ray direction
    # 90° toward the horizontal in the ray-vertical plane.
    cos_t = np.cos(to_rad)
    sin_t = np.sin(to_rad)
    cos_a = np.cos(az_rad)
    sin_a = np.sin(az_rad)

    sv_x = cos_t * cos_a
    sv_y = cos_t * sin_a
    sv_z = -sin_t

    m = to_matrix(mt)
    gamma = np.stack([gx, gy, gz], axis=-1)
    sv = np.stack([sv_x, sv_y, sv_z], axis=-1)

    # A_SV = sv_i M_ij gamma_j
    rad = np.einsum("...i,ij,...j->...", sv, m, gamma)

    return np.asarray(rad)


def radiation_pattern_sh(
    mt: MomentTensor,
    azimuth: NDArray[np.floating],
    takeoff: NDArray[np.floating],
) -> NDArray[np.floating]:
    r"""Compute SH-wave radiation pattern for a general moment tensor.

    The SH direction is perpendicular to both the ray direction and the
    SV direction (horizontal, transverse to the ray).

    Parameters
    ----------
    mt : MomentTensor
        Moment tensor.
    azimuth : NDArray
        Source-to-station azimuths in degrees.
    takeoff : NDArray
        Takeoff angles in degrees (0=up, 180=down).

    Returns
    -------
    NDArray
        SH-wave radiation amplitude.
    """
    azimuth = np.asarray(azimuth, dtype=np.float64)
    takeoff = np.asarray(takeoff, dtype=np.float64)

    az_rad = np.radians(azimuth)

    gx, gy, gz = _direction_cosines(np.radians(azimuth), np.radians(takeoff))

    # SH unit vector: horizontal, perpendicular to the azimuth
    # (90° clockwise from the horizontal projection of the ray)
    sin_a = np.sin(az_rad)
    cos_a = np.cos(az_rad)

    sh_x = -sin_a  # perpendicular to azimuth in horizontal plane
    sh_y = cos_a
    sh_z = np.zeros_like(az_rad)

    m = to_matrix(mt)
    gamma = np.stack([gx, gy, gz], axis=-1)
    sh = np.stack([sh_x, sh_y, sh_z], axis=-1)

    # A_SH = sh_i M_ij gamma_j
    rad = np.einsum("...i,ij,...j->...", sh, m, gamma)

    return np.asarray(rad)
