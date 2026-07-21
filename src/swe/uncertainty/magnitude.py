"""Analytical uncertainty propagation from location to magnitude.

Uses the Hutton & Boore (1987) ML formula to derive partial derivatives
of magnitude with respect to hypocentral distance, then combines
distance-induced uncertainty with inter-station scatter in quadrature.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from swe.uncertainty.types import LocationUncertainty, MagnitudeUncertainty

logger = logging.getLogger(__name__)

__all__ = ["propagate_location_to_magnitude"]


def propagate_location_to_magnitude(
    ml_std: float,
    ml_stderr: float,
    n_valid_stations: int,
    distances_km: NDArray[np.floating],
    location_uncertainty: LocationUncertainty,
    *,
    a: float = 1.110,
    b: float = 0.00189,
) -> MagnitudeUncertainty:
    """Propagate location uncertainty into magnitude uncertainty.

    The Hutton & Boore (1987) ML formula is::

        ML = log10(A) + a * log10(r/100) + b * (r - 100) + c + 3.0

    The partial derivative with respect to distance *r* (km) is::

        dML/dr = a / (r * ln(10)) + b

    This gives the magnitude sensitivity to distance changes.  The
    distance uncertainty is derived from the location uncertainty
    (horizontal component), and the magnitude contribution is the
    RMS of ``dML/dr * sigma_r`` across stations, combined in
    quadrature with the inter-station scatter.

    Parameters
    ----------
    ml_std : float
        Standard deviation of per-station ML values.
    ml_stderr : float
        Standard error of the mean ML.
    n_valid_stations : int
        Number of stations with valid ML values.
    distances_km : ndarray
        Hypocentral distances (km), one per station.
    location_uncertainty : LocationUncertainty
        Location uncertainty from the prior pipeline step.
    a : float
        Geometric spreading coefficient (Hutton & Boore).
    b : float
        Anelastic attenuation coefficient (Hutton & Boore).

    Returns
    -------
    MagnitudeUncertainty
        Combined magnitude uncertainty.
    """
    distances_km = np.asarray(distances_km, dtype=np.float64)

    # Distance uncertainty in km (from horizontal location uncertainty)
    sigma_r_km = location_uncertainty.horiz_uncertainty_m / 1000.0

    # Partial derivative dML/dr at each station distance
    valid_mask = distances_km > 0
    dml_dr = np.zeros_like(distances_km)
    dml_dr[valid_mask] = (
        a / (distances_km[valid_mask] * np.log(10.0)) + b
    )

    # Per-station magnitude uncertainty from distance error
    station_mag_unc = np.abs(dml_dr) * sigma_r_km

    # RMS distance contribution across valid stations
    if np.any(valid_mask):
        distance_contribution = float(
            np.sqrt(np.mean(station_mag_unc[valid_mask] ** 2))
        )
    else:
        distance_contribution = 0.0

    # Combine in quadrature: station scatter + distance contribution
    mag_uncertainty = float(
        np.sqrt(ml_stderr**2 + distance_contribution**2)
    )

    return MagnitudeUncertainty(
        mag_uncertainty=mag_uncertainty,
        station_std=ml_std,
        station_stderr=ml_stderr,
        distance_contribution=distance_contribution,
        n_valid_stations=n_valid_stations,
        method="analytical",
    )
