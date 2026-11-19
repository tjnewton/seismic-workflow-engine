"""Core uncertainty data structures for seismic workflow propagation.

These types have **zero imports** from other SWE modules, avoiding
circular dependencies.  They are frozen dataclasses with all-optional
fields so they compose cleanly through the pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

__all__ = [
    "LocationUncertainty",
    "MagnitudeUncertainty",
    "UncertainValue",
]


@dataclass(frozen=True)
class UncertainValue:
    """A scalar value with associated uncertainty metadata.

    Parameters
    ----------
    value : float
        Central or best-estimate value.
    uncertainty : float
        Symmetric 1-sigma uncertainty.
    confidence_level : float
        Confidence level of the uncertainty interval (default 0.68
        for 1-sigma).
    lower_bound : float or None
        Lower bound of the uncertainty interval.
    upper_bound : float or None
        Upper bound of the uncertainty interval.
    n_samples : int
        Number of samples used to estimate the uncertainty.
    method : str
        Method used to compute the uncertainty (e.g. ``"monte_carlo"``,
        ``"analytical"``, ``"bootstrap"``).
    """

    value: float
    uncertainty: float = 0.0
    confidence_level: float = 0.68
    lower_bound: float | None = None
    upper_bound: float | None = None
    n_samples: int = 0
    method: str = ""


@dataclass(frozen=True)
class LocationUncertainty:
    """Uncertainty envelope for a hypocenter location.

    Parameters
    ----------
    horiz_uncertainty_m : float
        Horizontal uncertainty (meters), typically the semi-major axis
        of the 68 % confidence ellipse.
    depth_uncertainty_m : float
        Vertical (depth) uncertainty in meters.
    rms : float
        RMS travel-time residual (seconds).
    covariance_matrix : ndarray or None
        3x3 covariance matrix ``[[Cxx, Cxy, Cxz], ...]`` in meters^2.
    n_samples : int
        Number of Monte Carlo realizations or scatter points.
    method : str
        Method used (e.g. ``"monte_carlo"``, ``"nll_scatter"``).
    """

    horiz_uncertainty_m: float = 0.0
    depth_uncertainty_m: float = 0.0
    rms: float = 0.0
    covariance_matrix: NDArray[np.floating] | None = field(default=None, repr=False)
    n_samples: int = 0
    method: str = ""

    class Config:
        arbitrary_types_allowed = True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LocationUncertainty):
            return NotImplemented
        base = (
            self.horiz_uncertainty_m == other.horiz_uncertainty_m
            and self.depth_uncertainty_m == other.depth_uncertainty_m
            and self.rms == other.rms
            and self.n_samples == other.n_samples
            and self.method == other.method
        )
        if not base:
            return False
        if self.covariance_matrix is None and other.covariance_matrix is None:
            return True
        if self.covariance_matrix is None or other.covariance_matrix is None:
            return False
        return bool(np.array_equal(self.covariance_matrix, other.covariance_matrix))

    def __hash__(self) -> int:
        cov_hash = (
            tuple(self.covariance_matrix.flat) if self.covariance_matrix is not None else None
        )
        return hash(
            (
                self.horiz_uncertainty_m,
                self.depth_uncertainty_m,
                self.rms,
                cov_hash,
                self.n_samples,
                self.method,
            )
        )


@dataclass(frozen=True)
class MagnitudeUncertainty:
    """Combined uncertainty for a magnitude estimate.

    Parameters
    ----------
    mag_uncertainty : float
        Total magnitude uncertainty (combined in quadrature).
    station_std : float
        Inter-station standard deviation.
    station_stderr : float
        Standard error of the mean across stations.
    distance_contribution : float
        Magnitude uncertainty contribution from hypocentral distance
        uncertainty.
    n_valid_stations : int
        Number of stations with valid magnitude estimates.
    method : str
        Method used (e.g. ``"analytical"``, ``"bootstrap"``).
    """

    mag_uncertainty: float = 0.0
    station_std: float = 0.0
    station_stderr: float = 0.0
    distance_contribution: float = 0.0
    n_valid_stations: int = 0
    method: str = ""
