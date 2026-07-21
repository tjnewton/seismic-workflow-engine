"""Conversions between SWE uncertainty types and ObsPy/dict formats."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from obspy.core.event import QuantityError

from swe.uncertainty.types import (
    LocationUncertainty,
    MagnitudeUncertainty,
    UncertainValue,
)

logger = logging.getLogger(__name__)

__all__ = [
    "location_uncertainty_to_dict",
    "location_uncertainty_from_dict",
    "magnitude_uncertainty_to_dict",
    "magnitude_uncertainty_from_dict",
    "uncertain_value_to_dict",
    "uncertain_value_from_dict",
    "location_uncertainty_to_quantity_errors",
    "quantity_error_to_uncertain_value",
]


# ---------------------------------------------------------------------------
# UncertainValue
# ---------------------------------------------------------------------------

def uncertain_value_to_dict(uv: UncertainValue) -> dict[str, Any]:
    """Convert an :class:`UncertainValue` to a JSON-safe dictionary.

    Parameters
    ----------
    uv : UncertainValue
        Value to convert.

    Returns
    -------
    dict
    """
    return {
        "value": uv.value,
        "uncertainty": uv.uncertainty,
        "confidence_level": uv.confidence_level,
        "lower_bound": uv.lower_bound,
        "upper_bound": uv.upper_bound,
        "n_samples": uv.n_samples,
        "method": uv.method,
    }


def uncertain_value_from_dict(d: dict[str, Any]) -> UncertainValue:
    """Reconstruct an :class:`UncertainValue` from a dictionary.

    Parameters
    ----------
    d : dict
        Dictionary previously produced by :func:`uncertain_value_to_dict`.

    Returns
    -------
    UncertainValue
    """
    return UncertainValue(
        value=float(d["value"]),
        uncertainty=float(d.get("uncertainty", 0.0)),
        confidence_level=float(d.get("confidence_level", 0.68)),
        lower_bound=d.get("lower_bound"),
        upper_bound=d.get("upper_bound"),
        n_samples=int(d.get("n_samples", 0)),
        method=str(d.get("method", "")),
    )


# ---------------------------------------------------------------------------
# LocationUncertainty
# ---------------------------------------------------------------------------

def location_uncertainty_to_dict(lu: LocationUncertainty) -> dict[str, Any]:
    """Convert a :class:`LocationUncertainty` to a JSON-safe dictionary.

    Parameters
    ----------
    lu : LocationUncertainty
        Uncertainty to convert.

    Returns
    -------
    dict
    """
    d: dict[str, Any] = {
        "horiz_uncertainty_m": lu.horiz_uncertainty_m,
        "depth_uncertainty_m": lu.depth_uncertainty_m,
        "rms": lu.rms,
        "n_samples": lu.n_samples,
        "method": lu.method,
    }
    if lu.covariance_matrix is not None:
        d["covariance_matrix"] = lu.covariance_matrix.tolist()
    else:
        d["covariance_matrix"] = None
    return d


def location_uncertainty_from_dict(d: dict[str, Any]) -> LocationUncertainty:
    """Reconstruct a :class:`LocationUncertainty` from a dictionary.

    Parameters
    ----------
    d : dict
        Dictionary previously produced by :func:`location_uncertainty_to_dict`.

    Returns
    -------
    LocationUncertainty
    """
    cov = d.get("covariance_matrix")
    if cov is not None:
        cov = np.asarray(cov, dtype=np.float64)
    return LocationUncertainty(
        horiz_uncertainty_m=float(d.get("horiz_uncertainty_m", 0.0)),
        depth_uncertainty_m=float(d.get("depth_uncertainty_m", 0.0)),
        rms=float(d.get("rms", 0.0)),
        covariance_matrix=cov,
        n_samples=int(d.get("n_samples", 0)),
        method=str(d.get("method", "")),
    )


# ---------------------------------------------------------------------------
# MagnitudeUncertainty
# ---------------------------------------------------------------------------

def magnitude_uncertainty_to_dict(mu: MagnitudeUncertainty) -> dict[str, Any]:
    """Convert a :class:`MagnitudeUncertainty` to a JSON-safe dictionary.

    Parameters
    ----------
    mu : MagnitudeUncertainty
        Uncertainty to convert.

    Returns
    -------
    dict
    """
    return {
        "mag_uncertainty": mu.mag_uncertainty,
        "station_std": mu.station_std,
        "station_stderr": mu.station_stderr,
        "distance_contribution": mu.distance_contribution,
        "n_valid_stations": mu.n_valid_stations,
        "method": mu.method,
    }


def magnitude_uncertainty_from_dict(d: dict[str, Any]) -> MagnitudeUncertainty:
    """Reconstruct a :class:`MagnitudeUncertainty` from a dictionary.

    Parameters
    ----------
    d : dict
        Dictionary previously produced by :func:`magnitude_uncertainty_to_dict`.

    Returns
    -------
    MagnitudeUncertainty
    """
    return MagnitudeUncertainty(
        mag_uncertainty=float(d.get("mag_uncertainty", 0.0)),
        station_std=float(d.get("station_std", 0.0)),
        station_stderr=float(d.get("station_stderr", 0.0)),
        distance_contribution=float(d.get("distance_contribution", 0.0)),
        n_valid_stations=int(d.get("n_valid_stations", 0)),
        method=str(d.get("method", "")),
    )


# ---------------------------------------------------------------------------
# ObsPy interop
# ---------------------------------------------------------------------------

def location_uncertainty_to_quantity_errors(
    lu: LocationUncertainty,
) -> dict[str, QuantityError]:
    """Convert a :class:`LocationUncertainty` to ObsPy QuantityError objects.

    Parameters
    ----------
    lu : LocationUncertainty
        Location uncertainty.

    Returns
    -------
    dict
        Keys: ``"depth_errors"``, ``"latitude_errors"``,
        ``"longitude_errors"``.
    """
    meters_per_deg = 111_320.0
    horiz_deg = lu.horiz_uncertainty_m / meters_per_deg
    return {
        "depth_errors": QuantityError(uncertainty=lu.depth_uncertainty_m),
        "latitude_errors": QuantityError(uncertainty=horiz_deg),
        "longitude_errors": QuantityError(uncertainty=horiz_deg),
    }


def quantity_error_to_uncertain_value(
    qe: QuantityError,
    *,
    value: float = 0.0,
) -> UncertainValue:
    """Convert an ObsPy :class:`QuantityError` to :class:`UncertainValue`.

    Parameters
    ----------
    qe : QuantityError
        ObsPy quantity error.
    value : float
        Central value to associate with the uncertainty.

    Returns
    -------
    UncertainValue
    """
    unc = qe.uncertainty if qe.uncertainty is not None else 0.0
    lb = qe.lower_uncertainty if hasattr(qe, "lower_uncertainty") else None
    ub = qe.upper_uncertainty if hasattr(qe, "upper_uncertainty") else None
    return UncertainValue(
        value=value,
        uncertainty=float(unc),
        lower_bound=float(lb) if lb is not None else None,
        upper_bound=float(ub) if ub is not None else None,
    )
