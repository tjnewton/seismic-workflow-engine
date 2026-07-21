"""Local magnitude (ML) estimation via Wood-Anderson simulation.

Simulates the Wood-Anderson seismograph response and computes ML
using the Hutton & Boore (1987) attenuation relation.

References
----------
Hutton, L.K. & Boore, D.M. (1987). The ML scale in Southern
    California. Bull. Seismol. Soc. Am., 77(6), 2074-2094.
"""

from __future__ import annotations

import logging
from typing import NamedTuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from obspy import Stream
    from obspy.core.trace import Trace

logger = logging.getLogger(__name__)

__all__ = [
    "LocalMagnitudeResult",
    "wood_anderson_amplitude",
    "local_magnitude",
]


class LocalMagnitudeResult(NamedTuple):
    """Result from local magnitude estimation."""

    ml: float
    """Mean ML across stations."""
    ml_per_station: NDArray[np.floating]
    """Individual station ML values."""
    station_names: list[str]
    """Station identifiers corresponding to ml_per_station."""
    residuals: NDArray[np.floating]
    """Residuals: ``ml_per_station - ml``."""
    ml_std: float = 0.0
    """Standard deviation of valid station ML values."""
    ml_stderr: float = 0.0
    """Standard error of the mean: ``ml_std / sqrt(n_valid)``."""
    n_valid_stations: int = 0
    """Number of stations with non-NaN ML values."""
    magnitude_uncertainty: object | None = None
    """Optional :class:`~swe.uncertainty.types.MagnitudeUncertainty`."""


def wood_anderson_amplitude(
    trace: Trace,
    *,
    water_level: float = 10.0,
) -> float:
    """Simulate the Wood-Anderson seismograph and return peak amplitude.

    Removes the instrument response (if available) and applies the
    standard Wood-Anderson transfer function to obtain the displacement
    amplitude in millimetres that a WA instrument would record.

    Parameters
    ----------
    trace : obspy.Trace
        Input waveform with attached instrument response
        (``trace.stats.response``).
    water_level : float
        Water level (dB) for spectral division during response removal.

    Returns
    -------
    float
        Peak Wood-Anderson amplitude in millimetres.  Returns ``NaN``
        if the response cannot be removed.
    """
    tr = trace.copy()

    # Wood-Anderson seismograph parameters
    wa_paz = {
        "poles": [
            -6.283185 + 4.712389j,
            -6.283185 - 4.712389j,
        ],
        "zeros": [0j, 0j],
        "gain": 2800.0,
        "sensitivity": 1.0,
    }

    try:
        if hasattr(tr.stats, "response") and tr.stats.response is not None:
            tr.remove_response(
                output="DISP",
                water_level=water_level,
            )
        tr.simulate(paz_simulate=wa_paz, water_level=water_level)
    except (ValueError, TypeError, OSError) as exc:
        logger.warning(
            "Failed to simulate WA response for %s.%s.%s: %s",
            tr.stats.network,
            tr.stats.station,
            tr.stats.channel,
            exc,
        )
        return float("nan")

    # Peak amplitude in mm (ObsPy returns metres after displacement)
    peak_amp = float(np.nanmax(np.abs(tr.data))) * 1000.0
    return peak_amp


def local_magnitude(
    stream: Stream,
    distances_km: NDArray[np.floating] | list[float],
    *,
    a: float = 1.110,
    b: float = 0.00189,
    c: float = -2.09,
    water_level: float = 10.0,
    location_uncertainty: object | None = None,
) -> LocalMagnitudeResult:
    """Compute local magnitude (ML) from Wood-Anderson amplitudes.

    Uses the Hutton & Boore (1987) attenuation relation::

        ML = log10(A) + a * log10(r / 100) + b * (r - 100) + c + 3.0

    where *A* is the peak WA amplitude in mm and *r* is the
    hypocentral distance in km.  The ``+ 3.0`` normalises to the
    original Richter definition at 100 km.

    Parameters
    ----------
    stream : obspy.Stream
        Waveform data with instrument response metadata.  One ML
        value is computed per trace.
    distances_km : array-like
        Hypocentral distances (km), one per trace in *stream*.
    a : float
        Geometric spreading coefficient.
    b : float
        Anelastic attenuation coefficient.
    c : float
        Station/baseline correction.
    water_level : float
        Water level (dB) for response removal.
    location_uncertainty : LocationUncertainty, optional
        If provided, the distance-contribution to magnitude uncertainty
        is computed via :func:`~swe.uncertainty.magnitude.propagate_location_to_magnitude`
        and stored in the returned result's ``magnitude_uncertainty``
        attribute (accessible as a dict key when the result is
        converted via pipeline provenance).

    Returns
    -------
    LocalMagnitudeResult
        Mean ML, per-station values, station names, and residuals.
    """
    distances = np.asarray(distances_km, dtype=np.float64)
    if len(distances) != len(stream):
        raise ValueError(
            f"distances_km length ({len(distances)}) must match "
            f"stream length ({len(stream)})"
        )

    mls: list[float] = []
    names: list[str] = []

    for i, trace in enumerate(stream):
        amp = wood_anderson_amplitude(trace, water_level=water_level)
        dist = distances[i]

        if np.isnan(amp) or amp <= 0 or dist <= 0:
            mls.append(float("nan"))
        else:
            ml_i = (
                np.log10(amp)
                + a * np.log10(dist / 100.0)
                + b * (dist - 100.0)
                + c
                + 3.0
            )
            mls.append(float(ml_i))

        names.append(
            f"{trace.stats.network}.{trace.stats.station}.{trace.stats.channel}"
        )

    ml_arr = np.asarray(mls, dtype=np.float64)
    mean_ml = float(np.nanmean(ml_arr))
    residuals = ml_arr - mean_ml

    valid = ml_arr[~np.isnan(ml_arr)]
    n_valid = len(valid)
    if n_valid > 1:
        ml_std = float(np.std(valid, ddof=1))
        ml_stderr = ml_std / np.sqrt(n_valid)
    elif n_valid == 1:
        ml_std = 0.0
        ml_stderr = 0.0
    else:
        ml_std = 0.0
        ml_stderr = 0.0

    mag_unc = None
    if location_uncertainty is not None:
        from swe.uncertainty.magnitude import propagate_location_to_magnitude

        mag_unc = propagate_location_to_magnitude(
            ml_std=ml_std,
            ml_stderr=ml_stderr,
            n_valid_stations=n_valid,
            distances_km=distances,
            location_uncertainty=location_uncertainty,
            a=a,
            b=b,
        )

    return LocalMagnitudeResult(
        ml=mean_ml,
        ml_per_station=ml_arr,
        station_names=names,
        residuals=residuals,
        ml_std=ml_std,
        ml_stderr=ml_stderr,
        n_valid_stations=n_valid,
        magnitude_uncertainty=mag_unc,
    )
