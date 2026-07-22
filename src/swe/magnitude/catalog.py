"""Magnitude-to-catalog integration for ObsPy Event objects.

Bridges the gap between SWE's magnitude estimation functions and
ObsPy's Catalog/Event data model by computing magnitudes and
attaching them as proper ObsPy Magnitude and StationMagnitude objects.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from obspy.core.event import (
    Magnitude,
    QuantityError,
    StationMagnitude,
    StationMagnitudeContribution,
)

from swe.magnitude.duration import duration_magnitude
from swe.magnitude.local import local_magnitude

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from obspy import Catalog, Stream
    from obspy.core.event import Event
    from obspy.core.utcdatetime import UTCDateTime

logger = logging.getLogger(__name__)

__all__ = [
    "attach_duration_magnitudes",
    "attach_local_magnitudes",
    "attach_magnitudes_to_catalog",
]


def attach_local_magnitudes(
    event: Event,
    stream: Stream,
    distances_km: NDArray[np.floating] | list[float],
    *,
    a: float = 1.110,
    b: float = 0.00189,
    c: float = -2.09,
    water_level: float = 10.0,
) -> Event:
    """Compute local magnitudes and attach to an ObsPy Event.

    Calls :func:`swe.magnitude.local.local_magnitude` internally and
    wraps the results in ObsPy ``Magnitude`` and ``StationMagnitude``
    objects, appending them to the event.

    Parameters
    ----------
    event : obspy.core.event.Event
        Event to attach magnitudes to (modified in place).
    stream : obspy.Stream
        Waveform data with instrument response metadata.
    distances_km : array-like
        Hypocentral distances (km), one per trace in *stream*.
    a, b, c : float
        Hutton & Boore attenuation relation coefficients.
    water_level : float
        Water level (dB) for response removal.

    Returns
    -------
    Event
        The same event with magnitudes appended.
    """
    result = local_magnitude(stream, distances_km, a=a, b=b, c=c, water_level=water_level)

    if np.isnan(result.ml):
        logger.warning("ML computation returned NaN, skipping attachment")
        return event

    # Create station magnitudes
    contributions: list[StationMagnitudeContribution] = []
    for i, name in enumerate(result.station_names):
        ml_i = result.ml_per_station[i]
        if np.isnan(ml_i):
            continue

        sta_mag = StationMagnitude(
            mag=float(ml_i),
            station_magnitude_type="ML",
        )
        sta_mag.waveform_id = _make_waveform_id(name)
        event.station_magnitudes.append(sta_mag)

        contrib = StationMagnitudeContribution(
            station_magnitude_id=sta_mag.resource_id,
            residual=float(result.residuals[i]),
        )
        contributions.append(contrib)

    # Create network magnitude
    mag = Magnitude(
        mag=result.ml,
        magnitude_type="ML",
        station_count=sum(1 for m in result.ml_per_station if not np.isnan(m)),
        station_magnitude_contributions=contributions,
    )
    if result.magnitude_uncertainty is not None:
        mag.mag_errors = QuantityError(uncertainty=result.magnitude_uncertainty.mag_uncertainty)
    elif result.ml_stderr > 0:
        mag.mag_errors = QuantityError(uncertainty=result.ml_stderr)
    event.magnitudes.append(mag)
    event.preferred_magnitude_id = mag.resource_id

    logger.info("Attached ML=%.2f to event %s", result.ml, event.resource_id)
    return event


def attach_duration_magnitudes(
    event: Event,
    stream: Stream,
    p_times: dict[str, UTCDateTime],
    *,
    a0: float = -0.87,
    a1: float = 2.00,
    a2: float = 0.0035,
    distances_km: dict[str, float] | None = None,
    noise_window: float = 5.0,
    snr_threshold: float = 2.0,
    smooth_window: float = 1.0,
) -> Event:
    """Compute duration magnitudes and attach to an ObsPy Event.

    Calls :func:`swe.magnitude.duration.duration_magnitude` for each
    trace and wraps results in ObsPy ``Magnitude`` and
    ``StationMagnitude`` objects.

    Parameters
    ----------
    event : obspy.core.event.Event
        Event to attach magnitudes to (modified in place).
    stream : obspy.Stream
        Waveform data.
    p_times : dict
        Mapping of station ID (``network.station.channel`` or
        ``station``) to P-wave arrival time.
    a0, a1, a2 : float
        Duration magnitude coefficients.
    distances_km : dict, optional
        Mapping of station ID to epicentral distance in km.
        If not provided, ``a2 * distance`` term is zero.
    noise_window, snr_threshold, smooth_window : float
        Parameters passed to :func:`duration_magnitude`.

    Returns
    -------
    Event
        The same event with magnitudes appended.
    """
    station_mds: list[float] = []
    contributions: list[StationMagnitudeContribution] = []

    for trace in stream:
        station_id = f"{trace.stats.network}.{trace.stats.station}.{trace.stats.channel}"

        # Look up P-time
        p_time = p_times.get(station_id) or p_times.get(trace.stats.station)
        if p_time is None:
            continue

        dist = 0.0
        if distances_km is not None:
            dist = distances_km.get(station_id, distances_km.get(trace.stats.station, 0.0))

        result = duration_magnitude(
            trace,
            p_time,
            a0=a0,
            a1=a1,
            a2=a2,
            distance_km=dist,
            noise_window=noise_window,
            snr_threshold=snr_threshold,
            smooth_window=smooth_window,
        )

        if np.isnan(result.md):
            continue

        sta_mag = StationMagnitude(
            mag=float(result.md),
            station_magnitude_type="Md",
        )
        sta_mag.waveform_id = _make_waveform_id(station_id)
        event.station_magnitudes.append(sta_mag)

        contrib = StationMagnitudeContribution(
            station_magnitude_id=sta_mag.resource_id,
        )
        contributions.append(contrib)
        station_mds.append(result.md)

    if not station_mds:
        logger.warning("No valid Md values computed, skipping attachment")
        return event

    mean_md = float(np.mean(station_mds))
    n_valid = len(station_mds)

    mag = Magnitude(
        mag=mean_md,
        magnitude_type="Md",
        station_count=n_valid,
        station_magnitude_contributions=contributions,
    )
    if n_valid > 1:
        md_std = float(np.std(station_mds, ddof=1))
        md_stderr = md_std / np.sqrt(n_valid)
        mag.mag_errors = QuantityError(uncertainty=md_stderr)
    event.magnitudes.append(mag)
    event.preferred_magnitude_id = mag.resource_id

    logger.info("Attached Md=%.2f to event %s", mean_md, event.resource_id)
    return event


def attach_magnitudes_to_catalog(
    catalog: Catalog,
    streams: list[Stream],
    distances: list[NDArray[np.floating] | list[float]],
    *,
    magnitude_type: str = "ML",
    **kwargs,
) -> Catalog:
    """Compute and attach magnitudes to all events in a catalog.

    Parameters
    ----------
    catalog : obspy.Catalog
        Catalog of events (modified in place).
    streams : list of Stream
        Waveform data, one per event.
    distances : list of array-like
        Hypocentral distances (km), one array per event.
    magnitude_type : str
        ``"ML"`` for local magnitude, ``"Md"`` for duration magnitude.
    **kwargs
        Additional keyword arguments passed to the magnitude function.

    Returns
    -------
    Catalog
        The same catalog with magnitudes attached.

    Raises
    ------
    ValueError
        If lengths of catalog, streams, and distances don't match.
    """
    if len(catalog) != len(streams) or len(catalog) != len(distances):
        raise ValueError(
            f"Lengths must match: catalog={len(catalog)}, "
            f"streams={len(streams)}, distances={len(distances)}"
        )

    for i, event in enumerate(catalog):
        if magnitude_type.upper() == "ML":
            attach_local_magnitudes(event, streams[i], distances[i], **kwargs)
        else:
            logger.warning(
                "magnitude_type='%s' not supported in batch mode; "
                "use attach_duration_magnitudes directly",
                magnitude_type,
            )

    return catalog


def _make_waveform_id(station_id: str):
    """Create a WaveformStreamID from a station identifier string.

    Parameters
    ----------
    station_id : str
        Station identifier, either ``network.station.channel`` or
        ``station`` format.

    Returns
    -------
    WaveformStreamID
    """
    from obspy.core.event import WaveformStreamID

    parts = station_id.split(".")
    if len(parts) >= 3:
        return WaveformStreamID(
            network_code=parts[0],
            station_code=parts[1],
            channel_code=parts[2],
        )
    return WaveformStreamID(station_code=parts[0])
