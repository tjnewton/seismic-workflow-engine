"""Converters between association results and ObsPy catalog objects."""

from __future__ import annotations

import logging

import numpy as np
from matplotlib.dates import num2date
from obspy import UTCDateTime
from obspy.core.event import (
    Catalog,
    Event,
    Origin,
    Pick,
    WaveformStreamID,
)

from swe.association.clustering import AssociationResult, get_cluster_stats
from swe.picks import SWEPick

logger = logging.getLogger(__name__)


def association_to_catalog(
    result: AssociationResult,
    *,
    network: str = "",
    channel: str = "",
    phase_hint: str = "P",
    default_latitude: float = 0.0,
    default_longitude: float = 0.0,
    default_depth: float = 8000.0,
    swe_picks: list[SWEPick] | None = None,
) -> Catalog:
    """Convert an :class:`AssociationResult` into an ObsPy :class:`Catalog`.

    Each non-noise cluster becomes an :class:`Event` with an :class:`Origin`
    placed at the cluster median time (using default coordinates) and one
    :class:`Pick` per associated pick.

    Parameters
    ----------
    result : AssociationResult
        Output from :func:`~swe.association.associate_picks`.
    network : str
        Network code written into each :class:`WaveformStreamID`.
    channel : str
        Channel code written into each :class:`WaveformStreamID`.
    phase_hint : str
        Phase label for every pick (e.g. ``"P"``).
    default_latitude : float
        Latitude assigned to each :class:`Origin`.
    default_longitude : float
        Longitude assigned to each :class:`Origin`.
    default_depth : float
        Depth in metres assigned to each :class:`Origin`.
    swe_picks : list of SWEPick, optional
        If provided, per-pick phase hints and confidence are preserved in
        the output catalog instead of using the global *phase_hint*.

    Returns
    -------
    obspy.core.event.Catalog
        One event per non-noise cluster.
    """
    unique_labels, cluster_medians = get_cluster_stats(
        result.labels, result.pick_times
    )

    pick_times = np.array(result.pick_times)
    pick_station_order = np.array(result.pick_station_order)

    # Build a lookup table from swe_picks for per-pick phase/confidence
    _pick_lookup: dict[tuple[str, float], SWEPick] = {}
    if swe_picks is not None:
        for sp in swe_picks:
            _pick_lookup[(sp.station, float(sp.time))] = sp

    events: list[Event] = []
    for label in sorted(unique_labels):
        if label == -1:
            continue

        origin_time = UTCDateTime(num2date(cluster_medians[label]))
        origin = Origin(
            time=origin_time,
            latitude=default_latitude,
            longitude=default_longitude,
            depth=default_depth,
        )

        member_indices = np.where(result.labels == label)[0]
        event_picks: list[Pick] = []
        for idx in member_indices:
            sta_idx = pick_station_order[idx]
            sta_name = result.reverse_station_distance[sta_idx]
            pick_time = UTCDateTime(num2date(pick_times[idx]))

            # Try to find matching SWEPick for per-pick metadata
            matched = _pick_lookup.get((sta_name, float(pick_time)))
            hint = matched.phase if matched else phase_hint
            net = matched.network if matched and matched.network else network
            cha = matched.channel if matched and matched.channel else channel

            event_picks.append(
                Pick(
                    time=pick_time,
                    waveform_id=WaveformStreamID(
                        network_code=net,
                        station_code=sta_name,
                        channel_code=cha,
                    ),
                    phase_hint=hint,
                )
            )

        event = Event(
            origins=[origin],
            picks=event_picks,
            preferred_origin_id=origin.resource_id,
        )
        events.append(event)

    return Catalog(events=events)
