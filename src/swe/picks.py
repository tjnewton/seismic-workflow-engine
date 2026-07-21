"""Unified pick format for the seismic workflow engine.

:class:`SWEPick` is the canonical pick representation used throughout SWE.
Converter functions translate to/from the formats used by individual modules
(``PhaseArrival``, ``dict[str, list[UTCDateTime]]``, ObsPy ``Pick``, and
``pandas.DataFrame``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

import pandas as pd
from obspy import UTCDateTime
from obspy.core.event import Pick, QuantityError, WaveformStreamID

if TYPE_CHECKING:
    from swe.picking.deep_learning import PhaseArrival

logger = logging.getLogger(__name__)

__all__ = [
    "SWEPick",
    "from_obspy_picks",
    "from_phase_arrivals",
    "picks_from_dataframe",
    "picks_to_dataframe",
    "to_obspy_picks",
    "to_station_picks",
]


class SWEPick(NamedTuple):
    """A single seismic phase pick with full metadata.

    Parameters
    ----------
    station : str
        Station code (e.g. ``"STA1"``).
    phase : str
        Phase type (``"P"``, ``"S"``, etc.).
    time : UTCDateTime
        Pick time.
    confidence : float
        Confidence in ``[0, 1]``.
    network : str
        Network code.
    channel : str
        Channel code.
    method : str
        Picking method (e.g. ``"PhaseNet"``, ``"STA/LTA"``, ``"manual"``).
    time_uncertainty_s : float
        Pick time uncertainty in seconds (1-sigma).  Zero means
        unknown/unset.
    """

    station: str
    phase: str
    time: UTCDateTime
    confidence: float
    network: str = ""
    channel: str = ""
    method: str = ""
    time_uncertainty_s: float = 0.0


def from_phase_arrivals(
    arrivals: list[PhaseArrival],
    *,
    max_time_error: float = 0.5,
) -> list[SWEPick]:
    """Convert :class:`~swe.picking.deep_learning.PhaseArrival` objects to :class:`SWEPick`.

    Parameters
    ----------
    arrivals : list of PhaseArrival
        Arrivals from :func:`~swe.picking.deep_learning.predict_phase_arrivals`.
    max_time_error : float
        Maximum time error in seconds, corresponding to ``confidence=0``.
        The derived uncertainty is ``(1 - confidence) * max_time_error``.

    Returns
    -------
    list of SWEPick
    """
    return [
        SWEPick(
            station=a.station,
            phase=a.phase,
            time=a.time,
            confidence=a.confidence,
            time_uncertainty_s=(1.0 - a.confidence) * max_time_error,
        )
        for a in arrivals
    ]


def to_station_picks(
    picks: list[SWEPick],
) -> tuple[dict[str, list], dict[str, list[str]]]:
    """Convert :class:`SWEPick` list to the dict format used by association.

    Parameters
    ----------
    picks : list of SWEPick
        Picks to convert.

    Returns
    -------
    station_picks : dict
        ``{station: [UTCDateTime, ...]}`` sorted by time.
    pick_phases : dict
        ``{station: ["P", "S", ...]}`` aligned with *station_picks*.
    """
    station_picks: dict[str, list] = {}
    pick_phases: dict[str, list[str]] = {}

    for p in picks:
        station_picks.setdefault(p.station, []).append(p.time)
        pick_phases.setdefault(p.station, []).append(p.phase)

    # Sort each station's picks by time, keeping phases aligned
    for sta in station_picks:
        pairs = sorted(zip(station_picks[sta], pick_phases[sta], strict=True))
        station_picks[sta] = [t for t, _ in pairs]
        pick_phases[sta] = [ph for _, ph in pairs]

    return station_picks, pick_phases


def from_obspy_picks(obspy_picks: list[Pick]) -> list[SWEPick]:
    """Convert ObsPy :class:`~obspy.core.event.Pick` objects to :class:`SWEPick`.

    Parameters
    ----------
    obspy_picks : list of obspy.core.event.Pick
        Picks from an ObsPy :class:`~obspy.core.event.Catalog`.

    Returns
    -------
    list of SWEPick
    """
    result: list[SWEPick] = []
    for op in obspy_picks:
        wid = op.waveform_id
        station = wid.station_code if wid else ""
        network = wid.network_code if wid else ""
        channel = wid.channel_code if wid else ""
        phase = op.phase_hint or "P"
        time = op.time if op.time is not None else UTCDateTime(0)
        confidence = 1.0
        time_uncertainty_s = 0.0
        if (
            op.time_errors is not None
            and op.time_errors.uncertainty is not None
        ):
            time_uncertainty_s = float(op.time_errors.uncertainty)
            confidence = max(0.0, min(1.0, 1.0 - time_uncertainty_s))

        result.append(
            SWEPick(
                station=station,
                phase=phase,
                time=time,
                confidence=confidence,
                network=network,
                channel=channel,
                time_uncertainty_s=time_uncertainty_s,
            )
        )
    return result


def to_obspy_picks(
    picks: list[SWEPick],
    *,
    network: str = "",
    map_confidence_to_time_error: bool = False,
    max_time_error: float = 1.0,
) -> list[Pick]:
    """Convert :class:`SWEPick` objects to ObsPy :class:`~obspy.core.event.Pick`.

    Parameters
    ----------
    picks : list of SWEPick
        Picks to convert.
    network : str
        Fallback network code when the pick has no network set.
    map_confidence_to_time_error : bool
        If *True*, set ``time_errors`` on each ObsPy Pick using
        ``(1 - confidence) * max_time_error``.
    max_time_error : float
        Maximum time error in seconds, corresponding to ``confidence=0``.

    Returns
    -------
    list of obspy.core.event.Pick
    """
    result: list[Pick] = []
    for p in picks:
        net = p.network or network
        pick = Pick(
            time=p.time,
            waveform_id=WaveformStreamID(
                network_code=net,
                station_code=p.station,
                channel_code=p.channel,
            ),
            phase_hint=p.phase,
        )
        if p.time_uncertainty_s > 0:
            pick.time_errors = QuantityError(
                uncertainty=p.time_uncertainty_s
            )
        elif map_confidence_to_time_error:
            pick.time_errors = QuantityError(
                uncertainty=(1.0 - p.confidence) * max_time_error
            )
        result.append(pick)
    return result


def picks_to_dataframe(picks: list[SWEPick]) -> pd.DataFrame:
    """Convert :class:`SWEPick` list to a :class:`~pandas.DataFrame`.

    Parameters
    ----------
    picks : list of SWEPick
        Picks to convert.

    Returns
    -------
    pandas.DataFrame
        Columns: ``station``, ``phase``, ``time``, ``confidence``,
        ``network``, ``channel``, ``method``, ``time_uncertainty_s``.
    """
    rows = [
        {
            "station": p.station,
            "phase": p.phase,
            "time": p.time.datetime,
            "timestamp": float(p.time),
            "confidence": p.confidence,
            "network": p.network,
            "channel": p.channel,
            "method": p.method,
            "time_uncertainty_s": p.time_uncertainty_s,
        }
        for p in picks
    ]
    return pd.DataFrame(rows)


def picks_from_dataframe(df: pd.DataFrame) -> list[SWEPick]:
    """Create :class:`SWEPick` list from a :class:`~pandas.DataFrame`.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns ``station``, ``phase``, ``time``.  Optional
        columns: ``confidence``, ``network``, ``channel``, ``method``.

    Returns
    -------
    list of SWEPick
    """
    result: list[SWEPick] = []
    for _, row in df.iterrows():
        t = row["time"]
        if not isinstance(t, UTCDateTime):
            t = UTCDateTime(t)
        result.append(
            SWEPick(
                station=str(row["station"]),
                phase=str(row["phase"]),
                time=t,
                confidence=float(row.get("confidence", 1.0)),
                network=str(row.get("network", "")),
                channel=str(row.get("channel", "")),
                method=str(row.get("method", "")),
                time_uncertainty_s=float(row.get("time_uncertainty_s", 0.0)),
            )
        )
    return result
