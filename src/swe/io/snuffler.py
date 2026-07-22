"""Snuffler marker file (.mrkr) parsing utilities.

The Snuffler marker format is a plain-text format produced by
`Pyrocko/Snuffler <https://pyrocko.org/>`_ for manual and automatic
phase picks, event markers, and other annotations.

Snuffler phase marker lines have the format::

    phase: YYYY-MM-DD HH:MM:SS.ssss  0 NET.STA.LOC.CHA  HASH DATE TIME PHASE_HINT  None False

Fields are whitespace-delimited. The NSLC is dot-delimited within field 4
(0-indexed), and the phase hint (P, S, etc.) is in field 8.
"""

from __future__ import annotations

import base64
import hashlib
import logging
from pathlib import Path
from statistics import median

from obspy.core.utcdatetime import UTCDateTime

logger = logging.getLogger(__name__)


def parse_phase_picks(
    marker_file_path: str | Path,
    phase_hint: str = "P",
) -> list[dict]:
    """Parse phase picks from a Snuffler marker file.

    Parameters
    ----------
    marker_file_path : str or Path
        Path to a ``.mrkr`` file.
    phase_hint : str
        Phase type to extract (default ``"P"``).

    Returns
    -------
    list of dict
        Each dict has keys ``station``, ``channel``, ``network``,
        ``location``, ``time`` (:class:`UTCDateTime`), and ``phase``.
    """
    picks = []
    for marker in _iter_phase_markers(marker_file_path):
        if marker["phase_hint"] == phase_hint:
            picks.append(
                {
                    "network": marker["network"],
                    "station": marker["station"],
                    "location": marker["location"],
                    "channel": marker["channel"],
                    "time": marker["time"],
                    "phase": phase_hint,
                }
            )

    logger.info(
        "Parsed %d %s-phase picks from %s",
        len(picks),
        phase_hint,
        marker_file_path,
    )
    return picks


def party_to_markers(
    party,
    output_path: str | Path,
    *,
    phase_hint: str | None = None,
) -> Path:
    """Convert an EQcorrscan Party to a Snuffler marker file.

    Parameters
    ----------
    party : eqcorrscan.core.match_filter.Party
        Detection party containing families with catalogs of events.
    output_path : str or Path
        Path to write the ``.mrkr`` file.
    phase_hint : str, optional
        If provided, override all pick phase hints with this value.
        Otherwise use each pick's own ``phase_hint``.

    Returns
    -------
    Path
        The written file path.
    """
    output_path = Path(output_path)
    events: dict[str, list[dict]] = {}

    for family in party.families:
        for event in family.catalog.events:
            pick_times = [p.time.timestamp for p in event.picks if p.time]
            if not pick_times:
                continue

            median_time = median(pick_times)
            event_time_str = UTCDateTime(median_time).strftime("%Y-%m-%d %H:%M:%S.%f")[:-1]
            event_hash = base64.urlsafe_b64encode(
                hashlib.sha1(event_time_str.encode("utf8")).digest()
            ).decode("ascii")

            picks_list: list[dict] = []
            for pick in event.picks:
                wid = pick.waveform_id
                net = wid.network_code or "XX"
                sta = wid.station_code or ""
                cha = wid.channel_code or ""
                hint = phase_hint if phase_hint is not None else (pick.phase_hint or "P")
                picks_list.append(
                    {"station": f"{sta}.{cha}", "time": pick.time, "_net": net, "_hint": hint}
                )

            events[event_hash] = picks_list

    # Write using the Snuffler format directly (we need per-pick network codes
    # and phase hints, which write_events_to_markers doesn't support)
    lines: list[str] = []
    for event_id, picks in events.items():
        if not picks:
            continue
        event_time = picks[0]["time"].datetime.strftime("%Y-%m-%d %H:%M:%S.%f")
        event_time = event_time[:-1]

        event_line = (
            f"event: {event_time}  0 {event_id}          "
            f"0.0          0.0 None         None None  Event None\n"
        )
        lines.append(event_line)

        for pick in picks:
            pick_parts = pick["station"].split(".")
            pick_station = pick_parts[0]
            pick_channel = pick_parts[1] if len(pick_parts) > 1 else ""
            net_code = pick["_net"]
            hint = pick["_hint"]

            pick_time = pick["time"].datetime.strftime("%Y-%m-%d %H:%M:%S.%f")
            pick_time = pick_time[:-1]

            nslc = f"{net_code}.{pick_station}..{pick_channel}"
            pick_line = (
                f"phase: {pick_time}  0 {nslc: <16}"
                f"{event_id} {event_time[:10]}  {event_time[10:]} "
                f"{hint: <9}None False\n"
            )
            lines.append(pick_line)

    with open(output_path, "w") as fh:
        fh.writelines(lines)

    logger.info("Wrote %d events from Party to %s", len(events), output_path)
    return output_path


def parse_all_markers(marker_file_path: str | Path) -> list[dict]:
    """Parse all marker lines from a Snuffler marker file.

    Returns a list of dicts with keys depending on marker type.
    Currently supports ``phase`` markers.  Other types are returned
    with ``type`` and ``raw_line`` keys.
    """
    markers = []
    with open(marker_file_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("phase:"):
                parsed = _parse_phase_line(line)
                if parsed is not None:
                    markers.append({"type": "phase", **parsed})
                else:
                    markers.append({"type": "phase", "raw_line": line})
            else:
                markers.append({"type": "unknown", "raw_line": line})
    return markers


def write_events_to_markers(
    events: dict,
    output_path: str | Path,
    *,
    station_order: list[str] | None = None,
    phase_hint: str = "P",
) -> Path:
    """Write event picks to a Snuffler marker file.

    Parameters
    ----------
    events : dict
        ``{event_id: [{"station": "STA.CHA", "time": UTCDateTime}, ...], ...}``
        Each event is a list of pick dicts.
    output_path : str or Path
        Path to write the ``.mrkr`` file.
    station_order : list of str, optional
        If provided, station names are replaced by their index in this list
        for the network code field (Snuffler distance hack for plotting).
        Otherwise the network code is left as ``"XX"``.
    phase_hint : str
        Phase label to use in the marker file.

    Returns
    -------
    Path
        The written file path.
    """
    output_path = Path(output_path)
    station_distance: dict[str, int] = {}
    if station_order is not None:
        station_distance = {sta: idx for idx, sta in enumerate(station_order)}

    lines: list[str] = []
    for event_id, picks in events.items():
        if not picks:
            continue
        event_time = picks[0]["time"].datetime.strftime("%Y-%m-%d %H:%M:%S.%f")
        event_time = event_time[:-1]  # 5-digit microseconds per Snuffler

        event_line = (
            f"event: {event_time}  0 {event_id}          "
            f"0.0          0.0 None         None None  Event None\n"
        )
        lines.append(event_line)

        for pick in picks:
            pick_parts = pick["station"].split(".")
            pick_station = pick_parts[0]
            pick_channel = pick_parts[1] if len(pick_parts) > 1 else ""

            net_code = str(station_distance.get(pick_station, "XX"))
            pick_time = pick["time"].datetime.strftime("%Y-%m-%d %H:%M:%S.%f")
            pick_time = pick_time[:-1]

            nslc = f"{net_code}.{pick_station}..{pick_channel}"
            pick_line = (
                f"phase: {pick_time}  0 {nslc: <16}"
                f"{event_id} {event_time[:10]}  {event_time[10:]} "
                f"{phase_hint: <9}None False\n"
            )
            lines.append(pick_line)

    with open(output_path, "w") as fh:
        fh.writelines(lines)

    logger.info("Wrote %d events to %s", len(events), output_path)
    return output_path


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _iter_phase_markers(
    marker_file_path: str | Path,
) -> list[dict]:
    """Yield parsed phase marker dicts from a Snuffler marker file."""
    results = []
    with open(marker_file_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.startswith("phase:"):
                continue
            parsed = _parse_phase_line(line)
            if parsed is not None:
                results.append(parsed)
    return results


def _parse_phase_line(line: str) -> dict | None:
    """Parse a single Snuffler phase marker line.

    Expected whitespace-delimited fields::

        0: "phase:"
        1: date (YYYY-MM-DD)
        2: time (HH:MM:SS.ssss)
        3: "0" (or other integer)
        4: NET.STA.LOC.CHA (dot-delimited NSLC)
        ...
        8: phase_hint (P, S, etc.)

    Returns None if the line cannot be parsed.
    """
    fields = line.split()
    if len(fields) < 9:
        return None

    nslc = fields[4]
    nslc_parts = nslc.split(".")
    if len(nslc_parts) < 4:
        return None

    try:
        pick_time = UTCDateTime(f"{fields[1]} {fields[2]}")
    except (TypeError, ValueError):
        return None

    return {
        "network": nslc_parts[0],
        "station": nslc_parts[1],
        "location": nslc_parts[2],
        "channel": nslc_parts[3],
        "time": pick_time,
        "phase_hint": fields[8],
    }
