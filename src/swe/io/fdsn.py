"""FDSN web-service data retrieval utilities.

Download waveform data, station metadata (instrument responses), station
inventories for a region and epoch, and event catalogs from FDSN-compliant
data centers using ObsPy's :class:`~obspy.clients.fdsn.Client`.

Every function takes a ``client`` argument that may be an ObsPy data-center
key (e.g. ``"IRIS"``, ``"USGS"``; ``"EARTHSCOPE"`` in ObsPy >= 1.5), a base
URL such as ``"https://service.earthscope.org"``, or a ready
:class:`~obspy.clients.fdsn.Client` instance (reused, not re-created).
"""

from __future__ import annotations

import calendar
import functools
import glob
import logging
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from obspy import Catalog, Inventory, Trace, UTCDateTime, read_inventory
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNException, FDSNNoDataException

logger = logging.getLogger(__name__)

#: Networks excluded from region inventories by default. ``SY`` is the
#: synthetic-seismogram network served by the IRIS/EarthScope DMC: its
#: "stations" reuse real station codes and coordinates but are not real
#: deployments, and they appear in any epoch (including before the real
#: station existed).
DEFAULT_EXCLUDED_NETWORKS: tuple[str, ...] = ("SY",)

_STATIONXML_NAME = "{network}.{station}.{location}.{channel}.xml"
_LEGACY_RESP_NAME = "RESP.{network}.{station}..{channel}"
# Errors read_inventory raises for unreadable or non-metadata files
# (lxml's XMLSyntaxError subclasses SyntaxError).
_PARSE_ERRORS = (TypeError, ValueError, OSError, SyntaxError)


def _client(client: str | Client) -> Client:
    """Return *client* itself, or a new Client for a data-center key or base URL."""
    if isinstance(client, str):
        return Client(client)
    return client


def get_waveforms(
    network: str,
    stations: str | Sequence[str],
    channels: str | Sequence[str],
    start_time: UTCDateTime,
    end_time: UTCDateTime,
    *,
    output_dir: str | Path = ".",
    location: str = "**",
    client: str | Client = "IRIS",
    filename_format: str = "{network}.{station}.{channel}.{date}.ms",
) -> list[Path]:
    """Download waveforms and write daily MiniSEED files.

    Parameters
    ----------
    network : str
        FDSN network code (e.g. ``"UW"``).
    stations : str or sequence of str
        Station code(s).
    channels : str or sequence of str
        Channel code(s).
    start_time, end_time : UTCDateTime
        Requested time window.
    output_dir : str or Path
        Directory to write daily files into.
    location : str
        Location code pattern (default ``"**"`` for any location).
    client : str or Client
        FDSN data-center key, base URL, or Client instance (default
        ``"IRIS"``; see the module docstring).
    filename_format : str
        Template for output filenames.  Available placeholders:
        ``{network}``, ``{station}``, ``{channel}``, ``{date}``
        (``YYYY-MM-DD``).

    Returns
    -------
    list of Path
        Paths to written MiniSEED files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(stations, str):
        stations = [stations]
    if isinstance(channels, str):
        channels = [channels]

    fdsn_client = _client(client)
    written: list[Path] = []

    for station in stations:
        for channel in channels:
            try:
                st = fdsn_client.get_waveforms(
                    network,
                    station,
                    location,
                    channel,
                    start_time,
                    end_time,
                )
            except (FDSNException, ConnectionError, OSError):
                logger.warning(
                    "No data for %s.%s.%s (%s - %s)",
                    network,
                    station,
                    channel,
                    start_time,
                    end_time,
                )
                continue

            for tr in st:
                tr.stats.sampling_rate = round(tr.stats.sampling_rate)
            st.merge(method=1, fill_value=0)

            if not st:
                continue

            stream_start = st[0].stats.starttime
            stream_end = st[0].stats.endtime

            day_start = UTCDateTime(
                f"{stream_start.year}-{stream_start.month:02}-{stream_start.day:02}T00:00:00.0Z"
            )

            while day_start < stream_end:
                day_end = UTCDateTime(
                    f"{day_start.year}-{day_start.month:02}-{day_start.day:02}T23:59:59.999999999Z"
                )
                day_st = st.copy()
                day_st.trim(day_start, day_end, nearest_sample=False)

                if day_st:
                    date_str = f"{day_start.year}-{day_start.month:02}-{day_start.day:02}"
                    fname = filename_format.format(
                        network=network,
                        station=station,
                        channel=channel,
                        date=date_str,
                    )
                    outpath = output_dir / fname
                    day_st.write(str(outpath), format="MSEED")
                    written.append(outpath)
                    logger.info("Wrote %s", outpath)

                # Advance to next day
                year = day_start.year
                month = day_start.month
                day = day_start.day
                month_end = calendar.monthrange(year, month)[1]
                if month == 12 and day == month_end:
                    year += 1
                    month = 1
                    day = 1
                elif day == month_end:
                    month += 1
                    day = 1
                else:
                    day += 1
                day_start = UTCDateTime(f"{year}-{month:02}-{day:02}T00:00:00.0Z")

    return written


def get_response_files(
    network: str,
    stations: Sequence[str],
    channel: str,
    start_time: UTCDateTime,
    end_time: UTCDateTime,
    *,
    output_dir: str | Path = ".",
    client: str | Client = "IRIS",
) -> list[Path]:
    """Download station response metadata as StationXML, one file per channel.

    Each channel the service returns is written to
    ``{network}.{station}.{location}.{channel}.xml``, using the real codes
    even when the request used patterns, with all of the channel's epochs
    that overlap the request. Downloads for different channels into one
    directory add files; a later download of the same channel replaces its
    file, so request the whole time span you will process (or use separate
    directories per span). :func:`load_response` reads them back.

    Parameters
    ----------
    network : str
        FDSN network code or pattern.
    stations : sequence of str
        Station codes or patterns.
    channel : str
        Channel code or pattern (e.g. ``"BH?"``).
    start_time, end_time : UTCDateTime
        Requested epoch.
    output_dir : str or Path
        Directory to write StationXML files into.
    client : str or Client
        FDSN data-center key, base URL, or Client instance.

    Returns
    -------
    list of Path
        Paths to written StationXML files, one per channel.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fdsn_client = _client(client)
    written: list[Path] = []

    for station in stations:
        try:
            inv = fdsn_client.get_stations(
                network=network,
                station=station,
                channel=channel,
                starttime=start_time,
                endtime=end_time,
                level="response",
            )
            # One file per (net, sta, loc, cha); a channel with several epochs
            # keeps them all in its one file.
            keys = dict.fromkeys(
                (net.code, sta.code, cha.location_code, cha.code)
                for net in inv
                for sta in net
                for cha in sta
            )
            for net_code, sta_code, loc, cha_code in keys:
                one = inv.select(network=net_code, station=sta_code, location=loc, channel=cha_code)
                outpath = output_dir / _STATIONXML_NAME.format(
                    network=net_code, station=sta_code, location=loc, channel=cha_code
                )
                one.write(str(outpath), format="STATIONXML")
                written.append(outpath)
            logger.info("Wrote responses for %s.%s", network, station)
        except (FDSNException, ConnectionError, OSError):
            logger.warning("Failed to get response for %s.%s", network, station)

    return written


@functools.lru_cache(maxsize=256)
def _read_inventory_cached(
    path: str, stamp: tuple[int, int, int, int], fmt: str | None
) -> Inventory:
    """Parse *path* once per file state; *stamp* (inode, size, mtime, ctime) keys the cache."""
    return read_inventory(path) if fmt is None else read_inventory(path, format=fmt)


def _read_inventory(path: Path, *, legacy: bool = False) -> Inventory:
    st = path.stat()
    stamp = (st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    try:
        return _read_inventory_cached(str(path), stamp, None)
    except _PARSE_ERRORS:
        if not legacy:
            raise
        # Autodetection can reject or misread files the RESP reader accepts
        # (e.g. a leading blank line); legacy loaders always forced format="RESP".
        return _read_inventory_cached(str(path), stamp, "RESP")


def _response_candidates(trace: Trace, response_dir: Path) -> Iterator[tuple[Path, bool]]:
    """Yield ``(path, is_legacy)`` in lookup order; the directory is listed only if needed."""
    stats = trace.stats
    exact = response_dir / _STATIONXML_NAME.format(
        network=stats.network, station=stats.station, location=stats.location, channel=stats.channel
    )
    yield exact, False
    pattern = f"{glob.escape(stats.network)}.{glob.escape(stats.station)}.*.xml"
    for path in sorted(response_dir.glob(pattern)):
        if path != exact:
            yield path, False
    yield (
        response_dir
        / _LEGACY_RESP_NAME.format(
            network=stats.network, station=stats.station, channel=stats.channel
        ),
        True,
    )


def load_response(trace: Trace, response_dir: str | Path) -> Inventory:
    """Read the instrument response metadata for *trace* from *response_dir*.

    Candidate files, in order:

    1. ``{network}.{station}.{location}.{channel}.xml`` for the trace, as
       written by :func:`get_response_files`;
    2. any other ``{network}.{station}.*.xml``;
    3. the legacy per-channel file ``RESP.{network}.{station}..{channel}``.

    The first readable file whose metadata covers the trace is returned.
    "Covers" means the same network, station, location and channel, with an
    epoch containing the trace start. Unreadable or unrelated files are
    skipped with a warning. Formats are detected from content; a legacy file
    that detection rejects is retried as RESP. Parsed files are cached by
    inode, size, mtime and ctime, so per-window calls do not re-parse. Treat
    the returned Inventory as read-only.

    Parameters
    ----------
    trace : Trace
        Trace whose ``network``, ``station``, ``location``, ``channel`` and
        start time select the metadata.
    response_dir : str or Path
        Directory holding the response files.

    Returns
    -------
    Inventory
        Station metadata, suitable for :meth:`obspy.core.trace.Trace.remove_response`.

    Raises
    ------
    FileNotFoundError
        If no candidate file exists, or none covers the trace. The message
        lists the files read and skipped.
    """
    response_dir = Path(response_dir)
    stats = trace.stats
    read: list[str] = []
    for path, legacy in _response_candidates(trace, response_dir):
        if not path.is_file():
            continue
        try:
            inv = _read_inventory(path, legacy=legacy)
        except _PARSE_ERRORS as exc:
            logger.warning("Skipping unreadable response file %s: %s", path, exc)
            read.append(f"{path.name} (unreadable)")
            continue
        read.append(path.name)
        covering = inv.select(
            network=stats.network,
            station=stats.station,
            location=stats.location,
            channel=stats.channel,
            time=stats.starttime,
        )
        if len(covering):
            return inv
    if read:
        raise FileNotFoundError(
            f"No response file in {response_dir} covers {trace.id} at {stats.starttime}; "
            f"read: {', '.join(read)}"
        )
    raise FileNotFoundError(f"No response file for {trace.id} in {response_dir}")


def get_stations_in_region(
    *,
    starttime: UTCDateTime,
    endtime: UTCDateTime,
    minlatitude: float,
    maxlatitude: float,
    minlongitude: float,
    maxlongitude: float,
    network: str = "*",
    station: str = "*",
    channel: str = "*",
    level: str = "station",
    exclude_networks: str | Iterable[str] = DEFAULT_EXCLUDED_NETWORKS,
    client: str | Client = "IRIS",
) -> Inventory:
    """Station metadata for stations operating in a lat/lon box during an epoch.

    A station is returned when its epoch overlaps ``[starttime, endtime]``
    (FDSN ``starttime``/``endtime`` semantics). A box that crosses the
    antimeridian needs two calls, one on each side.

    Parameters
    ----------
    starttime, endtime : UTCDateTime
        Epoch of interest.
    minlatitude, maxlatitude, minlongitude, maxlongitude : float
        Box bounds in degrees.
    network, station, channel : str
        FDSN code patterns (default ``"*"``).
    level : str
        FDSN detail level: ``"network"``, ``"station"``, ``"channel"`` or
        ``"response"``.
    exclude_networks : str or iterable of str
        Network code(s) dropped from the result (case-insensitive). Defaults to
        :data:`DEFAULT_EXCLUDED_NETWORKS` (the synthetic ``SY`` network); pass
        ``()`` to keep everything. A network named explicitly in *network*
        (e.g. ``network="SY"``) is never excluded.
    client : str or Client
        FDSN data-center key, base URL, or Client instance.

    Returns
    -------
    Inventory
        Matching networks and stations; empty when the service has no match.
    """
    try:
        inv = _client(client).get_stations(
            network=network,
            station=station,
            channel=channel,
            starttime=starttime,
            endtime=endtime,
            minlatitude=minlatitude,
            maxlatitude=maxlatitude,
            minlongitude=minlongitude,
            maxlongitude=maxlongitude,
            level=level,
        )
    except FDSNNoDataException:
        logger.info("No stations in region for %s - %s", starttime, endtime)
        return Inventory(networks=[], source="swe")
    if isinstance(exclude_networks, str):
        exclude_networks = (exclude_networks,)
    # A network the caller names explicitly (no wildcard) is never excluded.
    named = {c.strip().upper() for c in network.split(",") if c.strip() and not set(c) & set("*?")}
    excluded = {code.upper() for code in exclude_networks} - named
    dropped = [n.code for n in inv.networks if n.code.upper() in excluded]
    if dropped:
        logger.info("Excluded networks %s", sorted(set(dropped)))
    inv.networks = [n for n in inv.networks if n.code.upper() not in excluded]
    return inv


def _iso(t: UTCDateTime | None) -> str | None:
    return None if t is None else t.isoformat()


def inventory_to_dataframe(inventory: Inventory, *, level: str = "station") -> pd.DataFrame:
    """Flatten an Inventory into a table with one row per station or channel.

    Parameters
    ----------
    inventory : Inventory
        Station metadata (e.g. from :func:`get_stations_in_region`).
    level : str
        ``"station"`` (one row per network/station epoch) or ``"channel"``
        (one row per channel epoch; stations without channels yield no rows).

    Returns
    -------
    DataFrame
        Columns ``network, station, latitude, longitude, elevation,
        start_date, end_date``; channel level adds ``location, channel,
        sample_rate`` and uses the channel's own coordinates and epoch. Dates
        are ISO-8601 UTC strings (open epochs are ``None``), which avoids
        pandas' nanosecond date range for far-future end dates.

    Raises
    ------
    ValueError
        If *level* is not ``"station"`` or ``"channel"``.
    """
    if level not in ("station", "channel"):
        raise ValueError(f"level must be 'station' or 'channel', got {level!r}")
    columns = ["network", "station", "latitude", "longitude", "elevation", "start_date", "end_date"]
    if level == "channel":
        columns += ["location", "channel", "sample_rate"]
    rows: list[dict[str, Any]] = []
    for net in inventory:
        for sta in net:
            if level == "station":
                rows.append(
                    dict(
                        network=net.code,
                        station=sta.code,
                        latitude=sta.latitude,
                        longitude=sta.longitude,
                        elevation=sta.elevation,
                        start_date=_iso(sta.start_date),
                        end_date=_iso(sta.end_date),
                    )
                )
                continue
            for cha in sta:
                rows.append(
                    dict(
                        network=net.code,
                        station=sta.code,
                        latitude=cha.latitude,
                        longitude=cha.longitude,
                        elevation=cha.elevation,
                        start_date=_iso(cha.start_date),
                        end_date=_iso(cha.end_date),
                        location=cha.location_code,
                        channel=cha.code,
                        sample_rate=cha.sample_rate,
                    )
                )
    return pd.DataFrame(rows, columns=columns)


def get_events(
    *,
    starttime: UTCDateTime,
    endtime: UTCDateTime,
    client: str | Client = "USGS",
    **query: Any,
) -> Catalog:
    """Query an FDSN event service; an empty Catalog when nothing matches.

    Parameters
    ----------
    starttime, endtime : UTCDateTime
        Origin-time window.
    client : str or Client
        FDSN data-center key, base URL, or Client instance (default
        ``"USGS"``).
    **query
        Further FDSN event parameters passed through unchanged, e.g.
        ``minmagnitude``, ``latitude``/``longitude``/``maxradius``,
        ``minlatitude``... box bounds, or ``includearrivals``.

    Returns
    -------
    Catalog
        Matching events.
    """
    try:
        return _client(client).get_events(starttime=starttime, endtime=endtime, **query)
    except FDSNNoDataException:
        logger.info("No events for %s - %s", starttime, endtime)
        return Catalog()


def catalog_to_dataframe(catalog: Catalog) -> pd.DataFrame:
    """Flatten a Catalog into a table with one row per event.

    Uses each event's preferred origin and magnitude, falling back to the
    first listed; missing values are ``None``.

    Parameters
    ----------
    catalog : Catalog
        Events (e.g. from :func:`get_events`).

    Returns
    -------
    DataFrame
        Columns ``event_id, time, latitude, longitude, depth_km, magnitude,
        magnitude_type``; ``time`` is an ISO-8601 UTC string.
    """
    columns = [
        "event_id",
        "time",
        "latitude",
        "longitude",
        "depth_km",
        "magnitude",
        "magnitude_type",
    ]
    rows: list[dict[str, Any]] = []
    for ev in catalog:
        origin = ev.preferred_origin() or (ev.origins[0] if ev.origins else None)
        mag = ev.preferred_magnitude() or (ev.magnitudes[0] if ev.magnitudes else None)
        depth = origin.depth if origin is not None else None
        rows.append(
            dict(
                event_id=str(ev.resource_id),
                time=_iso(origin.time) if origin is not None else None,
                latitude=origin.latitude if origin is not None else None,
                longitude=origin.longitude if origin is not None else None,
                depth_km=depth / 1000.0 if depth is not None else None,
                magnitude=mag.mag if mag is not None else None,
                magnitude_type=mag.magnitude_type if mag is not None else None,
            )
        )
    return pd.DataFrame(rows, columns=columns)
