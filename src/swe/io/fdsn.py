"""FDSN web-service data retrieval utilities.

Download waveform data and station metadata (instrument responses) from
FDSN-compliant data centres using ObsPy's
:class:`~obspy.clients.fdsn.Client`.
"""

from __future__ import annotations

import calendar
import logging
from pathlib import Path
from typing import Sequence

from obspy import Inventory, Stream, UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNException

logger = logging.getLogger(__name__)


def get_waveforms(
    network: str,
    stations: str | Sequence[str],
    channels: str | Sequence[str],
    start_time: UTCDateTime,
    end_time: UTCDateTime,
    *,
    output_dir: str | Path = ".",
    location: str = "**",
    client: str = "IRIS",
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
    client : str
        FDSN data centre name (default ``"IRIS"``).
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

    fdsn_client = Client(client)
    written: list[Path] = []

    for station in stations:
        for channel in channels:
            try:
                st = fdsn_client.get_waveforms(
                    network, station, location, channel, start_time, end_time,
                )
            except (FDSNException, ConnectionError, OSError):
                logger.warning(
                    "No data for %s.%s.%s (%s - %s)",
                    network, station, channel, start_time, end_time,
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
                f"{stream_start.year}-{stream_start.month:02}-"
                f"{stream_start.day:02}T00:00:00.0Z"
            )

            while day_start < stream_end:
                day_end = UTCDateTime(
                    f"{day_start.year}-{day_start.month:02}-"
                    f"{day_start.day:02}T23:59:59.999999999Z"
                )
                day_st = st.copy()
                day_st.trim(day_start, day_end, nearest_sample=False)

                if day_st:
                    date_str = (
                        f"{day_start.year}-{day_start.month:02}-"
                        f"{day_start.day:02}"
                    )
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
                day_start = UTCDateTime(
                    f"{year}-{month:02}-{day:02}T00:00:00.0Z"
                )

    return written


def get_response_files(
    network: str,
    stations: Sequence[str],
    channel: str,
    start_time: UTCDateTime,
    end_time: UTCDateTime,
    *,
    output_dir: str | Path = ".",
    client: str = "IRIS",
) -> list[Path]:
    """Download station response files (StationXML).

    Parameters
    ----------
    network : str
        FDSN network code.
    stations : sequence of str
        Station codes.
    channel : str
        Channel code.
    start_time, end_time : UTCDateTime
        Requested epoch.
    output_dir : str or Path
        Directory to write StationXML files into.
    client : str
        FDSN data centre name.

    Returns
    -------
    list of Path
        Paths to written StationXML files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fdsn_client = Client(client)
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
            outpath = output_dir / f"RESP.{network}.{station}..{channel}"
            inv.write(str(outpath), format="STATIONXML")
            written.append(outpath)
            logger.info("Wrote response for %s.%s", network, station)
        except (FDSNException, ConnectionError, OSError):
            logger.warning("Failed to get response for %s.%s", network, station)

    return written
