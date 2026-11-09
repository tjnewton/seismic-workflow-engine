"""Automated seismic noise window detection using STA/LTA thresholding.

This module implements the core algorithm for finding noise windows in
continuous seismic waveform data by identifying periods where the STA/LTA
characteristic function remains below a user-defined threshold.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
from obspy import read
from obspy.core.utcdatetime import UTCDateTime
from obspy.signal.trigger import classic_sta_lta

from swe.io.fdsn import load_response

logger = logging.getLogger(__name__)


class NoiseWindowMeta(NamedTuple):
    """Metadata for a single noise window."""

    start: UTCDateTime
    """Window start time."""
    end: UTCDateTime
    """Window end time."""
    station: str
    """Station code."""
    channel: str
    """Channel code."""


class NoiseWindowResult(NamedTuple):
    """Result from noise window detection."""

    windows: np.ndarray
    """2-D array of noise window amplitudes."""
    times: list[NoiseWindowMeta]
    """Metadata for each window."""


def find_noise_windows(
    filepaths: Sequence[str | Path],
    detection_parameters: tuple[int, int, float, int],
    time_periods: list[tuple[UTCDateTime, UTCDateTime]],
    *,
    bandpass_filter: bool = False,
    freqmin: float = 20.0,
    freqmax: float = 60.0,
    corners: int = 4,
    remove_response: bool = False,
    response_dir: str | Path | None = None,
    station_parser: Callable[[str | Path], tuple[str, str]] | None = None,
) -> NoiseWindowResult:
    """Detect noise windows in continuous seismic data via STA/LTA thresholding.

    Reads MSEED files, computes the classic STA/LTA characteristic function,
    and extracts fixed-length windows where all samples fall below the
    threshold (i.e. are considered noise rather than signal).

    Parameters
    ----------
    filepaths : list of str or Path
        Paths to MSEED files, each containing data for a single
        station/channel.
    detection_parameters : tuple of (sta, lta, threshold, window_length)
        - sta: short-term-average window length in samples
        - lta: long-term-average window length in samples
        - threshold: STA/LTA ratio below which data is considered noise
        - window_length: desired noise window length in samples
    time_periods : list of (start, end) UTCDateTime tuples
        Time windows to search for noise within.
    bandpass_filter : bool
        If True, apply a bandpass filter and linear detrend before detection.
    freqmin, freqmax : float
        Bandpass filter corner frequencies in Hz (only used when
        ``bandpass_filter=True``).
    corners : int
        Number of filter corners (only used when ``bandpass_filter=True``).
    remove_response : bool
        If True, remove instrument response using the response files in
        *response_dir* (see :func:`swe.io.load_response`).
    response_dir : str, Path, or None
        Directory of response files: StationXML from
        :func:`swe.io.get_response_files`, or legacy ``RESP.*`` files.
        Required when ``remove_response=True``.
    station_parser : callable or None
        A function ``f(filepath) -> (station, channel)`` that extracts
        metadata from the file path.  If *None*, the default parser splits
        the filename on ``"."`` expecting the pattern
        ``NETWORK.STATION..CHANNEL.DATE.ms``.

    Returns
    -------
    noise_windows : numpy.ndarray
        2-D array of shape ``(n_windows, window_length)`` containing raw
        amplitude samples for each detected noise window.
    noise_windows_times : list of tuple
        Each entry is ``(start_time, end_time, station, channel)`` where
        times are :class:`~obspy.core.utcdatetime.UTCDateTime` objects.

    Examples
    --------
    >>> from obspy.core.utcdatetime import UTCDateTime
    >>> filepaths = ["5A.27..DP1.2018-03-13T00.00.00.ms"]
    >>> params = (15, 250, 3, 240)
    >>> periods = [(UTCDateTime("2018-03-13T01:06:00"),
    ...             UTCDateTime("2018-03-13T01:16:00"))]
    >>> windows, times = find_noise_windows(filepaths, params, periods)
    """
    sta, lta, threshold_value, threshold_period = detection_parameters
    parse = station_parser or _default_station_parser

    noise_windows: list[np.ndarray] = []
    noise_windows_times: list[NoiseWindowMeta] = []

    for filepath in filepaths:
        t0 = time.time()
        filepath = Path(filepath)

        if not filepath.exists():
            logger.warning("File does not exist: %s", filepath)
            continue

        station, channel = parse(filepath)
        logger.info("Processing station=%s channel=%s", station, channel)

        trace = read(str(filepath)).merge(method=1, fill_value=0)[0]

        for start_time, end_time in time_periods:
            period_trace = trace.copy().trim(start_time, end_time)

            if remove_response:
                if response_dir is None:
                    raise ValueError("response_dir must be provided when remove_response=True")
                inv = load_response(period_trace, response_dir)
                period_trace.remove_response(inventory=inv, output="VEL", plot=False)

            if bandpass_filter:
                period_trace.filter(
                    "bandpass",
                    freqmin=freqmin,
                    freqmax=freqmax,
                    corners=corners,
                )
                period_trace.detrend("linear")

            times = period_trace.times("utcdatetime")
            data = period_trace.data

            cft = classic_sta_lta(data, sta, lta)
            stalta_mask = cft < threshold_value

            # Exclude the STA/LTA ramp-up region at the start
            for i in range(len(cft)):
                if cft[i] != 0:
                    stalta_mask[:i] = False
                    break

            # Fixed-window extraction (non-overlapping)
            station_windows: list[np.ndarray] = []
            station_times: list[NoiseWindowMeta] = []

            for idx in range(0, len(stalta_mask) - threshold_period, threshold_period):
                if stalta_mask[idx] and stalta_mask[idx : idx + threshold_period].all():
                    station_windows.append(data[idx : idx + threshold_period])
                    station_times.append(
                        NoiseWindowMeta(times[idx], times[idx + threshold_period], station, channel)
                    )

            n_found = len(station_windows)
            if n_found == 0:
                logger.info("%s--%s  Found 0 noise windows.", start_time, end_time)
            else:
                logger.info(
                    "%s--%s  Found %d noise windows.",
                    start_time,
                    end_time,
                    n_found,
                )
                noise_windows.extend(station_windows)
                noise_windows_times.extend(station_times)

        elapsed = time.time() - t0
        logger.info("station/channel runtime: %.2f minutes.", elapsed / 60)

    noise_windows_arr = (
        np.array(noise_windows) if noise_windows else np.empty((0, threshold_period))
    )
    return NoiseWindowResult(noise_windows_arr, noise_windows_times)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _default_station_parser(filepath: Path) -> tuple[str, str]:
    """Extract station and channel from a MiniSEED filename.

    Supports two common formats:

    - ``NETWORK.STATION..CHANNEL.DATE.ms`` (double-dot, e.g. from manual FDSN)
    - ``NETWORK.STATION.CHANNEL.DATE.ms`` (no double-dot, e.g. from swe.io)
    """
    parts = filepath.name.split(".")
    station = parts[1]
    if len(parts) >= 6 and parts[2] == "":
        # Double-dot format: 5A.27..DP1.2018-03-13T00.00.00.ms
        channel = parts[3]
    else:
        # Simple format: CI.WBS.BHZ.2019-07-06.ms
        channel = parts[2]
    return station, channel
