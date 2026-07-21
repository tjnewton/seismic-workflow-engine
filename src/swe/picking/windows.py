"""Sliding-window construction for seismic phase picking.

Provides functions to extract overlapping data windows from waveform files,
apply threshold-based pick culling, and remove duplicate picks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np
from obspy import read
from obspy.core.inventory import read_inventory

logger = logging.getLogger(__name__)


class PickingWindowResult(NamedTuple):
    """Result from sliding-window extraction."""

    windows: np.ndarray
    """Shape ``(N, window_length, 2)`` -- log-amplitude and sign."""
    windows_untransformed: list[np.ndarray]
    """Raw amplitude windows."""
    times: list[tuple]
    """``(window_start, window_end, station, channel)`` metadata."""


class ThresholdPickResult(NamedTuple):
    """Result from threshold-based pick culling."""

    picked_windows: list[np.ndarray]
    """Windows containing picks above threshold."""
    picked_windows_times: list[tuple]
    """Metadata for retained windows."""
    index_of_max_prediction: list[int]
    """Sample index of the maximum prediction within each window."""


class DeduplicatedPickResult(NamedTuple):
    """Result from duplicate pick removal."""

    station_picks: dict[str, list]
    """``{station: [pick_time, ...]}`` of unique picks."""
    unique_windows: list[np.ndarray]
    """Deduplicated windows."""
    unique_windows_times: list[tuple]
    """Deduplicated metadata."""
    unique_index_of_max_prediction: list[int]
    """Deduplicated pick indices."""


def find_picking_windows(
    filepaths: Sequence[str | Path],
    window_length: int,
    window_offset: int,
    time_periods: list[tuple],
    *,
    sampling_rate: float = 250.0,
    apply_filter: bool = False,
    freqmin: float = 20.0,
    freqmax: float = 60.0,
    corners: int = 4,
    apply_response: bool = False,
    response_dir: str | Path | None = None,
    epsilon: float = 1e-6,
) -> PickingWindowResult:
    """Extract overlapping windows from waveform files for phase picking.

    Each window is log-transformed and paired with its sign for downstream
    model consumption.  Raw (untransformed) windows and metadata are also
    returned.

    Parameters
    ----------
    filepaths : sequence of str or Path
        Paths to ObsPy-readable waveform files.  Each file should contain
        data for a single station/channel combination.
    window_length : int
        Window length in samples.
    window_offset : int
        Window step (stride) in samples.
    time_periods : list of tuple
        ``[(start_time, end_time), ...]`` where each element is a pair of
        :class:`~obspy.UTCDateTime` objects defining a time period of
        interest.  All periods are assumed to fall on the same day.
    sampling_rate : float
        Target sampling rate in Hz.  Traces not at this rate are
        interpolated.
    apply_filter : bool
        If *True*, detrend and bandpass-filter each period.
    freqmin, freqmax : float
        Bandpass filter corner frequencies (Hz).
    corners : int
        Number of filter corners.
    apply_response : bool
        If *True*, remove instrument response.  Requires *response_dir*.
    response_dir : str or Path, optional
        Directory containing RESP files for response removal.
    epsilon : float
        Small value added before ``log`` to avoid ``log(0)``.

    Returns
    -------
    picking_windows : numpy.ndarray
        Shape ``(N, window_length, 2)`` -- log-amplitude and sign.
    picking_windows_untransformed : list of numpy.ndarray
        Raw amplitude windows.
    picking_windows_times : list of tuple
        ``(window_start, window_end, station, channel)`` metadata for each
        window.
    """
    picking_windows: list[np.ndarray] = []
    picking_windows_untransformed: list[np.ndarray] = []
    picking_windows_times: list[tuple] = []

    for filepath in filepaths:
        filepath = Path(filepath)
        filename = filepath.name
        parts = filename.split(".")
        station = parts[1] if len(parts) > 1 else "UNK"
        channel = parts[3] if len(parts) > 3 else "UNK"

        station_windows: list[np.ndarray] = []
        station_windows_times: list[tuple] = []

        trace = read(str(filepath)).merge(method=1, fill_value=0)[0]

        # Interpolate to target sampling rate if needed
        if trace.stats.sampling_rate != sampling_rate:
            starttime = time_periods[0][0] - 60
            endtime = time_periods[-1][1] + 60
            npts = int((endtime - starttime) * sampling_rate)
            trace.interpolate(
                sampling_rate,
                method="lanczos",
                starttime=starttime,
                endtime=endtime,
                npts=npts,
                a=30,
            )

        for start_time, end_time in time_periods:
            period_trace = trace.copy()
            period_trace = period_trace.trim(start_time, end_time)

            if apply_response and response_dir is not None:
                response_dir = Path(response_dir)
                resp_file = response_dir / (
                    f"RESP.{period_trace.stats.network}"
                    f".{period_trace.stats.station}"
                    f"..{period_trace.stats.channel}"
                )
                inv = read_inventory(
                    path_or_file_object=str(resp_file), format="RESP"
                )
                period_trace.remove_response(
                    inventory=inv, output="VEL", plot=False
                )

            if apply_filter:
                period_trace.detrend("linear")
                period_trace.filter(
                    "bandpass",
                    freqmin=freqmin,
                    freqmax=freqmax,
                    corners=corners,
                )

            times = period_trace.times("utcdatetime")
            data = period_trace.data

            for index in range(0, len(data) - window_length, window_offset):
                window_start = times[index]
                window_end = times[index + window_length - 1]
                station_windows.append(data[index : index + window_length])
                station_windows_times.append(
                    (window_start, window_end, station, channel)
                )

            for w_idx in range(len(station_windows)):
                window_data = station_windows[w_idx]
                reshaped = np.zeros((len(window_data), 2))
                picking_windows_untransformed.append(window_data)

                for row_idx in range(len(reshaped)):
                    reshaped[row_idx][1] = np.sign(window_data[row_idx])
                    reshaped[row_idx][0] = np.log(
                        np.abs(window_data[row_idx]) + epsilon
                    )

                picking_windows.append(reshaped)
                picking_windows_times.append(station_windows_times[w_idx])

    picking_windows_arr = np.array(picking_windows)

    return PickingWindowResult(picking_windows_arr, picking_windows_untransformed, picking_windows_times)


def get_threshold_picks(
    pick_predictions: np.ndarray,
    threshold: float,
    picking_windows_untransformed: list[np.ndarray],
    picking_windows_times: list[tuple],
) -> ThresholdPickResult:
    """Cull pick predictions to those exceeding a confidence threshold.

    Parameters
    ----------
    pick_predictions : numpy.ndarray
        Model output predictions, shape ``(N, window_length)``.
    threshold : float
        Minimum prediction value to retain a pick.
    picking_windows_untransformed : list of numpy.ndarray
        Raw amplitude windows corresponding to *pick_predictions*.
    picking_windows_times : list of tuple
        Metadata for each window.

    Returns
    -------
    picked_windows : list of numpy.ndarray
        Windows containing picks above threshold.
    picked_windows_times : list of tuple
        Metadata for retained windows.
    index_of_max_prediction : list of int
        Sample index of the maximum prediction within each retained window.
    """
    max_row_prediction = np.max(pick_predictions, axis=1)
    above_threshold_rows = np.where(max_row_prediction > threshold)[0]

    index_of_max_prediction: list[int] = []
    for row in above_threshold_rows:
        row_max_idx = np.where(
            pick_predictions[row] == max_row_prediction[row]
        )
        index_of_max_prediction.append(row_max_idx[0][0])

    picked_windows: list[np.ndarray] = []
    picked_windows_times: list[tuple] = []
    for row in above_threshold_rows:
        picked_windows.append(picking_windows_untransformed[row])
        picked_windows_times.append(picking_windows_times[row])

    return ThresholdPickResult(picked_windows, picked_windows_times, index_of_max_prediction)


def remove_duplicate_picks(
    picked_windows: list[np.ndarray],
    picked_windows_times: list[tuple],
    index_of_max_prediction: list[int],
    station_list: Sequence[str],
    *,
    sampling_rate: float = 250.0,
    duplicate_threshold: float = 10.0,
) -> DeduplicatedPickResult:
    """Remove picks that occur within a threshold window on the same station.

    When multiple picks on the same station fall within
    ``duplicate_threshold / sampling_rate`` seconds of each other, only the
    earliest is retained.

    Parameters
    ----------
    picked_windows : list of numpy.ndarray
        Windows containing picks.
    picked_windows_times : list of tuple
        ``(start_time, end_time, station, channel)`` for each window.
    index_of_max_prediction : list of int
        Sample index of the pick within each window.
    station_list : sequence of str
        All station names in the project.
    sampling_rate : float
        Sampling rate in Hz (used with *duplicate_threshold*).
    duplicate_threshold : float
        Minimum separation in samples for two picks on the same station
        to be considered distinct.

    Returns
    -------
    station_picks : dict
        ``{station: [pick_time, ...]}`` of unique picks.
    unique_picked_windows : list of numpy.ndarray
        Deduplicated windows.
    unique_picked_windows_times : list of tuple
        Deduplicated metadata.
    unique_index_of_max_prediction : list of int
        Deduplicated pick indices.
    """
    if len(picked_windows_times) == 0:
        return DeduplicatedPickResult({}, [], [], [])

    station_picks: dict[str, list] = {str(sta): [] for sta in station_list}
    duplicate_list: list[int] = []

    for index in range(len(picked_windows_times)):
        window_start, _window_end, station, _channel = picked_windows_times[
            index
        ]
        pick_time = window_start + (
            index_of_max_prediction[index] / sampling_rate
        )

        if station not in station_picks:
            station_picks[station] = []

        if len(station_picks[station]) == 0:
            station_picks[station].append(pick_time)
        else:
            duplicate_flag = False
            for pick in station_picks[station]:
                if abs(pick - pick_time) <= (
                    duplicate_threshold / sampling_rate
                ):
                    duplicate_flag = True
                    duplicate_list.append(index)

            if not duplicate_flag:
                station_picks[station].append(pick_time)

    unique_picked_windows = [
        item
        for idx, item in enumerate(picked_windows)
        if idx not in duplicate_list
    ]
    unique_picked_windows_times = [
        item
        for idx, item in enumerate(picked_windows_times)
        if idx not in duplicate_list
    ]
    unique_index_of_max_prediction = [
        item
        for idx, item in enumerate(index_of_max_prediction)
        if idx not in duplicate_list
    ]

    return DeduplicatedPickResult(
        station_picks,
        unique_picked_windows,
        unique_picked_windows_times,
        unique_index_of_max_prediction,
    )
