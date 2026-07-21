"""Noise window validation: automated and interactive.

Provides two approaches for removing false-positive noise windows:

1. **Automated** - cross-reference against P-wave first-arrival picks
   from a Snuffler marker file (:func:`verify_noise_windows`).
2. **Interactive** - visual batch review with user input
   (:func:`explore_noise_windows`).
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import numpy as np
from obspy.core.utcdatetime import UTCDateTime

from swe.noise._plotting import plot_noise_windows

logger = logging.getLogger(__name__)


def verify_noise_windows(
    noise_windows: np.ndarray,
    noise_windows_times: list[tuple],
    marker_file_path: str | Path,
) -> tuple[np.ndarray, list[tuple]]:
    """Remove noise windows that contain P-wave first-arrival picks.

    Parses a Snuffler ``.mrkr`` marker file for P-wave phase picks and
    removes any noise window whose time range and station/channel match
    a pick.

    Parameters
    ----------
    noise_windows : numpy.ndarray
        2-D array of noise window amplitudes from :func:`find_noise_windows`.
    noise_windows_times : list of tuple
        Metadata ``(start, end, station, channel)`` for each window.
    marker_file_path : str or Path
        Path to a Snuffler marker file.

    Returns
    -------
    verified_windows : numpy.ndarray
        Noise windows with false positives removed.
    verified_times : list of tuple
        Corresponding metadata.
    """
    pick_list = _parse_snuffler_picks(marker_file_path)

    false_indices: list[int] = []
    for idx, window in enumerate(noise_windows_times):
        win_start, win_end, win_station, win_channel = window
        for pick_station, pick_time, pick_channel in pick_list:
            if (
                pick_station == win_station
                and pick_channel == win_channel
                and win_start <= pick_time <= win_end
            ):
                false_indices.append(idx)
                break

    verified_windows = np.delete(noise_windows, false_indices, axis=0)
    verified_times = copy.deepcopy(noise_windows_times)
    for idx in sorted(false_indices, reverse=True):
        del verified_times[idx]

    logger.info(
        "Found %d false noise windows containing picks.", len(false_indices)
    )
    return verified_windows, verified_times


def explore_noise_windows(
    noise_windows: np.ndarray,
    noise_windows_times: list[tuple],
    progress_file: str | Path | None = None,
    batch_size: int = 172,
) -> tuple[np.ndarray | None, list[tuple] | None]:
    """Interactively review noise windows in batches for visual validation.

    Plots batches of noise windows and prompts the user to flag false
    positives by entering space-separated indices.  Progress is saved
    to a JSON file so the review can be resumed across sessions.

    Parameters
    ----------
    noise_windows : numpy.ndarray
        2-D array of noise window amplitudes.
    noise_windows_times : list of tuple
        Metadata for each window.
    progress_file : str, Path, or None
        Path to a JSON file with saved progress from a previous session.
    batch_size : int
        Number of windows to display per batch (default 172).

    Returns
    -------
    verified_windows : numpy.ndarray or None
        Validated windows (returned only when the end is reached).
    verified_times : list of tuple or None
        Corresponding metadata.
    """

    def save_progress(index: int, removal_indices: list[int]) -> None:
        save_path = progress_file if progress_file is not None else "explore_progress.json"
        with open(save_path, "w") as f:
            json.dump({"start_index": index, "removal_indices": removal_indices}, f)

    def remove_windows(
        windows: np.ndarray,
        times: list[tuple],
        removal_indices: list[int],
    ) -> tuple[np.ndarray, list[tuple]]:
        removal_indices_sorted = sorted(removal_indices)
        verified_w = np.delete(windows, removal_indices_sorted, axis=0)
        verified_t = copy.deepcopy(times)
        for i in sorted(removal_indices_sorted, reverse=True):
            del verified_t[i]
        return verified_w, verified_t

    verified_windows = None
    verified_times = None

    if progress_file is not None:
        with open(progress_file, "r") as f:
            priors = json.load(f)
        start_index = priors["start_index"]
        removal_indices = priors["removal_indices"]
    else:
        start_index = 0
        removal_indices = []

    for index in range(start_index, len(noise_windows), batch_size):
        remaining = len(noise_windows) - index
        n_windows = min(batch_size, remaining)
        is_last_batch = n_windows < batch_size

        plot_noise_windows(
            noise_windows[index : index + n_windows],
            indices=list(range(index, index + n_windows)),
        )

        user_input = input(
            "Enter 'quit', nothing, or space-separated indices to flag for "
            "removal: "
        )

        if user_input.lower() == "quit":
            save_progress(index, removal_indices)
            break

        if user_input.strip():
            for item in user_input.split():
                removal_indices.append(int(item))

        if is_last_batch:
            save_progress(len(noise_windows) - 1, removal_indices)
            if removal_indices:
                verified_windows, verified_times = remove_windows(
                    noise_windows, noise_windows_times, removal_indices
                )
            else:
                verified_windows = noise_windows
                verified_times = noise_windows_times

    return verified_windows, verified_times


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_snuffler_picks(
    marker_file_path: str | Path,
) -> list[tuple[str, UTCDateTime, str]]:
    """Parse P-wave picks from a Snuffler marker file.

    Returns a list of ``(station, pick_time, channel)`` tuples.
    """
    from swe.io.snuffler import parse_phase_picks

    picks = parse_phase_picks(marker_file_path, phase_hint="P")
    return [(p["station"], p["time"], p["channel"]) for p in picks]
