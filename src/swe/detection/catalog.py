"""Detection catalog management for matched-filter workflows.

Functions for sorting, filtering, and quality-controlling EQcorrscan
:class:`~eqcorrscan.Party` detection catalogs, plus the top-level
:func:`find_lfes` driver that chains detection, culling, stacking, and
iterative re-detection.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from obspy import Stream, UTCDateTime, read

from swe.utils.helpers import snr

logger = logging.getLogger(__name__)

try:
    from eqcorrscan import Family, Party
except ImportError as exc:  # pragma: no cover
    raise ImportError("eqcorrscan is required for swe.detection.catalog") from exc


class LFEResult(NamedTuple):
    """Result from the full LFE detection pipeline."""

    party: Party | None
    """Final detection catalog."""
    stack_pws: Stream
    """Phase-weighted stack."""
    stack_linear: Stream
    """Linear stack."""
    stack_ccs: list[float]
    """Per-detection cross-correlation values from stacking."""


def party_snrs(
    party: Party,
    streams_path: str | Path,
    main_trace: tuple[str, str, str],
    *,
    lowcut: float = 1.0,
    highcut: float = 15.0,
    samp_rate: float = 100.0,
    filename_pattern: str = "{net}.{sta}.{cha}.{year}-{month:02d}-{day:02d}.ms",
) -> list[float]:
    """Compute per-detection SNR for the primary trace.

    Parameters
    ----------
    party : eqcorrscan.Party
        Detection results.
    streams_path : str or Path
        Directory of day-long MiniSEED files.
    main_trace : tuple of str
        ``(network, station, channel)`` for the primary detection trace.
    lowcut, highcut : float
        Bandpass filter corners (Hz).
    samp_rate : float
        Target sampling rate (Hz).
    filename_pattern : str
        Filename pattern for day-long MiniSEED files.  Available
        placeholders: ``{net}``, ``{sta}``, ``{cha}``, ``{year}``,
        ``{month}``, ``{day}``.

    Returns
    -------
    list of float
        One SNR value per detection.  ``NaN`` where the file is missing.
    """
    streams_path = Path(streams_path)
    net, sta, cha = main_trace

    pick_times: list[UTCDateTime] = []
    for event in party.families[0].catalog.events:
        found = False
        for pick in event.picks:
            wf = pick.waveform_id
            if wf.station_code == sta and wf.network_code == net and wf.channel_code == cha:
                pick_times.append(pick.time)
                found = True
                break
        if not found:
            pick_times.append(event.picks[-1].time)

    snrs: list[float] = []
    for pick_time in pick_times:
        day_fname = filename_pattern.format(
            net=net,
            sta=sta,
            cha=cha,
            year=pick_time.year,
            month=pick_time.month,
            day=pick_time.day,
        )
        day_files = sorted(glob.glob(str(streams_path / day_fname)))
        if day_files:
            day_st = read(day_files[0])
            day_st.filter("bandpass", freqmin=lowcut, freqmax=highcut)
            day_st.interpolate(sampling_rate=samp_rate)
            day_st.trim(
                pick_time - 20,
                pick_time + 40,
                pad=True,
                fill_value=np.nan,
                nearest_sample=True,
            )
            snrs.append(snr(day_st)[0])
        else:
            snrs.append(np.nan)

    return snrs


def sort_party(
    party: Party,
    n: int,
    detection_files_path: str | Path,
    main_trace: tuple[str, str, str],
    *,
    save_csv: str | Path | None = None,
) -> Party:
    """Keep the top *n* detections ranked by correlation amplitude.

    Parameters
    ----------
    party : eqcorrscan.Party
        Input detections.
    n : int
        Number of top detections to keep.
    detection_files_path : str or Path
        Directory of day-long MiniSEED files (used for SNR computation).
    main_trace : tuple of str
        ``(network, station, channel)`` for SNR computation.
    save_csv : str or Path, optional
        If set, write a CSV of detection metadata to this path.

    Returns
    -------
    eqcorrscan.Party
        Party containing only the top *n* detections.
    """
    snrs_list = party_snrs(party, detection_files_path, main_trace)

    data = []
    for idx, detection in enumerate(party.families[0].detections):
        data.append(
            [
                idx,
                detection.detect_time,
                abs(detection.detect_val),
                detection.threshold,
                detection.threshold_type,
                detection.threshold_input,
                detection.template_name,
                snrs_list[idx],
            ]
        )

    df = pd.DataFrame(
        data,
        columns=[
            "index",
            "time",
            "abs_correlation_sum",
            "correlation_sum_threshold",
            "threshold_metric",
            "metric_input",
            "template",
            "snr",
        ],
    )

    top_n_df = df.nlargest(n, "abs_correlation_sum")
    keeper_indices = top_n_df["index"].tolist()

    detections = [party.families[0].detections[i] for i in keeper_indices]
    template = party.families[0].template
    top_n_party = Party(families=[Family(template, detections)])

    if save_csv is not None:
        df.to_csv(str(save_csv), index=False)
        logger.info("Saved detection CSV to %s", save_csv)

    return top_n_party


def get_day_detections(
    date: UTCDateTime,
    party: Party,
) -> Party:
    """Return detections from a single calendar day.

    Parameters
    ----------
    date : UTCDateTime
        Date of interest.
    party : eqcorrscan.Party
        Input detections.

    Returns
    -------
    eqcorrscan.Party
        Party containing only detections on the specified day.
    """
    detections = []
    for detection in party.families[0].detections:
        doi = detection.detect_time
        if doi.year == date.year and doi.month == date.month and doi.day == date.day:
            detections.append(detection)

    template = party.families[0].template
    return Party(families=[Family(template, detections)])


def cull_party_dates(
    dates: list[UTCDateTime],
    party: Party,
) -> Party:
    """Keep only detections on specified calendar dates.

    Parameters
    ----------
    dates : list of UTCDateTime
        Dates to keep.
    party : eqcorrscan.Party
        Input detections.

    Returns
    -------
    eqcorrscan.Party
        Filtered Party.
    """
    date_set = {(d.year, d.month, d.day) for d in dates}
    detections = []
    for detection in party.families[0].detections:
        doi = detection.detect_time
        if (doi.year, doi.month, doi.day) in date_set:
            detections.append(detection)

    template = party.families[0].template
    return Party(families=[Family(template, detections)])


def get_threshold_detections(
    party: Party,
    threshold: float,
) -> Party:
    """Keep only detections above a threshold value.

    The effective threshold for each detection is computed as
    ``detection.threshold / detection.threshold_input * threshold``.

    Parameters
    ----------
    party : eqcorrscan.Party
        Input detections.
    threshold : float
        Threshold multiplier.

    Returns
    -------
    eqcorrscan.Party
        Detections meeting the threshold criterion.
    """
    detections = []
    for detection in party.families[0].detections:
        detection_value = abs(detection.detect_val)
        detection_threshold = detection.threshold / detection.threshold_input
        threshold_value = detection_threshold * threshold
        if detection_value >= threshold_value:
            detections.append(detection)

    template = party.families[0].template
    return Party(families=[Family(template, detections)])


def find_lfes(
    templates: list[str],
    template_files: list[str | Path],
    station_dict: dict[str, dict],
    template_length: float,
    template_prepick: float,
    detection_files_path: str | Path,
    start_date: UTCDateTime,
    end_date: UTCDateTime,
    main_trace: tuple[str, str, str],
    *,
    detect_thresh: float = 8.0,
    thresh_type: str = "MAD",
    shift_method: str = "self",
    party_path: str | Path | None = None,
    stack_path: str | Path | None = None,
    cull: bool = False,
    snr_threshold: tuple[float, float] = (1.0, 15.0),
    top_n: int | None = None,
) -> LFEResult:
    """Full LFE detection pipeline: detect, cull, sort, stack.

    This is the top-level driver that chains matched-filter detection,
    SNR-based culling, correlation-amplitude sorting, and phase-aligned
    stacking into a single call.

    Parameters
    ----------
    templates : list of str
        HypoDD-format template lines.
    template_files : list of str or Path
        Paths to template waveform MiniSEED files.
    station_dict : dict
        ``{station: {"network": ..., "channel": ...}}`` metadata.
    template_length, template_prepick : float
        Template duration and pre-pick (seconds).
    detection_files_path : str or Path
        Directory of day-long MiniSEED files.
    start_date, end_date : UTCDateTime
        Detection date range.
    main_trace : tuple of str
        ``(network, station, channel)`` for the primary detection trace.
    detect_thresh : float
        Detection threshold.
    thresh_type : str
        Detection threshold type (``'MAD'``, ``'abs'``, etc.).
    shift_method : str
        Stacking alignment method (``'zero'``, ``'self'``, ``'stack'``, etc.).
    party_path : str or Path, optional
        Path to read a pre-computed Party (EQcorrscan native format).
        If *None*, detection is run from scratch.
    stack_path : str or Path, optional
        Path to read a pre-computed stack (ObsPy Stream format).
        If *None*, stacking is computed.
    cull : bool
        Whether to apply SNR-based culling.
    snr_threshold : tuple of float
        ``(min_snr, max_snr)`` for culling.
    top_n : int, optional
        If set, keep only the top *n* detections by correlation.

    Returns
    -------
    party : eqcorrscan.Party
        Final detection catalog.
    stack_pws : obspy.Stream
        Phase-weighted stack.
    stack_linear : obspy.Stream
        Linear stack.
    stack_ccs : list of float
        Per-detection cross-correlation values from stacking.
    """
    from swe.detection.matched_filter import (
        cull_detections,
        detect_signals,
    )
    from swe.detection.stacking import stack_template_detections

    # Step 1: Get detections
    if party_path is not None:
        party_path = Path(party_path)
        party = Party().read(str(party_path))
        logger.info("Loaded Party from %s", party_path)
    else:
        party = detect_signals(
            templates,
            template_files,
            station_dict,
            template_length,
            template_prepick,
            detection_files_path,
            start_date,
            end_date,
            thresh_type=thresh_type,
            detect_thresh=detect_thresh,
        )
        if party is None:
            return LFEResult(None, Stream(), Stream(), [])

    # Step 2: Cull by SNR
    if cull:
        party = cull_detections(
            party,
            detection_files_path,
            main_trace,
            snr_min=snr_threshold[0],
            snr_max=snr_threshold[1],
        )
        n_det = sum(len(f) for f in party.families)
        logger.info("After culling: %d detections", n_det)

    # Step 3: Sort by correlation amplitude
    if top_n is not None:
        party = sort_party(
            party,
            top_n,
            detection_files_path,
            main_trace,
        )
        n_det = sum(len(f) for f in party.families)
        logger.info("After top-%d sort: %d detections", top_n, n_det)

    # Step 4: Build stacks
    if stack_path is not None:
        stack_pws = read(str(stack_path))
        stack_linear = Stream()
        stack_ccs: list[float] = []
    else:
        family = sorted(party.families, key=lambda f: len(f))[-1]
        template_times: dict[str, list] = {}
        for trace in family.template.st:
            tid = f"{trace.stats.network}.{trace.stats.station}.{trace.stats.channel}"
            template_times[tid] = [
                trace.stats.starttime - template_prepick,
                trace.stats.endtime,
            ]

        stack_pws, stack_linear, stack_ccs = stack_template_detections(
            party,
            detection_files_path,
            main_trace,
            template_times,
            align_type=shift_method,
        )

    return LFEResult(party, stack_pws, stack_linear, stack_ccs)
