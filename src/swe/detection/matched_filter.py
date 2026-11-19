"""Matched-filter detection pipeline using EQcorrscan.

Provides functions to build templates from Snuffler marker picks, run
matched-filter detection across multi-day datasets, and quality-control
the resulting detections.
"""

from __future__ import annotations

import glob
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
from obspy import Stream, UTCDateTime, read
from obspy.core.event import Catalog, Event, Magnitude, Origin, Pick, WaveformStreamID, read_events

from swe.utils.helpers import snr

logger = logging.getLogger(__name__)


class MarkerTemplateResult(NamedTuple):
    """Result from converting Snuffler markers to template metadata."""

    templates: list[str]
    """HypoDD-format phase lines."""
    station_dict: dict[str, dict]
    """``{station: {"network": ..., "channel": ...}}`` metadata."""
    pick_offset: dict[str, float]
    """``{station: offset_seconds}`` relative to earliest pick."""
    template_windows: dict[str, list]
    """``{net.sta: [starttime, endtime]}`` for each station."""


try:
    from eqcorrscan import Family, Party, Template, Tribe
    from eqcorrscan.core.match_filter.matched_filter import MatchFilterError
except ImportError as exc:  # pragma: no cover
    raise ImportError("eqcorrscan is required for swe.detection.matched_filter") from exc


def _next_day(dt: UTCDateTime) -> UTCDateTime:
    """Advance a UTCDateTime to the start of the next calendar day."""
    return UTCDateTime(dt.year, dt.month, dt.day) + 86400


def markers_to_template(
    marker_file_path: str | Path,
    prepick_offset: float,
    *,
    phase_hint: str = "S",
    template_duration: float = 16.0,
) -> MarkerTemplateResult:
    """Convert Snuffler marker picks into EQcorrscan template metadata.

    Reads an S-phase pick file and generates HypoDD-format template lines,
    a station dictionary, per-station pick offsets, and template time windows.

    Parameters
    ----------
    marker_file_path : str or Path
        Path to a ``.mrkr`` file with S-phase picks for a single event.
    prepick_offset : float
        Seconds before the S-wave pick to start each template window.
    phase_hint : str
        Phase type to extract from the marker file.
    template_duration : float
        Total duration of each template window in seconds.

    Returns
    -------
    templates : list of str
        HypoDD-format phase lines for input to
        :func:`obspy.core.event.read_events`.
    station_dict : dict
        ``{station: {"network": ..., "channel": ...}}`` metadata.
    pick_offset : dict
        ``{station: offset_seconds}`` relative to the earliest pick.
    template_windows : dict
        ``{net.sta: [starttime, endtime]}`` for each station.
    """
    marker_file_path = Path(marker_file_path)
    pick_dict: dict[str, UTCDateTime] = {}
    station_dict: dict[str, dict] = {}
    template_windows: dict[str, list] = {}
    pick_offset: dict[str, float] = {}
    earliest_pick_time: UTCDateTime | None = None
    earliest_pick_station: str = ""

    with open(marker_file_path) as fh:
        for line in fh:
            if len(line) <= 52:
                continue
            if not line.startswith("phase"):
                continue
            if line[-20:-19] != phase_hint:
                continue
            nslc_parts = line[35:51].split(".")
            if len(nslc_parts) < 4:
                continue

            pick_station = nslc_parts[1]
            # Preserve the marker's real channel code (do not force "...Z"):
            # borehole channels such as "DP1" have no "Z" component.
            pick_channel = nslc_parts[3].split()[0]
            pick_network = nslc_parts[0]
            pick_time = UTCDateTime(line[7:31])

            pick_dict[pick_station] = pick_time
            station_dict[pick_station] = {
                "network": pick_network,
                "channel": pick_channel,
            }
            template_windows[f"{pick_network}.{pick_station}"] = [
                pick_time - prepick_offset,
                (pick_time - prepick_offset) + template_duration,
            ]

            if earliest_pick_time is None or pick_time < earliest_pick_time:
                earliest_pick_time = pick_time
                earliest_pick_station = pick_station

    if earliest_pick_time is None:
        return MarkerTemplateResult([], {}, {}, {})

    earliest_pick_time_adj = earliest_pick_time - prepick_offset

    templates = [
        f"# {earliest_pick_time_adj.year} "
        f"{earliest_pick_time_adj.month:02} "
        f"{earliest_pick_time_adj.day:02} "
        f"{earliest_pick_time_adj.hour:02} "
        f"{earliest_pick_time_adj.minute:02} "
        f"{earliest_pick_time_adj.second:02}."
        f"{earliest_pick_time_adj.microsecond // 10000:02}  "
        f"0.0000 0.0000  0.00  1.00  0.0  0.0  0.00  1\n"
    ]
    templates.append(f"{earliest_pick_station}    0.000  1       P\n")
    pick_offset[earliest_pick_station] = 0.0

    for station, pick_time in pick_dict.items():
        if station == earliest_pick_station:
            continue
        time_diff = pick_time - (earliest_pick_time_adj + prepick_offset)
        microseconds = int((time_diff - int(time_diff)) * 1000)
        templates.append(f"{station:4}   {int(time_diff):2}.{microseconds:03}  1       P\n")
        pick_offset[station] = time_diff

    return MarkerTemplateResult(templates, station_dict, pick_offset, template_windows)


def _component_set(channel: str) -> tuple[str, ...]:
    """Return the three component codes matching a channel's convention.

    Surface sensors use ``Z``/``N``/``E``; borehole sensors (e.g. the Parkfield
    HRSN) use ``1``/``2``/``3``.  The convention is inferred from the final
    character of *channel* so templates build correctly for either.  An
    unrecognized final character yields just that single component.
    """
    last = channel[-1]
    if last in "ZNE":
        return ("Z", "N", "E")
    if last in "123":
        return ("1", "2", "3")
    return (last,)


def make_templates(
    templates: list[str],
    template_files: Sequence[str | Path],
    station_dict: dict[str, dict],
    template_length: float,
    template_prepick: float,
    *,
    lowcut: float = 1.0,
    highcut: float = 15.0,
    samp_rate: float = 100.0,
    filt_order: int = 4,
    swin: str = "all",
    process_len: float = 86400,
    three_component: bool = True,
) -> Tribe:
    """Build an EQcorrscan :class:`~eqcorrscan.Tribe` from template metadata.

    Parameters
    ----------
    templates : list of str
        HypoDD-format phase lines (from :func:`markers_to_template`).
    template_files : sequence of str or Path
        Paths to MiniSEED files containing template waveform data.
    station_dict : dict
        ``{station: {"network": ..., "channel": ...}}`` metadata.
    template_length : float
        Template duration in seconds.
    template_prepick : float
        Pre-pick padding in seconds.
    lowcut, highcut : float
        Bandpass filter corners (Hz).
    samp_rate : float
        Target sampling rate (Hz).
    filt_order : int
        Filter order.
    swin : str
        Which channels to include (``'all'``, ``'P'``, ``'S'``).
    process_len : float
        Processing length in seconds for EQcorrscan.
    three_component : bool
        If *True*, expand each pick to Z, N, E components.

    Returns
    -------
    eqcorrscan.Tribe
    """
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pha", delete=False) as tmp:
        for line in templates:
            tmp.write(line)
        tmp_path = tmp.name

    catalog = read_events(tmp_path, format="HYPODDPHA")
    Path(tmp_path).unlink(missing_ok=True)

    for event in catalog.events:
        picks = event.picks.copy()
        for idx, pick in enumerate(event.picks):
            sta = pick.waveform_id.station_code
            if sta not in station_dict:
                continue
            picks[idx].waveform_id.network_code = station_dict[sta]["network"]
            chan = station_dict[sta]["channel"]
            # Use the channel as given rather than forcing a "...Z" code: HRSN
            # borehole sensors use 1/2/3 component codes (e.g. "SP1"), which have
            # no "Z" equivalent.  _component_set yields the right sibling codes.
            picks[idx].waveform_id.channel_code = chan
            if three_component:
                for comp in _component_set(chan):
                    if comp == chan[-1]:
                        continue
                    copy = picks[idx].copy()
                    copy.waveform_id.channel_code = chan[:-1] + comp
                    picks.append(copy)
        event.picks = picks

    st = Stream()
    for f in template_files:
        st += read(str(f))
    st.interpolate(sampling_rate=samp_rate)

    tribe = Tribe().construct(
        method="from_meta_file",
        meta_file=catalog,
        st=st,
        lowcut=lowcut,
        highcut=highcut,
        samp_rate=samp_rate,
        length=template_length,
        filt_order=filt_order,
        prepick=template_prepick,
        swin=swin,
        process_len=process_len,
        parallel=False,
    )
    return tribe


def detect_signals(
    templates: list[str],
    template_files: Sequence[str | Path],
    station_dict: dict[str, dict],
    template_length: float,
    template_prepick: float,
    detection_files_path: str | Path,
    start_date: UTCDateTime,
    end_date: UTCDateTime,
    *,
    thresh_type: str = "MAD",
    detect_thresh: float = 8.0,
    trig_int: float = 8.0,
    lowcut: float = 1.0,
    highcut: float = 15.0,
    samp_rate: float = 100.0,
    output_path: str | Path | None = None,
) -> Party | None:
    """Run matched-filter detection over a date range.

    Parameters
    ----------
    templates : list of str
        HypoDD-format phase lines.
    template_files : sequence of str or Path
        Paths to template waveform files.
    station_dict : dict
        Station metadata.
    template_length, template_prepick : float
        Template duration and pre-pick padding (seconds).
    detection_files_path : str or Path
        Directory of day-long MiniSEED files.
    start_date, end_date : UTCDateTime
        Detection date range.
    thresh_type : str
        Threshold type (``'MAD'``, ``'abs'``, ``'av_chan_corr'``).
    detect_thresh : float
        Detection threshold value.
    trig_int : float
        Minimum trigger interval in seconds.
    lowcut, highcut : float
        Bandpass corners for template construction.
    samp_rate : float
        Target sampling rate for template construction.
    output_path : str or Path, optional
        If set, write the Party to this path using EQcorrscan's native
        file format.

    Returns
    -------
    eqcorrscan.Party or None
        Accumulated detections, or *None* if nothing was found.
    """
    detection_files_path = Path(detection_files_path)

    tribe = make_templates(
        templates,
        template_files,
        station_dict,
        template_length,
        template_prepick,
        lowcut=lowcut,
        highcut=highcut,
        samp_rate=samp_rate,
    )

    party_list: list[Party] = []
    iteration_date = start_date

    while iteration_date < end_date:
        logger.info("Processing %s", iteration_date.date)
        day_files = sorted(
            glob.glob(
                str(
                    detection_files_path / f"*.{iteration_date.year}-{iteration_date.month:02}"
                    f"-{iteration_date.day:02}.ms"
                )
            )
        )

        st = Stream()
        for f in day_files:
            st += read(f)

        try:
            party = tribe.detect(
                stream=st,
                threshold=detect_thresh,
                daylong=True,
                threshold_type=thresh_type,
                trig_int=trig_int,
                plot=False,
                return_stream=False,
                parallel_process=False,
                ignore_bad_data=True,
            )
        except (MatchFilterError, NotImplementedError, ValueError) as exc:
            logger.warning("Detection failed for %s: %s", iteration_date.date, exc)
            party = Party(families=[Family(template=Template())])

        if len(party.families[0]) > 0:
            party_list.append(party)

        iteration_date = _next_day(iteration_date)

    if not party_list:
        return None

    combined = party_list[0]
    for p in party_list[1:]:
        combined.families[0] = combined.families[0].append(p.families[0])

    if output_path is not None:
        combined.write(str(output_path))
        logger.info("Wrote Party to %s", output_path)

    return combined


def cull_detections(
    party: Party,
    detection_files_path: str | Path,
    main_trace: tuple[str, str, str],
    *,
    snr_min: float = 1.0,
    snr_max: float = 25.0,
    zero_fraction: float = 0.3,
    lowcut: float | None = 1.0,
    highcut: float | None = 15.0,
) -> Party:
    """Remove low-quality detections from a Party based on SNR.

    Parameters
    ----------
    party : eqcorrscan.Party
        Input detection results.
    detection_files_path : str or Path
        Directory of day-long MiniSEED files.
    main_trace : tuple of str
        ``(network, station, channel)`` for quality assessment.
    snr_min, snr_max : float
        Acceptable SNR range.  Detections outside this range are removed.
    zero_fraction : float
        Maximum fraction of near-zero samples allowed.
    lowcut, highcut : float, optional
        Bandpass filter corners applied before SNR measurement.

    Returns
    -------
    eqcorrscan.Party
        Filtered detection results.
    """
    detection_files_path = Path(detection_files_path)
    net, sta, cha = main_trace
    detections_to_keep = []

    for detection in party.families[0].detections:
        doi = detection.detect_time
        day_files = sorted(
            glob.glob(
                str(
                    detection_files_path / f"{net}.{sta}.{cha}.{doi.year}-{doi.month:02}"
                    f"-{doi.day:02}.ms"
                )
            )
        )
        if not day_files:
            continue

        try:
            st = read(day_files[0])
            if lowcut is not None and highcut is not None:
                st.filter("bandpass", freqmin=lowcut, freqmax=highcut)
            st.trim(doi - 20, doi + 40)
            trace = st[0]

            rounded = np.rint(trace.data.copy())
            zero_count = rounded[np.where(rounded == 0)].size
            trace_snr = snr(trace)
            if not trace_snr:
                continue
            trace_snr_val = trace_snr[0]

            if np.isnan(trace_snr_val):
                continue
            if trace_snr_val < snr_min or trace_snr_val > snr_max:
                continue
            if zero_count > zero_fraction * trace.stats.npts:
                continue

            detections_to_keep.append(detection)
        except (OSError, ValueError, IndexError, TypeError) as exc:
            logger.warning("Skipping detection at %s: %s: %s", doi, type(exc).__name__, exc)

    template = party.families[0].template
    return Party(families=[Family(template, detections_to_keep)])


def get_detections(
    party: Party,
    streams_path: str | Path,
    main_trace: tuple[str, str, str],
    *,
    lowcut: float = 1.0,
    highcut: float = 15.0,
    samp_rate: float = 100.0,
    pre_window: float = 20.0,
    post_window: float = 40.0,
) -> Stream:
    """Extract waveform snippets for each detection in a Party.

    Parameters
    ----------
    party : eqcorrscan.Party
        Detection results.
    streams_path : str or Path
        Directory of day-long MiniSEED files.
    main_trace : tuple of str
        ``(network, station, channel)`` to extract.
    lowcut, highcut : float
        Bandpass filter corners (Hz).
    samp_rate : float
        Target sampling rate (Hz).
    pre_window, post_window : float
        Seconds before/after pick time to include.

    Returns
    -------
    obspy.Stream
        One trace per detection.
    """
    from obspy import Trace

    streams_path = Path(streams_path)
    net, sta, cha = main_trace

    pick_times: list[UTCDateTime] = []
    for event in party.families[0].catalog.events:
        for pick in event.picks:
            wf = pick.waveform_id
            if wf.station_code == sta and wf.network_code == net and wf.channel_code == cha:
                pick_times.append(pick.time)
                break

    result = Stream()
    for pick_time in pick_times:
        day_files = sorted(
            glob.glob(
                str(
                    streams_path / f"{net}.{sta}.{cha}.{pick_time.year}"
                    f"-{pick_time.month:02}-{pick_time.day:02}.ms"
                )
            )
        )
        if day_files:
            day_st = read(day_files[0])
            day_st.filter("bandpass", freqmin=lowcut, freqmax=highcut)
            day_st.interpolate(sampling_rate=samp_rate)
            day_st.trim(
                pick_time - pre_window,
                pick_time + post_window,
                pad=True,
                fill_value=np.nan,
                nearest_sample=True,
            )
            result += day_st
        else:
            result += Trace()

    return result


def detections_from_stacks(
    stack: Stream,
    detection_files_path: str | Path,
    start_date: UTCDateTime,
    end_date: UTCDateTime,
    *,
    pick_times: dict[str, UTCDateTime] | None = None,
    lowcut: float = 1.0,
    highcut: float = 15.0,
    samp_rate: float = 100.0,
    template_length: float = 12.0,
    template_prepick: float = 2.0,
    thresh_type: str = "MAD",
    detect_thresh: float = 8.0,
    trig_int: float = 8.0,
    process_len: float | None = None,
) -> Party | None:
    """Use stacked waveforms as templates for a second detection pass.

    Parameters
    ----------
    stack : obspy.Stream
        Stacked waveforms to use as template.
    detection_files_path : str or Path
        Directory of day-long MiniSEED files.
    start_date, end_date : UTCDateTime
        Detection date range.
    pick_times : dict, optional
        ``{station_code: pick_time}`` for constructing the catalog.
        If *None*, a default pick at the trace midpoint is used.
    lowcut, highcut : float
        Bandpass filter corners (Hz).
    samp_rate : float
        Target sampling rate (Hz).
    template_length, template_prepick : float
        Template duration and pre-pick (seconds).
    thresh_type : str
        Threshold type for EQcorrscan detection.
    detect_thresh : float
        Detection threshold.
    trig_int : float
        Minimum trigger interval (seconds).
    process_len : float, optional
        EQcorrscan processing length (seconds) for the stacked template.  This
        **must** match the length of the data passed to each detection call.
        Detection runs ``daylong=True`` on whole-day streams, so the correct
        value is the day length (``86400``); pass that explicitly.  When *None*
        (default, kept for backward compatibility) it falls back to
        ``max(60, stack_duration + 10)``, which is only appropriate when the
        detection stream is itself that short - using the default against
        day-long data forces EQcorrscan to chunk the day into ``process_len``
        windows and severely under-detects.

    Returns
    -------
    eqcorrscan.Party or None
    """
    detection_files_path = Path(detection_files_path)

    st = Stream()
    picks = []
    for trace in stack:
        st += trace
        if pick_times and trace.stats.station in pick_times:
            pt = pick_times[trace.stats.station]
        else:
            pt = trace.stats.starttime + (trace.stats.endtime - trace.stats.starttime) / 2
        picks.append(
            Pick(
                time=pt,
                phase_hint="P",
                waveform_id=WaveformStreamID(
                    network_code=trace.stats.network,
                    station_code=trace.stats.station,
                    channel_code=trace.stats.channel,
                ),
            )
        )

    event = Event(
        origins=[
            Origin(
                latitude=0.0,
                longitude=0.0,
                depth=0,
                time=stack[0].stats.starttime,
            )
        ],
        magnitudes=[Magnitude(mag=0.0)],
        picks=picks,
    )
    catalog = Catalog([event])

    if process_len is None:
        process_len = max(60, int(stack[0].stats.endtime - stack[0].stats.starttime) + 10)

    stack_tribe = Tribe().construct(
        method="from_meta_file",
        meta_file=catalog,
        st=st,
        lowcut=lowcut,
        highcut=highcut,
        samp_rate=samp_rate,
        length=template_length,
        filt_order=4,
        prepick=template_prepick,
        swin="all",
        process_len=process_len,
        parallel=True,
        skip_short_chans=False,
    )

    party_list: list[Party] = []
    iteration_date = start_date

    while iteration_date < end_date:
        logger.info("Stack detection: %s", iteration_date.date)
        day_files = sorted(
            glob.glob(
                str(
                    detection_files_path / f"*.{iteration_date.year}-{iteration_date.month:02}"
                    f"-{iteration_date.day:02}.ms"
                )
            )
        )
        day_st = Stream()
        for f in day_files:
            day_st += read(f)

        try:
            party = stack_tribe.detect(
                stream=day_st,
                threshold=detect_thresh,
                daylong=True,
                threshold_type=thresh_type,
                trig_int=trig_int,
                plot=False,
                return_stream=False,
                parallel_process=False,
                ignore_bad_data=True,
            )
        except (MatchFilterError, NotImplementedError, ValueError) as exc:
            logger.warning("Stack detection failed for %s: %s", iteration_date.date, exc)
            party = Party(families=[Family(template=Template())])

        if len(party.families[0]) > 0:
            party_list.append(party)

        iteration_date = _next_day(iteration_date)

    if not party_list:
        return None

    combined = party_list[0]
    for p in party_list[1:]:
        combined.families[0] = combined.families[0].append(p.families[0])

    return combined
