"""Relative moment estimation from matched-filter detections.

Computes relative seismic moments from EQcorrscan Party detections,
either from detection metadata (correlation values and channel counts)
or from direct waveform amplitude comparisons.

References
----------
Shelly, D.R. et al. (2007). Non-volcanic tremor and low-frequency
    earthquake swarms. Nature, 446(7133), 305-307.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd

from swe.utils.helpers import max_amplitude

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

__all__ = [
    "RelativeMomentResult",
    "relative_moment_dataframe",
    "relative_moments_from_party",
    "relative_moments_from_streams",
]


class RelativeMomentResult(NamedTuple):
    """Result from relative moment estimation for a single template family."""

    relative_moments: NDArray[np.floating]
    """Dimensionless moments relative to the template event."""
    relative_magnitudes: NDArray[np.floating]
    """Relative magnitudes: ``(2/3) * log10(relative_moment)``."""
    detect_times: list
    """Detection times (UTCDateTime objects)."""
    template_name: str
    """Name of the template family."""


def relative_moments_from_party(
    party,
    *,
    min_cc: float = 0.0,
) -> list[RelativeMomentResult]:
    """Estimate relative moments from EQcorrscan Party detection metadata.

    Uses the ratio ``|detect_val| / no_chans`` as a proxy for the
    amplitude ratio between each detection and the template, following
    the approach of Shelly et al. (2007).  This is a quick estimate
    that does not require the original waveform data.

    Parameters
    ----------
    party : eqcorrscan.Party
        Detection results containing one or more families.
    min_cc : float
        Minimum cross-correlation value to include a detection.
        Detections below this threshold are excluded.

    Returns
    -------
    list of RelativeMomentResult
        One result per family in the party.
    """
    results: list[RelativeMomentResult] = []

    for family in party.families:
        template_name = family.template.name
        moments = []
        times = []

        for detection in family.detections:
            cc = abs(detection.detect_val)
            if cc < min_cc:
                continue

            n_chans = detection.no_chans if detection.no_chans > 0 else 1
            rel_moment = cc / n_chans
            moments.append(rel_moment)
            times.append(detection.detect_time)

        if not moments:
            logger.warning(
                "Family '%s': no detections above min_cc=%.2f",
                template_name,
                min_cc,
            )
            moments_arr = np.array([], dtype=np.float64)
            mags_arr = np.array([], dtype=np.float64)
        else:
            moments_arr = np.asarray(moments, dtype=np.float64)
            with np.errstate(divide="ignore"):
                mags_arr = (2.0 / 3.0) * np.log10(moments_arr)

        results.append(
            RelativeMomentResult(
                relative_moments=moments_arr,
                relative_magnitudes=mags_arr,
                detect_times=times,
                template_name=template_name,
            )
        )

    return results


def relative_moments_from_streams(
    family,
    detection_streams: list,
    *,
    min_cc: float = 0.0,
) -> RelativeMomentResult:
    """Estimate relative moments from waveform amplitude ratios.

    For each detection, computes the amplitude ratio between the
    detection waveform and the template waveform across all shared
    channels, then takes the median ratio as the relative moment.
    This is more precise than :func:`relative_moments_from_party`
    but requires access to the detection waveforms.

    Parameters
    ----------
    family : eqcorrscan.Family
        A single template family.
    detection_streams : list of obspy.Stream
        One Stream per detection, trimmed around the detection time.
        Must be in the same order as ``family.detections``.
    min_cc : float
        Minimum cross-correlation value to include a detection.

    Returns
    -------
    RelativeMomentResult
        Relative moments computed from amplitude ratios.
    """
    template_st = family.template.st
    template_name = family.template.name
    template_amp = max_amplitude(template_st)
    if template_amp == 0 or np.isnan(template_amp):
        template_amp = 1.0
        logger.warning(
            "Template '%s' has zero/NaN amplitude; using 1.0",
            template_name,
        )

    moments = []
    times = []

    n_detections = min(len(family.detections), len(detection_streams))
    for i in range(n_detections):
        detection = family.detections[i]
        cc = abs(detection.detect_val)
        if cc < min_cc:
            continue

        det_st = detection_streams[i]
        ratios = []

        for tmpl_tr in template_st:
            sel = det_st.select(
                network=tmpl_tr.stats.network,
                station=tmpl_tr.stats.station,
                channel=tmpl_tr.stats.channel,
            )
            if sel:
                tmpl_amp = max_amplitude(tmpl_tr)
                det_amp = max_amplitude(sel[0])
                if tmpl_amp > 0 and not np.isnan(tmpl_amp) and not np.isnan(det_amp):
                    ratios.append(det_amp / tmpl_amp)

        if ratios:
            rel_moment = float(np.median(np.asarray(ratios, dtype=float)))
        else:
            rel_moment = cc / max(detection.no_chans, 1)

        moments.append(rel_moment)
        times.append(detection.detect_time)

    if not moments:
        moments_arr = np.array([], dtype=np.float64)
        mags_arr = np.array([], dtype=np.float64)
    else:
        moments_arr = np.asarray(moments, dtype=np.float64)
        with np.errstate(divide="ignore"):
            mags_arr = (2.0 / 3.0) * np.log10(moments_arr)

    return RelativeMomentResult(
        relative_moments=moments_arr,
        relative_magnitudes=mags_arr,
        detect_times=times,
        template_name=template_name,
    )


def relative_moment_dataframe(
    party,
    *,
    min_cc: float = 0.0,
) -> pd.DataFrame:
    """Compute relative moments and return as a DataFrame.

    Convenience wrapper around :func:`relative_moments_from_party`
    that produces a flat DataFrame suitable for downstream analysis.

    Parameters
    ----------
    party : eqcorrscan.Party
        Detection results containing one or more families.
    min_cc : float
        Minimum cross-correlation value to include a detection.

    Returns
    -------
    pandas.DataFrame
        Columns: ``template_name``, ``detect_time``,
        ``relative_moment``, ``relative_magnitude``.
    """
    results = relative_moments_from_party(party, min_cc=min_cc)

    rows: list[dict] = []
    for result in results:
        for i in range(len(result.relative_moments)):
            rows.append(
                {
                    "template_name": result.template_name,
                    "detect_time": result.detect_times[i],
                    "relative_moment": result.relative_moments[i],
                    "relative_magnitude": result.relative_magnitudes[i],
                }
            )

    return pd.DataFrame(rows)
