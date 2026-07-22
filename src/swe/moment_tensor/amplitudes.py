"""Three-component phase amplitudes for amplitude moment-tensor inversion.

Measures P- and S-phase amplitudes from three-component records as the peak of
the **3-component vector magnitude** in each phase window.  The vector
magnitude is invariant under any rotation of the sensor frame, so these
observables do **not** require knowing the component orientations -- which is
essential for borehole arrays such as the Parkfield HRSN, whose downhole
sensor azimuths (components 1/2/3) are poorly constrained.  The resulting total
S/P amplitude ratio feeds
:func:`swe.moment_tensor.moment_tensor_from_s_p_ratios`.

True SV/SH separation (ray-frame rotation to P-SV-SH) would constrain the
mechanism more tightly but requires validated component orientations; it is a
deliberate future extension gated on that validation, not implemented here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from obspy import Stream, UTCDateTime

logger = logging.getLogger(__name__)

__all__ = [
    "ThreeComponentAmplitudes",
    "three_component_amplitudes",
]


class ThreeComponentAmplitudes(NamedTuple):
    """Orientation-independent P and S amplitudes for one station."""

    station: str
    """Station code."""
    p_amplitude: float
    """Peak 3-component vector magnitude in the P window."""
    s_amplitude: float
    """Peak 3-component vector magnitude in the S window."""
    sp_ratio: float
    """``s_amplitude / p_amplitude`` (NaN if the P amplitude is zero)."""


def _vector_magnitude_peak(
    stream: Stream,
    start: UTCDateTime,
    end: UTCDateTime,
) -> float:
    """Peak 3-component vector magnitude over ``[start, end]``.

    The three component traces are sliced to the window and truncated to a
    common length; the per-sample magnitude ``sqrt(sum c_i**2)`` is formed and
    its maximum returned.  Rotation-invariant.
    """
    arrays = []
    for tr in stream:
        seg = tr.slice(start, end)
        if seg.stats.npts > 0:
            arrays.append(seg.data.astype(np.float64))
    if not arrays:
        return 0.0
    n = min(len(a) for a in arrays)
    if n == 0:
        return 0.0
    stacked = np.vstack([a[:n] for a in arrays])
    magnitude = np.sqrt(np.sum(stacked**2, axis=0))
    return float(np.max(magnitude))


def three_component_amplitudes(
    stream: Stream,
    *,
    p_window: tuple[float, float],
    s_window: tuple[float, float],
    reference_time: UTCDateTime | None = None,
) -> ThreeComponentAmplitudes:
    """Measure orientation-independent P and S amplitudes for one station.

    Parameters
    ----------
    stream : obspy.Stream
        Exactly three component traces for a single station (e.g. the stacked
        DP1/DP2/DP3 waveforms).  Component orientation is irrelevant because the
        vector magnitude is rotation-invariant.
    p_window : tuple of float
        ``(start, end)`` of the P window in seconds, relative to
        ``reference_time``.
    s_window : tuple of float
        ``(start, end)`` of the S window in seconds, relative to
        ``reference_time``.
    reference_time : obspy.UTCDateTime, optional
        Zero-time for the window offsets.  Defaults to the earliest trace start
        time in *stream*.

    Returns
    -------
    ThreeComponentAmplitudes

    Raises
    ------
    ValueError
        If *stream* does not contain exactly three traces, the traces are not
        all from one station, or a window start is not before its end.
    """
    if len(stream) != 3:
        raise ValueError(f"expected exactly 3 component traces, got {len(stream)}")
    stations = {tr.stats.station for tr in stream}
    if len(stations) != 1:
        raise ValueError(f"all traces must share one station, got {stations}")
    if p_window[0] >= p_window[1]:
        raise ValueError(f"p_window start must precede end: {p_window}")
    if s_window[0] >= s_window[1]:
        raise ValueError(f"s_window start must precede end: {s_window}")

    ref = reference_time if reference_time is not None else min(tr.stats.starttime for tr in stream)

    p_amp = _vector_magnitude_peak(stream, ref + p_window[0], ref + p_window[1])
    s_amp = _vector_magnitude_peak(stream, ref + s_window[0], ref + s_window[1])
    ratio = s_amp / p_amp if p_amp > 0 else float("nan")

    return ThreeComponentAmplitudes(
        station=stream[0].stats.station,
        p_amplitude=p_amp,
        s_amplitude=s_amp,
        sp_ratio=ratio,
    )
