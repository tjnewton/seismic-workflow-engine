"""First-motion polarity measurement from P-wave arrivals.

Measures the initial polarity (up or down) of P-wave arrivals on
seismic traces.  Polarity observations are the primary input for
focal mechanism determination via grid-search.

References
----------
Hardebeck, J.L. & Shearer, P.M. (2002). A new method for
    determining first-motion focal mechanisms. Bull. Seismol.
    Soc. Am., 92(6), 2264-2276.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from obspy import Stream
    from obspy.core.trace import Trace
    from obspy.core.utcdatetime import UTCDateTime

logger = logging.getLogger(__name__)

__all__ = [
    "PolarityObservation",
    "PolarityResult",
    "measure_polarities",
    "measure_polarity",
]


class PolarityObservation(NamedTuple):
    """Result from measuring polarity on a single trace."""

    station: str
    """Station identifier (network.station.channel)."""
    polarity: int
    """+1 for compressional (up), -1 for dilatational (down), 0 for indeterminate."""
    confidence: float
    """Confidence in [0, 1] based on SNR."""
    snr: float
    """Signal-to-noise ratio of the first motion."""


class PolarityResult(NamedTuple):
    """Batch polarity measurement result across multiple stations."""

    observations: list[PolarityObservation]
    """Individual polarity observations."""
    stations: list[str]
    """Station identifiers."""
    polarities: NDArray[np.signedinteger]
    """Array of polarity values (+1, -1, 0)."""
    confidences: NDArray[np.floating]
    """Array of confidence values."""


def measure_polarity(
    trace: Trace,
    p_time: UTCDateTime,
    *,
    window_before: float = 0.5,
    window_after: float = 0.1,
    min_snr: float = 3.0,
    highcut: float | None = None,
    lowcut: float | None = None,
) -> PolarityObservation:
    """Measure first-motion polarity on a single trace.

    Examines the first significant amplitude excursion after the
    P-wave arrival to determine whether the initial motion is
    compressional (up, +1) or dilatational (down, -1).

    Parameters
    ----------
    trace : obspy.Trace
        Waveform containing the P-wave arrival.
    p_time : UTCDateTime
        P-wave arrival time.
    window_before : float
        Seconds before P-time used to estimate noise level.
    window_after : float
        Seconds after P-time to search for the first motion.
    min_snr : float
        Minimum signal-to-noise ratio required to assign a polarity.
        Below this threshold, polarity is set to 0 (indeterminate).
    highcut : float, optional
        High-cut frequency (Hz) for bandpass filter.
    lowcut : float, optional
        Low-cut frequency (Hz) for bandpass filter.

    Returns
    -------
    PolarityObservation
        Station ID, polarity (+1/-1/0), confidence, and SNR.
    """
    station_id = f"{trace.stats.network}.{trace.stats.station}.{trace.stats.channel}"
    sr = trace.stats.sampling_rate
    t_start = trace.stats.starttime

    # Work on a copy to avoid modifying the original
    tr = trace.copy()

    # Optional bandpass filtering
    if lowcut is not None and highcut is not None:
        tr.filter("bandpass", freqmin=lowcut, freqmax=highcut, zerophase=True)
    elif lowcut is not None:
        tr.filter("highpass", freq=lowcut, zerophase=True)
    elif highcut is not None:
        tr.filter("lowpass", freq=highcut, zerophase=True)

    # Compute sample indices
    p_idx = int((p_time - t_start) * sr)
    noise_start = max(0, int((p_time - window_before - t_start) * sr))
    noise_end = max(0, p_idx)
    signal_end = min(len(tr.data), int((p_time + window_after - t_start) * sr))

    # Validate indices
    if noise_end <= noise_start or p_idx < 0 or p_idx >= len(tr.data):
        logger.warning("Trace too short for polarity measurement on %s", station_id)
        return PolarityObservation(station=station_id, polarity=0, confidence=0.0, snr=0.0)

    if signal_end <= p_idx:
        logger.warning("Signal window too short for polarity on %s", station_id)
        return PolarityObservation(station=station_id, polarity=0, confidence=0.0, snr=0.0)

    # Noise RMS
    noise_data = tr.data[noise_start:noise_end].astype(np.float64)
    noise_rms = np.sqrt(np.mean(noise_data**2))
    if noise_rms < 1e-30:
        noise_rms = 1e-30

    # Find the first significant excursion after P-time
    signal_data = tr.data[p_idx:signal_end].astype(np.float64)
    peak_idx = int(np.argmax(np.abs(signal_data)))
    peak_amp = signal_data[peak_idx]
    snr = abs(peak_amp) / noise_rms

    # Determine polarity
    if snr < min_snr:
        polarity = 0
        confidence = 0.0
    else:
        polarity = int(np.sign(peak_amp))
        # Confidence scales from 0 at min_snr to 1 at 3*min_snr
        confidence = float(np.clip((snr - min_snr) / (2.0 * min_snr), 0.0, 1.0))

    return PolarityObservation(
        station=station_id,
        polarity=polarity,
        confidence=confidence,
        snr=float(snr),
    )


def measure_polarities(
    stream: Stream,
    p_times: dict[str, UTCDateTime],
    *,
    window_before: float = 0.5,
    window_after: float = 0.1,
    min_snr: float = 3.0,
    highcut: float | None = None,
    lowcut: float | None = None,
) -> PolarityResult:
    """Measure first-motion polarities across all stations in a stream.

    Parameters
    ----------
    stream : obspy.Stream
        Waveform data for multiple stations.
    p_times : dict
        Mapping of station identifier to P-wave arrival time.
        Keys should match ``network.station.channel`` or
        ``station`` format.
    window_before : float
        Seconds before P-time for noise estimation.
    window_after : float
        Seconds after P-time to search for first motion.
    min_snr : float
        Minimum SNR to assign a polarity.
    highcut : float, optional
        High-cut frequency (Hz) for bandpass filter.
    lowcut : float, optional
        Low-cut frequency (Hz) for bandpass filter.

    Returns
    -------
    PolarityResult
        Batch result with observations, stations, polarities, and
        confidences arrays.
    """
    observations: list[PolarityObservation] = []

    for trace in stream:
        station_id = f"{trace.stats.network}.{trace.stats.station}.{trace.stats.channel}"

        # Try full station ID first, then just station name
        p_time = p_times.get(station_id) or p_times.get(trace.stats.station)
        if p_time is None:
            logger.debug("No P-time for %s, skipping", station_id)
            continue

        obs = measure_polarity(
            trace,
            p_time,
            window_before=window_before,
            window_after=window_after,
            min_snr=min_snr,
            highcut=highcut,
            lowcut=lowcut,
        )
        observations.append(obs)

    stations = [obs.station for obs in observations]
    polarities = np.array([obs.polarity for obs in observations], dtype=np.int32)
    confidences = np.array([obs.confidence for obs in observations], dtype=np.float64)

    return PolarityResult(
        observations=observations,
        stations=stations,
        polarities=polarities,
        confidences=confidences,
    )
