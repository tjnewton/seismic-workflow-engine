"""Shared fixtures for focal mechanism tests.

Provides synthetic traces with known P-onset polarities, known
geometries, and synthetic focal mechanism data for round-trip tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime


def make_polarity_trace(
    *,
    network: str = "XX",
    station: str = "STA1",
    channel: str = "BHZ",
    sampling_rate: float = 100.0,
    starttime: UTCDateTime | None = None,
    p_onset: float = 10.0,
    polarity_sign: float = 1.0,
    signal_amplitude: float = 10.0,
    noise_amplitude: float = 0.1,
    duration: float = 20.0,
) -> Trace:
    """Return a synthetic trace with a known P-onset polarity.

    Parameters
    ----------
    p_onset : float
        Time in seconds from starttime for the P-arrival.
    polarity_sign : float
        +1 for compressional (up) first motion, -1 for dilatational (down).
    signal_amplitude : float
        Peak amplitude of the first motion.
    noise_amplitude : float
        RMS amplitude of the pre-P noise.
    """
    if starttime is None:
        starttime = UTCDateTime(2024, 1, 1)

    npts = int(duration * sampling_rate)
    rng = np.random.default_rng(42)
    data = rng.standard_normal(npts) * noise_amplitude

    # Insert a clear first-motion pulse after P-onset
    p_idx = int(p_onset * sampling_rate)
    pulse_length = int(0.05 * sampling_rate)  # 50 ms pulse
    for i in range(pulse_length):
        if p_idx + i < npts:
            data[p_idx + i] += polarity_sign * signal_amplitude * np.exp(-i / (pulse_length / 3))

    header = {
        "network": network,
        "station": station,
        "channel": channel,
        "sampling_rate": sampling_rate,
        "npts": npts,
        "starttime": starttime,
    }
    return Trace(data=data.astype(np.float64), header=header)


@pytest.fixture()
def up_trace():
    """Trace with clear compressional (up) first motion."""
    return make_polarity_trace(polarity_sign=1.0, station="UP01")


@pytest.fixture()
def down_trace():
    """Trace with clear dilatational (down) first motion."""
    return make_polarity_trace(polarity_sign=-1.0, station="DN01")


@pytest.fixture()
def noisy_trace():
    """Trace with very low SNR (indeterminate polarity)."""
    return make_polarity_trace(
        signal_amplitude=0.05,
        noise_amplitude=1.0,
        station="NOISY",
    )


@pytest.fixture()
def polarity_stream(up_trace, down_trace, noisy_trace):
    """Stream with up, down, and noisy traces."""
    return Stream([up_trace, down_trace, noisy_trace])


@pytest.fixture()
def p_times():
    """P-times for the polarity stream."""
    t0 = UTCDateTime(2024, 1, 1)
    return {
        "UP01": t0 + 10.0,
        "DN01": t0 + 10.0,
        "NOISY": t0 + 10.0,
    }
