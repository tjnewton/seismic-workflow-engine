"""Shared fixtures for spectral analysis tests.

All data is synthetic - no real data files required.
"""

from __future__ import annotations

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime


@pytest.fixture()
def three_component_stream():
    """Three-component stream with a 5 Hz peak on horizontals for HVSR tests.

    Vertical: white noise.
    Horizontals: white noise + strong 5 Hz sinusoid -> expected HVSR peak near 5 Hz.
    """
    sr = 100.0
    npts = 10000  # 100 seconds
    starttime = UTCDateTime(2024, 1, 1, 0, 0, 0)
    rng = np.random.default_rng(42)
    t = np.arange(npts) / sr

    # Vertical: white noise
    vert = rng.standard_normal(npts).astype(np.float64) * 0.1

    # Horizontals: noise + 5 Hz sinusoid
    signal_5hz = 5.0 * np.sin(2.0 * np.pi * 5.0 * t)
    horiz_n = rng.standard_normal(npts).astype(np.float64) * 0.1 + signal_5hz
    horiz_e = rng.standard_normal(npts).astype(np.float64) * 0.1 + signal_5hz

    st = Stream()
    for channel, data in [("BHZ", vert), ("BHN", horiz_n), ("BHE", horiz_e)]:
        header = {
            "network": "XX",
            "station": "STA1",
            "channel": channel,
            "sampling_rate": sr,
            "npts": npts,
            "starttime": starttime,
        }
        st += Trace(data=data.copy(), header=header)

    return st


@pytest.fixture()
def event_trace_pair():
    """Paired target/reference traces for SSR tests.

    Reference: flat white noise.
    Target: white noise + 3x amplification in 2-8 Hz band.
    """
    sr = 100.0
    npts = 5000  # 50 seconds
    starttime = UTCDateTime(2024, 1, 1, 0, 0, 0)
    rng = np.random.default_rng(123)

    # Reference: broadband noise
    ref_data = rng.standard_normal(npts).astype(np.float64)

    # Target: same noise but amplified in 2-8 Hz
    # Generate using filtered amplification
    target_data = ref_data.copy()
    t = np.arange(npts) / sr
    # Add a strong 4 Hz signal to the target
    target_data += 3.0 * np.sin(2.0 * np.pi * 4.0 * t)

    target_header = {
        "network": "XX",
        "station": "TGT1",
        "channel": "BHZ",
        "sampling_rate": sr,
        "npts": npts,
        "starttime": starttime,
    }
    ref_header = {
        "network": "XX",
        "station": "REF1",
        "channel": "BHZ",
        "sampling_rate": sr,
        "npts": npts,
        "starttime": starttime,
    }
    target = Trace(data=target_data, header=target_header)
    reference = Trace(data=ref_data, header=ref_header)
    return target, reference


@pytest.fixture()
def coda_trace():
    """Synthetic trace with a known decaying coda (Qc ≈ 100).

    Envelope follows: A0 * exp(-pi*f*t/Qc) / t for the single-scattering model.
    """
    sr = 100.0
    npts = 20000  # 200 seconds
    starttime = UTCDateTime(2024, 1, 1, 0, 0, 0)
    rng = np.random.default_rng(77)

    qc_true = 100.0
    central_freq = 3.0  # sqrt(1 * 10) ≈ 3.16, but we use exact for synthesis
    t = np.arange(npts) / sr

    # P-onset at 10s, origin at 5s
    origin_offset = 5.0  # seconds from trace start
    p_onset = 10.0

    data = rng.standard_normal(npts).astype(np.float64) * 0.001  # background

    for i in range(int(p_onset * sr), npts):
        t_lapse = t[i] - origin_offset
        if t_lapse > 0.1:
            envelope_val = 10.0 * np.exp(
                -np.pi * central_freq * t_lapse / qc_true
            ) / t_lapse
            data[i] += envelope_val * np.sin(
                2.0 * np.pi * central_freq * t[i]
            )

    header = {
        "network": "XX",
        "station": "COD1",
        "channel": "BHZ",
        "sampling_rate": sr,
        "npts": npts,
        "starttime": starttime,
    }
    return Trace(data=data.astype(np.float64), header=header)


@pytest.fixture()
def brune_trace():
    """Synthetic trace from IFFT of a Brune spectrum with known parameters.

    omega_0 = 1e-6, corner_frequency = 5.0 Hz.
    """
    sr = 100.0
    npts = 4096
    starttime = UTCDateTime(2024, 1, 1, 0, 0, 0)

    freqs = np.fft.rfftfreq(npts, d=1.0 / sr)

    # Brune spectrum
    omega_0 = 1e-6
    fc = 5.0
    spectrum = omega_0 / (1.0 + (freqs / fc) ** 2)

    # Random phase
    rng = np.random.default_rng(99)
    phase = rng.uniform(-np.pi, np.pi, len(spectrum))
    complex_spec = spectrum * np.exp(1j * phase)
    complex_spec[0] = 0  # zero DC

    data = np.fft.irfft(complex_spec, n=npts).astype(np.float64)

    header = {
        "network": "XX",
        "station": "BRN1",
        "channel": "BHZ",
        "sampling_rate": sr,
        "npts": npts,
        "starttime": starttime,
    }
    return Trace(data=data, header=header)
