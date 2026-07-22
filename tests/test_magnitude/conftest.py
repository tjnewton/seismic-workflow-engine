"""Shared fixtures for magnitude tests.

Provides mock EQcorrscan Party/Family/Detection objects that do not
require eqcorrscan to be installed.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime


def _make_trace(
    *,
    network: str = "XX",
    station: str = "STA1",
    channel: str = "BHZ",
    sampling_rate: float = 100.0,
    npts: int = 1000,
    starttime: UTCDateTime | None = None,
    amplitude: float = 1.0,
) -> Trace:
    """Return a synthetic Trace with controlled amplitude."""
    if starttime is None:
        starttime = UTCDateTime(2024, 1, 1)
    rng = np.random.default_rng(42)
    data = rng.standard_normal(npts).astype(np.float64) * amplitude
    header = {
        "network": network,
        "station": station,
        "channel": channel,
        "sampling_rate": sampling_rate,
        "npts": npts,
        "starttime": starttime,
    }
    return Trace(data=data, header=header)


def _make_detection(
    *,
    detect_val: float = 0.8,
    no_chans: int = 3,
    detect_time: UTCDateTime | None = None,
    template_name: str = "tmpl_001",
) -> SimpleNamespace:
    """Return a mock Detection-like object."""
    if detect_time is None:
        detect_time = UTCDateTime(2024, 1, 1, 12, 0, 0)
    return SimpleNamespace(
        detect_val=detect_val,
        no_chans=no_chans,
        detect_time=detect_time,
        threshold=8.0,
        threshold_type="MAD",
        threshold_input=8.0,
        template_name=template_name,
    )


def _make_template(
    *,
    name: str = "tmpl_001",
    n_traces: int = 3,
) -> SimpleNamespace:
    """Return a mock Template-like object with a Stream."""
    st = Stream()
    for i in range(n_traces):
        st += _make_trace(station=f"ST{i:02d}")
    return SimpleNamespace(name=name, st=st)


def _make_family(
    *,
    template_name: str = "tmpl_001",
    n_detections: int = 10,
    n_traces: int = 3,
) -> SimpleNamespace:
    """Return a mock Family-like object."""
    rng = np.random.default_rng(42)
    template = _make_template(name=template_name, n_traces=n_traces)
    detections = []
    for i in range(n_detections):
        det = _make_detection(
            detect_val=float(rng.uniform(0.3, 1.5)),
            no_chans=n_traces,
            detect_time=UTCDateTime(2024, 1, 1, 12, 0, 0) + i * 3600,
            template_name=template_name,
        )
        detections.append(det)
    return SimpleNamespace(
        template=template,
        detections=detections,
    )


def _make_party(
    *,
    n_families: int = 1,
    n_detections: int = 10,
) -> SimpleNamespace:
    """Return a mock Party-like object."""
    families = []
    for i in range(n_families):
        fam = _make_family(
            template_name=f"tmpl_{i:03d}",
            n_detections=n_detections,
        )
        families.append(fam)
    return SimpleNamespace(families=families)


@pytest.fixture()
def mock_party():
    """A Party with 1 family and 10 detections."""
    return _make_party(n_families=1, n_detections=10)


@pytest.fixture()
def mock_party_multi():
    """A Party with 2 families and 5 detections each."""
    return _make_party(n_families=2, n_detections=5)


@pytest.fixture()
def mock_family():
    """A single Family with 10 detections."""
    return _make_family(n_detections=10)


@pytest.fixture()
def synthetic_trace():
    """A synthetic trace suitable for duration magnitude tests."""
    sr = 100.0
    npts = 6000  # 60 seconds
    starttime = UTCDateTime(2024, 1, 1, 0, 0, 0)
    t = np.arange(npts) / sr

    # Build a signal: noise + decaying sinusoid starting at t=10s
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(npts) * 0.01
    signal = np.zeros(npts)
    p_onset_idx = int(10.0 * sr)
    for i in range(p_onset_idx, npts):
        elapsed = (i - p_onset_idx) / sr
        signal[i] = 5.0 * np.exp(-elapsed / 10.0) * np.sin(2.0 * np.pi * 5.0 * t[i])

    data = (noise + signal).astype(np.float64)
    header = {
        "network": "XX",
        "station": "STA1",
        "channel": "BHZ",
        "sampling_rate": sr,
        "npts": npts,
        "starttime": starttime,
    }
    return Trace(data=data, header=header)
