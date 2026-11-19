"""Shared fixtures for integration tests."""

from __future__ import annotations

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime


@pytest.fixture()
def synthetic_station_network():
    """Six-station network with geographic and local Cartesian coordinates.

    Returns a dict with keys:
    - ``stations``: list of station codes
    - ``location_dict``: ``{station: [lon, lat, elev]}`` for association (geographic)
    - ``station_coords``: ``(6, 3)`` ndarray in meters for location (local Cartesian)
    """
    stations = ["STA1", "STA2", "STA3", "STA4", "STA5", "STA6"]
    location_dict = {
        "STA1": [-118.50, 34.00, 0.0],
        "STA2": [-118.48, 34.01, 0.0],
        "STA3": [-118.46, 34.02, 0.0],
        "STA4": [-118.44, 34.03, 0.0],
        "STA5": [-118.42, 34.04, 0.0],
        "STA6": [-118.40, 34.05, 0.0],
    }
    # Local Cartesian coordinates in meters (roughly 2 km spacing)
    station_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [2000.0, 1000.0, 0.0],
            [4000.0, 2000.0, 0.0],
            [6000.0, 3000.0, 0.0],
            [8000.0, 4000.0, 0.0],
            [10000.0, 5000.0, 0.0],
        ]
    )
    return {
        "stations": stations,
        "location_dict": location_dict,
        "station_coords": station_coords,
    }


@pytest.fixture()
def synthetic_multi_station_stream(synthetic_station_network):
    """3-component ObsPy Stream for all 6 stations, 60s at 100 Hz.

    Embeds amplitude spikes at t=10s and t=30s to simulate two events.
    """
    stations = synthetic_station_network["stations"]
    t0 = UTCDateTime("2024-01-01T00:00:00")
    npts = 6000  # 60s at 100 Hz
    sampling_rate = 100.0

    rng = np.random.default_rng(42)
    st = Stream()
    for sta in stations:
        for cha in ("BHZ", "BHN", "BHE"):
            data = rng.normal(0, 0.01, npts).astype(np.float64)
            # Event 1 at t=10s
            data[1000:1050] += 1.0
            # Event 2 at t=30s
            data[3000:3050] += 1.0

            header = {
                "network": "XX",
                "station": sta,
                "channel": cha,
                "sampling_rate": sampling_rate,
                "starttime": t0,
                "npts": npts,
            }
            tr = Trace(data=data, header=header)
            st.append(tr)
    return st


@pytest.fixture()
def synthetic_pick_times(synthetic_station_network):
    """Pre-built pick times simulating two events across 6 stations.

    Returns ``{station: [UTCDateTime, ...]}``.
    Event 1 picks cluster around t0+10s, event 2 around t0+610s,
    with small inter-station delays (0.1-0.5 s).  The 600 s gap
    ensures DBSCAN separates the two events cleanly in the
    standardized feature space.
    """
    t0 = UTCDateTime("2024-01-01T00:00:00")
    stations = synthetic_station_network["stations"]
    delays = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    station_picks: dict[str, list[UTCDateTime]] = {}
    for sta, delay in zip(stations, delays, strict=True):
        station_picks[sta] = [
            t0 + 10.0 + delay,  # Event 1
            t0 + 610.0 + delay,  # Event 2
        ]
    return station_picks
