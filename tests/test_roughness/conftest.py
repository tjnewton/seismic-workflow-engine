"""Shared fixtures for roughness module tests.

All data is synthetic - no real data files required.
"""

from __future__ import annotations

import numpy as np
import pytest
from obspy import Trace, UTCDateTime


@pytest.fixture()
def synthetic_surface_profile():
    """1-D surface profile with known roughness.

    Sum of two sinusoids: dominant lambda_1 = 0.5 m (A=1.0)
    and secondary lambda_2 = 0.1 m (A=0.3).
    Profile length: 10 m, spacing: 0.005 m (2000 points).
    """
    dx = 0.005
    x = np.arange(0, 10.0, dx)
    lambda_1 = 0.5
    lambda_2 = 0.1
    z = 1.0 * np.sin(2.0 * np.pi * x / lambda_1) + 0.3 * np.sin(2.0 * np.pi * x / lambda_2)
    return x, z, lambda_1


@pytest.fixture()
def synthetic_elevation_grid():
    """2-D elevation grid with known roughness.

    Dominant wavelength: 2.0 m in x-direction, 3.0 m in y-direction.
    Grid: 50 x 60 points, spacing 0.5 m.
    Returns an ElevationGrid NamedTuple.
    """
    from swe.topo.raster import ElevationGrid

    dx = 0.5
    nx, ny = 60, 50
    x = np.arange(nx) * dx
    y = np.arange(ny) * dx

    xx, yy = np.meshgrid(x, y)
    elevation = 1.0 * np.sin(2.0 * np.pi * xx / 2.0) + 0.5 * np.sin(2.0 * np.pi * yy / 3.0)
    rng = np.random.default_rng(42)
    elevation += rng.standard_normal(elevation.shape) * 0.05

    return ElevationGrid(x=x, y=y, elevation=elevation, crs=None)


@pytest.fixture()
def frequency_traces():
    """ObsPy Trace objects with ~34.5 Hz sinusoid + noise.

    Returns (traces, pick_times): 20 traces with picks spread
    over 10 days (matplotlib date numbers).
    """
    sr = 200.0
    npts = 400  # 2 seconds
    freq_true = 34.5
    rng = np.random.default_rng(42)
    starttime = UTCDateTime(2024, 1, 1, 0, 0, 0)

    traces = []
    pick_times = []

    for i in range(20):
        t = np.arange(npts) / sr
        # Signal: sinusoid with Gaussian envelope centred at 1.0 s
        envelope = np.exp(-((t - 1.0) ** 2) / (2 * 0.1**2))
        data = envelope * np.sin(2.0 * np.pi * freq_true * t)
        data += rng.standard_normal(npts) * 0.05

        header = {
            "network": "XX",
            "station": f"S{i:02d}",
            "channel": "BHZ",
            "sampling_rate": sr,
            "npts": npts,
            "starttime": starttime + i * 3600,
        }
        traces.append(Trace(data=data.astype(np.float64), header=header))

        # Pick times spread over 10 days (matplotlib date numbers)
        pick_times.append(730120.0 + i * 0.5)

    return traces, pick_times, freq_true


@pytest.fixture()
def slip_velocity_series():
    """Slip velocity time series with known ramp.

    Velocity ramps from 1e-7 to 1e-6 m/s over 10 days.
    Times as matplotlib date numbers matching frequency_traces fixture.
    """
    times = np.linspace(730120.0, 730130.0, 50)
    velocities = np.linspace(1e-7, 1e-6, 50)
    return times, velocities
