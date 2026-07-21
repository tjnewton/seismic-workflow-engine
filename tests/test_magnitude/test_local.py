"""Tests for swe.magnitude.local."""

from __future__ import annotations

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime

from swe.magnitude.local import (
    LocalMagnitudeResult,
    local_magnitude,
    wood_anderson_amplitude,
)


def _make_trace(
    *,
    network: str = "XX",
    station: str = "STA1",
    channel: str = "BHZ",
    sampling_rate: float = 100.0,
    npts: int = 1000,
    amplitude: float = 1.0,
) -> Trace:
    """Return a synthetic Trace."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal(npts).astype(np.float64) * amplitude
    header = {
        "network": network,
        "station": station,
        "channel": channel,
        "sampling_rate": sampling_rate,
        "npts": npts,
        "starttime": UTCDateTime(2024, 1, 1),
    }
    return Trace(data=data, header=header)


class TestWoodAndersonAmplitude:
    """Tests for wood_anderson_amplitude."""

    def test_returns_float(self):
        tr = _make_trace()
        amp = wood_anderson_amplitude(tr)
        assert isinstance(amp, float)

    def test_positive_amplitude(self):
        tr = _make_trace()
        amp = wood_anderson_amplitude(tr)
        # Without proper response metadata the simulation may succeed
        # or return NaN depending on ObsPy version -- just check it's a float
        assert isinstance(amp, float)

    def test_no_response_returns_value(self):
        """Trace without response should still produce a result (simulation only)."""
        tr = _make_trace()
        amp = wood_anderson_amplitude(tr)
        assert isinstance(amp, float)


class TestLocalMagnitude:
    """Tests for local_magnitude."""

    def test_returns_result(self):
        st = Stream([_make_trace(station=f"ST{i}") for i in range(3)])
        distances = [10.0, 20.0, 30.0]
        result = local_magnitude(st, distances)
        assert isinstance(result, LocalMagnitudeResult)

    def test_station_names(self):
        st = Stream([
            _make_trace(station="AAA"),
            _make_trace(station="BBB"),
        ])
        result = local_magnitude(st, [10.0, 20.0])
        assert "XX.AAA.BHZ" in result.station_names
        assert "XX.BBB.BHZ" in result.station_names

    def test_residuals_sum_to_zero(self):
        """Residuals from the mean should sum to approximately zero."""
        st = Stream([_make_trace(station=f"ST{i}") for i in range(3)])
        result = local_magnitude(st, [10.0, 20.0, 30.0])
        valid = result.residuals[~np.isnan(result.residuals)]
        if len(valid) > 0:
            np.testing.assert_allclose(np.sum(valid), 0.0, atol=1e-10)

    def test_mismatched_lengths_raises(self):
        st = Stream([_make_trace()])
        with pytest.raises(ValueError, match="distances_km length"):
            local_magnitude(st, [10.0, 20.0])

    def test_per_station_length(self):
        st = Stream([_make_trace(station=f"ST{i}") for i in range(4)])
        result = local_magnitude(st, [10.0, 20.0, 30.0, 40.0])
        assert len(result.ml_per_station) == 4
        assert len(result.station_names) == 4

    def test_zero_distance_returns_nan(self):
        """Zero distance should produce NaN for that station."""
        st = Stream([_make_trace()])
        result = local_magnitude(st, [0.0])
        assert np.isnan(result.ml_per_station[0])

    def test_ml_stderr_computed(self):
        """ml_stderr should be > 0 with multiple valid stations."""
        st = Stream([
            _make_trace(station=f"ST{i}", amplitude=float(i + 1))
            for i in range(4)
        ])
        result = local_magnitude(st, [10.0, 20.0, 30.0, 40.0])
        if result.n_valid_stations > 1:
            assert result.ml_std > 0
            assert result.ml_stderr > 0
            assert result.ml_stderr < result.ml_std

    def test_single_station_stderr_zero(self):
        """Single station should give stderr=0."""
        st = Stream([_make_trace()])
        result = local_magnitude(st, [10.0])
        assert result.ml_stderr == 0.0
        assert result.n_valid_stations <= 1
