"""Tests for swe.magnitude.duration."""

from __future__ import annotations

import numpy as np
from obspy import Trace, UTCDateTime

from swe.magnitude.duration import (
    DurationMagnitudeResult,
    coda_duration,
    duration_magnitude,
)


class TestCodaDuration:
    """Tests for coda_duration."""

    def test_positive_duration(self, synthetic_trace):
        p_time = synthetic_trace.stats.starttime + 10.0
        tau = coda_duration(synthetic_trace, p_time)
        assert tau > 0.0

    def test_duration_reasonable(self, synthetic_trace):
        """Duration should be less than the trace length after P."""
        p_time = synthetic_trace.stats.starttime + 10.0
        tau = coda_duration(synthetic_trace, p_time)
        max_possible = synthetic_trace.stats.endtime - p_time
        assert tau <= max_possible + 1.0

    def test_higher_snr_threshold_shorter_duration(self, synthetic_trace):
        """A lower SNR threshold should give a longer coda."""
        p_time = synthetic_trace.stats.starttime + 10.0
        tau_low = coda_duration(
            synthetic_trace,
            p_time,
            snr_threshold=1.5,
        )
        tau_high = coda_duration(
            synthetic_trace,
            p_time,
            snr_threshold=5.0,
        )
        assert tau_low >= tau_high

    def test_short_trace_returns_zero(self):
        """Trace too short for measurement should return 0."""
        data = np.zeros(10, dtype=np.float64)
        header = {
            "network": "XX",
            "station": "STA",
            "channel": "BHZ",
            "sampling_rate": 100.0,
            "npts": 10,
            "starttime": UTCDateTime(2024, 1, 1),
        }
        tr = Trace(data=data, header=header)
        # P time beyond trace
        tau = coda_duration(tr, UTCDateTime(2024, 1, 1, 0, 1, 0))
        assert tau == 0.0

    def test_noise_only_trace(self):
        """Pure noise should still return a valid (short) duration."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(5000).astype(np.float64) * 0.01
        header = {
            "network": "XX",
            "station": "STA",
            "channel": "BHZ",
            "sampling_rate": 100.0,
            "npts": 5000,
            "starttime": UTCDateTime(2024, 1, 1),
        }
        tr = Trace(data=data, header=header)
        p_time = UTCDateTime(2024, 1, 1) + 10.0
        tau = coda_duration(tr, p_time)
        assert isinstance(tau, float)
        assert tau >= 0.0


class TestDurationMagnitude:
    """Tests for duration_magnitude."""

    def test_returns_result(self, synthetic_trace):
        p_time = synthetic_trace.stats.starttime + 10.0
        result = duration_magnitude(synthetic_trace, p_time)
        assert isinstance(result, DurationMagnitudeResult)

    def test_md_is_finite(self, synthetic_trace):
        p_time = synthetic_trace.stats.starttime + 10.0
        result = duration_magnitude(synthetic_trace, p_time)
        assert np.isfinite(result.md)

    def test_coda_duration_stored(self, synthetic_trace):
        p_time = synthetic_trace.stats.starttime + 10.0
        result = duration_magnitude(synthetic_trace, p_time)
        assert result.coda_duration > 0.0

    def test_snr_threshold_stored(self, synthetic_trace):
        p_time = synthetic_trace.stats.starttime + 10.0
        result = duration_magnitude(
            synthetic_trace,
            p_time,
            snr_threshold=3.0,
        )
        assert result.snr_threshold_used == 3.0

    def test_distance_effect(self, synthetic_trace):
        """Increasing distance should increase Md (positive a2)."""
        p_time = synthetic_trace.stats.starttime + 10.0
        result_near = duration_magnitude(
            synthetic_trace,
            p_time,
            distance_km=0.0,
        )
        result_far = duration_magnitude(
            synthetic_trace,
            p_time,
            distance_km=100.0,
        )
        assert result_far.md > result_near.md

    def test_custom_coefficients(self, synthetic_trace):
        p_time = synthetic_trace.stats.starttime + 10.0
        result = duration_magnitude(
            synthetic_trace,
            p_time,
            a0=0.0,
            a1=1.0,
            a2=0.0,
        )
        # With a0=0, a1=1, a2=0: Md = log10(tau)
        expected = np.log10(result.coda_duration)
        np.testing.assert_allclose(result.md, expected, rtol=1e-6)

    def test_zero_coda_returns_nan(self):
        """Zero-length coda should produce NaN magnitude."""
        data = np.zeros(10, dtype=np.float64)
        header = {
            "network": "XX",
            "station": "STA",
            "channel": "BHZ",
            "sampling_rate": 100.0,
            "npts": 10,
            "starttime": UTCDateTime(2024, 1, 1),
        }
        tr = Trace(data=data, header=header)
        result = duration_magnitude(tr, UTCDateTime(2024, 1, 1, 0, 1, 0))
        assert np.isnan(result.md)
        assert result.coda_duration == 0.0
