"""Tests for swe.utils.signal."""

import numpy as np
import pytest
from obspy import Trace, UTCDateTime

from swe.utils.signal import get_trace_properties, instantaneous_frequency


class TestInstantaneousFrequency:
    def test_sine_wave(self):
        """A pure sine wave should yield ~its known frequency."""
        sr = 1000.0
        freq = 40.0
        t = np.arange(0, 1.0, 1 / sr)
        data = np.sin(2 * np.pi * freq * t)
        tr = Trace(data=data, header={"sampling_rate": sr})
        result = instantaneous_frequency(tr, halfwin=0.06)
        assert abs(result - freq) < 5.0

    def test_peak_near_boundary_returns_nan(self):
        """Peak at the very start should return NaN."""
        data = np.zeros(100)
        data[0] = 10.0  # peak at first sample
        tr = Trace(data=data, header={"sampling_rate": 100.0})
        result = instantaneous_frequency(tr, halfwin=0.06)
        assert np.isnan(result)

    def test_custom_halfwin(self):
        sr = 500.0
        freq = 30.0
        t = np.arange(0, 1.0, 1 / sr)
        data = np.sin(2 * np.pi * freq * t)
        tr = Trace(data=data, header={"sampling_rate": sr})
        result = instantaneous_frequency(tr, halfwin=0.02)
        assert abs(result - freq) < 10.0


class TestGetTraceProperties:
    def test_basic_output_shapes(self):
        """Verify output structure on a synthetic step-like trace."""
        sr = 250.0
        duration_s = 2.0
        n = int(sr * duration_s)
        data = np.zeros(n)
        # Step function at the midpoint
        data[n // 2 :] = 1.0
        # Add small noise to avoid degenerate polynomial fits
        rng = np.random.default_rng(42)
        data += rng.normal(0, 0.001, n)

        start = UTCDateTime("2020-01-01T00:00:00")
        tr = Trace(data=data, header={"sampling_rate": sr, "starttime": start})
        pick_time = start + duration_s / 2  # middle of trace

        dy, d2y, curvature, fits, mads = get_trace_properties(
            tr, pick_time, 0.4, sampling_rate=sr
        )

        assert len(dy) == 1000
        assert len(d2y) == 1000
        assert len(curvature) == 1000
        assert len(fits) == 6
        assert len(mads) == 6

    def test_mads_are_non_negative(self):
        sr = 250.0
        n = int(sr * 2.0)
        rng = np.random.default_rng(99)
        data = rng.normal(0, 1.0, n)
        data[n // 2] = 10.0  # spike

        start = UTCDateTime("2020-01-01T00:00:00")
        tr = Trace(data=data, header={"sampling_rate": sr, "starttime": start})
        pick_time = start + 1.0

        _, _, _, _, mads = get_trace_properties(tr, pick_time, 0.4)
        for val in mads:
            assert val >= 0.0
