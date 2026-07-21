"""Tests for swe.detection.stacking."""

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime

from swe.detection.stacking import Stack, time_shift


class TestStack:
    def test_is_tribe_subclass(self):
        """Stack should be a subclass of eqcorrscan.Tribe."""
        from eqcorrscan import Tribe

        assert issubclass(Stack, Tribe)

    def test_instantiation(self):
        """Empty Stack should be instantiable."""
        s = Stack()
        assert len(s) == 0


class TestTimeShift:
    def test_zero_shift(self):
        """Zero shift should return identical data."""
        sr = 100.0
        n = 1000
        data = np.sin(2 * np.pi * 5 * np.arange(n) / sr)
        tr = Trace(
            data=data.copy(),
            header={"sampling_rate": sr, "starttime": UTCDateTime(0)},
        )
        result = time_shift(tr, 0.0)
        np.testing.assert_allclose(result.data, data, atol=1e-10)

    def test_starttime_adjusted(self):
        """Shift should adjust starttime."""
        sr = 100.0
        data = np.zeros(100)
        start = UTCDateTime("2020-01-01T00:00:00")
        tr = Trace(data=data, header={"sampling_rate": sr, "starttime": start})
        shift_secs = 1.5
        result = time_shift(tr, shift_secs)
        assert result.stats.starttime == start + shift_secs

    def test_does_not_modify_original(self):
        """time_shift should return a copy, not modify in-place."""
        sr = 100.0
        data = np.ones(100)
        start = UTCDateTime("2020-01-01T00:00:00")
        tr = Trace(data=data.copy(), header={"sampling_rate": sr, "starttime": start})
        result = time_shift(tr, 2.0)
        assert tr.stats.starttime == start
        assert result.stats.starttime == start + 2.0

    def test_sine_shift(self):
        """Shifting a sine wave by half a period should approximately flip it."""
        sr = 1000.0
        freq = 5.0
        half_period = 1.0 / (2.0 * freq)
        t = np.arange(0, 1.0, 1 / sr)
        data = np.sin(2 * np.pi * freq * t)
        tr = Trace(
            data=data,
            header={"sampling_rate": sr, "starttime": UTCDateTime(0)},
        )
        result = time_shift(tr, half_period)
        # Interior samples should be roughly negated
        mid = len(data) // 4
        end = 3 * len(data) // 4
        np.testing.assert_allclose(
            result.data[mid:end], -data[mid:end], atol=0.15
        )
