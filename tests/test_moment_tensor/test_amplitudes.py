"""Tests for swe.moment_tensor.amplitudes (3-component phase amplitudes)."""

from __future__ import annotations

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime

from swe.moment_tensor.amplitudes import (
    ThreeComponentAmplitudes,
    three_component_amplitudes,
)

_SR = 250.0
_START = UTCDateTime("2015-02-17T00:00:00")


def _comp(data, comp, station="CCRB"):
    return Trace(
        data=np.asarray(data, dtype=np.float64),
        header={
            "sampling_rate": _SR,
            "starttime": _START,
            "station": station,
            "channel": f"DP{comp}",
        },
    )


def _three_component(c1, c2, c3, station="CCRB"):
    return Stream([_comp(c1, 1, station), _comp(c2, 2, station), _comp(c3, 3, station)])


def _pulse(n, center_s, amp, sr=_SR, width=8):
    data = np.zeros(n)
    i = int(center_s * sr)
    data[i : i + width] = amp
    return data


class TestThreeComponentAmplitudes:
    def test_basic_p_and_s_amplitudes(self):
        n = int(20 * _SR)
        # P at 2 s (amp 1 on comp1), S at 5 s (amp 4 spread over comps)
        c1 = _pulse(n, 2.0, 1.0) + _pulse(n, 5.0, 2.0)
        c2 = _pulse(n, 5.0, 2.0)
        c3 = _pulse(n, 5.0, 1.0)
        st = _three_component(c1, c2, c3)
        res = three_component_amplitudes(st, p_window=(1.5, 3.0), s_window=(4.5, 6.0))
        assert isinstance(res, ThreeComponentAmplitudes)
        assert res.station == "CCRB"
        # P vector magnitude = 1.0; S vector magnitude = sqrt(2^2+2^2+1^2)=3
        assert res.p_amplitude == pytest.approx(1.0)
        assert res.s_amplitude == pytest.approx(3.0)
        assert res.sp_ratio == pytest.approx(3.0)

    def test_rotation_invariance(self):
        """Vector-magnitude amplitudes are invariant under sensor rotation."""
        n = int(20 * _SR)
        c1 = _pulse(n, 2.0, 1.0) + _pulse(n, 5.0, 3.0)
        c2 = _pulse(n, 2.0, 0.5) + _pulse(n, 5.0, 2.0)
        c3 = _pulse(n, 2.0, 0.2) + _pulse(n, 5.0, 1.0)
        st = _three_component(c1, c2, c3)
        base = three_component_amplitudes(st, p_window=(1.5, 3.0), s_window=(4.5, 6.0))

        # Apply an arbitrary 3D rotation to the component vectors.
        theta, phi = 0.7, 1.3
        Rz = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        Ry = np.array(
            [
                [np.cos(phi), 0, np.sin(phi)],
                [0, 1, 0],
                [-np.sin(phi), 0, np.cos(phi)],
            ]
        )
        R = Rz @ Ry
        stacked = np.vstack([c1, c2, c3])
        rot = R @ stacked
        st_rot = _three_component(rot[0], rot[1], rot[2])
        rotated = three_component_amplitudes(st_rot, p_window=(1.5, 3.0), s_window=(4.5, 6.0))
        assert rotated.p_amplitude == pytest.approx(base.p_amplitude, rel=1e-9)
        assert rotated.s_amplitude == pytest.approx(base.s_amplitude, rel=1e-9)
        assert rotated.sp_ratio == pytest.approx(base.sp_ratio, rel=1e-9)

    def test_reference_time(self):
        n = int(20 * _SR)
        c = _pulse(n, 5.0, 1.0)
        st = _three_component(c, c.copy(), c.copy())
        # reference 3 s in: the 5 s pulse is now at +2 s
        res = three_component_amplitudes(
            st,
            p_window=(-1.0, 0.0),  # quiet
            s_window=(1.5, 2.5),  # contains the pulse
            reference_time=_START + 3.0,
        )
        assert res.s_amplitude > res.p_amplitude

    def test_zero_p_gives_nan_ratio(self):
        n = int(20 * _SR)
        zero = np.zeros(n)
        s = _pulse(n, 5.0, 1.0)
        st = _three_component(zero, zero.copy(), s)
        res = three_component_amplitudes(st, p_window=(1.0, 2.0), s_window=(4.5, 6.0))
        assert res.p_amplitude == 0.0
        assert np.isnan(res.sp_ratio)

    def test_requires_three_traces(self):
        n = int(20 * _SR)
        st = Stream([_comp(np.zeros(n), 1), _comp(np.zeros(n), 2)])
        with pytest.raises(ValueError, match="exactly 3"):
            three_component_amplitudes(st, p_window=(1, 2), s_window=(3, 4))

    def test_requires_one_station(self):
        n = int(20 * _SR)
        st = Stream(
            [
                _comp(np.zeros(n), 1, "CCRB"),
                _comp(np.zeros(n), 2, "CCRB"),
                _comp(np.zeros(n), 3, "EADB"),
            ]
        )
        with pytest.raises(ValueError, match="one station"):
            three_component_amplitudes(st, p_window=(1, 2), s_window=(3, 4))

    def test_invalid_windows(self):
        n = int(20 * _SR)
        st = _three_component(np.zeros(n), np.zeros(n), np.zeros(n))
        with pytest.raises(ValueError, match="p_window"):
            three_component_amplitudes(st, p_window=(2, 1), s_window=(3, 4))
        with pytest.raises(ValueError, match="s_window"):
            three_component_amplitudes(st, p_window=(1, 2), s_window=(4, 3))
