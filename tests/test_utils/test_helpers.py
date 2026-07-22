"""Tests for swe.utils.helpers."""

import numpy as np
import pytest
from obspy import Stream, Trace

from swe.utils.helpers import (
    ellipsoidal_distance,
    geo_to_local,
    max_amplitude,
    nearest,
    nearest_ind,
    snr,
)


class TestNearest:
    def test_basic(self):
        items = np.array([1.0, 3.0, 5.0, 7.0])
        assert nearest(items, 4.2) == 5.0

    def test_exact_match(self):
        items = np.array([10, 20, 30])
        assert nearest(items, 20) == 20


class TestNearestInd:
    def test_basic(self):
        items = np.array([1.0, 3.0, 5.0, 7.0])
        assert nearest_ind(items, 4.2) == 2  # index of 5.0

    def test_first_element(self):
        items = np.array([1.0, 3.0, 5.0])
        assert nearest_ind(items, 0.5) == 0


class TestMaxAmplitude:
    def test_1d_array(self):
        data = np.array([-5.0, 3.0, 1.0])
        assert max_amplitude(data) == 5.0

    def test_2d_array(self):
        data = np.array([[-2.0, 1.0], [0.5, -3.0]])
        result = max_amplitude(data)
        np.testing.assert_array_equal(result, [2.0, 3.0])

    def test_trace(self):
        tr = Trace(data=np.array([-4.0, 2.0, 3.0]))
        assert max_amplitude(tr) == 4.0

    def test_stream(self):
        tr1 = Trace(data=np.array([1.0, -2.0]))
        tr2 = Trace(data=np.array([5.0, 0.0]))
        st = Stream([tr1, tr2])
        assert max_amplitude(st) == 5.0

    def test_3d_raises(self):
        data = np.ones((2, 3, 4))
        with pytest.raises(ValueError, match="3-D"):
            max_amplitude(data)

    def test_unsupported_type(self):
        with pytest.raises(TypeError):
            max_amplitude("not an array")

    def test_nan_handling(self):
        data = np.array([np.nan, 2.0, -3.0])
        assert max_amplitude(data) == 3.0


class TestSnr:
    def test_stream(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1.0, 1000)
        data[500] = 10.0  # spike
        tr = Trace(data=data)
        st = Stream([tr])
        result = snr(st)
        assert len(result) == 1
        assert result[0] > 1.0  # peak is well above RMS

    def test_trace(self):
        data = np.ones(100)
        data[50] = 5.0
        tr = Trace(data=data)
        result = snr(tr)
        assert len(result) == 1
        # SNR = max(|data|) / rms(data)
        assert result[0] > 1.0

    def test_empty_trace(self):
        tr = Trace(data=np.array([]))
        assert snr(tr) == []

    def test_unsupported_type(self):
        with pytest.raises(TypeError):
            snr("not a trace")


class TestGeoToLocal:
    def test_centroid_origin(self):
        """Without explicit origin, the centroid should be used."""
        loc = {
            "A": [-118.0, 35.0, 1.0],
            "B": [-117.0, 36.0, 1.0],
        }
        coords, origin = geo_to_local(loc)
        assert origin[0] == pytest.approx(-117.5)
        assert origin[1] == pytest.approx(35.5)
        assert origin[2] == pytest.approx(1.0)
        assert set(coords.keys()) == {"A", "B"}

    def test_explicit_origin(self):
        """An explicit origin should override centroid calculation."""
        loc = {"A": [-118.0, 35.0, 1.0]}
        coords, origin = geo_to_local(loc, origin=(-118.0, 35.0, 1.0))
        assert origin == (-118.0, 35.0, 1.0)
        np.testing.assert_allclose(coords["A"], [0.0, 0.0, 0.0], atol=1.0)

    def test_sign_conventions(self):
        """Stations east/north of origin should have positive x/y."""
        origin = (-118.0, 35.0, 0.5)
        loc = {
            "NE": [-117.5, 35.5, 0.5],  # east + north
            "SW": [-118.5, 34.5, 0.5],  # west + south
        }
        coords, _ = geo_to_local(loc, origin=origin)
        assert coords["NE"][0] > 0  # east
        assert coords["NE"][1] > 0  # north
        assert coords["SW"][0] < 0  # west
        assert coords["SW"][1] < 0  # south

    def test_depth_convention(self):
        """Station below origin elevation should have positive z."""
        origin = (0.0, 0.0, 2.0)
        loc = {"LOW": [0.0, 0.0, 1.0]}  # 1 km lower than origin
        coords, _ = geo_to_local(loc, origin=origin)
        assert coords["LOW"][2] == pytest.approx(1000.0)

    def test_known_distance(self):
        """Two points ~1 degree apart should be ~111 km."""
        origin = (0.0, 0.0, 0.0)
        loc = {"E": [1.0, 0.0, 0.0]}
        coords, _ = geo_to_local(loc, origin=origin)
        # ~111 km at equator
        assert 100_000 < abs(coords["E"][0]) < 120_000

    def test_empty_dict(self):
        """Empty input should return empty dict."""
        coords, _origin = geo_to_local({})
        assert coords == {}

    def test_returns_numpy_arrays(self):
        """Values should be numpy arrays."""
        loc = {"A": [0.0, 0.0, 0.0]}
        coords, _ = geo_to_local(loc, origin=(0.0, 0.0, 0.0))
        assert isinstance(coords["A"], np.ndarray)
        assert coords["A"].shape == (3,)


class TestEllipsoidalDistance:
    def test_same_point(self):
        p = [0.0, 0.0, 0.0]
        assert ellipsoidal_distance(p, p) == pytest.approx(0.0, abs=1e-6)

    def test_known_distance(self):
        # Portland, OR to Seattle, WA (~233 km surface)
        portland = [-122.6765, 45.5231, 0.0]
        seattle = [-122.3321, 47.6062, 0.0]
        dist = ellipsoidal_distance(portland, seattle)
        assert 230 < dist < 240

    def test_two_dimensional(self):
        p1 = [-120.0, 46.0, 5.0]
        p2 = [-120.0, 46.0, 10.0]
        dist_2d = ellipsoidal_distance(p1, p2, two_dimensional=True)
        assert dist_2d == pytest.approx(0.0, abs=1e-3)

    def test_depth_contribution(self):
        p1 = [-120.0, 46.0, 0.0]
        p2 = [-120.0, 46.0, 10.0]
        dist_3d = ellipsoidal_distance(p1, p2)
        assert dist_3d == pytest.approx(10.0, abs=0.1)
